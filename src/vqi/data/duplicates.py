"""Duplicate and near-duplicate detection.

Two mechanisms, two policies:

* **Exact duplicates** (identical SHA-256) are unambiguous. Within the official train set one
  copy is kept; a train image identical to a test image is excluded from *train* so that the
  official test set stays intact and comparable with the literature. Test images are never
  excluded.
* **Near-duplicates** (small pHash Hamming distance) need a threshold that can only be chosen
  after looking at the distance distribution (EDA). Until ``split.duplicates.max_hamming_distance``
  is set, nothing is grouped or excluded on that basis — a *sensitivity table* over candidate
  thresholds is produced instead so that the decision is informed.

Groups (``group_id``) are connected components of the "is a duplicate of" relation (union-find):
the split step keeps every group on one side of the train/validation boundary.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from vqi.config import Config
from vqi.data.hashing import hamming_distance_matrix
from vqi.logging_utils import get_logger, log_event
from vqi.provenance import provenance

logger = get_logger(__name__)

DUPLICATES_REPORT_FILENAME = "duplicates_report.json"
NN_DISTANCES_FILENAME = "phash_nearest_neighbours.csv"


class UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, i: int) -> int:
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def exact_duplicate_groups(df: pd.DataFrame) -> list[list[str]]:
    """Lists of image_ids sharing the same SHA-256 (size >= 2), ordered by acquisition index."""
    valid = df[(~df["is_corrupted"]) & (df["sha256"] != "")]
    groups = []
    for _, grp in valid.groupby("sha256", sort=False):
        if len(grp) > 1:
            grp = grp.sort_values(["official_split", "acquisition_index"], ascending=[False, True])
            groups.append(grp["image_id"].tolist())
    return groups


def phash_distances(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Full Hamming-distance matrix over valid rows; returns (row positions, matrix)."""
    valid_idx = np.flatnonzero((~df["is_corrupted"]).to_numpy() & (df["phash"] != "").to_numpy())
    matrix = hamming_distance_matrix(df["phash"].to_numpy()[valid_idx].tolist())
    return valid_idx, matrix


def nearest_neighbours(df: pd.DataFrame, valid_idx: np.ndarray, matrix: np.ndarray) -> pd.DataFrame:
    """For each valid image, its closest other image (by pHash) and the distance."""
    m = matrix.astype(np.int32).copy()
    np.fill_diagonal(m, np.iinfo(np.int32).max)
    nn_pos = m.argmin(axis=1)
    nn_dist = m[np.arange(len(m)), nn_pos]
    ids = df["image_id"].to_numpy()[valid_idx]
    splits = df["official_split"].to_numpy()[valid_idx]
    labels = df["label"].to_numpy()[valid_idx]
    return pd.DataFrame(
        {
            "image_id": ids,
            "official_split": splits,
            "label": labels,
            "nn_image_id": ids[nn_pos],
            "nn_official_split": splits[nn_pos],
            "nn_label": labels[nn_pos],
            "nn_distance": nn_dist,
        }
    )


def pairs_under_threshold(
    df: pd.DataFrame, valid_idx: np.ndarray, matrix: np.ndarray, max_distance: int
) -> pd.DataFrame:
    """All unordered pairs (i < j) with Hamming distance <= ``max_distance``."""
    iu, ju = np.triu_indices(len(matrix), k=1)
    d = matrix[iu, ju]
    keep = d <= max_distance
    ids = df["image_id"].to_numpy()[valid_idx]
    splits = df["official_split"].to_numpy()[valid_idx]
    labels = df["label"].to_numpy()[valid_idx]
    return (
        pd.DataFrame(
            {
                "image_id_a": ids[iu[keep]],
                "image_id_b": ids[ju[keep]],
                "split_a": splits[iu[keep]],
                "split_b": splits[ju[keep]],
                "label_a": labels[iu[keep]],
                "label_b": labels[ju[keep]],
                "distance": d[keep],
            }
        )
        .sort_values(["distance", "image_id_a"])
        .reset_index(drop=True)
    )


def sensitivity_table(
    df: pd.DataFrame, valid_idx: np.ndarray, matrix: np.ndarray, thresholds: tuple[int, ...]
) -> list[dict[str, Any]]:
    """How many pairs / cross-split pairs / label-conflicting pairs each candidate threshold
    would flag. This is what the pHash threshold decision is made from."""
    rows = []
    for t in thresholds:
        pairs = pairs_under_threshold(df, valid_idx, matrix, t)
        cross = pairs[pairs["split_a"] != pairs["split_b"]]
        conflict = pairs[pairs["label_a"] != pairs["label_b"]]
        rows.append(
            {
                "max_hamming_distance": int(t),
                "n_pairs": int(len(pairs)),
                "n_images_involved": int(
                    pd.unique(pd.concat([pairs["image_id_a"], pairs["image_id_b"]])).size
                ),
                "n_cross_split_pairs": int(len(cross)),
                "n_label_conflict_pairs": int(len(conflict)),
            }
        )
    return rows


def assign_groups(
    df: pd.DataFrame, pairs: pd.DataFrame, exact_groups: list[list[str]]
) -> pd.Series:
    """Connected components over exact + near-duplicate links → ``group_id`` per image.

    Singletons get their own id. Ids are the smallest image_id in the component, prefixed so
    they never collide with a real part id supplied through ``split.group_column``.
    """
    ids = df["image_id"].tolist()
    pos = {img_id: i for i, img_id in enumerate(ids)}
    uf = UnionFind(len(ids))
    for grp in exact_groups:
        for other in grp[1:]:
            uf.union(pos[grp[0]], pos[other])
    for a, b in zip(pairs["image_id_a"], pairs["image_id_b"], strict=True):
        uf.union(pos[a], pos[b])
    roots = [uf.find(i) for i in range(len(ids))]
    root_min_id: dict[int, str] = {}
    for i, r in enumerate(roots):
        root_min_id[r] = min(root_min_id.get(r, ids[i]), ids[i])
    return pd.Series([f"g{root_min_id[r]}" for r in roots], index=df.index, name="group_id")


def exclusion_flags(
    df: pd.DataFrame, exact_groups: list[list[str]], near_pairs: pd.DataFrame
) -> pd.Series:
    """Reason to exclude each *train* image from modelling (empty string = keep).

    Policy (validated in Phase 1):
    * corrupted images are excluded wherever they are (they cannot be used at all);
    * test images are otherwise never excluded;
    * exact duplicates: keep one copy in train; drop the train copy of any train/test duplicate;
    * near-duplicates: only when a threshold is set — drop the train side of a train/test
      near-duplicate pair (same rationale: keep the test set intact and uncontaminated).
      Train/train near-duplicates are *kept* and merely grouped (same split), because they are
      legitimate training variety.
    """
    reason = pd.Series([""] * len(df), index=df.index, dtype=object)
    nonconforming = (~df["name_conforms"]).to_numpy()
    reason[nonconforming] = "nonconforming_filename"
    corrupted = df["is_corrupted"].to_numpy()
    reason[corrupted] = "corrupted:" + df.loc[corrupted, "corruption_reason"].astype(str)

    split = df.set_index("image_id")["official_split"]
    by_id = {img_id: idx for idx, img_id in zip(df.index, df["image_id"], strict=True)}

    conforms = df.set_index("image_id")["name_conforms"]
    for grp in exact_groups:
        grp = [g for g in grp if conforms[g]]  # nonconforming files already carry a reason
        test_members = [g for g in grp if split[g] == "test"]
        train_members = [g for g in grp if split[g] == "train"]
        if test_members:
            for g in train_members:
                reason[by_id[g]] = f"exact_duplicate_of_test:{test_members[0]}"
        else:
            for g in train_members[1:]:
                reason[by_id[g]] = f"exact_duplicate_of_train:{train_members[0]}"

    for a, b, sa, sb in zip(
        near_pairs["image_id_a"],
        near_pairs["image_id_b"],
        near_pairs["split_a"],
        near_pairs["split_b"],
        strict=True,
    ):
        if sa != sb and conforms[a] and conforms[b]:
            train_side, test_side = (a, b) if sa == "train" else (b, a)
            if reason[by_id[train_side]] == "":
                reason[by_id[train_side]] = f"near_duplicate_of_test:{test_side}"
    return reason.rename("exclude_reason")


def analyse_duplicates(
    df: pd.DataFrame, cfg: Config
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    """Run the full analysis. Returns (manifest with group_id/exclude_reason, report, nn table)."""
    dup = cfg.split.duplicates
    exact_groups = exact_duplicate_groups(df)
    valid_idx, matrix = phash_distances(df)
    nn = nearest_neighbours(df, valid_idx, matrix)
    sensitivity = sensitivity_table(df, valid_idx, matrix, dup.candidate_thresholds)

    if dup.max_hamming_distance is None:
        near_pairs = pairs_under_threshold(df, valid_idx, matrix, -1)  # empty
        near_status = "disabled_until_threshold_chosen"
    else:
        near_pairs = pairs_under_threshold(df, valid_idx, matrix, dup.max_hamming_distance)
        near_status = f"enabled_threshold_{dup.max_hamming_distance}"

    out = df.copy()
    out["group_id"] = assign_groups(out, near_pairs, exact_groups)
    out["exclude_reason"] = exclusion_flags(out, exact_groups, near_pairs)

    split_of = out.set_index("image_id")["official_split"]
    cross_exact = [g for g in exact_groups if len({split_of[m] for m in g}) > 1]
    report = provenance(
        cfg,
        exact={
            "n_groups": len(exact_groups),
            "n_extra_images": int(sum(len(g) - 1 for g in exact_groups)),
            "n_cross_split_groups": len(cross_exact),
            "groups": exact_groups,
        },
        near={
            "status": near_status,
            "phash_bits": dup.phash_size**2,
            "n_pairs_flagged": int(len(near_pairs)),
            "nn_distance_quantiles": {
                q: float(np.quantile(nn["nn_distance"], float(q)))
                for q in ("0.0", "0.01", "0.05", "0.25", "0.5", "0.75")
            },
            "n_images_with_nn_distance_0": int((nn["nn_distance"] == 0).sum()),
            "sensitivity": sensitivity,
        },
        exclusions=out["exclude_reason"]
        .str.split(":")
        .str[0]
        .replace("", "kept")
        .value_counts()
        .to_dict(),
        n_groups_total=int(out["group_id"].nunique()),
        n_multi_image_groups=int((out["group_id"].value_counts() > 1).sum()),
    )
    log_event(
        logger,
        "duplicates.analysed",
        exact_groups=len(exact_groups),
        near_status=near_status,
        excluded=int((out["exclude_reason"] != "").sum()),
    )
    return out, report, nn


def write_duplicates_report(report: dict[str, Any], nn: pd.DataFrame, cfg: Config) -> Path:
    cfg.paths.interim_dir.mkdir(parents=True, exist_ok=True)
    path = cfg.paths.interim_dir / DUPLICATES_REPORT_FILENAME
    path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8", newline="\n"
    )
    nn.to_csv(cfg.paths.interim_dir / NN_DISTANCES_FILENAME, index=False, lineterminator="\n")
    return path


def read_duplicates_report(cfg: Config) -> dict[str, Any]:
    path = cfg.paths.interim_dir / DUPLICATES_REPORT_FILENAME
    return json.loads(path.read_text(encoding="utf-8"))


def read_nn_distances(cfg: Config) -> pd.DataFrame:
    return pd.read_csv(
        cfg.paths.interim_dir / NN_DISTANCES_FILENAME, dtype={"image_id": str, "nn_image_id": str}
    )
