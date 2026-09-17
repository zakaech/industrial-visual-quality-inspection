"""Grad-CAM (Selvaraju et al., 2017) for :class:`DefectClassifier`.

CAM = ReLU(Σ_k α_k A_k) with A_k the activations of the target layer (``layer4``, 20×7 cells on a
640×224 input) and α_k the spatially averaged gradient of the defect logit w.r.t. A_k. The map is
normalised to [0, 1] and upsampled to the input size. Masks are used *only* to measure how much
of the map falls inside the annotated defect; they never enter the model.

Limits (documented in the report): coarse resolution (one cell ≈ 32×32 px), sensitivity to the
chosen layer, a heat-map is evidence of *where* the logit is sensitive, not a segmentation, and
it says nothing about calibration or about defects smaller than a cell.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
import torch
from torch import nn

from vqi.models.factory import DefectClassifier


class GradCAM:
    def __init__(self, model: DefectClassifier, target_layer: str = "layer4") -> None:
        self.model = model.eval()
        self.layer: nn.Module = model.group(target_layer)
        self._act: torch.Tensor | None = None
        self._grad: torch.Tensor | None = None
        self._h1 = self.layer.register_forward_hook(self._save_act)
        self._h2 = self.layer.register_full_backward_hook(self._save_grad)

    def _save_act(self, _m: nn.Module, _i: Any, out: torch.Tensor) -> None:
        self._act = out.detach()

    def _save_grad(self, _m: nn.Module, _gi: Any, grad_out: tuple[torch.Tensor, ...]) -> None:
        self._grad = grad_out[0].detach()

    def close(self) -> None:
        self._h1.remove()
        self._h2.remove()

    def __call__(self, x: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        """Returns (cams [B, H, W] in [0, 1] at input resolution, probabilities [B])."""
        self.model.zero_grad(set_to_none=True)
        x = x.detach().requires_grad_(False)
        logits = self.model(x).squeeze(1)
        logits.sum().backward()  # d logit_i / d A_i for each sample (samples are independent)
        assert self._act is not None and self._grad is not None
        weights = self._grad.mean(dim=(2, 3), keepdim=True)  # α_k
        cam = torch.relu((weights * self._act).sum(dim=1))  # [B, h, w]
        cams = []
        for c in cam:
            c = c.cpu().numpy().astype(np.float32)
            c = c - c.min()
            c = c / c.max() if c.max() > 0 else c
            cams.append(cv2.resize(c, (x.shape[3], x.shape[2]), interpolation=cv2.INTER_LINEAR))
        return np.stack(cams), torch.sigmoid(logits).detach().cpu().numpy().astype(np.float64)


def overlay(canvas_rgb: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    heat = cv2.applyColorMap((255 * cam).astype(np.uint8), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(canvas_rgb, 1 - alpha, heat, alpha, 0)


def cam_mask_overlap(cam: np.ndarray, mask: np.ndarray) -> dict[str, float | bool]:
    """How much of the CAM mass lies inside the defect mask, and whether its peak does."""
    inside = mask > 0
    total = float(cam.sum())
    share = float(cam[inside].sum() / total) if total > 0 and inside.any() else float("nan")
    peak = np.unravel_index(int(cam.argmax()), cam.shape)
    return {
        "cam_mass_in_mask": share,
        "mask_area_share": float(inside.mean()),
        "peak_in_mask": bool(inside[peak]) if inside.any() else False,
        "concentration_ratio": (
            share / float(inside.mean()) if inside.any() and inside.mean() > 0 else float("nan")
        ),
    }
