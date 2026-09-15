"""Reproducibility helpers.

A single seed (``reproducibility.seed`` in the config) drives Python's ``random``, NumPy, PyTorch
and every DataLoader worker. On CPU some operations remain non-deterministic across library
versions or thread counts; that residual variance is documented (risk R9), not hidden.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = True, num_threads: int | None = None) -> None:
    """Seed every random source used by the pipeline.

    Args:
        seed: the global seed.
        deterministic: request deterministic PyTorch algorithms (``warn_only`` so that an op
            without a deterministic implementation logs instead of crashing).
        num_threads: pin the intra-op thread count; different thread counts can change the
            order of floating-point reductions and therefore the last bits of a result.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
    if num_threads is not None:
        torch.set_num_threads(num_threads)


def worker_init_fn(worker_id: int) -> None:
    """Seed NumPy/random inside a DataLoader worker from the torch per-worker seed."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_generator(seed: int) -> torch.Generator:
    """A torch generator to pass to DataLoader/sampler so shuffling is reproducible."""
    gen = torch.Generator()
    gen.manual_seed(seed)
    return gen
