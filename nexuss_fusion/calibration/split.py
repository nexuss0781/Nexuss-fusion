"""Deterministic example-level splits (by key, ensures no image/caption leakage)."""
from __future__ import annotations

import hashlib

import torch


def split_by_key(keys: list[str], te: float = 0.2, seed: int = 42) -> tuple[list[str], list[str]]:
    if not 0.0 <= te < 1.0:
        raise ValueError("te must satisfy 0 <= te < 1")
    rng = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(keys), generator=rng).tolist()
    n_test = int(round(len(keys) * te))
    test_keys = [keys[i] for i in idx[:n_test]]
    train_keys = [keys[i] for i in idx[n_test:]]
    return train_keys, test_keys


def key_for(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()