"""Backend: resolve torch/eigen/auto for hot non-trainable kernels.

- torch: primary, autograd-capable reference implementation.
- eigen: optional C++/Eigen fallback built by `ci/build_eigen.sh`; used only
  for non-trainable hot kernels (procrustes init, whitening) after a parity
  self-test against torch passes.
- auto: torch first; eigen only when compiled and parity OK.

Trainable modules stay torch-only (autograd). The eigen path never computes
gradients, so it is opted into explicitly via NEXUSS_FUSION_BACKEND=eigen.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Callable

import torch

log = logging.getLogger(__name__)

ENV_KEY = "NEXUSS_FUSION_BACKEND"

BackendFn = Callable[..., torch.Tensor]


@dataclass(frozen=True)
class Backend:
    name: str
    procrustes: BackendFn
    ridge_least_squares: BackendFn
    whiten: BackendFn


def _torch_backend() -> Backend:
    from . import torch_kernels as tk

    return Backend(
        name="torch",
        procrustes=tk.procrustes,
        ridge_least_squares=tk.ridge_least_squares,
        whiten=tk.whiten,
    )


def _eigen_backend() -> Backend | None:
    try:
        from . import _eigen as wrapper
    except ImportError as exc:  # pragma: no cover - unexpected
        log.debug("eigen wrapper unavailable: %s", exc)
        return None
    if not wrapper.loadable() or not wrapper.parity_check():
        log.debug("eigen backend not loadable or failed parity check")
        return None
    return Backend(
        name="eigen",
        procrustes=wrapper.procrustes,
        ridge_least_squares=wrapper.ridge_least_squares,
        whiten=wrapper.whiten,
    )


_backends: dict[str, Backend | None] | None = None


def _resolve_all() -> None:
    global _backends  # noqa: PLW0603
    if _backends is not None:
        return
    _backends = {
        "torch": _torch_backend(),
        "eigen": _eigen_backend(),
    }


def eigen_available() -> bool:
    _resolve_all()
    return _backends is not None and _backends["eigen"] is not None


def eigen_parity_ok() -> bool:
    return eigen_available()


def get_backend(name: str | None = None) -> Backend:
    _resolve_all()
    requested = (name or os.environ.get(ENV_KEY) or "auto").lower()
    if requested not in _backends:
        requested = "auto"
    if requested == "auto":
        backend = _backends["eigen"] if _backends["eigen"] is not None else _backends["torch"]
    else:
        backend = _backends[requested]
    if backend is None:
        raise RuntimeError(f"backend '{requested}' unavailable (native module missing or parity failed)")
    return backend