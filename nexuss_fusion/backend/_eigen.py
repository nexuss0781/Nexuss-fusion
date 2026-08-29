"""Optional C++/Eigen backend loader + parity self-test.

The native module `_eigen_native`.so is built by `ci/build_eigen.sh`; until it
exists this module reports `loadable() == False` and the torch backend is used.
"""

from __future__ import annotations

import logging
from typing import Any

import torch

from . import torch_kernels as _tk

log = logging.getLogger(__name__)

_native: Any = None


def _try_import() -> bool:
    global _native  # noqa: PLW0603
    if _native is not None:
        return bool(_native)
    try:
        from . import _eigen_native as mod  # type: ignore[attr-defined]

        _native = mod
    except ImportError as exc:  # pragma: no cover - platform dependent
        log.debug("eigen native module not available: %s", exc)
        _native = False
    return bool(_native)


def loadable() -> bool:
    return _try_import()


def procrustes(source: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if not _try_import():
        raise RuntimeError("eigen native module unavailable")
    src = source.detach().cpu().double().numpy()
    tgt = target.detach().cpu().double().numpy()
    r, scale = _native.procrustes(src, tgt)
    return torch.from_numpy(r).to(source.device), torch.as_tensor(scale, dtype=torch.float64)


def ridge_least_squares(source: torch.Tensor, target: torch.Tensor, lam: float = 1e-3) -> torch.Tensor:
    return _tk.ridge_least_squares(source, target, lam)


def whiten(h: torch.Tensor, mean: torch.Tensor, std: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    if not _try_import():
        raise RuntimeError("eigen native module unavailable")
    out = _native.whiten(
        h.detach().cpu().double().numpy(),
        mean.detach().cpu().double().numpy(),
        std.detach().cpu().double().numpy(),
        scale.detach().cpu().double().numpy(),
    )
    return torch.from_numpy(out).to(h.device)


def parity_check(atol: float = 1e-5, rtol: float = 1e-4) -> bool:
    """Compare eigen kernels against torch kernels on seeded input."""
    if not _try_import():
        return False
    torch.manual_seed(0)
    src = torch.randn(60, 12).double()
    tgt = torch.randn(60, 8).double()
    h = torch.randn(64, 12).double()
    mean = torch.randn(12).double()
    std = torch.rand(12).double() + 0.5
    scale = torch.rand(12).double()

    r_eig, s_eig = procrustes(src, tgt)
    r_ref, s_ref = _tk.procrustes(src, tgt)
    if not torch.allclose(r_eig, r_ref, atol=atol, rtol=rtol) or not torch.isclose(
        s_eig, s_ref, atol=atol, rtol=rtol
    ):
        log.warning("eigen procrustes parity mismatch")
        return False
    w_eig = whiten(h, mean, std, scale)
    w_ref = _tk.whiten(h, mean, std, scale)
    if not torch.allclose(w_eig, w_ref, atol=atol, rtol=rtol):
        log.warning("eigen whiten parity mismatch")
        return False
    return True
