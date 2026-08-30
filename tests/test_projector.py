"""Tests for VisionProjector + AttentionPooling (stage 1b)."""

from __future__ import annotations

import torch

from nexuss_fusion.math.projector import VisionProjector
from nexuss_fusion.math.resample import AttentionPooling, budget_for_image


class TestAttentionPooling:
    def test_output_shape(self) -> None:
        pool = AttentionPooling(d_in=64, d_q=64, d_out=128, budget=8)
        x = torch.randn(2, 16, 64)
        out = pool(x)
        assert out.shape == (2, 8, 128)

    def test_single_batch(self) -> None:
        pool = AttentionPooling(d_in=32, d_q=32, d_out=64, budget=4)
        x = torch.randn(1, 10, 32)
        out = pool(x)
        assert out.shape == (1, 4, 64)

    def test_large_sequence(self) -> None:
        pool = AttentionPooling(d_in=256, d_q=256, d_out=512, budget=16)
        x = torch.randn(4, 1024, 256)
        out = pool(x)
        assert out.shape == (4, 16, 512)

    def test_budget_clamp(self) -> None:
        pool = AttentionPooling(d_in=16, d_q=16, d_out=32, budget=128)
        x = torch.randn(1, 8, 16)
        out = pool(x)
        assert out.shape == (1, 8, 32)

    def test_grad_flows(self) -> None:
        pool = AttentionPooling(d_in=32, d_q=32, d_out=64, budget=4)
        x = torch.randn(2, 10, 32, requires_grad=True)
        out = pool(x)
        out.sum().backward()
        assert x.grad is not None


class TestVisionProjector:
    def test_output_shape(self) -> None:
        proj = VisionProjector(d_in=64, d_hidden=128, d_out=256, budget=8)
        x = torch.randn(2, 16, 64)
        out = proj(x)
        assert out.shape == (2, 8, 256)

    def test_unbatched_input(self) -> None:
        proj = VisionProjector(d_in=64, d_hidden=128, d_out=256, budget=8)
        x = torch.randn(16, 64)
        out = proj(x)
        assert out.shape == (1, 8, 256)

    def test_output_dim_matches_d_out(self) -> None:
        proj = VisionProjector(d_in=768, d_hidden=1024, d_out=960, budget=64)
        x = torch.randn(1, 128, 768)
        out = proj(x)
        assert out.shape == (1, 64, 960)

    def test_grad_flows(self) -> None:
        proj = VisionProjector(d_in=32, d_hidden=64, d_out=128, budget=4)
        x = torch.randn(2, 8, 32)
        out = proj(x)
        out.sum().backward()
        for p in proj.parameters():
            assert p.grad is not None

    def test_deterministic(self) -> None:
        proj = VisionProjector(d_in=32, d_hidden=64, d_out=128, budget=4)
        proj.eval()
        x = torch.randn(1, 10, 32)
        out1 = proj(x)
        out2 = proj(x)
        assert torch.allclose(out1, out2)


class TestBudgetForImage:
    def test_bounded(self) -> None:
        assert budget_for_image(1024, 64) == 64
        assert budget_for_image(8, 64) == 8
        assert budget_for_image(1024) == 64
