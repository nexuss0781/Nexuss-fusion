"""Model specifications: the framework's 'any architecture' registry."""
from __future__ import annotations

from .models import AUDIO_MICRO, SPECS, TEXT_MICRO, VISION_MICRO, ModelSpec, spec_for

__all__ = ["AUDIO_MICRO", "SPECS", "TEXT_MICRO", "VISION_MICRO", "ModelSpec", "spec_for"]