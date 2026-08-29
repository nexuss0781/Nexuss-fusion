"""Declarative model descriptors: the framework's 'any architecture' registry."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ModelSpec:
    name: str
    kind: str
    hidden_dim: int
    token_law: str
    source: str
    license: str
    role: str
    notes: str = ""
    meta: dict = field(default_factory=dict)


TEXT_MICRO = ModelSpec(
    name="SmolLM2-360M-Instruct",
    kind="text",
    hidden_dim=960,
    token_law="native token ids",
    source="HuggingFaceTB/SmolLM2-360M-Instruct",
    license="Apache-2.0",
    role="native decoder (text)",
    notes="Dual hidden dim 960; llama-layout; 0.27 GB q4_k_m; 57.73 tok/s",
)

VISION_MICRO = ModelSpec(
    name="SmolVLM2-500M-Video-Instruct",
    kind="vision",
    hidden_dim=768,
    token_law="SigLIP 1024 patches -> pixel-shuffle x4 -> ~64 soft tokens",
    source="HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
    license="Apache-2.0",
    role="native vision in Option A; frozen vision source in Option B",
    notes="siglip hidden 768; smollm2 decoder hidden 960; 0.44 GB q8_0; 67.15 tok/s",
)

AUDIO_MICRO = ModelSpec(
    name="Qwen3-ASR-0.6B",
    kind="audio",
    hidden_dim=896,
    token_law="AuT d_model 896; ~12.5 states/s after downsample (conv_chunksize 500)",
    source="Qwen/Qwen3-ASR-0.6B",
    license="Apache-2.0",
    role="frozen audio encoder source",
    notes="own projector 896->1024 NOT reused; nexus fusion trains 896->960 bridge; 0.80 GB q8_0",
)

SPECS: dict[str, ModelSpec] = {s.name: s for s in (TEXT_MICRO, VISION_MICRO, AUDIO_MICRO)}


def spec_for(name: str) -> ModelSpec:
    try:
        return SPECS[name]
    except KeyError as exc:
        raise KeyError(f"unregistered model: {name}; register a ModelSpec first") from exc
