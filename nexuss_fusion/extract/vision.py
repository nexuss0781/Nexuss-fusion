"""Frozen vision feature extractor: SigLIP patch states of SmolVLM2-500M."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import torch
from transformers import AutoProcessor, SmolVLMForConditionalGeneration

log = logging.getLogger(__name__)

SIGLIP_HIDDEN = 768
PATCH_SIZE = 16


@dataclass
class VisionExtractor:
    model_id: str = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
    revision: str = "main"
    device: str = "cpu"
    torch_dtype: torch.dtype = torch.float32
    _processor: AutoProcessor | None = field(init=False, default=None)
    _model: SmolVLMForConditionalGeneration | None = field(init=False, default=None)

    def _ensure(self) -> tuple[AutoProcessor, SmolVLMForConditionalGeneration]:
        if self._processor is None or self._model is None:
            self._processor = AutoProcessor.from_pretrained(self.model_id, revision=self.revision)
            self._model = SmolVLMForConditionalGeneration.from_pretrained(
                self.model_id, revision=self.revision, torch_dtype=self.torch_dtype
            ).to(self.device)
            self._model.eval()
        return self._processor, self._model

    def encode_pixel_values(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """SigLIP patch hidden states before the connector: (num_patches, 768)."""
        _, model = self._ensure()
        vision_tower = model.model.vision_model
        with torch.no_grad():
            out = vision_tower(pixel_values.to(device=self.device, dtype=self.torch_dtype))
        states = out.last_hidden_state  # (B, T, hidden)
        return states[0].float().cpu()

    def encode_image(self, image_pil) -> torch.Tensor:
        processor, _ = self._ensure()
        inputs = processor(images=image_pil, return_tensors="pt")
        return self.encode_pixel_values(inputs["pixel_values"])

    def encode_image_batch_cached(self, image_paths: list[str], cache) -> torch.Tensor:
        from PIL import Image

        rows = []
        for path in image_paths:
            key = cache.key(
                {"kind": "image", "model": self.model_id, "revision": self.revision, "image": path}
            )
            states, _ = cache.compute_or_get(
                key, lambda p=path: self._encode_single(Image.open(p).convert("RGB"))
            )
            rows.append(states.mean(dim=0, keepdim=True))  # image-level source row
        return torch.cat(rows, dim=0)

    def _encode_single(self, image_pil) -> torch.Tensor:
        return self.encode_image(image_pil)


def validate_patch_states(states: torch.Tensor, expected_hidden: int = SIGLIP_HIDDEN) -> None:
    if states.dim() != 2 or states.shape[-1] != expected_hidden:
        raise ValueError(
            f"unexpected SigLIP state shape {tuple(states.shape)} (expected (patches, {expected_hidden}))"
        )
