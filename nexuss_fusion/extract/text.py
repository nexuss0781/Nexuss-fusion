"""Frozen text embedder: mean-pooled caption embeddings from SmolLM2-360M."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import torch
from transformers import AutoModel, AutoTokenizer

log = logging.getLogger(__name__)

SMOL_HIDDEN = 960


def normalize_caption(caption: str) -> str:
    return " ".join(caption.split()).lower()


def caption_cache_key(cache, model_id: str, revision: str, caption: str) -> str:
    return cache.key(
        {"kind": "text", "model": model_id, "revision": revision, "caption": normalize_caption(caption)}
    )


@dataclass
class TextEmbedder:
    model_id: str = "HuggingFaceTB/SmolLM2-360M-Instruct"
    revision: str = "main"
    device: str = "cpu"
    torch_dtype: torch.dtype = torch.float32
    _tokenizer: AutoTokenizer | None = field(init=False, default=None)
    _model: AutoModel | None = field(init=False, default=None)

    def _ensure(self) -> tuple[AutoTokenizer, AutoModel]:
        if self._tokenizer is None or self._model is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_id, revision=self.revision)
            self._model = AutoModel.from_pretrained(
                self.model_id, revision=self.revision, torch_dtype=self.torch_dtype
            ).to(self.device)
            self._model.eval()
        return self._tokenizer, self._model

    def embed_caption(self, caption: str) -> torch.Tensor:
        """Mean-pooled embedding over non-padding tokens: shape (1, hidden)."""
        tokenizer, model = self._ensure()
        enc = tokenizer(caption, return_tensors="pt", truncation=True, max_length=512)
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)
        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attention_mask)
        rows = out.last_hidden_state.float()
        masked = rows * attention_mask.unsqueeze(-1).float()
        pooled = masked.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True).clamp(min=1).float()
        return pooled

    def embed_batch_cached(self, captions: list[str], cache) -> torch.Tensor:
        rows = []
        for caption in captions:
            key = caption_cache_key(cache, self.model_id, self.revision, caption)
            emb, _ = cache.compute_or_get(key, lambda c=caption: self.embed_caption(c))
            rows.append(emb)
        return torch.cat(rows, dim=0)
