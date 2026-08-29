"""One-native-sequence construction: typed interleaving + causal masks (torch)."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class TypedBlock:
    kind: str
    token_count: int
    payload: torch.Tensor | None = None


class Interleaver:
    """Assemble text + modality soft-token blocks into one causal sequence."""

    def __init__(self, marker_ids: dict[str, int], sep_ids: dict[str, int] | None = None) -> None:
        self.marker_ids = marker_ids
        self.sep_ids = sep_ids or {}

    def build(self, blocks: list[TypedBlock]) -> tuple[torch.Tensor, torch.Tensor]:
        ids: list[int] = []
        for block in blocks:
            if block.kind == "text":
                if block.payload is None:
                    raise ValueError("text blocks require a payload of token ids")
                ids.extend(int(t) for t in block.payload.tolist())
                continue
            marker = self.marker_ids.get(block.kind)
            if marker is None:
                raise ValueError(f"no marker registered for {block.kind}")
            ids.extend([marker] * block.token_count)
            sep = self.sep_ids.get(block.kind)
            if sep is not None:
                ids.append(sep)
        ids_t = torch.tensor(ids, dtype=torch.long)
        n = ids_t.shape[0]
        mask = torch.tril(torch.ones((n, n), dtype=torch.bool))
        return ids_t, mask


def build_sequence(blocks: list[TypedBlock], marker_ids: dict[str, int]) -> tuple[torch.Tensor, torch.Tensor]:
    return Interleaver(marker_ids).build(blocks)
