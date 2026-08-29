"""One-native-sequence construction: typed interleaving + causal block masks."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TypedBlock:
    kind: str
    token_count: int
    payload: np.ndarray | None = None


class Interleaver:
    """Assemble text + modality soft-token blocks into a single causal sequence.

    Builds an additive causal mask so every position attends to all tokens of
    earlier blocks and to earlier tokens of its own block (block-causal).
    """

    def __init__(self, marker_ids: dict[str, int], sep_ids: dict[str, int] | None = None) -> None:
        self.marker_ids = marker_ids
        self.sep_ids = sep_ids or {}

    def build(self, blocks: list[TypedBlock]) -> tuple[np.ndarray, np.ndarray]:
        """Return (token_ids, causal_mask). One marker token per block (except text).

        For soft-token blocks the ids are reserved placeholder marker ids (the
        real embeddings are substituted at model time); text blocks contribute
        their token ids directly.
        """
        ids: list[int] = []
        for block in blocks:
            if block.kind == "text":
                ids.extend(int(t) for t in block.payload)
            else:
                marker = self.marker_ids.get(block.kind)
                if marker is None:
                    raise ValueError(f"no marker registered for {block.kind}")
                ids.extend([marker] * block.token_count)
                sep = self.sep_ids.get(block.kind)
                if sep is not None:
                    ids.append(sep)
        ids_arr = np.asarray(ids, dtype=np.int64)
        n = ids_arr.shape[0]
        mask = np.zeros((n, n), dtype=bool)
        for i in range(n):
            mask[i, : i + 1] = True
        return ids_arr, mask


def build_sequence(blocks: list[TypedBlock], marker_ids: dict[str, int]) -> tuple[np.ndarray, np.ndarray]:
    return Interleaver(marker_ids).build(blocks)