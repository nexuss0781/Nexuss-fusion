import numpy as np

from nexuss_fusion.math.resample import AttentionPooling, budget_for_audio
from nexuss_fusion.sequence.interleave import Interleaver, TypedBlock


def test_budget_for_audio_scales_with_duration():
    assert budget_for_audio(2.0) == 25
    assert budget_for_audio(0.5) == 7
    assert budget_for_audio(1000.0) == 256


def test_attention_pooling_shapes():
    d_in, d_q, T, b = 896, 960, 40, 8
    pool = AttentionPooling(d_in, d_q, budget=b)
    keys = np.random.default_rng(0).normal(size=(T, d_q))
    values = np.random.default_rng(1).normal(size=(T, d_in))
    queries = np.random.default_rng(2).normal(size=(b, d_q))
    out = pool(keys, values, queries)
    assert out.shape == (b, d_in)


def test_interleaver_builds_causal_sequence():
    inter = Interleaver(marker_ids={"image": 200, "audio": 201, "end": 2})
    text_block = TypedBlock("text", token_count=3, payload=np.array([10, 20, 30]))
    img_block = TypedBlock("image", token_count=4)
    aud_block = TypedBlock("audio", token_count=2)
    ids, mask = inter.build([text_block, img_block, aud_block, text_block])

    assert ids.shape[0] == 3 + 4 + 2 + 3
    assert np.issubdtype(ids.dtype, np.integer)
    assert mask.shape == (ids.shape[0], ids.shape[0])
    assert np.all(mask.sum(axis=1) == np.arange(1, ids.shape[0] + 1))


def test_interleaver_two_text_blocks_are_preserved():
    inter = Interleaver({})
    ids, _ = inter.build(
        [
            TypedBlock("text", 0, payload=np.array([5, 6, 7])),
            TypedBlock("text", 0, payload=np.array([8, 9])),
        ]
    )
    assert ids.tolist() == [5, 6, 7, 8, 9]