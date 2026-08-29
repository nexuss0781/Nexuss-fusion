import torch

from nexuss_fusion.math.resample import AttentionPooling, budget_for_audio
from nexuss_fusion.sequence.interleave import Interleaver, TypedBlock


def test_budget_for_audio_scales_with_duration():
    assert budget_for_audio(2.0) == 25
    assert budget_for_audio(0.5) == 7
    assert budget_for_audio(1000.0) == 256


def test_attention_pooling_shapes():
    d_in, d_out, budget = 896, 960, 8
    T = 40
    pool = AttentionPooling(d_in=d_in, d_q=d_out, d_out=d_out, budget=budget)
    states = torch.randn(2, T, d_in)
    out = pool(states)
    assert out.shape == (2, budget, d_out)


def test_attention_pooling_bounded_batch():
    d_in = 768
    pool = AttentionPooling(d_in=d_in, d_q=960, d_out=960, budget=8)
    out = pool(torch.randn(1, 15, d_in))
    assert out.shape == (1, 8, 960)


def test_interleaver_builds_causal_sequence():
    inter = Interleaver(marker_ids={"image": 200, "audio": 201, "end": 2})
    text_block = TypedBlock("text", token_count=3, payload=torch.tensor([10, 20, 30]))
    img_block = TypedBlock("image", token_count=4)
    aud_block = TypedBlock("audio", token_count=2)
    ids, mask = inter.build([text_block, img_block, aud_block, text_block])

    assert ids.shape[0] == 3 + 4 + 2 + 3
    assert ids.dtype == torch.long
    assert mask.shape == (ids.shape[0], ids.shape[0])
    assert torch.equal(mask.sum(dim=1), torch.arange(1, ids.shape[0] + 1))


def test_interleaver_two_text_blocks_are_preserved():
    inter = Interleaver({})
    ids, _ = inter.build(
        [
            TypedBlock("text", 0, payload=torch.tensor([5, 6, 7])),
            TypedBlock("text", 0, payload=torch.tensor([8, 9])),
        ]
    )
    assert ids.tolist() == [5, 6, 7, 8, 9]
