from nexuss_fusion.math.resample import budget_for_image


def test_budget_for_image_caps_at_limit():
    assert budget_for_image(1024) == 64
    assert budget_for_image(1024, budget=32) == 32
    assert budget_for_image(100, budget=16) == 16


def test_budget_for_image_respects_patches():
    assert budget_for_image(50) == 50
