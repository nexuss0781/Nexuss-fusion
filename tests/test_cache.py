import torch

from nexuss_fusion.data.features_cache import FeaturesCache
from nexuss_fusion.data.manifest import load_manifest, sha256_file, verify_manifest, write_manifest


def test_cache_roundtrip_and_hit(tmp_path):
    cache = FeaturesCache(tmp_path / "cache")
    key = cache.key({"kind": "image", "image": "a.jpg"})
    tensor = torch.randn(5, 768)
    assert not cache.has(key)
    cache.put(key, tensor)
    assert cache.has(key)
    cached, fresh = cache.compute_or_get(key, lambda: torch.randn(5, 768))
    assert not fresh
    assert torch.equal(cached, tensor)


def test_cache_recomputes_when_missing(tmp_path):
    cache = FeaturesCache(tmp_path / "cache")
    key = cache.key({"kind": "image", "image": "b.jpg"})
    got, fresh = cache.compute_or_get(key, lambda: torch.full((2, 3), 7.0))
    assert fresh
    assert torch.equal(got, torch.full((2, 3), 7.0))


def test_manifest_write_verify(tmp_path):
    manifest = {"a": 1, "b": "two"}
    path = tmp_path / "m.json"
    write_manifest(manifest, path)
    assert verify_manifest(path) is None
    assert load_manifest(path)["a"] == 1


def test_manifest_detects_corruption(tmp_path):
    path = tmp_path / "m.json"
    write_manifest({"a": 1}, path)
    path.write_text('{"a": 2}')
    assert verify_manifest(path) is not None


def test_sha256_file_consistent(tmp_path):
    p = tmp_path / "x.txt"
    p.write_bytes(b"hello")
    assert sha256_file(p) == "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
