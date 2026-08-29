# Nexuss-Fusion

A machine-learning framework that fuses models of **different modalities and
different architectures** into **one native model** — by working on the
mathematics of representation alignment, not by averaging weights.

The pilot target is the SPACE portfolio of always-on micro models:

| Modality | Expert | Weights | Hidden |
|---|---|---|---|
| Text | SmolLM2-360M-Instruct | 0.27 GB | 960 |
| Vision | SmolVLM2-500M-Video-Instruct | 0.44 GB | 768 → 960 (SigLIP → SmolLM2) |
| Audio | Qwen3-ASR-0.6B | 0.80 GB | 896 → 1024 (Qwen3 audio encoder) |

The result is **Nexuss-Fusion-360M**: one decoder that natively consumes
interleaved text, image/video, and speech signals and answers in text — a
teacher/base model for the always-on runtime.

## The idea in one line

> Never average weights that live in different coordinate systems. Instead,
> **canonicalize** every representation space, learn an **optimal affine map
> (Procrustes-initiated)** from each expert's space into the decoder's space,
> **compress** variable-length modality sequences with learned query
> resamplers, and reason over **one interleaved sequence** with one decoder.

This makes *any* architecture compatible: the only requirements are that each
expert emits sequential features and that paired data exists to estimate the
alignment maps. No shared initialization, matching layers, or compatible
tokenizers are ever required.

## Repository layout

```
PROPOSAL.md                  # what, why, architecture, roadmap, gates
docs/MATHS.md                # the fusion mathematics (core contribution)
docs/ARCHITECTURE.md         # system design + training stages
docs/FUSION-MATRIX.md        # measured config of the three target models
nexuss_fusion/
  math/                      # alignment primitives (Procrustes, whitening, resampling)
  sequence/                  # one-native-sequence interleaving + masks
  losses/                    # fused training objective
  specs/                     # declarative model descriptors
  cli.py                     # prints the fusion plan for any spec set
tests/                       # pytest for the math/spec primitives
pyproject.toml
```

## Status

Proposal + mathematical core scaffold. No training/data pipelines yet.
See `PROPOSAL.md` for the staged plan and gates.

Related: `../Space` hosts the benchmark suite, the GGUF micro specialists, and
the Nexuss-AO late-fusion hub (production baseline). Nexuss-Fusion is the
research arm that tries to produce the single unified checkpoint.