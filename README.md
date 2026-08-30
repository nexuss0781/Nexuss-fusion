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

## Runtime & backends

PyTorch-first. All trainable math runs in autograd-capable torch. A validated
C++/Eigen fallback (`cpp/eigen_kernels.cpp`, built by `ci/build_eigen.sh`) can
be opted into for hot non-trainable kernels (Procrustes init, whitening) via
`NEXUSS_FUSION_BACKEND=eigen`; it is only selected after a parity self-test
against the torch kernels passes. `auto` prefers torch.

## Repository layout

```
PROPOSAL.md                  # what, why, architecture, roadmap, gates
docs/MATHS.md                # the fusion mathematics (core contribution)
docs/ARCHITECTURE.md         # system design + training stages
docs/FUSION-MATRIX.md        # measured config of the three target models
docs/PHASE2-RESULTS.md       # first experiment campaign (recorded + interpreted)
nexuss_fusion/
  backend/                   # torch primary + optional eigen kernels + parity
  math/                      # alignment primitives (Procrustes, whitening, resampling)
  sequence/                  # one-native-sequence interleaving + masks
  losses/                    # fused training objective
  specs/                     # declarative model descriptors
  extract/                   # frozen modality encoders (SigLIP states, SmolLM2 text)
  data/                      # content-addressed feature cache + immutable manifests
  calibration/               # Procrustes/Ridge bridges + deterministic splits
  eval/                      # held-out alignment metrics + acceptance gates
  run/phase2.py              # phase 2 experiment entrypoint
  run/phase1b.py             # stage 1b: train vision projector + resampler
  run/phase2a.py             # stage 2a: decoder alignment (vision prefix + LM loss)
  run/phase2b.py             # stage 2b: unfreeze decoder + e2e training
  run/evaluate.py            # generation quality eval (greedy decode + BLEU/ROUGE)
ci/                          # eigen build script
cpp/                         # Eigen fallback kernels
scripts/                     # synthetic calibration-set generator (CI smoke)
.github/workflows/           # ci.yml (unit/lint) + phase2-vision.yml (experiment)
tests/                       # pytest for math/sequence/data/calibration/eval
pyproject.toml
```

## Status

Phase 2 (vision→text bridge) shipped end-to-end on CI:

- [x] torch-first math core (backend interface, Procrustes, ridge, whitening, resamplers, interleaver, fused loss)
- [x] content-addressed feature cache + immutable manifests
- [x] CalibrationBridge (Procrustes-init ridge + whitening) + held-out eval with acceptance gates
- [x] CI: lint/typecheck/unit + `phase2-vision` experiment workflow (green)
- [x] First experiment campaign + write-up: `docs/PHASE2-RESULTS.md`
- [x] SPACE strict validation: cosine gates PASS on 9 real image-caption pairs
- [x] Stage 1b: VisionProjector (resampler + MLP) — retention gate PASS at 281.5%
- [x] Stage 2a: decoder alignment — LM loss 11.35→9.82, vision-caption cosine 0.78
- [x] Stage 2b: unfrozen decoder (4 layers) — LM loss 11.35→9.64, cosine 0.94
- [x] Generation quality eval: pipeline works end-to-end; generation needs 1K+ pairs (currently 9)
- [ ] Expand training data (1K+ image-caption pairs)
- [ ] Audio bridge + multi-branch fusion (Phase 3)

The first campaign ran the pipeline end-to-end on 24 synthetic pairs (all
gates below threshold — synthetic signal too weak). The SPACE strict
validation on 9 real photograph-caption pairs passed both cosine gates
(`cosine_ridge = 0.20`, above zero and random baselines), confirming the
alignment thesis works on real data. Stage 1b trained a VisionProjector
(AttentionPooling + MLP) that achieves 281% retention vs the ridge baseline,
producing soft tokens with cosine 0.94 against caption embeddings. Stage 2a
trained the projector to condition the frozen SmolLM2 decoder via vision prefix
tokens, reducing LM loss from 11.35 to 9.82. Stage 2b unfroze the last 4
decoder layers, further reducing loss to 9.64 while maintaining 0.94 cosine
alignment.
See `docs/PHASE2-RESULTS.md`.

See `PROPOSAL.md` for the staged plan and gates.

Related: `../Space` hosts the benchmark suite, the GGUF micro specialists, and
the Nexuss-AO late-fusion hub (production baseline). Nexuss-Fusion is the
research arm that tries to produce the single unified checkpoint.