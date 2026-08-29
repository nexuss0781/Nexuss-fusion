# Nexuss-Fusion — Architecture

High-level system design. See `MATHS.md` for the mathematics and
`PROPOSAL.md` for the plan and gates.

## 1. System diagram

```
                 ┌──────────────────────────────────────────────┐
                 │              MODEL SPECS (registry)          │
                 │  text   vision   audio   …future…            │
                 └──────────────┬───────────────────────────────┘
                                │
   frozen experts               v
  E_text   E_vision   E_audio    →  features cache (precomputed, on CPU/GPU)
                                │
                 ┌──────────────┴───────────────────────────────┐
                 │          ALIGNMENT LAYER (trainable)         │
                 │  canonicalize → procrustes-init projector    │
                 │  → query resampler → typed soft tokens       │
                 └──────────────┬───────────────────────────────┘
                                │
                 ┌──────────────┴───────────────────────────────┐
                 │          ONE NATIVE SEQUENCE (L decoder)     │
                 │  [text] ⟨image⟩ Z_v ⟨audio⟩ Z_a [text] …     │
                 │  causal next-token objective                  │
                 └──────────────┬───────────────────────────────┘
                                │
                             text answer
```

## 2. Components

### 2.1 Spec registry (`nexuss_fusion/specs/`)
Declarative descriptors so any model can be registered without touching the
fusion machinery: `hidden_dim`, token-length law, source, license, tokenizer,
expert kind. The three SPACE targets are pre-registered
(`nexuss_fusion/specs/models.py`); adding a model = adding a spec + a loader.

### 2.2 Math primitives (`nexuss_fusion/math/`)
Pure/reference implementations (numpy-first, torch-adaptable):
- `procrustes.py` — orthogonal/scale init via thin SVD, ridge least squares,
  low-rank MLP init helper.
- `normalize.py` — whitening canonicalizer (population stats + learned scale).
- `resample.py` — Q-Former-style AttentionPooling with a budget `b_m`.

### 2.3 Sequence layer (`nexuss_fusion/sequence/`)
`interleave.py` builds the unified embedding sequence and causal block-attention
structures from typed blocks. `markers.py` manages special-token registration
and embedding resize.

### 2.4 Losses (`nexuss_fusion/losses/`)
Reference `losses.py` implementing the fused objective's signature so the pieces
can be wired and unit-checked before any training runs.

### 2.5 Train/eval (future phases)
Features-cache runner → calibration set builder → alignment trainer
(projector-only) → evaluator (WER / exact-match / ANLS / CIDEr listeners).

## 3. Target fusion: Nexuss-Fusion-360M

- Decoder `L` = SmolLM2-family 360M LM (960-d, 32 layers), kept as the native
  text engine. Recommended base checkpoint: `SmolVLM2-500M-Video-Instruct`
  (its decoder *is* the SmolLM2-family 360M and the vision path is already
  native). Option B: start from `SmolLM2-360M-Instruct` and train both bridges.
- Vision: reuse frozen SigLIP path of SmolVLM2 when on Option A; else train a
  768→960 Procrustes-init bridge + resampler budget ≤64 on `SmolLM2`.
- Audio: frozen Qwen3-ASR AuT encoder (896) → canonicalize → Procrustes-init
  projector 896→960 → query resampler ≤12.5 tok/s (or fixed 4–16 per segment).
- Text stays the native SmolLM2 path untouched during alignment stages.

## 4. Training stages (summary)

| Stage | Trains | Frozen | Data |
|---|---|---|---|
| 0 | — | all | pins, scorers, splits |
| 1 | vision bridge | experts + L | image↔caption calibration |
| 2 | audio bridge | experts + L | speech↔transcript calibration |
| 3 | both bridges end-to-end | experts | text+img, text+aud, all |
| 4 | LoRA on L | experts | mixed instruction + replay + teacher logits |
| 5 | scale/shift + dropout | experts | all modality subsets |

Replay: 15–25%. Feature caching keeps CPU training tractable.

## 5. CI / reproducibility

- Deterministic fixture record + llama.cpp-pinned builds (Space
  `research/github-actions.md` applies).
- Every training/calibration run writes `manifest.json` (model specs, hashes,
  dataset split, seed, features-cache digest) and per-example artifacts.
- Smoke workflow stays label-accurate: alignment *training* is never called
  "inference" and vice versa.