# Nexuss-Fusion Proposal

A framework that fuses models of **different modalities and different
architectures into one native model** by making their representation spaces
compatible through mathematics, then fusing the three SPACE specialists into a
single checkpoint: **Nexuss-Fusion-360M**.

Status: proposal + mathematical core scaffold. Nothing trained yet.

---

## 1. Problem statement

We selected three always-on micro specialists that run comfortably on a small
CPU box:

| Modality | Model | Weights | Decoder dim / encoder dim |
|---|---|---|---|
| Text | SmolLM2-360M-Instruct | 0.27 GB | 960 (decoder) |
| Vision | SmolVLM2-500M-Video-Instruct | 0.44 GB | 768 (SigLIP) → 960 |
| Audio | Qwen3-ASR-0.6B | 0.80 GB | 896 (AuT) → 1024 (Qwen3 LM) |

They are **architecturally incompatible**:

- different computation graphs, layer counts, attention layouts, positional
  schemes (RoPE θ, interleaved), normalization statistics, vocabularies,
  quantization;
- their hidden features live in different vector spaces that differ by unknown
  rotations, permutations, scalings, and per-layer reparameterizations.

Direct weight merging (tensor average, task arithmetic, embedding substitution)
fails because averaging assumes **aligned, compatible coordinates** — which is
exactly what these models do not have. This is the conclusion of the Space
research (`../Space/research/fusion.md`, `final-synthesis.md`).

**Goal of Nexuss-Fusion:** fuse them into **one native model** anyway — by
solving the *representation-alignment* problem, not the *weight-averaging*
problem.

## 2. Core thesis

A single native multimodal model is achievable with only:

1. each frozen expert emitting **sequential features** `E_m(x_m) ∈ R^{T×d_m}`;
2. **paired calibration data** (same content through two experts) to estimate
   alignment maps;
3. an autoregressive **decoder LM** that defines the shared semantic space.

Compatibility across *any* architecture is manufactured in four mathematical
steps:

1. **Canonicalize** each space (whitening / RMS norm) so global scale, mean,
   and variance differences vanish.
2. **Align** each space to the decoder via an **optimal affine map** whose
   initial solution is the closed-form **Orthogonal Procrustes** estimate
   (SVD of the cross-correlation between paired features), then fine-tuned as a
   low-rank MLP that cannot collapse.
3. **Compress/unify sequence length** with learned **query resamplers**
   (Perceiver/Q-Former style cross-attention) so any `T_m` maps to a bounded
   token budget compatible with text.
4. **Interleave** everything as typed soft tokens into **one causal sequence**
   and train a single fused objective.

Because alignment is learned per-expert into a *target space* rather than by
matching *between* experts, the framework is architecture-agnostic. New
encoders (depth, thermal, IMU, music, code) plug in with their own
canonicalizer + projector + resampler.

Precedents the design rests on: BLIP-2 (frozen encoder + Q-Former + frozen
LLM), LLaVA (linear connector + instruction tuning), Flamingo (Perceiver +
gated cross-attention), OneLLM (progressive per-modality alignment), Qwen3-ASR
(cascade encoder → LM). Nexuss-Fusion's contribution is making the **alignment
mathematics** the explicit, primary mechanism — Procrustes-initiated affine
alignment, whitened canonicalization, and a single fused loss — so that
architecture identity never blocks fusion.

## 3. Architecture

### 3.1 Nexuss-Fusion-360M

```
text              image/video              speech
 │                  │                       │
 │           SigLIP (frozed 768)     Qwen3-AuT (froz, 896)
 │                  │                       │
 │           canonical_vision         canonical_audio
 │                  │                       │
 │           Procrustes-init projector 96  Procrustes-init projector 960
 │                  │                       │
 │           query resampler (≤64 tok) query resampler (≤12.5 tok/s)
 │                  │                       │
 └──────┬───────────┴───────────┬───────────┘
        │                       │
      <img> tokens           <aud> tokens
        └───────────────────────┼────────────────┐
                          SmolLM2 decoder (360M)
                                │
                       one causal next-token objective
                                │
                             text answer
```

Design decisions:

- **Base checkpoint:** `SmolVLM2-500M-Video-Instruct` already *is* a
  text(image/video) native model — its decoder is a SmolLM2-family 360M LM and
  its vision path (SigLIP + pixel-shuffle connector) is proven and lives in the
  same family as our text specialist. Starting from it gives native text+
  image/video for free. Option B (start from `SmolLM2-360M` and add both
  vision and audio bridges) is the fallback if we want full control of the
  vision bridge.
- **Audio bridge:** reuse the frozen Qwen3-ASR audio encoder (AuT, 896-dim,
  mel frontend). Its own "audio→text" projector (896→1024) is *not* reused;
  Nexuss-Fusion trains 896→960 into the SmolLM2 space. We use the encoder's
  **semantic hidden states**, not raw acoustic frames.
- **Speech interface:** initially the bridge is fed **ASR semantic states**
  aligned to transcripts (the exactness-preserving path). Transcript-text
  evidence (the late-fusion style) remains a fallback and a calibration source.
- **Markers:** canonical SmolLM2 tokenizer + new special tokens `<|image|>`,
  `<|audio|>`, `<|end|>`; embeddings resized once.
- **Frozen-first:** all experts frozen; only canonicalizers, projectors,
  resamplers, and later LoRA/scale-shift adapters are trainable.

### 3.2 Why this is a framework, not a one-off

`nexuss_fusion/math/*` and the `ModelSpec` descriptors decouple "what an
expert is" from "how we fuse it". Any expert that can emit `T×d_m` features and
any calibration corpus that pairs its content with decoder-language content can
be added: Qwen3-ASR-1.7B as audio-quality teacher, new encoders, future
modalities. LLMs of any width plug in as the hub decoder.

### 3.3 Explicitly out of scope (phase 0)

- Direct weight averaging / GGUF tensor merging.
- Any-to-any *generation* (speech/image output) — requires discrete codecs.
- Claiming the micro specialists natively understand each others' raw signals
  before the fused training happens.

## 4. Training plan (staged)

Curriculum follows Space's `research/training.md` and OneLLM's progressive
recipe, with projector-only first and specialists kept frozen.

| Stage | What is trained | Data | Gate to advance |
|---|---|---|---|
| 0 | Manifest, scorers, held-out splits, deterministic runners | all suites pinned | every metric reproducible from a manifest |
| 1a | Vision canonicalizer + Procrustes init (forecasting own alignment) | image↔caption calibration set | init reduces alignment error vs. random |
| 1b | Vision projector + resampler (vs frozen SmolVLM2 bridge) | image→caption pairs | ≥95% specialist unimodal retention |
| 2a | Audio canonicalizer + Procrustes init | speech↔transcript hidden states | alignment error lower than random init |
| 2b | Audio projector + resampler | speech→transcript pairs | exact WER within 10% relative of Qwen3-ASR |
| 3 | Combined soft-token system, decoder frozen | text+image, text+audio, all-three | ≥95% each specialist; ≤2% text replay regression |
| 4 | Distillation + replay + modality dropout | mixed instruction, teacher logits | 10% cross-modal gain over single-modality baseline |
| 5 | Selective unfreezing (LoRA, upper layers) | interleaved multi-turn | 90% of full-input quality on missing-modality subsets |
| 6 | Packaging + GGUF (research) + release decision | full eval | retention, groundedness, latency/RSS gates |

Objective at stage 3 and later:

```
L = L_answer
  + λ_feat · Σ_m MSE(norm(A_m · E_m(x_m)), norm(h_teacher_m))
  + λ_logit · T² · KL(teacher || student)
  + λ_align · L_contrastive
  + λ_replay · L_replay
  + λ_missing · L_missing
```

## 5. Datasets and "teacher data" plan

- **Calibration (Procrustes init):** paired (vision encoder states ↔ SmolLM2
  caption embeddings) and (Qwen3-ASR encoder states ↔ transcript text
  embeddings). Small: hundreds of examples suffice for a good linear init, and
  we can synthesize from public captions (CC0/Apache-safe subsets).
- **Alignment:** LibriSpeech/Common Voice slices (Apache/CC-by where
  permitted) for audio; public image-caption slices for vision (e.g. CC3M
  safe subset or DoCC using the frozen SmolVLM2 bridge as a weak teacher).
- **Distillation/mixing:** transcripts produced by **Qwen3-ASR-1.7B**, captions
  produced by **SmolVLM2-2.2B**, text data near the text micro's domain. All
  teacher-generated labels are confidence-gated and audited; hallucinated or
  contradictory outputs are dropped (never used as authoritative soft labels).
- **Replay:** fixed 15–25% text-only batches to protect SmolLM2 quality.
- **Missing-modality:** availability-conditioned dropout over all 2³ subsets,
  with calibrated abstention for exactness-critical tasks.

All licenses/provenance recorded per-sample (mirrors Space evaluation gates).

## 6. Evaluation gates (from Space research, applied)

| Area | Gate |
|---|---|
| Specialist retention | ≥95% unimodal quality; ≤2% text replay regression |
| Cross-modal value | ≥10% relative gain over strongest single-modality baseline |
| Missing modalities | ≥90% of full-input quality for nonessential omissions |
| Audio exactness | WER ≤10% relative above Qwen3-ASR on held-out clips |
| Grounding | shuffled-media score ≥5pt below true-media score |
| Systems | ≥45/54/30 tok/s paths; peak RSS ≤1.3× micro; cold ≤+20% |
| Reproducibility | clean rerun within 1pt; per-example artifacts manifest-pinned |

## 7. Compute plan (old CPU box)

- Projectors/resamplers: ~1–10M trainable params — fully CPU-feasible.
- Forward passes through frozen 500M–0.6B experts are slow on CPU but
  acceptable for a few-hundred-example calibration and alignment loops.
- Strategy: **cache all frozen expert features once** (`features cache`), then
  train only the small bridge heads on precomputed states — this turns CPU
  training into tiny-matrix work.
- Optional later: single T4 session (free tier) for the distillation/mixing
  stage; results land as checked-in artifacts, run through the same pins.

## 8. Repository plan

Phase 1 (this commit): proposal, mathematics, framework scaffold with working
primitive code + tests.

Phase 2: features-cache runners → calibration set → Procrustes init for vision.

Phase 3: projector/resampler training loop (CPU) + evaluator (WER / EM / ANLS).

Phase 4: audio bridge; combined sequence; fused loss; replay.

Phase 5: missing-modality + instruction stage; evaluation gates; GGUF research.

Phase 6: decide release vs. keep Nexuss-AO hub as production baseline.

## 9. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Connector learns shortcuts | paired-data splits by speaker/scene; shuffled-media controls; contrastive + replay |
| Catastrophic text forgetting | frozen decoder first; LoRA; 15–25% replay |
| Audio latency/context explosion | query resampling to ≤12.5 tok/s; chunking; cached features |
| Bridge cannot preserve exact ASR | gold-transcript fallback; confidence-gated distillation; WER gate |
| CPU training too slow | feature caching: train bridges on precomputed frozen states |
| Quantization breaks bridge | adapters run fp16/fp32 independent of GGUF; quantize only after calibration |

## 10. Open questions

1. Base: reuse `SmolVLM2-500M-Video-Instruct` (recommended) vs. start from
   `SmolLM2-360M-Instruct` and train both bridges?
2. Audio budget: fixed segment queries (4–16) vs. time-proportional ≤12.5/s?
3. Should the vision bridge replace SmolVLM2's connector (Option B) or keep it
   (Option A)?
4. Do we eventually want any-to-any generation (adds codecs — separate
   program)?

## 11. Decision

This proposal defines **Nexuss-Fusion** as the single-native-model research
track, running in parallel with the Nexuss-AO late-fusion hub which remains the
auditable production baseline. The framework's mathematics are the deliverable
of this commit; the three specialists are the first fusion target.