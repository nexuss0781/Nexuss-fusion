# The Mathematics of Nexuss-Fusion

Framework-level mathematics for fusing models of different modalities and
architectures into one native autoregressive model, via representation
alignment instead of weight averaging.

Notation:

- `m` — modality (text, image, audio, …).
- `E_m : X_m → R^{T_m × d_m}` — frozen expert producing sequential features.
- `L` — the decoder LM (hidden width `d_L`), the "shared semantic space".
- `P_m : R^{d_m} → R^{d_L}` — trained per-modality projection.
- `C_m` — canonicalizer (`R^{d_m} → R^{d_m}`), whitening / scale harmonization.
- `R_m` — query resampler compressing `T_m` tokens into a budget `b_m`.
- `G` — the fused embedding sequence fed to `L`.

---

## 1. Why weight averaging fails (formal)

Two experts rarely share a coordinate system. Hidden features of model families
differ by unknown bijections `φ_m` on the representation manifolds. A layer's
parameter tensors are coordinates *of* a transformation `f_m`; averaging
`αf_A + (1−α)f_B` in coordinates is only meaningful if `φ_A = φ_B = id`, i.e.
the models were trained from a shared initialization in a shared parameter
space (Model Soups, Task Arithmetic conditions). For independently trained
models that condition does not hold, so tensor averaging is not even a
well-defined operation on the spaces we care about.

Nexuss-Fusion replaces "average the weights" with "align the spaces":

```
            φ_A                                φ_B
   space_A ───────→ shared semantic space ←─────── space_B
       →  z_A  =  P_A(C_A(E_A(x_A)))          z_B  =  P_B(C_B(E_B(x_B)))
```

Define compatibility as: there exist bounded, trainable maps
`P_m ∘ C_m` such that content-matched examples land near each other in
`space_L` (the decoder embedding space), measured by a fusible norm. This
definition is satisfied for *any* two architectures as long as paired data
exists; no layer match, initializer match, or vocabulary union is needed.

## 2. Step 1 — Canonicalization (whitening / scale harmonization)

Each space is normalized so its global statistics no longer dominate, while
information along every principal direction is preserved (full-rank affine):

```
μ_m  = mean_t h_m(t),   σ_m = std_t h_m(t)          per feature channel
C_m(h) = (h − μ_m) / (σ_m + ε)                     (plus learned per-channel scale γ_m)
```

This diagonal reparameterization make inner products and attention scores
comparable *across* spaces before any cross-space map is applied. It also
defuses the "normalization statistics differ" incompatibility highlighted by
the prior research.

## 3. Step 2 — Optimal linear alignment (Procrustes-initiated)

The heart of "make any architecture compatible".

**Setup.** We have paired data: the same semantic content `c` through the
source expert and through the decoder's own representation `h*` (e.g. for the
audio bridge: Qwen3-ASR encoder states aligned with SmolLM2 transcript
embeddings). Collect matrices

```
H_m ∈ R^{T×d_m}    (source features, whitened)
H*  ∈ R^{T×d_L}    (target features, e.g. text embedding rows)
```

**Goal.** Find the best linear map `A_m` minimizing

```
min_A ‖ H* − H_m A ‖_F² + λ ‖A‖_F²            (Ridge-regularized least squares)
```

**Closed form (general solution).**

```
A = (H_mᵀ H_m + λ I)⁻¹ H_mᵀ H*
```

**Orthogonal-constrained (Procrustes) initialization.** Restrict to an
orthogonal+scale step so the map is a rigid motion plus isotropic scale (the
safest prior: no anisotropic distortion initially). Let the cross-correlation
be `H*ᵀ H_m = U Σ Vᵀ` (thin SVD). Then the optimal orthogonal map is

```
R_m = U Vᵀ
```

and the best isotropic scale is `s_m = tr(Σ)/‖H_m‖_F²`. Starting the trainer at
`P_m(t=0) = s_m R_m` (Option: `A` from Ridge LS) gives a solution that already
encodes the true cross-space geometry, so gradient fine-tuning refines rather
than searches from scratch, and it cannot collapse (orthogonality retained
under small updates / re-projection).

**Fine-tuning.** After the closed-form init, `P_m` is continued as a low-rank
MLP `P_m(z) = W₂ · gelu(W₁ z + b₁) + b₂` with `W₁` initialized from the
autoencoder-aligned basis so incremental capacity adds nonlinear corrections.
`λ` and the SVD rank threshold set the effective capacity.

**Why this is universal.** The only data requirement is paired content
`(x_m, y*)`; every modality has a natural pairing (image↔caption, speech↔
transcript, camera↔audio event). Encoder width / decoder width are arbitrary;
`A_m` always exists as a `d_L × d_m` linear map. Architecture differences
(graphex, ROPE, heads, quant) never enter this equation.

## 4. Step 3 — Sequence-length unification (query resampler)

Even after projection, a 30 s clip yields ~375 raw audio state rows while an
image yields ~1024 SigLIP patches — incompatible with text-like attention
budgets. Learned query resamplers fix the token count per segment:

```
Z_m = Softmax((Q W_q)(K W_k)ᵀ / √d) · (V W_v)          Q ∈ R^{b_m × d_q}
K, V = from projected source states H_m P_m
```

- `b_m` is a per-modality budget: vision `≤ 64` tokens/segment (SmolVLM2 uses
  ~64 at 512×512 via pixel-shuffle ×4); audio `≤ 12.5` tokens/s (Qwen3-ASR
  downsample `conv_chunksize=500`, then queried) or a fixed `4–16` per segment.
- This is mathematically the Perceiver/Q-Former pooling; it bounds `seq_len` so
  the fused native sequence stays text-like regardless of source length.

## 5. Step 4 — One native sequence

Canonical text tokenizer + new special markers `⟨image⟩`, `⟨audio⟩`, end
markers; embedding matrix resized once (HF-compliant). The fused input is a
single causal sequence of text embeddings and soft modality tokens:

```
S = [t_1 … t_k] ⟨image⟩ Z_v ⟨audio⟩ Z_a [t_{k+1} …]
```

`L` applies ordinary causal next-token prediction (shared attention, shared
vocabulary, single objective). No gated cross-attention is required at first:
we are *injecting* aligned soft tokens into self-attention, exactly like the
since-proven SmolVLM vision path. Gated cross-attention (Flamingo-style) is the
reserve mechanism only if ablations show lost grounding/temporal evidence.

## 6. The fused objective

At stage ≥ 3:

```
L = L_answer
  + λ_feat   Σ_m MSE(norm(P_m C_m E_m(x_m)), norm(h_teacher_m))
  + λ_logit  T² KL(p_teacher ‖ p_student)
  + λ_align  L_contrastive           (paired positives, in-batch negatives)
  + λ_replay L_replay                (fixed text-only buffer)
  + λ_missing L_missing              (availability-conditioned subsets)
```

- `L_answer`: CE on assistant tokens only.
- Feature/logit terms ramp from 0 after warm-up; teacher targets are
  confidence-gated.
- Replay ratio 15–25% with a stratified text/specialist/rare-hard buffer.
- Missing modalities: explicit availability mask per sample over all non-empty
  subsets; abstain/request when the missing signal is exactness-critical.

## 7. Compatibility for emerging architectures (the framework promise)

The pipeline for a new expert `E_new` is: declare its spec (`hidden_dim`,
`token length law`), canonicalize, paired calibration, Procrustes init, small
trainable projector+resampler. No architectural constraint on `E_new`
whatsoever. If a future expert has *no* natural text pairing (e.g. IMU),
pairing is created through a second expert (ImageBind-style transitive
alignment): calibrate `E_new` against an expert already fused.

## 8. Optional advanced: layerwise bridge (weight-level cross-family fusion)

For the case where a genuinely *shared decoder trunk* is built by combining
two LMs of the same width but different training, Nexuss-Fusion falls back on
operator-compatible reparameterization:

1. resample both models to a canonical parameterization (RMSNorm, common
   head_dim, common RoPE);
2. initialize bridges `B_l` per layer to transform `state_l` of source into
   `state_l` of target via the same Procrustes machinery (layer-output pairs);
3. gate toward zero (Flamingo gate) and anneal.

This is a separate, gated experiment, not the phase-1 path.

## 9. Summary equations (cheat sheet)

```
whiten:        C_m(h) = (h − μ_m)/(σ_m + ε)
procrustes:    H*ᵀ H_m = U Σ Vᵀ   →   P_m,0 = s·UVᵀ
ridge-LS:      A = (H_mᵀ H_m + λI)⁻¹ H_mᵀ H*
reproject:     z_m = P_m(C_m(E_m(x_m)))
resample:      Z_m = Attn(Q, KV=z_m) ∈ R^{b_m × d_L}
sequence:      S = [text] ⟨image⟩ Z_v ⟨audio⟩ Z_a [text] …
objective:     L = L_ans + λ_f L_feat + λ_l L_KL + λ_a L_align + λ_r L_replay + λ_m L_missing
gate:          95% specialist retention · ≤2% text replay regression ·
               10% cross-modal gain · WER ≤ 10% rel. above Qwen3-ASR
```