# Fusion Matrix — the three target models

Config-derived facts for the pilot fusion. Dims marked "verify" must be
confirmed by loading the actual checkpoints at calibration time.

## Text — SmolLM2-360M-Instruct

| Property | Value |
|---|---|
| Family | SmolLM2 (llama-layout) |
| Hidden `d_L` | 960 |
| Layers / heads / kv-heads | 32 / 15 / 5 |
| Intermediate / head_dim | 2560 / 64 |
| RoPE θ / interleaved | 100000 / false |
| Vocab | 49152 |
| Weights (q4_k_m) | 0.27 GB |
| Measured | 57.73 tok/s, 474 MB RSS, 37.7 s/10 prompts |

Role in fusion: the **native decoder** `L`. Unchanged during alignment.

## Vision — SmolVLM2-500M-Video-Instruct

| Property | Value |
|---|---|
| Family | smolvlm (`SmolVLMForConditionalGeneration`) |
| Text decoder | SmolLM2-family, hidden 960, 32L, vocab 49280 |
| Vision encoder | SigLIP, hidden 768, 12 heads, image_size 512, patch 16 |
| Connector | pixel-shuffle (`scale_factor=4`) + projector into 960 |
| image_token_id | 49190 |
| Effective visual tokens @512×512 | ≈ 64 (1024 patches / 4²) — verify |
| Weights (q8_0) | 0.44 GB |
| Measured | 67.15 tok/s, 1026 MB RSS, 73.7 s/10 imgs |

Role in fusion: Option A base (already native text+vision) OR a frozen SigLIP
source for a 768→960 bridge on `SmolLM2-360M`. Verify connector output width =
text hidden 960.

## Audio — Qwen3-ASR-0.6B

| Property | Value |
|---|---|
| Family | qwen3_asr (cascade) |
| Audio encoder (AuT) | d_model 896, downsample_hidden_size 480, conv_chunksize 500 |
| Own audio→LM projector | 896 → 1024 (Qwen3-0.6B LM width) — not reused |
| Input | mel (16 kHz, Whisper-style bins) — verify bins/stride |
| Languages | 52 |
| Weights (q8_0) | 0.80 GB |
| Measured | 37.17 tok/s, 2073 MB RSS, 24.1 s/10 clips |

Role in fusion: frozen audio encoder source; Nexuss-Fusion trains a NEW
canonicalizer + Procrustes-init 896→960 projector + query resampler
(≤12.5 tok/s or 4–16/segment). Gold transcripts remain the exactness fallback.

## Budget assumptions (verify at load)

```
text:   native token ids
vision: ≤ 64 soft tokens/frame
audio:  ≤ 12.5 soft tokens/s  (≈ 1 token/80 ms)  or fixed 4–16/segment
```

These keep the fused sequence text-like so the 360M decoder stays fast.

## Calibration pairings

| Bridge | Source states | Target states |
|---|---|---|
| vision | SigLIP 768 state rows (per image) | SmolLM2 caption embeddings (mean-pooled rows) |
| audio | Qwen3-ASR AuT 896 state rows (per clip) | SmolLM2 transcript embeddings (mean-pooled rows) |

Both produce the paired matrices `(H_m, H*)` needed for Procrustes init.