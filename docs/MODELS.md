# Ḍād — Model Registry

Single source of truth for every released/planned version. See
[VERSIONING.md](VERSIONING.md) for the scheme and release process.

Name: **Ḍād** (ضاد) — *لغة الضاد*, the language of the ḍād (= Arabic).

---

## Versions

| Version | Status | Params | Tokenizer | Corpus (unique) | ValW | Notes |
|---|---|---:|---|---:|---:|---|
| **Ḍād-v1** | ✅ validated baseline | 14.62M | BPE 16k **Metaspace** | 170.7M tok | **4.947** | decoder-only LLaMA-style; نحو strongest; Qur'an-masked |
| **Ḍād-v2** | 🔜 planned | ~46.7M (Config C, **tied**) | BPE 16k (same family) | ~0.9–1.0B tok target | goal **≤ 4.0** | see [RESEARCH_v1_to_v2.md](RESEARCH_v1_to_v2.md) |

---

## Ḍād-v1 — reproducibility record

| Field | Value |
|---|---|
| **Code commit** | `d1bd116` (to be tagged `Ḍād-v1`) |
| **Tokenizer sha256** | `b39c30820893f459…` (`tokenizer/arabic_bpe_16k.json`) |
| **train.bin** | 170,733,821 tokens · sha256 `c639aadcce54824f…` · mask 1.118% |
| **val.bin** | 4,132,711 tokens · book-level split, zero leakage |
| **Config** | n_embd 256 · n_layer 8 · n_head 8 · head_dim 32 · hidden 704 · block 512 · vocab 16000 · untied · no bias |
| **Recipe** | AdamW, peak LR 1.5e-4 → 1.5e-5 cosine, 400 warmup, wd 0.1, grad-clip 1.0, 10k steps, fp16+GradScaler (T4) |
| **Tokens seen** | 655M (≈3.84 epochs) |
| **Seed** | `SHAMELA_SHUFFLE_SEED = 42` |
| **Best checkpoint** | step 9500 → `checkpoints/metaspace_baseline_step9500.pt` (local safety copy) |
| **Eval (per-category)** | نحو/صرف 4.666 · بلاغة 4.839 · معاجم 4.976 · لغة 5.254 · أدب 5.353 · شعر 5.717 |
| **Known limits** | under-capacity looping; poetry weakest (diacritics stripped); 56% params in embedding/head |

> ⚠️ **Cross-version note:** Ḍād-v1's ValW (4.947) is **not comparable** to the pre-Metaspace
> figure (4.674) — different tokenizer = different per-token loss scale. Only compare ValW
> **within the same tokenizer family**. Across tokenizer changes, judge by generation quality.

---

## Ḍād-v2 — planned spec (summary; full design in RESEARCH_v1_to_v2.md)

- **Config C:** n_embd 512 · n_layer 12 · n_head 8 · hidden 1408 · **weight-tied** → 46.7M total,
  38.5M non-embedding (6× v1's compute params), embedding share 17.5%.
- **Corpus:** ~0.9–1.0B unique tokens (Chinchilla-matched). Gating dependency = corpus expansion
  (targeted prose: تاريخ/تراجم/سير → شروح حديث → تفسير, dedup-first; + possibly Hindawi/Wikipedia).
- **Recipe:** larger micro-batch (MFU headroom), restored LR ~3e-4, final-step eval fix (known-issues #7).
- **Success:** ValW ≤ 4.0, poetry < 5.0, no degenerate looping, embedding share < 20%.
- **Open decision (b):** ship 50M (Config C) first, or stretch to ~70–80M (Config D) — pending.
