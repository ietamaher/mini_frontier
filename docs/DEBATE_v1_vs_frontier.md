# Ḍād v1 vs. Today's Models — Debate Memo & Enhancement Shortlist

**Companion to** `research/minifrontier.tex` and `docs/RESEARCH_v1_to_v2.md`.
**Date:** 2026-06-23 · **Subject:** Ḍād-v1 (14.62M, ValW 4.947, BPB 1.25 measured)

> This memo captures the analytical debate held over Ḍād-v1: *(1)* is 4.95 nats the
> best we can do on this data? *(2)* what do today's frontier models actually reach,
> and how far are we? *(3)* are we even using a modern architecture? It is grounded
> in the **now-measured** anchors (`bpb_results.json`, `scaling_results.json`) rather
> than estimates, and ends with a **ranked shortlist of enhancements** — including a
> free-lunch win available *before* v2.

---

## TL;DR

1. **4.95 is not the floor — but the data caps the big wins.** v1 was still descending at the last step, the train/val gap is small (~0.25 nats), and 56% of its parameters never compute. There is recoverable headroom from *architecture and recipe* before data becomes the hard wall. The exact data ceiling is **measurement-pending** (the scaling sweep harness is validated but only the 5M/200-iter pilot has run).
2. **The frontier gap is far narrower than perplexity suggests.** On the portable ruler, v1 sits at a **measured 1.25 bits/byte** vs frontier ≈0.6–0.8 bpb — an *efficiency* gap of ~1.7×, part of which is the task (stripped classical Arabic over a 16k vocab is intrinsically high-entropy per byte), not the model. The real difference is that frontier models sit *at* the entropy floor (= fluency); v1 sits *above* it (= looping).
3. **The architecture is current, not outdated.** Ḍād-v1 is a small LLaMA; its skeleton is identical to 2024–26 frontier models. The deltas (GQA, MoE, long-context RoPE) are efficiency/scale features that would **not** raise quality at 14–50M params.
4. **Biggest non-obvious finding:** a **weight-tied retrain at the *same* 14.6M size** gives **+63% compute parameters for free** (6.42M → 10.44M) at identical inference cost — a cheap "v1.5" that de-risks v2 and should noticeably beat v1 on the same corpus and hardware.

---

## Debate thread 1 — "Is 4.95 the best we can do with this data?"

**No, but be precise about *which* ceiling.** Three tiers:

| Tier | Lever | On the same 170M data? | Expected |
|---|---|---|---|
| Free wins | weight tying + longer/bigger-batch run, same ~15M model | yes | ValW ~4.7–4.8 |
| Capacity | scale to ~47M (v2-C) on the same data | yes, ~4–6 epochs | ValW ~4.3–4.6, overfitting risk rises |
| Data | grow unique corpus to ~1B | no — needs new data | breaks ≤4.0, route to fluency |

**Evidence it isn't model-saturated yet:**
- The v1 ValW curve was **still falling** at step 9500 (4.955 → 4.947 over the last 500 steps), with LR already at floor — an under-trained tail, not convergence.
- The **train/val gap is only ~0.25 nats** (train ≈4.7 vs ValW 4.947). A data-saturated 15M model would show train collapsing while val stalls — it doesn't. Capacity isn't exhausted on these 170M tokens.
- **Data-constrained scaling** (repeating data is near-free up to ~4 epochs, decaying after) places v1 at **3.84 epochs** — right at the edge of the free-repetition zone. ~1 more epoch of near-free value remains in this corpus; beyond that, *more unique data* is mandatory.

**Status of the empirical test:** the fixed-corpus scaling sweep (`scaling_sweep.py`) is the instrument that turns this from argument into a number. Harness validated on the **5M pilot** (200 iters → ValW 8.44, as expected for a barely-trained run); the real 4-epoch sweep across {5M, 15M, 30M, 47M} on the RTX 3090 is **pending** and will pin the crossover size where 170M tokens stop paying off.

---

## Debate thread 2 — "What do frontier models reach, and how far are we?"

**Per-token loss is not portable** — comparing v1's perplexity 141 to a frontier model's ~2 nats is meaningless (different tokenizer, vocab, language, units). The portable metric is **bits-per-byte**, and we now have it **measured** (per-token UTF-8 decode, mask-aware, not assumed):

| | v1 (measured) | Frontier (English, reported) |
|---|---|---|
| Bits-per-byte | **1.25 bpb** | ~0.6–0.8 bpb |
| Position vs text entropy floor (~1 bit/char) | **above** it (looping) | **at** it (fluent) |

So the honest gap is an **efficiency ratio of ~1.7×**, not the ~70× the raw perplexities imply — and part of even that 1.7× is the task itself.

**The measured per-category BPB exposes the single most actionable finding:**

| Category | nats/tok | bytes/tok | **BPB** |
|---|---:|---:|---:|
| بلاغة rhetoric | 4.84 | 6.44 | **1.09** |
| نحو/صرف grammar | 4.68 | 6.00 | 1.12 |
| معاجم lexicons | 4.98 | 5.92 | 1.21 |
| لغة language | 5.27 | 6.15 | 1.24 |
| أدب literature | 5.33 | 5.64 | 1.36 |
| **شعر poetry** | 5.70 | **5.05** | **1.63** |

Poetry is **both** the highest-loss **and** the least byte-efficient domain — its tokens are short, diacritic-bearing fragments stripped to bare consonants. This is hard quantitative proof that **tashkīl stripping caps quality exactly where vocalisation carries the signal**, and it points to a concrete fix (below).

**The meaningful target** is not a frontier model's English loss; it is the **entropy floor of *our* corpus and tokenizer** — which, for stripped classical Arabic, likely sits *higher* than the MSA figures usually quoted. We deliberately do not assert "3.5 nats" as settled; the sweep + BPB are the anchors that will locate it.

---

## Debate thread 3 — "Are we using the same architecture as today's models?"

**Yes.** Ḍād-v1's skeleton is identical to the 2024–26 frontier recipe; nothing is outdated.

| Component | Ḍād-v1 | Frontier 2024–26 | Same? |
|---|---|---|---|
| Topology | Decoder-only | Decoder-only | ✅ |
| Norm | RMSNorm pre-norm | RMSNorm pre-norm | ✅ |
| Positions | RoPE | RoPE + scaling | ✅ core |
| FFN | **Dense** SwiGLU | SwiGLU, often **MoE** | ⚠️ |
| Attention | Full **MHA** | **GQA/MQA/MLA** | ⚠️ |
| Kernel | Flash-Attention | Flash-Attention 2/3 | ✅ |
| Biases | None | None | ✅ |
| Vocab | 16k Arabic | 128k–256k multiling. | ⚠️ scale |

The three real deltas — **GQA, MoE, long-context RoPE** — are **efficiency/scale** features, not quality features at this size:
- **GQA** shrinks a KV-cache that is already negligible at 512 context → ~0 quality gain now.
- **MoE** starves each expert of tokens below the B-param / T-token regime → counterproductive at 50M.
- **Long-context RoPE** is irrelevant until `block_size` grows past 512.

The distance to the frontier is **scale + post-training (SFT, RLHF/RLAIF — neither begun)**, not the network design. The one scale-appropriate borrow worth an ablation is **QK-norm** (a few lines, stabilises the higher LR a larger batch permits).

---

## Noticeable enhancements — ranked by ROI

Ordered by (impact ÷ cost). Items 1–3 are cheap and partly available *before* the full v2.

### ★ 1. Weight tying — a free lunch, even at v1 size
Tying `lm_head` to `tok_emb` costs nothing and, because the 16k embedding is 56% of v1, redeploys that capacity into computation. **At the same ~14.6M total size**, a tied retrain buys dramatically more "thinking" parameters at **identical inference cost and speed**:

| Variant | Total | Compute params | Emb share | Compute gain vs v1 |
|---|---:|---:|---:|---:|
| v1 (untied, L8) | 14.61M | 6.42M | 56.1% | — |
| v1.5 tied, L11 | 12.93M | 8.83M | 31.7% | **+38%** |
| v1.5 tied, L12 | 13.73M | 9.63M | 29.8% | **+50%** |
| **v1.5 tied, L13** | **14.53M** | **10.44M** | 28.2% | **+63%** |

> **Recommendation:** run **v1.5 = tied, n_embd 256, n_layer 13** on the *same* 170M corpus and recipe. Same size, same speed, +63% compute capacity. This is the cleanest controlled experiment we can run, should noticeably beat ValW 4.947, and validates tying before committing v2 to it. Expected: a real step down in loss with zero hardware change.

### ★ 2. Exploit the idle GPU (MFU 0.3%) → bigger batch + restored LR
The T4 ran at **0.3% MFU** — almost entirely launch/overhead-bound. Raising the micro-batch (16 → 48–96) costs little wall-clock and gives calmer gradients, which in turn **re-enables peak LR 3e-4** (the 1.5e-4 cap existed *only* because the 65k-token batch was noisy). Net: more tokens/hour **and** better optimisation, for free.

### ★ 3. Context 512 → 1024 — directly attacks the looping
The degenerate enumeration is an under-capacity + short-history failure. Doubling context gives the model more discourse to condition on; VRAM headroom is ample at this scale. Cheap, and targets the most visible quality defect.

### 4. Grow unique data to ~1B tokens — *the* lever for v2
The #1 constraint for any model >15M. Routes: targeted Arabic prose (تاريخ/تراجم/سير → شروح حديث → تفسير, dedup-first), Hindawi, Arabic Wikipedia, and synthetic textbook-style data. **De-duplicate before counting** — repeats inflate tokens without signal. Gating dependency for Config C.

### 5. Diacritised poetry/vocalised sub-corpus — fixes the measured BPB outlier
The 1.63-bpb poetry result is hard evidence that stripping tashkīl removes the signal poetry needs. Add a vocalised sub-corpus (vocalised dīwāns, the Tashkeela corpus) so diacritics are at least in-vocabulary. Higher-ceiling variant (tashkīl as first-class tokens throughout) is a v2.x experiment — it roughly doubles sequence length but is the only true path to "correct" vocalised Arabic.

### 6. Tokenizer compression — consider 16k → 24–32k vocab (ablate)
Measured compression is only **~5.86 bytes/token ≈ 2.9 Arabic chars/token** — low. A larger Arabic vocab would pack more bytes per token, lengthening effective context and likely lowering BPB. With **weight tying** the extra embedding rows are cheap. Caveat: it shifts the per-token loss scale, so judge by **BPB and generation**, not raw ValW. Worth one ablation arm.

### 7. QK-norm — cheap stability borrow from modern models
Normalise Q,K before attention. A few lines; buys headroom for the higher LR that item 2 enables. Low risk, small but real.

---

## What is measured vs. still open

| Claim | Status |
|---|---|
| ValW 4.947, per-category losses | ✅ measured (step 9500) |
| BPB 1.25 overall + per-category | ✅ measured (`bpb_results.json`) |
| 56% embedding share / +63% tied free-lunch | ✅ computed (exact) |
| Architecture parity with frontier | ✅ established |
| Frontier ≈0.6–0.8 bpb | ◐ from literature (English) — not re-measured here |
| Data ceiling of the 170M corpus | ⬜ **pending** full scaling sweep (pilot only) |
| v2 → ValW ≤4.0 / ppl 33–55 | ⬜ projection, explicitly not a settled floor |

---

## One-line verdict

> **The architecture is already modern and the frontier gap is narrower than the perplexity implies; what holds Ḍād back is capacity wasted on an untied embedding, a small corpus, and an idle GPU. Tie the embeddings (free, +63% compute even at v1 size), feed the GPU a bigger batch, grow the data toward 1B tokens, and restore diacritics where they carry signal — in that order.**
