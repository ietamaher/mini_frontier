# Known Issues — open, not yet fixed

Ordered by priority. Each has the observed symptom and the recommended approach so a fix
doesn't start from scratch.

---

## 1. Tokenizer decode inserts spurious intra-word spaces  ✅ RESOLVED (2026-06-22)

- **Symptom (from `generate.py`):** `( نا )`, `ت علمت`, `اح رم`, `س منت`, `الك تيبة`.
- **Confirmed root cause (empirical):** the saved tokenizer had `decoder: None`,
  `pre_tokenizer: Whitespace`, and NO `continuing_subword_prefix`/`end_of_word_suffix`.
  So word-boundary info was discarded at encode time and the default decoder space-joins
  EVERY token. Proof: `الكلب يركض في` → pieces `['الكلب','يرك','ض','في',...]` → decode
  `الكلب يرك ض في` — the subword space (`يرك`↔`ض`) is indistinguishable from the real-word
  space (`ض`↔`في`). **A decode-only fix is impossible** — the information is not in the IDs.
- **Fix applied:** switched the tokenizer to **Metaspace** (SentencePiece-style `▁`
  word-boundary marker) in `tokenizer_arabic.py` — `pre_tokenizer = Metaspace(...)` +
  `decoder = MetaspaceDecoder(...)`. Retrained the 16k BPE on `data/shamela/stream.txt`,
  re-encoded all bins (`--encode_shamela_masked`). Round-trip is now clean; grammar terms
  are 1 token each; ~1.00 tok/word on grammar (was 1.42). Old tokenizer kept as
  `tokenizer/arabic_bpe_16k.OLD_whitespace.json`.
- **Cascade (because token IDs changed):** new `arabic_bpe_16k.json`; new bins
  (train 170,733,821 tok; val 4,132,711 tok); **old checkpoints are invalid** → full model
  retrain required. See operations-log §L.
- **Lesson:** any BPE tokenizer for a script without explicit word delimiters MUST use a
  boundary-preserving scheme (Metaspace `▁` or ByteLevel) AND set a matching `decoder`.
  Never ship a tokenizer with `decoder: None`. Verify with a decode round-trip before
  encoding the corpus.

## 2. Qur'an masking only anecdotally verified

- **Symptom:** a single `قال تعالى` prompt behaved well (no ﴿﴾, no confident fabricated
  verse), but that's one sample.
- **Recommended:** generate a batch of scripture-trigger prompts (`قال تعالى`, `بسم الله`,
  `إن الله`, `يا أيها الذين آمنوا`) at a few temperatures and measure how often the model
  produces confident long verse-like spans. Compare against the `checkpoints/no_masker/`
  baseline to quantify the masking effect.

## 3. `generate.py` reports two different param counts

- **Symptom:** logs `Paramètres : 10.52M` while the model banner says `Total: 14.62M`.
- **Cause:** `get_num_params()` subtracts the 4.1M token-embedding (16000×256), nanoGPT
  convention. The true size is 14.62M.
- **Fix (cosmetic):** either label it `non-embedding params` or print both; harmless as-is.

## 4. No long-range coherence / doesn't "answer" prompts

- **Symptom:** the model continues in-genre rather than reasoning or responding.
- **Status:** expected and documented (14.6M params, ~3.7 epochs). Not a bug. Only
  addressable by a bigger model / more tokens, which is out of current scope.

## 5. عروض (cat 33) has no validation signal

- **Status:** by design — too small to spare a val book, left entirely in train. عروض is
  excluded from the weighted val loss (Z renormalized to 0.985). Accept; revisit only if
  عروض becomes a priority.

## 6. MFU ~0.3% on T4

- **Status:** not a problem — a 14.6M model can't saturate a T4; it's launch/memory-bound,
  not compute-bound. Don't "optimize" this away.

## 7. No final eval/checkpoint at the last step  ✅ FIXED (2026-06-24)

Implemented in `train.py`: `is_last_step = (step == max_iters-1)` now forces both the eval
block and the checkpoint save on the final iteration. (Original entry below.)


- **Symptom:** eval+save run every 500 steps, so on a 10k-step run the last evaluated
  checkpoint is **step 9500**; steps 9550–9999 train but are never evaluated or saved
  (`ckpt_best.pt` = step 9500). The final ~500 steps' improvement (~0.02 loss) is discarded.
- **Fix:** in `train.py`, force an eval + checkpoint when `step == max_iters - 1` (or run a
  final eval after the loop). Trivial, free gain. Apply before the larger 50/80M run where
  the wasted tail is proportionally bigger if cadence isn't adjusted.

## 8. Qur'an fabrication persists despite loss-masking (frame leak)  ◑ mitigated at inference

- **Symptom:** generation still emits `قال تعالي: "..." [التوبة ١٤٩]` — confident but FABRICATED
  citations (fake verse text + fake sūra/āya reference).
- **Root cause (verified):** loss-masking set the VERSE target tokens (inside `﴿…﴾`) to -1, but
  NOT the citation FRAME (`قال تعالى`/`قوله تعالى`, ~3700 in a 40-file نحو sample) nor the bracket
  references. So the model (a) learned to emit the frame freely (never masked), and (b) got ZERO
  gradient on the first verse token → never learned a real continuation → at generation it fills
  the slot with hallucinated Arabic. Masking prevented *memorising real scripture* (output is
  fake, not real verses) but NOT *emitting citation-shaped output*. Note: normalisation ى→ي means
  the live token is `تعالي`, not `تعالى`.
- **Mitigation (inference, no retrain):** `generate.py` now bans at sampling every vocab token
  containing `تعالى/تعالي/﴿/﴾` (6 tokens) → the divine epithet can't be emitted → no citation
  frame can form. Added `model.generate(bad_token_ids=...)` (generation-only, v1 weights/training
  untouched) + `quran_frame_token_ids()` + `--allow_quran` to disable. Verified: the anomaly
  disappears; model stays in grammatical register. Residual: `قال الله` without `تعالى` is not
  banned (الله can't be banned), a weaker trigger.
- **Proper fix (next retrain):** extend the training mask to cover the FRAME tokens + references,
  not just the `﴿…﴾` verse span. Real cure = alignment/SFT ("don't fabricate scripture") at v2.
- **UPDATE 2026-06-24 (training fix implemented, behind flag):** `data_pipeline.py` now masks
  citation frames (`قال تعالى`/`قوله تعالى`/`قال الله تعالى`/`سبحانه وتعالى`/bare `تعالى`) in
  addition to `﴿…﴾`, via `--mask_frames` / `prepare_shamela_split_masked(mask_frames=True)`.
  Default OFF (v1 bins reproducible). Ḍād-v2 will encode with `--mask_frames`.

---

# STRATEGIC FORK — vocalized track vs general LM (de-vocalized)  ⚠️ decision pending (2026-06-22)

**The point (user):** a model can only *do* نحو if tashkīl is active. الإعراب IS the diacritic —
مرفوع = ضمة, منصوب = فتحة, مجرور = كسرة; علامة الإعراب is a حركة. Our current pipeline strips
tashkīl, so the model learns the *discourse about* grammar (the words مرفوع/في محل رفع/… are
content tokens) but cannot read a case ending or vocalize — its نحو is hollow.

**Three compounding causes (all in `normalize_arabic`, mini_frontier):**
1. `strip_tashkeel=True` removes every حركة → علامة الإعراب never reaches the model.
2. The corpus itself is only **partially vocalized** — per-category mean tashkīl density
   (from `arabic-corpus/staging/corpus_manifest.csv`): شعر 0.235, معاجم 0.174,
   **نحو/صرف 0.127**, لغة 0.097, عروض 0.092, أدب 0.075, بلاغة 0.069. Even نحو is ~1 char in 8
   vocalized → keeping tashkīl as-is would give an *inconsistent* signal (same word sometimes
   bare, sometimes marked), not a real إعراب engine.
3. `unify_alif`, `unify_ya` (ى→ي), and ة→ه ALSO erase grammatical/morphological signal:
   ة = تاء التأنيث, ى = اسم مقصور, إ/أ = همزة قطع/وصل. For a grammar model these must stay.

**Good news — reversible without re-extraction.** Diacritics + original forms still live in
`arabic-corpus/staging/clean`; stripping happens only at stream-build time (mini_frontier).
Rebuilding a vocalized stream = a `strip_tashkeel=False` + unifications-off variant of
`normalize_arabic`, then retrain tokenizer. No re-extraction.

**The fork (a goal decision, not a bug):**

| Goal | Pipeline | Verdict |
|---|---|---|
| General classical-Arabic LM | current de-vocalized | ✅ valid — the in-progress run is a fine baseline (don't call it a نحو model) |
| Real نحو / تشكيل engine | vocalized + unifications off + dense-vocalization corpus | ❌ impossible with the current pipeline |

**If the نحو/تشكيل track is chosen — recommended path:**
1. Keep tashkīl; disable at least ة/ه and ى/ي unification (new stream variant).
2. Bring in **Tashkeela** (densely/fully vocalized corpus, already named in project
   `CLAUDE.md §8`) as the diacritization backbone — Shamela's 0.13 is not enough alone.
3. Optionally add an explicit **diacritization objective**: bare input → vocalized output
   (model learns to *place* the حركة = to *apply* the نحو).
4. Keep Shamela's كتب الإعراب for step-by-step grammatical CoT (see the SFT/CoT plan).

**Current decision:** let the in-progress de-vocalized run finish as the generalist baseline;
the vocalized نحو/تشكيل track is a SEPARATE, later effort to be planned deliberately. Not started.

---

# SCALING ROADMAP — 50–80M tier + full Shamela extraction  ⏸️ ON HOLD (2026-06-23)

**Gate:** do NOT start this until the corrected Metaspace run (new bins) is validated as
healthy. The previous "8500" run was found to have trained on the OLD bins (see the
data-provenance recurrence) — first confirm a clean run on `train.bin`=170,733,821 tokens.

**Why scale:** at 14.6M params the model is already past Chinchilla-optimal on the
language-core (~150M unique tokens ≈ optimal for ~8M params). More raw text barely helps at
this size — the binding constraint is MODEL SIZE. Scaling model + data together is the real
quality lever. Full Shamela (~1.3B words ≈ ~1.6B raw tokens → ~1.0–1.2B after dedup) unlocks
exactly the 50–80M tier.

## Model configs (exact, derived from real arch — formula reproduces 14,618,880 exactly)

`P = 2·(vocab·d) + n_layer·(4·d² + 3·d·hidden + 2·d) + d`  (vocab=16k, no bias, untied emb)

| param | current 14.6M | **50M** | **80M** |
|---|---|---|---|
| n_embd (d) | 256 | **512** | **640** |
| n_layer | 8 | **10** | **12** |
| n_head | 8 | **8** | **10** |
| head_dim | 32 | **64** | **64** |
| hidden_dim (SwiGLU) | 704 | **1408** | **1728** |
| block_size | 512 | **1024** | **1024** |
| vocab_size | 16000 | 16000 | 16000 |
| **total params** | 14,618,880 | **48,507,392** | **79,970,560** |

Notes: head_dim 32→64 (32 too small); block_size 512→1024 costs 0 params (RoPE); keep vocab
16k at these tiers (32k only justified at 125M+, else emb+head dominates).

## Token budget (Chinchilla ~20×)

| tier | optimal tokens | Shamela (deduped) supply | verdict |
|---|---|---|---|
| 50M | ~1.0B | ~1.0–1.2B | ✅ ~1 epoch |
| 80M | ~1.6B | ~1.1B → ~1.5 epoch | ⚙️ repetition (≤4 ep) or +some MSA |

## Compute (honest, T4)

| tier | FLOPs vs 14.6M | ~tok/s (T4) | 1 epoch (~1B tok) |
|---|---|---|---|
| 50M | ~3.4× | ~45k | ~6 h |
| 80M | ~5.5× | ~28k | ~10–16 h |

Feasible on T4/RTX with checkpoint-resume across sessions (multi-day). >150M would need
rented A100. 50–80M stays within current hardware.

## Full extraction plan (~7,680 remaining books, 33 non-language-core categories)

State: 8,593 catalog books, 911 already extracted (cats 29–35). ~7,680 remain.

⚠️ **Decision before launch:** extracting everything changes the model's nature
(language-focused → general classical Arabic incl. religious sciences). hadith/tafsir/fiqh
bring heavy isnād repetition + Qur'an citations → dedup and Qur'an-masking become critical.
Either accept this shift or pick a subset.

Steps (corpus-prep work — belongs in `arabic-corpus`, not mini_frontier):
0. Generalize `scripts/extract_language_core.py` → `extract_categories.py` (any category
   list). Type B protocol: clean Shamela text, strip noise, keep `[PAGE n]`, write to a
   SEPARATE `full_manifest.csv` (don't disturb the validated language-core manifest).
1. **Pilot first** (CLAUDE.md §8): 1 book per new family (hadith/tafsir/fiqh/history),
   verify noise-stripping + `[QURAN-VERIFY]` + isnad_ratio + tashkil_density; human sign-off.
2. Batch per category → append manifest rows (real token_count, isnad_ratio, tashkil_density,
   license_status). `raw/` immutable.
3. Provenance/license (CLAUDE.md §2): flag `license: UNCLEAR`; no UNCLEAR text into clean/
   without sign-off.
4. **Cross-book dedup (the heavy, decisive step):** (a) exact-dup lines/paragraphs;
   (b) near-dup via MinHash+LSH (shared matn/isnād across shurūḥ). Expected −25–40%.
5. Stats/report: `scripts/corpus_stats.py` — tokens/category, dedup ratio, tashkīl density,
   `[QURAN-VERIFY]` count.

Expected: ~9–10 GB raw text; extraction ~1–3 h CPU; dedup a few hours; ~1.0–1.2B tokens
after dedup; final uint16 bins ~2.2–2.5 GB.

## Decision (a) — RESOLVED (2026-06-23): targeted prose expansion

Not "all categories". Expand into **classical prose in use** (the corpus is currently
meta-linguistic — books *about* Arabic; the model needs prose/verse *in* Arabic), in this
fixed priority order, **dedup-first**:

1. **التاريخ + التراجم + السير** (~200M raw tokens) — purest narrative prose, lowest
   Qur'an/isnād overhead, stylistically continuous with the existing أدب. Start here.
2. **شروح الحديث** (~129M) — commentary prose, rich in غريب and إعراب.
3. **التفسير** (~138M, selective, Qur'an-masking at full strength) — best for نحو
   (dense iʿrāb) but highest overhead (most Qur'an to mask + heavy isnād in narration-tafsīr).

**كتب السنة (hadith متون, 1243 books, ~248M raw) is OUT OF SCOPE** until cross-book dedup
is built and proven. It is the most redundant reservoir — the same matn repeats across
dozens of collections with different isnād; raw token count is massively inflated, unique
content is a fraction. Never treat its big number as "easy volume".

Rationale: the "3–5× tokens" figure is RAW, not unique. Real gain depends on the dedup
haircut (unmeasured yet). Realistic post-dedup+masking target ≈ 450–500M unique words
≈ ~600M tokens → comfortably covers the 50M tier, 80M with light repetition.

Caveat (catalog gap): `isnad_ratio`/`tashkil_density` are NOT populated for un-extracted
books (computed only at extraction time), so categories can't be pre-ranked by redundancy
without sampling — the dedup haircut is only knowable after extraction.

### Grammar floor — second clause of decision (a)

When rebalancing the mix toward general prose (grammar-core target drops from ~45% to
~25–30%, prose 50–55%, lexica ~15%), the **vocalized / explicit-grammar fraction gets its
own hard FLOOR in the mix config**, protected from the general-prose rebalance. The prose
expansion improves fluency but is (mostly) unvocalized — letting it dilute the already-thin
grammatical/vocalized signal would undercut the نحو goal. So: a minimum % reserved for the
نحو/صرف core (and, when the vocalized track lands, for vocalized data) that the prose ratio
cannot push below. Concretely: add a `MIN_GRAMMAR_FRAC` (and later `MIN_VOCALIZED_FRAC`)
guard in the weighting config; the rebalance fills the remainder, never the floor.

### Pilot gate before locking the ordering

Before committing the full ordering, **pilot 2–3 books each from التاريخ and التراجم**,
run them through the masking pipeline, and confirm two numbers:
1. **Actual token density vs the page estimate** (we used ~165 words/page × 1.25 — verify).
2. **How much isnād the stripping catches in التراجم specifically** — biographical
   dictionaries (طبقات/تراجم) can carry MORE chains than pure narrative تاريخ, so تراجم may
   behave more like the hadith dedup problem than expected.
If both hold, the ordering is locked. If تراجم turns out isnād-heavy, demote it below
تاريخ/سير within step 1.

**PILOT RESULT (2026-06-23) — ordering LOCKED.** Extracted 3 books each from تاريخ (cat 25:
9872, 9911, 401) and تراجم (cat 26: 5777, 14132, 5885) into `staging/_pilot_prose/` (isolated,
manifest untouched). Findings:
- **Token density beats the estimate:** تاريخ **246 tok/page (+49% vs 165)**, تراجم 175 (+6%).
  → the prose reservoir is BIGGER than the ~200M estimate (likely ~280–300M raw for hist+tarājim).
- **تراجم is NOT isnād-heavy:** isnād ratio ~0.001 in both cats → no demotion; ordering stays
  تاريخ/تراجم/سير together in step 1.
- Caveats: (1) the isnād regex (حدثنا/أخبرنا/عن X عن) may under-catch biographical-style chains
  (روى عنه…وعنه…) — eyeball 1–2 تراجم before generalizing; (2) تراجم is heterogeneous (book 5777
  = sparse 65 tok/page طبقات-style entries vs 14132 = dense 284) and notably more vocalized
  (tashkīl 0.202 vs تاريخ 0.015 — a bonus for the نحو/vocalization angle).

## Open decision (b)
- Target 50M first (safe, ~1 epoch) or 80M (needs repetition/MSA)?

**Status: not started. Resume after the corrected Metaspace training run is validated.**
