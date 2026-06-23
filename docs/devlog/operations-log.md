# Operations Log

Chronological record of every operation. Newest section last. Dates from file
timestamps / session history (June 2026).

---

## A. Corpus catalog (arabic-corpus)

- Built `staging/catalog.csv` via `scripts/build_catalog.py` — **8,593 rows**, of which
  **8,590 downloaded** (3 pending, none in the focus categories).
- Columns: book_id, title_ar, category_id, category_name_ar, download_status, extracted,
  page_count, token_count (whitespace estimate), byte_size, author_ar, death_year_hijri,
  isnad_ratio, tashkil_density. Plus per-category summary table + flags
  (category mismatches, top-10 isnad, duplicate titles).
- Constraints honored: read-only, no network, no new extraction, no renaming.

## B. Language-core extraction (arabic-corpus)

- `scripts/extract_language_core.py` extracted **only categories 29–35** (language core);
  the other 33 categories were left untouched.
- Result: **911 books, 122.9M whitespace-tokens**, appended to
  `staging/corpus_manifest.csv` (real token counts, `[PAGE n]` markers kept, blank
  `mix_weight` column).
- Per-category token totals printed.

## C. Mix weights (arabic-corpus → consumed by mini_frontier)

- Filled `mix_weight` per category in the manifest:
  `29:2.57, 30:0.56, 31:2.57, 32:0.45, 33:4.0, 34:3.8, 35:1.83`.
- Weighting is by **document repetition during sampling**, NOT on-disk duplication:
  a book with weight `w` is emitted `floor(w)` full passes + a `(w - floor(w))` fraction
  of its lines.
- Target effective distribution achieved: نحو 45.5%, أدب 22.1%, معاجم 15.2%, لغة 7.1%,
  بلاغة 7.1%, شعر 2.5%, عروض 0.5%.

## D. Pipeline + tokenizer (mini_frontier)

> ⚠️ First attempt mistakenly created these files in `arabic-corpus/` (wrong repo) —
> deleted. See bugs-and-gotchas #1.

- `config.py`: added `ARABIC_CORPUS_DIR`, `SHAMELA_STAGING`, `CORPUS_MANIFEST`,
  `SHAMELA_STREAM_DIR`, `SHAMELA_STREAM`, `SHAMELA_SHUFFLE_SEED=42`,
  `CATEGORY_WEIGHTS`, `VAL_EVAL_MIX`, `VAL_CAT_NAMES`, `val_cat_bin()`.
- `data_pipeline.py` extended: `_scan_shamela_clean`, `_content_lines`, `_stream_book`
  (block-based, RAM-safe normalization), `build_weighted_stream`, `_load_manifest_rows`,
  `select_val_books`, `_write_chunks`, `encode_stream_to_bin`, `_count_tokens`,
  `prepare_shamela_split`, `emit_val_category_bins`.
- `--prepare_shamela` → weighted **de-vocalized** text stream (tashkīl stripped,
  alif/ya/waw unified) used to train the tokenizer.
- Retrained the 16k BPE tokenizer on that stream → `tokenizer/arabic_bpe_16k.json`
  (the previous tokenizer was a throwaway trained on a tiny test set).
- Tokenizer sanity before encoding: grammar terms (منصوب/مرفوع/الفاعل/مجرور/إعراب/النحو/الصرف)
  collapse to single tokens; ~1.42 tok/word on grammar vs ~1.63 on prose.

## E. Train/val split + encoding (mini_frontier)

- **Book-level** split (no tail-slice): ~5% of books per category held out, top-10%
  largest books excluded from val (protect unique reference works), uniform sampling by size.
- `train.bin` = **175,918,892 tokens (866 books)**; `val.bin` = **4,231,363 tokens (45 books)**;
  **zero book overlap**.
- عروض (cat 33) too small for a val book → left entirely in train (no val signal), by design.
- `--emit_val_cat_bins` → per-category `val_cat{N}.bin`. They sum *exactly* to `val.bin`
  (integrity check passes).
- `[PAGE n]` markers stripped at binary-prep time.

## F. Eval/reporting change (mini_frontier, train.py — eval only)

- Added `evaluate_categories()` and `weighted_val_loss()`.
- Val loss now logged per-category (نحو/صرف, أدب, معاجم at minimum) **and** as a single
  weighted loss using the training effective-mix (`config.VAL_EVAL_MIX`, renormalized;
  عروض excluded, Z=0.985).
- Best checkpoint now tracks the **weighted** val loss.
- `train.bin` and book selection were NOT changed — this was purely eval/reporting.
- `model.py` left untouched.

## G. Smoke test (laptop, GTX 1650)

- 200 steps, healthy: ValW 9.73 → 7.42.
- Fit-VRAM command: `python3 train.py --max_iters 200 --batch_size 4 --block_size 256 --no_compile`.

## H. First Colab run + truncation incident

- First full run loaded only **9,437,184 tokens** (exactly 18 MiB) instead of 175.9M —
  incomplete Drive upload. See bugs-and-gotchas #5. Fixed with fail-loud size+hash asserts.

## I. Qur'an span masking (feature/quran-masking branch)

- Root cause of fabricated scripture: `normalize_arabic` strips ﴿ ﴾ markers
  (U+FD3E/FD3F, outside the kept ranges), so Qur'an spans entered training unmarked.
- Added loss masking with `ignore_index = -1`: `QURAN_SPAN_RE = re.compile(r'﴿([^﴾]*)﴾')`,
  `_quran_segments`, `_encode_chunks_masked`, `prepare_shamela_split_masked`;
  `MemmapDataset` gained `mask_path`/`mask_for()`; batch builder sets `y[m==0] = -1`.
- Verified: train **1.05%** masked (بلاغة 5.9%, شعر 0%); token counts unchanged
  (﴿﴾ sit at whitespace boundaries); X never masked, only Y.

## J. Git + final run

- Repo: `https://github.com/ietamaher/mini_frontier.git`; author `ietamaher <ieta_maher@hotmail.fr>`.
- Tokenizer JSON force-tracked via `.gitignore` exception. Feature branch merged to `main`
  (--no-ff).
- Colab archive rebuilt excluding `checkpoints/`, `logs/`, `*.txt` → ~235 MB.
- **Final 10k-step run (T4, float16+GradScaler, ~0.66B tokens ≈ 3.7 epochs):**
  best ValW **4.6738** at step 9500. No overfitting. Per-category final:
  نحو/صرف 4.21 < بلاغة 4.70 < معاجم 4.76 < لغة 5.22 < أدب 5.28 < شعر 5.56.
- Masked checkpoints synced to `checkpoints/`; previous run preserved under
  `checkpoints/no_masker/`.

## K. Generation test (laptop)

- `python3 generate.py` on `ckpt_best.pt` (step 9500). Register-correct classical Arabic;
  نحو strongest (correct إعراب template). Qur'an-trigger prompt did not emit ﴿﴾ or a
  confident fabricated verse (consistent with masking; not yet batch-verified).

## L. Tokenizer Metaspace fix + full re-encode (2026-06-22)

- Diagnosed and fixed the intra-word-space bug (known-issues #1): old tokenizer had
  `decoder: None` + `Whitespace` pre-tokenizer → word boundaries lost, default decoder
  space-joins every token. Decode-only fix proven impossible.
- `tokenizer_arabic.py`: `Whitespace` → **Metaspace** pre-tokenizer (`▁`) + matching
  `MetaspaceDecoder`. Added `WORD_BOUNDARY = "▁"`.
- `generate.py`: param log now prints total + non-embedding (was the confusing 10.52M).
- Backed up old tokenizer → `tokenizer/arabic_bpe_16k.OLD_whitespace.json`.
- Retrained 16k BPE on `data/shamela/stream.txt` (1.18 GB) → new `arabic_bpe_16k.json`
  (decoder=Metaspace persisted). Round-trip clean; grammar terms 1 token each; 1.00 tok/word
  on grammar passage (was 1.42).
- Re-encoded ALL bins (`python3 data_pipeline.py --encode_shamela_masked`), same
  deterministic 866/45 book split:
  - `train.bin` **170,733,821** tokens (was 175.9M), masked 1.118%
  - `val.bin` **4,132,711** tokens (was 4.23M), masked 1.469%
  - per-category val bins sum exactly to val.bin ✓; every `*_mask.bin` size == its token count ✓
  - decoded a real train.bin slice → clean, no intra-word spaces ✓
- **Token counts dropped** because the new tokenizer packs fewer tokens/word — so the next
  run's absolute val loss is NOT comparable to the previous 4.6738 (per-token CE rises when
  tokens are more informative). Judge the new run by curve shape + generation quality.
- **Old checkpoints (`checkpoints/ckpt_*.pt`, `checkpoints/no_masker/`) are now invalid**
  (trained on old token IDs). Next: full model retrain on Colab with the new tokenizer+bins.

## M. Metaspace run — completed & VALIDATED (2026-06-23)

The corrected run (new Metaspace tokenizer + new bins) ran 10k steps on Colab T4.
- **First attempt was a false start:** a run trained on the OLD bins (loaded stale data from
  Drive) — caught via the provenance cross-check (clean decode only with the OLD tokenizer).
  See bugs-and-gotchas (data provenance). Re-run on the correct new bins.
- **Final validated model = `ckpt_best.pt` = step 9500, ValW 4.9471** (saved as
  `checkpoints/metaspace_baseline_step9500.pt` to protect it from the next run; checkpoints/
  is gitignored — local protection only).
- Curve: 1000→6.18, 5000→5.13, 7500→4.98, 9500→**4.95**. Smooth, converged, no overfitting.
- Per-category @ 9500: نحو/صرف **4.666** (lowest ✓) < بلاغة 4.839 < معاجم 4.976 < لغة 5.254
  < أدب 5.353 < شعر 5.717 — healthy weighting signature, نحو strongest.
- **Generation VALIDATED:** clean decode, NO intra-word spaces (the bug is fixed); coherent
  إعراب register over ~55 tokens; provenance cross-check passes (OLD tokenizer → garbage →
  confirms training on new bins). قال تعالى prompt → no fabricated verse / no ﴿﴾ (masking OK).
- ⚠️ **NOT comparable to the old 4.6738** — different token scale (Metaspace = fewer,
  more-informative tokens → higher per-token CE). This is the real reference baseline now.
- **Note for next run (50/80M):** eval/save cadence is every 500 steps, so the LAST eval was
  step 9500; steps 9550–9999 trained but were never evaluated/checkpointed (the final ~500
  steps' small gain, ~0.02 loss, is lost). Force a final eval+save at the very last step next
  time. Minor, but free.
