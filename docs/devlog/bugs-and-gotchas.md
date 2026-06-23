# Bugs & Gotchas — read before any fix

Every bug we already hit, its root cause, the fix, and the rule that prevents redoing it.
If you're about to change pipeline / .gitignore / Colab packaging / checkpoints, scan this first.

---

## 1. Pipeline/tokenizer/training code created in the wrong repo

- **Symptom:** tokenizer, data-pipeline and training files were written under
  `arabic-corpus/staging/...` and `arabic-corpus/`.
- **Root cause:** confused the two repos. `arabic-corpus` is corpus-preparation ONLY.
- **Fix:** deleted the misplaced files (`scripts/config.py`, `scripts/data_pipeline.py`,
  `pipeline_output/`) from `arabic-corpus`; all training/tokenizer/pipeline code lives in
  `mini_frontier`.
- **Rule:** `arabic-corpus/` = extraction + manifest + provenance, and `raw/` is immutable.
  Never put tokenizer/pipeline/model/train code there. Training code → `mini_frontier/`.

## 2. `.gitignore` inline comments break patterns (happened TWICE)

- **Symptom:** `data/shamela/  # flux texte` and `!tokenizer/...json # exception` did not
  match — files were not ignored / the exception did not apply.
- **Root cause:** `.gitignore` does NOT support trailing inline comments; the `#...` becomes
  part of the pattern.
- **Fix:** put every comment on its own line, pattern alone on its line.
- **Rule:** in `.gitignore`, comments are full-line only. Never `pattern  # note`.

## 3. `torch.amp.GradScaler("cuda", ...)` missing / API mismatch

- **Symptom:** AttributeError on `torch.amp.GradScaler` at startup.
- **Root cause:** env had torch 2.0.1; the unified `torch.amp` GradScaler API needs **≥2.3**.
- **Fix:** upgraded laptop to `torch==2.4.1+cu121` (driver 535 → CUDA 12.2; GTX 1650 = sm_75).
- **Rule:** every training machine (laptop dev, RTX, Colab) must be torch ≥ 2.3. Check
  `python -c "import torch; print(torch.__version__)"` before launching.

## 4. Backgrounded Python buffered its output (no live log)

- **Symptom:** background `python train.py > log` produced nothing until exit.
- **Fix:** run with `python3 -u` (unbuffered) for backgrounded/long runs.
- **Rule:** always `-u` when piping a long run to a log file.

## 5. `train.bin` truncated on Colab — silent 18 MiB instead of 175.9M tokens ⚠️ most dangerous

- **Symptom:** training log showed `Dataset chargé : 9,437,184 tokens` (exactly 18 MiB)
  instead of `175,918,892`. Training "worked" but on 5% of the data.
- **Root cause:** incomplete Google Drive upload of `train.bin`. The local archive was
  intact (sha256 `05aa0cd1…`).
- **Fix:** added a fail-loud Colab verification cell asserting both **byte size** and
  **sha256** of `train.bin`/`val.bin` before training starts.
- **Rule:** never trust an uploaded/extracted bin by existence alone. Assert exact size +
  hash. Expected: `train.bin` 175,918,892 tokens; `val.bin` 4,231,363 tokens; per-category
  val bins sum exactly to `val.bin`.

## 6. Colab archive bloat — `tar` ignored `.gitignore` (1029 MB)

- **Symptom:** the upload archive ballooned to ~1 GB; swept in 893 MB of `checkpoints/`.
- **Root cause:** `tar` does NOT honor `.gitignore`.
- **Fix:** rebuilt with explicit `--exclude='checkpoints'  --exclude='logs' --exclude='*.txt'`
  → ~235 MB.
- **Rule:** packaging tools (`tar`, `zip`) need explicit `--exclude`. Never assume
  `.gitignore` protects you outside of git. Exclude `checkpoints/`, `logs/`, `data/*.bin`,
  `data/*.txt` from any upload archive.

## 7. Smoke test overwrote the real `ckpt_best.pt`

- **Symptom:** the step-5000 best checkpoint was replaced by a 14-step throwaway from a
  masking smoke test (both wrote `checkpoints/ckpt_best.pt`).
- **Fix:** restored `ckpt_best.pt` from `ckpt_step05000.pt` (verified step=5000, val=4.8126);
  removed the throwaway 00000/00007/00014 checkpoints.
- **Rule:** smoke/throwaway runs must write to a separate dir (e.g. `checkpoints/smoke/`) or
  use a distinct prefix. Never let a smoke test share the real checkpoint dir. (We now keep
  the prior real run under `checkpoints/no_masker/`.)

## 8. Stale `.git/index.lock`

- **Symptom:** git operations failed with "index.lock exists" after an interrupted
  background `git add`.
- **Fix:** removed `.git/index.lock`.
- **Rule:** don't background `git add` of huge trees; if interrupted, delete the stale lock.

## 9. 2.2 GB of `data/shamela/*.txt` almost pushed to GitHub

- **Symptom:** the weighted text streams (`.txt`, ~2 GB) were not ignored initially; would
  have blown past GitHub's 100 MB file limit.
- **Fix:** added `data/shamela/`, `data/*.txt`, `data/*.bin` to `.gitignore` **before** the
  first commit.
- **Rule:** generated data (streams, bins, checkpoints, logs) is never committed. Only code,
  docs, config, and the trained tokenizer JSON (via explicit `!` exception).

## 10. Fabricated Qur'an — root cause was normalization stripping ﴿ ﴾

- **Symptom:** model emitted confident fabricated scripture; couldn't be masked because the
  spans were unmarked in training data.
- **Root cause:** `normalize_arabic` strips the ornate parenthesis markers ﴿ (U+FD3E) and
  ﴾ (U+FD3F) because they fall outside the kept Unicode ranges — so Qur'an spans entered the
  stream indistinguishable from prose.
- **Fix:** mask Qur'an spans (`﴿…﴾`) for the loss **before** normalization removes the
  markers; `ignore_index = -1` on the Y targets inside the span.
- **Rule:** any orthographic feature you rely on downstream must be detected BEFORE
  `normalize_arabic` runs (it strips anything outside its kept ranges). Don't add new
  marker-dependent logic after normalization.

## 11. Reusable build artifacts left in /tmp got wiped (Shamela Java extractor)

- **Symptom:** `ShamelaQueryExtract.class` + its `.java` source lived in `/tmp` (per the old
  scripts' `-cp /tmp`). `/tmp` was cleared → both gone; pilot extraction blocked.
- **Recovery:** the source was recoverable from the session transcript jsonl (grep for the
  class body). Re-saved + recompiled.
- **Fix:** moved source+class to **`arabic-corpus/scripts/java/`** (in-repo, persistent),
  added `scripts/java/README.md` (build commands), and repointed both
  `extract_language_core.py` and `shamela_watch_extract.py` classpath from `/tmp` →
  `scripts/java`. Compile with system `javac 11`, run with bundled JRE 21.
- **Rule:** never store reusable build artifacts, compiled classes, or anything you'll need
  again in `/tmp`. Keep them in the repo. `/tmp` is scratch-only.

---

## Standing constraints (don't violate)

- `arabic-corpus/raw/` is immutable; arabic-corpus is corpus-prep only.
- Never normalize orthography in `clean/`; flag-don't-guess; Qur'an = maximum sensitivity.
- `model.py` stays untouched unless explicitly requested.
- Don't change `train.bin` or book selection when changing eval/reporting.
- Don't commit huge data files; exclude `*.txt` streams + checkpoints from Colab archives.
- Laptop (GTX 1650, 4 GB) = dev/smoke only. Real training = RTX station / Colab; don't
  retune `config.py` defaults for the laptop.
