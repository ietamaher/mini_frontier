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
