# Devlog — MiniFrontier Arabic LLM

Running record of what we did, what broke, and what not to repeat. Filed before any
new fix so the same bugs don't get re-introduced.

Two repos are involved:
- **`arabic-corpus/`** — corpus preparation ONLY (extraction, manifest, provenance).
  No tokenizer/pipeline/training code lives here. `raw/` is immutable.
- **`mini_frontier/`** — training repo (tokenizer, model, pipeline, train/eval, generation).

## Files

| File | Purpose |
|---|---|
| [operations-log.md](operations-log.md) | Chronological record of every operation performed. |
| [bugs-and-gotchas.md](bugs-and-gotchas.md) | **Read before any fix.** Bug → root cause → fix → how to not redo it. |
| [known-issues.md](known-issues.md) | Open items not yet fixed (with the recommended approach). |

## One-line status (2026-06-21)

Full 10k-step Colab run complete. Best weighted val loss **4.6738** (step 9500),
beats the pre-masking baseline (4.81). Qur'an span-masking merged to `main`.
Generation is register-correct (نحو strongest). Top open issue: tokenizer decode
inserts spurious intra-word spaces — see [known-issues.md](known-issues.md).
