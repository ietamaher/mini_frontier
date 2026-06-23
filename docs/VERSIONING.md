# Ḍād — Model Versioning & Release Strategy

How we take the model from **Ḍād-v1 → v2 → v3 …** inside this repo: what bumps a version,
how code/weights/data are tracked, and the exact steps to cut a release.

> **Model name:** **Ḍād** (ضاد) — from *لغة الضاد*, "the language of the ḍād", the epithet of
> Arabic (the one letter no other language has). Versions: `Ḍād-v1` (14.6M, current baseline),
> `Ḍād-v2` (50M, planned — see [RESEARCH_v1_to_v2.md](RESEARCH_v1_to_v2.md)), `Ḍād-v3` …

---

## 1. The core problem

A trained model is **3 coupled artifacts**, not one:

| Artifact | Size | Lives in | Versioned how |
|---|---|---|---|
| **Code** (model.py, train.py, pipeline, config) | KBs | git, `main` | commit SHA + tag |
| **Tokenizer** (`arabic_bpe_16k.json`) | ~1 MB | git (tracked exception) | committed; hash recorded |
| **Weights** (`ckpt_best.pt`) | ~175 MB+ | **NOT git** (too big) | GitHub Release asset / HF Hub |
| **Data bins** (`train.bin`, masks) | ~340 MB+ | **NOT git** | Drive archive + recorded hashes |

A "version" is the **frozen tuple** of all four. The repo tracks the small ones and *records
the hashes/locations* of the big ones so any version is reproducible. (We already learned the
cost of breaking this coupling: the Metaspace tokenizer change silently invalidated the old
bins and checkpoints — see devlog. Versioning exists to make that impossible to do by accident.)

---

## 2. Version-numbering scheme

`Ḍād-vMAJOR.MINOR.PATCH` (e.g. `Ḍād-v2.1.0`). What bumps which:

| Bump | Trigger | Examples | Breaks compat with… |
|---|---|---|---|
| **MAJOR** (v1→v2) | **Architecture or tokenizer change** | param count (n_embd/n_layer), weight-tying, vocab size, **any tokenizer retrain** | bins + checkpoints + tokenizer all change |
| **MINOR** (v2.0→v2.1) | **Corpus or recipe change, same arch+tokenizer** | new categories added, new mix weights, new LR/batch schedule | bins change; checkpoints retrained; tokenizer **same** |
| **PATCH** (v2.1.0→v2.1.1) | **Fix / continuation, same spec** | bug re-run, longer training of same config, eval-only fix | nothing structural |

**Hard rule:** a **tokenizer retrain is always at least a MAJOR bump.** It re-maps every token
ID → all bins and all checkpoints become invalid. Never ship a new tokenizer under a MINOR/PATCH.

---

## 3. Reproducibility record (per version)

Every version pins this quintuple so it can be rebuilt exactly:

1. **Code commit SHA** (the exact `main` commit that produced it) → captured by the git tag.
2. **Tokenizer hash** (`sha256 tokenizer/arabic_bpe_16k.json`).
3. **Data bins hashes** (`train.bin`, `val.bin`, masks) + the corpus manifest used.
4. **Config snapshot** (the `ModelConfig` + training hyperparameters, copied into the model card).
5. **Seed** (`SHAMELA_SHUFFLE_SEED`, train seed).

These four+seed go into the **model card** (§5) and the **registry** ([MODELS.md](MODELS.md)).

---

## 4. Git workflow

- **`main`** = latest stable code. Always reproducible against the most recent released model.
- **One annotated tag per released model**, on the exact commit that trained it:
  ```
  git tag -a Ḍād-v1 <commit> -m "Ḍād-v1 — 14.6M Metaspace baseline, ValW 4.947"
  git push origin Ḍād-v1
  ```
- **Feature branch per experiment** (as with `feature/quran-masking`): e.g.
  `feature/v2-tied-50m`. Merge to `main` (`--no-ff`) once the version is adopted, then tag.
- **In-flight version lines** can keep a branch (`v2-dev`) until released; release = merge + tag.

```
main ──●────●────●────────●────────────●───►
       │    │              │            │
   Ḍād-v1  (work)     feature/v2     Ḍād-v2
    tag                tied-50m        tag
                       (merged)
```

---

## 5. Per-version deliverables (release checklist)

Cutting `Ḍād-vN`:

1. `docs/cards/MODEL_CARD_vN.md` — config, tokenizer hash, data manifest+hashes, recipe,
   eval metrics (ValW + per-category), generation samples, known limitations.
2. Add/flip the row in [MODELS.md](MODELS.md) (the registry — single source of truth).
3. **Tag** the commit (`Ḍād-vN`) and push the tag.
4. **GitHub Release** named `Ḍād-vN` on that tag, with assets attached:
   - `ckpt_best.pt` (the weights — GitHub Releases allow up to 2 GB/file),
   - `arabic_bpe_16k.json` (the matching tokenizer),
   - `data_manifest_vN.txt` (bins sizes + sha256, so the corpus is verifiable).
   *(Optionally also publish to the Hugging Face Hub for sharing/inference.)*
5. Keep the local weights under a stable name so the next run can't overwrite them
   (e.g. `checkpoints/Ḍād-v1_step9500.pt` — `checkpoints/` is gitignored, local safety copy).

> Weights never go in git. The **tag** freezes the code+tokenizer; the **release assets** carry
> the weights+data-manifest. `git checkout Ḍād-v1` + download that release = the exact model.

---

## 6. Where the science lives

- **Design/justification per major version** → a research report like
  [RESEARCH_v1_to_v2.md](RESEARCH_v1_to_v2.md) (v1 analysis + v2 proposal). Write one per MAJOR.
- **Decisions/bugs/roadmap** → `docs/devlog/` (running log).
- **What exists right now** → [MODELS.md](MODELS.md) (the registry).

---

## 7. Immediate next actions for Ḍād

1. **Tag the current baseline** as `Ḍād-v1` (commit that produced the Metaspace run) and cut its
   GitHub Release with `metaspace_baseline_step9500.pt` + tokenizer + data manifest.
2. Write `docs/cards/MODEL_CARD_v1.md`.
3. v2 work (Config C, 46.7M tied, ~1B tokens) proceeds on `feature/v2-tied-50m`; apply the
   known-issues #7 fix (final-step eval) and the corpus expansion (تاريخ/تراجم → … ) first.
