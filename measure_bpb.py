#!/usr/bin/env python3
"""
measure_bpb.py — TRUE bits-per-byte (BPB) for Ḍād-v1 on the val split.

Measures (not estimates):
  • bytes/token  : UTF-8 byte length of each decoded token over the val set
                   (distribution: mean, median, p10, p90).
  • nats/token   : mean cross-entropy on the val set, respecting the Qur'an
                   loss-mask (mask==0 → ignored, exactly as in training).
  • BPB          : (nats_per_token / ln2) / bytes_per_token, point value + a
                   low/high range from p90/p10 bytes/token.
  • per-category BPB over the val_cat{N}.bin sets.

Writes bpb_results.json. Inference only — no training, no corpus changes.

Usage: python3 measure_bpb.py [--ckpt checkpoints/ckpt_best.pt] [--batch 4]
"""
import argparse, json, math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from config import (ModelConfig, get_device, get_autocast_ctx, train_cfg,
                    TOKENIZER_PATH, VAL_CAT_NAMES, val_cat_bin, CHECKPOINT_DIR)
from model import MiniFrontierLLM
from tokenizer_arabic import ArabicTokenizer
from data_pipeline import VAL_BIN, mask_for

LN2 = math.log(2.0)


def load_model(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ModelConfig(**ck["model_cfg"])
    m = MiniFrontierLLM(cfg).to(device)
    m.load_state_dict(ck["model"])
    m.eval()
    return m, cfg, ck


def byte_table(tok, vocab_size):
    """bytes[i] = UTF-8 byte length that token id i contributes to the text.
    Special tokens (skip_special=True → '') contribute 0 bytes."""
    tbl = np.zeros(vocab_size, dtype=np.int32)
    for i in range(vocab_size):
        tbl[i] = len(tok.decode([i]).encode("utf-8"))
    return tbl


@torch.no_grad()
def eval_split(model, block, bin_path, byte_tbl, device, batch):
    """Returns (total_nats, n_tokens, per_token_byte_array) over non-masked targets."""
    data = np.memmap(str(bin_path), dtype=np.uint16, mode="r")
    mpath = mask_for(bin_path)
    mask = np.memmap(str(mpath), dtype=np.uint8, mode="r") if Path(mpath).exists() else None
    n = len(data)
    ctx = get_autocast_ctx(device, train_cfg.dtype)

    total_nats = 0.0
    n_tokens = 0
    byte_chunks = []

    starts = list(range(0, n - block - 1, block))
    xb, yb = [], []

    def flush():
        nonlocal total_nats, n_tokens
        if not xb:
            return
        x = torch.from_numpy(np.stack(xb).astype(np.int64)).to(device)
        y = torch.from_numpy(np.stack(yb).astype(np.int64)).to(device)
        with ctx:
            logits, _ = model(x, y)
        ce = F.cross_entropy(logits.view(-1, logits.size(-1)).float(),
                             y.view(-1), ignore_index=-1, reduction="sum")
        total_nats += ce.item()
        n_tokens += int((y.view(-1) != -1).sum().item())
        xb.clear(); yb.clear()

    for s in starts:
        x_np = data[s:s + block]
        y_np = np.asarray(data[s + 1:s + block + 1]).astype(np.int64)
        if mask is not None:
            m_np = np.asarray(mask[s + 1:s + block + 1])
            valid = m_np == 1
            y_np[~valid] = -1
        else:
            valid = np.ones(block, dtype=bool)
        # bytes only for kept (non-masked) targets
        kept_ids = np.asarray(data[s + 1:s + block + 1])[valid]
        byte_chunks.append(byte_tbl[kept_ids])
        xb.append(np.asarray(x_np)); yb.append(y_np)
        if len(xb) >= batch:
            flush()
    flush()

    per_token_bytes = np.concatenate(byte_chunks) if byte_chunks else np.array([], dtype=np.int32)
    return total_nats, n_tokens, per_token_bytes


def summarize(name, total_nats, n_tokens, ptb):
    nats_per_tok = total_nats / max(n_tokens, 1)
    bits_per_tok = nats_per_tok / LN2
    mean_b = float(ptb.mean()) if ptb.size else 0.0
    med_b = float(np.median(ptb)) if ptb.size else 0.0
    p10_b = float(np.percentile(ptb, 10)) if ptb.size else 0.0
    p90_b = float(np.percentile(ptb, 90)) if ptb.size else 0.0
    bpb = bits_per_tok / mean_b if mean_b else 0.0
    # range: more bytes/token → lower BPB ; fewer → higher
    bpb_low = bits_per_tok / p90_b if p90_b else 0.0
    bpb_high = bits_per_tok / p10_b if p10_b else 0.0
    return {
        "name": name,
        "n_tokens": int(n_tokens),
        "nats_per_token": round(nats_per_tok, 4),
        "bits_per_token": round(bits_per_tok, 4),
        "bytes_per_token_mean": round(mean_b, 3),
        "bytes_per_token_median": round(med_b, 3),
        "bytes_per_token_p10": round(p10_b, 3),
        "bytes_per_token_p90": round(p90_b, 3),
        "bpb": round(bpb, 4),
        "bpb_low": round(bpb_low, 4),
        "bpb_high": round(bpb_high, 4),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, default=CHECKPOINT_DIR / "ckpt_best.pt")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--out", type=Path, default=Path("bpb_results.json"))
    args = ap.parse_args()

    device = get_device()
    print(f"Device: {device}")
    model, cfg, ck = load_model(args.ckpt, device)
    block = cfg.block_size
    print(f"Model: step={ck.get('step')} val_loss={ck.get('val_loss'):.4f} "
          f"block={block} vocab={cfg.vocab_size}")

    tok = ArabicTokenizer(TOKENIZER_PATH)
    print("Building byte/token table…")
    btbl = byte_table(tok, cfg.vocab_size)

    results = {}

    print("\n=== FULL VAL ===")
    tn, nt, ptb = eval_split(model, block, VAL_BIN, btbl, device, args.batch)
    full = summarize("val_all", tn, nt, ptb)
    results["overall"] = full
    print(f"  nats/tok={full['nats_per_token']}  bits/tok={full['bits_per_token']}  "
          f"bytes/tok={full['bytes_per_token_mean']} (p10={full['bytes_per_token_p10']}, "
          f"p90={full['bytes_per_token_p90']})")
    print(f"  BPB = {full['bpb']}  (range {full['bpb_low']}–{full['bpb_high']})")

    print("\n=== PER CATEGORY ===")
    per_cat = {}
    for cat, name in VAL_CAT_NAMES.items():
        b = val_cat_bin(cat)
        if not Path(b).exists():
            continue
        tn, nt, ptb = eval_split(model, block, b, btbl, device, args.batch)
        s = summarize(f"cat{cat}_{name}", tn, nt, ptb)
        per_cat[str(cat)] = {"name_ar": name, **s}
        print(f"  {name:<8} BPB={s['bpb']:.4f}  nats/tok={s['nats_per_token']:.3f}  "
              f"bytes/tok={s['bytes_per_token_mean']:.3f}  (n={s['n_tokens']:,})")
    results["per_category"] = per_cat
    results["meta"] = {
        "ckpt": str(args.ckpt), "step": ck.get("step"), "model_val_loss": ck.get("val_loss"),
        "block_size": block, "vocab_size": cfg.vocab_size,
        "tokenizer": str(TOKENIZER_PATH), "mask": "Quran spans excluded (mask==0)",
        "note": "bytes/token MEASURED via per-token UTF-8 decode, not assumed",
    }

    args.out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n✅ wrote {args.out}")
    print(f"HEADLINE: BPB = {full['bpb']} bits/byte  (range {full['bpb_low']}–{full['bpb_high']}), "
          f"from {full['nats_per_token']} nats/tok ÷ {full['bytes_per_token_mean']} bytes/tok")


if __name__ == "__main__":
    main()
