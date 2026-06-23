#!/usr/bin/env python3
"""
scaling_sweep.py — locate the data ceiling of the FIXED 170M-token corpus.

Trains a short sweep of weight-TIED configs (~5M, ~15M, ~30M, ~47M params) on the
SAME corpus / tokenizer / recipe (only size varies), ~4 epochs each, and records
final ValW, final train loss, train/val gap, and the val-loss curve. Produces
scaling_results.json + scaling_curve.png.

Weight tying is applied at RUNTIME (lm_head.weight = tok_emb.weight) so model.py is
untouched and v1 reproducibility is preserved.

PILOT FIRST:
    python3 scaling_sweep.py --pilot                # ~5M, few hundred steps, validate harness
Full sweep (prefer RTX 3090 / bf16):
    python3 scaling_sweep.py --configs 5m,15m,30m,47m --epochs 4 --micro_batch 32

All artifacts go under mini_frontier/sweep/ (gitignored weights).
"""
import argparse, json, math, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from config import (ModelConfig, get_device, get_autocast_ctx, resolve_dtype,
                    train_cfg, VAL_EVAL_MIX, VAL_CAT_NAMES, val_cat_bin)
from model import MiniFrontierLLM
from data_pipeline import TRAIN_BIN, VAL_BIN, MemmapDataset, mask_for

SWEEP_DIR = Path(__file__).parent / "sweep"
SWEEP_DIR.mkdir(exist_ok=True)

# head_dim = 64 throughout (research-recommended), n_head = n_embd // 64
CONFIGS = {
    "5m":  dict(n_embd=128, n_layer=14, n_head=2,  hidden_dim=384),
    "15m": dict(n_embd=256, n_layer=14, n_head=4,  hidden_dim=704),
    "30m": dict(n_embd=384, n_layer=14, n_head=6,  hidden_dim=1024),
    "47m": dict(n_embd=512, n_layer=12, n_head=8,  hidden_dim=1408),  # Config C
}


def build_tied_model(spec, device):
    cfg = ModelConfig(**spec)
    m = MiniFrontierLLM(cfg).to(device)
    # runtime weight tying — do NOT modify model.py
    m.lm_head.weight = m.tok_emb.weight
    return m, cfg


def lr_at(step, peak, mn, warmup, total):
    if step < warmup:
        return peak * (step + 1) / warmup
    if step >= total:
        return mn
    r = (step - warmup) / max(1, total - warmup)
    return mn + 0.5 * (peak - mn) * (1 + math.cos(math.pi * r))


@torch.no_grad()
def masked_mean_ce(model, ds, block, device, ctx, max_batches, batch):
    """Deterministic mean nats/token over a MemmapDataset (mask-aware), capped."""
    data, mask = ds.data, ds.mask
    n = ds.n_tokens
    total_nats, total_tok = 0.0, 0
    starts = list(range(0, n - block - 1, block))[:max_batches * batch]
    xb, yb = [], []
    def flush():
        nonlocal total_nats, total_tok
        if not xb: return
        x = torch.from_numpy(np.stack(xb).astype(np.int64)).to(device)
        y = torch.from_numpy(np.stack(yb).astype(np.int64)).to(device)
        with ctx:
            logits, _ = model(x, y)
        ce = F.cross_entropy(logits.view(-1, logits.size(-1)).float(), y.view(-1),
                             ignore_index=-1, reduction="sum")
        total_nats += ce.item()
        total_tok += int((y.view(-1) != -1).sum().item())
        xb.clear(); yb.clear()
    for s in starts:
        y_np = np.asarray(data[s+1:s+block+1]).astype(np.int64)
        if mask is not None:
            y_np[np.asarray(mask[s+1:s+block+1]) == 0] = -1
        xb.append(np.asarray(data[s:s+block])); yb.append(y_np)
        if len(xb) >= batch: flush()
    flush()
    return total_nats / max(total_tok, 1)


def weighted_val(model, block, device, ctx, batch, max_batches):
    """ValW over per-category val bins, using VAL_EVAL_MIX (renormalized)."""
    per_cat, Z, acc = {}, 0.0, 0.0
    for cat, w in VAL_EVAL_MIX.items():
        b = val_cat_bin(cat)
        if not Path(b).exists():
            continue
        ds = MemmapDataset(b, block, mask_for(b))
        L = masked_mean_ce(model, ds, block, device, ctx, max_batches, batch)
        per_cat[VAL_CAT_NAMES.get(cat, str(cat))] = round(L, 4)
        acc += w * L; Z += w
    return acc / max(Z, 1e-9), per_cat


def train_one(name, spec, args, device):
    ctx = get_autocast_ctx(device, train_cfg.dtype)
    use_scaler = (resolve_dtype(train_cfg.dtype) == "float16" and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    model, cfg = build_tied_model(spec, device)
    n_total = sum(p.numel() for p in model.parameters())
    n_nonemb = n_total - model.tok_emb.weight.numel()
    print(f"\n=== {name} : {n_total/1e6:.2f}M params (non-emb {n_nonemb/1e6:.2f}M) "
          f"d={cfg.n_embd} L={cfg.n_layer} h={cfg.n_head} ===")

    block = args.block
    train_ds = MemmapDataset(TRAIN_BIN, block, mask_for(TRAIN_BIN))
    flat_val_ds = MemmapDataset(VAL_BIN, block, mask_for(VAL_BIN))

    eff_batch = args.micro_batch * args.grad_accum * block
    if args.pilot:
        max_iters = args.pilot_steps
    else:
        max_iters = math.ceil(args.epochs * train_ds.n_tokens / eff_batch)
    warmup = min(train_cfg.warmup_iters, max(10, max_iters // 25))
    peak, mn = train_cfg.learning_rate, train_cfg.min_lr

    decay = [p for p in model.parameters() if p.dim() >= 2]
    nodecay = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([
        {"params": decay, "weight_decay": train_cfg.weight_decay},
        {"params": nodecay, "weight_decay": 0.0},
    ], lr=peak, betas=(train_cfg.beta1, train_cfg.beta2))

    print(f"  eff_batch={eff_batch:,} tok/step │ max_iters={max_iters:,} "
          f"(~{max_iters*eff_batch/1e6:.0f}M tok, {max_iters*eff_batch/train_ds.n_tokens:.2f} ep) "
          f"│ warmup={warmup}")

    curve, t0 = [], time.time()
    best_valw, final_train = float("inf"), float("nan")
    diverged = False
    eval_every = max(50, max_iters // 12) if not args.pilot else max(20, max_iters // 3)

    model.train()
    for step in range(max_iters):
        lr = lr_at(step, peak, mn, warmup, max_iters)
        for g in opt.param_groups: g["lr"] = lr

        opt.zero_grad(set_to_none=True)
        loss_acc = 0.0
        for _ in range(args.grad_accum):
            x, y = train_ds.get_random_batch(args.micro_batch)
            x, y = x.to(device), y.to(device)
            with ctx:
                _, loss = model(x, y)
            loss = loss / args.grad_accum
            scaler.scale(loss).backward()
            loss_acc += loss.item()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
        scaler.step(opt); scaler.update()

        final_train = loss_acc
        if not math.isfinite(loss_acc):
            print(f"  ⚠️ diverged at step {step} (loss={loss_acc}) — recording & stopping")
            diverged = True
            break

        if step % args.log_interval == 0:
            print(f"  step {step:>5}/{max_iters} │ train {loss_acc:.4f} │ lr {lr:.2e} "
                  f"│ {time.time()-t0:.0f}s")

        if step > 0 and (step % eval_every == 0):
            model.eval()
            valw, per_cat = weighted_val(model, block, device, ctx, args.eval_batch, args.eval_max_batches)
            flat = masked_mean_ce(model, flat_val_ds, block, device, ctx, args.eval_max_batches, args.eval_batch)
            model.train()
            curve.append({"step": step, "valw": round(valw, 4), "flat": round(flat, 4),
                          "train": round(loss_acc, 4)})
            print(f"  [eval] step {step} │ ValW {valw:.4f} │ ValFlat {flat:.4f} │ "
                  f"train {loss_acc:.4f} │ gap {flat-loss_acc:+.3f}")
            if valw < best_valw:
                best_valw = valw
                torch.save({"step": step, "val_loss": valw, "model_cfg": vars(cfg),
                            "model": model.state_dict()}, SWEEP_DIR / f"{name}_best.pt")

        if args.max_seconds and (time.time() - t0) > args.max_seconds:
            print(f"  ⏱️ wall-clock cap ({args.max_seconds}s) hit at step {step}")
            break

    # final eval
    model.eval()
    valw, per_cat = weighted_val(model, block, device, ctx, args.eval_batch, args.eval_max_batches)
    flat = masked_mean_ce(model, flat_val_ds, block, device, ctx, args.eval_max_batches, args.eval_batch)
    if valw < best_valw:
        best_valw = valw
        torch.save({"step": "final", "val_loss": valw, "model_cfg": vars(cfg),
                    "model": model.state_dict()}, SWEEP_DIR / f"{name}_best.pt")
    print(f"  FINAL {name}: ValW {valw:.4f} │ ValFlat {flat:.4f} │ train {final_train:.4f} "
          f"│ gap {flat-final_train:+.3f} │ best ValW {best_valw:.4f}")

    return {
        "name": name, "params_total": int(n_total), "params_nonemb": int(n_nonemb),
        "n_embd": cfg.n_embd, "n_layer": cfg.n_layer, "n_head": cfg.n_head,
        "hidden_dim": cfg.hidden_dim, "max_iters": max_iters, "eff_batch": eff_batch,
        "final_valw": round(valw, 4), "best_valw": round(best_valw, 4),
        "final_flat": round(flat, 4), "final_train": round(final_train, 4),
        "gap_flat_minus_train": round(flat - final_train, 4),
        "per_cat": per_cat, "curve": curve, "diverged": diverged,
        "wall_seconds": round(time.time() - t0, 1),
    }


def plot(results, out):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"(plot skipped: {e})"); return
    rs = [r for r in results if r]
    xs = [r["params_total"] / 1e6 for r in rs]
    ys = [r["best_valw"] for r in rs]
    plt.figure(figsize=(6, 4))
    plt.plot(xs, ys, "o-", color="#c0392b")
    for r in rs:
        plt.annotate(r["name"], (r["params_total"]/1e6, r["best_valw"]),
                     textcoords="offset points", xytext=(6, 6), fontsize=9)
    plt.xlabel("Parameters (M)"); plt.ylabel("Best weighted val loss (ValW, nats)")
    plt.title("Ḍād scaling sweep on fixed 170M-token corpus")
    plt.grid(alpha=0.3); plt.tight_layout(); plt.savefig(out, dpi=140)
    print(f"✅ wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default="5m,15m,30m,47m")
    ap.add_argument("--epochs", type=float, default=4.0)
    ap.add_argument("--micro_batch", type=int, default=16)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--block", type=int, default=512)
    ap.add_argument("--log_interval", type=int, default=50)
    ap.add_argument("--eval_batch", type=int, default=8)
    ap.add_argument("--eval_max_batches", type=int, default=40)
    ap.add_argument("--max_seconds", type=int, default=0, help="0 = no wall-clock cap")
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--pilot_steps", type=int, default=300)
    ap.add_argument("--out", type=Path, default=Path("scaling_results.json"))
    ap.add_argument("--plot", type=Path, default=Path("scaling_curve.png"))
    args = ap.parse_args()

    device = get_device()
    print(f"Device: {device} │ pilot={args.pilot}")

    names = ["5m"] if args.pilot else [c.strip() for c in args.configs.split(",")]
    results = []
    for name in names:
        if name not in CONFIGS:
            print(f"skip unknown config {name}"); continue
        r = train_one(name, CONFIGS[name], args, device)
        results.append(r)
        # write incrementally so a crash mid-sweep still leaves data
        args.out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.pilot:
        r = results[0]
        ok = (not r["diverged"]) and r["curve"] and r["curve"][-1]["train"] < r["curve"][0]["train"] \
             and (SWEEP_DIR / "5m_best.pt").exists()
        print("\n" + "="*60)
        print(f"PILOT VERDICT: {'✅ harness OK' if ok else '⚠️ LOOKS WRONG — inspect before full sweep'}")
        print(f"  checkpoint written: {(SWEEP_DIR/'5m_best.pt').exists()}")
        print(f"  train loss moved: {r['curve'][0]['train'] if r['curve'] else '?'} → {r['final_train']}")
        print(f"  eval ran: {len(r['curve'])} eval point(s); final ValW {r['final_valw']}")
        print("="*60)
        return

    plot(results, args.plot)
    print("\n=== SWEEP SUMMARY (best ValW vs params) ===")
    for r in results:
        print(f"  {r['name']:<4} {r['params_total']/1e6:5.1f}M → ValW {r['best_valw']:.4f} "
              f"│ gap {r['gap_flat_minus_train']:+.3f} │ {'DIVERGED' if r['diverged'] else 'ok'}")
    print(f"\n✅ wrote {args.out} and {args.plot}")


if __name__ == "__main__":
    main()
