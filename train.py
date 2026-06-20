"""
MiniFrontier - Boucle d'Entraînement
======================================
Entraînement production-ready avec :
  • Cosine LR schedule avec warmup
  • Mixed precision BFloat16 (natif RTX 3000+)
  • Gradient accumulation (simule un large batch)
  • Gradient clipping
  • Checkpointing automatique (meilleur val loss)
  • Logging structuré (console + fichier JSON)
  • Estimation MFU (efficacité GPU)
  • torch.compile() pour +10-20% de vitesse
  • Reprise depuis un checkpoint

Usage :
    python train.py                         # nouvel entraînement
    python train.py --resume                # reprendre depuis le dernier checkpoint
    python train.py --resume --ckpt checkpoints/ckpt_step4000.pt
"""

import argparse
import json
import logging
import math
import time
from pathlib import Path

import torch
import torch.nn as nn

from config import (
    TrainConfig, ModelConfig, train_cfg, model_cfg,
    CHECKPOINT_DIR, LOG_DIR,
    get_device, get_autocast_ctx,
    VAL_EVAL_MIX, VAL_CAT_NAMES, val_cat_bin,
)
from model import MiniFrontierLLM
from data_pipeline import MemmapDataset, TRAIN_BIN, VAL_BIN
from tokenizer_arabic import ArabicTokenizer
from config import TOKENIZER_PATH

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)s │ %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / "train.log"),
    ],
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Learning Rate Scheduler (Cosine Warmup)
# ─────────────────────────────────────────────────────────────────────────────
def get_lr(step: int, cfg: TrainConfig) -> float:
    """
    Schedule en 3 phases :
      1. Warmup linéaire : 0 → learning_rate sur warmup_iters steps
      2. Cosine decay   : learning_rate → min_lr jusqu'à lr_decay_iters
      3. Plateau        : min_lr constant au-delà
    """
    if step < cfg.warmup_iters:
        return cfg.learning_rate * step / max(cfg.warmup_iters, 1)
    if step >= cfg.lr_decay_iters:
        return cfg.min_lr
    progress = (step - cfg.warmup_iters) / max(cfg.lr_decay_iters - cfg.warmup_iters, 1)
    coeff    = 0.5 * (1.0 + math.cos(math.pi * progress))
    return cfg.min_lr + coeff * (cfg.learning_rate - cfg.min_lr)


# ─────────────────────────────────────────────────────────────────────────────
# Optimiseur (AdamW avec weight decay sélectif)
# ─────────────────────────────────────────────────────────────────────────────
def build_optimizer(model: nn.Module, cfg: TrainConfig, device: torch.device) -> torch.optim.Optimizer:
    """
    Sépare les paramètres 2D+ (matrices de poids → weight decay)
    des paramètres 1D (biais, RMSNorm weight → pas de weight decay).
    Utilise le fused AdamW CUDA si disponible (+15% de vitesse).
    """
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.dim() >= 2:
            decay.append(p)
        else:
            no_decay.append(p)

    log.info(f"Optimizer │ decay params: {sum(p.numel() for p in decay)/1e6:.2f}M "
             f"│ no_decay: {sum(p.numel() for p in no_decay)/1e6:.2f}M")

    optim_groups = [
        {"params": decay,    "weight_decay": cfg.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]

    use_fused = device.type == "cuda"
    optimizer = torch.optim.AdamW(
        optim_groups,
        lr=cfg.learning_rate,
        betas=(cfg.beta1, cfg.beta2),
        fused=use_fused,
    )
    return optimizer


# ─────────────────────────────────────────────────────────────────────────────
# Évaluation
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(
    model: nn.Module,
    val_ds: MemmapDataset,
    cfg_train: TrainConfig,
    cfg_model: ModelConfig,
    device: torch.device,
    autocast_ctx,
) -> float:
    """Calcule la val loss sur eval_iters micro-batches."""
    model.eval()
    losses = []
    for _ in range(cfg_train.eval_iters):
        X, Y = val_ds.get_random_batch(cfg_train.micro_batch_size)
        X, Y = X.to(device, non_blocking=True), Y.to(device, non_blocking=True)
        with autocast_ctx:
            _, loss = model(X, Y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


@torch.no_grad()
def evaluate_categories(
    model: nn.Module,
    val_datasets: dict,
    cfg_train: TrainConfig,
    device: torch.device,
    autocast_ctx,
) -> dict:
    """
    Val loss SÉPARÉE par catégorie : échantillonne eval_iters micro-batches dans
    chaque val_cat{N}.bin indépendamment. Retourne {cat_id: loss}.
    """
    model.eval()
    out = {}
    for cat, ds in val_datasets.items():
        losses = []
        for _ in range(cfg_train.eval_iters):
            X, Y = ds.get_random_batch(cfg_train.micro_batch_size)
            X, Y = X.to(device, non_blocking=True), Y.to(device, non_blocking=True)
            with autocast_ctx:
                _, loss = model(X, Y)
            losses.append(loss.item())
        out[cat] = sum(losses) / len(losses)
    model.train()
    return out


def weighted_val_loss(per_cat: dict, mix: dict = VAL_EVAL_MIX) -> float:
    """
    Val loss pondérée par le MIX EFFECTIF d'entraînement (نحو 45%, أدب 22%…),
    et non une moyenne plate par token. Les poids sont renormalisés sur les
    catégories réellement présentes (عروض absent → somme < 1 → renorm).
    """
    avail = {c: mix[c] for c in per_cat if c in mix}
    Z = sum(avail.values())
    if Z <= 0:
        return float("nan")
    return sum(per_cat[c] * w for c, w in avail.items()) / Z


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint
# ─────────────────────────────────────────────────────────────────────────────
def save_checkpoint(
    step: int, model: nn.Module, optimizer: torch.optim.Optimizer,
    val_loss: float, cfg_model: ModelConfig, cfg_train: TrainConfig,
    scaler=None, is_best: bool = False,
):
    raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    ckpt = {
        "step":       step,
        "val_loss":   val_loss,
        "model":      raw_model.state_dict(),
        "optimizer":  optimizer.state_dict(),
        "scaler":     scaler.state_dict() if scaler is not None else None,
        "model_cfg":  cfg_model.__dict__,
        "train_cfg":  cfg_train.__dict__,
    }
    path = CHECKPOINT_DIR / f"ckpt_step{step:05d}.pt"
    torch.save(ckpt, path)
    log.info(f"💾 Checkpoint sauvegardé → {path.name} (val_loss={val_loss:.4f})")

    if is_best:
        best_path = CHECKPOINT_DIR / "ckpt_best.pt"
        torch.save(ckpt, best_path)
        log.info(f"⭐ Nouveau meilleur modèle → {best_path.name}")


def load_checkpoint(
    ckpt_path: Path, model: nn.Module, optimizer: torch.optim.Optimizer,
    device: torch.device, scaler=None,
) -> int:
    """Charge un checkpoint. Retourne le step de reprise."""
    log.info(f"Chargement du checkpoint : {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    # Charger les poids dans le modèle raw (avant torch.compile)
    raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    raw_model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])

    # Restaurer l'état du GradScaler (FP16) s'il existe
    if scaler is not None and ckpt.get("scaler") is not None:
        scaler.load_state_dict(ckpt["scaler"])

    step = ckpt["step"]
    log.info(f"Reprise à l'étape {step} (val_loss précédente: {ckpt['val_loss']:.4f})")
    return step


# ─────────────────────────────────────────────────────────────────────────────
# Boucle principale d'entraînement
# ─────────────────────────────────────────────────────────────────────────────
def train(cfg_train: TrainConfig, cfg_model: ModelConfig, resume: bool = False,
          resume_ckpt: Path | None = None):

    device       = get_device()
    autocast_ctx = get_autocast_ctx(device, cfg_train.dtype)

    # GradScaler OBLIGATOIRE en float16 (T4, V100…) : la plage dynamique étroite
    # du FP16 fait overflow/underflow les gradients sans scaling → divergence.
    # BF16 (RTX 3090, A100…) n'en a PAS besoin (même plage que FP32).
    from config import resolve_dtype
    resolved_dtype = resolve_dtype(cfg_train.dtype)
    use_scaler = (device.type == "cuda" and resolved_dtype == "float16")
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    if use_scaler:
        log.info("🔧 GradScaler ACTIVÉ (float16) — protège contre l'overflow des gradients")

    log.info(f"🖥️  Device : {device}  │  Dtype : {resolved_dtype}")
    log.info(f"Effective batch size : {cfg_train.micro_batch_size * cfg_train.gradient_accumulation_steps * cfg_model.block_size:,} tokens/step")

    # ── Datasets ──────────────────────────────────────────────────────────────
    train_ds = MemmapDataset(TRAIN_BIN, cfg_model.block_size)
    val_ds   = MemmapDataset(VAL_BIN,   cfg_model.block_size)   # plat (référence)

    # Val par catégorie : un dataset par val_cat{N}.bin présent sur disque.
    # Permet la val loss séparée + la val loss pondérée par le mix d'entraînement.
    val_datasets = {}
    for cat in VAL_EVAL_MIX:
        p = val_cat_bin(cat)
        if p.exists():
            val_datasets[cat] = MemmapDataset(p, cfg_model.block_size)
        else:
            log.warning(f"val_cat{cat}.bin absent ({p}) — catégorie exclue de l'éval. "
                        f"Lancer : python data_pipeline.py --emit_val_cat_bins")
    if not val_datasets:
        log.warning("Aucun val_cat{N}.bin trouvé — repli sur la val loss plate uniquement.")

    # ── Modèle ────────────────────────────────────────────────────────────────
    model = MiniFrontierLLM(cfg_model).to(device)

    # torch.compile (PyTorch >= 2.0) — réduit les overheads Python
    if cfg_train.compile_model and device.type == "cuda":
        log.info("⚡ Compilation du modèle avec torch.compile…")
        model = torch.compile(model)
        log.info("✅ Compilation terminée")

    # ── Optimiseur ────────────────────────────────────────────────────────────
    optimizer = build_optimizer(model, cfg_train, device)

    # ── Reprise ───────────────────────────────────────────────────────────────
    start_step = 0
    if resume:
        ckpt_path = resume_ckpt or (CHECKPOINT_DIR / "ckpt_best.pt")
        if ckpt_path.exists():
            start_step = load_checkpoint(ckpt_path, model, optimizer, device, scaler=scaler)
        else:
            log.warning(f"Checkpoint introuvable : {ckpt_path}. Démarrage à zéro.")

    # ── Log JSON ──────────────────────────────────────────────────────────────
    log_file = LOG_DIR / "metrics.jsonl"
    log_fh   = open(log_file, "a")

    best_val_loss = float("inf")
    model.train()

    # Premier batch (CPU → GPU asynchrone)
    X, Y = train_ds.get_random_batch(cfg_train.micro_batch_size)
    X = X.to(device, non_blocking=True)
    Y = Y.to(device, non_blocking=True)

    t0          = time.perf_counter()
    tokens_seen = start_step * cfg_train.micro_batch_size * cfg_train.gradient_accumulation_steps * cfg_model.block_size

    log.info(f"\n{'─'*60}")
    log.info(f"🚀 Démarrage de l'entraînement — {cfg_train.max_iters - start_step} steps restants")
    log.info(f"{'─'*60}\n")

    for step in range(start_step, cfg_train.max_iters):

        # ── Learning Rate ──────────────────────────────────────────────────
        lr = get_lr(step, cfg_train)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # ── Évaluation périodique ──────────────────────────────────────────
        if step % cfg_train.eval_interval == 0:
            # Val plate (référence) + val par catégorie + val pondérée (mix train).
            flat_val = evaluate(model, val_ds, cfg_train, cfg_model, device, autocast_ctx)
            per_cat  = (evaluate_categories(model, val_datasets, cfg_train, device, autocast_ctx)
                        if val_datasets else {})
            w_val    = weighted_val_loss(per_cat) if per_cat else flat_val

            # Le "best" suit la val PONDÉRÉE (ce qu'on optimise), pas la plate.
            is_best  = w_val < best_val_loss
            if is_best:
                best_val_loss = w_val

            cat_str = " ".join(
                f"{VAL_CAT_NAMES.get(c, c)}={per_cat[c]:.3f}" for c in sorted(per_cat)
            )
            log.info(
                f"Étape {step:>5d}/{cfg_train.max_iters} │ "
                f"ValW: {w_val:.4f} │ Best: {best_val_loss:.4f} │ "
                f"ValFlat: {flat_val:.4f} │ LR: {lr:.2e} │ "
                f"Tokens: {tokens_seen/1e6:.1f}M"
            )
            if cat_str:
                log.info(f"        par catégorie │ {cat_str}")

            log_fh.write(json.dumps({
                "step": step,
                "val_weighted": w_val,
                "val_flat": flat_val,
                "val_per_cat": {str(c): per_cat[c] for c in per_cat},
                "lr": lr,
                "tokens_M": tokens_seen / 1e6,
            }) + "\n")
            log_fh.flush()

            # Sauvegarde si meilleure val pondérée
            if is_best or step % cfg_train.save_interval == 0:
                save_checkpoint(step, model, optimizer, w_val, cfg_model, cfg_train, scaler=scaler, is_best=is_best)

        # ── Forward / Backward avec Gradient Accumulation ─────────────────
        optimizer.zero_grad(set_to_none=True)
        train_loss_accum = 0.0

        for micro_step in range(cfg_train.gradient_accumulation_steps):
            with autocast_ctx:
                _, loss = model(X, Y)
                loss    = loss / cfg_train.gradient_accumulation_steps

            # Charger le prochain batch en parallèle du backward
            X_next, Y_next = train_ds.get_random_batch(cfg_train.micro_batch_size)
            X_next = X_next.to(device, non_blocking=True)
            Y_next = Y_next.to(device, non_blocking=True)

            # backward sur la loss scalée (no-op si scaler désactivé en BF16)
            scaler.scale(loss).backward()
            train_loss_accum += loss.item()

            X, Y = X_next, Y_next

        tokens_seen += (
            cfg_train.micro_batch_size
            * cfg_train.gradient_accumulation_steps
            * cfg_model.block_size
        )

        # ── Gradient Clipping ─────────────────────────────────────────────
        # Unscale AVANT le clipping : sinon on clippe des gradients scalés
        # (×~65000 en FP16), ce qui rend grad_clip inopérant.
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg_train.grad_clip)

        # ── Mise à jour ───────────────────────────────────────────────────
        # scaler.step skip l'update si des inf/nan sont détectés (overflow FP16),
        # puis scaler.update ajuste dynamiquement le facteur d'échelle.
        scaler.step(optimizer)
        scaler.update()

        # ── Monitoring ────────────────────────────────────────────────────
        if step % cfg_train.log_interval == 0:
            t1 = time.perf_counter()
            dt = t1 - t0
            t0 = t1

            # train_loss_accum contient DÉJÀ la loss moyenne réelle : dans la
            # micro-boucle, chaque loss est divisée par grad_accum AVANT le
            # .item(), donc la somme des grad_accum termes = loss moyenne.
            # NE PAS re-multiplier par grad_accum ici (sinon ×grad_accum, ce qui
            # affichait 77 au lieu de 9.7 — au-dessus du max théorique ln(vocab)).
            real_loss = train_loss_accum

            # Estimation MFU
            raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
            mfu = raw_model.estimate_mfu(
                cfg_train.micro_batch_size * cfg_train.gradient_accumulation_steps, dt
            )

            tokens_per_sec = (
                cfg_train.micro_batch_size
                * cfg_train.gradient_accumulation_steps
                * cfg_model.block_size
                * cfg_train.log_interval
            ) / dt

            log.info(
                f"  step {step:>5d} │ "
                f"loss {real_loss:.4f} │ "
                f"grad_norm {grad_norm:.3f} │ "
                f"tok/s {tokens_per_sec:,.0f} │ "
                f"MFU {mfu*100:.1f}%"
            )

    log.info(f"\n{'─'*60}")
    log.info(f"✅ Entraînement terminé — {cfg_train.max_iters} steps")
    log.info(f"   Meilleur val loss : {best_val_loss:.4f}")
    log.info(f"   Tokens vus        : {tokens_seen/1e9:.2f}B")
    log.info(f"{'─'*60}\n")
    log_fh.close()


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniFrontier — Entraînement")
    parser.add_argument("--resume",  action="store_true",  help="Reprendre depuis un checkpoint")
    parser.add_argument("--ckpt",    type=Path, default=None, help="Chemin du checkpoint à charger")

    # Overrides CLI pour les hyperparamètres fréquents
    parser.add_argument("--lr",              type=float, default=None)
    parser.add_argument("--max_iters",       type=int,   default=None)
    parser.add_argument("--lr_decay_iters",  type=int,   default=None,
                        help="Step où le cosine LR atteint min_lr. "
                             "Auto-synchronisé sur max_iters si omis.")
    parser.add_argument("--warmup_iters",    type=int,   default=None)
    parser.add_argument("--batch_size",      type=int,   default=None)
    parser.add_argument("--block_size",      type=int,   default=None)
    parser.add_argument("--no_compile",      action="store_true")
    args = parser.parse_args()

    # Appliquer les overrides
    if args.lr         is not None: train_cfg.learning_rate       = args.lr
    if args.batch_size is not None: train_cfg.micro_batch_size    = args.batch_size
    if args.block_size is not None: model_cfg.block_size          = args.block_size
    if args.no_compile:             train_cfg.compile_model       = False

    if args.max_iters is not None:
        train_cfg.max_iters = args.max_iters
        # CRITIQUE : si --max_iters est réduit (ex: run de test à 1000 steps)
        # mais lr_decay_iters reste à sa valeur par défaut (10000), le cosine
        # schedule n'a quasiment pas le temps de décroître : le LR reste collé
        # près du pic pendant tout le run, ce qui produit une loss qui descend
        # puis REMONTE en fin de run (overshoot permanent, pas de consolidation).
        # On resynchronise donc lr_decay_iters sur max_iters sauf override explicite.
        if args.lr_decay_iters is not None:
            train_cfg.lr_decay_iters = args.lr_decay_iters
        else:
            train_cfg.lr_decay_iters = args.max_iters
            log.info(
                f"⚠️  --max_iters={args.max_iters} fourni sans --lr_decay_iters : "
                f"lr_decay_iters resynchronisé à {args.max_iters} "
                f"(sinon le LR ne décroît jamais sur un run court)."
            )
    elif args.lr_decay_iters is not None:
        train_cfg.lr_decay_iters = args.lr_decay_iters

    if args.warmup_iters is not None:
        train_cfg.warmup_iters = args.warmup_iters
    elif args.max_iters is not None and train_cfg.warmup_iters >= train_cfg.max_iters:
        # Garde-fou : le warmup ne doit jamais dépasser le run entier
        train_cfg.warmup_iters = max(1, train_cfg.max_iters // 10)
        log.info(f"⚠️  warmup_iters resynchronisé à {train_cfg.warmup_iters} (10% de max_iters)")

    if args.max_iters is not None and train_cfg.eval_interval > train_cfg.max_iters:
        # Garde-fou : sinon aucune évaluation/checkpoint entre step 0 et la fin
        train_cfg.eval_interval = max(1, train_cfg.max_iters // 4)
        log.info(f"⚠️  eval_interval resynchronisé à {train_cfg.eval_interval}")

    # Sync vocab_size depuis le tokenizer
    if TOKENIZER_PATH.exists():
        tok = ArabicTokenizer(TOKENIZER_PATH)
        model_cfg.vocab_size = tok.vocab_size
        log.info(f"Vocab size depuis tokenizer : {model_cfg.vocab_size}")
    else:
        log.warning("Tokenizer non trouvé — utilisation de vocab_size par défaut (16000)")

    train(train_cfg, model_cfg, resume=args.resume, resume_ckpt=args.ckpt)
