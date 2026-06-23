"""
MiniFrontier - Inférence & Génération Interactive
===================================================
Charge un checkpoint entraîné et génère du texte en arabe.

Usage :
    python generate.py --prompt "كان يا ما كان"
    python generate.py --prompt "السماء" --temperature 0.7 --max_tokens 200
    python generate.py --interactive
"""

import argparse
import logging
from pathlib import Path

import torch

from config import ModelConfig, InferConfig, infer_cfg, CHECKPOINT_DIR, get_device
from model import MiniFrontierLLM
from tokenizer_arabic import ArabicTokenizer, normalize_arabic
from config import TOKENIZER_PATH

logging.basicConfig(level=logging.INFO, format="%(levelname)s │ %(message)s")
log = logging.getLogger(__name__)


def load_model(ckpt_path: Path, device: torch.device) -> tuple[MiniFrontierLLM, ArabicTokenizer]:
    """Charge le modèle et le tokenizer depuis un checkpoint."""
    log.info(f"Chargement du modèle : {ckpt_path.name}")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg  = ModelConfig(**ckpt["model_cfg"])

    model = MiniFrontierLLM(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    tok = ArabicTokenizer(TOKENIZER_PATH)

    step      = ckpt.get("step", "?")
    val_loss  = ckpt.get("val_loss", float("nan"))
    log.info(f"✅ Modèle chargé │ step={step} │ val_loss={val_loss:.4f}")
    log.info(f"   Paramètres  : {model.get_num_params(non_embedding=False)/1e6:.2f}M "
             f"(dont {model.get_num_params(non_embedding=True)/1e6:.2f}M hors embedding)")
    log.info(f"   Vocab size  : {cfg.vocab_size} │ block_size: {cfg.block_size}")

    return model, tok


# ── Guard anti-fabrication coranique ────────────────────────────────────────────
# Le masquage à l'entraînement empêche de MÉMORISER les versets (﴿…﴾), mais pas
# d'ÉMETTRE le CADRE « قال تعالى / قوله تعالى / قال الله تعالى » puis de fabriquer
# un faux verset dans le trou. On supprime donc au sampling tout token portant
# l'épithète divine « تعالى » (forme normalisée « تعالي » car ى→ي), ce qui rend le
# cadre de citation impossible à former. Surgical : « الله » reste autorisé.
_QURAN_FRAME_SUBSTRINGS = ("تعالى", "تعالي", "﴿", "﴾")


def quran_frame_token_ids(tok: ArabicTokenizer) -> list[int]:
    """IDs de tous les tokens du vocab contenant un marqueur de cadre coranique."""
    subs = set()
    for s in _QURAN_FRAME_SUBSTRINGS:
        for v in (s, normalize_arabic(s)):
            if v:
                subs.add(v)
    bad = [tid for tid in range(tok.vocab_size)
           if any(sub in tok.decode([tid]) for sub in subs)]
    return bad


@torch.inference_mode()
def generate_text(
    model: MiniFrontierLLM,
    tok:   ArabicTokenizer,
    prompt: str,
    cfg:   InferConfig = infer_cfg,
    device: torch.device | None = None,
    bad_token_ids: list[int] | None = None,
) -> str:
    """Génère du texte à partir d'un prompt arabe."""
    if device is None:
        device = get_device()

    # Encodage du prompt
    norm_prompt = normalize_arabic(prompt)
    ids = tok.encode(norm_prompt, add_bos=True, add_eos=False)
    x   = torch.tensor([ids], dtype=torch.long, device=device)

    # Génération
    out = model.generate(
        x,
        max_new_tokens=cfg.max_new_tokens,
        temperature=cfg.temperature,
        top_k=cfg.top_k,
        top_p=cfg.top_p,
        repetition_penalty=cfg.repetition_penalty,
        eos_id=tok.eos_id,
        bad_token_ids=bad_token_ids,
    )

    # Décodage (exclure le prompt d'origine)
    generated_ids = out[0, len(ids):].tolist()
    return tok.decode(generated_ids)


def interactive_mode(model: MiniFrontierLLM, tok: ArabicTokenizer,
                     cfg: InferConfig, device: torch.device,
                     bad_token_ids: list[int] | None = None):
    """REPL interactif pour générer du texte en arabe."""
    print("\n" + "═"*60)
    print("  MiniFrontier — Mode Interactif (Ctrl+C pour quitter)")
    print("═"*60)
    print(f"  Température : {cfg.temperature} │ Top-P : {cfg.top_p} │ Top-K : {cfg.top_k}")
    if bad_token_ids:
        print(f"  Guard Coran : ON ({len(bad_token_ids)} tokens « تعالى/﴿﴾ » supprimés)")
    print("═"*60 + "\n")

    while True:
        try:
            prompt = input("📝 Prompt : ").strip()
            if not prompt:
                continue

            print("\n📖 Génération…\n")
            result = generate_text(model, tok, prompt, cfg, device, bad_token_ids)
            print("─"*50)
            print(f"  {prompt}{result}")
            print("─"*50 + "\n")

        except KeyboardInterrupt:
            print("\n\nAu revoir !")
            break
        except Exception as e:
            log.error(f"Erreur : {e}")


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniFrontier — Génération de texte arabe")
    parser.add_argument("--prompt",       type=str,   default=None)
    parser.add_argument("--interactive",  action="store_true")
    parser.add_argument("--ckpt",         type=Path,  default=CHECKPOINT_DIR / "ckpt_best.pt")
    parser.add_argument("--temperature",  type=float, default=infer_cfg.temperature)
    parser.add_argument("--top_p",        type=float, default=infer_cfg.top_p)
    parser.add_argument("--top_k",        type=int,   default=infer_cfg.top_k)
    parser.add_argument("--max_tokens",   type=int,   default=infer_cfg.max_new_tokens)
    parser.add_argument("--rep_penalty",  type=float, default=infer_cfg.repetition_penalty)
    parser.add_argument("--allow_quran",  action="store_true",
                        help="désactive le guard anti-fabrication coranique (par défaut: ON)")
    args = parser.parse_args()

    # Appliquer les overrides
    infer_cfg.temperature        = args.temperature
    infer_cfg.top_p              = args.top_p
    infer_cfg.top_k              = args.top_k
    infer_cfg.max_new_tokens     = args.max_tokens
    infer_cfg.repetition_penalty = args.rep_penalty

    if not args.ckpt.exists():
        log.error(f"Checkpoint introuvable : {args.ckpt}")
        log.error("Entraînez d'abord le modèle avec : python train.py")
        raise SystemExit(1)

    device = get_device()
    model, tok = load_model(args.ckpt, device)

    # Guard anti-fabrication coranique (actif par défaut, désactivable)
    bad_ids = None if args.allow_quran else quran_frame_token_ids(tok)
    if bad_ids:
        log.info(f"🛡️  Guard Coran ACTIF — {len(bad_ids)} tokens « تعالى/﴿﴾ » supprimés "
                 f"du sampling (--allow_quran pour désactiver)")

    if args.interactive:
        interactive_mode(model, tok, infer_cfg, device, bad_ids)
    elif args.prompt:
        result = generate_text(model, tok, args.prompt, infer_cfg, device, bad_ids)
        print(f"\n{args.prompt}{result}\n")
    else:
        # Mode démo avec quelques prompts de test
        test_prompts = [
            "كان يا ما كان في قديم الزمان",
            "الكتاب خير صديق",
            "السماء صافية و",
            "قال الولد الصغير",
        ]
        for p in test_prompts:
            result = generate_text(model, tok, p, infer_cfg, device, bad_ids)
            print(f"\n  [{p}] → {result[:120]}…")
