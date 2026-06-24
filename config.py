"""
MiniFrontier - Configuration Centralisée
Toutes les hyperparamètres et chemins au même endroit.
"""
from dataclasses import dataclass, field
from pathlib import Path
import torch


# ── Chemins du projet ─────────────────────────────────────────────────────────
ROOT_DIR       = Path(__file__).parent
DATA_DIR       = ROOT_DIR / "data"
TOKENIZER_DIR  = ROOT_DIR / "tokenizer"
CHECKPOINT_DIR = ROOT_DIR / "checkpoints"
LOG_DIR        = ROOT_DIR / "logs"

for d in (DATA_DIR, TOKENIZER_DIR, CHECKPOINT_DIR, LOG_DIR):
    d.mkdir(parents=True, exist_ok=True)

TOKENIZER_PATH = TOKENIZER_DIR / "arabic_bpe_16k.json"
CORPUS_BIN     = DATA_DIR / "corpus.bin"        # données tokenisées (numpy memmap)
CORPUS_RAW_DIR = DATA_DIR / "raw"               # fichiers .txt arabes bruts

# ── Corpus Shamela (dépôt voisin arabic-corpus) ──────────────────────────────
# Le corpus réel est préparé séparément dans arabic-corpus/ ; mini_frontier le
# consomme via le manifeste pondéré (colonne mix_weight) SANS toucher aux sources.
ARABIC_CORPUS_DIR = Path("/home/rapit/Desktop/arabic-corpus")
SHAMELA_STAGING   = ARABIC_CORPUS_DIR / "staging"
CORPUS_MANIFEST   = SHAMELA_STAGING / "corpus_manifest.csv"

# Flux texte pondéré intermédiaire (de-vocalisé + normalisé + mélangé).
# (1) sert à entraîner le tokenizer BPE, puis (2) est encodé en train/val.bin.
SHAMELA_STREAM_DIR   = DATA_DIR / "shamela"
SHAMELA_STREAM       = SHAMELA_STREAM_DIR / "stream.txt"
SHAMELA_SHUFFLE_SEED = 42

# ── Évaluation pondérée par catégorie ────────────────────────────────────────
# La val loss "plate" (moyenne par token) est dominée par أدب/معاجم, justement
# les catégories sous-pondérées à l'entraînement. On rapporte donc :
#   (1) la val loss PAR catégorie (séparément), et
#   (2) une val loss pondérée selon le MIX EFFECTIF d'entraînement ci-dessous,
#       pour que la métrique suive ce qu'on optimise réellement.
# عروض (cat 33, 1.5%) n'a aucun livre de validation → absent ici ; les poids
# sont renormalisés sur les catégories réellement présentes (somme = 0.985).
VAL_EVAL_MIX = {
    31: 0.45,    # النحو والصرف
    32: 0.22,    # الأدب
    30: 0.15,    # الغريب والمعاجم
    29: 0.07,    # كتب اللغة
    35: 0.07,    # البلاغة
    34: 0.025,   # الشعر ودواوينه
}
VAL_CAT_NAMES = {
    29: "لغة", 30: "معاجم", 31: "نحو/صرف", 32: "أدب",
    33: "عروض", 34: "شعر", 35: "بلاغة",
}


def val_cat_bin(cat: int) -> Path:
    """Chemin du .bin de validation pour une catégorie donnée."""
    return DATA_DIR / f"val_cat{cat}.bin"


# ── Architecture du modèle ────────────────────────────────────────────────────
@dataclass
class ModelConfig:
    # Vocabulaire : 16k tokens, optimisé pour l'arabe mono-langue
    vocab_size:  int = 16_000

    # Contexte : 512 tokens (256 pour debug rapide)
    block_size:  int = 512

    # Profondeur : 8 blocs — bon équilibre capacité/vitesse pour ~10M params
    n_layer:     int = 8

    # Têtes d'attention : 8 — head_dim = 256/8 = 32
    n_head:      int = 8

    # Taille des embeddings
    n_embd:      int = 256

    # FFN SwiGLU : ≈ (8/3) * n_embd, arrondi à un multiple de 64 pour les kernels CUDA
    hidden_dim:  int = 704   # 2.75 × 256, multiple de 64

    # Dropout (0.0 = désactivé pour un pré-entraînement standard)
    dropout:     float = 0.0

    # Bias : False — standard post-LLaMA (économise ~1% de VRAM)
    bias:        bool  = False

    # RoPE base (10_000 standard, 500_000 pour YaRN long-context)
    rope_base:   float = 10_000.0

    # Weight tying entre l'embedding d'entrée et lm_head.
    # v1 : False (standard Llama). v2 : True — libère ~4.1M params (28% du modèle)
    # redéployés dans le calcul. Voir docs/RESEARCH_v1_to_v2.md §6.2.
    tie_embeddings: bool = False


# ── Entraînement ──────────────────────────────────────────────────────────────
@dataclass
class TrainConfig:
    # --- Batch & accumulation ---
    # Effective batch = micro_batch_size × block_size × grad_accum
    # = 16 × 512 × 8 = 65 536 tokens/step  (confortable sur RTX 3090 24 GB)
    micro_batch_size:           int   = 16
    gradient_accumulation_steps: int  = 8

    # --- Scheduler LR (Cosine Warmup + Decay) ---
    # peak LR abaissé de 3e-4 → 1.5e-4 : 3e-4 (valeur GPT-2/nanoGPT) suppose un
    # batch ~500k tokens. Ici le batch effectif = 65k tokens (8× plus petit),
    # donc gradients plus bruités → un LR trop élevé fait diverger la loss dès la
    # fin du warmup (loss qui descend puis remonte, grad_norm qui gonfle).
    learning_rate:  float = 1.5e-4   # peak LR (sûr pour batch 65k + FP16 T4)
    min_lr:         float = 1.5e-5   # fin du cosine (= peak/10)
    warmup_iters:   int   = 400      # warmup plus long = montée plus douce
    max_iters:      int   = 10_000   # ~655 M tokens vus (65 536 × 10 000)
    lr_decay_iters: int   = 10_000   # aligner avec max_iters

    # --- Régularisation ---
    weight_decay:   float = 0.1
    grad_clip:      float = 1.0
    beta1:          float = 0.9
    beta2:          float = 0.95

    # --- Précision ---
    # "auto" détecte automatiquement : bfloat16 si Ampere+ (RTX 3090, A100…)
    #                                   float16  sinon (T4, V100, GTX 1080…)
    # Forcer manuellement : "bfloat16" ou "float16"
    dtype: str = "auto"

    # --- Évaluation & sauvegarde ---
    eval_interval:  int   = 500
    eval_iters:     int   = 100
    save_interval:  int   = 1000
    log_interval:   int   = 50

    # --- Tokenizer ---
    tokenizer_vocab_size: int = 16_000

    # --- Compilation (torch.compile) ---
    compile_model: bool = True   # PyTorch >= 2.0  — accélère de ~10-20%

    # --- Validation split ---
    val_ratio: float = 0.05      # 5% du corpus pour la validation


# ── Inférence ─────────────────────────────────────────────────────────────────
@dataclass
class InferConfig:
    temperature:    float = 0.8
    top_p:          float = 0.9     # nucleus sampling
    top_k:          int   = 50
    max_new_tokens: int   = 300
    repetition_penalty: float = 1.1


# ── Device auto-detect ────────────────────────────────────────────────────────
def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_dtype(dtype_str: str) -> str:
    """
    Résout "auto" vers le dtype optimal selon la capacité du GPU.
    • Ampere+ (compute capability >= 8.0) → bfloat16 (RTX 3090, A100, RTX 5060 Ti)
    • Turing / Volta / Pascal (T4, V100, GTX…) → float16
    • CPU / MPS → float32
    """
    if dtype_str != "auto":
        return dtype_str
    if torch.cuda.is_available():
        major, _ = torch.cuda.get_device_capability()
        chosen = "bfloat16" if major >= 8 else "float16"
        gpu_name = torch.cuda.get_device_name(0)
        print(f"[config] GPU: {gpu_name} (cc {major}.x) → dtype={chosen}")
        return chosen
    return "float32"


def get_autocast_ctx(device: torch.device, dtype_str: str):
    """Retourne le contexte autocast approprié selon le GPU détecté."""
    if device.type == "cuda":
        resolved = resolve_dtype(dtype_str)
        pt_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}.get(resolved, torch.float32)
        return torch.autocast(device_type="cuda", dtype=pt_dtype)
    return torch.autocast(device_type="cpu", enabled=False)  # no-op sur CPU


# ── Instances globales (importées par les autres modules) ─────────────────────
model_cfg = ModelConfig()
train_cfg = TrainConfig()
infer_cfg = InferConfig()

# ── Ḍād-v2 — Config C (~47M, weight-tied) ─────────────────────────────────────
# Cible v2 : n_embd 512 · 12 couches · head_dim 64 · hidden 1408 · TIED.
# 46.7M total / 38.5M non-embedding (6× le calcul de v1). Corpus optimal ~935M
# tokens (Chinchilla). block_size reste 512 ; passer à 1024 est recommandé contre
# le looping si la VRAM 3090 le permet (voir RESEARCH §7.2). NE PAS utiliser pour
# v1 — c'est un modèle séparé (nouveau tokenizer + corpus élargi).
model_cfg_v2 = ModelConfig(
    n_embd=512,
    n_layer=12,
    n_head=8,            # head_dim = 512/8 = 64
    hidden_dim=1408,     # ≈ (8/3)·512, multiple de 64
    block_size=512,
    tie_embeddings=True,
)
