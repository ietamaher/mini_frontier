"""
Tests fumée (smoke tests) — MiniFrontier
=========================================
Vérifications rapides que les composants de base fonctionnent, SANS nécessiter
de GPU ni de modèle entraîné. À lancer après toute modification structurelle.

    pytest tests/
    # ou sans pytest :
    python tests/test_smoke.py
"""

import sys
from pathlib import Path

# Permet d'importer les modules du projet depuis tests/
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from config import ModelConfig, TrainConfig, get_device, get_autocast_ctx
from model import MiniFrontierLLM, RMSNorm, SwiGLU, RotaryEmbedding


def test_config_instantiation():
    """Les dataclasses de config s'instancient sans erreur."""
    mc = ModelConfig()
    tc = TrainConfig()
    assert mc.vocab_size == 16000
    assert mc.n_embd % mc.n_head == 0, "n_embd doit être divisible par n_head"
    assert tc.learning_rate > tc.min_lr


def test_model_forward_shapes():
    """Le forward pass produit les bonnes shapes et une loss finie."""
    cfg = ModelConfig(vocab_size=256, block_size=32, n_layer=2, n_head=2, n_embd=64, hidden_dim=128)
    model = MiniFrontierLLM(cfg)

    B, T = 2, 16
    idx = torch.randint(0, cfg.vocab_size, (B, T))
    targets = torch.randint(0, cfg.vocab_size, (B, T))

    logits, loss = model(idx, targets)
    assert logits.shape == (B, T, cfg.vocab_size), f"shape logits inattendue : {logits.shape}"
    assert loss.ndim == 0 and torch.isfinite(loss), "la loss doit être un scalaire fini"

    # Loss initiale ~ ln(vocab) pour un modèle non entraîné
    import math
    expected = math.log(cfg.vocab_size)
    assert abs(loss.item() - expected) < 1.0, f"loss initiale {loss.item():.2f} loin de ln(vocab)={expected:.2f}"


def test_model_inference_mode():
    """En mode inférence (targets=None), seul le dernier token est projeté."""
    cfg = ModelConfig(vocab_size=256, block_size=32, n_layer=2, n_head=2, n_embd=64, hidden_dim=128)
    model = MiniFrontierLLM(cfg)
    idx = torch.randint(0, cfg.vocab_size, (1, 10))
    logits, loss = model(idx)
    assert logits.shape == (1, 1, cfg.vocab_size), "inférence : doit projeter 1 seul token"
    assert loss is None


def test_generation_runs():
    """generate() produit le bon nombre de tokens sans crash."""
    cfg = ModelConfig(vocab_size=256, block_size=32, n_layer=2, n_head=2, n_embd=64, hidden_dim=128)
    model = MiniFrontierLLM(cfg)
    model.eval()
    idx = torch.randint(0, cfg.vocab_size, (1, 5))
    out = model.generate(idx, max_new_tokens=10, temperature=0.8, top_k=20)
    assert out.shape[1] == 15, f"attendu 5+10=15 tokens, obtenu {out.shape[1]}"


def test_rmsnorm():
    """RMSNorm préserve la shape."""
    norm = RMSNorm(64)
    x = torch.randn(2, 10, 64)
    assert norm(x).shape == x.shape


def test_swiglu():
    """SwiGLU préserve la dimension d'entrée."""
    cfg = ModelConfig(n_embd=64, hidden_dim=128)
    ffn = SwiGLU(cfg)
    x = torch.randn(2, 10, 64)
    assert ffn(x).shape == x.shape


def test_rope_application():
    """RoPE préserve la shape de Q/K."""
    rope = RotaryEmbedding(head_dim=32, max_seq_len=64)
    x = torch.randn(2, 4, 16, 32)  # (B, n_head, T, head_dim)
    out = rope.apply_rotary(x, seq_len=16)
    assert out.shape == x.shape


def test_param_count_reasonable():
    """Le modèle full-size fait bien ~14-15M params."""
    model = MiniFrontierLLM(ModelConfig())
    total = sum(p.numel() for p in model.parameters())
    assert 13e6 < total < 16e6, f"param count hors plage attendue : {total/1e6:.1f}M"


def test_dtype_resolution():
    """resolve_dtype renvoie une valeur valide."""
    from config import resolve_dtype
    for forced in ["bfloat16", "float16", "float32"]:
        assert resolve_dtype(forced) == forced
    # "auto" renvoie un dtype concret selon le matériel
    assert resolve_dtype("auto") in ("bfloat16", "float16", "float32")


# ── Runner sans pytest ────────────────────────────────────────────────────────
if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"✅ {t.__name__}")
        except Exception as e:
            print(f"❌ {t.__name__}: {e}")
            failed += 1
    print(f"\n{len(tests)-failed}/{len(tests)} tests passés")
    sys.exit(1 if failed else 0)
