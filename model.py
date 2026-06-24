"""
MiniFrontier - Architecture du Modèle
=======================================
LLM Llama-style ~10M paramètres, optimisé pour l'arabe mono-langue.

Composants :
  • RMSNorm         — plus rapide et stable que LayerNorm
  • RoPE            — position encoding relatif, supporte l'extrapolation de contexte
  • SwiGLU          — FFN gated, supérieur à GELU/ReLU sur les petits modèles
  • Flash Attention — via F.scaled_dot_product_attention (PyTorch >= 2.0)
  • GQA-ready       — architecture compatible Grouped Query Attention (extension future)
"""

import math
import logging
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import ModelConfig, model_cfg

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# RMSNorm
# ─────────────────────────────────────────────────────────────────────────────
class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (Zhang & Sennrich, 2019).
    Pas de calcul de moyenne → plus rapide que LayerNorm standard.
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._norm(x.float()).type_as(x) * self.weight


# ─────────────────────────────────────────────────────────────────────────────
# Rotary Position Embeddings (RoPE)
# ─────────────────────────────────────────────────────────────────────────────
class RotaryEmbedding(nn.Module):
    """
    RoPE (Su et al., 2021) — Encodage positionnel relatif injecté dans Q et K.
    Pré-calcule les fréquences une seule fois, gère le cache pour éviter
    le recalcul à chaque forward pass.
    """
    def __init__(self, head_dim: int, base: float = 10_000.0, max_seq_len: int = 2048):
        super().__init__()
        self.head_dim = head_dim
        self.base     = base
        # Fréquences inverses : shape (head_dim // 2,)
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        # Cache cos/sin pré-calculé
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, dtype=self.inv_freq.dtype, device=self.inv_freq.device)
        freqs = torch.outer(t, self.inv_freq)                  # (T, d/2)
        emb   = torch.cat((freqs, freqs), dim=-1)              # (T, d)
        self.register_buffer("cos_cache", emb.cos(), persistent=False)
        self.register_buffer("sin_cache", emb.sin(), persistent=False)
        self._cache_len = seq_len

    def _get_cos_sin(self, seq_len: int, device: torch.device):
        if seq_len > self._cache_len:
            self._build_cache(seq_len * 2)        # recrée avec marge
        return (
            self.cos_cache[:seq_len].to(device),
            self.sin_cache[:seq_len].to(device),
        )

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        half = x.shape[-1] // 2
        x1, x2 = x[..., :half], x[..., half:]
        return torch.cat((-x2, x1), dim=-1)

    def apply_rotary(self, x: torch.Tensor, seq_len: int) -> torch.Tensor:
        """
        x : (B, n_head, T, head_dim)
        Retourne x avec RoPE appliqué.
        """
        cos, sin = self._get_cos_sin(seq_len, x.device)
        cos = cos.unsqueeze(0).unsqueeze(1)    # (1, 1, T, d)
        sin = sin.unsqueeze(0).unsqueeze(1)
        return (x * cos) + (self._rotate_half(x) * sin)


# ─────────────────────────────────────────────────────────────────────────────
# Causal Self-Attention + RoPE
# ─────────────────────────────────────────────────────────────────────────────
class CausalSelfAttention(nn.Module):
    """
    Attention causale (auto-régressive) avec :
      • Flash Attention (F.scaled_dot_product_attention, is_causal=True)
      • RoPE appliqué sur Q et K
      • Projections sans biais (standard LLaMA/Mistral)
    """
    def __init__(self, cfg: ModelConfig, rope: RotaryEmbedding):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0, "n_embd doit être divisible par n_head"

        self.n_head  = cfg.n_head
        self.n_embd  = cfg.n_embd
        self.head_dim = cfg.n_embd // cfg.n_head
        self.dropout  = cfg.dropout
        self.rope     = rope

        # Projections Q, K, V, O — sans biais
        self.wq = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.wk = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.wv = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.wo = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)

        self.attn_drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        # Projections et reshape → (B, n_head, T, head_dim)
        def _proj(w, inp):
            return w(inp).view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        q, k, v = _proj(self.wq, x), _proj(self.wk, x), _proj(self.wv, x)

        # RoPE sur Q et K uniquement (pas V)
        q = self.rope.apply_rotary(q, T)
        k = self.rope.apply_rotary(k, T)

        # Flash Attention — utilise le noyau CUDA optimisé si disponible
        # is_causal=True génère automatiquement le masque causal
        attn_dropout = self.dropout if self.training else 0.0
        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=attn_dropout,
            is_causal=True,
        )

        # Recombinaison des têtes → (B, T, C)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.wo(y)


# ─────────────────────────────────────────────────────────────────────────────
# SwiGLU Feed-Forward Network
# ─────────────────────────────────────────────────────────────────────────────
class SwiGLU(nn.Module):
    """
    SwiGLU (Shazeer, 2020) — remplace GELU/ReLU par une gate multiplicative.
    Formule : FFN(x) = W2(SiLU(W1·x) ⊙ W3·x)
    Ratio hidden_dim ≈ 8/3 × dim (standard Llama).
    """
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.w1 = nn.Linear(cfg.n_embd, cfg.hidden_dim, bias=cfg.bias)   # gate
        self.w3 = nn.Linear(cfg.n_embd, cfg.hidden_dim, bias=cfg.bias)   # up
        self.w2 = nn.Linear(cfg.hidden_dim, cfg.n_embd, bias=cfg.bias)   # down
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.w2(F.silu(self.w1(x)) * self.w3(x)))


# ─────────────────────────────────────────────────────────────────────────────
# Bloc Transformer (Llama-style)
# ─────────────────────────────────────────────────────────────────────────────
class FrontierBlock(nn.Module):
    """
    Bloc élémentaire avec :
      • Pre-normalization (RMSNorm avant l'attention et le FFN)
      • Connexions résiduelles
    """
    def __init__(self, cfg: ModelConfig, rope: RotaryEmbedding):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.n_embd)
        self.attn      = CausalSelfAttention(cfg, rope)
        self.ffn_norm  = RMSNorm(cfg.n_embd)
        self.ffn       = SwiGLU(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x))
        x = x + self.ffn(self.ffn_norm(x))
        return x


# ─────────────────────────────────────────────────────────────────────────────
# MiniFrontier LLM
# ─────────────────────────────────────────────────────────────────────────────
class MiniFrontierLLM(nn.Module):
    """
    LLM Llama-style ~10M paramètres, mono-langue arabe.

    Graphe :
        tokens → tok_emb → [FrontierBlock × n_layer] → RMSNorm → lm_head → logits
    """

    def __init__(self, cfg: ModelConfig = model_cfg):
        super().__init__()
        self.cfg = cfg

        # RoPE partagé entre tous les blocs (économise de la mémoire)
        self.rope = RotaryEmbedding(
            head_dim=cfg.n_embd // cfg.n_head,
            base=cfg.rope_base,
            max_seq_len=cfg.block_size * 2,
        )

        # Embedding des tokens
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)

        # Blocs Transformer
        self.layers = nn.ModuleList([
            FrontierBlock(cfg, self.rope) for _ in range(cfg.n_layer)
        ])

        # Normalisation finale
        self.norm_f = RMSNorm(cfg.n_embd)

        # Projection vers le vocabulaire
        # v1 : pas de weight tying (standard Llama). v2 : tying activé via
        # cfg.tie_embeddings (libère ~4.1M params, le levier #1 du scaling — voir
        # docs/RESEARCH_v1_to_v2.md §6.2). Défaut False → comportement v1 inchangé.
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)

        # Initialisation des poids
        self.apply(self._init_weights)

        # Scaling spécial pour les résidus (GPT-2 paper trick)
        # Divise l'init des projections de sortie par √(2 × n_layer)
        scale = (2 * cfg.n_layer) ** -0.5
        for name, p in self.named_parameters():
            if name.endswith("wo.weight") or name.endswith("w2.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 * scale)

        # Weight tying (après init, façon nanoGPT) — lm_head partage l'embedding
        # d'entrée. model.parameters() dé-duplique le tenseur partagé.
        if getattr(cfg, "tie_embeddings", False):
            self.lm_head.weight = self.tok_emb.weight

        self._log_params()

    def _init_weights(self, module: nn.Module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if hasattr(module, "bias") and module.bias is not None:
                nn.init.zeros_(module.bias)

    def _log_params(self):
        total  = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        log.info(f"MiniFrontierLLM │ Total: {total/1e6:.2f}M params │ Trainable: {trainable/1e6:.2f}M")

    # ── Forward ───────────────────────────────────────────────────────────────
    def forward(
        self,
        idx:     torch.Tensor,              # (B, T) — indices de tokens
        targets: torch.Tensor | None = None # (B, T) — cibles pour la loss
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, T = idx.shape
        assert T <= self.cfg.block_size, (
            f"Séquence trop longue ({T} > block_size {self.cfg.block_size})"
        )

        x = self.tok_emb(idx)               # (B, T, n_embd)

        for layer in self.layers:
            x = layer(x)

        x = self.norm_f(x)

        if targets is not None:
            # Mode entraînement — logits complets pour calculer la cross-entropy
            logits = self.lm_head(x)        # (B, T, vocab_size)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,            # pour le masquage futur (SFT)
            )
            return logits, loss
        else:
            # Mode inférence — on ne projette que le dernier token
            logits = self.lm_head(x[:, [-1], :])   # (B, 1, vocab_size)
            return logits, None

    # ── Génération ────────────────────────────────────────────────────────────
    @torch.inference_mode()
    def generate(
        self,
        idx:          torch.Tensor,    # (1, T) — contexte initial
        max_new_tokens: int = 200,
        temperature:  float = 0.8,
        top_k:        int   = 50,
        top_p:        float = 0.9,     # nucleus sampling
        repetition_penalty: float = 1.1,
        eos_id:       int | None = None,
        bad_token_ids: list[int] | None = None,   # tokens interdits (logits → -inf)
    ) -> torch.Tensor:
        """
        Génération auto-régressive avec :
          • Temperature sampling
          • Top-K filtering
          • Nucleus (top-p) filtering
          • Repetition penalty
          • Arrêt sur EOS token
          • bad_token_ids : interdiction dure de certains tokens (ex. cadres de
            citation coranique « تعالى » pour éviter la fabrication d'Écriture).
            Génération uniquement — n'affecte ni l'architecture ni l'entraînement.
        """
        for _ in range(max_new_tokens):
            # Tronquer si on dépasse le contexte max
            idx_cond = idx[:, -self.cfg.block_size:]

            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]       # (1, vocab_size) — dernier token

            # Interdiction dure (avant tout filtrage) — exclut du sampling
            if bad_token_ids:
                logits[:, bad_token_ids] = float("-inf")

            # Repetition penalty (diminue la proba des tokens déjà générés)
            if repetition_penalty != 1.0:
                for token_id in set(idx[0].tolist()):
                    logits[0, token_id] /= repetition_penalty

            # Temperature
            logits = logits / max(temperature, 1e-8)

            # Top-K
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            # Top-P (nucleus)
            if 0.0 < top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                cumprobs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                # Supprime les tokens dont la proba cumulée dépasse top_p
                remove = cumprobs - F.softmax(sorted_logits, dim=-1) > top_p
                sorted_logits[remove] = float("-inf")
                logits = torch.zeros_like(logits).scatter_(1, sorted_idx, sorted_logits)

            probs     = F.softmax(logits, dim=-1)
            idx_next  = torch.multinomial(probs, num_samples=1)  # (1, 1)

            idx = torch.cat((idx, idx_next), dim=1)

            if eos_id is not None and idx_next.item() == eos_id:
                break

        return idx

    # ── Utilitaires ───────────────────────────────────────────────────────────
    def get_num_params(self, non_embedding: bool = True) -> int:
        """Nombre de paramètres (sans les embeddings si non_embedding=True)."""
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.tok_emb.weight.numel()
        return n

    # Peak FLOPs théoriques par GPU (TFLOPs tensor cores — identique BF16/FP16
    # sur ces cartes, donc inutile de distinguer par dtype)
    _GPU_PEAK_TFLOPS = {
        "rtx 3090":    142e12,
        "rtx 5060 ti":  23.7e12,
        "a100":        312e12,
        "t4":           65e12,   # T4 : pas de coeurs BF16 natifs, FP16 only
        "v100":        125e12,
    }
    _DEFAULT_PEAK_TFLOPS = 50e12   # fallback prudent pour GPU non listé

    def estimate_mfu(self, batch_size: int, dt_seconds: float) -> float:
        """
        Estime le Model FLOPs Utilization (MFU) — un indicateur d'efficacité GPU.
        Basé sur PaLM paper : 6 × N × T flops par token.

        Détecte automatiquement le GPU via torch.cuda.get_device_name() au lieu
        de supposer une RTX 3090 — sinon le MFU affiché est faux (collé à ~0%)
        sur tout autre GPU (T4, A100, etc.) car le dénominateur est démesuré
        par rapport à la puissance réelle de la carte utilisée.
        """
        import torch as _torch

        N = self.get_num_params()
        T = self.cfg.block_size
        flops_per_token = 6 * N
        flops_per_step  = flops_per_token * batch_size * T

        peak_tflops = self._DEFAULT_PEAK_TFLOPS
        if _torch.cuda.is_available():
            gpu_name = _torch.cuda.get_device_name(0).lower()
            for key, tflops in self._GPU_PEAK_TFLOPS.items():
                if key in gpu_name:
                    peak_tflops = tflops
                    break

        mfu = flops_per_step / (dt_seconds * peak_tflops)
        return mfu
