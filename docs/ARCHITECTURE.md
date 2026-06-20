# Architecture Technique — MiniFrontier

> Référence détaillée de chaque composant. Pour les décisions (le *pourquoi*),
> voir `DECISIONS.md`. Pour l'utilisation, voir le `README.md`.

---

## Vue d'ensemble

```
                    idx (B, T) entiers de tokens
                            │
                  ┌─────────▼─────────┐
                  │  Token Embedding   │  nn.Embedding(16000, 256)
                  └─────────┬─────────┘
                            │  x (B, T, 256)
        ┌───────────────────▼───────────────────┐
        │         FrontierBlock × 8              │
        │  ┌─────────────────────────────────┐  │
        │  │ x += Attn(RMSNorm(x))   [RoPE]  │  │   ← pre-norm + résiduel
        │  │ x += SwiGLU(RMSNorm(x))         │  │
        │  └─────────────────────────────────┘  │
        └───────────────────┬───────────────────┘
                            │
                  ┌─────────▼─────────┐
                  │     RMSNorm        │  norme finale
                  └─────────┬─────────┘
                            │
                  ┌─────────▼─────────┐
                  │     LM Head        │  nn.Linear(256, 16000, bias=False)
                  └─────────┬─────────┘
                            │
                      logits (B, T, 16000)
```

**Budget paramètres (~14.6M) :**
- Token embedding : 16000 × 256 ≈ 4.1M
- 8 blocs × (4 matrices attn 256² + 3 matrices SwiGLU 256×704) ≈ 6.4M
- LM head : 256 × 16000 ≈ 4.1M
- RMSNorm + divers ≈ négligeable

---

## Composants

### RMSNorm (`model.py`)
Root Mean Square Layer Normalization. Normalise par la racine de la moyenne des carrés, **sans soustraire la moyenne** (contrairement à LayerNorm). Plus rapide, aussi stable.

```
RMSNorm(x) = x / sqrt(mean(x²) + eps) * weight
```

Calcul fait en float32 puis re-cast (stabilité numérique en AMP).

### RotaryEmbedding / RoPE (`model.py`)
Encode la position de manière **relative** en faisant tourner les vecteurs Q et K dans le plan complexe par paires de dimensions.

- Fréquences inverses pré-calculées : `1 / base^(2i/head_dim)`, base = 10000.
- Cache cos/sin construit une fois pour `max_seq_len`, étendu automatiquement si dépassement.
- Appliqué à Q et K **avant** l'attention, jamais à V.
- Partagé entre tous les blocs (une seule instance, économise la mémoire).

Avantages : aucun paramètre appris, extrapolation de contexte possible.

### CausalSelfAttention (`model.py`)
Attention multi-têtes auto-régressive.

- 8 têtes, head_dim = 256/8 = 32.
- Projections Q/K/V/O **sans biais** (convention Llama).
- RoPE appliqué à Q et K.
- **Flash Attention** via `F.scaled_dot_product_attention(..., is_causal=True)` : le masque causal est généré automatiquement, le noyau CUDA fusionné économise la VRAM (pas de matrice d'attention T×T matérialisée).

### SwiGLU (`model.py`)
Feed-forward network avec gate multiplicative.

```
SwiGLU(x) = W2( SiLU(W1·x) ⊙ W3·x )
```

- 3 matrices : W1 (gate), W3 (up), W2 (down).
- `hidden_dim = 704` ≈ (8/3) × 256, multiple de 64.
- SiLU (Swish) sur la branche gate.

### FrontierBlock (`model.py`)
Bloc Transformer Llama-style avec **pre-normalization** :

```
x = x + Attn(RMSNorm(x))      # sous-couche attention
x = x + SwiGLU(RMSNorm(x))    # sous-couche FFN
```

Les connexions résiduelles partent de `x` avant normalisation (pre-norm), ce qui stabilise l'entraînement profond.

### MiniFrontierLLM (`model.py`)
Le modèle complet.

- **Initialisation** : poids normaux std=0.02 ; les projections de sortie (`wo`, `w2`) re-scalées par `1/sqrt(2 × n_layer)` (trick GPT-2 pour stabiliser les résiduels).
- **forward(idx, targets)** : si `targets` fourni → logits complets + cross-entropy. Sinon → logits du dernier token seulement (inférence rapide).
- **generate()** : auto-régressif avec temperature, top-k, top-p (nucleus), repetition penalty, arrêt sur EOS.
- **estimate_mfu()** : Model FLOPs Utilization, table de TFLOPs par GPU.

---

## Pipeline de données (`data_pipeline.py`)

### Tokenisation streaming (RAM-safe)
1. **Passe 1** : `_iter_chunks()` lit le corpus normalisé ligne par ligne, tokenise par blocs (budget double : N lignes OU N caractères), écrit directement en binaire dans `corpus.tmp`. RAM = 1 chunk.
2. **Passe 2** : split train/val via `np.memmap` zero-copy, copie par blocs de 64M tokens. `corpus.tmp` supprimé.

Sortie : `data/train.bin` et `data/val.bin`, format uint16.

### MemmapDataset
Charge un .bin via `np.memmap` (zero-copy, paging géré par l'OS). `get_random_batch(B)` tire B fenêtres aléatoires de `block_size+1` tokens, renvoie (x, y) décalés d'une position.

---

## Tokenizer (`tokenizer_arabic.py`)

### Normalisation Fusha (`normalize_arabic`)
1. Unicode NFC.
2. Unification Alif (أإآٱ → ا), Ya (ىٮ → ي), Waw (ؤ → و).
3. Suppression des tashkeel (diacritiques U+0610-061A, U+064B-065F, etc.).
4. Suppression des caractères non-arabes (sauf ponctuation utile).
5. Nettoyage des espaces.

### BPE 16k
- `tokenizers.BPE` (backend Rust), pré-tokenizer Whitespace.
- Tokens spéciaux : `[UNK] [BOS] [EOS] [PAD] [SEP] [MASK]`.
- `min_frequency=2`, alphabet de base arabe garanti.

### ArabicTokenizer (wrapper)
`encode()` (+ BOS/EOS optionnels), `decode()`, `encode_batch()` (**sans padding**), `encode_batch_padded()` (avec padding, pour SFT futur uniquement).

---

## Boucle d'entraînement (`train.py`)

| Étape | Mécanisme |
|-------|-----------|
| LR schedule | Cosine warmup → decay → plateau min_lr (`get_lr`) |
| Précision | AMP autocast, dtype auto-détecté (bf16/fp16) |
| GradScaler | Activé en fp16 uniquement (anti-overflow) |
| Grad accumulation | `grad_accum` micro-batches → batch effectif 65k tokens |
| Grad clipping | `clip_grad_norm_` à 1.0, après `unscale_` |
| Optimizer | AdamW fused, weight decay sélectif (2D+ only), betas (0.9, 0.95) |
| Évaluation | `evaluate()` toutes les `eval_interval` steps sur la val |
| Checkpoint | Sauvegarde sur meilleure val loss (`ckpt_best.pt`) + périodique |
| Resume | Restaure model + optimizer + scaler + step |
| Monitoring | loss, grad_norm, tok/s, MFU + log JSON (`metrics.jsonl`) |

---

## Inférence (`generate.py`)

Charge un checkpoint (reconstruit `ModelConfig` depuis les métadonnées sauvegardées), encode le prompt, génère token par token. Modes : `--prompt`, `--interactive` (REPL), démo par défaut.
