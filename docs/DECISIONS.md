# Journal des Décisions Techniques

> Ce fichier documente les choix d'ingénierie et **tous les bugs résolus**.
> Lis-le avant de débugger : un problème que tu rencontres a peut-être déjà été
> diagnostiqué et corrigé ici. Ne réintroduis pas un bug listé comme résolu.

---

## Bugs résolus (par ordre chronologique)

### BUG-01 — Crash RAM lors de `data_pipeline.py --prepare`
**Symptôme :** `^C` (kill OOM) sur Colab après "Lecture de …", avant tout log de progression.
**Cause :** la v1 accumulait tous les tokens dans une liste Python (`all_ids: list[int]`) avant écriture disque. Avec 2 milliards de tokens, ça faisait ~16 GB de liste Python → kill sur Colab (12 GB).
**Correctif :** tokenisation **streaming en deux passes**. Passe 1 écrit chunk par chunk dans `corpus.tmp` (RAM = 1 chunk). Passe 2 split train/val via `np.memmap` zero-copy. Voir `tokenize_corpus()` et `_iter_chunks()`.
**Ne pas régresser :** ne jamais charger le corpus tokenisé entier en RAM.

### BUG-02 — Freeze tokenizer sur gros chunks (padding global)
**Symptôme :** `^C` à l'intérieur du premier chunk, même après le fix streaming.
**Cause :** `self._tok.enable_padding()` actif dans `__init__`. `encode_batch` paddait chaque article Wikipedia à la longueur du plus long du chunk. Un article de 8000 tokens × 50000 lignes = 400M d'entiers pour un seul chunk.
**Correctif :** padding désactivé par défaut. `enable_padding` n'existe QUE dans `encode_batch_padded()` (méthode séparée, jamais appelée par le pipeline). Ajout d'un budget caractères par chunk (`max_chars_per_chunk`) en plus du budget lignes.
**Ne pas régresser :** `enable_padding` ne doit JAMAIS apparaître dans `ArabicTokenizer.__init__`.

### BUG-03 — MFU affiché à 0.0%
**Symptôme :** la colonne MFU reste à 0.0% quel que soit le GPU.
**Cause :** `peak_tflops` codé en dur à 142e12 (RTX 3090). Sur T4 (vrai pic ~65 TFLOPs), le dénominateur trop grand écrasait le MFU à ~0.
**Correctif :** table `_GPU_PEAK_TFLOPS` indexée par `torch.cuda.get_device_name()`, fallback 50 TFLOPs. Voir `MiniFrontierLLM.estimate_mfu()`.

### BUG-04 — LR ne décroît jamais sur run court
**Symptôme :** loss qui descend puis REMONTE ; LR collé près du pic toute la durée.
**Cause :** `--max_iters 1000` ne touchait que `max_iters`, pas `lr_decay_iters` (resté à 10000). Le cosine schedule n'avait pas le temps de décroître.
**Correctif :** `--max_iters` resynchronise automatiquement `lr_decay_iters` (et `warmup_iters`, `eval_interval` si nécessaire) sauf override explicite. Voir bloc CLI dans `train.py`.
**Ne pas régresser :** garder la resynchronisation automatique.

### BUG-05 — Train loss affichée ×8 trop grande
**Symptôme :** train loss ~77 (impossible : max théorique = ln(16000) ≈ 9.68), val loss correcte à ~9.7.
**Cause :** `real_loss = train_loss_accum * grad_accum`. Mais `train_loss_accum` contenait déjà la loss moyenne (chaque micro-loss est divisée par `grad_accum` AVANT le `.item()`). Double comptage.
**Correctif :** `real_loss = train_loss_accum` tout court.

### BUG-06 — Divergence réelle de l'entraînement (LR trop haut + pas de GradScaler)
**Symptôme :** après BUG-05 corrigé, loss descend jusqu'à l'étape 250 puis remonte sans arrêt, grad_norm gonfle (1.0 → 3.3).
**Cause double :**
  1. LR pic 3e-4 trop élevé pour un batch effectif de 65k tokens (8× plus petit que le batch ~500k pour lequel 3e-4 est calibré). La divergence démarrait pile à la fin du warmup, quand le LR atteint son pic.
  2. **Aucun GradScaler en float16.** Le code avait `scaler = None` en dur. Le FP16 (T4) a une plage dynamique étroite → overflow/underflow des gradients sans scaling.
**Correctif :**
  1. LR pic abaissé à 1.5e-4, warmup allongé à 400, min_lr = peak/10.
  2. `GradScaler` activé automatiquement quand `resolve_dtype()` renvoie "float16". `scaler.scale(loss).backward()`, `scaler.unscale_()` avant le clipping, `scaler.step()` + `scaler.update()`. État du scaler sauvegardé/restauré dans les checkpoints.
**Ne pas régresser :** GradScaler indispensable en FP16. `unscale_` avant `clip_grad_norm_`.

---

## Décisions d'architecture

### Pourquoi mono-langue arabe
À 14.6M paramètres, la capacité est le facteur limitant. Un modèle multilingue dilue ses embeddings et ses circuits sur plusieurs langues. Concentrer sur l'arabe seul maximise la qualité par paramètre. (Référence : philosophie "Textbook is All You Need" + observation empirique sur les micro-modèles.)

### Pourquoi un tokenizer BPE custom et pas tiktoken
Les tokenizers occidentaux (cl100k_base de GPT-4) fragmentent l'arabe en 4-6 sous-tokens par mot. Un BPE 16k entraîné sur le corpus arabe encode un mot courant en 1-2 tokens, ce qui préserve le contexte effectif et évite l'explosion VRAM de l'attention.

### Pourquoi la normalisation Fusha agressive
Suppression des tashkeel (voyelles courtes), unification des formes Alif/Ya/Waw. À 14.6M params, la variabilité est l'ennemie : `كَتَبَ` et `كتب` doivent mapper au même token pour ne pas gaspiller le vocabulaire. Voir `normalize_arabic()`.

### Pourquoi pas de weight tying
Llama/Mistral ne partagent pas les poids entre `tok_emb` et `lm_head` (contrairement à GPT-2). On suit cette convention. Coût : ~4M params en plus, acceptable.

### Pourquoi RoPE et pas d'embeddings positionnels appris
RoPE (Rotary Position Embeddings) encode la position de façon relative, supporte l'extrapolation de contexte, et n'ajoute aucun paramètre. Cache cos/sin pré-calculé pour la vitesse.

### Pourquoi SwiGLU et pas GELU/ReLU
SwiGLU (gate multiplicative) surpasse empiriquement GELU sur les petits modèles. `hidden_dim ≈ (8/3) × n_embd`, arrondi à un multiple de 64 pour les kernels CUDA.

### Pourquoi uint16 sur disque
Vocab 16k < 65535 → uint16 (2 octets) suffit. Économise 50% d'espace disque et de bande passante I/O vs int32. Le DataLoader convertit en int64 au moment du batch (requis par PyTorch embedding).

### Pourquoi memmap et pas un Dataset HuggingFace
Pour un pré-entraînement simple sur un flux dense de tokens, `np.memmap` est zero-copy et laisse le kernel OS gérer le paging. Pas besoin du surcoût d'un Dataset/DataLoader HuggingFace.

---

## Réglages de référence validés

| Paramètre | Valeur | Justification |
|-----------|--------|---------------|
| `learning_rate` | 1.5e-4 | Sûr pour batch 65k + FP16 (voir BUG-06) |
| `min_lr` | 1.5e-5 | peak / 10 |
| `warmup_iters` | 400 | Montée douce, évite la divergence précoce |
| `micro_batch_size` | 16 | Confortable sur RTX 3090 24 GB |
| `gradient_accumulation_steps` | 8 | Batch effectif = 65 536 tokens |
| `grad_clip` | 1.0 | Standard, stabilise les petits modèles |
| `weight_decay` | 0.1 | Sur les tenseurs 2D+ uniquement |
| `block_size` | 512 | Contexte ; réduire à 256 pour debug rapide |
| `vocab_size` | 16000 | Optimal pour arabe mono-langue |

---

## Points de vigilance connus (non-bugs)

- **Double log `[config] GPU: …`** : `resolve_dtype()` est appelé deux fois (autocast + scaler). Cosmétique, sans impact.
- **Overfitting attendu** : sur un corpus de 69M tokens, un run de 10000 steps = ~9 epochs. La val loss finira par stagner pendant que la train loss descend. `ckpt_best.pt` protège (sauvegarde sur amélioration val uniquement). Le vrai correctif est plus de données (cible 500M-1B tokens).
- **`torch.compile` + T4** : warning `Not enough SMs to use max_autotune_gemm` — bénin, la compilation fonctionne quand même.

---

## Mesures empiriques (2026-06-23) — remplacent des estimations dans le papier

### MEAS-01 — Bits-per-byte (BPB) v1 : MESURÉ, pas estimé
**Symptôme :** le papier raisonnait en perplexité (dépendante du tokenizer, non comparable entre modèles) sans métrique invariante.
**Mesure :** `measure_bpb.py` → `bpb_results.json`. bytes/token = longueur UTF-8 réelle de chaque token décodé du val (pas une constante supposée) ; nats/token = CE flat masquée (spans coraniques exclus, comme à l'entraînement) ; BPB = (nats/ln2)/bytes.
**Résultat :** **BPB = 1.25 bits/byte** sur tout le val (5.07 nats/tok ÷ 5.86 bytes/tok ; médiane 6, p10/p90 = 2/10 → enveloppe 0.73–3.66, point = 1.25). Par catégorie : بلاغة 1.09, نحو/صرف 1.12, معاجم 1.21, لغة 1.24, أدب 1.36, شعر 1.63 (poésie la moins compressible : tashkīl strippé = signal perdu). → §`sec:bpb` du `.tex`.

### MEAS-02 — Plafond de données : sweep d'échelle sur corpus fixe (170M)
**Symptôme :** Chinchilla donne l'optimum compute, pas le plafond d'un corpus FIXE ; le papier projetait 3.5–4.0 nats sans ancre empirique.
**Mesure :** `scaling_sweep.py` (configs weight-TIED ~5/15/30/47M, head_dim 64, ~4 époques, recette v1, tie au runtime sans toucher `model.py`) → `scaling_results.json` + `scaling_curve.png`. **Harnais validé sur le pilote 5M** (loss 9.70→8.59, eval par catégorie + masque OK, checkpoint écrit) ; sweep complet à lancer sur RTX 3090 (infaisable sur la GTX 1650 4 Go locale).
**Résultat :** en attente du run 3090. Interprétation : si ValW descend encore à 47M → corpus pas encore limitant ; si plat / gap qui s'ouvre → mur des 170M atteint, et la taille de croisement = point « best-on-this-data ». → §`sec:ceiling` du `.tex`.
**Conséquence papier :** plancher 3.5 nats adouci en projection (pas un objectif acquis) ; l'arabe classique tashkīl-strippé peut plafonner plus haut que le MSA.
