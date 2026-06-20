# MiniFrontier — LLM Arabe classique ~14.6M paramètres

**Architecture Llama-style** (RMSNorm · RoPE · SwiGLU · Flash-Attention) · **BPE 16k dé-vocalisé** · corpus **arabe classique pondéré** (المكتبة الشاملة).

> Modèle de langage causal pré-entraîné *from scratch*, **mono-langue arabe**, spécialisé sur les sciences de la langue (نحو، صرف، لغة، معاجم، بلاغة، أدب، شعر، عروض). Objectif : maîtriser toute la chaîne d'un LLM (corpus → tokenizer → pré-entraînement → évaluation) à petite échelle sur matériel grand public.
>
> 📚 [`CLAUDE.md`](CLAUDE.md) · [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) · [`docs/DECISIONS.md`](docs/DECISIONS.md) · [`docs/ROADMAP.md`](docs/ROADMAP.md)

---

## 1. Vue d'ensemble

```
tokens ─► Token Embedding (16000 × 256)
       ─► 8 × FrontierBlock :
              RMSNorm ─► CausalSelfAttention (RoPE + Flash-Attention)
              RMSNorm ─► SwiGLU FFN (256 → 704 → 256)
       ─► RMSNorm ─► LM Head (256 × 16000, sans weight-tying)
```

| Hyperparamètre | Valeur |
|---|---|
| `vocab_size` | 16 000 |
| `n_embd` (d_model) | 256 |
| `n_layer` | 8 |
| `n_head` | 8 → `head_dim` = 32 |
| `hidden_dim` (SwiGLU) | 704 ≈ 8⁄3 × 256, multiple de 64 |
| `block_size` (contexte) | 512 |
| `rope_base` | 10 000 |
| biais / weight-tying | aucun (post-LLaMA) |
| **Paramètres totaux** | **14.62 M** (10.52 M hors embedding) |

### Décompte exact des paramètres

Le total n'est pas magique, il se dérive directement de la config :

```
Embedding  tok_emb     = vocab × d            = 16000 × 256          = 4 096 000
LM head    lm_head     = d × vocab            = 256 × 16000          = 4 096 000   (non tié)

Par bloc (× 8) :
  Attention  Wq,Wk,Wv,Wo = 4 × d²            = 4 × 256²             =   262 144
  SwiGLU     W1,W3       = 2 × d × hidden     = 2 × 256 × 704        =   360 448
  SwiGLU     W2          =     hidden × d     =     704 × 256        =   180 224
  2 × RMSNorm            = 2 × d              = 2 × 256              =       512
  ───────────────────────────────────────────────────────────────────────────
  sous-total bloc                                                    =   803 328
  × 8 blocs                                                          = 6 426 624

RMSNorm final            = d                                         =       256
═══════════════════════════════════════════════════════════════════════════════
TOTAL = 4 096 000 + 4 096 000 + 6 426 624 + 256                      = 14 618 880
```

Les embeddings (entrée + sortie) pèsent **8.19 M** soit 56 % des poids — typique d'un petit modèle à vocabulaire relativement large. Le « cœur » Transformer ne fait que **10.52 M** (chiffre rapporté hors embedding).

---

## 2. Le corpus — arabe classique pondéré

Le corpus est préparé **dans un dépôt séparé** (`arabic-corpus`, extraction depuis une installation locale de المكتبة الشاملة) et consommé ici via un manifeste pondéré. Les sources sont du **texte numérique propre** (pas d'OCR), faithful-transcrites.

- **911 livres**, 7 catégories Shamela (sciences de la langue).
- Chaque livre porte un poids de mélange `mix_weight` dans `staging/corpus_manifest.csv`.

### Pondération du mélange (mix weights)

L'idée : sur-représenter le **cœur grammatical** (نحو، لغة، عروض، شعر, peu volumineux mais ciblés) et sous-représenter l'أدب/معاجم (volumineux mais hors-cible). Pour un livre de `T_raw` tokens et de poids `w` :

```
T_effectif = T_raw × w          (tokens « vus » après pondération)
part(cat)  = Σ T_eff(cat) / Σ T_eff(tous)
```

La pondération est réalisée **par répétition de documents** (pas de duplication de fichiers texte permanente) : un livre de poids `w` est inclus `⌊w⌋` fois en entier, plus une fraction `w − ⌊w⌋` de ses lignes.

| Cat | Catégorie | Poids | Tokens bruts | Tokens effectifs | Part effective |
|----:|-----------|------:|-------------:|-----------------:|---------------:|
| 31 | النحو والصرف | 2.57 | 21.5 M | 55.2 M | **45.5 %** |
| 32 | الأدب | 0.45 | 59.4 M | 26.7 M | 22.1 % |
| 30 | الغريب والمعاجم | 0.56 | 33.0 M | 18.5 M | 15.2 % |
| 29 | كتب اللغة | 2.57 | 3.3 M | 8.6 M | 7.1 % |
| 35 | البلاغة | 1.83 | 4.7 M | 8.6 M | 7.1 % |
| 34 | الشعر ودواوينه | 3.80 | 0.8 M | 3.1 M | 2.5 % |
| 33 | العروض والقوافي | 4.00 | 0.15 M | 0.58 M | 0.5 % |

> L'أدب passe de 48 % du corpus brut à 22 % effectif ; le نحو monte de 17 % à 45 %. C'est tout l'objet du mélange.

### Découpage train/val **au niveau livre** (zéro fuite)

Le split N'EST PAS une coupe en queue du flux concaténé (qui ferait fuiter des copies d'un même livre sur-pondéré des deux côtés). À la place :

1. ~5 % des **livres** de chaque catégorie sont réservés à la validation (échantillonnage déterministe sur la distribution de taille, gros dictionnaires/références **protégés** = gardés en train).
2. La pondération `mix_weight` n'est appliquée **qu'aux livres d'entraînement**.
3. Les livres de validation sont encodés **une seule fois (1×)**.

→ **0 livre commun** entre train et val : la val loss est un vrai signal de généralisation.

| | Tokens | Livres |
|---|------:|------:|
| `train.bin` (pondéré) | **175 918 892** | 866 |
| `val.bin` (1×, held-out) | **4 231 363** | 45 |

---

## 3. Le tokenizer — BPE 16k dé-vocalisé

BPE 16 000 entraîné **sur le flux pondéré** (le grammatical est donc sur-représenté dans les merges → tokenisation plus dense des termes ciblés). Normalisation appliquée *avant* l'entraînement ET à l'inférence (cohérence stricte) :

- suppression des **tashkīl** (harakāt) ;
- unification **alif** (أ إ آ ٱ → ا), **yā/alif maqṣūra** (ى → ي), **wāw hamza** (ؤ → و) ;
- retrait des caractères non-arabes hors ponctuation utile.

**Efficacité mesurée** (tokens / mot, plus bas = mieux) :

| Texte | tokens/mot | chars/token |
|---|---:|---:|
| Grammaire (held-out) | **1.42** | 3.19 |
| Prose générale (أدب) | 1.63 | — |

Les termes grammaticaux clés sont des **tokens uniques** : `منصوب`, `مرفوع`, `مجرور`, `الفاعل`, `إعراب`, `النحو`, `الصرف` → 1 token chacun.

> ⚠️ La normalisation strippe aussi les marqueurs coraniques `﴿…﴾` (bloc Presentation-Forms). Conséquence connue : le modèle apprend la *forme* des citations coraniques sans pouvoir les distinguer → voir §7 (limites) et la roadmap (masquage des spans coraniques).

---

## 4. Entraînement

### Batch effectif & volume de tokens

```
batch_effectif = micro_batch × grad_accum × block_size
               = 16 × 8 × 512 = 65 536 tokens / step

tokens_total   = batch_effectif × max_iters
               = 65 536 × 10 000 = 655 360 000 tokens  (≈ 655 M)

epochs         = tokens_total / corpus_train
               = 655.36 M / 175.92 M ≈ 3.73 époques
```

Ratio **tokens/params ≈ 655 M / 14.6 M ≈ 45** : régime « compute-rich » (au-delà de l'optimum Chinchilla ~20×), assumé pour un petit modèle — la répétition à ~3.7 époques reste dans la zone sûre.

### Learning-rate : cosine avec warmup

```
        ┌ lr_peak · t/warmup                                   si t < warmup
lr(t) = ┤ min_lr + ½(lr_peak − min_lr)(1 + cos(π·p))           sinon, p = (t−warmup)/(decay−warmup)
        └ min_lr                                                si t ≥ decay
```

| | Valeur | Pourquoi |
|---|---|---|
| `lr_peak` | 1.5e-4 | abaissé vs 3e-4 (nanoGPT) car batch 65k ≪ 500k → gradients plus bruités |
| `min_lr` | 1.5e-5 | = peak/10 |
| `warmup_iters` | 400 | montée douce, évite la divergence post-warmup |
| `weight_decay` / `grad_clip` | 0.1 / 1.0 | sur les poids 2D uniquement (pas RMSNorm/biais) |
| `betas` | (0.9, 0.95) | AdamW fused |

### Précision adaptée au GPU (automatique)

- **Ampere+** (RTX 3090/5060 Ti, A100) → **bfloat16**, pas de GradScaler.
- **Turing/Volta** (T4, V100) → **float16 + GradScaler** (plage dynamique étroite du fp16 → overflow sans scaling).
- CPU → float32.

`torch >= 2.3` requis (API `torch.amp.GradScaler`). Tokens stockés en **uint16** (vocab 16k < 65535 → 2 octets/token).

### Évaluation par catégorie + val loss pondérée

La val loss « plate » (moyenne par token) est dominée par أدب/معاجم — justement les catégories *sous*-pondérées. On rapporte donc :

1. la val loss **séparée par catégorie** (`val_cat{N}.bin`), et
2. une val loss **pondérée par le mix d'entraînement** :

```
L_pondérée = ( Σ_c  w_c · L_c ) / Z      avec  Z = Σ_c w_c
w = { نحو 0.45, أدب 0.22, معاجم 0.15, لغة 0.07, بلاغة 0.07, شعر 0.025 }
Z = 0.985   (عروض absent de la val → renormalisation)
```

Le **meilleur checkpoint** (`ckpt_best.pt`) suit `L_pondérée`, pas la moyenne plate → on optimise ce qui compte.

---

## 5. Pipeline de bout en bout

```bash
# 0. Dépendances (torch>=2.3 ; cu121 conseillé)
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# 1. Construire le flux pondéré dé-vocalisé (depuis arabic-corpus/staging)
python data_pipeline.py --prepare_shamela --held_out <book_id>

# 2. (Ré)entraîner le tokenizer BPE 16k sur ce flux
#    → tokenizer/arabic_bpe_16k.json
#    (cf. data_pipeline / tokenizer_arabic)

# 3. Encoder : split AU NIVEAU LIVRE → train.bin / val.bin (uint16)
python data_pipeline.py --encode_shamela

# 4. Émettre les bins de validation PAR catégorie
python data_pipeline.py --emit_val_cat_bins

# 5. Entraîner (config par défaut : 65 536 tok/step × 10 000)
python train.py
#    reprise :  python train.py --resume
#    Colab T4 :  réduire si OOM → --batch_size 8 --block_size 256

# 6. Générer
python generate.py --prompt "الفاعل مرفوع وعلامة رفعه" --temperature 0.7
python generate.py --interactive
```

> Le chemin du corpus (`ARABIC_CORPUS_DIR`) et tous les poids/chemins sont centralisés dans **`config.py`**. Les `.bin`, `.txt` de flux, checkpoints et tokenizer sont **git-ignorés** (volumineux / régénérables) — voir §6.

---

## 6. Matériel & artefacts

| Machine | Rôle | Précision |
|---|---|---|
| Laptop GTX 1650 (4 Go) | dev / smoke-test uniquement | float16 |
| Station RTX 3090 / 5060 Ti | entraînement réel | bfloat16 |
| Google Colab T4 | entraînement réel | float16 + GradScaler, ~120k tok/s |

Un run complet (10k steps) ≈ **30 min sur RTX 3090**, **~90 min sur T4**.

**Non versionné** (`.gitignore`) : `data/*.bin`, `data/shamela/*.txt` (~2 Go de flux), `checkpoints/`, `logs/`, `tokenizer/*.json`. Le dépôt ne contient que le **code** ; les `.bin` se régénèrent via le pipeline §5 ou se transfèrent à part (trop gros pour la limite 100 Mo de GitHub).

---

## 7. Résultats & limites (honnêtes)

**Run T4, checkpoint `ckpt_best.pt` (step 5000)** : val loss pondérée ≈ **4.81**, en descente. Le نحو est systématiquement la catégorie de plus faible loss — la pondération fonctionne. Génération : registre grammatical fluide et correct localement (« الضمة الظاهرة على آخره »), iʿrāb bien formé, citations de références plausibles (المغني، شرح الكافية).

**Limites (inhérentes à ~14.6 M params) :**

1. **Pas de raisonnement, imitation de forme.** Les phrases d'analyse sont localement bien formées mais globalement non valides. Plafond de capacité — non corrigeable par plus de pré-entraînement à cette taille (lever : modèle 50–100 M+).
2. **Effondrement répétitif à basse température** (boucles `لم يقم لم يقم…`) : augmenter `--rep_penalty` (~1.3), ou ajouter min-p / no-repeat-ngram.
3. **Citations coraniques fabriquées** (problème de *sûreté*, prioritaire) : le modèle génère du texte coranique-like avec des références inventées, car la normalisation a effacé les marqueurs `﴿…﴾`. **Correctif v2** (dans le pipeline de données) : masquage des spans coraniques dans la loss (`ignore_index`, déjà supporté par `model.py`) ou tokens spéciaux `[QURAN]…[/QURAN]`.

---

## 8. Roadmap

1. **Masquage des spans coraniques** (sûreté) — *prochaine étape*.
2. **SFT** (paires instruction/réponse arabes).
3. **Base plus grande (50–100 M)** pour une réelle compétence grammaticale.
4. Constitutional AI / RLAIF · GQA · extension de contexte (YaRN) · export GGUF.

> Note : `generate_data.py` (génération synthétique via Ollama/vLLM) reste disponible comme outil optionnel, mais n'est **pas** la source du corpus actuel (qui est Shamela).
