# MiniFrontier — LLM Arabe ~14.6M Paramètres
## Architecture Llama-style • BPE 16k • RoPE • SwiGLU • Flash Attention

> 📚 **Documentation projet** : [`CLAUDE.md`](CLAUDE.md) (contexte IA) · [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) (détail technique) · [`docs/DECISIONS.md`](docs/DECISIONS.md) (journal des bugs résolus) · [`docs/ROADMAP.md`](docs/ROADMAP.md) (SFT & Constitutional AI à venir)

---

## Vue d'ensemble

Un modèle de langage causal ~10M paramètres entraîné exclusivement en **arabe standard moderne (Fusha)**.  
Architecture inspirée de LLaMA 2 / Mistral, dimensionnée pour s'entraîner sur une **RTX 3090 en moins de 2 heures**.

```
Architecture :
  Token Embedding  (vocab_size=16k, n_embd=256)
  ↓
  8× FrontierBlock :
     RMSNorm → CausalSelfAttention (RoPE, Flash Attention)
     RMSNorm → SwiGLU FFN
  ↓
  RMSNorm → LM Head (vocab_size=16k)
```

---

## Structure du projet

```
mini_frontier/
├── config.py           # Hyperparamètres centralisés (ModelConfig, TrainConfig)
├── tokenizer_arabic.py # BPE 16k arabe + normalisation Fusha
├── data_pipeline.py    # Corpus → memmap binaire (train.bin / val.bin)
├── model.py            # Architecture (RMSNorm, RoPE, SwiGLU, FlashAttn)
├── train.py            # Boucle d'entraînement production-ready
├── generate.py         # Inférence + mode interactif
├── generate_data.py    # Génération de données synthétiques (vLLM / Ollama)
├── requirements.txt
│
├── data/
│   ├── raw/            # 📂 Mettre vos fichiers .txt arabes bruts ici
│   ├── normalized/     # (auto-généré) textes normalisés
│   ├── train.bin       # (auto-généré) corpus tokenisé train
│   └── val.bin         # (auto-généré) corpus tokenisé validation
│
├── tokenizer/
│   └── arabic_bpe_16k.json   # (auto-généré) tokenizer BPE
│
├── checkpoints/
│   ├── ckpt_best.pt    # Meilleur modèle (val loss minimale)
│   └── ckpt_stepXXXXX.pt
│
└── logs/
    ├── train.log
    └── metrics.jsonl
```

---

## Pipeline complet — Étape par étape

### Étape 0 — Installation

```bash
# Créer un environnement virtuel
python -m venv .venv && source .venv/bin/activate

# PyTorch CUDA 12.1 (RTX 3090 / RTX 5060 Ti)
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Dépendances
pip install -r requirements.txt
```

---

### Étape 1 — Préparer le corpus brut arabe

Placez vos fichiers `.txt` arabes dans `data/raw/`.

**Sources recommandées :**

| Source | Volume estimé | Qualité |
|--------|--------------|---------|
| [Hindawi Books](https://www.hindawi.org/books/) | ~2 Go | ⭐⭐⭐⭐⭐ |
| [Wikipédia Arabe (filtré)](https://huggingface.co/datasets/wikipedia) | ~1 Go | ⭐⭐⭐⭐ |
| [Cosmopedia (arabe)](https://huggingface.co/datasets/HuggingFaceTB/cosmopedia) | ~500 Mo | ⭐⭐⭐⭐⭐ |
| Synthétique (generate_data.py) | illimité | ⭐⭐⭐⭐⭐ |

**Cible : 4–5 Go de texte brut ≈ 1 milliard de tokens après normalisation.**

---

### Étape 2 — Générer des données synthétiques (optionnel)

Utilisez votre Qwen3:27B local pour générer des histoires TinyStories arabes :

```bash
# Via Ollama (maktab-dev-ollama ou instance locale)
python generate_data.py \
    --n 10000 \
    --backend ollama \
    --host 192.168.1.190 \
    --port 8085 \
    --model qwen3:27b \
    --output data/raw/synth_stories_ar.txt \
    --workers 4

# Via vLLM (si servi localement)
python generate_data.py \
    --n 50000 \
    --backend vllm \
    --host 192.168.1.190 \
    --port 8085 \
    --model Qwen/Qwen2.5-7B-Instruct
```

---

### Étape 3 — Entraîner le tokenizer BPE arabe

```bash
python tokenizer_arabic.py --train --corpus_dir data/raw --vocab_size 16000
```

Ce que ça fait :
- Normalise l'arabe (supprime les tashkeel, unifie Alif/Ya/Waw)
- Entraîne un tokenizer BPE 16k sur votre corpus
- Sauvegarde dans `tokenizer/arabic_bpe_16k.json`

Test du tokenizer :
```bash
python tokenizer_arabic.py --test "الكلب يركض في الحديقة"
```

---

### Étape 4 — Préparer le corpus binaire

```bash
python data_pipeline.py --prepare --corpus_dir data/raw

# Vérifier les stats
python data_pipeline.py --stats
```

Génère `data/train.bin` et `data/val.bin` (format numpy uint16, chargement memmap).

---

### Étape 5 — Entraîner le modèle

```bash
python train.py
```

Options utiles :
```bash
# Entraînement court pour test (1000 steps)
python train.py --max_iters 1000

# Reprendre depuis un checkpoint
python train.py --resume
python train.py --resume --ckpt checkpoints/ckpt_step05000.pt

# Ajuster la taille de batch si OOM
python train.py --batch_size 8

# Sans torch.compile (debug)
python train.py --no_compile
```

**Monitoring en temps réel :**
```bash
tail -f logs/train.log
```

---

### Étape 6 — Générer du texte

```bash
# Génération simple
python generate.py --prompt "كان يا ما كان"

# Mode interactif (REPL)
python generate.py --interactive

# Paramètres de génération
python generate.py \
    --prompt "السماء" \
    --temperature 0.7 \
    --top_p 0.9 \
    --max_tokens 300
```

---

## Estimation des performances

### Sur votre setup (RTX 3090 24 GB)

| Paramètre | Valeur |
|-----------|--------|
| Tokens/step | 65 536 (16 × 8 × 512) |
| Vitesse estimée | ~350 000 tok/s |
| 10 000 steps | ~30 min |
| 1 milliard de tokens | ~48 min |
| Val loss attendue (10M, Fusha) | ~2.5–3.0 |

### Sur Google Colab T4

| Paramètre | Valeur |
|-----------|--------|
| Vitesse estimée | ~80 000 tok/s (4× plus lent) |
| 1 milliard de tokens | ~3.5–4 heures |
| Recommandation | Réduire block_size à 256, batch_size à 8 |

---

## Comparaison avec le code original

| Aspect | Code original | MiniFrontier v2 |
|--------|---------------|-----------------|
| Tokenizer | tiktoken (mauvais pour l'arabe) | BPE 16k arabe custom |
| Position encoding | RoPE (ok) | RoPE avec cache optimisé |
| Normalisation arabe | ❌ aucune | ✅ Tashkeel + Alif + Ya |
| Dataset | Shakespeare/TinyStories | Corpus arabe natif |
| Checkpointing | ❌ absent | ✅ meilleur val loss |
| Génération | Basic sampling | Top-K + Top-P + Rep penalty |
| Mixed precision | BF16 ok | BF16 + autocast propre |
| torch.compile | ❌ | ✅ (+15% vitesse) |
| Monitoring | print basique | JSON metrics + MFU |
| Reprise training | ❌ | ✅ --resume |

---

## Connexion avec votre stack

- **Ollama** : `generate_data.py` appelle directement votre instance locale (Qwen3:27B)
- **vLLM** : compatible avec le port 8085 de votre lab
- **LXD `ai-stable`** : les scripts s'exécutent dans le container sans modification
- **RTX 3090 + RTX 5060 Ti** : en ajoutant `CUDA_VISIBLE_DEVICES=0,1` pour multi-GPU (via DataParallel ou DDP)

---

## Prochaines étapes (roadmap)

1. **SFT** (Supervised Fine-Tuning) — paires Q/R arabes pour l'instruct
2. **Constitutional AI** — alignement selon vos principes
3. **GQA** — Grouped Query Attention (extension du code actuel)
4. **YaRN** — context extension > 512 tokens
5. **Quantisation GGUF** — export llama.cpp pour inférence locale
