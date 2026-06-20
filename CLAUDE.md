# CLAUDE.md

> Contexte projet pour Claude Code. Lis ce fichier en entier avant toute modification.

## Qu'est-ce que ce projet

**MiniFrontier** — un LLM causal ~14.6M paramètres entraîné **exclusivement en arabe standard moderne (Fusha)**, depuis zéro. Architecture Llama-style (RMSNorm, RoPE, SwiGLU, Flash Attention). Objectif pédagogique et expérimental : comprendre toute la chaîne d'un LLM frontier à petite échelle, du tokenizer jusqu'à l'alignement, sur du matériel grand public (RTX 3090 locale, ou Google Colab T4).

Le modèle est **mono-langue par choix** : à 14.6M paramètres, concentrer toute la capacité sur une seule langue (l'arabe) donne de bien meilleurs résultats qu'un modèle multilingue dilué.

## État actuel

- ✅ Pré-entraînement : pipeline complet et fonctionnel, entraînement stable (val loss ~5.0 et en descente sur corpus 69M tokens).
- ✅ Tokenizer BPE 16k arabe : opérationnel.
- ✅ Pipeline de données streaming (RAM-safe).
- ✅ Génération / inférence avec Top-K, Top-P, repetition penalty.
- ✅ Génération de données synthétiques via Ollama/vLLM.
- ⬜ **À venir** : SFT (Supervised Fine-Tuning), puis Constitutional AI / RLAIF. Voir `docs/ROADMAP.md`.

## Règles d'or (NE PAS VIOLER)

1. **Le code de pré-entraînement est validé et stable. Ne le refactore pas sans raison.** `train.py`, `model.py`, `data_pipeline.py`, `config.py` ont été débuggés ligne par ligne (voir `docs/DECISIONS.md` pour l'historique des bugs résolus). Toute modification doit préserver le comportement existant.

2. **Ne réintroduis JAMAIS les bugs déjà corrigés.** Les pièges connus sont documentés dans `docs/DECISIONS.md`. Les plus critiques :
   - Ne PAS multiplier `train_loss_accum` par `grad_accum` (la loss est déjà moyennée dans la micro-boucle).
   - Ne PAS activer `enable_padding()` globalement sur le tokenizer (fait exploser la RAM lors du `prepare`).
   - Ne PAS accumuler tout le corpus tokenisé dans une liste Python (utiliser le streaming memmap).
   - GradScaler OBLIGATOIRE en float16 (T4), inutile mais inoffensif en bfloat16 (RTX 3090).
   - `--max_iters` doit resynchroniser `lr_decay_iters` sinon le LR ne décroît jamais.

3. **Précision adaptée au GPU.** Le code détecte automatiquement : bfloat16 sur Ampere+ (RTX 3090, RTX 5060 Ti, A100), float16 + GradScaler sur Turing/Volta (T4). Ne hardcode pas un dtype.

4. **uint16 pour les tokens.** Le vocab fait 16k < 65535, donc les .bin sont en uint16 (2 octets/token). Ne passe pas en int32/int64 sur disque.

5. **Pas de dépendance lourde non justifiée.** Le projet tient sur torch + tokenizers + numpy. N'ajoute pas transformers/datasets/accelerate dans le chemin d'entraînement principal (ils ne servent qu'à la génération de données synthétiques, en optionnel).

## Architecture des fichiers

| Fichier | Rôle | Toucher avec prudence ? |
|---------|------|------------------------|
| `config.py` | Hyperparamètres centralisés (ModelConfig, TrainConfig, InferConfig) + détection GPU/dtype | Oui — c'est le point d'entrée des réglages |
| `model.py` | Architecture : RMSNorm, RoPE, SwiGLU, CausalSelfAttention, MiniFrontierLLM | ⚠️ Validé, ne pas refactorer |
| `tokenizer_arabic.py` | BPE 16k + normalisation Fusha (tashkeel, Alif, Ya, Waw) | ⚠️ Padding désactivé volontairement |
| `data_pipeline.py` | Corpus .txt → memmap binaire (streaming, RAM-safe) | ⚠️ Le streaming est critique |
| `train.py` | Boucle d'entraînement : cosine LR, AMP, grad accum, checkpoint, resume | ⚠️ Cœur validé |
| `generate.py` | Inférence + mode interactif | OK à étendre |
| `generate_data.py` | Données synthétiques via Ollama/vLLM | OK à étendre |
| `fix_colab.py` | Diagnostic du freeze padding sur Colab | Utilitaire ponctuel |

## Flux de travail standard

```bash
# 0. Installation
pip install -r requirements.txt

# 1. Préparer le corpus : placer les .txt arabes dans data/raw/
#    (sources : Hindawi, Wikipedia AR, Cosmopedia, ou synthétique)

# 2. (optionnel) Générer des données synthétiques
python generate_data.py --n 10000 --backend ollama --host 192.168.1.190 --port 8085 --model qwen3:27b

# 3. Entraîner le tokenizer BPE arabe
python tokenizer_arabic.py --train --corpus_dir data/raw --vocab_size 16000

# 4. Tokeniser le corpus en binaire
python data_pipeline.py --prepare --corpus_dir data/raw

# 5. Entraîner
python train.py --max_iters 10000

# 6. Reprendre depuis un checkpoint
python train.py --resume --ckpt checkpoints/ckpt_best.pt

# 7. Générer du texte
python generate.py --interactive
python generate.py --prompt "في يوم من الأيام" --temperature 0.8
```

## Cibles matérielles

- **RTX 3090 (24 GB)** : cible principale. bfloat16 natif, ~350k tok/s, run complet < 1h.
- **RTX 5060 Ti (16 GB)** : bfloat16 natif aussi (Blackwell).
- **Google Colab T4 (16 GB)** : float16 + GradScaler, ~125k tok/s. Réduire `block_size`/`batch_size` si OOM.
- **CPU** : fonctionne pour le debug, lent. dtype float32.

## Conventions de code

- Commentaires et logs en **français** (l'auteur travaille en FR). Le code (noms de variables, fonctions) en anglais.
- Docstrings sur toute fonction publique.
- Type hints partout.
- Pas de `print()` dans le code de production : utiliser le `logging` configuré.
- Les nouveaux scripts d'étape (sft_train.py, etc.) suivent le même squelette : config dataclass → fonction principale → CLI argparse.

## Vérifications avant de livrer une modification

1. `python -c "import ast; ast.parse(open('FICHIER.py').read())"` — syntaxe OK.
2. Si tu touches `train.py` : vérifier que `--resume` fonctionne encore (charge model + optimizer + scaler).
3. Si tu touches `config.py` : vérifier que `resolve_dtype()` renvoie le bon dtype pour T4 (float16) ET RTX 3090 (bfloat16).
4. Si tu touches le tokenizer ou le pipeline : `enable_padding` ne doit JAMAIS apparaître dans `__init__`.
5. Ne jamais committer le contenu de `data/`, `checkpoints/`, `logs/` (voir `.gitignore`).

## Où trouver le reste

- `docs/ARCHITECTURE.md` — détail technique de chaque composant du modèle.
- `docs/DECISIONS.md` — journal des décisions et des bugs résolus (lis-le avant de débugger).
- `docs/ROADMAP.md` — phases SFT et Constitutional AI à venir.
- `README.md` — guide utilisateur pas-à-pas.
