# Roadmap — MiniFrontier

> Plan des phases à venir. Les phases 0-1 sont faites. Les suivantes sont à
> construire. Chaque nouveau script doit suivre le squelette du projet
> (config dataclass → fonction principale → CLI argparse) et charger le
> checkpoint de la phase précédente comme point de départ.

---

## ✅ Phase 0 — Tokenizer & Données (FAIT)
- Tokenizer BPE 16k arabe avec normalisation Fusha.
- Pipeline de données streaming RAM-safe.
- Génération de données synthétiques (Ollama/vLLM).

## ✅ Phase 1 — Pré-entraînement (FAIT)
- Architecture Llama-style 14.6M params.
- Boucle d'entraînement stable (cosine LR, AMP, grad accum, checkpoint, resume).
- Inférence avec sampling avancé.
- **État** : val loss ~5.0 sur 69M tokens. Stable.
- **Prochaine amélioration** : scaler le corpus à 500M-1B tokens pour un meilleur plancher de loss.

---

## ⬜ Phase 2 — SFT (Supervised Fine-Tuning)

**Objectif :** apprendre au modèle pré-entraîné à suivre des instructions (format Question → Réponse en arabe).

**Nouveau fichier : `sft_train.py`**
- Charge `checkpoints/ckpt_best.pt` (le modèle pré-entraîné) comme point de départ.
- Dataset de paires `{instruction, réponse}` en arabe. Format suggéré : JSONL avec champs `instruction` / `output`.
  - Sources : Bactrian-X (subset arabe), Aya Dataset (filtré arabe), ou génération synthétique via Qwen3 local.
- **Loss masking** : ne calculer la loss QUE sur les tokens de la réponse, pas sur le prompt. C'est pourquoi `model.forward` utilise déjà `ignore_index=-1` dans le cross_entropy — mettre les tokens du prompt à -1 dans `targets`.
- Format de template à fixer, ex :
  ```
  [BOS] السؤال: {instruction} [SEP] الجواب: {réponse} [EOS]
  ```
- LR plus bas qu'en pré-entraînement (typiquement 1e-5 à 5e-5).
- Réutiliser `encode_batch_padded()` du tokenizer (les paires ont des longueurs variables → padding nécessaire ici, contrairement au pré-entraînement).

**À NE PAS casser :** le pré-entraînement reste intact. `sft_train.py` est un fichier séparé.

---

## ⬜ Phase 3 — Constitutional AI / RLAIF

**Objectif :** aligner le modèle SFT selon des principes (constitution), en utilisant un modèle juge IA au lieu d'annotateurs humains (RLAIF = RL from AI Feedback).

C'est la méthodologie d'Anthropic (Bai et al., 2022). Trois sous-étapes :

### 3a. Génération de critiques/révisions (phase SL de Constitutional AI)
**Nouveau fichier : `cai_revise.py`**
- Le modèle SFT génère des réponses à des prompts (dont certains adverses).
- Un modèle juge (Qwen3:27B local) critique chaque réponse selon une **constitution** (liste de principes textuels en arabe : honnêteté, refus du contenu nuisible, utilité…).
- Le juge réécrit la réponse pour respecter les principes.
- On fine-tune le modèle sur les réponses révisées (re-SFT).

### 3b. Modèle de récompense (Reward Model)
**Nouveau fichier : `reward_model.py`**
- Le modèle juge génère des paires de préférences `{réponse_A préférée à réponse_B}` selon la constitution.
- Entraîner un modèle de récompense (le LLM + une tête scalaire) à prédire ces préférences.

### 3c. Optimisation par RL
**Nouveau fichier : `rlaif_train.py`**
- Optimiser le modèle SFT avec **PPO** ou, plus simple et plus stable à cette échelle, **DPO** (Direct Preference Optimization — pas besoin de reward model séparé, apprend directement des paires de préférences).
- **Recommandation : commencer par DPO**, beaucoup plus simple à implémenter et à stabiliser sur un micro-modèle que PPO.

**Chaîne complète :**
```
ckpt_best.pt (pré-entraîné)
    → sft_train.py        → ckpt_sft.pt
    → cai_revise.py       → ckpt_cai_sl.pt
    → rlaif_train.py (DPO)→ ckpt_aligned.pt
```

**Fichier à créer : `constitution_ar.txt`** — les principes en arabe, un par ligne.

---

## ⬜ Phase 4 — Déploiement & Optimisation

- **Export GGUF** : convertir vers le format llama.cpp pour inférence locale optimisée (s'intègre avec le lab Ollama/vLLM existant de l'auteur).
- **Quantisation** : Q4/Q5/Q6 pour réduire l'empreinte mémoire.
- **Serveur d'inférence** : endpoint OpenAI-compatible (comme le port 8085 du lab).

---

## Idées d'extension (non prioritaires)

- **GQA (Grouped Query Attention)** : partager les têtes K/V entre plusieurs têtes Q pour réduire le cache KV en inférence. L'architecture actuelle est déjà proche, extension naturelle.
- **Context extension (YaRN)** : passer block_size de 512 à 2048+ via interpolation des fréquences RoPE.
- **Mixture of Experts** : remplacer le SwiGLU dense par des experts routés (s'aligne avec l'intérêt de l'auteur pour les gros modèles MoE).
- **Scaling du corpus** : Hindawi complet + Wikipédia AR + Cosmopedia AR + synthétique → 1B+ tokens.
