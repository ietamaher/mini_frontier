# Makefile — MiniFrontier
# Raccourcis pour le pipeline complet. Usage : make <cible>

.PHONY: help install tokenizer prepare train resume generate stats clean lint check synth

# Variables surchargeables : make train MAX_ITERS=20000
CORPUS_DIR ?= data/raw
VOCAB_SIZE ?= 16000
MAX_ITERS  ?= 10000
CKPT       ?= checkpoints/ckpt_best.pt
PROMPT     ?= في يوم من الأيام
OLLAMA_HOST ?= 192.168.1.190
OLLAMA_PORT ?= 8085
OLLAMA_MODEL ?= qwen3:27b
N_STORIES  ?= 10000

help:  ## Affiche cette aide
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install:  ## Installe les dépendances
	pip install -r requirements.txt

synth:  ## Génère des données synthétiques arabes (Ollama local)
	python generate_data.py --n $(N_STORIES) --backend ollama \
		--host $(OLLAMA_HOST) --port $(OLLAMA_PORT) --model $(OLLAMA_MODEL) \
		--output $(CORPUS_DIR)/synth_stories_ar.txt

tokenizer:  ## Entraîne le tokenizer BPE arabe
	python tokenizer_arabic.py --train --corpus_dir $(CORPUS_DIR) --vocab_size $(VOCAB_SIZE)

prepare:  ## Tokenise le corpus en binaire (train.bin / val.bin)
	python data_pipeline.py --prepare --corpus_dir $(CORPUS_DIR)

stats:  ## Affiche les stats du corpus binaire
	python data_pipeline.py --stats

train:  ## Lance l'entraînement (MAX_ITERS=10000 par défaut)
	python train.py --max_iters $(MAX_ITERS)

resume:  ## Reprend l'entraînement depuis ckpt_best.pt
	python train.py --resume --ckpt $(CKPT) --max_iters $(MAX_ITERS)

generate:  ## Génère du texte (PROMPT="...")
	python generate.py --prompt "$(PROMPT)" --temperature 0.8

interactive:  ## Mode génération interactif (REPL)
	python generate.py --interactive

# ── Pipeline complet depuis zéro ──────────────────────────────────────────────
all: tokenizer prepare train  ## tokenizer → prepare → train (pipeline complet)

# ── Qualité de code ───────────────────────────────────────────────────────────
lint:  ## Lint avec ruff
	ruff check .

format:  ## Formate avec ruff
	ruff format .

check:  ## Vérifie la syntaxe de tous les modules
	@for f in *.py; do \
		python -c "import ast; ast.parse(open('$$f').read())" && echo "✅ $$f" || echo "❌ $$f"; \
	done

# ── Nettoyage ─────────────────────────────────────────────────────────────────
clean:  ## Supprime les caches Python
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

clean-all: clean  ## Supprime AUSSI données, checkpoints, logs (DESTRUCTIF)
	rm -rf data/*.bin data/*.tmp data/normalized checkpoints/* logs/* tokenizer/*.json
	@echo "⚠️  Données, checkpoints, logs et tokenizer supprimés."
