"""
MiniFrontier - Générateur de Données Synthétiques Arabes
==========================================================
Génère des mini-histoires arabes (TinyStories-style) via :
  • vLLM   — si un modèle est déjà servi localement (192.168.1.190:8085)
  • Ollama  — fallback vers l'instance maktab-dev-ollama
  • OpenAI-compatible API — compatible avec tout endpoint LLM

L'objectif : créer ~500 000 histoires courtes en Fusha (arabe standard)
pour constituer la fondation du corpus d'entraînement.

Usage :
    python generate_data.py --n 1000 --output data/raw/synth_stories.txt
    python generate_data.py --n 5000 --backend ollama --model qwen3:27b
    python generate_data.py --n 10000 --backend vllm   --host 192.168.1.190 --port 8085
"""

import argparse
import logging
import random
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

from tokenizer_arabic import normalize_arabic

logging.basicConfig(level=logging.INFO, format="%(levelname)s │ %(message)s")
log = logging.getLogger(__name__)


# ── Vocabulaire de seed pour la diversité des histoires ──────────────────────
NOUNS_ARABIC = [
    "قطة", "كلب", "طفل", "بنت", "ولد", "شجرة", "بحر", "جبل", "قمر", "شمس",
    "نهر", "حصان", "سمكة", "طائر", "زهرة", "مدينة", "قرية", "بيت", "مدرسة",
    "كتاب", "قلم", "سيارة", "قطار", "سفينة", "فراشة", "نملة", "أسد", "ذئب",
    "غيمة", "مطر", "ريح", "رمل", "حجر", "نار", "ماء", "تفاحة", "خبز",
]

VERBS_ARABIC = [
    "يركض", "يلعب", "يجد", "يفكر", "يساعد", "يتعلم", "يقرأ", "يكتب",
    "يسبح", "يطير", "يمشي", "يتكلم", "يضحك", "يبكي", "يتساءل",
]

SETTINGS_ARABIC = [
    "في الغابة الكثيفة", "على شاطئ البحر", "في القرية الهادئة",
    "في المدينة الكبيرة", "على قمة الجبل", "في حديقة جميلة",
    "تحت ضوء القمر", "في يوم مشمس", "في ليلة باردة", "بجوار النهر",
]

MORALS_ARABIC = [
    "الصداقة أثمن من الذهب",
    "الصبر مفتاح الفرج",
    "العمل الجاد يؤتي ثماره",
    "الكذب لا يفيد أحدا",
    "من ساعد غيره نال المساعدة",
    "الشجاعة تفتح الأبواب",
    "التعاون يصنع المعجزات",
]


def build_prompt(noun1: str, noun2: str, verb: str, setting: str, moral: str) -> str:
    """
    Construit un prompt structuré pour générer une histoire simple en arabe.
    Le format Instruction-Following améliore la qualité et la cohérence.
    """
    return f"""اكتب قصة قصيرة جداً باللغة العربية الفصحى البسيطة لطفل في الخامسة من عمره.

المتطلبات:
- استخدم هذه الكلمات: {noun1}، {noun2}، {verb}
- المكان: {setting}
- الدرس المستفاد: {moral}
- الطول: 80 إلى 120 كلمة فقط
- الأسلوب: جمل قصيرة وبسيطة، فعل وفاعل ونتيجة واضحة
- لا تذكر العنوان، ابدأ مباشرة بالقصة

القصة:"""


# ── Backends de génération ────────────────────────────────────────────────────
class VLLMBackend:
    """Appel vers un serveur vLLM ou tout endpoint OpenAI-compatible."""
    def __init__(self, host: str = "192.168.1.190", port: int = 8085,
                 model: str = "Qwen/Qwen2.5-7B-Instruct"):
        self.url   = f"http://{host}:{port}/v1/chat/completions"
        self.model = model

    def generate(self, prompt: str, max_tokens: int = 200, temperature: float = 0.85) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "أنت كاتب قصص أطفال متمكن. تكتب بالعربية الفصحى السهلة."},
                {"role": "user",   "content": prompt},
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": 0.9,
        }
        r = requests.post(self.url, json=payload, timeout=60)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()


class OllamaBackend:
    """Appel vers Ollama (maktab-dev-ollama ou instance locale)."""
    def __init__(self, host: str = "localhost", port: int = 11434,
                 model: str = "qwen3:27b"):
        self.url   = f"http://{host}:{port}/api/generate"
        self.model = model

    def generate(self, prompt: str, max_tokens: int = 200, temperature: float = 0.85) -> str:
        payload = {
            "model":  self.model,
            "prompt": f"/no_think\n{prompt}",
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature,
                "top_p": 0.9,
                "repeat_penalty": 1.1,
            },
        }
        r = requests.post(self.url, json=payload, timeout=120)
        r.raise_for_status()
        return r.json()["response"].strip()


# ── Pipeline de génération ────────────────────────────────────────────────────
def generate_dataset(
    backend,
    n_stories: int,
    output_path: Path,
    max_workers: int = 4,
    max_tokens: int = 180,
):
    """
    Génère n_stories histoires et les sauvegarde dans output_path.
    Utilise un pool de threads pour paralléliser les requêtes API.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    generated, errors = 0, 0
    t0 = time.time()

    # Génération des paramètres de seed
    seeds = [
        (
            random.choice(NOUNS_ARABIC),
            random.choice(NOUNS_ARABIC),
            random.choice(VERBS_ARABIC),
            random.choice(SETTINGS_ARABIC),
            random.choice(MORALS_ARABIC),
        )
        for _ in range(n_stories)
    ]

    def _generate_one(seed):
        noun1, noun2, verb, setting, moral = seed
        prompt = build_prompt(noun1, noun2, verb, setting, moral)
        try:
            text = backend.generate(prompt, max_tokens=max_tokens)
            # Filtrage qualité basique
            if len(text) < 50 or len(text) > 1500:
                return None
            return normalize_arabic(text)
        except Exception as e:
            log.debug(f"Erreur génération : {e}")
            return None

    log.info(f"🚀 Génération de {n_stories} histoires ({max_workers} workers)…")

    with open(output_path, "a", encoding="utf-8") as f_out, \
         ThreadPoolExecutor(max_workers=max_workers) as executor:

        futures = {executor.submit(_generate_one, seed): seed for seed in seeds}

        for future in as_completed(futures):
            result = future.result()
            if result:
                f_out.write(result + "\n\n")
                generated += 1
            else:
                errors += 1

            # Progression toutes les 50 histoires
            if (generated + errors) % 50 == 0:
                elapsed = time.time() - t0
                rate = generated / elapsed
                eta  = (n_stories - generated - errors) / max(rate, 1e-9)
                log.info(
                    f"  ✓ {generated} | ✗ {errors} | "
                    f"{rate:.1f} hist/s | ETA {eta/60:.0f}min"
                )

    elapsed = time.time() - t0
    log.info(f"\n✅ Terminé : {generated}/{n_stories} histoires ({elapsed/60:.1f} min)")
    log.info(f"   Fichier → {output_path} ({output_path.stat().st_size/1e6:.1f} MB)")
    return generated


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Génération de données synthétiques arabes")
    parser.add_argument("--n",        type=int,  default=1000,    help="Nombre d'histoires")
    parser.add_argument("--output",   type=Path, default=Path("data/raw/synth_stories_ar.txt"))
    parser.add_argument("--backend",  choices=["vllm", "ollama"], default="ollama")
    parser.add_argument("--host",     type=str,  default="192.168.1.190")
    parser.add_argument("--port",     type=int,  default=8085)
    parser.add_argument("--model",    type=str,  default="qwen3:27b")
    parser.add_argument("--workers",  type=int,  default=4)
    parser.add_argument("--max_tokens", type=int, default=180)
    args = parser.parse_args()

    if args.backend == "vllm":
        backend = VLLMBackend(host=args.host, port=args.port, model=args.model)
        log.info(f"Backend : vLLM @ {args.host}:{args.port} ({args.model})")
    else:
        backend = OllamaBackend(host=args.host, port=args.port, model=args.model)
        log.info(f"Backend : Ollama @ {args.host}:{args.port} ({args.model})")

    generate_dataset(
        backend=backend,
        n_stories=args.n,
        output_path=args.output,
        max_workers=args.workers,
        max_tokens=args.max_tokens,
    )
