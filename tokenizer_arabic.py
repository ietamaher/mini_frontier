"""
MiniFrontier - Tokenizer Arabe BPE
===================================
Entraîne un tokenizer BPE 16k spécialisé pour l'arabe standard moderne.
Gère la normalisation Unicode, la suppression des tashkeel, et l'unification
des formes de caractères.

Usage :
    python tokenizer_arabic.py --train --corpus_dir data/raw
    python tokenizer_arabic.py --test  "الكلب يركض في الحديقة"
"""

import re
import sys
import unicodedata
import argparse
import logging
from pathlib import Path
from typing import List

from tokenizers import Tokenizer, AddedToken
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Sequence, Whitespace, UnicodeScripts
from tokenizers.normalizers import (
    Sequence as NormSequence, NFD, Strip, Replace, Lowercase
)

from config import TOKENIZER_PATH, CORPUS_RAW_DIR, train_cfg

logging.basicConfig(level=logging.INFO, format="%(levelname)s │ %(message)s")
log = logging.getLogger(__name__)


# ── Tokens spéciaux ───────────────────────────────────────────────────────────
SPECIAL_TOKENS = ["[UNK]", "[BOS]", "[EOS]", "[PAD]", "[SEP]", "[MASK]"]
BOS_TOKEN = "[BOS]"
EOS_TOKEN = "[EOS]"
UNK_TOKEN = "[UNK]"
PAD_TOKEN = "[PAD]"


# ── Normalisation du texte arabe ──────────────────────────────────────────────
# Tashkeel (diacritiques / voyelles courtes) — à supprimer pour un micro-modèle
_TASHKEEL_PATTERN = re.compile(
    "[\u0610-\u061A"    # Extended Arabic
    "\u064B-\u065F"    # Harakat (Fatha, Kasra, Damma, Shadda, Sukun…)
    "\u0670"           # Superscript Alef
    "\u06D6-\u06DC"    # Quranic annotation signs
    "\u06DF-\u06E8"
    "\u06EA-\u06ED]"
)

# Unification des formes de Alif
_ALIF_MAP = str.maketrans(
    "أإآٱ",  # Alif hamza above, below, madda, wasla
    "اااا",  # → Alif simple
)

# Unification Ya / Alif Maqsura
_YA_MAP = str.maketrans(
    "ىٮ",   # Alif Maqsura + dotless Ba
    "يي",
)

# Ta Marbuta → Ha (optionnel, améliore la généralisation nominale)
_TA_MARBUTA_MAP = str.maketrans("ة", "ه")

# Waw formes alternatives
_WAW_MAP = str.maketrans("ؤ", "و")

# Supprimer les caractères non arabes sauf ponctuation minimale
_NON_ARABIC = re.compile(
    r"[^\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF"  # blocs Arabic
    r"\u0020\n\r"                                   # espace, newlines
    r"\.\،\؟\!\:\;\-\(\)\[\]\"]"                   # ponctuation utile
)

# Espaces multiples
_MULTI_SPACE = re.compile(r" {2,}")
_MULTI_NL    = re.compile(r"\n{3,}")


def normalize_arabic(text: str, *, strip_tashkeel: bool = True,
                     unify_alif: bool = True, unify_ya: bool = True,
                     unify_waw: bool = True, strip_non_arabic: bool = True) -> str:
    """
    Normalise un texte arabe pour l'entraînement.
    Retourne une chaîne propre, prête à être tokenisée.
    """
    # 1. Normalisation Unicode NFC (cohérence des composants)
    text = unicodedata.normalize("NFC", text)

    # 2. Unification des caractères variant
    if unify_alif:
        text = text.translate(_ALIF_MAP)
    if unify_ya:
        text = text.translate(_YA_MAP)
    if unify_waw:
        text = text.translate(_WAW_MAP)

    # 3. Suppression des tashkeel (voyelles courtes)
    if strip_tashkeel:
        text = _TASHKEEL_PATTERN.sub("", text)

    # 4. Suppression des caractères non-arabes (optionnel)
    if strip_non_arabic:
        text = _NON_ARABIC.sub(" ", text)

    # 5. Nettoyage des espaces parasites
    text = _MULTI_SPACE.sub(" ", text)
    text = _MULTI_NL.sub("\n\n", text)

    return text.strip()


def normalize_file(src: Path, dst: Path) -> int:
    """Normalise un fichier .txt arabe. Retourne le nombre de lignes traitées."""
    lines_out = 0
    with open(src, "r", encoding="utf-8", errors="ignore") as f_in, \
         open(dst, "w", encoding="utf-8") as f_out:
        for line in f_in:
            cleaned = normalize_arabic(line)
            if len(cleaned) > 10:           # filtre les lignes quasi-vides
                f_out.write(cleaned + "\n")
                lines_out += 1
    return lines_out


# ── Entraînement du Tokenizer BPE ─────────────────────────────────────────────
def train_tokenizer(corpus_files: List[Path], save_path: Path,
                    vocab_size: int = 16_000) -> Tokenizer:
    """
    Entraîne un tokenizer BPE sur les fichiers arabes fournis.
    Les fichiers doivent être PRÉ-NORMALISÉS (passer normalize_file d'abord).
    """
    log.info(f"Entraînement BPE — vocab_size={vocab_size}, fichiers={len(corpus_files)}")

    # Modèle BPE avec Unknown token
    tokenizer = Tokenizer(BPE(unk_token=UNK_TOKEN))

    # Pré-tokenisation : séparation sur les espaces blancs (approprié pour l'arabe)
    # UnicodeScripts évite de fusionner des tokens de scripts différents
    tokenizer.pre_tokenizer = Sequence([Whitespace()])

    # Formateur BPE
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,            # ignore les mots-racines ultra-rares
        special_tokens=SPECIAL_TOKENS,
        show_progress=True,
        initial_alphabet=list(                       # alphabet de base arabe garanti
            "ابتثجحخدذرزسشصضطظعغفقكلمنهوي"
            "ءآأإاةوى"
        ),
    )

    # Entraînement
    tokenizer.train(
        files=[str(f) for f in corpus_files],
        trainer=trainer,
    )

    # Sauvegarde
    save_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(save_path))
    log.info(f"✅ Tokenizer sauvegardé → {save_path}")
    log.info(f"   Vocab effectif : {tokenizer.get_vocab_size()} tokens")

    return tokenizer


# ── Wrapper léger pour PyTorch ────────────────────────────────────────────────
class ArabicTokenizer:
    """
    Wrapper autour du tokenizer HuggingFace pour l'intégration PyTorch.
    Expose .encode(), .decode(), et les IDs des tokens spéciaux.
    """
    def __init__(self, tokenizer_path: Path = TOKENIZER_PATH):
        if not tokenizer_path.exists():
            raise FileNotFoundError(
                f"Tokenizer introuvable : {tokenizer_path}\n"
                f"Lancez d'abord : python tokenizer_arabic.py --train"
            )
        self._tok = Tokenizer.from_file(str(tokenizer_path))
        # NOTE : padding désactivé par défaut. enable_padding() pad TOUTES les
        # séquences d'un encode_batch() à la longueur de la plus longue ligne
        # du batch. Sur un corpus à longueur variable (articles Wikipedia
        # entiers, certains de 50 mots, d'autres de 5000), ça gonfle chaque
        # chunk de tokenisation avec des dizaines de milliers de [PAD] inutiles
        # — exactement le bug qui causait le freeze RAM/CPU lors du prepare.
        # Le pré-entraînement n'a de toute façon pas besoin de padding : on
        # concatène un flux dense de tokens, pas des batches à taille fixe.
        # Utiliser .encode_batch_padded() explicitement si un jour nécessaire
        # (ex: SFT par paires instruction/réponse).

        vocab = self._tok.get_vocab()
        self.bos_id  = vocab[BOS_TOKEN]
        self.eos_id  = vocab[EOS_TOKEN]
        self.unk_id  = vocab[UNK_TOKEN]
        self.pad_id  = vocab[PAD_TOKEN]
        self.vocab_size = self._tok.get_vocab_size()

    # ── Encodage ──────────────────────────────────────────────────────────────
    def encode(self, text: str, add_bos: bool = True, add_eos: bool = True,
               normalize: bool = True) -> List[int]:
        """Encode un texte arabe en liste d'entiers."""
        if normalize:
            text = normalize_arabic(text)
        ids = self._tok.encode(text).ids
        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]
        return ids

    def encode_batch(self, texts: List[str], normalize: bool = True) -> List[List[int]]:
        """
        Encodage en batch SANS padding — chaque séquence garde sa longueur
        réelle. C'est le mode correct pour le pré-entraînement (on concatène
        un flux dense de tokens, pas des batches alignés).
        """
        if normalize:
            texts = [normalize_arabic(t) for t in texts]
        return [e.ids for e in self._tok.encode_batch(texts)]

    def encode_batch_padded(self, texts: List[str], normalize: bool = True) -> List[List[int]]:
        """
        Encodage en batch AVEC padding à la longueur max du batch.
        À utiliser uniquement pour des tâches nécessitant des séquences de
        taille fixe (ex: SFT par paires instruction/réponse, classification).
        Active le padding seulement le temps de cet appel, puis le désactive
        pour ne pas affecter encode_batch().
        """
        if normalize:
            texts = [normalize_arabic(t) for t in texts]
        self._tok.enable_padding(pad_token=PAD_TOKEN)
        try:
            return [e.ids for e in self._tok.encode_batch(texts)]
        finally:
            self._tok.no_padding()

    # ── Décodage ──────────────────────────────────────────────────────────────
    def decode(self, ids: List[int], skip_special: bool = True) -> str:
        """Décode une liste d'entiers en texte arabe."""
        return self._tok.decode(ids, skip_special_tokens=skip_special)

    def __len__(self) -> int:
        return self.vocab_size

    def __repr__(self) -> str:
        return f"ArabicTokenizer(vocab_size={self.vocab_size}, path={TOKENIZER_PATH.name})"


# ── CLI ───────────────────────────────────────────────────────────────────────
def _cli():
    parser = argparse.ArgumentParser(description="Tokenizer Arabe BPE — MiniFrontier")
    parser.add_argument("--train",      action="store_true", help="Entraîner le tokenizer")
    parser.add_argument("--normalize",  action="store_true", help="Normaliser les fichiers du corpus")
    parser.add_argument("--test",       type=str,   default=None, help="Tester l'encodage d'une phrase")
    parser.add_argument("--corpus_dir", type=Path,  default=CORPUS_RAW_DIR)
    parser.add_argument("--vocab_size", type=int,   default=train_cfg.tokenizer_vocab_size)
    parser.add_argument("--output",     type=Path,  default=TOKENIZER_PATH)
    args = parser.parse_args()

    if args.normalize or args.train:
        raw_dir = args.corpus_dir
        norm_dir = raw_dir.parent / "normalized"
        norm_dir.mkdir(parents=True, exist_ok=True)

        txt_files = list(raw_dir.glob("*.txt"))
        if not txt_files:
            log.error(f"Aucun fichier .txt dans {raw_dir}")
            sys.exit(1)

        log.info(f"Normalisation de {len(txt_files)} fichier(s)…")
        norm_files = []
        for src in txt_files:
            dst = norm_dir / src.name
            n = normalize_file(src, dst)
            log.info(f"  {src.name} → {n} lignes")
            norm_files.append(dst)

        if args.train:
            train_tokenizer(norm_files, args.output, vocab_size=args.vocab_size)

    if args.test:
        tok = ArabicTokenizer(args.output)
        ids = tok.encode(args.test)
        decoded = tok.decode(ids)
        print(f"\nTexte original : {args.test}")
        print(f"Après normalisation : {normalize_arabic(args.test)}")
        print(f"Tokens ({len(ids)}) : {ids}")
        print(f"Décodé  : {decoded}")


if __name__ == "__main__":
    _cli()
