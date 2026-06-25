"""
MiniFrontier - Pipeline de Données (v2 — RAM-safe streaming)
=============================================================
Convertit les fichiers .txt arabes normalisés en fichiers binaires numpy
(memmap) pour un chargement ultra-rapide pendant l'entraînement.

CORRECTIF vs v1 : la v1 accumulait TOUS les tokens dans une liste Python
avant d'écrire sur disque → crash RAM sur Colab (12 GB) avec un corpus de
2 milliards de tokens (~16 GB de liste Python).

Solution v2 :
  • Passe 1 — tokenisation chunk par chunk, écriture streaming dans corpus.tmp
              (RAM max ≈ chunk_size × ~30 octets ≈ 150 MB)
  • Passe 2 — split train/val via np.memmap zero-copy (aucun chargement RAM)
  • corpus.tmp supprimé automatiquement à la fin.

Usage :
    python data_pipeline.py --prepare        # normalise + tokenise tout
    python data_pipeline.py --stats          # stats sur le corpus binaire
"""

import argparse
import csv
import logging
import math
import random
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Tuple

import numpy as np

from config import (
    CORPUS_RAW_DIR, CORPUS_BIN, DATA_DIR,
    CORPUS_MANIFEST, SHAMELA_STAGING, SHAMELA_STREAM_DIR, SHAMELA_STREAM,
    SHAMELA_SHUFFLE_SEED, val_cat_bin,
    CORPUS_MANIFEST_V2,
    model_cfg, train_cfg,
)
from tokenizer_arabic import ArabicTokenizer, normalize_arabic, normalize_file

logging.basicConfig(level=logging.INFO, format="%(levelname)s │ %(message)s")
log = logging.getLogger(__name__)

NORM_DIR  = DATA_DIR / "normalized"
TRAIN_BIN = DATA_DIR / "train.bin"
VAL_BIN   = DATA_DIR / "val.bin"
# Ḍād-v2 — chemins séparés (activés par --v2 dans le CLI, ne touchent pas v1)
TRAIN_BIN_V2 = DATA_DIR / "train_v2.bin"
VAL_BIN_V2   = DATA_DIR / "val_v2.bin"


# ── Pipeline Shamela : flux texte pondéré (depuis arabic-corpus/staging) ───────
# Construit un flux texte unique, de-vocalisé et normalisé, où chaque livre est
# répété proportionnellement à sa colonne mix_weight du manifeste. Ce flux sert
# (1) à entraîner le tokenizer BPE puis (2) à produire train.bin/val.bin via la
# machinerie de tokenisation existante (tokenize_corpus, inchangée).
_PAGE_LINE_RE = re.compile(r'^\s*\[PAGE \d+\]\s*$')


def _scan_shamela_clean() -> dict:
    """Associe book_id → chemin du *_clean.txt présent dans staging/."""
    index = {}
    for cat_dir in sorted(SHAMELA_STAGING.iterdir()):
        if not cat_dir.is_dir():
            continue
        for book_dir in sorted(cat_dir.iterdir()):
            if not book_dir.is_dir():
                continue
            cleans = list(book_dir.glob("*_clean.txt"))
            if not cleans:
                continue
            m = re.match(r'^book_(\d+)$', book_dir.name)
            if m:
                index[int(m.group(1))] = cleans[0]
            else:
                prov = book_dir / "PROVENANCE.md"
                if prov.exists():
                    pm = re.search(r'book_id[^\d]*(\d+)', prov.read_text(encoding='utf-8'))
                    if pm:
                        index[int(pm.group(1))] = cleans[0]
    return index


def _content_lines(path: Path):
    """Yield les lignes de contenu : sans commentaires (#) ni marqueurs [PAGE n]."""
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.rstrip('\n')
            if line.startswith('#'):
                continue
            if _PAGE_LINE_RE.match(line):
                continue
            yield line


def _stream_book(path: Path, fraction: float, out, block_chars: int = 2_000_000) -> int:
    """
    Normalise et écrit le contenu d'un livre dans le flux (par blocs, RAM-safe).
    • fraction<1.0 → ne prend que les premières (fraction × n_lignes) lignes.
    • normalize_arabic : strip tashkeel + unification alif/ya/waw (défauts).
    Retourne le nombre de caractères normalisés écrits.
    """
    if fraction < 1.0:
        total = sum(1 for _ in _content_lines(path))
        line_budget = max(1, int(total * fraction))
    else:
        line_budget = None

    buf, buf_len, written = [], 0, 0

    def _flush():
        nonlocal buf, buf_len, written
        if not buf:
            return
        norm = normalize_arabic("\n".join(buf))
        if norm:
            out.write(norm)
            out.write("\n")
            written += len(norm)
        buf, buf_len = [], 0

    for i, line in enumerate(_content_lines(path)):
        if line_budget is not None and i >= line_budget:
            break
        buf.append(line)
        buf_len += len(line)
        if buf_len >= block_chars:
            _flush()
    _flush()
    return written


def build_weighted_stream(
    manifest: Path = CORPUS_MANIFEST,
    out_path: Path = SHAMELA_STREAM,
    seed: int = SHAMELA_SHUFFLE_SEED,
    exclude_ids=None,
) -> dict:
    """
    Construit le flux texte pondéré, de-vocalisé et normalisé.

    • poids = colonne mix_weight du manifeste (répétition floor(w)× + fraction)
    • [PAGE n] retirés, normalize_arabic appliqué (tashkeel off, alif/ya/waw on)
    • mélange au niveau book-pass (graine fixe → build reproductible)

    Retourne un dict de stats (par catégorie, exclusions, taille).
    """
    manifest = Path(manifest)
    out_path = Path(out_path)
    exclude  = set(exclude_ids or [])

    index = _scan_shamela_clean()
    log.info(f"Livres clean trouvés dans staging/ : {len(index)}")

    rows = []
    with open(manifest, encoding='utf-8') as f:
        for r in csv.DictReader(f):
            bid = int(r['book_id'])
            if bid in exclude:
                continue
            w = float(r['mix_weight']) if r.get('mix_weight') else 0.0
            if w <= 0:
                continue
            path = index.get(bid)
            if path is None:
                continue
            rows.append((bid, int(r['category_id']), w, path))

    # Liste de passes (path, fraction, cat) — répétition entière + fraction
    chunks = []
    for bid, cat, w, path in rows:
        full = int(w)
        frac = w - full
        for _ in range(full):
            chunks.append((path, 1.0, cat))
        if frac > 1e-3:
            chunks.append((path, frac, cat))

    random.Random(seed).shuffle(chunks)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cat_chars = defaultdict(int)
    cat_books = defaultdict(int)
    for bid, cat, w, path in rows:
        cat_books[cat] += 1

    log.info(f"Écriture du flux pondéré → {out_path}  "
             f"({len(rows)} livres, {len(chunks)} passes, exclus={sorted(exclude)})")
    t0 = time.time()
    with open(out_path, 'w', encoding='utf-8') as out:
        for n, (path, fraction, cat) in enumerate(chunks, 1):
            cat_chars[cat] += _stream_book(path, fraction, out)
            if n % 200 == 0:
                log.info(f"  {n}/{len(chunks)} passes │ {time.time()-t0:.0f}s")

    size_mb = out_path.stat().st_size / 1e6
    log.info(f"✅ Flux écrit : {size_mb:.0f} MB en {time.time()-t0:.0f}s")
    return {
        "out_path":  out_path,
        "n_books":   len(rows),
        "n_chunks":  len(chunks),
        "size_mb":   size_mb,
        "cat_chars": dict(cat_chars),
        "cat_books": dict(cat_books),
        "excluded":  sorted(exclude),
    }


# ── Split train/val AU NIVEAU LIVRE (zéro chevauchement) ───────────────────────
# Au lieu de découper la queue du flux pondéré (fuite : des copies d'un même
# livre sur-pondéré tombent des deux côtés), on choisit ~5% des livres par
# catégorie pour la validation AVANT pondération. Les livres train sont répétés
# selon mix_weight ; les livres val sont pris une seule fois (1×). Garantit une
# val loss = signal de généralisation propre.

def _load_manifest_rows(index: dict) -> list:
    """Charge les lignes du manifeste joignables à un *_clean.txt."""
    rows = []
    with open(CORPUS_MANIFEST, encoding='utf-8') as f:
        for r in csv.DictReader(f):
            bid = int(r['book_id'])
            w   = float(r['mix_weight']) if r.get('mix_weight') else 0.0
            if w <= 0:
                continue
            path = index.get(bid)
            if path is None:
                continue
            rows.append({
                "bid":      bid,
                "cat":      int(r['category_id']),
                "cat_name": r['category_name_ar'],
                "w":        w,
                "tokens":   int(r['token_count']) if r['token_count'] else 0,
                "title":    r['title_ar'],
                "path":     path,
            })
    return rows


def select_val_books(rows: list, val_frac: float = 0.05,
                     exclude_top_frac: float = 0.10, min_tokens: int = 500) -> tuple:
    """
    Sélectionne ~val_frac des livres PAR catégorie pour la validation.

    • exclut les plus gros livres de chaque catégorie (top exclude_top_frac) :
      dictionnaires / mutūn de référence irremplaçables → gardés en train.
    • exclut les fragments (< min_tokens) : trop courts pour une val utile.
    • échantillonne uniformément sur la distribution de taille (représentatif,
      déterministe, PAS aléatoire-par-token).

    Retourne (set des book_ids val, dict stats par catégorie).
    """
    by_cat = defaultdict(list)
    for r in rows:
        by_cat[r['cat']].append(r)

    val_ids = set()
    sel = {}
    for cat in sorted(by_cat):
        books = sorted(by_cat[cat], key=lambda r: r['tokens'])
        n = len(books)

        n_top = round(n * exclude_top_frac)
        big_ids = {r['bid'] for r in books[n - n_top:]} if n_top else set()
        candidates = [r for r in books
                      if r['bid'] not in big_ids and r['tokens'] >= min_tokens]

        n_val = round(n * val_frac)
        n_val = min(n_val, len(candidates))

        chosen = []
        if n_val > 0:
            step = len(candidates) / n_val
            idxs = sorted({min(int(i * step + step / 2), len(candidates) - 1)
                           for i in range(n_val)})
            chosen = [candidates[j] for j in idxs]

        for r in chosen:
            val_ids.add(r['bid'])
        sel[cat] = {
            "cat_name":   books[0]['cat_name'],
            "n_total":    n,
            "n_val":      len(chosen),
            "n_excluded_big": len(big_ids),
            "val_books":  [(r['bid'], r['title'], r['tokens']) for r in chosen],
        }
    return val_ids, sel


def _write_chunks(out_path: Path, chunks: list, seed: int) -> dict:
    """Mélange et écrit une liste de passes (path, fraction, cat) dans un flux."""
    random.Random(seed).shuffle(chunks)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cat_chars = defaultdict(int)
    with open(out_path, 'w', encoding='utf-8') as out:
        for path, fraction, cat in chunks:
            cat_chars[cat] += _stream_book(path, fraction, out)
    return dict(cat_chars)


def encode_stream_to_bin(stream_path: Path, tok: ArabicTokenizer, out_bin: Path,
                         chunk_size: int = 20_000,
                         max_chars_per_chunk: int = 5_000_000) -> int:
    """Tokenise un flux (déjà normalisé) → un .bin uint16 unique, SANS split."""
    n_total = 0
    t0 = time.time()
    with open(out_bin, "wb") as f_out:
        for chunk_arr in _iter_chunks(Path(stream_path), tok, chunk_size, max_chars_per_chunk):
            chunk_arr.tofile(f_out)
            n_total += len(chunk_arr)
    log.info(f"  {out_bin.name} : {n_total:,} tokens "
             f"({n_total*2/1e6:.0f} MB) en {time.time()-t0:.0f}s")
    return n_total


def _count_tokens(rows: list, tok: ArabicTokenizer) -> int:
    """Compte les tokens BPE d'un ensemble de livres (1×, normalisés)."""
    total = 0
    batch = []
    for r in rows:
        for line in _content_lines(r['path']):
            norm = normalize_arabic(line)
            if norm:
                batch.append(norm)
                if len(batch) >= 2000:
                    for e in tok._tok.encode_batch(batch):
                        total += len(e.ids)
                    batch = []
    for e in tok._tok.encode_batch(batch):
        total += len(e.ids)
    return total


def prepare_shamela_split(tok: ArabicTokenizer, val_frac: float = 0.05,
                          seed: int = SHAMELA_SHUFFLE_SEED) -> dict:
    """
    Pipeline complet split-livre :
      1. sélection val (~val_frac/catégorie, gros livres protégés)
      2. flux train pondéré (mix_weight) → train.bin
      3. flux val (livres held-out, 1×) → val.bin
      4. stats par catégorie pour la val
    """
    index = _scan_shamela_clean()
    rows  = _load_manifest_rows(index)
    log.info(f"Livres joignables : {len(rows)}")

    val_ids, sel = select_val_books(rows, val_frac=val_frac)
    train_rows = [r for r in rows if r['bid'] not in val_ids]
    val_rows   = [r for r in rows if r['bid'] in val_ids]
    log.info(f"Split livre : {len(train_rows)} train / {len(val_rows)} val "
             f"(0 chevauchement)")

    # Passes train pondérées (répétition floor(w)× + fraction)
    train_chunks = []
    for r in train_rows:
        full = int(r['w'])
        frac = r['w'] - full
        for _ in range(full):
            train_chunks.append((r['path'], 1.0, r['cat']))
        if frac > 1e-3:
            train_chunks.append((r['path'], frac, r['cat']))

    # Passes val : 1× chacune (aucune répétition)
    val_chunks = [(r['path'], 1.0, r['cat']) for r in val_rows]

    train_stream = SHAMELA_STREAM_DIR / "train_stream.txt"
    val_stream   = SHAMELA_STREAM_DIR / "val_stream.txt"

    log.info(f"Écriture flux train ({len(train_chunks)} passes)…")
    _write_chunks(train_stream, train_chunks, seed)
    log.info(f"Écriture flux val ({len(val_chunks)} livres, 1×)…")
    _write_chunks(val_stream, val_chunks, seed + 1)

    log.info("Encodage → train.bin / val.bin (uint16)…")
    n_tr  = encode_stream_to_bin(train_stream, tok, TRAIN_BIN)
    n_val = encode_stream_to_bin(val_stream, tok, VAL_BIN)

    # Tokens val par catégorie (pour vérifier la représentativité)
    val_by_cat = defaultdict(list)
    for r in val_rows:
        val_by_cat[r['cat']].append(r)
    val_cat_tokens = {cat: _count_tokens(rs, tok) for cat, rs in val_by_cat.items()}

    return {
        "n_train_tokens": n_tr,
        "n_val_tokens":   n_val,
        "n_train_books":  len(train_rows),
        "n_val_books":    len(val_rows),
        "selection":      sel,
        "val_cat_tokens": dict(val_cat_tokens),
        "val_ids":        sorted(val_ids),
    }


def emit_val_category_bins(tok: ArabicTokenizer, val_frac: float = 0.05,
                           seed: int = SHAMELA_SHUFFLE_SEED) -> dict:
    """
    Émet un .bin de validation PAR catégorie (val_cat{N}.bin), à partir de la
    MÊME sélection déterministe de livres val (aucune re-sélection). Ne touche
    NI train.bin NI val.bin NI la sélection — c'est un artefact d'évaluation
    portable (à expédier avec les autres .bin sur Colab/remote).

    La somme des val_cat{N}.bin == val.bin (mêmes 45 livres, 1×).
    """
    index = _scan_shamela_clean()
    rows  = _load_manifest_rows(index)
    val_ids, _ = select_val_books(rows, val_frac=val_frac)
    val_rows = [r for r in rows if r['bid'] in val_ids]

    by_cat = defaultdict(list)
    for r in val_rows:
        by_cat[r['cat']].append(r)

    results = {}
    for cat in sorted(by_cat):
        rs = by_cat[cat]
        chunks = [(r['path'], 1.0, cat) for r in rs]
        stream = SHAMELA_STREAM_DIR / f"val_cat{cat}_stream.txt"
        _write_chunks(stream, chunks, seed=seed)
        out_bin = val_cat_bin(cat)
        n = encode_stream_to_bin(stream, tok, out_bin)
        stream.unlink(missing_ok=True)
        results[cat] = {"n_books": len(rs), "n_tokens": n, "bin": out_bin}
    return results


# ── Helpers de streaming ──────────────────────────────────────────────────────
def _iter_chunks(src: Path, tok: ArabicTokenizer, chunk_size: int,
                  max_chars_per_chunk: int = 5_000_000):
    """
    Générateur : lit src ligne par ligne, tokenise par blocs, et yield des
    numpy arrays uint16.

    Double limite par chunk :
      • chunk_size lignes (limite haute)
      • max_chars_per_chunk caractères (limite réelle de RAM)
    Un corpus avec des lignes de longueur très variable (articles Wikipedia
    entiers : 50 mots à 5000 mots) peut faire exploser un chunk borné
    uniquement en nombre de lignes. Le budget en caractères garantit une
    empreinte mémoire prévisible quel que soit le contenu.
    """
    buffer = []
    buffer_chars = 0
    with open(src, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            buffer.append(line)
            buffer_chars += len(line)
            if len(buffer) >= chunk_size or buffer_chars >= max_chars_per_chunk:
                batch = tok.encode_batch(buffer, normalize=False)
                flat  = [tid for ids in batch for tid in ids]
                yield np.array(flat, dtype=np.uint16)
                buffer.clear()
                buffer_chars = 0
    if buffer:
        batch = tok.encode_batch(buffer, normalize=False)
        flat  = [tid for ids in batch for tid in ids]
        yield np.array(flat, dtype=np.uint16)
        buffer.clear()


# ── Tokenisation streaming (RAM-safe) ────────────────────────────────────────
def tokenize_corpus(
    norm_dir: Path,
    tok: ArabicTokenizer,
    val_ratio: float = train_cfg.val_ratio,
    chunk_size: int = 20_000,            # lignes par chunk (limite haute)
    max_chars_per_chunk: int = 5_000_000, # ~5M caractères ≈ vraie limite RAM
) -> Tuple[int, int]:
    """
    Tokenise le corpus en deux passes sans jamais l'accumuler en RAM.

    Passe 1 — streaming vers corpus.tmp :
        Lit chunk par chunk (borné en lignes ET en caractères), tokenise,
        écrit directement en binaire. RAM utilisée : 1 chunk à la fois,
        plafonnée par max_chars_per_chunk indépendamment de la longueur
        des lignes du corpus.

    Passe 2 — split train / val via memmap :
        Copie les slices [0:n_train] et [n_train:] depuis le .tmp
        vers train.bin et val.bin en blocs de 64M tokens.
        RAM utilisée : 1 bloc de 128 MB à la fois.

    Returns (n_train_tokens, n_val_tokens).
    """
    txt_files = sorted(norm_dir.glob("*.txt"))
    if not txt_files:
        raise FileNotFoundError(f"Aucun fichier normalisé dans {norm_dir}")

    dtype    = np.uint16
    tmp_path = DATA_DIR / "corpus.tmp"
    n_total  = 0
    t0       = time.time()

    # ── Passe 1 ───────────────────────────────────────────────────────────────
    log.info(f"Tokenisation streaming de {len(txt_files)} fichier(s) → {tmp_path.name}")
    log.info(f"  chunk≤{chunk_size} lignes OU ≤{max_chars_per_chunk/1e6:.1f}M caractères")

    with open(tmp_path, "wb") as f_tmp:
        for src in txt_files:
            log.info(f"  Fichier : {src.name}")
            for i, chunk_arr in enumerate(_iter_chunks(src, tok, chunk_size, max_chars_per_chunk)):
                chunk_arr.tofile(f_tmp)
                n_total += len(chunk_arr)
                if (i + 1) % 20 == 0:
                    elapsed = time.time() - t0
                    rate    = n_total / max(elapsed, 1e-9) / 1e6
                    log.info(
                        f"    {n_total/1e6:>8.1f}M tokens │ "
                        f"{elapsed:>5.0f}s │ "
                        f"{rate:.1f}M tok/s"
                    )

    elapsed = time.time() - t0
    log.info(f"  Passe 1 OK : {n_total:,} tokens en {elapsed:.1f}s "
             f"({n_total*2/1e9:.2f} GB sur disque)")

    # ── Passe 2 ───────────────────────────────────────────────────────────────
    log.info("  Passe 2/2 — split train/val via memmap…")
    n_val   = int(n_total * val_ratio)
    n_train = n_total - n_val

    corpus_mm  = np.memmap(str(tmp_path), dtype=dtype, mode="r", shape=(n_total,))
    COPY_BLOCK = 64_000_000           # 64M tokens = 128 MB par opération

    log.info(f"    train.bin ← {n_train:,} tokens ({n_train*2/1e9:.2f} GB)…")
    train_mm = np.memmap(str(TRAIN_BIN), dtype=dtype, mode="w+", shape=(n_train,))
    for start in range(0, n_train, COPY_BLOCK):
        end = min(start + COPY_BLOCK, n_train)
        train_mm[start:end] = corpus_mm[start:end]
    train_mm.flush()
    del train_mm

    log.info(f"    val.bin   ← {n_val:,} tokens ({n_val*2/1e6:.0f} MB)…")
    val_mm = np.memmap(str(VAL_BIN), dtype=dtype, mode="w+", shape=(n_val,))
    for start in range(0, n_val, COPY_BLOCK):
        end = min(start + COPY_BLOCK, n_val)
        val_mm[start:end] = corpus_mm[n_train + start : n_train + end]
    val_mm.flush()
    del val_mm

    del corpus_mm
    tmp_path.unlink(missing_ok=True)

    log.info(f"✅ Corpus prêt — train: {TRAIN_BIN.stat().st_size/1e9:.2f} GB │ "
             f"val: {VAL_BIN.stat().st_size/1e6:.0f} MB")
    return n_train, n_val


# ── DataLoader rapide (memmap zero-copy) ─────────────────────────────────────
class MemmapDataset:
    """
    Chargement zero-copy depuis un fichier binaire numpy.
    np.memmap ne charge rien en RAM — le kernel OS gère le paging.

    Masquage optionnel (loss masking) : si `mask_path` pointe sur un .bin uint8
    aligné token-à-token (1 = apprendre, 0 = ignorer, ex. spans coraniques),
    les cibles Y correspondantes sont mises à -1 → ignorées par cross_entropy
    (ignore_index=-1). X (contexte) reste inchangé : le modèle LIT le span mais
    n'est pas entraîné à le GÉNÉRER. Sans mask_path → comportement identique.
    """
    def __init__(self, bin_path: Path, block_size: int, mask_path: Path = None):
        if not bin_path.exists():
            raise FileNotFoundError(
                f"Fichier binaire introuvable : {bin_path}\n"
                f"Lancez : python data_pipeline.py --prepare"
            )
        self.data       = np.memmap(str(bin_path), dtype=np.uint16, mode="r")
        self.block_size = block_size
        self.n_tokens   = len(self.data)

        self.mask = None
        if mask_path is not None and Path(mask_path).exists():
            self.mask = np.memmap(str(mask_path), dtype=np.uint8, mode="r")
            if len(self.mask) != self.n_tokens:
                raise ValueError(
                    f"Désalignement mask/tokens : {len(self.mask)} ≠ {self.n_tokens} "
                    f"({mask_path.name} vs {bin_path.name})"
                )
            n_masked = int((np.asarray(self.mask) == 0).sum())
            log.info(f"Dataset chargé : {self.n_tokens:,} tokens depuis {bin_path.name} "
                     f"│ masque actif : {n_masked:,} tokens ignorés "
                     f"({n_masked/max(self.n_tokens,1)*100:.2f}%)")
        else:
            log.info(f"Dataset chargé : {self.n_tokens:,} tokens depuis {bin_path.name}")

    def __len__(self) -> int:
        return self.n_tokens - self.block_size - 1

    def get_random_batch(self, batch_size: int):
        """Retourne (x, y) tensors shape (B, T) — transfert GPU géré en externe."""
        import torch
        ix = np.random.randint(0, len(self), size=(batch_size,))
        x  = np.stack([self.data[i     : i + self.block_size    ] for i in ix]).astype(np.int64)
        y  = np.stack([self.data[i + 1 : i + self.block_size + 1] for i in ix]).astype(np.int64)
        if self.mask is not None:
            # mask aligné sur les CIBLES (positions i+1 … i+block)
            m = np.stack([self.mask[i + 1 : i + self.block_size + 1] for i in ix])
            y[m == 0] = -1            # -1 = ignore_index dans model.forward
        return torch.from_numpy(x), torch.from_numpy(y)


def mask_for(bin_path: Path) -> Path:
    """train.bin → train_mask.bin ; val_cat31.bin → val_cat31_mask.bin."""
    p = Path(bin_path)
    return p.with_name(p.stem + "_mask.bin")


# ── Masquage des spans coraniques (loss masking) ──────────────────────────────
# Les fichiers clean/ conservent les marqueurs ﴿…﴾ ; la normalisation les efface.
# On détecte donc les spans AVANT normalisation, on tokenise par segments, et on
# émet un masque uint8 parallèle (0 = span coranique → cible ignorée à la loss).
# Seuls les versets ENTRE ﴿…﴾ sont masqués (haute précision). Le Coran cité sans
# crochets (fréquent en grammaire) n'est pas détectable et reste non masqué.
QURAN_SPAN_RE = re.compile(r'﴿([^﴾]*)﴾')

# Fix #8 — cadres de citation coranique (texte BRUT, avant normalisation, donc ى).
# Masquer ces tokens à la loss empêche le modèle d'APPRENDRE à émettre le cadre
# « قال تعالى » puis de fabriquer un faux verset (cf. guard d'inférence dans
# generate.py qui bannit « تعالى » au sampling). Activé via mask_frames=True (v2).
QURAN_FRAME_RE = re.compile(
    r'قال\s+الله\s+تعال[ىي]|قال\s+تعال[ىي]|قوله\s+تعال[ىي]|سبحانه\s+وتعال[ىي]|تعال[ىي]'
)


def _split_frames(text: str, mask_frames: bool):
    """Sous-découpe un segment NON-verset en (texte, is_masked) selon les cadres."""
    if not text:
        return []
    if not mask_frames:
        return [(text, False)]
    segs, pos = [], 0
    for m in QURAN_FRAME_RE.finditer(text):
        if m.start() > pos:
            segs.append((text[pos:m.start()], False))
        segs.append((m.group(0), True))     # cadre → masqué
        pos = m.end()
    if pos < len(text):
        segs.append((text[pos:], False))
    return segs


def _quran_segments(line: str, mask_frames: bool = False):
    """Découpe une ligne brute en segments (texte, is_masked) : versets ﴿…﴾
    (toujours) + cadres de citation (si mask_frames, fix #8)."""
    segs = []
    pos = 0
    for m in QURAN_SPAN_RE.finditer(line):
        if m.start() > pos:
            segs.extend(_split_frames(line[pos:m.start()], mask_frames))
        inner = m.group(1)
        if inner:
            segs.append((inner, True))
        pos = m.end()
    if pos < len(line):
        segs.extend(_split_frames(line[pos:], mask_frames))
    return segs


def _encode_chunks_masked(chunks: list, tok: ArabicTokenizer,
                          out_bin: Path, out_mask: Path, seed: int,
                          flush_tokens: int = 4_000_000,
                          mask_frames: bool = False) -> tuple:
    """
    Encode une liste de passes (path, fraction, cat) → (out_bin uint16,
    out_mask uint8), avec masquage des spans coraniques. RAM-safe (flush par
    blocs). Lignes sans Coran → encode_batch (rapide) ; lignes avec ﴿ → encode
    par segments (préserve l'ordre). Retourne (n_tokens, n_masked).
    """
    random.Random(seed).shuffle(chunks)
    out_bin.parent.mkdir(parents=True, exist_ok=True)

    tok_buf, mask_buf = [], []
    n_tokens = n_masked = 0
    t0 = time.time()

    with open(out_bin, "wb") as fb, open(out_mask, "wb") as fm:
        def flush_buffers():
            nonlocal tok_buf, mask_buf, n_tokens, n_masked
            if not tok_buf:
                return
            arr = np.array(tok_buf, dtype=np.uint16)
            msk = np.array(mask_buf, dtype=np.uint8)
            arr.tofile(fb)
            msk.tofile(fm)
            n_tokens += len(arr)
            n_masked += int((msk == 0).sum())
            tok_buf, mask_buf = [], []

        plain_batch = []
        def flush_plain():
            if not plain_batch:
                return
            normed = [normalize_arabic(t) for t in plain_batch]
            for e in tok._tok.encode_batch(normed):
                tok_buf.extend(e.ids)
                mask_buf.extend([1] * len(e.ids))
            plain_batch.clear()

        for n, (path, fraction, cat) in enumerate(chunks, 1):
            if fraction < 1.0:
                total = sum(1 for _ in _content_lines(path))
                budget = max(1, int(total * fraction))
            else:
                budget = None

            for i, line in enumerate(_content_lines(path)):
                if budget is not None and i >= budget:
                    break
                if '﴿' in line or (mask_frames and 'تعال' in line):
                    flush_plain()                      # préserve l'ordre
                    for seg, is_q in _quran_segments(line, mask_frames):
                        norm = normalize_arabic(seg)
                        if not norm:
                            continue
                        ids = tok._tok.encode(norm).ids
                        tok_buf.extend(ids)
                        mask_buf.extend([0 if is_q else 1] * len(ids))
                else:
                    plain_batch.append(line)
                    if len(plain_batch) >= 2000:
                        flush_plain()
                if len(tok_buf) >= flush_tokens:
                    flush_buffers()
            flush_plain()
            if n % 200 == 0:
                log.info(f"  {n}/{len(chunks)} passes │ {n_tokens/1e6:.0f}M tokens │ "
                         f"{time.time()-t0:.0f}s")
        flush_plain()
        flush_buffers()

    return n_tokens, n_masked


def prepare_shamela_split_masked(tok: ArabicTokenizer, val_frac: float = 0.05,
                                 seed: int = SHAMELA_SHUFFLE_SEED,
                                 mask_frames: bool = False) -> dict:
    """
    Comme prepare_shamela_split, mais avec MASQUAGE des spans coraniques.
    Écrit train.bin + train_mask.bin, val.bin + val_mask.bin, et les
    val_cat{N}.bin + val_cat{N}_mask.bin. Même sélection val déterministe.
    mask_frames=True (fix #8, v2) masque aussi les cadres « قال تعالى » etc.
    """
    index = _scan_shamela_clean()
    rows  = _load_manifest_rows(index)
    val_ids, sel = select_val_books(rows, val_frac=val_frac)
    train_rows = [r for r in rows if r['bid'] not in val_ids]
    val_rows   = [r for r in rows if r['bid'] in val_ids]
    log.info(f"Split livre : {len(train_rows)} train / {len(val_rows)} val (0 chevauchement)")

    train_chunks = []
    for r in train_rows:
        full = int(r['w']); frac = r['w'] - full
        for _ in range(full):
            train_chunks.append((r['path'], 1.0, r['cat']))
        if frac > 1e-3:
            train_chunks.append((r['path'], frac, r['cat']))
    val_chunks = [(r['path'], 1.0, r['cat']) for r in val_rows]

    log.info(f"Encodage MASQUÉ train ({len(train_chunks)} passes) → train.bin + train_mask.bin…")
    n_tr, n_tr_m = _encode_chunks_masked(train_chunks, tok, TRAIN_BIN, mask_for(TRAIN_BIN), seed,
                                         mask_frames=mask_frames)
    log.info(f"Encodage MASQUÉ val ({len(val_chunks)} livres) → val.bin + val_mask.bin…")
    n_va, n_va_m = _encode_chunks_masked(val_chunks, tok, VAL_BIN, mask_for(VAL_BIN), seed + 1,
                                         mask_frames=mask_frames)

    # val par catégorie (masqué aussi, pour cohérence de l'éval)
    val_by_cat = defaultdict(list)
    for r in val_rows:
        val_by_cat[r['cat']].append(r)
    cat_stats = {}
    for cat in sorted(val_by_cat):
        chunks = [(r['path'], 1.0, cat) for r in val_by_cat[cat]]
        cb = val_cat_bin(cat)
        nt, nm = _encode_chunks_masked(chunks, tok, cb, mask_for(cb), seed,
                                       mask_frames=mask_frames)
        cat_stats[cat] = {"n_books": len(val_by_cat[cat]), "n_tokens": nt, "n_masked": nm}

    return {
        "n_train_tokens": n_tr, "n_train_masked": n_tr_m,
        "n_val_tokens":   n_va, "n_val_masked":   n_va_m,
        "n_train_books":  len(train_rows), "n_val_books": len(val_rows),
        "selection":      sel,
        "val_cat_stats":  cat_stats,
        "val_ids":        sorted(val_ids),
    }


# ── Statistiques ─────────────────────────────────────────────────────────────
def print_corpus_stats():
    for name, path in [("Train", TRAIN_BIN), ("Val", VAL_BIN)]:
        if path.exists():
            n       = path.stat().st_size // 2
            size_gb = path.stat().st_size / 1e9
            log.info(f"{name} corpus : {n:>14,} tokens │ {size_gb:.2f} GB")
        else:
            log.warning(f"{name} corpus introuvable : {path}")


# ── CLI ───────────────────────────────────────────────────────────────────────
def _cli():
    parser = argparse.ArgumentParser(description="Data Pipeline v2 — MiniFrontier")
    parser.add_argument("--prepare",         action="store_true")
    parser.add_argument("--normalize_only",  action="store_true")
    parser.add_argument("--tokenize_only",   action="store_true")
    parser.add_argument("--stats",           action="store_true")
    parser.add_argument("--prepare_shamela", action="store_true",
                        help="Construire le flux texte pondéré depuis arabic-corpus/staging")
    parser.add_argument("--encode_shamela",  action="store_true",
                        help="Tokeniser le flux Shamela → train.bin/val.bin")
    parser.add_argument("--emit_val_cat_bins", action="store_true",
                        help="Émettre val_cat{N}.bin par catégorie (éval pondérée)")
    parser.add_argument("--encode_shamela_masked", action="store_true",
                        help="Encoder avec masquage des spans coraniques "
                             "(train/val/val_cat + *_mask.bin)")
    parser.add_argument("--mask_frames", action="store_true",
                        help="Fix #8 (v2) : masquer AUSSI les cadres « قال تعالى » etc., "
                             "pas seulement les versets ﴿…﴾")
    parser.add_argument("--v2", action="store_true",
                        help="Ḍād-v2 : utiliser manifeste/bins/tokenizer v2 (chemins séparés)")
    parser.add_argument("--held_out",        type=int, default=None,
                        help="book_id exclu du flux (mesure tokenizer held-out)")
    parser.add_argument("--val_book_frac",   type=float, default=0.05,
                        help="Fraction de livres PAR catégorie réservés à la val")
    parser.add_argument("--corpus_dir",      type=Path,  default=CORPUS_RAW_DIR)
    parser.add_argument("--val_ratio",       type=float, default=train_cfg.val_ratio)
    parser.add_argument("--chunk_size",      type=int,   default=20_000,
                        help="Lignes par chunk, limite haute (réduire si OOM persiste)")
    parser.add_argument("--max_chars_per_chunk", type=int, default=5_000_000,
                        help="Caractères max par chunk — vraie garantie de RAM, "
                             "réduire si OOM persiste (ex: 2000000)")
    args = parser.parse_args()

    # Ḍād-v2 : remappe les globals vers les chemins v2 (n'écrase pas v1)
    if getattr(args, "v2", False):
        global CORPUS_MANIFEST, TRAIN_BIN, VAL_BIN, val_cat_bin
        CORPUS_MANIFEST = CORPUS_MANIFEST_V2
        TRAIN_BIN = TRAIN_BIN_V2
        VAL_BIN = VAL_BIN_V2
        val_cat_bin = lambda cat: DATA_DIR / f"val_cat{cat}_v2.bin"
        log.info("🅥2 MODE V2 — manifeste/bins/tokenizer v2 (v1 intact)")

    if args.stats:
        print_corpus_stats()
        return

    if args.prepare_shamela:
        stats = build_weighted_stream(
            exclude_ids={args.held_out} if args.held_out else None,
        )
        log.info(f"Flux Shamela prêt : {stats['n_books']} livres │ "
                 f"{stats['n_chunks']} passes │ {stats['size_mb']:.0f} MB")
        return

    if args.encode_shamela:
        from config import TOKENIZER_PATH
        tok = ArabicTokenizer(TOKENIZER_PATH)
        r = prepare_shamela_split(tok, val_frac=args.val_book_frac)

        # ── Rapport final ──────────────────────────────────────────────────
        tr, va = r["n_train_tokens"], r["n_val_tokens"]
        total  = tr + va
        print("\n" + "=" * 74)
        print("SPLIT TRAIN / VAL AU NIVEAU LIVRE — zéro chevauchement")
        print("=" * 74)
        print(f"  train.bin : {tr:>14,} tokens │ {r['n_train_books']:>3} livres "
              f"(pondérés) │ {tr*2/1e6:.0f} MB")
        print(f"  val.bin   : {va:>14,} tokens │ {r['n_val_books']:>3} livres "
              f"(1×)       │ {va*2/1e6:.1f} MB")
        print(f"  val share : {va/total*100:.2f}% des tokens")

        print("\nVAL SET — répartition par catégorie")
        print(f"  {'CID':>3}  {'Catégorie':<26} {'val/tot livres':>14} "
              f"{'val tokens':>12} {'% du val':>9}")
        print("  " + "-" * 70)
        sel = r["selection"]
        vct = r["val_cat_tokens"]
        for cat in sorted(sel):
            s = sel[cat]
            vt = vct.get(cat, 0)
            share = vt / va * 100 if va else 0
            flag = "  ⚠ 0 val" if s["n_val"] == 0 else ""
            print(f"  {cat:>3}  {s['cat_name'][:26]:<26} "
                  f"{s['n_val']:>5}/{s['n_total']:<8} {vt:>12,} {share:>8.1f}%{flag}")
        print("  " + "-" * 70)
        print(f"  Graine de sélection : déterministe (taille-uniforme). "
              f"Gros livres/réfs protégés (gardés en train).")
        return

    if args.emit_val_cat_bins:
        from config import TOKENIZER_PATH, VAL_CAT_NAMES
        tok = ArabicTokenizer(TOKENIZER_PATH)
        res = emit_val_category_bins(tok, val_frac=args.val_book_frac)
        total = sum(v["n_tokens"] for v in res.values())
        print("\nVAL BINS PAR CATÉGORIE")
        print(f"  {'CID':>3}  {'Catégorie':<10} {'livres':>6} {'tokens':>12}  bin")
        for cat in sorted(res):
            v = res[cat]
            print(f"  {cat:>3}  {VAL_CAT_NAMES.get(cat,''):<10} {v['n_books']:>6} "
                  f"{v['n_tokens']:>12,}  {v['bin'].name}")
        print(f"  {'':>3}  {'TOTAL':<10} {'':>6} {total:>12,}  "
              f"(doit == val.bin : 4 231 363)")
        return

    if args.encode_shamela_masked:
        from config import TOKENIZER_PATH, TOKENIZER_PATH_V2, VAL_CAT_NAMES
        tok = ArabicTokenizer(TOKENIZER_PATH_V2 if args.v2 else TOKENIZER_PATH)
        if args.mask_frames:
            log.info("Fix #8 ACTIF — masquage des cadres « قال تعالى » en plus des versets ﴿…﴾")
        r = prepare_shamela_split_masked(tok, val_frac=args.val_book_frac,
                                         mask_frames=args.mask_frames)
        tr, trm = r["n_train_tokens"], r["n_train_masked"]
        va, vam = r["n_val_tokens"],   r["n_val_masked"]
        print("\n" + "=" * 70)
        print("ENCODAGE MASQUÉ (spans coraniques ﴿…﴾ → cible ignorée)")
        print("=" * 70)
        print(f"  train.bin : {tr:>14,} tokens │ masqués {trm:>10,} ({trm/max(tr,1)*100:.3f}%)")
        print(f"  val.bin   : {va:>14,} tokens │ masqués {vam:>10,} ({vam/max(va,1)*100:.3f}%)")
        print("\n  val par catégorie :")
        for cat in sorted(r["val_cat_stats"]):
            s = r["val_cat_stats"][cat]
            print(f"    {VAL_CAT_NAMES.get(cat,cat):<8} {s['n_tokens']:>12,} tokens │ "
                  f"masqués {s['n_masked']:>8,} ({s['n_masked']/max(s['n_tokens'],1)*100:.3f}%)")
        print("\n  → train.py charge automatiquement *_mask.bin si présents.")
        return

    if args.prepare or args.normalize_only:
        NORM_DIR.mkdir(parents=True, exist_ok=True)
        raw_files = list(args.corpus_dir.glob("*.txt"))
        if not raw_files:
            log.error(f"Aucun .txt dans {args.corpus_dir}")
            return
        log.info(f"Normalisation de {len(raw_files)} fichier(s)…")
        for src in raw_files:
            dst = NORM_DIR / src.name
            n   = normalize_file(src, dst)
            log.info(f"  {src.name} → {n:,} lignes propres")

    if args.prepare or args.tokenize_only:
        from config import TOKENIZER_PATH
        tok = ArabicTokenizer(TOKENIZER_PATH)
        n_tr, n_val = tokenize_corpus(
            NORM_DIR, tok,
            val_ratio=args.val_ratio,
            chunk_size=args.chunk_size,
            max_chars_per_chunk=args.max_chars_per_chunk,
        )
        log.info(f"Train : {n_tr:,} tokens | Val : {n_val:,} tokens")


if __name__ == "__main__":
    _cli()
