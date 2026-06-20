#!/usr/bin/env python3
"""
fix_colab.py — Diagnostic + correctif one-shot pour le freeze de data_pipeline.

Cause du freeze "^C juste après 'Fichier :'" :
  Le tokenizer a enable_padding() actif globalement. Du coup, encode_batch()
  sur un chunk de 50 000 articles Wikipedia (longueurs très variables) pad
  CHAQUE article à la longueur du plus long du chunk. Un seul article de
  8 000 tokens => 50 000 × 8 000 = 400 M d'entiers pour un seul chunk =>
  saturation RAM/CPU instantanée, AVANT le premier log de progression.

Ce script :
  1. Vérifie si le tokenizer déployé a le padding global actif.
  2. Vérifie si encode_batch pad (test réel sur 2 phrases de longueurs différentes).
  3. Affiche un verdict clair et la marche à suivre.

Usage sur Colab :
    !python fix_colab.py
"""

import sys
from pathlib import Path


def check_source_file():
    """Inspecte tokenizer_arabic.py pour un enable_padding() dans __init__."""
    p = Path("tokenizer_arabic.py")
    if not p.exists():
        print("❌ tokenizer_arabic.py introuvable dans le dossier courant.")
        return None

    lines = p.read_text(encoding="utf-8").splitlines()
    in_init = False
    bad_line = None
    for i, ln in enumerate(lines, 1):
        if "def __init__" in ln:
            in_init = True
        elif ln.strip().startswith("def "):
            in_init = False
        if in_init and "enable_padding" in ln and not ln.strip().startswith("#"):
            bad_line = (i, ln.strip())
            break

    if bad_line:
        print(f"⚠️  PADDING GLOBAL DÉTECTÉ dans __init__ (ligne {bad_line[0]}) :")
        print(f"      {bad_line[1]}")
        print("    => C'est la cause du freeze. Fichier tokenizer_arabic.py PÉRIMÉ.")
        return False
    else:
        print("✅ Aucun enable_padding() global dans __init__ (fichier source OK).")
        return True


def check_runtime_behavior():
    """Test réel : encode_batch pad-il les séquences ?"""
    try:
        from tokenizer_arabic import ArabicTokenizer
    except Exception as e:
        print(f"❌ Import impossible : {e}")
        return None

    try:
        tok = ArabicTokenizer()
    except FileNotFoundError:
        print("⚠️  Tokenizer .json pas encore entraîné — test runtime sauté.")
        print("    (Lancez d'abord : python tokenizer_arabic.py --train …)")
        return None

    # Deux phrases de longueurs très différentes
    short = "السماء زرقاء"
    long_ = "الكلب يركض في الحديقة الكبيرة الجميلة تحت أشجار النخيل العالية"
    out = tok.encode_batch([short, long_], normalize=True)

    len_short, len_long = len(out[0]), len(out[1])
    print(f"\n    Test encode_batch :")
    print(f"      phrase courte → {len_short} tokens")
    print(f"      phrase longue → {len_long} tokens")

    if len_short == len_long:
        print("    ❌ PADDING ACTIF : les deux séquences ont la même longueur.")
        print("       encode_batch pad au max du batch → freeze sur gros chunks.")
        return False
    else:
        print("    ✅ Pas de padding : longueurs réelles préservées. encode_batch OK.")
        return True


def main():
    print("═" * 60)
    print("  Diagnostic du freeze data_pipeline.py")
    print("═" * 60)

    src_ok = check_source_file()
    rt_ok  = check_runtime_behavior()

    print("\n" + "═" * 60)
    print("  VERDICT")
    print("═" * 60)

    if src_ok is False or rt_ok is False:
        print("""
  ❌ Ton tokenizer_arabic.py sur Colab est PÉRIMÉ (padding global actif).

  CORRECTIF — réuploade les fichiers à jour, puis :

    1. Vérifie :
         !grep -n enable_padding tokenizer_arabic.py
       enable_padding ne doit JAMAIS apparaître dans __init__,
       seulement dans encode_batch_padded().

    2. Relance dans l'ordre :
         !python tokenizer_arabic.py --train --corpus_dir data/raw --vocab_size 16000
         !python data_pipeline.py --prepare --corpus_dir data/raw

  Si tu ne peux pas réuploader, applique le hotfix runtime ci-dessous.
""")
    elif src_ok and rt_ok:
        print("""
  ✅ Tokenizer correct, pas de padding global.

  Si data_pipeline.py freeze encore, c'est le data_pipeline lui-même
  qui est périmé (pas de budget caractères par chunk). Réuploade-le et
  relance. Tu peux aussi réduire la taille des chunks :

    !python data_pipeline.py --prepare --corpus_dir data/raw --chunk_size 10000
""")
    else:
        print("  ⚠️  Diagnostic incomplet — entraîne d'abord le tokenizer.")


if __name__ == "__main__":
    main()
