"""
download_data.py
================
Downloads ALL datasets for the Multilingual OT project and saves them as JSON
files under raw_data/:

  raw_data/
    english/
      SQuAD/     {train, validation}.json
      SNLI/      {train, validation, test}.json
      MMLU/      {test, validation, dev, auxiliary_train}.json
    alignment/
      FLORES-200/
        {dev, devtest}.json          ← config "all", 200 languages as columns
      OPUS-100/
        <lang_pair>/                 ← 99 English-centric pairs
          {train, validation, test}.json
    downstream/
      XSQuAD/
        <lang>/  {validation}.json   ← 12 languages
      XNLI/
        <lang>/  {train, validation, test}.json  ← 15 languages
      MMMLU/
        <lang>/  {test}.json         ← all available configs

Skip logic: if ALL expected split files already exist in the target directory,
the dataset/config is skipped entirely (no re-download).

Usage:
    pip install datasets tqdm
    python download_data.py
    python download_data.py --root /data/my_project/raw_data
"""

import argparse
import os
import json
from datasets import load_dataset, get_dataset_config_names
from dotenv import load_dotenv
from tqdm import tqdm

from huggingface_hub import login
load_dotenv()

hf_token = os.getenv("HF_TOKEN")
if hf_token:
    login(token=hf_token)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download all Multilingual OT datasets to a local directory."
    )
    parser.add_argument(
        "--root",
        type=str,
        default="raw_data",
        help=(
            "Root directory to save all downloaded data. "
            "Default: raw_data"
        ),
    )
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def already_downloaded(directory: str) -> bool:
    """Return True if the directory exists and contains at least one .json file."""
    if not os.path.isdir(directory):
        return False
    json_files = [f for f in os.listdir(directory) if f.endswith(".json")]
    return len(json_files) > 0


def save_dataset(ds_dict, base_dir: str):
    """Persist every split of a DatasetDict as a JSON file."""
    os.makedirs(base_dir, exist_ok=True)
    for split in ds_dict.keys():
        out_path = os.path.join(base_dir, f"{split}.json")
        records = [dict(row) for row in ds_dict[split]]
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
        tqdm.write(f"    ✔ [{split}] {len(records):,} records  →  {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 – English Datasets
# ─────────────────────────────────────────────────────────────────────────────

def download_squad(ROOT: str):
    out = os.path.join(ROOT, "english", "SQuAD")
    print("\n━━  SQuAD v2  ━━")
    if already_downloaded(out):
        print(f"  ⏭  Skipped (already downloaded): {out}")
        return
    ds = load_dataset("rajpurkar/squad_v2")
    save_dataset(ds, out)


def download_snli(ROOT: str):
    out = os.path.join(ROOT, "english", "SNLI")
    print("\n━━  SNLI  ━━")
    if already_downloaded(out):
        print(f"  ⏭  Skipped (already downloaded): {out}")
        return
    ds = load_dataset("stanfordnlp/snli")
    save_dataset(ds, out)


def download_mmlu(ROOT: str):
    out = os.path.join(ROOT, "english", "MMLU")
    print("\n━━  MMLU  ━━")
    if already_downloaded(out):
        print(f"  ⏭  Skipped (already downloaded): {out}")
        return
    ds = load_dataset("cais/mmlu", "all")
    save_dataset(ds, out)


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 – Alignment Datasets
# ─────────────────────────────────────────────────────────────────────────────

def download_flores200(ROOT: str):
    out = os.path.join(ROOT, "alignment", "FLORES-200")
    print("\n━━  FLORES-200 / FLORES+ (all languages — openlanguagedata/flores_plus)  ━━")
    if already_downloaded(out):
        print(f"  ⏭  Skipped (already downloaded): {out}")
        return
    ds = load_dataset("openlanguagedata/flores_plus")
    save_dataset(ds, out)


# Complete list of OPUS-100 English-centric language pairs (99 pairs)
OPUS100_PAIRS = [
    "af-en", "am-en", "an-en", "ar-de", "ar-en", "ar-fr", "ar-nl",
    "ar-ru", "ar-zh", "as-en", "az-en", "be-en", "bg-en", "bn-en",
    "br-en", "bs-en", "ca-en", "cs-en", "cy-en", "da-en", "de-en",
    "dz-en", "el-en", "en-eo", "en-es", "en-et", "en-eu", "en-fa",
    "en-fi", "en-fr", "en-fy", "en-ga", "en-gd", "en-gl", "en-gu",
    "en-ha", "en-he", "en-hi", "en-hr", "en-hu", "en-hy", "en-id",
    "en-ig", "en-is", "en-it", "en-ja", "en-ka", "en-kk", "en-km",
    "en-kn", "en-ko", "en-ku", "en-ky", "en-li", "en-lt", "en-lv",
    "en-mg", "en-mk", "en-ml", "en-mn", "en-mr", "en-ms", "en-mt",
    "en-my", "en-nb", "en-ne", "en-nl", "en-nn", "en-no", "en-oc",
    "en-or", "en-pa", "en-pl", "en-ps", "en-pt", "en-ro", "en-ru",
    "en-rw", "en-se", "en-sh", "en-si", "en-sk", "en-sl", "en-sq",
    "en-sr", "en-sv", "en-ta", "en-te", "en-tg", "en-th", "en-tk",
    "en-tr", "en-tt", "en-ug", "en-uk", "en-ur", "en-uz", "en-vi",
    "en-wa", "en-xh", "en-yi", "en-yo", "en-zh", "en-zu",
]


def download_opus100(ROOT: str):
    """OPUS-100: 99 English-centric language pairs."""
    base = os.path.join(ROOT, "alignment", "OPUS-100")
    print(f"\n━━  OPUS-100 ({len(OPUS100_PAIRS)} language pairs)  ━━")

    pairs_to_download = [
        p for p in OPUS100_PAIRS
        if not already_downloaded(os.path.join(base, p))
    ]
    skipped = len(OPUS100_PAIRS) - len(pairs_to_download)
    if skipped:
        tqdm.write(f"  ⏭  Skipping {skipped} already-downloaded pairs.")

    for pair in tqdm(pairs_to_download, desc="OPUS-100 pairs", unit="pair"):
        try:
            ds = load_dataset("Helsinki-NLP/opus-100", pair)
            save_dataset(ds, os.path.join(base, pair))
        except Exception as e:
            tqdm.write(f"  [WARN] OPUS-100 '{pair}': {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Downstream Tasks
# ─────────────────────────────────────────────────────────────────────────────

XQUAD_LANGS = ["ar", "de", "el", "en", "es", "hi", "ro", "ru", "th", "tr", "vi", "zh"]


def download_xquad(ROOT: str):
    base = os.path.join(ROOT, "downstream", "XSQuAD")
    print(f"\n━━  XSQuAD / XQuAD ({len(XQUAD_LANGS)} languages)  ━━")

    langs_to_download = [
        l for l in XQUAD_LANGS
        if not already_downloaded(os.path.join(base, l))
    ]
    skipped = len(XQUAD_LANGS) - len(langs_to_download)
    if skipped:
        tqdm.write(f"  ⏭  Skipping {skipped} already-downloaded languages.")

    for lang in tqdm(langs_to_download, desc="XQuAD langs", unit="lang"):
        try:
            ds = load_dataset("google/xquad", f"xquad.{lang}")
            save_dataset(ds, os.path.join(base, lang))
        except Exception as e:
            tqdm.write(f"  [WARN] XQuAD '{lang}': {e}")


XNLI_LANGS = [
    "ar", "bg", "de", "el", "en", "es", "fr",
    "hi", "ru", "sw", "th", "tr", "ur", "vi", "zh",
]


def download_xnli(ROOT: str):
    base = os.path.join(ROOT, "downstream", "XNLI")
    print(f"\n━━  XNLI ({len(XNLI_LANGS)} languages)  ━━")

    langs_to_download = [
        l for l in XNLI_LANGS
        if not already_downloaded(os.path.join(base, l))
    ]
    skipped = len(XNLI_LANGS) - len(langs_to_download)
    if skipped:
        tqdm.write(f"  ⏭  Skipping {skipped} already-downloaded languages.")

    for lang in tqdm(langs_to_download, desc="XNLI langs", unit="lang"):
        try:
            ds = load_dataset("facebook/xnli", lang)
            save_dataset(ds, os.path.join(base, lang))
        except Exception as e:
            tqdm.write(f"  [WARN] XNLI '{lang}': {e}")


def download_mmmlu(ROOT: str):
    """MMMLU: auto-discovers all available language configs at runtime."""
    base = os.path.join(ROOT, "downstream", "MMMLU")
    print("\n━━  MMMLU (all available language configs)  ━━")

    try:
        configs = get_dataset_config_names("openai/MMMLU")
        print(f"  Found {len(configs)} configs.")
    except Exception as e:
        tqdm.write(f"  [WARN] Could not fetch MMMLU config names ({e}), using fallback list.")
        configs = [
            "AR_XY", "BN_BD", "DE_DE", "ES_LA", "FR_FR",
            "HI_IN", "ID_ID", "IT_IT", "JA_JP", "KO_KR",
            "PT_BR", "SW_KE", "YO_NG", "ZH_CN", "default",
        ]

    configs_to_download = [
        c for c in configs
        if not already_downloaded(os.path.join(base, c))
    ]
    skipped = len(configs) - len(configs_to_download)
    if skipped:
        tqdm.write(f"  ⏭  Skipping {skipped} already-downloaded configs.")

    for cfg in tqdm(configs_to_download, desc="MMMLU configs", unit="cfg"):
        try:
            ds = load_dataset("openai/MMMLU", cfg)
            save_dataset(ds, os.path.join(base, cfg))
        except Exception as e:
            tqdm.write(f"  [WARN] MMMLU '{cfg}': {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = parse_args()
    ROOT = args.root

    print("=" * 60)
    print("  Multilingual OT — Dataset Download Script")
    print(f"  Output root: {os.path.abspath(ROOT)}")
    print("=" * 60)

    # ── Stage 1: English ──────────────────────────────────────────────────
    download_squad(ROOT)
    download_snli(ROOT)
    download_mmlu(ROOT)

    # ── Stage 2: Alignment ────────────────────────────────────────────────
    download_flores200(ROOT)
    download_opus100(ROOT)

    # ── Downstream Tasks ──────────────────────────────────────────────────
    download_xquad(ROOT)
    download_xnli(ROOT)
    download_mmmlu(ROOT)

    print("\n" + "=" * 60)
    print("  ✅  All downloads complete.")
    print("=" * 60)