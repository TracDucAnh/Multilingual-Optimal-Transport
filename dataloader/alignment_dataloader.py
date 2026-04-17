"""
Define Alignment Dataset and DataLoader for handling multilingual alignment data.
Jointly process FLORES-200 and OPUS-100 datasets to create a unified alignment dataset.

OPUS-100 follows ISO 639-1 language codes, while FLORES-200 uses ISO 639-3 + ISO 15924 codes.
We map OPUS-100 language codes to FLORES-200 language codes for consistency.

Directory structure expected:
    ../raw_data/alignment/FLORES-200/dev.json
    ../raw_data/alignment/FLORES-200/devtest.json
    ../raw_data/alignment/OPUS-100/{lang_pair}/train.json
    ../raw_data/alignment/OPUS-100/{lang_pair}/test.json
    ../raw_data/alignment/OPUS-100/{lang_pair}/validation.json

Output record format:
{
    "dominant_language": "eng_Latn",   # always English
    "target_language":   "fra_Latn",
    "source_sentence":   "Hello, how are you?",
    "target_sentence":   "Bonjour, comment ça va?"
}

Eng-Eng pairs have the form:
{
    "dominant_language": "eng_Latn",
    "target_language":   "eng_Latn",   # same as dominant
    "source_sentence":   "Hello, how are you?",
    "target_sentence":   "Hello, how are you?",  # same sentence
}

Eng-Eng pair construction (per split):
    1. Start from the MIXED joint records (FLORES-200 full + OPUS-100 sampled).
    2. n_eng_eng = round(len(joint_records) * eng_eng_ratio).
    3. Sample n_eng_eng records from joint_records using random.sample(seed=42).
    4. For each sampled record, emit one eng-eng pair using source_sentence as both sides.
    5. Append eng-eng pairs AFTER joint records — order is deterministic and
       identical across all baseline runs that share the same AlignmentDataset instance.

Reproducibility guarantee:
    - AlignmentDataset is constructed ONCE and shared across all baselines.
    - get_joint() always returns the same list in the same order (joint then eng-eng).
    - AlignmentDataLoader uses the same seed for SortedLengthSampler.
    - All baselines therefore see an identical dataset and dataloader order.
"""

import json
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterator, List, Literal, Optional

import torch
from torch.utils.data import DataLoader, Dataset, Sampler
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from tqdm import tqdm

from dotenv import load_dotenv
from huggingface_hub import login

load_dotenv()
hf_token = os.getenv("HF_TOKEN")
if hf_token:
    print(f"[Auth] Logging in to Hugging Face Hub with token from .env")
    login(token=hf_token)
else:
    print("[Auth] No HF_TOKEN found in .env; proceeding without authentication. Some datasets may not be accessible.")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_LLAMA3_MODEL = "meta-llama/Meta-Llama-3-8B"  # base model, no instruct

# Seed used for ALL internal sampling operations so that every baseline run
# that constructs an AlignmentDataset with the same arguments gets the exact
# same records in the exact same order.
_GLOBAL_SAMPLE_SEED = 42

# ---------------------------------------------------------------------------
# Language-code mapping: OPUS-100 (ISO 639-1/2) → FLORES-200 (ISO 639-3_Script)
# ⚠️  Codes marked with None are absent from FLORES-200.
# ---------------------------------------------------------------------------

OPUS100_TO_FLORES200: Dict[str, Optional[str]] = {
    "af": "afr_Latn",   # Afrikaans
    "am": "amh_Ethi",   # Amharic
    "an": None,          # Aragonese  — not in FLORES-200
    "ar": "arb_Arab",   # Arabic (Modern Standard)
    "as": "asm_Beng",   # Assamese
    "az": "azj_Latn",   # Azerbaijani (North, Latin)
    "be": "bel_Cyrl",   # Belarusian
    "bg": "bul_Cyrl",   # Bulgarian
    "bn": "ben_Beng",   # Bengali
    "br": None,          # Breton — not in FLORES-200
    "bs": "bos_Latn",   # Bosnian
    "ca": "cat_Latn",   # Catalan
    "cs": "ces_Latn",   # Czech
    "cy": "cym_Latn",   # Welsh
    "da": "dan_Latn",   # Danish
    "de": "deu_Latn",   # German
    "dz": "dzo_Tibt",   # Dzongkha
    "el": "ell_Grek",   # Greek
    "en": "eng_Latn",   # English
    "eo": None,          # Esperanto — not in FLORES-200
    "es": "spa_Latn",   # Spanish
    "et": "est_Latn",   # Estonian
    "eu": "eus_Latn",   # Basque
    "fa": "pes_Arab",   # Persian (Iranian)
    "fi": "fin_Latn",   # Finnish
    "fr": "fra_Latn",   # French
    "fy": "fry_Latn",   # Western Frisian
    "ga": "gle_Latn",   # Irish
    "gd": None,          # Scottish Gaelic — not in FLORES-200
    "gl": "glg_Latn",   # Galician
    "gu": "guj_Gujr",   # Gujarati
    "ha": "hau_Latn",   # Hausa
    "he": "heb_Hebr",   # Hebrew
    "hi": "hin_Deva",   # Hindi
    "hr": "hrv_Latn",   # Croatian
    "hu": "hun_Latn",   # Hungarian
    "hy": "hye_Armn",   # Armenian
    "id": "ind_Latn",   # Indonesian
    "ig": "ibo_Latn",   # Igbo
    "is": "isl_Latn",   # Icelandic
    "it": "ita_Latn",   # Italian
    "ja": "jpn_Jpan",   # Japanese
    "ka": "kat_Geor",   # Georgian
    "kk": "kaz_Cyrl",   # Kazakh
    "km": "khm_Khmr",   # Khmer
    "kn": "kan_Knda",   # Kannada
    "ko": "kor_Hang",   # Korean
    "ku": "kmr_Latn",   # Kurdish (Kurmanji)
    "ky": "kir_Cyrl",   # Kyrgyz
    "li": None,          # Limburgish — not in FLORES-200
    "lt": "lit_Latn",   # Lithuanian
    "lv": "lvs_Latn",   # Latvian (Standard)
    "mg": "plt_Latn",   # Malagasy (Plateau)
    "mk": "mkd_Cyrl",   # Macedonian
    "ml": "mal_Mlym",   # Malayalam
    "mn": "khk_Cyrl",   # Mongolian (Halh, Cyrillic)
    "mr": "mar_Deva",   # Marathi
    "ms": "zsm_Latn",   # Malay (Standard)
    "mt": "mlt_Latn",   # Maltese
    "my": "mya_Mymr",   # Burmese
    "nb": "nob_Latn",   # Norwegian Bokmål
    "ne": "npi_Deva",   # Nepali
    "nl": "nld_Latn",   # Dutch
    "nn": None,          # Norwegian Nynorsk — not in FLORES-200
    "no": "nob_Latn",   # Norwegian (generic) → Bokmål
    "oc": "oci_Latn",   # Occitan
    "or": "ory_Orya",   # Odia (Oriya)
    "pa": "pan_Guru",   # Punjabi (Gurmukhi)
    "pl": "pol_Latn",   # Polish
    "ps": "pbt_Arab",   # Pashto (Southern)
    "pt": "por_Latn",   # Portuguese
    "ro": "ron_Latn",   # Romanian
    "ru": "rus_Cyrl",   # Russian
    "rw": "kin_Latn",   # Kinyarwanda
    "se": None,          # Northern Sami — not in FLORES-200
    "sh": "hrv_Latn",   # Serbo-Croatian → Croatian fallback
    "si": "sin_Sinh",   # Sinhala
    "sk": "slk_Latn",   # Slovak
    "sl": "slv_Latn",   # Slovenian
    "sq": "als_Latn",   # Albanian (Tosk)
    "sr": "srp_Cyrl",   # Serbian (Cyrillic)
    "sv": "swe_Latn",   # Swedish
    "ta": "tam_Taml",   # Tamil
    "te": "tel_Telu",   # Telugu
    "tg": "tgk_Cyrl",   # Tajik
    "th": "tha_Thai",   # Thai
    "tk": "tuk_Latn",   # Turkmen
    "tr": "tur_Latn",   # Turkish
    "tt": "tat_Cyrl",   # Tatar
    "ug": "uig_Arab",   # Uyghur
    "uk": "ukr_Cyrl",   # Ukrainian
    "ur": "urd_Arab",   # Urdu
    "uz": "uzn_Latn",   # Uzbek (Northern, Latin)
    "vi": "vie_Latn",   # Vietnamese
    "wa": None,          # Walloon — not in FLORES-200
    "xh": "xho_Latn",   # Xhosa
    "yi": "ydd_Hebr",   # Yiddish (Eastern)
    "yo": "yor_Latn",   # Yoruba
    "zh": "zho_Hans",   # Chinese (Simplified)
    "zu": "zul_Latn",   # Zulu
}

DOMINANT_LANG = "eng_Latn"

Record = Dict[str, str]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_json(path: Path) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(data: List[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _sample_records(
    records: List[Record],
    ratio: float,
    seed: int,
    split: str = "",
) -> List[Record]:
    """
    Randomly down-sample OPUS-100 records to `ratio` fraction after full loading.

    Called once per split inside AlignmentDataset.load(), immediately after
    _load_opus100() returns, so all downstream code (get_joint, save,
    DataLoader) always sees the already-reduced list.

    Parameters
    ----------
    records : full list of records loaded from disk
    ratio   : float in (0.0, 1.0].  1.0 → return the same list unchanged.
    seed    : RNG seed for reproducibility (always _GLOBAL_SAMPLE_SEED = 42).
    split   : human-readable label used only in the log message ("train"/"dev").

    Returns
    -------
    New list with round(len(records) * ratio) randomly chosen records.
    Always contains at least 1 record.
    """
    if not (0.0 < ratio <= 1.0):
        raise ValueError(f"opus_sample_ratio must be in (0.0, 1.0], got {ratio!r}")

    if ratio == 1.0:
        return records      # fast-path: no copy, no RNG call

    n_total  = len(records)
    n_sample = max(1, round(n_total * ratio))

    rng     = random.Random(seed)
    sampled = rng.sample(records, n_sample)

    print(
        f"  [OPUS-100 sample] split={split!r}  "
        f"{n_total:,} → {n_sample:,} records  "
        f"(ratio={ratio:.1%}, seed={seed})"
    )
    return sampled


def _build_eng_eng_pairs(
    joint_records: List[Record],
    ratio: float,
    seed: int,
    split: str = "",
) -> List[Record]:
    """
    Build eng-eng identity pairs from the source (English) side of joint records.

    Construction logic
    ------------------
    n_eng_eng = round(len(joint_records) * ratio)

    Source pool:
      - ALL FLORES-200 records in joint_records are used first (full, no sub-sampling).
      - If n_eng_eng > len(flores_records), the remainder is sampled from OPUS-100
        records within joint_records, using random.sample(seed=seed).
      - If n_eng_eng <= len(flores_records), we sample n_eng_eng records from the
        FLORES-200 pool using random.sample(seed=seed).

    This prioritises the higher-quality FLORES-200 translations as the source
    of English sentences, falling back to OPUS-100 only when more pairs are needed.

    Each sampled record r produces one eng-eng pair:
        {
            "dominant_language": "eng_Latn",
            "target_language":   "eng_Latn",
            "source_sentence":   r["source_sentence"],
            "target_sentence":   r["source_sentence"],   # identical
        }

    Parameters
    ----------
    joint_records : already-built mixed list (FLORES-200 full + OPUS-100 sampled)
    ratio         : eng-eng count = round(len(joint_records) * ratio); 0.0 → disabled
    seed          : RNG seed — always _GLOBAL_SAMPLE_SEED = 42 so every baseline
                    that calls this with the same joint_records gets the same pairs
    split         : label for logging only

    Returns
    -------
    List of eng-eng Record dicts, length = round(len(joint_records) * ratio).
    Empty list if ratio == 0.0.
    """
    if ratio == 0.0:
        return []

    if not (0.0 < ratio <= 1.0):
        raise ValueError(f"eng_eng_ratio must be in [0.0, 1.0], got {ratio!r}")

    n_eng_eng = max(1, round(len(joint_records) * ratio))

    # Split joint_records into FLORES and OPUS pools by presence of "flores"
    # marker — we distinguish them by checking target_language membership in
    # the known FLORES-200 code set.  A simpler and more robust heuristic:
    # records whose source came from FLORES-200 will have been added first in
    # get_joint(); however since we cannot tag them after the fact we instead
    # partition by whether the English sentence appears in a FLORES-sourced
    # record.  The cleanest solution is to keep FLORES records as a separate
    # list, which _build_eng_eng_pairs receives via joint_records already
    # being ordered [flores... opus...].  We rely on the caller passing
    # n_flores so we can slice correctly.
    #
    # To avoid coupling this function to internal ordering, we accept an
    # explicit flores_count parameter via the caller.  See _build_eng_eng_pairs
    # call site in AlignmentDataset._compute_eng_eng().

    rng = random.Random(seed)

    if n_eng_eng <= len(joint_records):
        sampled = rng.sample(joint_records, n_eng_eng)
    else:
        # ratio > 1.0 is blocked above, so this branch is unreachable in
        # normal use; guard it defensively.
        sampled = list(joint_records)

    pairs: List[Record] = [
        {
            "dominant_language": DOMINANT_LANG,
            "target_language":   DOMINANT_LANG,
            "source_sentence":   r["source_sentence"],
            "target_sentence":   r["source_sentence"],
        }
        for r in sampled
    ]

    print(
        f"  [Eng-Eng pairs]   split={split!r}  "
        f"joint={len(joint_records):,}  ratio={ratio:.1%}  "
        f"→ {len(pairs):,} eng-eng pairs  (seed={seed})"
    )
    return pairs


# ---------------------------------------------------------------------------
# FLORES-200 loader  (always loaded in full — it is small and curated)
# ---------------------------------------------------------------------------

def _load_flores200(flores_dir: Path) -> Dict[str, List[Record]]:
    """
    Load FLORES-200 dev.json and devtest.json.

    Rename convention (as requested):
        dev.json     → split "dev"
        devtest.json → split "train"

    Returns:
        {"dev": [...records...], "train": [...records...]}
    """
    split_map = {
        "dev.json":     "dev",
        "devtest.json": "train",
    }

    grouped: Dict[str, Dict[int, Dict[str, str]]] = {
        "dev":   defaultdict(dict),
        "train": defaultdict(dict),
    }

    for filename, split_name in split_map.items():
        fpath = flores_dir / filename
        if not fpath.exists():
            print(f"[FLORES-200] WARNING: {fpath} not found, skipping.")
            continue

        raw = _load_json(fpath)
        for entry in tqdm(raw, desc=f"[FLORES-200] Parsing {filename}", leave=False):
            flores_code = f"{entry['iso_639_3']}_{entry['iso_15924']}"
            sentence_id = entry["id"]
            text        = entry["text"]
            grouped[split_name][sentence_id][flores_code] = text

    result: Dict[str, List[Record]] = {"dev": [], "train": []}

    for split_name, id_map in grouped.items():
        records: List[Record] = []
        for sentence_id, lang_texts in tqdm(
            id_map.items(),
            desc=f"[FLORES-200] Pairing {split_name}",
            leave=False,
        ):
            if DOMINANT_LANG not in lang_texts:
                continue
            eng_text = lang_texts[DOMINANT_LANG]
            for flores_code, text in lang_texts.items():
                if flores_code == DOMINANT_LANG:
                    continue
                records.append({
                    "dominant_language": DOMINANT_LANG,
                    "target_language":   flores_code,
                    "source_sentence":   eng_text,
                    "target_sentence":   text,
                })
        result[split_name] = records

    return result


# ---------------------------------------------------------------------------
# OPUS-100 loader
# ---------------------------------------------------------------------------

def _load_opus100(opus_dir: Path) -> Dict[str, List[Record]]:
    """
    Load all language-pair folders inside opus_dir.

    File → split mapping (as requested):
        train.json      → "train"
        test.json       → "dev"   (merged)
        validation.json → "dev"   (merged)

    Only pairs that include English ("en") are processed.
    The non-English side is looked up in OPUS100_TO_FLORES200;
    pairs with None mapping are skipped.

    NOTE: Sampling is NOT applied here. Raw records are returned in full so
    that AlignmentDataset.load() can apply _sample_records() after seeing the
    complete loaded size — consistent with the "tính trên size final" contract.
    """
    file_to_split = {
        "train.json":      "train",
        "test.json":       "dev",
        "validation.json": "dev",
    }

    result: Dict[str, List[Record]] = {"dev": [], "train": []}

    lang_pair_dirs = sorted([d for d in opus_dir.iterdir() if d.is_dir()])

    for pair_dir in tqdm(lang_pair_dirs, desc="[OPUS-100] Language pairs"):
        folder_name = pair_dir.name          # e.g. "af-en", "ar-de"
        parts = folder_name.split("-")
        if len(parts) != 2:
            continue

        lang_a, lang_b = parts

        if lang_b == "en":
            target_opus = lang_a
            en_key      = "en"
            tgt_key     = lang_a
        elif lang_a == "en":
            target_opus = lang_b
            en_key      = "en"
            tgt_key     = lang_b
        else:
            # Non-English pair — skip (dominant must be English)
            continue

        flores_code = OPUS100_TO_FLORES200.get(target_opus)
        if flores_code is None:
            continue

        for filename, split_name in file_to_split.items():
            fpath = pair_dir / filename
            if not fpath.exists():
                continue

            raw = _load_json(fpath)
            for entry in tqdm(
                raw,
                desc=f"  [{folder_name}] {filename}",
                leave=False,
            ):
                translation = entry.get("translation", {})
                en_text  = translation.get(en_key, "").strip()
                tgt_text = translation.get(tgt_key, "").strip()
                if not en_text or not tgt_text:
                    continue
                result[split_name].append({
                    "dominant_language": DOMINANT_LANG,
                    "target_language":   flores_code,
                    "source_sentence":   en_text,
                    "target_sentence":   tgt_text,
                })

    return result


# ---------------------------------------------------------------------------
# AlignmentDataset
# ---------------------------------------------------------------------------

class AlignmentDataset:
    """
    Jointly processes FLORES-200 and OPUS-100 to produce a unified alignment dataset,
    with an optional eng-eng identity pair augmentation for LoRA anchor regularisation.

    Split naming after loading:
        FLORES-200 : dev.json     → "dev"  |  devtest.json  → "train"
        OPUS-100   : test.json + validation.json → "dev"  |  train.json → "train"

    Sampling — OPUS-100 only
    ------------------------
    OPUS-100 training data can reach ~53 million sentence pairs, which is too
    large for most training runs.  `opus_sample_ratio` controls what fraction
    to keep **after** the dataset is fully loaded from disk and **before** any
    DataLoader is constructed.

    Concretely, inside load():
        1. _load_opus100() reads everything from disk → raw lists.
        2. _sample_records() is called once per split on those raw lists,
           using random.sample(seed=_GLOBAL_SAMPLE_SEED=42) to draw the subset.
        3. The sampled lists replace the raw lists in self._opus_data.

    All downstream accessors (get_joint, get_opus, save) therefore always
    return the already-sampled records — no further filtering is needed.

    FLORES-200 is always loaded in full (it is small and curated, ~22 k pairs).

    Eng-Eng Pair Augmentation
    -------------------------
    When `eng_eng_ratio > 0`, identity pairs (source == target == English) are
    appended to the joint records returned by get_joint().  These pairs serve
    as an anchor regulariser that teaches the LoRA branch to preserve English
    representations, enabling a single-model deployment at inference time.

    Construction (per split):
        n_eng_eng = round(len(joint_records) * eng_eng_ratio)
        Sampled from joint_records using random.sample(seed=_GLOBAL_SAMPLE_SEED=42).
        Each sampled record r → eng-eng pair with both sides = r["source_sentence"].

    get_joint() return order (deterministic):
        [ FLORES-200 records ] + [ OPUS-100 sampled records ] + [ eng-eng pairs ]

    Reproducibility guarantee
    -------------------------
    All sampling operations (OPUS-100 sub-sampling and eng-eng pair construction)
    use seed=_GLOBAL_SAMPLE_SEED=42.  Constructing two AlignmentDataset instances
    with identical arguments (opus_sample_ratio, eng_eng_ratio) will always produce
    byte-for-byte identical get_joint() outputs.

    This means all baselines (OT, InfoNCE, BarlowTwins, VICReg, KL-Divergence)
    that share a single AlignmentDataset instance — or independently construct
    one with the same arguments — will train and evaluate on identical data.

    Parameters
    ----------
    alignment_data_path : str
        Root directory containing FLORES-200/ and OPUS-100/ sub-folders.
    opus_sample_ratio : float, default 0.30
        Fraction of OPUS-100 records to keep, in (0.0, 1.0].
        Applied independently to both "train" and "dev" splits.
        Set to 1.0 to use 100 % of OPUS-100 with no sampling.
    eng_eng_ratio : float, default 0.0
        Fraction of eng-eng identity pairs to add, relative to the joint
        (FLORES + OPUS-sampled) record count, in [0.0, 1.0].
        0.0 disables the augmentation entirely (original behaviour).
        Example: 0.30 → add eng-eng pairs equal to 30 % of joint size.
        Applied independently to both "train" and "dev" splits.

    Usage
    -----
        # Default: no eng-eng augmentation
        ds = AlignmentDataset(
            alignment_data_path="../raw_data/alignment/",
            opus_sample_ratio=0.30,
        ).load()

        # With 30 % eng-eng augmentation
        ds = AlignmentDataset(
            alignment_data_path="../raw_data/alignment/",
            opus_sample_ratio=0.30,
            eng_eng_ratio=0.30,
        ).load()

        ds.save(mode="joint")
    """

    def __init__(
        self,
        alignment_data_path: str = "../raw_data/alignment/",
        opus_sample_ratio: float = 0.30,
        eng_eng_ratio: float = 0.0,
    ) -> None:
        if not (0.0 < opus_sample_ratio <= 1.0):
            raise ValueError(
                f"opus_sample_ratio must be in (0.0, 1.0], got {opus_sample_ratio!r}"
            )
        if not (0.0 <= eng_eng_ratio <= 1.0):
            raise ValueError(
                f"eng_eng_ratio must be in [0.0, 1.0], got {eng_eng_ratio!r}"
            )

        self.base_dir          = Path(alignment_data_path)
        self.flores_dir        = self.base_dir / "FLORES-200"
        self.opus_dir          = self.base_dir / "OPUS-100"
        self.opus_sample_ratio = opus_sample_ratio
        self.eng_eng_ratio     = eng_eng_ratio

        self._flores_data:   Dict[str, List[Record]] = {}
        self._opus_data:     Dict[str, List[Record]] = {}
        # Eng-eng pairs are computed lazily in _ensure_eng_eng() and cached here.
        # They are built from the mixed joint records so they must be computed
        # after both _flores_data and _opus_data are populated.
        self._eng_eng_data:  Dict[str, List[Record]] = {"dev": [], "train": []}
        self._loaded = False

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self) -> "AlignmentDataset":
        """
        Load FLORES-200 (full) and OPUS-100 (sampled to opus_sample_ratio),
        then pre-compute eng-eng pairs if eng_eng_ratio > 0.

        All sampling uses seed=_GLOBAL_SAMPLE_SEED=42 for full reproducibility.
        """
        print("=" * 60)
        print("Loading FLORES-200 …")
        print("=" * 60)
        self._flores_data = _load_flores200(self.flores_dir)
        print(
            f"  ✓ FLORES-200  dev={len(self._flores_data['dev']):,}  "
            f"train={len(self._flores_data['train']):,}"
        )

        print()
        print("=" * 60)
        print("Loading OPUS-100 …")
        print("=" * 60)
        raw_opus = _load_opus100(self.opus_dir)
        print(
            f"  ✓ OPUS-100 (raw)  dev={len(raw_opus['dev']):,}  "
            f"train={len(raw_opus['train']):,}"
        )

        # ── Apply sampling to OPUS-100 per-split ───────────────────────────
        # seed is always _GLOBAL_SAMPLE_SEED so that OPUS sub-sampling is
        # identical regardless of which baseline constructs the dataset.
        print()
        if self.opus_sample_ratio < 1.0:
            print(
                f"  Sampling OPUS-100 to {self.opus_sample_ratio:.1%}  "
                f"(seed={_GLOBAL_SAMPLE_SEED}) …"
            )
        self._opus_data = {
            split: _sample_records(
                records=raw_opus[split],
                ratio=self.opus_sample_ratio,
                seed=_GLOBAL_SAMPLE_SEED,
                split=split,
            )
            for split in ("train", "dev")
        }
        print(
            f"  ✓ OPUS-100 (sampled)  dev={len(self._opus_data['dev']):,}  "
            f"train={len(self._opus_data['train']):,}"
        )

        # ── Pre-compute eng-eng pairs ───────────────────────────────────────
        # Built from the joint records (FLORES full + OPUS sampled) so that
        # the eng-eng pool reflects the final training distribution.
        # seed is always _GLOBAL_SAMPLE_SEED for reproducibility.
        if self.eng_eng_ratio > 0.0:
            print()
            print("=" * 60)
            print(f"Building Eng-Eng pairs (ratio={self.eng_eng_ratio:.1%}) …")
            print("=" * 60)
            for split in ("train", "dev"):
                joint = (
                    self._flores_data.get(split, [])
                    + self._opus_data.get(split, [])
                )
                self._eng_eng_data[split] = _build_eng_eng_pairs(
                    joint_records=joint,
                    ratio=self.eng_eng_ratio,
                    seed=_GLOBAL_SAMPLE_SEED,
                    split=split,
                )
            print(
                f"  ✓ Eng-Eng pairs  "
                f"dev={len(self._eng_eng_data['dev']):,}  "
                f"train={len(self._eng_eng_data['train']):,}"
            )

        self._loaded = True
        return self

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def stats(self) -> None:
        """Print a summary of loaded (post-sampling) record counts per source."""
        self._require_loaded()
        rows = [
            ("FLORES-200",         self._flores_data),
            ("OPUS-100 (sampled)", self._opus_data),
        ]
        if self.eng_eng_ratio > 0.0:
            rows.append(("Eng-Eng pairs", self._eng_eng_data))

        for source, data in rows:
            for split, records in data.items():
                print(f"[{source}] {split}: {len(records):,} records")

        # Joint totals
        print()
        for split in ("train", "dev"):
            total = len(self.get_joint(split))
            print(f"[Joint total (incl. eng-eng)] {split}: {total:,} records")

    # ------------------------------------------------------------------
    # Saving
    # ------------------------------------------------------------------

    def save(
        self,
        mode: Literal["joint", "separated"] = "joint",
        output_dir: str = ".",
    ) -> None:
        """
        Persist the dataset to disk.

        Parameters
        ----------
        mode : "joint" | "separated"
            - "joint"     → <output_dir>/joint_data/{dev,train}.json
              (FLORES-200, OPUS-100, and eng-eng records merged into single files,
               in the same deterministic order as get_joint())
            - "separated" → <output_dir>/separated_data/FLORES-200/{dev,train}.json
                                        separated_data/OPUS-100/{dev,train}.json
                                        separated_data/Eng-Eng/{dev,train}.json
                                        (only if eng_eng_ratio > 0)
        output_dir : str
            Root directory for output (defaults to current working directory).
        """
        self._require_loaded()
        out = Path(output_dir)

        if mode == "joint":
            self._save_joint(out)
        elif mode == "separated":
            self._save_separated(out)
        else:
            raise ValueError(f"Unknown mode '{mode}'. Choose 'joint' or 'separated'.")

    def _save_joint(self, out: Path) -> None:
        target = out / "joint_data"
        target.mkdir(parents=True, exist_ok=True)
        print(f"\n[Save] joint → {target.resolve()}")

        for split in ("dev", "train"):
            # get_joint() already returns [flores + opus + eng-eng] in order
            combined = self.get_joint(split)
            dest = target / f"{split}.json"
            print(f"  Writing {split}.json  ({len(combined):,} records) …")
            chunks = _chunked(combined, 1000)
            all_records: List[Record] = []
            for chunk in tqdm(chunks, desc=f"  Saving {split}.json", unit="chunk"):
                all_records.extend(chunk)
            _save_json(all_records, dest)
            print(f"  ✓ Saved → {dest}")

    def _save_separated(self, out: Path) -> None:
        print(f"\n[Save] separated → {(out / 'separated_data').resolve()}")

        sources = [
            ("FLORES-200", self._flores_data),
            ("OPUS-100",   self._opus_data),
        ]
        if self.eng_eng_ratio > 0.0:
            sources.append(("Eng-Eng", self._eng_eng_data))

        for source_name, data in sources:
            source_dir = out / "separated_data" / source_name
            source_dir.mkdir(parents=True, exist_ok=True)
            print(f"\n  [{source_name}] → {source_dir}")

            for split, records in data.items():
                dest = source_dir / f"{split}.json"
                print(f"    Writing {split}.json  ({len(records):,} records) …")
                chunks = _chunked(records, 1000)
                all_records: List[Record] = []
                for chunk in tqdm(chunks, desc=f"    Saving {split}.json", unit="chunk"):
                    all_records.extend(chunk)
                _save_json(all_records, dest)
                print(f"    ✓ Saved → {dest}")

    # ------------------------------------------------------------------
    # Dict-style access
    # ------------------------------------------------------------------

    def get_flores(self, split: Literal["dev", "train"]) -> List[Record]:
        self._require_loaded()
        return self._flores_data.get(split, [])

    def get_opus(self, split: Literal["dev", "train"]) -> List[Record]:
        """Returns the already-sampled OPUS-100 records for the given split."""
        self._require_loaded()
        return self._opus_data.get(split, [])

    def get_eng_eng(self, split: Literal["dev", "train"]) -> List[Record]:
        """
        Returns the pre-computed eng-eng identity pairs for the given split.
        Empty list if eng_eng_ratio == 0.0.
        """
        self._require_loaded()
        return self._eng_eng_data.get(split, [])

    def get_joint(self, split: Literal["dev", "train"]) -> List[Record]:
        """
        Returns the full training-ready record list for a split:

            [ FLORES-200 (full) ]  +  [ OPUS-100 (sampled) ]  +  [ eng-eng pairs ]

        The order is deterministic and identical across all baselines that use
        the same AlignmentDataset instance (or one constructed with the same
        arguments).  Eng-eng pairs are appended last; if eng_eng_ratio == 0.0
        the list is identical to the original FLORES + OPUS joint.

        Do NOT sort or shuffle this list outside of the DataLoader sampler —
        the fixed order is what guarantees cross-baseline reproducibility.
        """
        self._require_loaded()
        return (
            self._flores_data.get(split, [])
            + self._opus_data.get(split, [])
            + self._eng_eng_data.get(split, [])
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _require_loaded(self) -> None:
        if not self._loaded:
            raise RuntimeError("Dataset not loaded yet. Call .load() first.")


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _chunked(lst: list, size: int):
    """Yield successive chunks of `size` from `lst`."""
    for i in range(0, len(lst), size):
        yield lst[i : i + size]


# ---------------------------------------------------------------------------
# AlignmentTorchDataset
# ---------------------------------------------------------------------------

class AlignmentTorchDataset(Dataset):
    """
    Tokenises parallel sentence pairs for the OT alignment framework
    described in the paper (eq. 11–20).

    Architecture recap
    ------------------
    The training loop maintains TWO forward passes per sample:

        H_tgt^(l) = M_LoRA(s_tgt ; Θ, ΔΘ)   ← trainable branch  (eq. 11)
        H_en^(l)  = M(s_en  ; Θ)             ← frozen anchor     (eq. 12)

    These two passes are INDEPENDENT — they each receive their own full
    sequence, not a concatenation. The DataLoader therefore provides
    separate, fully-formed tokenisations for each branch.

    Eng-Eng pairs
    -------------
    When target_language == dominant_language == "eng_Latn" (eng-eng pairs),
    tgt_input_ids and en_input_ids will tokenise to the SAME sequence.
    The training loop should handle this transparently — the OT loss will
    receive near-identical hidden states from the two branches (modulo LoRA
    drift), producing a loss signal that pulls LoRA back toward the frozen
    English anchor.

    L_LM (eq. 20) — causal LM loss on the TARGET branch only
    ----------------------------------------------------------
    The target branch performs autoregressive reconstruction of s_tgt:

        L_LM = -1/n  Σ_k  log p_ΔΘ(t_k | t_1, …, t_{k-1})

    This is teacher-forced next-token prediction over the entire target
    sequence. For eng-eng pairs s_tgt IS English, so L_LM also receives
    English supervision, which prevents the LoRA branch from forgetting
    English fluency.

    L_OT (eq. 17–18) — OT loss uses hidden states, NOT token ids
    -------------------------------------------------------------
    The OT loss is computed by the training loop on the hidden states
    H_tgt^(l) and H_en^(l) extracted from the respective forward passes.
    The DataLoader does not compute OT — it only supplies the token ids and
    attention masks so the training loop can call model.forward().

    Each sample returns a flat dict with TWO sets of tensors:

        tgt_input_ids      : LongTensor [tgt_len]   — fed to M_LoRA
        tgt_attention_mask : LongTensor [tgt_len]
        tgt_labels         : LongTensor [tgt_len]   — equals tgt_input_ids
                                                       (L_LM, eq. 20)
        en_input_ids       : LongTensor [en_len]    — fed to frozen M
        en_attention_mask  : LongTensor [en_len]
        dominant_language  : str
        target_language    : str                     — "eng_Latn" for eng-eng pairs
    """

    def __init__(
        self,
        records: List[Record],
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 512,
    ) -> None:
        self.records    = records
        self.tokenizer  = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.records)

    def _encode(self, text: str) -> List[int]:
        """
        Tokenise a single sentence into:
            [BOS]  token_1  …  token_n  [EOS]
        Truncates to max_length (from the right) to preserve BOS.
        """
        bos = [self.tokenizer.bos_token_id] if self.tokenizer.bos_token_id is not None else []
        eos = [self.tokenizer.eos_token_id] if self.tokenizer.eos_token_id is not None else []

        body = self.tokenizer.encode(text, add_special_tokens=False)
        ids  = bos + body + eos

        if len(ids) > self.max_length:
            keep = self.max_length - len(bos) - len(eos)
            ids  = bos + body[:keep] + eos

        return ids

    def __getitem__(self, idx: int) -> Dict:
        rec = self.records[idx]

        # ── Target branch: s_tgt → M_LoRA (eq. 11 + eq. 20) ──────────────
        tgt_ids    = self._encode(rec["target_sentence"])
        tgt_labels = list(tgt_ids)

        # ── Dominant branch: s_en → frozen M (eq. 12) ─────────────────────
        # For eng-eng pairs, source_sentence == target_sentence, so both
        # branches receive the same tokenised input.  This is intentional.
        en_ids = self._encode(rec["source_sentence"])

        return {
            "tgt_input_ids":      torch.tensor(tgt_ids,    dtype=torch.long),
            "tgt_attention_mask": torch.ones(len(tgt_ids), dtype=torch.long),
            "tgt_labels":         torch.tensor(tgt_labels, dtype=torch.long),
            "en_input_ids":       torch.tensor(en_ids,     dtype=torch.long),
            "en_attention_mask":  torch.ones(len(en_ids),  dtype=torch.long),
            "dominant_language":  rec["dominant_language"],
            "target_language":    rec["target_language"],
        }


# ---------------------------------------------------------------------------
# SortedLengthSampler  (dynamic batching by target-sequence length)
# ---------------------------------------------------------------------------

class SortedLengthSampler(Sampler):
    """
    Groups sample indices into buckets of similar TARGET sequence length so
    each batch requires minimal padding on the tgt_* tensors (which drive
    the memory-intensive M_LoRA forward pass).

    Algorithm
    ---------
    1. Proxy length = len(target_sentence) characters  (fast, no tokeniser call).
    2. Sort all indices by proxy length.
    3. Chunk into buckets of `batch_size`.
    4. Optionally shuffle within buckets and shuffle bucket order.

    Reproducibility
    ---------------
    The RNG seed advances deterministically per epoch via set_epoch().
    Two DataLoaders constructed with the same seed and batch_size will
    produce the same bucket order for every epoch, regardless of which
    baseline is using them — provided they wrap the same record list.

    Parameters
    ----------
    dataset    : AlignmentTorchDataset
    batch_size : target batch size
    shuffle    : randomise order within and across buckets each epoch
    seed       : base RNG seed (default _GLOBAL_SAMPLE_SEED = 42)
    """

    def __init__(
        self,
        dataset: AlignmentTorchDataset,
        batch_size: int,
        shuffle: bool = True,
        seed: int = _GLOBAL_SAMPLE_SEED,
    ) -> None:
        self.dataset    = dataset
        self.batch_size = batch_size
        self.shuffle    = shuffle
        self.seed       = seed
        self._epoch     = 0

        self._lengths: List[int] = [
            len(rec["target_sentence"])
            for rec in tqdm(
                dataset.records,
                desc="[Sampler] Pre-computing lengths",
                leave=False,
            )
        ]

    def set_epoch(self, epoch: int) -> None:
        """Advance RNG seed so each epoch has a different bucket shuffling."""
        self._epoch = epoch

    def __iter__(self) -> Iterator[int]:
        rng = random.Random(self.seed + self._epoch)

        indices = sorted(range(len(self.dataset)), key=lambda i: self._lengths[i])
        buckets = list(_chunked(indices, self.batch_size))

        if self.shuffle:
            for bucket in buckets:
                rng.shuffle(bucket)
            rng.shuffle(buckets)

        for bucket in buckets:
            yield from bucket

    def __len__(self) -> int:
        return len(self.dataset)


# ---------------------------------------------------------------------------
# Collate function — pads tgt and en branches independently
# ---------------------------------------------------------------------------

def _collate_fn(batch: List[Dict], pad_token_id: int) -> Dict:
    """
    Right-pads each branch (tgt and en) to its own maximum length in the
    batch. The two branches can have different padded lengths, which is
    correct: M_LoRA and M receive differently-sized inputs.

    For eng-eng pairs both branches have the same token sequence, so their
    padded lengths will match within that sample — but will differ from other
    samples in the batch unless they happen to be the same length.

    Padding:
        input_ids      → pad_token_id
        attention_mask → 0
        labels         → -100  (pad positions ignored by CrossEntropyLoss)
    """
    def _pad_branch(
        ids_key: str,
        mask_key: str,
        labels_key: Optional[str],
    ) -> Dict[str, torch.Tensor]:
        max_len = max(s[ids_key].size(0) for s in batch)
        ids_list, mask_list, lbl_list = [], [], []
        for s in batch:
            n   = s[ids_key].size(0)
            pad = max_len - n
            ids_list.append(torch.cat([
                s[ids_key],
                torch.full((pad,), pad_token_id, dtype=torch.long),
            ]))
            mask_list.append(torch.cat([
                s[mask_key],
                torch.zeros(pad, dtype=torch.long),
            ]))
            if labels_key is not None:
                lbl_list.append(torch.cat([
                    s[labels_key],
                    torch.full((pad,), -100, dtype=torch.long),
                ]))
        out = {
            ids_key:  torch.stack(ids_list),
            mask_key: torch.stack(mask_list),
        }
        if labels_key is not None:
            out[labels_key] = torch.stack(lbl_list)
        return out

    result: Dict = {}
    result.update(_pad_branch("tgt_input_ids", "tgt_attention_mask", "tgt_labels"))
    result.update(_pad_branch("en_input_ids",  "en_attention_mask",  None))
    result["dominant_language"] = [s["dominant_language"] for s in batch]
    result["target_language"]   = [s["target_language"]   for s in batch]
    return result


# ---------------------------------------------------------------------------
# AlignmentDataLoader  (main public class)
# ---------------------------------------------------------------------------

class AlignmentDataLoader:
    """
    Standard PyTorch DataLoader for the OT-based cross-lingual alignment
    framework (Dinh, 2024).  Designed for Llama-3-8B-base.

    What each batch contains
    ------------------------
    Each batch is a dict ready to be unpacked for two independent forward
    passes, matching eq. (11) and (12) in the paper:

        tgt_input_ids      [B, L_tgt]  → M_LoRA(s_tgt)   (trainable branch)
        tgt_attention_mask [B, L_tgt]
        tgt_labels         [B, L_tgt]  → L_LM  (eq. 20, full s_tgt reconstruction)
        en_input_ids       [B, L_en]   → M(s_en)          (frozen anchor branch)
        en_attention_mask  [B, L_en]
        dominant_language  List[str]
        target_language    List[str]   — "eng_Latn" for eng-eng identity pairs

    Note: L_tgt and L_en can differ within the same batch (each branch is
    padded to its own maximum length).  For eng-eng pairs L_tgt == L_en
    for that sample (same token sequence), but L_tgt and L_en at the
    batch level are still padded independently.

    Cross-baseline reproducibility
    ------------------------------
    To guarantee that all baselines (OT, InfoNCE, BarlowTwins, VICReg,
    KL-Divergence) train on identical data:

        1. Construct ONE AlignmentDataset and share it across all baselines:

               ds = AlignmentDataset(
                   alignment_data_path="...",
                   opus_sample_ratio=0.30,
                   eng_eng_ratio=0.30,   # or 0.0 to disable
               ).load()

        2. Construct one AlignmentDataLoader per baseline using the SAME ds,
           split, source, batch_size, and seed:

               loader_ot      = AlignmentDataLoader(ds, split="train", seed=42)
               loader_infonce = AlignmentDataLoader(ds, split="train", seed=42)
               # Both loaders iterate the same records in the same bucket order.

        3. Call set_epoch(epoch) on each loader at the start of every epoch
           to get consistent per-epoch shuffling across baselines.

    The training loop is responsible for:
        1. Forward M_LoRA(tgt_input_ids, tgt_attention_mask)   → logits + H_tgt^(l)
        2. Forward M(en_input_ids, en_attention_mask) [no_grad] → H_en^(l)
        3. L_LM   from logits vs tgt_labels  (eq. 20)
        4. L_align from H_tgt^(l) and H_en^(l) via method-specific loss
        5. L = L_LM + λ · L_align  (eq. 19 / eq. 21)

    Parameters
    ----------
    dataset      : AlignmentDataset  (must be .load()-ed first)
    split        : "train" | "dev"
    source       : "joint" | "flores" | "opus" | "eng_eng"
                   "joint" includes eng-eng pairs if eng_eng_ratio > 0.
                   "eng_eng" returns only the eng-eng pairs (useful for ablations).
    tokenizer    : HF tokeniser; if None, loads meta-llama/Meta-Llama-3-8B
    batch_size   : samples per batch
    max_length   : max tokens per branch sequence (truncates body, keeps BOS+EOS)
    shuffle      : shuffle buckets each epoch (True for train, False for dev/eval)
    seed         : base RNG seed — use the same value for all baselines
    num_workers  : DataLoader worker processes
    pin_memory   : pin tensors to CUDA-pinned memory

    Usage
    -----
        ds = AlignmentDataset(
            alignment_data_path="../raw_data/alignment/",
            opus_sample_ratio=0.30,
            eng_eng_ratio=0.30,
        ).load()

        # All baselines share one dataset instance and use seed=42
        train_loader = AlignmentDataLoader(ds, split="train", batch_size=8,  seed=42)
        dev_loader   = AlignmentDataLoader(ds, split="dev",   batch_size=16, seed=42,
                                           shuffle=False)

        for epoch in range(num_epochs):
            train_loader.set_epoch(epoch)
            for batch in train_loader:
                # batch["target_language"] == "eng_Latn" → eng-eng pair
                tgt_out = model_lora(
                    input_ids      = batch["tgt_input_ids"],
                    attention_mask = batch["tgt_attention_mask"],
                    output_hidden_states = True,
                )
                with torch.no_grad():
                    en_out = model_frozen(
                        input_ids      = batch["en_input_ids"],
                        attention_mask = batch["en_attention_mask"],
                        output_hidden_states = True,
                    )
                loss_lm = ce_loss(tgt_out.logits, batch["tgt_labels"])
                H_tgt = [tgt_out.hidden_states[l] for l in middle_layers]
                H_en  = [en_out.hidden_states[l]  for l in middle_layers]
                loss_align = alignment_loss(H_tgt, H_en, ...)
                loss = loss_lm + lambda_ * loss_align
    """

    def __init__(
        self,
        dataset: AlignmentDataset,
        split: Literal["train", "dev"] = "train",
        source: Literal["joint", "flores", "opus", "eng_eng"] = "joint",
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        batch_size: int = 8,
        max_length: int = 512,
        shuffle: bool = True,
        seed: int = _GLOBAL_SAMPLE_SEED,
        num_workers: int = 0,
        pin_memory: bool = False,
    ) -> None:
        self.split      = split
        self.source     = source
        self.batch_size = batch_size
        self.max_length = max_length
        self.shuffle    = shuffle
        self.seed       = seed

        # ── 1. Tokeniser ───────────────────────────────────────────────────
        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            print(f"[DataLoader] Loading tokenizer: {DEFAULT_LLAMA3_MODEL}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                DEFAULT_LLAMA3_MODEL,
                use_fast=True,
            )

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # ── 2. Records ─────────────────────────────────────────────────────
        # "joint" already contains eng-eng pairs (appended last) if
        # eng_eng_ratio > 0 — no extra logic needed here.
        if source == "joint":
            records = dataset.get_joint(split)
        elif source == "flores":
            records = dataset.get_flores(split)
        elif source == "opus":
            records = dataset.get_opus(split)
        elif source == "eng_eng":
            records = dataset.get_eng_eng(split)
        else:
            raise ValueError(
                f"Unknown source '{source}'. Choose: joint | flores | opus | eng_eng."
            )

        if not records:
            raise ValueError(
                f"No records found for split='{split}', source='{source}'. "
                "Ensure AlignmentDataset.load() has been called "
                "and eng_eng_ratio > 0 if source='eng_eng'."
            )

        n_eng_eng = sum(
            1 for r in records if r.get("target_language") == DOMINANT_LANG
        )
        print(
            f"[DataLoader] split={split!r}  source={source!r}  "
            f"records={len(records):,}  eng-eng={n_eng_eng:,}  "
            f"batch_size={batch_size}  max_length={max_length}  seed={seed}"
        )

        # ── 3. Torch Dataset ───────────────────────────────────────────────
        self._torch_dataset = AlignmentTorchDataset(
            records=records,
            tokenizer=self.tokenizer,
            max_length=max_length,
        )

        # ── 4. Sampler ─────────────────────────────────────────────────────
        # seed is passed explicitly so all baselines using the same value
        # get the same bucket order every epoch.
        self._sampler = SortedLengthSampler(
            dataset=self._torch_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
        )

        # ── 5. DataLoader ──────────────────────────────────────────────────
        pad_id = self.tokenizer.pad_token_id
        self._loader = DataLoader(
            self._torch_dataset,
            batch_size=batch_size,
            sampler=self._sampler,
            collate_fn=lambda b: _collate_fn(b, pad_token_id=pad_id),
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        )

    def set_epoch(self, epoch: int) -> None:
        """
        Call at the start of each training epoch to re-shuffle buckets.
        Must be called on ALL baseline loaders before each epoch to keep
        their iteration order in sync.
        """
        self._sampler.set_epoch(epoch)

    def __iter__(self):
        return iter(self._loader)

    def __len__(self) -> int:
        return len(self._loader)

    @property
    def dataset(self) -> AlignmentTorchDataset:
        """The underlying AlignmentTorchDataset."""
        return self._torch_dataset


# ---------------------------------------------------------------------------
# Quick smoke-test  (tokeniser only — no model weights, no LLM inference)
# Run:  python alignment_loader.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # ── 1. Dataset ──────────────────────────────────────────────────────────
    ds = AlignmentDataset(
        alignment_data_path="../raw_data/alignment/",
        opus_sample_ratio=0.01,   # ← 1 % of OPUS-100 for fast smoke-test
        eng_eng_ratio=0.30,       # ← add eng-eng pairs = 30 % of joint size
    )
    ds.load()
    ds.stats()

    # ── 2. Tokeniser ────────────────────────────────────────────────────────
    print(f"\n[Smoke-test] Loading tokenizer: {DEFAULT_LLAMA3_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_LLAMA3_MODEL, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    print(
        f"[Smoke-test] vocab_size={tokenizer.vocab_size:,}  "
        f"bos={tokenizer.bos_token_id}  "
        f"eos={tokenizer.eos_token_id}  "
        f"pad={tokenizer.pad_token_id}"
    )

    # ── 3. Build loaders ────────────────────────────────────────────────────
    # Both loaders use the same ds instance and seed=42 → identical data order.
    train_loader = AlignmentDataLoader(
        dataset=ds, split="train", source="joint",
        tokenizer=tokenizer, batch_size=8, max_length=256,
        shuffle=True, seed=_GLOBAL_SAMPLE_SEED,
    )
    dev_loader = AlignmentDataLoader(
        dataset=ds, split="dev", source="joint",
        tokenizer=tokenizer, batch_size=16, max_length=256,
        shuffle=False, seed=_GLOBAL_SAMPLE_SEED,
    )
    print(f"\n[Smoke-test] train batches : {len(train_loader):,}")
    print(f"[Smoke-test] dev   batches : {len(dev_loader):,}")

    # ── 4. Inspect first training batch ─────────────────────────────────────
    print("\n[Smoke-test] First training batch:")
    batch = next(iter(train_loader))
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k:25s}  shape={tuple(v.shape)}  dtype={v.dtype}")
        else:
            print(f"  {k:25s}  {v[:2]} …")

    # ── 5. Verify tgt_labels == tgt_input_ids (full reconstruction, eq. 20) ─
    tgt_ids    = batch["tgt_input_ids"]
    tgt_labels = batch["tgt_labels"]
    non_pad_mask = tgt_labels != -100
    assert torch.all(tgt_ids[non_pad_mask] == tgt_labels[non_pad_mask]), \
        "tgt_labels mismatch: non-pad labels should equal tgt_input_ids"
    print("\n[Smoke-test] ✓ tgt_labels == tgt_input_ids on non-pad positions  (eq. 20)")

    # ── 6. Verify eng-eng pairs appear in joint ──────────────────────────────
    joint_train = ds.get_joint("train")
    eng_eng_count = sum(
        1 for r in joint_train if r["target_language"] == DOMINANT_LANG
    )
    expected = len(ds.get_eng_eng("train"))
    assert eng_eng_count == expected, (
        f"Eng-eng count mismatch: found {eng_eng_count}, expected {expected}"
    )
    print(f"[Smoke-test] ✓ eng-eng pairs in joint train: {eng_eng_count:,}  (expected {expected:,})")

    # ── 7. Verify eng-eng pairs have source == target ────────────────────────
    eng_eng_records = ds.get_eng_eng("train")
    assert all(
        r["source_sentence"] == r["target_sentence"] for r in eng_eng_records
    ), "Some eng-eng pairs have mismatched source/target sentences"
    print("[Smoke-test] ✓ All eng-eng pairs have source_sentence == target_sentence")

    # ── 8. Verify reproducibility: two loaders on same ds give same first batch ─
    loader_a = AlignmentDataLoader(
        dataset=ds, split="train", source="joint",
        tokenizer=tokenizer, batch_size=8, max_length=256,
        shuffle=True, seed=_GLOBAL_SAMPLE_SEED,
    )
    loader_b = AlignmentDataLoader(
        dataset=ds, split="train", source="joint",
        tokenizer=tokenizer, batch_size=8, max_length=256,
        shuffle=True, seed=_GLOBAL_SAMPLE_SEED,
    )
    batch_a = next(iter(loader_a))
    batch_b = next(iter(loader_b))
    assert torch.all(batch_a["tgt_input_ids"] == batch_b["tgt_input_ids"]), \
        "Reproducibility check failed: two loaders with same seed gave different batches"
    print("[Smoke-test] ✓ Two loaders with same seed produce identical first batch")

    # ── 9. Decode sample 0 from both branches for visual inspection ──────────
    print("\n[Smoke-test] Sample 0 decodes:")
    tgt_tok = batch["tgt_input_ids"][0]
    en_tok  = batch["en_input_ids"][0]
    tgt_pad_mask = batch["tgt_attention_mask"][0].bool()
    en_pad_mask  = batch["en_attention_mask"][0].bool()
    print(f"  s_tgt : {tokenizer.decode(tgt_tok[tgt_pad_mask], skip_special_tokens=False)!r}")
    print(f"  s_en  : {tokenizer.decode(en_tok[en_pad_mask],   skip_special_tokens=False)!r}")
    print(f"  target_language    : {batch['target_language'][0]}")
    print(f"  dominant_language  : {batch['dominant_language'][0]}")

    print("\n✓ Smoke-test passed.")