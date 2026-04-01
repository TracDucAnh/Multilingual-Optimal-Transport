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
    seed    : RNG seed for reproducibility (default kept at 42 by the caller).
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
    Jointly processes FLORES-200 and OPUS-100 to produce a unified alignment dataset.

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
           using random.sample(seed=opus_sample_seed) to draw the subset.
        3. The sampled lists replace the raw lists in self._opus_data.

    All downstream accessors (get_joint, get_opus, save) therefore always
    return the already-sampled records — no further filtering is needed.

    FLORES-200 is always loaded in full (it is small and curated, ~22 k pairs).

    Parameters
    ----------
    alignment_data_path : str
        Root directory containing FLORES-200/ and OPUS-100/ sub-folders.
    opus_sample_ratio : float, default 0.30
        Fraction of OPUS-100 records to keep, in (0.0, 1.0].
        Applied independently to both "train" and "dev" splits.
        Set to 1.0 to use 100 % of OPUS-100 with no sampling.
    opus_sample_seed : int, default 42
        RNG seed passed to random.sample for reproducibility.

    Usage
    -----
        # 30 % of OPUS-100 (default)
        ds = AlignmentDataset(
            alignment_data_path="../raw_data/alignment/",
            opus_sample_ratio=0.30,
            opus_sample_seed=42,
        ).load()

        # 100 % — use full OPUS-100
        ds_full = AlignmentDataset(
            alignment_data_path="../raw_data/alignment/",
            opus_sample_ratio=1.0,
        ).load()

        ds.save(mode="joint")
    """

    def __init__(
        self,
        alignment_data_path: str = "../raw_data/alignment/",
        opus_sample_ratio: float = 0.30,
        opus_sample_seed: int = 42,
    ) -> None:
        if not (0.0 < opus_sample_ratio <= 1.0):
            raise ValueError(
                f"opus_sample_ratio must be in (0.0, 1.0], got {opus_sample_ratio!r}"
            )

        self.base_dir          = Path(alignment_data_path)
        self.flores_dir        = self.base_dir / "FLORES-200"
        self.opus_dir          = self.base_dir / "OPUS-100"
        self.opus_sample_ratio = opus_sample_ratio
        self.opus_sample_seed  = opus_sample_seed

        self._flores_data: Dict[str, List[Record]] = {}
        self._opus_data:   Dict[str, List[Record]] = {}
        self._loaded = False

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self) -> "AlignmentDataset":
        """
        Load FLORES-200 (full) and OPUS-100 (sampled to opus_sample_ratio).

        Sampling is applied per-split immediately after _load_opus100() returns,
        so the final sizes are visible in the log before any DataLoader is built.
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
        # Sampling is done here — after full loading — so the ratio is computed
        # against the real final size of each split, exactly as specified.
        print()
        if self.opus_sample_ratio < 1.0:
            print(
                f"  Sampling OPUS-100 to {self.opus_sample_ratio:.1%}  "
                f"(seed={self.opus_sample_seed}) …"
            )
        self._opus_data = {
            split: _sample_records(
                records=raw_opus[split],
                ratio=self.opus_sample_ratio,
                seed=self.opus_sample_seed,
                split=split,
            )
            for split in ("train", "dev")
        }
        print(
            f"  ✓ OPUS-100 (sampled)  dev={len(self._opus_data['dev']):,}  "
            f"train={len(self._opus_data['train']):,}"
        )

        self._loaded = True
        return self

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def stats(self) -> None:
        """Print a summary of loaded (post-sampling) record counts."""
        self._require_loaded()
        for source, data in [
            ("FLORES-200",          self._flores_data),
            ("OPUS-100 (sampled)",  self._opus_data),
        ]:
            for split, records in data.items():
                print(f"[{source}] {split}: {len(records):,} records")

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
              (FLORES-200 and OPUS-100 records merged into single files)
            - "separated" → <output_dir>/separated_data/FLORES-200/{dev,train}.json
                                        separated_data/OPUS-100/{dev,train}.json
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
            combined = self._flores_data.get(split, []) + self._opus_data.get(split, [])
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

        for source_name, data in [
            ("FLORES-200", self._flores_data),
            ("OPUS-100",   self._opus_data),
        ]:
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

    def get_joint(self, split: Literal["dev", "train"]) -> List[Record]:
        """Returns FLORES-200 (full) + OPUS-100 (sampled) for the given split."""
        self._require_loaded()
        return self._flores_data.get(split, []) + self._opus_data.get(split, [])

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

    L_LM (eq. 20) — causal LM loss on the TARGET branch only
    ----------------------------------------------------------
    The target branch performs autoregressive reconstruction of s_tgt:

        L_LM = -1/n  Σ_k  log p_ΔΘ(t_k | t_1, …, t_{k-1})

    This is teacher-forced next-token prediction over the entire target
    sequence. The sequence fed to M_LoRA is:

        [BOS]  t_1  t_2  …  t_n  [EOS]

    and the labels are the same sequence shifted by one (standard causal LM):

        labels[i] = input_ids[i+1]   (implemented via labels = input_ids,
                                       PyTorch's CrossEntropyLoss handles shift)

    Concretely we set labels == input_ids for ALL positions (no masking),
    meaning every next-token prediction step contributes to L_LM, exactly
    as written in eq. 20.

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
        target_language    : str
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

    Parameters
    ----------
    dataset    : AlignmentTorchDataset
    batch_size : target batch size
    shuffle    : randomise order within and across buckets each epoch
    seed       : base RNG seed; call set_epoch() to advance per epoch
    """

    def __init__(
        self,
        dataset: AlignmentTorchDataset,
        batch_size: int,
        shuffle: bool = True,
        seed: int = 42,
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
        target_language    List[str]

    Note: L_tgt and L_en can differ within the same batch (each branch is
    padded to its own maximum length).

    The training loop is responsible for:
        1. Forward M_LoRA(tgt_input_ids, tgt_attention_mask)   → logits + H_tgt^(l)
        2. Forward M(en_input_ids, en_attention_mask) [no_grad] → H_en^(l)
        3. L_LM   from logits vs tgt_labels  (eq. 20)
        4. L_OT   from H_tgt^(l) and H_en^(l) via Sinkhorn  (eq. 17–18)
        5. L = L_LM + λ · L_OT  (eq. 19)

    Features
    --------
    - Dynamic batching: bucketed by target-sentence length → minimal tgt padding.
    - Shuffle + seed: reproducible; call set_epoch() before each training epoch.
    - Pluggable tokeniser: defaults to Llama-3-8B; pass any HF tokeniser.

    Parameters
    ----------
    dataset      : AlignmentDataset  (must be .load()-ed first; OPUS-100 already
                   sampled to opus_sample_ratio at construction time)
    split        : "train" | "dev"
    source       : "joint" | "flores" | "opus"
    tokenizer    : HF tokeniser; if None, loads meta-llama/Meta-Llama-3-8B
    batch_size   : samples per batch
    max_length   : max tokens per branch sequence (truncates body, keeps BOS+EOS)
    shuffle      : shuffle buckets each epoch (True for train, False for dev/eval)
    seed         : base RNG seed
    num_workers  : DataLoader worker processes
    pin_memory   : pin tensors to CUDA-pinned memory

    Usage
    -----
        ds = AlignmentDataset(
            alignment_data_path="../raw_data/alignment/",
            opus_sample_ratio=0.30,   # 30 % of OPUS-100; set 1.0 for full data
            opus_sample_seed=42,
        ).load()

        train_loader = AlignmentDataLoader(ds, split="train", batch_size=8)
        dev_loader   = AlignmentDataLoader(ds, split="dev",   batch_size=16,
                                           shuffle=False)

        for epoch in range(num_epochs):
            train_loader.set_epoch(epoch)
            for batch in train_loader:
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
                loss_ot = ot_loss(H_tgt, H_en,
                                  batch["tgt_attention_mask"],
                                  batch["en_attention_mask"])
                loss = loss_lm + lambda_ * loss_ot
    """

    def __init__(
        self,
        dataset: AlignmentDataset,
        split: Literal["train", "dev"] = "train",
        source: Literal["joint", "flores", "opus"] = "joint",
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        batch_size: int = 8,
        max_length: int = 512,
        shuffle: bool = True,
        seed: int = 42,
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

        # ── 2. Records — OPUS-100 already sampled inside AlignmentDataset ──
        if source == "joint":
            records = dataset.get_joint(split)
        elif source == "flores":
            records = dataset.get_flores(split)
        elif source == "opus":
            records = dataset.get_opus(split)
        else:
            raise ValueError(f"Unknown source '{source}'. Choose: joint | flores | opus.")

        if not records:
            raise ValueError(
                f"No records found for split='{split}', source='{source}'. "
                "Ensure AlignmentDataset.load() has been called."
            )

        print(
            f"[DataLoader] split={split!r}  source={source!r}  "
            f"records={len(records):,}  batch_size={batch_size}  "
            f"max_length={max_length}"
        )

        # ── 3. Torch Dataset ───────────────────────────────────────────────
        self._torch_dataset = AlignmentTorchDataset(
            records=records,
            tokenizer=self.tokenizer,
            max_length=max_length,
        )

        # ── 4. Sampler ─────────────────────────────────────────────────────
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
        """Call at the start of each training epoch to re-shuffle buckets."""
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
        opus_sample_ratio=0.01,   # ← 1 % of OPUS-100; set 1.0 for full data
        opus_sample_seed=42,
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
    train_loader = AlignmentDataLoader(
        dataset=ds, split="train", source="joint",
        tokenizer=tokenizer, batch_size=8, max_length=256,
        shuffle=True, seed=42,
    )
    dev_loader = AlignmentDataLoader(
        dataset=ds, split="dev", source="joint",
        tokenizer=tokenizer, batch_size=16, max_length=256,
        shuffle=False, seed=42,
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

    # ── 6. Verify tgt and en branches are independent tensors ───────────────
    assert batch["tgt_input_ids"].shape != batch["en_input_ids"].shape or \
           not torch.all(batch["tgt_input_ids"] == batch["en_input_ids"]), \
        "tgt and en branches should generally differ"
    print("[Smoke-test] ✓ tgt branch and en branch are independent  (eq. 11-12)")

    # ── 7. Decode sample 0 from both branches for visual inspection ──────────
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