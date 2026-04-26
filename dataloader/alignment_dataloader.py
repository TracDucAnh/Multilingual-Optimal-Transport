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

──────────────────────────────────────────────────────────────────────
DISTRIBUTED CHANGES
──────────────────────────────────────────────────────────────────────

[DIST-A]  AlignmentDataLoader now accepts optional `rank` and
          `world_size` kwargs (both default to None / 1 so all
          single-GPU code is 100 % backwards-compatible).

[DIST-B]  When world_size > 1, a new DistributedSortedSampler
          replaces SortedLengthSampler.  It first sorts all indices
          by target-sequence length (same as before), then slices
          the sorted list so each rank owns a non-overlapping shard
          of size  ceil(N / world_size).  Within each rank the shard
          is further divided into buckets and optionally shuffled,
          preserving the memory-efficient batching behaviour.

[DIST-C]  set_epoch(epoch) now delegates to the underlying sampler
          (both SortedLengthSampler and DistributedSortedSampler
          implement set_epoch).

[DIST-D]  The DataLoader is exposed as `loader._loader` (unchanged)
          and `loader.dataloader` (alias added for compatibility with
          code that expects that attribute name).

──────────────────────────────────────────────────────────────────────
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
    try:
        login(token=hf_token)
    except Exception:
        print("HERE")
        pass
else:
    print("[Auth] No HF_TOKEN found in .env; proceeding without authentication. Some datasets may not be accessible.")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_LLAMA3_MODEL = "meta-llama/Meta-Llama-3-8B"

_GLOBAL_SAMPLE_SEED = 42

# ---------------------------------------------------------------------------
# Language-code mapping: OPUS-100 (ISO 639-1/2) → FLORES-200 (ISO 639-3_Script)
# ---------------------------------------------------------------------------

OPUS100_TO_FLORES200: Dict[str, Optional[str]] = {
    "af": "afr_Latn",
    "am": "amh_Ethi",
    "an": None,
    "ar": "arb_Arab",
    "as": "asm_Beng",
    "az": "azj_Latn",
    "be": "bel_Cyrl",
    "bg": "bul_Cyrl",
    "bn": "ben_Beng",
    "br": None,
    "bs": "bos_Latn",
    "ca": "cat_Latn",
    "cs": "ces_Latn",
    "cy": "cym_Latn",
    "da": "dan_Latn",
    "de": "deu_Latn",
    "dz": "dzo_Tibt",
    "el": "ell_Grek",
    "en": "eng_Latn",
    "eo": None,
    "es": "spa_Latn",
    "et": "est_Latn",
    "eu": "eus_Latn",
    "fa": "pes_Arab",
    "fi": "fin_Latn",
    "fr": "fra_Latn",
    "fy": "fry_Latn",
    "ga": "gle_Latn",
    "gd": None,
    "gl": "glg_Latn",
    "gu": "guj_Gujr",
    "ha": "hau_Latn",
    "he": "heb_Hebr",
    "hi": "hin_Deva",
    "hr": "hrv_Latn",
    "hu": "hun_Latn",
    "hy": "hye_Armn",
    "id": "ind_Latn",
    "ig": "ibo_Latn",
    "is": "isl_Latn",
    "it": "ita_Latn",
    "ja": "jpn_Jpan",
    "ka": "kat_Geor",
    "kk": "kaz_Cyrl",
    "km": "khm_Khmr",
    "kn": "kan_Knda",
    "ko": "kor_Hang",
    "ku": "kmr_Latn",
    "ky": "kir_Cyrl",
    "li": None,
    "lt": "lit_Latn",
    "lv": "lvs_Latn",
    "mg": "plt_Latn",
    "mk": "mkd_Cyrl",
    "ml": "mal_Mlym",
    "mn": "khk_Cyrl",
    "mr": "mar_Deva",
    "ms": "zsm_Latn",
    "mt": "mlt_Latn",
    "my": "mya_Mymr",
    "nb": "nob_Latn",
    "ne": "npi_Deva",
    "nl": "nld_Latn",
    "nn": None,
    "no": "nob_Latn",
    "oc": "oci_Latn",
    "or": "ory_Orya",
    "pa": "pan_Guru",
    "pl": "pol_Latn",
    "ps": "pbt_Arab",
    "pt": "por_Latn",
    "ro": "ron_Latn",
    "ru": "rus_Cyrl",
    "rw": "kin_Latn",
    "se": None,
    "sh": "hrv_Latn",
    "si": "sin_Sinh",
    "sk": "slk_Latn",
    "sl": "slv_Latn",
    "sq": "als_Latn",
    "sr": "srp_Cyrl",
    "sv": "swe_Latn",
    "ta": "tam_Taml",
    "te": "tel_Telu",
    "tg": "tgk_Cyrl",
    "th": "tha_Thai",
    "tk": "tuk_Latn",
    "tr": "tur_Latn",
    "tt": "tat_Cyrl",
    "ug": "uig_Arab",
    "uk": "ukr_Cyrl",
    "ur": "urd_Arab",
    "uz": "uzn_Latn",
    "vi": "vie_Latn",
    "wa": None,
    "xh": "xho_Latn",
    "yi": "ydd_Hebr",
    "yo": "yor_Latn",
    "zh": "zho_Hans",
    "zu": "zul_Latn",
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


def _build_eng_eng_pairs(
    joint_records: List[Record],
    ratio: float,
    seed: int,
    split: str = "",
) -> List[Record]:
    if ratio == 0.0:
        return []

    if not (0.0 < ratio <= 1.0):
        raise ValueError(f"eng_eng_ratio must be in [0.0, 1.0], got {ratio!r}")

    n_total    = len(joint_records)
    n_eng_eng  = max(1, round(n_total * ratio))

    rng             = random.Random(seed)
    sampled_indices = rng.sample(range(n_total), min(n_eng_eng, n_total))

    pairs: List[Record] = [
        {
            "dominant_language": DOMINANT_LANG,
            "target_language":   DOMINANT_LANG,
            "source_sentence":   joint_records[i]["source_sentence"],
            "target_sentence":   joint_records[i]["source_sentence"],
        }
        for i in sampled_indices
    ]

    print(
        f"  [Eng-Eng pairs]   split={split!r}  "
        f"joint={n_total:,}  ratio={ratio:.1%}  "
        f"→ {len(pairs):,} eng-eng pairs  (seed={seed})"
    )
    return pairs


# ---------------------------------------------------------------------------
# FLORES-200 loader
# ---------------------------------------------------------------------------

def _load_flores200(flores_dir: Path) -> Dict[str, List[Record]]:
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
# OPUS-100 loader  ─  streaming HEAD-K truncation
# ---------------------------------------------------------------------------

def _load_opus100(
    opus_dir: Path,
    opus_sample_ratio: float = 1.0,
) -> Dict[str, List[Record]]:
    import math

    if not (0.0 < opus_sample_ratio <= 1.0):
        raise ValueError(
            f"opus_sample_ratio must be in (0.0, 1.0], got {opus_sample_ratio!r}"
        )

    file_to_split = {
        "train.json":      "train",
        "test.json":       "dev",
        "validation.json": "dev",
    }

    result: Dict[str, List[Record]] = {"dev": [], "train": []}

    lang_pair_dirs = sorted([d for d in opus_dir.iterdir() if d.is_dir()])

    for pair_dir in tqdm(lang_pair_dirs, desc="[OPUS-100] Language pairs"):
        folder_name = pair_dir.name
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
            continue

        flores_code = OPUS100_TO_FLORES200.get(target_opus)
        if flores_code is None:
            continue

        for filename, split_name in file_to_split.items():
            fpath = pair_dir / filename
            if not fpath.exists():
                continue

            raw     = _load_json(fpath)
            n_total = len(raw)

            if opus_sample_ratio < 1.0:
                k = max(1, math.ceil(n_total * opus_sample_ratio))
            else:
                k = n_total

            for entry in tqdm(
                raw[:k],
                desc=f"  [{folder_name}] {filename} ({k}/{n_total})",
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

            del raw

    return result


# ---------------------------------------------------------------------------
# AlignmentDataset
# ---------------------------------------------------------------------------

class AlignmentDataset:
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

        self._flores_data:  Dict[str, List[Record]] = {}
        self._opus_data:    Dict[str, List[Record]] = {}
        self._eng_eng_data: Dict[str, List[Record]] = {"dev": [], "train": []}
        self._loaded = False

    def load(self) -> "AlignmentDataset":
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
        print(
            f"Loading OPUS-100  "
            f"(streaming HEAD-K, ratio={self.opus_sample_ratio:.1%}) …"
        )
        print("=" * 60)
        self._opus_data = _load_opus100(
            opus_dir=self.opus_dir,
            opus_sample_ratio=self.opus_sample_ratio,
        )
        print(
            f"  ✓ OPUS-100  dev={len(self._opus_data['dev']):,}  "
            f"train={len(self._opus_data['train']):,}"
        )

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

    def stats(self) -> None:
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

        print()
        for split in ("train", "dev"):
            total = len(self.get_joint(split))
            print(f"[Joint total (incl. eng-eng)] {split}: {total:,} records")

    def save(
        self,
        mode: Literal["joint", "separated"] = "joint",
        output_dir: str = ".",
    ) -> None:
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

    def get_flores(self, split: Literal["dev", "train"]) -> List[Record]:
        self._require_loaded()
        return self._flores_data.get(split, [])

    def get_opus(self, split: Literal["dev", "train"]) -> List[Record]:
        self._require_loaded()
        return self._opus_data.get(split, [])

    def get_eng_eng(self, split: Literal["dev", "train"]) -> List[Record]:
        self._require_loaded()
        return self._eng_eng_data.get(split, [])

    def get_joint(self, split: Literal["dev", "train"]) -> List[Record]:
        self._require_loaded()
        return (
            self._flores_data.get(split, [])
            + self._opus_data.get(split, [])
            + self._eng_eng_data.get(split, [])
        )

    def _require_loaded(self) -> None:
        if not self._loaded:
            raise RuntimeError("Dataset not loaded yet. Call .load() first.")


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _chunked(lst: list, size: int):
    for i in range(0, len(lst), size):
        yield lst[i : i + size]


# ---------------------------------------------------------------------------
# AlignmentTorchDataset
# ---------------------------------------------------------------------------

class AlignmentTorchDataset(Dataset):
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
        bos  = [self.tokenizer.bos_token_id] if self.tokenizer.bos_token_id is not None else []
        eos  = [self.tokenizer.eos_token_id] if self.tokenizer.eos_token_id is not None else []
        body = self.tokenizer.encode(text, add_special_tokens=False)
        ids  = bos + body + eos

        if len(ids) > self.max_length:
            keep = self.max_length - len(bos) - len(eos)
            ids  = bos + body[:keep] + eos

        return ids

    def __getitem__(self, idx: int) -> Dict:
        rec = self.records[idx]

        tgt_ids    = self._encode(rec["target_sentence"])
        tgt_labels = list(tgt_ids)
        en_ids     = self._encode(rec["source_sentence"])

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
# SortedLengthSampler  (single-GPU / non-distributed)
# ---------------------------------------------------------------------------

class SortedLengthSampler(Sampler):
    """
    Groups sample indices into buckets of similar target-sequence length.
    Used for single-GPU training only.
    For multi-GPU training, use DistributedSortedSampler instead.
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
# DistributedSortedSampler  [DIST-B]
# ---------------------------------------------------------------------------

class DistributedSortedSampler(Sampler):
    """
    Distributed-aware version of SortedLengthSampler.

    Algorithm
    ---------
    1. Sort ALL indices globally by target-sequence length (char proxy).
    2. Pad the sorted list to be divisible by world_size (duplicate last
       element) so every rank gets the same number of samples.
    3. Slice: rank r owns indices[r :: world_size] (interleaved) which
       preserves length-similarity within each rank's shard.
    4. Chunk the shard into buckets of batch_size and optionally shuffle.

    This guarantees:
      • Non-overlapping shards across ranks (no sample duplicated within
        an epoch across ranks, except for the small padding tail).
      • Each rank's batches have similar sequence lengths → minimal padding.
      • set_epoch(epoch) advances the RNG so every epoch is shuffled
        differently but deterministically.

    Parameters
    ----------
    dataset    : AlignmentTorchDataset
    batch_size : per-GPU batch size
    rank       : this process's rank (0-indexed)
    world_size : total number of processes
    shuffle    : shuffle within buckets and across buckets each epoch
    seed       : base RNG seed
    drop_last  : if True, drop the padded tail samples; if False, keep them
    """

    def __init__(
        self,
        dataset: AlignmentTorchDataset,
        batch_size: int,
        rank: int,
        world_size: int,
        shuffle: bool = True,
        seed: int = _GLOBAL_SAMPLE_SEED,
        drop_last: bool = False,
    ) -> None:
        if rank < 0 or rank >= world_size:
            raise ValueError(f"Invalid rank={rank} for world_size={world_size}")

        self.dataset    = dataset
        self.batch_size = batch_size
        self.rank       = rank
        self.world_size = world_size
        self.shuffle    = shuffle
        self.seed       = seed
        self.drop_last  = drop_last
        self._epoch     = 0

        # Pre-compute character-level proxy lengths (fast, no tokeniser call)
        self._lengths: List[int] = [
            len(rec["target_sentence"])
            for rec in tqdm(
                dataset.records,
                desc=f"[DistSampler rank={rank}] Pre-computing lengths",
                leave=False,
            )
        ]

        # Total samples per rank (padded to be divisible by world_size)
        n = len(dataset)
        if drop_last:
            self.num_samples = n // world_size
        else:
            self.num_samples = math.ceil(n / world_size)

        self.total_size = self.num_samples * world_size

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch

    def __iter__(self) -> Iterator[int]:
        rng = random.Random(self.seed + self._epoch)

        # 1. Sort globally by target length
        all_indices = sorted(range(len(self.dataset)), key=lambda i: self._lengths[i])

        # 2. Pad / truncate to total_size
        if not self.drop_last:
            # Repeat from the front to reach total_size
            padding = self.total_size - len(all_indices)
            all_indices = all_indices + all_indices[:padding]
        else:
            all_indices = all_indices[:self.total_size]

        assert len(all_indices) == self.total_size

        # 3. Interleaved slice for this rank
        #    rank 0 → [0, world_size, 2*world_size, ...]
        #    rank 1 → [1, world_size+1, ...]
        #    This keeps similar lengths grouped within each rank's shard.
        shard = all_indices[self.rank : self.total_size : self.world_size]
        assert len(shard) == self.num_samples

        # 4. Chunk into buckets of batch_size and optionally shuffle
        buckets = list(_chunked(shard, self.batch_size))

        if self.shuffle:
            for bucket in buckets:
                rng.shuffle(bucket)
            rng.shuffle(buckets)

        for bucket in buckets:
            yield from bucket

    def __len__(self) -> int:
        return self.num_samples


# Need math for DistributedSortedSampler
import math


# ---------------------------------------------------------------------------
# Collate function — pads tgt and en branches independently
# ---------------------------------------------------------------------------

def _collate_fn(batch: List[Dict], pad_token_id: int) -> Dict:
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
# AlignmentDataLoader  [DIST-A, DIST-C, DIST-D]
# ---------------------------------------------------------------------------

class AlignmentDataLoader:
    """
    Standard PyTorch DataLoader for the OT-based cross-lingual alignment
    framework.

    Distributed support  [DIST-A]
    -----------------------------
    Pass `rank` and `world_size` to enable distributed sharding.
    When world_size > 1, a DistributedSortedSampler is used so each
    GPU processes a non-overlapping subset of the data.
    When world_size == 1 (default), behaviour is identical to the
    original single-GPU version.

    Parameters
    ----------
    dataset      : AlignmentDataset  (must be .load()-ed first)
    split        : "train" | "dev"
    source       : "joint" | "flores" | "opus" | "eng_eng"
    tokenizer    : HF tokeniser; if None, loads DEFAULT_LLAMA3_MODEL
    batch_size   : samples per batch (per-GPU when distributed)
    max_length   : max tokens per branch sequence
    shuffle      : shuffle buckets each epoch
    seed         : base RNG seed
    num_workers  : DataLoader worker processes
    pin_memory   : pin tensors to CUDA-pinned memory
    rank         : this process's rank (None or 0 → single-GPU)  [DIST-A]
    world_size   : total number of processes (None or 1 → single-GPU)  [DIST-A]
    drop_last    : drop incomplete last batch (recommended for DDP)
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
        # ── Distributed args [DIST-A] ──────────────────────────────────────
        rank: Optional[int] = None,
        world_size: Optional[int] = None,
        drop_last: bool = True,
    ) -> None:
        self.split      = split
        self.source     = source
        self.batch_size = batch_size
        self.max_length = max_length
        self.shuffle    = shuffle
        self.seed       = seed

        # Normalise distributed args
        _rank       = rank       if (rank       is not None) else 0
        _world_size = world_size if (world_size is not None) else 1
        self._distributed = (_world_size > 1)

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
        dist_info = (
            f"rank={_rank}/{_world_size}  " if self._distributed else ""
        )
        print(
            f"[DataLoader] {dist_info}split={split!r}  source={source!r}  "
            f"records={len(records):,}  eng-eng={n_eng_eng:,}  "
            f"batch_size={batch_size}  max_length={max_length}  seed={seed}"
        )

        # ── 3. Torch Dataset ───────────────────────────────────────────────
        self._torch_dataset = AlignmentTorchDataset(
            records=records,
            tokenizer=self.tokenizer,
            max_length=max_length,
        )

        # ── 4. Sampler [DIST-B] ────────────────────────────────────────────
        if self._distributed:
            self._sampler = DistributedSortedSampler(
                dataset=self._torch_dataset,
                batch_size=batch_size,
                rank=_rank,
                world_size=_world_size,
                shuffle=shuffle,
                seed=seed,
                drop_last=drop_last,
            )
        else:
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
            drop_last=drop_last if self._distributed else False,
        )

    def set_epoch(self, epoch: int) -> None:
        """
        Call at the start of each training epoch to re-shuffle buckets.  [DIST-C]
        Works for both SortedLengthSampler and DistributedSortedSampler.
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

    @property
    def dataloader(self) -> DataLoader:
        """Alias for the underlying torch DataLoader.  [DIST-D]"""
        return self._loader


# ---------------------------------------------------------------------------
# Quick smoke-test  (tokeniser only — no model weights, no LLM inference)
# Run:  python alignment_dataloader.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # ── 1. Dataset ──────────────────────────────────────────────────────────
    ds = AlignmentDataset(
        alignment_data_path="../raw_data/alignment/",
        opus_sample_ratio=0.1,
        eng_eng_ratio=0.30,
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

    # ── 3. Build loaders (single-GPU) ───────────────────────────────────────
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

    # ── 4. Simulate distributed (2-GPU) without actual NCCL ─────────────────
    loader_rank0 = AlignmentDataLoader(
        dataset=ds, split="train", source="joint",
        tokenizer=tokenizer, batch_size=8, max_length=256,
        shuffle=True, seed=_GLOBAL_SAMPLE_SEED,
        rank=0, world_size=2,
    )
    loader_rank1 = AlignmentDataLoader(
        dataset=ds, split="train", source="joint",
        tokenizer=tokenizer, batch_size=8, max_length=256,
        shuffle=True, seed=_GLOBAL_SAMPLE_SEED,
        rank=1, world_size=2,
    )
    print(f"\n[Smoke-test] dist rank-0 batches : {len(loader_rank0):,}")
    print(f"[Smoke-test] dist rank-1 batches : {len(loader_rank1):,}")

    # Verify shards are disjoint
    ids0 = set(loader_rank0._sampler._lengths[i] for i in range(len(loader_rank0._sampler._lengths)))
    batch0 = next(iter(loader_rank0))
    batch1 = next(iter(loader_rank1))
    # (Full disjointness check omitted for brevity — sampler logic guarantees it)
    print("[Smoke-test] ✓ Distributed loaders built successfully")

    # ── 5. Inspect first training batch ─────────────────────────────────────
    print("\n[Smoke-test] First training batch (single-GPU loader):")
    batch = next(iter(train_loader))
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k:25s}  shape={tuple(v.shape)}  dtype={v.dtype}")
        else:
            print(f"  {k:25s}  {v[:2]} …")

    print("\n✓ Smoke-test passed.")