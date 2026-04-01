"""
downstream_dataloader.py
========================
Dataset và DataLoader cho zero-shot evaluation trên ba multilingual benchmarks:

    raw_data/downstream/MMMLU/   — Multilingual Multiple Choice QA
    raw_data/downstream/XNLI/    — Cross-lingual Natural Language Inference
    raw_data/downstream/XSQuAD/ — Cross-lingual Extractive QA

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Generation Strategy (MMMLU / XNLI / XSQuAD)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Tất cả 3 tasks đều dùng generation mode:
    - Prompt-only batches (left-padded cho batch generation)
    - model.generate() → decode → parse/normalize output
    - So sánh output với gold label

MMMLU : model phải output đúng "A", "B", "C", hoặc "D"
XNLI  : model phải output đúng "entailment", "neutral", hoặc "contradiction"
XSQuAD: model output câu trả lời tự do → F1 / EM vs gold answers

Batch keys (tất cả tasks)
──────────────────────────
    input_ids       LongTensor [B, L]    — prompt left-padded
    attention_mask  LongTensor [B, L]
    gold_label      List[str]            — ground-truth string label
    task            List[str]            — "mmmlu" / "xnli" / "xsquad"
    lang            List[str]            — language code

    (XSQuAD thêm)
    answers         List[List[str]]      — tất cả gold answers (cho F1/EM)
"""

import json
import random
from pathlib import Path
from typing import Dict, Iterator, List, Optional

import torch
from torch.utils.data import DataLoader, Dataset, Sampler
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "meta-llama/Meta-Llama-3-8B"

XNLI_LABEL_MAP  = {0: "entailment", 1: "neutral", 2: "contradiction"}
XNLI_CANDIDATES = ["entailment", "neutral", "contradiction"]

MCQ_OPTIONS = ["A", "B", "C", "D"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_json(path: Path) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _chunked(lst: list, size: int):
    for i in range(0, len(lst), size):
        yield lst[i : i + size]


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def _build_mmmlu_prompt(record: dict) -> str:
    """
    Prompt MMMLU — model phải trả lời đúng 1 chữ cái: A, B, C hoặc D.
    Prompt được thiết kế để model complete ngay bằng chữ cái đó.
    """
    subject     = record.get("Subject", "general knowledge").replace("_", " ")
    question    = record["Question"].strip()
    options_str = "\n".join(
        f"{letter}. {record[letter]}" for letter in MCQ_OPTIONS if letter in record
    )
    return (
        f"The following is a multiple choice question about {subject}.\n\n"
        f"Question: {question}\n"
        f"{options_str}\n\n"
        f"Answer with only the letter (A, B, C, or D).\n"
        f"Answer:"
    )


def _build_xnli_prompt(record: dict) -> str:
    """
    Prompt XNLI — model phải trả lời đúng 1 từ:
    entailment, neutral, hoặc contradiction.
    """
    premise    = record["premise"].strip()
    hypothesis = record["hypothesis"].strip()
    return (
        f"Determine the logical relationship between the premise and hypothesis.\n"
        f"Answer with only one word: entailment, neutral, or contradiction.\n\n"
        f"Premise: {premise}\n"
        f"Hypothesis: {hypothesis}\n"
        f"Relationship:"
    )


def _build_xsquad_prompt(record: dict) -> str:
    """Prompt XSQuAD — generation tự do."""
    context  = record["context"].strip()
    question = record["question"].strip()
    return (
        f"Read the following passage carefully and answer the question based on it.\n\n"
        f"Passage: {context}\n\n"
        f"Question: {question}\n"
        f"Answer:"
    )


# ---------------------------------------------------------------------------
# Base Generation Dataset  (left-padded, prompt-only)
# ---------------------------------------------------------------------------

class _BaseGenerationDataset(Dataset):
    """
    Prompt-only dataset cho generation mode.

    Mỗi sample chứa:
        input_ids      : [BOS] <prompt tokens>  (left-padded trong collate)
        attention_mask : tương ứng
        gold_label     : str  — ground-truth label để so sánh với output
        task           : str
        lang           : str
    """

    task_name: str = "base"

    def __init__(
        self,
        records: List[dict],
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 512,
    ) -> None:
        self.records    = records
        self.tokenizer  = tokenizer
        self.max_length = max_length
        self.bos = [tokenizer.bos_token_id] if tokenizer.bos_token_id is not None else []

    def _build_prompt(self, record: dict) -> str:
        raise NotImplementedError

    def _encode(self, text: str) -> List[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict:
        record = self.records[idx]
        lang       = record["_lang"]
        gold_label = record["_gold_label"]   # str

        prompt_ids = self._encode(self._build_prompt(record))
        # Truncate prompt từ trái nếu vượt budget
        budget     = self.max_length - len(self.bos)
        prompt_ids = prompt_ids[-max(0, budget):]
        full_ids   = self.bos + prompt_ids

        item = {
            "input_ids":      torch.tensor(full_ids, dtype=torch.long),
            "attention_mask": torch.ones(len(full_ids), dtype=torch.long),
            "gold_label":     gold_label,
            "task":           self.task_name,
            "lang":           lang,
        }

        # XSQuAD cần thêm tất cả gold answers cho F1/EM
        if "answers" in record:
            answers = record["answers"].get("text", [])
            item["answers"] = [a.strip() for a in answers if a.strip()] or ["no answer"]

        return item


# ---------------------------------------------------------------------------
# Left-pad collate (dùng cho tất cả generation tasks)
# ---------------------------------------------------------------------------

def _collate_fn_generation(batch: List[Dict], pad_token_id: int) -> Dict:
    """Left-pad input_ids / attention_mask cho generation batches."""
    max_len = max(s["input_ids"].size(0) for s in batch)
    input_ids_list, mask_list = [], []

    for s in batch:
        n   = s["input_ids"].size(0)
        pad = max_len - n
        input_ids_list.append(torch.cat([
            torch.full((pad,), pad_token_id, dtype=torch.long),
            s["input_ids"],
        ]))
        mask_list.append(torch.cat([
            torch.zeros(pad, dtype=torch.long),
            s["attention_mask"],
        ]))

    out = {
        "input_ids":      torch.stack(input_ids_list),   # [B, L]
        "attention_mask": torch.stack(mask_list),         # [B, L]
        "gold_label":     [s["gold_label"] for s in batch],
        "task":           [s["task"]        for s in batch],
        "lang":           [s["lang"]        for s in batch],
    }

    # Thêm answers nếu có (XSQuAD)
    if "answers" in batch[0]:
        out["answers"] = [s["answers"] for s in batch]

    return out


# ---------------------------------------------------------------------------
# SortedLengthSampler
# ---------------------------------------------------------------------------

class SortedLengthSampler(Sampler):
    """Sort by prompt length để giảm padding trong batch."""

    def __init__(
        self,
        lengths: List[int],
        batch_size: int,
        shuffle: bool = False,
        seed: int = 42,
    ) -> None:
        self._lengths   = lengths
        self.batch_size = batch_size
        self.shuffle    = shuffle
        self.seed       = seed
        self._epoch     = 0

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch

    def __iter__(self) -> Iterator[int]:
        rng     = random.Random(self.seed + self._epoch)
        indices = sorted(range(len(self._lengths)), key=lambda i: self._lengths[i])
        buckets = list(_chunked(indices, self.batch_size))
        if self.shuffle:
            for bucket in buckets:
                rng.shuffle(bucket)
            rng.shuffle(buckets)
        for bucket in buckets:
            yield from bucket

    def __len__(self) -> int:
        return len(self._lengths)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_tokenizer(tok: Optional[PreTrainedTokenizerBase]) -> PreTrainedTokenizerBase:
    if tok is None:
        print(f"[Tokenizer] Loading {DEFAULT_MODEL}")
        tok = AutoTokenizer.from_pretrained(DEFAULT_MODEL, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    return tok


def _build_generation_loader(
    torch_ds: _BaseGenerationDataset,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int,
    pin_memory: bool,
    length_fn,
) -> tuple:
    lengths = [length_fn(r) for r in torch_ds.records]
    sampler = SortedLengthSampler(lengths, batch_size, shuffle, seed)
    pad_id  = torch_ds.tokenizer.pad_token_id
    loader  = DataLoader(
        torch_ds,
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=lambda b: _collate_fn_generation(b, pad_token_id=pad_id),
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    return sampler, loader


# ---------------------------------------------------------------------------
# MMMLU Dataset & DataLoader
# ---------------------------------------------------------------------------

class MMLUDownstreamDataset(_BaseGenerationDataset):
    """
    Generation-mode dataset cho MMMLU.
    gold_label: "A" / "B" / "C" / "D"
    """
    task_name = "mmmlu"

    def _build_prompt(self, record: dict) -> str:
        return _build_mmmlu_prompt(record)


def _load_mmmlu(mmmlu_dir: Path) -> List[dict]:
    records: List[dict] = []
    lang_dirs = sorted(p for p in mmmlu_dir.iterdir() if p.is_dir())
    if not lang_dirs:
        print(f"[MMMLU] WARNING: no language sub-folders found in {mmmlu_dir}")
        return records

    for lang_dir in lang_dirs:
        fpath = lang_dir / "test.json"
        if not fpath.exists():
            print(f"[MMMLU] WARNING: {fpath} not found, skipping.")
            continue
        raw   = _load_json(fpath)
        valid = [
            r for r in raw
            if r.get("Answer", "").strip() in MCQ_OPTIONS and "Question" in r
        ]
        skipped = len(raw) - len(valid)
        if skipped:
            print(f"[MMMLU] {lang_dir.name}: skipped {skipped} malformed records.")
        for r in valid:
            r["_lang"]       = lang_dir.name
            r["_gold_label"] = r["Answer"].strip()   # "A" / "B" / "C" / "D"
        records.extend(valid)
        print(f"[MMMLU] Loaded {len(valid):,} records from {lang_dir.name}/test.json")

    return records


class MMLUDownstreamDataLoader:
    """
    Generation DataLoader cho MMMLU.

    Batch keys
    ----------
    input_ids       LongTensor [B, L]   — prompt left-padded
    attention_mask  LongTensor [B, L]
    gold_label      List[str]           — "A" / "B" / "C" / "D"
    task            List[str]           — ["mmmlu", ...]
    lang            List[str]           — language code
    """

    def __init__(
        self,
        data_root: str = "../raw_data/",
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        batch_size: int = 8,
        max_length: int = 512,
        shuffle: bool = False,
        seed: int = 42,
        num_workers: int = 0,
        pin_memory: bool = False,
    ) -> None:
        mmmlu_dir = Path(data_root) / "downstream" / "MMMLU"
        records   = _load_mmmlu(mmmlu_dir)
        self.tokenizer = _get_tokenizer(tokenizer)
        torch_ds  = MMLUDownstreamDataset(records, self.tokenizer, max_length)
        self._sampler, self._loader = _build_generation_loader(
            torch_ds, batch_size, shuffle, seed, num_workers, pin_memory,
            length_fn=lambda r: len(r.get("Question", "")) + sum(
                len(r.get(k, "")) for k in MCQ_OPTIONS
            ),
        )
        langs = sorted({r["_lang"] for r in records})
        print(
            f"[MMLUDownstreamDataLoader]  samples={len(records):,}  "
            f"batches={len(self._loader):,}  batch_size={batch_size}\n"
            f"  languages ({len(langs)}): {langs}"
        )

    def set_epoch(self, epoch: int) -> None:
        self._sampler.set_epoch(epoch)

    def __iter__(self):
        return iter(self._loader)

    def __len__(self) -> int:
        return len(self._loader)

    @property
    def dataset(self) -> MMLUDownstreamDataset:
        return self._loader.dataset


# ---------------------------------------------------------------------------
# XNLI Dataset & DataLoader
# ---------------------------------------------------------------------------

class XNLIDataset(_BaseGenerationDataset):
    """
    Generation-mode dataset cho XNLI.
    gold_label: "entailment" / "neutral" / "contradiction"
    """
    task_name = "xnli"

    def _build_prompt(self, record: dict) -> str:
        return _build_xnli_prompt(record)


def _load_xnli(xnli_dir: Path) -> List[dict]:
    records: List[dict] = []
    lang_dirs = sorted(p for p in xnli_dir.iterdir() if p.is_dir())
    if not lang_dirs:
        print(f"[XNLI] WARNING: no language sub-folders found in {xnli_dir}")
        return records

    for lang_dir in lang_dirs:
        fpath = lang_dir / "test.json"
        if not fpath.exists():
            print(f"[XNLI] WARNING: {fpath} not found, skipping.")
            continue
        raw     = _load_json(fpath)
        valid   = [r for r in raw if r.get("label") in XNLI_LABEL_MAP]
        skipped = len(raw) - len(valid)
        if skipped:
            print(f"[XNLI] {lang_dir.name}: skipped {skipped} records with label=-1.")
        for r in valid:
            r["_lang"]       = lang_dir.name
            r["_gold_label"] = XNLI_LABEL_MAP[r["label"]]  # "entailment" / "neutral" / "contradiction"
        records.extend(valid)
        print(f"[XNLI] Loaded {len(valid):,} records from {lang_dir.name}/test.json")

    return records


class XNLIDataLoader:
    """
    Generation DataLoader cho XNLI.

    Batch keys
    ----------
    input_ids       LongTensor [B, L]   — prompt left-padded
    attention_mask  LongTensor [B, L]
    gold_label      List[str]           — "entailment" / "neutral" / "contradiction"
    task            List[str]           — ["xnli", ...]
    lang            List[str]           — language code
    """

    def __init__(
        self,
        data_root: str = "../raw_data/",
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        batch_size: int = 16,
        max_length: int = 256,
        shuffle: bool = False,
        seed: int = 42,
        num_workers: int = 0,
        pin_memory: bool = False,
    ) -> None:
        xnli_dir = Path(data_root) / "downstream" / "XNLI"
        records  = _load_xnli(xnli_dir)
        self.tokenizer = _get_tokenizer(tokenizer)
        torch_ds = XNLIDataset(records, self.tokenizer, max_length)
        self._sampler, self._loader = _build_generation_loader(
            torch_ds, batch_size, shuffle, seed, num_workers, pin_memory,
            length_fn=lambda r: len(r.get("premise", "")) + len(r.get("hypothesis", "")),
        )
        langs = sorted({r["_lang"] for r in records})
        print(
            f"[XNLIDataLoader]  samples={len(records):,}  "
            f"batches={len(self._loader):,}  batch_size={batch_size}\n"
            f"  languages ({len(langs)}): {langs}"
        )

    def set_epoch(self, epoch: int) -> None:
        self._sampler.set_epoch(epoch)

    def __iter__(self):
        return iter(self._loader)

    def __len__(self) -> int:
        return len(self._loader)

    @property
    def dataset(self) -> XNLIDataset:
        return self._loader.dataset


# ---------------------------------------------------------------------------
# XSQuAD Dataset & DataLoader
# ---------------------------------------------------------------------------

class XSQuADDataset(_BaseGenerationDataset):
    """Generation-mode dataset cho XSQuAD."""
    task_name = "xsquad"

    def _build_prompt(self, record: dict) -> str:
        return _build_xsquad_prompt(record)


def _load_xsquad(xsquad_dir: Path) -> List[dict]:
    records: List[dict] = []
    lang_dirs = sorted(p for p in xsquad_dir.iterdir() if p.is_dir())
    if not lang_dirs:
        print(f"[XSQuAD] WARNING: no language sub-folders found in {xsquad_dir}")
        return records
    for lang_dir in lang_dirs:
        fpath = lang_dir / "validation.json"
        if not fpath.exists():
            print(f"[XSQuAD] WARNING: {fpath} not found, skipping.")
            continue
        raw          = _load_json(fpath)
        valid        = [r for r in raw if "answers" in r and "context" in r and "question" in r]
        unanswerable = sum(1 for r in valid if not r["answers"].get("text"))
        skipped      = len(raw) - len(valid)
        if skipped:
            print(f"[XSQuAD] {lang_dir.name}: skipped {skipped} malformed records.")
        for r in valid:
            r["_lang"]       = lang_dir.name
            r["_gold_label"] = (r["answers"].get("text") or ["no answer"])[0].strip()
        records.extend(valid)
        print(
            f"[XSQuAD] Loaded {len(valid):,} from {lang_dir.name}/validation.json "
            f"(answerable={len(valid)-unanswerable:,}, unanswerable={unanswerable:,})"
        )
    return records


class XSQuADDataLoader:
    """Generation DataLoader cho XSQuAD."""

    def __init__(
        self,
        data_root: str = "../raw_data/",
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        batch_size: int = 4,
        max_length: int = 1024,
        shuffle: bool = False,
        seed: int = 42,
        num_workers: int = 0,
        pin_memory: bool = False,
    ) -> None:
        xsquad_dir = Path(data_root) / "downstream" / "XSQuAD"
        records    = _load_xsquad(xsquad_dir)
        self.tokenizer = _get_tokenizer(tokenizer)
        torch_ds   = XSQuADDataset(records, self.tokenizer, max_length)
        self._sampler, self._loader = _build_generation_loader(
            torch_ds, batch_size, shuffle, seed, num_workers, pin_memory,
            length_fn=lambda r: len(r.get("context", "")),
        )
        langs = sorted({r["_lang"] for r in records})
        print(
            f"[XSQuADDataLoader]  records={len(records):,}  "
            f"batches={len(self._loader):,}  batch_size={batch_size}\n"
            f"  languages ({len(langs)}): {langs}"
        )

    def set_epoch(self, epoch: int) -> None:
        self._sampler.set_epoch(epoch)

    def __iter__(self):
        return iter(self._loader)

    def __len__(self) -> int:
        return len(self._loader)

    @property
    def dataset(self) -> XSQuADDataset:
        return self._loader.dataset


# ---------------------------------------------------------------------------
# Smoke-test
# Run: python downstream_dataloader.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    from huggingface_hub import login

    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        login(token=hf_token)

    DATA_ROOT = "../raw_data/"

    print("=" * 60)
    print("Loading shared tokenizer ...")
    print("=" * 60)
    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    print(f"  vocab={tokenizer.vocab_size:,}  bos={tokenizer.bos_token_id}  "
          f"eos={tokenizer.eos_token_id}  pad={tokenizer.pad_token_id}\n")

    loaders = {
        "MMMLU":  MMLUDownstreamDataLoader(data_root=DATA_ROOT, tokenizer=tokenizer, batch_size=8),
        "XNLI":   XNLIDataLoader(data_root=DATA_ROOT, tokenizer=tokenizer, batch_size=16),
        "XSQuAD": XSQuADDataLoader(data_root=DATA_ROOT, tokenizer=tokenizer, batch_size=4),
    }

    print("\n" + "=" * 60)
    print("Smoke-tests: MMMLU / XNLI / XSQuAD (generation mode)")
    print("=" * 60)

    for name, loader in loaders.items():
        batch = next(iter(loader))

        assert batch["input_ids"].dim()      == 2
        assert batch["attention_mask"].dim() == 2
        assert batch["input_ids"].dtype      == torch.long
        assert "labels"     not in batch,  f"{name}: should NOT have labels"
        assert "gold_label" in batch,      f"{name}: missing gold_label"
        assert isinstance(batch["gold_label"], list)
        assert isinstance(batch["gold_label"][0], str)
        assert "lang"       in batch
        assert "task"       in batch

        # Kiểm tra gold_label hợp lệ
        if name == "MMMLU":
            assert all(g in MCQ_OPTIONS for g in batch["gold_label"]), \
                f"MMMLU: gold_label phải trong {MCQ_OPTIONS}"
        elif name == "XNLI":
            assert all(g in XNLI_CANDIDATES for g in batch["gold_label"]), \
                f"XNLI: gold_label phải trong {XNLI_CANDIDATES}"
        elif name == "XSQuAD":
            assert "answers" in batch, "XSQuAD: missing answers"
            assert all(isinstance(a, list) for a in batch["answers"])

        # In prompt đầu tiên
        ids_0  = batch["input_ids"][0]
        mask_0 = batch["attention_mask"][0]
        prompt = tokenizer.decode(ids_0[mask_0 == 1], skip_special_tokens=True)

        print(f"\n  ✓ {name}  shape={tuple(batch['input_ids'].shape)}  "
              f"task={batch['task'][0]}  langs={set(batch['lang'])}")
        print(f"  gold_label[0] = '{batch['gold_label'][0]}'")
        print(f"  PROMPT (trunc 300 chars):\n  {prompt[:300]!r}")

    print("\n" + "=" * 60)
    print("✓ All smoke-tests passed.")
    print("=" * 60)