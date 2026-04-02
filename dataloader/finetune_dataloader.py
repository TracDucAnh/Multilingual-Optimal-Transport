"""
finetune_dataloader.py
======================
Dataset và DataLoader cho supervised fine-tuning (SFT) của LLMs trên
ba English benchmark datasets, sử dụng GENERATE MODE với L_LM loss:

    raw_data/english/MMLU/   — Multiple Choice QA
    raw_data/english/SQuAD/  — Extractive QA
    raw_data/english/SNLI/   — Natural Language Inference

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Generate Mode + L_LM Loss
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Tất cả 3 task đều dùng Instruct chat template (meta-llama/Meta-Llama-3-8B-Instruct).

SFT format (causal LM, L_LM = loss trên TOÀN BỘ sequence):
    [BOS] <system_turn> <user_turn> <assistant_turn: answer> [EOS]
    labels: -100 trên system+user turns, loss trên assistant answer + EOS

    Lý do mask system+user: chúng là prompt cố định, không muốn model
    "học thuộc" lại prompt — chỉ học mapping prompt → answer đúng.
    L_LM = CrossEntropy(shift_logits, shift_labels, ignore_index=-100)

Validation — Generate Mode:
    - Trả về prompt-only (left-padded) để gọi model.generate()
    - Parse output text → so sánh với gold label
    - MMLU / SNLI: Accuracy
    - SQuAD: F1 + Exact Match

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SFT Batch keys (tất cả tasks):
    input_ids       LongTensor [B, L]   full sequence (right-padded)
    attention_mask  LongTensor [B, L]
    labels          LongTensor [B, L]   -100 trên prompt; answer tokens + EOS

Val Batch keys (MMLU / SNLI):
    input_ids       LongTensor [B, L]   prompt only (left-padded)
    attention_mask  LongTensor [B, L]
    gold_label      List[str]           ground-truth answer string
    task            List[str]

Val Batch keys (SQuAD):
    input_ids       LongTensor [B, L]   prompt only (left-padded)
    attention_mask  LongTensor [B, L]
    gold_label      List[str]           first gold answer (for reference)
    answers         List[List[str]]     ALL gold answers (for F1/EM)
    task            List[str]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Mixed DataLoader — balanced batching (unchanged strategy)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MixedFinetuneDataLoader kết hợp 3 task với balanced per-batch mixing:
  1. n_balanced = min(len(mmlu), len(squad), len(snli)), làm tròn xuống
     đến bội số gần nhất của per_task = batch_size // 3
  2. Balanced pool: interleave [mmlu, squad, snli, ...]; mỗi mixed batch có
     per_task samples từ mỗi task.
  3. Remainder pools: single-task batches với kích thước batch_size.
  4. Thứ tự yield: mixed batches trước, sau đó remainder batches theo task.
"""

import json
import random
import re
import string
from collections import Counter
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, Sampler
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"

SNLI_LABEL_MAP   = {0: "entailment", 1: "neutral", 2: "contradiction"}
SNLI_CANDIDATES  = ["entailment", "neutral", "contradiction"]

MMLU_OPTIONS     = ["A", "B", "C", "D"]

# System prompt ngắn gọn, rõ ràng cho Instruct model
_SYSTEM_PROMPT = (
    "You are a precise answer extraction assistant. "
    "Always respond with only the answer and nothing else. "
    "Do not explain, do not repeat the question."
)


# ---------------------------------------------------------------------------
# Helpers — data loading
# ---------------------------------------------------------------------------

def _load_json(path: Path) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _chunked(lst: list, size: int):
    for i in range(0, len(lst), size):
        yield lst[i : i + size]


# ---------------------------------------------------------------------------
# MMLU normalisation
# ---------------------------------------------------------------------------

def _normalise_mmlu_record(record: dict) -> dict:
    """
    Normalise raw MMLU record in-place:
        record["answer"]  : int  →  str  e.g. 1 → "B"
        record["choices"] : plain → prefixed  e.g. ["A. Paris", "B. Rome", ...]
    """
    idx = record["answer"]
    record["answer"]  = MMLU_OPTIONS[idx]
    record["choices"] = [
        f"{MMLU_OPTIONS[i]}. {record['choices'][i]}"
        for i in range(len(record["choices"]))
    ]
    return record


# ---------------------------------------------------------------------------
# Prompt builders — trả về (user_message_str, answer_str)
# ---------------------------------------------------------------------------

def _build_mmlu_user_message(record: dict) -> str:
    """User message cho MMLU — yêu cầu trả lời đúng 1 ký tự A/B/C/D."""
    subject     = record.get("subject", "general knowledge").replace("_", " ")
    question    = record["question"].strip()
    options_str = "\n".join(record["choices"])   # đã prefix bởi normalisation
    return (
        f"The following is a multiple choice question about {subject}.\n\n"
        f"Question: {question}\n"
        f"{options_str}\n\n"
        f"Reply with only the single letter of the correct answer (A, B, C, or D). "
        f"Do not include any other text."
    )


def _build_mmlu_answer(record: dict) -> str:
    """Answer string cho MMLU: " A" / " B" / " C" / " D"."""
    return record["answer"]   # e.g. "A"


def _build_squad_user_message(record: dict) -> str:
    """User message cho SQuAD — trả lời ngắn gọn, đúng span."""
    context  = record["context"].strip()
    question = record["question"].strip()
    return (
        f"Read the passage below and answer the question with a short phrase or span "
        f"taken directly from the passage. Do not include any other text.\n\n"
        f"Passage: {context}\n\n"
        f"Question: {question}"
    )


def _build_squad_answer(record: dict) -> str:
    """Answer string cho SQuAD: lấy answer đầu tiên."""
    answers = record["answers"]
    return answers["text"][0].strip() if answers.get("text") else "no answer"


def _build_snli_user_message(record: dict) -> str:
    """User message cho SNLI — trả lời đúng 1 từ: entailment/neutral/contradiction."""
    return (
        f"Determine the logical relationship between the following premise and hypothesis.\n\n"
        f"Premise: {record['premise'].strip()}\n"
        f"Hypothesis: {record['hypothesis'].strip()}\n\n"
        f"Reply with exactly one word — either 'entailment', 'neutral', or 'contradiction'. "
        f"Do not include any other text."
    )


def _build_snli_answer(record: dict) -> str:
    """Answer string cho SNLI: 'entailment' / 'neutral' / 'contradiction'."""
    return SNLI_LABEL_MAP[record["label"]]


# ---------------------------------------------------------------------------
# Chat-template encoder
# ---------------------------------------------------------------------------

def _apply_chat_template_sft(
    tokenizer: PreTrainedTokenizerBase,
    user_message: str,
    answer_str: str,
    max_length: int,
) -> Dict[str, torch.Tensor]:
    """
    Tạo SFT sample với Llama-3-Instruct chat template.

    Layout token:
        [BOS] <|system|> ... <|user|> ... <|assistant|> <answer_tokens> [EOS]
        labels: -100 trên system+user+assistant_header; answer_tokens + EOS có loss.

    Truncation: nếu quá dài, cắt từ trái của user message (giữ answer nguyên vẹn).

    Trả về:
        input_ids      LongTensor [L]
        attention_mask LongTensor [L]
        labels         LongTensor [L]   -100 trên prompt, answer+EOS có loss
    """
    # 1. Build messages list (không có assistant turn để tách prompt / answer rõ ràng)
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user",   "content": user_message},
    ]

    # 2. Tokenize prompt (với add_generation_prompt=True → kết thúc bằng <|start_header_id|>assistant<|end_header_id|>\n\n)
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)

    # 3. Tokenize answer + EOS
    answer_ids = tokenizer.encode(answer_str, add_special_tokens=False)
    eos_ids    = [tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else []

    # 4. Truncate prompt từ trái nếu cần (giữ nguyên answer)
    max_prompt_len = max_length - len(answer_ids) - len(eos_ids)
    if max_prompt_len <= 0:
        # answer quá dài (hiếm xảy ra), truncate answer
        answer_ids = answer_ids[:max_length - len(eos_ids) - 1]
        max_prompt_len = 1
    prompt_ids = prompt_ids[-max_prompt_len:]

    # 5. Ghép full sequence
    full_ids = prompt_ids + answer_ids + eos_ids

    # 6. Labels: -100 trên prompt, answer + EOS có loss
    prompt_len = len(prompt_ids)
    labels     = [-100] * prompt_len + answer_ids + eos_ids

    assert len(full_ids) == len(labels), (
        f"Length mismatch: full_ids={len(full_ids)}, labels={len(labels)}"
    )

    return {
        "input_ids":      torch.tensor(full_ids, dtype=torch.long),
        "attention_mask": torch.ones(len(full_ids), dtype=torch.long),
        "labels":         torch.tensor(labels,    dtype=torch.long),
    }


def _apply_chat_template_val(
    tokenizer: PreTrainedTokenizerBase,
    user_message: str,
    max_length: int,
) -> Dict[str, torch.Tensor]:
    """
    Tạo prompt-only sample để dùng với model.generate().

    Trả về:
        input_ids      LongTensor [L]   prompt only (chưa left-pad)
        attention_mask LongTensor [L]
    """
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user",   "content": user_message},
    ]
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    encoded = tokenizer(
        prompt_text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
        padding=False,
        return_tensors="pt",
    )
    return {
        "input_ids":      encoded["input_ids"].squeeze(0),
        "attention_mask": encoded["attention_mask"].squeeze(0),
    }


# ---------------------------------------------------------------------------
# Base SFT Dataset
# ---------------------------------------------------------------------------

class _BaseSFTDataset(Dataset):
    """
    SFT dataset với Instruct chat template + L_LM loss trên answer tokens.

    Subclass cần implement:
        _build_user_message(record) → str
        _build_answer(record)       → str
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

    def _build_user_message(self, record: dict) -> str:
        raise NotImplementedError

    def _build_answer(self, record: dict) -> str:
        raise NotImplementedError

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict:
        record      = self.records[idx]
        user_msg    = self._build_user_message(record)
        answer_str  = self._build_answer(record)
        encoded     = _apply_chat_template_sft(
            self.tokenizer, user_msg, answer_str, self.max_length
        )
        return {
            "input_ids":      encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "labels":         encoded["labels"],
            "task":           self.task_name,
        }


# ---------------------------------------------------------------------------
# Base Val Dataset (generate mode)
# ---------------------------------------------------------------------------

class _BaseValDataset(Dataset):
    """
    Validation dataset: prompt-only, left-pad trong collate.
    Dùng với model.generate() → parse output → so sánh gold.

    Subclass cần implement:
        _build_user_message(record) → str
        _get_gold_label(record)     → str
    """

    task_name: str = "base_val"

    def __init__(
        self,
        records: List[dict],
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 512,
    ) -> None:
        self.records    = records
        self.tokenizer  = tokenizer
        self.max_length = max_length

    def _build_user_message(self, record: dict) -> str:
        raise NotImplementedError

    def _get_gold_label(self, record: dict) -> str:
        raise NotImplementedError

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict:
        record     = self.records[idx]
        user_msg   = self._build_user_message(record)
        gold_label = self._get_gold_label(record)
        encoded    = _apply_chat_template_val(
            self.tokenizer, user_msg, self.max_length
        )
        item = {
            "input_ids":      encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "gold_label":     gold_label,
            "task":           self.task_name,
        }
        return item


# ---------------------------------------------------------------------------
# Collate functions
# ---------------------------------------------------------------------------

def _collate_fn_sft(batch: List[Dict], pad_token_id: int) -> Dict:
    """Right-pad collate cho SFT batches."""
    max_len = max(s["input_ids"].size(0) for s in batch)
    input_ids_list, mask_list, labels_list = [], [], []
    for s in batch:
        n   = s["input_ids"].size(0)
        pad = max_len - n
        input_ids_list.append(torch.cat([
            s["input_ids"],
            torch.full((pad,), pad_token_id, dtype=torch.long),
        ]))
        mask_list.append(torch.cat([
            s["attention_mask"],
            torch.zeros(pad, dtype=torch.long),
        ]))
        labels_list.append(torch.cat([
            s["labels"],
            torch.full((pad,), -100, dtype=torch.long),
        ]))
    return {
        "input_ids":      torch.stack(input_ids_list),
        "attention_mask": torch.stack(mask_list),
        "labels":         torch.stack(labels_list),
        "task":           [s["task"] for s in batch],
    }


def _collate_fn_val(batch: List[Dict], pad_token_id: int) -> Dict:
    """Left-pad collate cho generation/val batches."""
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
        "input_ids":      torch.stack(input_ids_list),
        "attention_mask": torch.stack(mask_list),
        "gold_label":     [s["gold_label"] for s in batch],
        "task":           [s["task"]        for s in batch],
    }
    # XSQuAD / SQuAD thêm all gold answers
    if "answers" in batch[0]:
        out["answers"] = [s["answers"] for s in batch]
    return out


# ---------------------------------------------------------------------------
# SortedLengthSampler
# ---------------------------------------------------------------------------

class SortedLengthSampler(Sampler):
    """
    Bucket-based sampler: sort by proxy length → group into buckets of batch_size
    → optionally shuffle within/across buckets. Minimises intra-batch padding.
    """

    def __init__(
        self,
        lengths: List[int],
        batch_size: int,
        shuffle: bool = True,
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
    # SFT dùng right-pad; val dùng left-pad → set trong từng loader
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    return tok


def _build_sft_loader(
    torch_ds: _BaseSFTDataset,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int,
    pin_memory: bool,
    length_fn,
) -> Tuple[SortedLengthSampler, DataLoader]:
    lengths = [
        length_fn(rec)
        for rec in tqdm(torch_ds.records, desc="[Sampler] lengths", leave=False)
    ]
    sampler = SortedLengthSampler(lengths, batch_size, shuffle, seed)
    pad_id  = torch_ds.tokenizer.pad_token_id
    loader  = DataLoader(
        torch_ds,
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=lambda b: _collate_fn_sft(b, pad_token_id=pad_id),
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    return sampler, loader


def _build_val_loader(
    torch_ds: _BaseValDataset,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int,
    pin_memory: bool,
    length_fn,
) -> Tuple[SortedLengthSampler, DataLoader]:
    lengths = [
        length_fn(rec)
        for rec in tqdm(torch_ds.records, desc="[Sampler] lengths", leave=False)
    ]
    sampler = SortedLengthSampler(lengths, batch_size, shuffle, seed)
    pad_id  = torch_ds.tokenizer.pad_token_id
    loader  = DataLoader(
        torch_ds,
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=lambda b: _collate_fn_val(b, pad_token_id=pad_id),
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    return sampler, loader


def _count_batches(n_records: int, batch_size: int) -> int:
    if n_records == 0:
        return 0
    return (n_records + batch_size - 1) // batch_size


# ---------------------------------------------------------------------------
# MMLU — data loading
# ---------------------------------------------------------------------------

def _load_mmlu(mmlu_dir: Path, splits: List[str]) -> List[dict]:
    split_to_file = {
        "auxiliary_train": "auxiliary_train.json",
        "dev":             "dev.json",
        "test":            "test.json",
        "validation":      "validation.json",
    }
    records: List[dict] = []
    for split in splits:
        fname = split_to_file.get(split)
        if fname is None:
            raise ValueError(f"Unknown MMLU split '{split}'.")
        fpath = mmlu_dir / fname
        if not fpath.exists():
            print(f"[MMLU] WARNING: {fpath} not found, skipping.")
            continue
        raw   = _load_json(fpath)
        valid = [
            r for r in raw
            if isinstance(r.get("answer"), int)
            and 0 <= r["answer"] < len(r.get("choices", []))
        ]
        skipped = len(raw) - len(valid)
        if skipped:
            print(f"[MMLU] {fpath.name}: skipped {skipped} records with invalid answer index.")
        valid = [_normalise_mmlu_record(r) for r in valid]
        records.extend(valid)
        print(f"[MMLU] Loaded {len(valid):,} records from {fpath.name}")
    return records


# ---------------------------------------------------------------------------
# MMLU — SFT Dataset / DataLoader
# ---------------------------------------------------------------------------

class MMLUDataset(_BaseSFTDataset):
    task_name = "mmlu"

    def _build_user_message(self, record: dict) -> str:
        return _build_mmlu_user_message(record)

    def _build_answer(self, record: dict) -> str:
        return _build_mmlu_answer(record)


class MMLUDataLoader:
    """
    SFT DataLoader cho MMLU — Instruct chat template, L_LM loss trên answer token.

    Batch keys
    ----------
    input_ids / attention_mask / labels  LongTensor [B, L]
    task                                 List[str]  ["mmlu", ...]
    """

    def __init__(
        self,
        data_root: str = "../raw_data/english/",
        splits: List[str] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        batch_size: int = 8,
        max_length: int = 512,
        shuffle: bool = True,
        seed: int = 42,
        num_workers: int = 0,
        pin_memory: bool = False,
    ) -> None:
        if splits is None:
            splits = ["auxiliary_train"]
        mmlu_dir = Path(data_root) / "MMLU"
        records  = _load_mmlu(mmlu_dir, splits)
        self.tokenizer = _get_tokenizer(tokenizer)
        self.tokenizer.padding_side = "right"
        torch_ds = MMLUDataset(records, self.tokenizer, max_length)
        self._sampler, self._loader = _build_sft_loader(
            torch_ds, batch_size, shuffle, seed, num_workers, pin_memory,
            length_fn=lambda r: len(r["question"]) + sum(len(c) for c in r.get("choices", [])),
        )
        print(f"[MMLUDataLoader] splits={splits}  records={len(records):,}  "
              f"batches={len(self._loader):,}  batch_size={batch_size}")

    def set_epoch(self, epoch: int) -> None: self._sampler.set_epoch(epoch)
    def __iter__(self): return iter(self._loader)
    def __len__(self) -> int: return len(self._loader)

    @property
    def dataset(self) -> MMLUDataset:
        return self._loader.dataset


# ---------------------------------------------------------------------------
# MMLU — Validation Dataset / DataLoader (generate mode)
# ---------------------------------------------------------------------------

class MMLUValDataset(_BaseValDataset):
    """
    Val dataset cho MMLU — generate mode.
    gold_label: "A" / "B" / "C" / "D"
    """
    task_name = "mmlu"

    def _build_user_message(self, record: dict) -> str:
        return _build_mmlu_user_message(record)

    def _get_gold_label(self, record: dict) -> str:
        return record["answer"]   # đã normalise → "A"/"B"/"C"/"D"


class MMLUValDataLoader:
    """
    Validation DataLoader cho MMLU — generate mode.

    Caller gọi model.generate(), decode, parse letter đầu tiên,
    so sánh với gold_label để tính Accuracy.

    Batch keys
    ----------
    input_ids       LongTensor [B, L]   prompt only (left-padded)
    attention_mask  LongTensor [B, L]
    gold_label      List[str]           "A" / "B" / "C" / "D"
    task            List[str]           ["mmlu", ...]
    """

    def __init__(
        self,
        data_root: str = "../raw_data/english/",
        splits: List[str] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        batch_size: int = 8,
        max_length: int = 512,
        shuffle: bool = False,
        seed: int = 42,
        num_workers: int = 0,
        pin_memory: bool = False,
    ) -> None:
        if splits is None:
            splits = ["validation"]
        mmlu_dir = Path(data_root) / "MMLU"
        records  = _load_mmlu(mmlu_dir, splits)
        self.tokenizer = _get_tokenizer(tokenizer)
        self.tokenizer.padding_side = "left"
        torch_ds = MMLUValDataset(records, self.tokenizer, max_length)
        self._sampler, self._loader = _build_val_loader(
            torch_ds, batch_size, shuffle, seed, num_workers, pin_memory,
            length_fn=lambda r: len(r["question"]) + sum(len(c) for c in r.get("choices", [])),
        )
        print(f"[MMLUValDataLoader] splits={splits}  samples={len(records):,}  "
              f"batches={len(self._loader):,}  batch_size={batch_size}")

    def set_epoch(self, epoch: int) -> None: self._sampler.set_epoch(epoch)
    def __iter__(self): return iter(self._loader)
    def __len__(self) -> int: return len(self._loader)

    @property
    def dataset(self) -> MMLUValDataset:
        return self._loader.dataset


# ---------------------------------------------------------------------------
# SQuAD — data loading
# ---------------------------------------------------------------------------

def _load_squad(squad_dir: Path, splits: List[str]) -> List[dict]:
    split_to_file = {"train": "train.json", "validation": "validation.json"}
    records: List[dict] = []
    for split in splits:
        fname = split_to_file.get(split)
        if fname is None:
            raise ValueError(f"Unknown SQuAD split '{split}'.")
        fpath = squad_dir / fname
        if not fpath.exists():
            print(f"[SQuAD] WARNING: {fpath} not found, skipping.")
            continue
        raw          = _load_json(fpath)
        valid        = [r for r in raw if "answers" in r and "context" in r and "question" in r]
        unanswerable = sum(1 for r in valid if not r["answers"].get("text"))
        skipped      = len(raw) - len(valid)
        if skipped:
            print(f"[SQuAD] {fpath.name}: skipped {skipped} malformed records.")
        print(f"[SQuAD] Loaded {len(valid):,} records from {fpath.name} "
              f"(answerable={len(valid)-unanswerable:,}, unanswerable={unanswerable:,})")
        records.extend(valid)
    return records


# ---------------------------------------------------------------------------
# SQuAD — SFT Dataset / DataLoader
# ---------------------------------------------------------------------------

class SQuADDataset(_BaseSFTDataset):
    task_name = "squad"

    def _build_user_message(self, record: dict) -> str:
        return _build_squad_user_message(record)

    def _build_answer(self, record: dict) -> str:
        return _build_squad_answer(record)


class SQuADDataLoader:
    """
    SFT DataLoader cho SQuAD — Instruct chat template, L_LM loss trên answer span.

    Batch keys
    ----------
    input_ids / attention_mask / labels  LongTensor [B, L]
    task                                 List[str]  ["squad", ...]
    """

    def __init__(
        self,
        data_root: str = "../raw_data/english/",
        splits: List[str] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        batch_size: int = 4,
        max_length: int = 1024,
        shuffle: bool = True,
        seed: int = 42,
        num_workers: int = 0,
        pin_memory: bool = False,
    ) -> None:
        if splits is None:
            splits = ["train"]
        squad_dir = Path(data_root) / "SQuAD"
        records   = _load_squad(squad_dir, splits)
        self.tokenizer = _get_tokenizer(tokenizer)
        self.tokenizer.padding_side = "right"
        torch_ds = SQuADDataset(records, self.tokenizer, max_length)
        self._sampler, self._loader = _build_sft_loader(
            torch_ds, batch_size, shuffle, seed, num_workers, pin_memory,
            length_fn=lambda r: len(r["context"]),
        )
        print(f"[SQuADDataLoader] splits={splits}  records={len(records):,}  "
              f"batches={len(self._loader):,}  batch_size={batch_size}")

    def set_epoch(self, epoch: int) -> None: self._sampler.set_epoch(epoch)
    def __iter__(self): return iter(self._loader)
    def __len__(self) -> int: return len(self._loader)

    @property
    def dataset(self) -> SQuADDataset:
        return self._loader.dataset


# ---------------------------------------------------------------------------
# SQuAD — Validation Dataset / DataLoader (generate mode)
# ---------------------------------------------------------------------------

class SQuADValDataset(_BaseValDataset):
    """
    Val dataset cho SQuAD — generate mode.
    gold_label: answer đầu tiên (string)
    answers:    ALL gold answers (List[str]) cho F1/EM
    """
    task_name = "squad"

    def _build_user_message(self, record: dict) -> str:
        return _build_squad_user_message(record)

    def _get_gold_label(self, record: dict) -> str:
        answers = record["answers"].get("text", [])
        return answers[0].strip() if answers else "no answer"

    def __getitem__(self, idx: int) -> Dict:
        item    = super().__getitem__(idx)
        record  = self.records[idx]
        answers = record["answers"].get("text", [])
        item["answers"] = [a.strip() for a in answers if a.strip()] or ["no answer"]
        return item


class SQuADValDataLoader:
    """
    Validation DataLoader cho SQuAD — generate mode.

    Caller gọi model.generate(), decode new tokens, score vs answers với F1/EM.

    Batch keys
    ----------
    input_ids       LongTensor [B, L]   prompt only (left-padded)
    attention_mask  LongTensor [B, L]
    gold_label      List[str]           first gold answer
    answers         List[List[str]]     ALL gold answers (cho F1/EM)
    task            List[str]           ["squad", ...]
    """

    def __init__(
        self,
        data_root: str = "../raw_data/english/",
        splits: List[str] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        batch_size: int = 4,
        max_length: int = 1024,
        shuffle: bool = False,
        seed: int = 42,
        num_workers: int = 0,
        pin_memory: bool = False,
    ) -> None:
        if splits is None:
            splits = ["validation"]
        squad_dir = Path(data_root) / "SQuAD"
        records   = _load_squad(squad_dir, splits)
        self.tokenizer = _get_tokenizer(tokenizer)
        self.tokenizer.padding_side = "left"
        torch_ds  = SQuADValDataset(records, self.tokenizer, max_length)
        self._sampler, self._loader = _build_val_loader(
            torch_ds, batch_size, shuffle, seed, num_workers, pin_memory,
            length_fn=lambda r: len(r.get("context", "")),
        )
        print(f"[SQuADValDataLoader] splits={splits}  records={len(records):,}  "
              f"batches={len(self._loader):,}  batch_size={batch_size}")

    def set_epoch(self, epoch: int) -> None: self._sampler.set_epoch(epoch)
    def __iter__(self): return iter(self._loader)
    def __len__(self) -> int: return len(self._loader)

    @property
    def dataset(self) -> SQuADValDataset:
        return self._loader.dataset


# Alias cho backward compat
SQuADValidationDataLoader = SQuADValDataLoader


# ---------------------------------------------------------------------------
# SNLI — data loading
# ---------------------------------------------------------------------------

def _load_snli(snli_dir: Path, splits: List[str]) -> List[dict]:
    split_to_file = {
        "train": "train.json", "test": "test.json", "validation": "validation.json"
    }
    records: List[dict] = []
    for split in splits:
        fname = split_to_file.get(split)
        if fname is None:
            raise ValueError(f"Unknown SNLI split '{split}'.")
        fpath = snli_dir / fname
        if not fpath.exists():
            print(f"[SNLI] WARNING: {fpath} not found, skipping.")
            continue
        raw     = _load_json(fpath)
        valid   = [r for r in raw if r.get("label") in SNLI_LABEL_MAP]
        skipped = len(raw) - len(valid)
        if skipped:
            print(f"[SNLI] {fpath.name}: skipped {skipped} records with label=-1.")
        records.extend(valid)
        print(f"[SNLI] Loaded {len(valid):,} records from {fpath.name}")
    return records


# ---------------------------------------------------------------------------
# SNLI — SFT Dataset / DataLoader
# ---------------------------------------------------------------------------

class SNLIDataset(_BaseSFTDataset):
    task_name = "snli"

    def _build_user_message(self, record: dict) -> str:
        return _build_snli_user_message(record)

    def _build_answer(self, record: dict) -> str:
        return _build_snli_answer(record)


class SNLIDataLoader:
    """
    SFT DataLoader cho SNLI — Instruct chat template, L_LM loss trên label word.

    Batch keys
    ----------
    input_ids / attention_mask / labels  LongTensor [B, L]
    task                                 List[str]  ["snli", ...]
    """

    def __init__(
        self,
        data_root: str = "../raw_data/english/",
        splits: List[str] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        batch_size: int = 16,
        max_length: int = 256,
        shuffle: bool = True,
        seed: int = 42,
        num_workers: int = 0,
        pin_memory: bool = False,
    ) -> None:
        if splits is None:
            splits = ["train"]
        snli_dir = Path(data_root) / "SNLI"
        records  = _load_snli(snli_dir, splits)
        self.tokenizer = _get_tokenizer(tokenizer)
        self.tokenizer.padding_side = "right"
        torch_ds = SNLIDataset(records, self.tokenizer, max_length)
        self._sampler, self._loader = _build_sft_loader(
            torch_ds, batch_size, shuffle, seed, num_workers, pin_memory,
            length_fn=lambda r: len(r["premise"]) + len(r["hypothesis"]),
        )
        print(f"[SNLIDataLoader] splits={splits}  records={len(records):,}  "
              f"batches={len(self._loader):,}  batch_size={batch_size}")

    def set_epoch(self, epoch: int) -> None: self._sampler.set_epoch(epoch)
    def __iter__(self): return iter(self._loader)
    def __len__(self) -> int: return len(self._loader)

    @property
    def dataset(self) -> SNLIDataset:
        return self._loader.dataset


# ---------------------------------------------------------------------------
# SNLI — Validation Dataset / DataLoader (generate mode)
# ---------------------------------------------------------------------------

class SNLIValDataset(_BaseValDataset):
    """
    Val dataset cho SNLI — generate mode.
    gold_label: "entailment" / "neutral" / "contradiction"
    """
    task_name = "snli"

    def _build_user_message(self, record: dict) -> str:
        return _build_snli_user_message(record)

    def _get_gold_label(self, record: dict) -> str:
        return SNLI_LABEL_MAP[record["label"]]


class SNLIValDataLoader:
    """
    Validation DataLoader cho SNLI — generate mode.

    Caller gọi model.generate(), decode, parse label word,
    so sánh với gold_label để tính Accuracy.

    Batch keys
    ----------
    input_ids       LongTensor [B, L]   prompt only (left-padded)
    attention_mask  LongTensor [B, L]
    gold_label      List[str]           "entailment"/"neutral"/"contradiction"
    task            List[str]           ["snli", ...]
    """

    def __init__(
        self,
        data_root: str = "../raw_data/english/",
        splits: List[str] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        batch_size: int = 16,
        max_length: int = 256,
        shuffle: bool = False,
        seed: int = 42,
        num_workers: int = 0,
        pin_memory: bool = False,
    ) -> None:
        if splits is None:
            splits = ["validation"]
        snli_dir = Path(data_root) / "SNLI"
        records  = _load_snli(snli_dir, splits)
        self.tokenizer = _get_tokenizer(tokenizer)
        self.tokenizer.padding_side = "left"
        torch_ds = SNLIValDataset(records, self.tokenizer, max_length)
        self._sampler, self._loader = _build_val_loader(
            torch_ds, batch_size, shuffle, seed, num_workers, pin_memory,
            length_fn=lambda r: len(r["premise"]) + len(r["hypothesis"]),
        )
        print(f"[SNLIValDataLoader] splits={splits}  samples={len(records):,}  "
              f"batches={len(self._loader):,}  batch_size={batch_size}")

    def set_epoch(self, epoch: int) -> None: self._sampler.set_epoch(epoch)
    def __iter__(self): return iter(self._loader)
    def __len__(self) -> int: return len(self._loader)

    @property
    def dataset(self) -> SNLIValDataset:
        return self._loader.dataset


# ---------------------------------------------------------------------------
# MixedFinetuneDataLoader — balanced multi-task batching (SFT)
# ---------------------------------------------------------------------------

class MixedFinetuneDataLoader:
    """
    DataLoader kết hợp MMLU + SQuAD + SNLI với balanced per-batch mixing.
    Tất cả samples đều dùng Instruct chat template + L_LM loss.

    Xem module docstring để biết chiến lược batching.

    Batch keys
    ----------
    input_ids / attention_mask / labels  LongTensor [B', L]
    task                                 List[str]
    """

    def __init__(
        self,
        data_root: str = "../raw_data/english/",
        mmlu_splits: List[str] = None,
        squad_splits: List[str] = None,
        snli_splits: List[str] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        batch_size: int = 12,
        mmlu_max_length: int = 512,
        squad_max_length: int = 1024,
        snli_max_length: int = 256,
        shuffle: bool = True,
        seed: int = 42,
        num_workers: int = 0,
        pin_memory: bool = False,
    ) -> None:
        if batch_size < 3:
            raise ValueError("batch_size must be >= 3.")

        if mmlu_splits  is None: mmlu_splits  = ["auxiliary_train"]
        if squad_splits is None: squad_splits = ["train"]
        if snli_splits  is None: snli_splits  = ["train"]

        self.tokenizer  = _get_tokenizer(tokenizer)
        self.tokenizer.padding_side = "right"
        self.batch_size = batch_size
        self.shuffle    = shuffle
        self.seed       = seed
        self._epoch     = 0
        self.num_workers = num_workers
        self.pin_memory  = pin_memory

        mmlu_records  = _load_mmlu(Path(data_root) / "MMLU",  mmlu_splits)
        squad_records = _load_squad(Path(data_root) / "SQuAD", squad_splits)
        snli_records  = _load_snli(Path(data_root) / "SNLI",  snli_splits)

        self._per_task = batch_size // 3
        min_len        = min(len(mmlu_records), len(squad_records), len(snli_records))
        n_balanced     = (min_len // self._per_task) * self._per_task

        self._balanced_mmlu  = mmlu_records[:n_balanced]
        self._balanced_squad = squad_records[:n_balanced]
        self._balanced_snli  = snli_records[:n_balanced]
        self._remainder_mmlu  = mmlu_records[n_balanced:]
        self._remainder_squad = squad_records[n_balanced:]
        self._remainder_snli  = snli_records[n_balanced:]

        self._ds_mmlu  = MMLUDataset(mmlu_records,  self.tokenizer, mmlu_max_length)
        self._ds_squad = SQuADDataset(squad_records, self.tokenizer, squad_max_length)
        self._ds_snli  = SNLIDataset(snli_records,  self.tokenizer, snli_max_length)

        n_mixed_batches = n_balanced // self._per_task
        n_rem_mmlu  = _count_batches(len(self._remainder_mmlu),  batch_size)
        n_rem_squad = _count_batches(len(self._remainder_squad), batch_size)
        n_rem_snli  = _count_batches(len(self._remainder_snli),  batch_size)
        self._n_batches = n_mixed_batches + n_rem_mmlu + n_rem_squad + n_rem_snli

        print(
            f"\n[MixedFinetuneDataLoader]\n"
            f"  batch_size={batch_size}  per_task={self._per_task}\n"
            f"  MMLU  total={len(mmlu_records):,}  balanced={n_balanced:,}  "
            f"remainder={len(self._remainder_mmlu):,}\n"
            f"  SQuAD total={len(squad_records):,}  balanced={n_balanced:,}  "
            f"remainder={len(self._remainder_squad):,}\n"
            f"  SNLI  total={len(snli_records):,}  balanced={n_balanced:,}  "
            f"remainder={len(self._remainder_snli):,}\n"
            f"  mixed batches={n_mixed_batches:,}  "
            f"remainder: MMLU={n_rem_mmlu} SQuAD={n_rem_squad} SNLI={n_rem_snli}\n"
            f"  total batches={self._n_batches:,}"
        )

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch

    def __len__(self) -> int:
        return self._n_batches

    def __iter__(self) -> Iterator[Dict]:
        rng = random.Random(self.seed + self._epoch)
        bal_mmlu_idx  = list(range(len(self._balanced_mmlu)))
        bal_squad_idx = list(range(len(self._balanced_squad)))
        bal_snli_idx  = list(range(len(self._balanced_snli)))
        if self.shuffle:
            rng.shuffle(bal_mmlu_idx)
            rng.shuffle(bal_squad_idx)
            rng.shuffle(bal_snli_idx)

        pad_id = self.tokenizer.pad_token_id
        n_mixed = len(self._balanced_mmlu) // self._per_task
        for i in range(n_mixed):
            s = i * self._per_task
            e = s + self._per_task
            samples = (
                [self._ds_mmlu[j]  for j in bal_mmlu_idx[s:e]]
                + [self._ds_squad[j] for j in bal_squad_idx[s:e]]
                + [self._ds_snli[j]  for j in bal_snli_idx[s:e]]
            )
            yield _collate_fn_sft(samples, pad_id)

        yield from self._iter_remainder(
            self._remainder_mmlu,  self._ds_mmlu,  len(self._balanced_mmlu),  rng, pad_id)
        yield from self._iter_remainder(
            self._remainder_squad, self._ds_squad, len(self._balanced_squad), rng, pad_id)
        yield from self._iter_remainder(
            self._remainder_snli,  self._ds_snli,  len(self._balanced_snli),  rng, pad_id)

    def _iter_remainder(self, remainder_records, full_dataset, offset, rng, pad_id):
        if not remainder_records:
            return
        indices = list(range(offset, offset + len(remainder_records)))
        if self.shuffle:
            rng.shuffle(indices)
        for chunk in _chunked(indices, self.batch_size):
            samples = [full_dataset[j] for j in chunk]
            yield _collate_fn_sft(samples, pad_id)


# ---------------------------------------------------------------------------
# Evaluation helpers — parse generated output
# ---------------------------------------------------------------------------

def parse_mmlu_output(generated_text: str) -> str:
    """
    Parse output của model cho MMLU.
    Trả về letter đầu tiên trong {A, B, C, D} tìm thấy trong generated_text.
    Nếu không tìm thấy, trả về "".
    """
    text = generated_text.strip().upper()
    # Ưu tiên: letter đứng một mình (cả dòng)
    if text in MMLU_OPTIONS:
        return text
    # Tìm letter đầu tiên có trong text
    for ch in text:
        if ch in MMLU_OPTIONS:
            return ch
    return ""


def parse_snli_output(generated_text: str) -> str:
    """
    Parse output của model cho SNLI.
    Trả về label word đầu tiên tìm thấy trong generated_text (lowercase).
    Nếu không tìm thấy, trả về "".
    """
    text = generated_text.strip().lower()
    for label in SNLI_CANDIDATES:
        if label in text:
            return label
    return ""


def _normalize_answer(s: str) -> str:
    """Lower, remove punctuation, articles, extra whitespace — chuẩn SQuAD."""
    def remove_articles(t: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", t)

    def white_space_fix(t: str) -> str:
        return " ".join(t.split())

    def remove_punc(t: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in t if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(s.lower())))


def compute_exact_match(prediction: str, gold_answers: List[str]) -> float:
    """EM = 1 nếu prediction (sau normalize) khớp ít nhất 1 gold answer."""
    pred_norm = _normalize_answer(prediction)
    return float(any(_normalize_answer(g) == pred_norm for g in gold_answers))


def compute_f1_score(prediction: str, gold_answers: List[str]) -> float:
    """F1 token-level, lấy max F1 qua tất cả gold answers (chuẩn SQuAD)."""
    pred_tokens = _normalize_answer(prediction).split()

    def _f1_single(pred_toks: List[str], gold: str) -> float:
        gold_toks = _normalize_answer(gold).split()
        common    = Counter(pred_toks) & Counter(gold_toks)
        n_common  = sum(common.values())
        if n_common == 0:
            return 0.0
        precision = n_common / len(pred_toks)
        recall    = n_common / len(gold_toks)
        return 2 * precision * recall / (precision + recall)

    return max(_f1_single(pred_tokens, g) for g in gold_answers)


# ---------------------------------------------------------------------------
# Smoke-test
# Run: python finetune_dataloader.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    from huggingface_hub import login

    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        login(token=hf_token)

    DATA_ROOT  = "../raw_data/english/"
    BATCH_SIZE = 12

    print("=" * 60)
    print("Loading shared tokenizer (Llama-3-8B-Instruct) ...")
    print("=" * 60)
    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    print(f"  vocab={tokenizer.vocab_size:,}  bos={tokenizer.bos_token_id}  "
          f"eos={tokenizer.eos_token_id}  pad={tokenizer.pad_token_id}\n")

    # ── SFT Mixed DataLoader ─────────────────────────────────────────────────
    print("=" * 60)
    print("Smoke-test: MixedFinetuneDataLoader (Instruct + L_LM)")
    print("=" * 60)
    mixed_loader = MixedFinetuneDataLoader(
        data_root=DATA_ROOT, batch_size=BATCH_SIZE, tokenizer=tokenizer
    )
    mixed_loader.set_epoch(0)
    batch = next(iter(mixed_loader))
    tasks_in_batch = batch["task"]
    found_tasks    = set()
    for i in range(len(tasks_in_batch)):
        t = tasks_in_batch[i]
        if t not in found_tasks:
            found_tasks.add(t)
            ids    = batch["input_ids"][i]
            labels = batch["labels"][i]
            # Decode full sequence
            full_text = tokenizer.decode(ids[ids != tokenizer.pad_token_id],
                                         skip_special_tokens=False)
            # Decode answer only (non -100 labels)
            ans_mask  = labels != -100
            ans_ids   = ids[ans_mask]
            ans_text  = tokenizer.decode(ans_ids, skip_special_tokens=True)
            print(f"\n>>> [TASK: {t.upper()}]")
            print(f"ANSWER TARGET: '{ans_text}'")
            # Kiểm tra có loss token
            n_loss = ans_mask.sum().item()
            n_mask = (labels == -100).sum().item()
            print(f"  loss_tokens={n_loss}  masked_tokens={n_mask}")
        if len(found_tasks) == 3:
            break

    assert (batch["labels"] == -100).any(), "Phải có tokens bị mask"
    assert (batch["labels"] != -100).any(), "Phải có tokens có loss"
    assert found_tasks == {"mmlu", "squad", "snli"}, f"Missing tasks: {found_tasks}"
    print(f"\n  ✓ SFT smoke-test passed. Tasks: {found_tasks}")

    # ── MMLU Val DataLoader ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Smoke-test: MMLUValDataLoader (generate mode)")
    print("=" * 60)
    mmlu_val = MMLUValDataLoader(
        data_root=DATA_ROOT, tokenizer=tokenizer, batch_size=4, max_length=512
    )
    batch = next(iter(mmlu_val))
    assert batch["input_ids"].dim()      == 2
    assert "labels"     not in batch,    "Val batch không được có labels"
    assert "gold_label" in batch
    assert all(g in MMLU_OPTIONS for g in batch["gold_label"])
    prompt_sample = tokenizer.decode(
        batch["input_ids"][0][batch["attention_mask"][0] == 1],
        skip_special_tokens=False
    )
    print(f"  ✓ shape={tuple(batch['input_ids'].shape)}  task={batch['task'][0]}")
    print(f"  gold_label[0] = '{batch['gold_label'][0]}'")
    print(f"  prompt ends: '...{prompt_sample[-60:]}'")

    # ── SNLI Val DataLoader ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Smoke-test: SNLIValDataLoader (generate mode)")
    print("=" * 60)
    snli_val = SNLIValDataLoader(
        data_root=DATA_ROOT, tokenizer=tokenizer, batch_size=8, max_length=256
    )
    batch = next(iter(snli_val))
    assert batch["input_ids"].dim()      == 2
    assert "labels"     not in batch,    "Val batch không được có labels"
    assert "gold_label" in batch
    assert all(g in SNLI_CANDIDATES for g in batch["gold_label"])
    print(f"  ✓ shape={tuple(batch['input_ids'].shape)}  task={batch['task'][0]}")
    print(f"  gold_label[0] = '{batch['gold_label'][0]}'")

    # ── SQuAD Val DataLoader ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Smoke-test: SQuADValDataLoader (generate mode)")
    print("=" * 60)
    squad_val = SQuADValDataLoader(
        data_root=DATA_ROOT, tokenizer=tokenizer, batch_size=4, max_length=1024
    )
    batch = next(iter(squad_val))
    assert batch["input_ids"].dim()      == 2
    assert "labels"     not in batch,    "Val batch không được có labels"
    assert "answers"    in batch
    assert isinstance(batch["answers"][0], list)
    prompt_sample = tokenizer.decode(
        batch["input_ids"][0][batch["attention_mask"][0] == 1],
        skip_special_tokens=True
    )
    print(f"  ✓ shape={tuple(batch['input_ids'].shape)}  task={batch['task'][0]}")
    print(f"  gold_label[0] = '{batch['gold_label'][0]}'")
    print(f"  answers[0] = {batch['answers'][0]}")
    assert prompt_sample.strip().endswith("Question:") or "Question" in prompt_sample

    # ── Parse helpers ────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Smoke-test: parse helpers")
    print("=" * 60)
    assert parse_mmlu_output("A")         == "A"
    assert parse_mmlu_output("The answer is B.")  == "B"
    assert parse_mmlu_output("c")         == "C"
    assert parse_snli_output("entailment")         == "entailment"
    assert parse_snli_output("This is neutral.")   == "neutral"
    assert compute_exact_match("New York", ["New York", "NYC"]) == 1.0
    assert compute_f1_score("the cat sat", ["the cat sat on the mat"]) > 0.5
    print("  ✓ parse helpers OK")

    print("\n" + "=" * 60)
    print("✓ All smoke-tests passed.")
    print("=" * 60)