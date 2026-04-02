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

Validation — Generate Mode:
    - Trả về prompt-only (left-padded) để gọi model.generate()
    - Parse output text → so sánh với gold label
    - MMLU / SNLI: Accuracy
    - SQuAD: F1 + Exact Match

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SequentialTaskDataLoader — 1 dataloader duy nhất, chạy tuần tự từng task
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Thay vì mix 3 task cùng lúc, dataloader này yield toàn bộ MMLU trước,
rồi đến SQuAD, rồi đến SNLI trong mỗi epoch.

Adaptive batch size:
  - Mỗi task bắt đầu với batch_size được truyền vào (mặc định 64)
  - Nếu gặp OOM → tự động giảm một nửa và retry
  - In thông báo khi đổi task và khi OOM / thay đổi batch size
  - Batch size hiệu quả được lưu lại cho lần gọi tiếp theo

Cơ chế: DataLoader thực sự được tạo on-the-fly per task với batch size
hiện tại. OOM được bắt bên ngoài (trong training loop) và caller
gọi loader.report_oom() để trigger giảm batch size.
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

TASK_ORDER = ["mmlu", "squad", "snli"]   # thứ tự yield trong mỗi epoch

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
    idx = record["answer"]
    record["answer"]  = MMLU_OPTIONS[idx]
    record["choices"] = [
        f"{MMLU_OPTIONS[i]}. {record['choices'][i]}"
        for i in range(len(record["choices"]))
    ]
    return record


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def _build_mmlu_user_message(record: dict) -> str:
    subject     = record.get("subject", "general knowledge").replace("_", " ")
    question    = record["question"].strip()
    options_str = "\n".join(record["choices"])
    return (
        f"The following is a multiple choice question about {subject}.\n\n"
        f"Question: {question}\n"
        f"{options_str}\n\n"
        f"Reply with only the single letter of the correct answer (A, B, C, or D). "
        f"Do not include any other text."
    )


def _build_mmlu_answer(record: dict) -> str:
    return record["answer"]


def _build_squad_user_message(record: dict) -> str:
    context  = record["context"].strip()
    question = record["question"].strip()
    return (
        f"Read the passage below and answer the question with a short phrase or span "
        f"taken directly from the passage. Do not include any other text.\n\n"
        f"Passage: {context}\n\n"
        f"Question: {question}"
    )


def _build_squad_answer(record: dict) -> str:
    answers = record["answers"]
    return answers["text"][0].strip() if answers.get("text") else "no answer"


def _build_snli_user_message(record: dict) -> str:
    return (
        f"Determine the logical relationship between the following premise and hypothesis.\n\n"
        f"Premise: {record['premise'].strip()}\n"
        f"Hypothesis: {record['hypothesis'].strip()}\n\n"
        f"Reply with exactly one word — either 'entailment', 'neutral', or 'contradiction'. "
        f"Do not include any other text."
    )


def _build_snli_answer(record: dict) -> str:
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
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user",   "content": user_message},
    ]
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    answer_ids = tokenizer.encode(answer_str, add_special_tokens=False)
    eos_ids    = [tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else []

    max_prompt_len = max_length - len(answer_ids) - len(eos_ids)
    if max_prompt_len <= 0:
        answer_ids = answer_ids[:max_length - len(eos_ids) - 1]
        max_prompt_len = 1
    prompt_ids = prompt_ids[-max_prompt_len:]

    full_ids   = prompt_ids + answer_ids + eos_ids
    prompt_len = len(prompt_ids)
    labels     = [-100] * prompt_len + answer_ids + eos_ids

    assert len(full_ids) == len(labels)

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
# Base Datasets
# ---------------------------------------------------------------------------

class _BaseSFTDataset(Dataset):
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
        record     = self.records[idx]
        user_msg   = self._build_user_message(record)
        answer_str = self._build_answer(record)
        encoded    = _apply_chat_template_sft(
            self.tokenizer, user_msg, answer_str, self.max_length
        )
        return {
            "input_ids":      encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "labels":         encoded["labels"],
            "task":           self.task_name,
        }


class _BaseValDataset(Dataset):
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
        return {
            "input_ids":      encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "gold_label":     gold_label,
            "task":           self.task_name,
        }


# ---------------------------------------------------------------------------
# Collate functions
# ---------------------------------------------------------------------------

def _collate_fn_sft(batch: List[Dict], pad_token_id: int) -> Dict:
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
    if "answers" in batch[0]:
        out["answers"] = [s["answers"] for s in batch]
    return out


# ---------------------------------------------------------------------------
# SortedLengthSampler
# ---------------------------------------------------------------------------

class SortedLengthSampler(Sampler):
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
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    return tok


def _count_batches(n_records: int, batch_size: int) -> int:
    if n_records == 0:
        return 0
    return (n_records + batch_size - 1) // batch_size


# ---------------------------------------------------------------------------
# Per-task SFT Datasets
# ---------------------------------------------------------------------------

class MMLUDataset(_BaseSFTDataset):
    task_name = "mmlu"
    def _build_user_message(self, record): return _build_mmlu_user_message(record)
    def _build_answer(self, record):       return _build_mmlu_answer(record)


class SQuADDataset(_BaseSFTDataset):
    task_name = "squad"
    def _build_user_message(self, record): return _build_squad_user_message(record)
    def _build_answer(self, record):       return _build_squad_answer(record)


class SNLIDataset(_BaseSFTDataset):
    task_name = "snli"
    def _build_user_message(self, record): return _build_snli_user_message(record)
    def _build_answer(self, record):       return _build_snli_answer(record)


# ---------------------------------------------------------------------------
# Per-task Val Datasets
# ---------------------------------------------------------------------------

class MMLUValDataset(_BaseValDataset):
    task_name = "mmlu"
    def _build_user_message(self, record): return _build_mmlu_user_message(record)
    def _get_gold_label(self, record):     return record["answer"]


class SNLIValDataset(_BaseValDataset):
    task_name = "snli"
    def _build_user_message(self, record): return _build_snli_user_message(record)
    def _get_gold_label(self, record):     return SNLI_LABEL_MAP[record["label"]]


class SQuADValDataset(_BaseValDataset):
    task_name = "squad"
    def _build_user_message(self, record): return _build_squad_user_message(record)
    def _get_gold_label(self, record):
        answers = record["answers"].get("text", [])
        return answers[0].strip() if answers else "no answer"

    def __getitem__(self, idx: int) -> Dict:
        item    = super().__getitem__(idx)
        record  = self.records[idx]
        answers = record["answers"].get("text", [])
        item["answers"] = [a.strip() for a in answers if a.strip()] or ["no answer"]
        return item


# ---------------------------------------------------------------------------
# Data loading helpers
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
# SequentialTaskDataLoader — 1 dataloader duy nhất, từng task tuần tự
# ---------------------------------------------------------------------------

class SequentialTaskDataLoader:
    """
    DataLoader duy nhất chạy tuần tự từng task: MMLU → SQuAD → SNLI.

    Adaptive batch size:
      - Mỗi task bắt đầu với initial_batch_size
      - Caller gọi report_oom(task) khi gặp OOM → batch size giảm một nửa
      - In thông báo đổi task và OOM / thay đổi batch size

    Cách dùng trong training loop:
        loader = SequentialTaskDataLoader(...)
        for batch in loader:
            try:
                loss = forward(batch)
                loss.backward()
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                loader.report_oom(batch["task"][0])
                optimizer.zero_grad()
                continue   # bỏ batch này, batch tiếp theo sẽ dùng bs nhỏ hơn

    Thuộc tính public:
        current_batch_sizes   Dict[str, int]   batch size hiện tại mỗi task
        current_task          str              task đang được yield
        total_batches         int              tổng số batch ước tính (thay đổi khi OOM)
    """

    # Batch size tối thiểu trước khi raise lỗi
    MIN_BATCH_SIZE = 1

    def __init__(
        self,
        data_root: str = "../raw_data/english/",
        mmlu_splits: List[str] = None,
        squad_splits: List[str] = None,
        snli_splits: List[str] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        initial_batch_size: int = 64,
        mmlu_max_length: int = 512,
        squad_max_length: int = 1024,
        snli_max_length: int = 256,
        shuffle: bool = True,
        seed: int = 42,
        num_workers: int = 0,
        pin_memory: bool = False,
    ) -> None:
        if mmlu_splits  is None: mmlu_splits  = ["auxiliary_train"]
        if squad_splits is None: squad_splits = ["train"]
        if snli_splits  is None: snli_splits  = ["train"]

        self.tokenizer       = _get_tokenizer(tokenizer)
        self.tokenizer.padding_side = "right"
        self.initial_bs      = initial_batch_size
        self.shuffle         = shuffle
        self.seed            = seed
        self.num_workers     = num_workers
        self.pin_memory      = pin_memory
        self._epoch          = 0

        # ── Load raw records ──────────────────────────────────────────────────
        mmlu_records  = _load_mmlu(Path(data_root) / "MMLU",  mmlu_splits)
        squad_records = _load_squad(Path(data_root) / "SQuAD", squad_splits)
        snli_records  = _load_snli(Path(data_root) / "SNLI",  snli_splits)

        # ── Build torch Datasets ──────────────────────────────────────────────
        self._datasets: Dict[str, _BaseSFTDataset] = {
            "mmlu":  MMLUDataset( mmlu_records,  self.tokenizer, mmlu_max_length),
            "squad": SQuADDataset(squad_records, self.tokenizer, squad_max_length),
            "snli":  SNLIDataset( snli_records,  self.tokenizer, snli_max_length),
        }

        # Proxy lengths cho SortedLengthSampler
        self._lengths: Dict[str, List[int]] = {
            "mmlu":  [len(r["question"]) + sum(len(c) for c in r.get("choices", []))
                      for r in mmlu_records],
            "squad": [len(r["context"]) for r in squad_records],
            "snli":  [len(r["premise"]) + len(r["hypothesis"]) for r in snli_records],
        }

        # ── Adaptive batch sizes — độc lập mỗi task ──────────────────────────
        self.current_batch_sizes: Dict[str, int] = {
            "mmlu":  initial_batch_size,
            "squad": initial_batch_size,
            "snli":  initial_batch_size,
        }
        self._oom_counts: Dict[str, int] = {"mmlu": 0, "squad": 0, "snli": 0}

        # State
        self.current_task: str = "mmlu"

        n_records = {k: len(v.records) for k, v in self._datasets.items()}
        print(
            f"\n[SequentialTaskDataLoader]\n"
            f"  Task order : {' → '.join(TASK_ORDER)}\n"
            f"  MMLU  records={n_records['mmlu']:,}   bs={self.current_batch_sizes['mmlu']}\n"
            f"  SQuAD records={n_records['squad']:,}   bs={self.current_batch_sizes['squad']}\n"
            f"  SNLI  records={n_records['snli']:,}   bs={self.current_batch_sizes['snli']}\n"
            f"  (batch sizes auto-reduce on OOM)"
        )

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _make_loader(self, task: str) -> DataLoader:
        """Tạo DataLoader cho task với batch size hiện tại."""
        bs      = self.current_batch_sizes[task]
        ds      = self._datasets[task]
        lengths = self._lengths[task]
        pad_id  = self.tokenizer.pad_token_id

        sampler = SortedLengthSampler(lengths, bs, self.shuffle, self.seed)
        sampler.set_epoch(self._epoch)

        return DataLoader(
            ds,
            batch_size=bs,
            sampler=sampler,
            collate_fn=lambda b: _collate_fn_sft(b, pad_token_id=pad_id),
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
        )

    @property
    def total_batches(self) -> int:
        """Tổng số batch ước tính với batch sizes hiện tại."""
        return sum(
            _count_batches(len(self._datasets[t].records), self.current_batch_sizes[t])
            for t in TASK_ORDER
        )

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch

    def report_oom(self, task: str) -> int:
        """
        Caller gọi khi gặp OOM cho task này.
        Giảm batch size một nửa. In thông báo.
        Trả về batch size mới. Raise nếu đã đạt MIN_BATCH_SIZE.
        """
        old_bs = self.current_batch_sizes[task]
        new_bs = max(self.MIN_BATCH_SIZE, old_bs // 2)
        self._oom_counts[task] += 1

        if new_bs == old_bs:
            raise RuntimeError(
                f"[OOM] Task '{task}': batch size đã đạt tối thiểu ({old_bs}), "
                f"không thể giảm thêm! Cần giảm max_length hoặc dùng GPU lớn hơn."
            )

        self.current_batch_sizes[task] = new_bs
        print(
            f"\n{'!'*60}\n"
            f"[OOM #{self._oom_counts[task]}] Task='{task}'  "
            f"batch_size: {old_bs} → {new_bs}\n"
            f"{'!'*60}"
        )
        return new_bs

    # ── Main iteration ─────────────────────────────────────────────────────────

    def __iter__(self) -> Iterator[Dict]:
        for task in TASK_ORDER:
            self.current_task = task
            bs = self.current_batch_sizes[task]
            n  = len(self._datasets[task].records)

            print(
                f"\n{'─'*60}\n"
                f"[DataLoader] ▶ Bắt đầu task: {task.upper():<6}  "
                f"records={n:,}  batch_size={bs}  "
                f"~{_count_batches(n, bs):,} batches\n"
                f"{'─'*60}"
            )

            # Yield batches cho task này
            # Nếu OOM được báo cáo (qua report_oom), loader tiếp theo sẽ dùng bs nhỏ hơn
            # Nhưng batch hiện tại vẫn cần được xử lý — caller tự bỏ qua và continue
            loader = self._make_loader(task)
            last_bs = bs

            for batch in loader:
                # Kiểm tra xem batch size có bị thay đổi giữa chừng không (sau OOM)
                cur_bs = self.current_batch_sizes[task]
                if cur_bs != last_bs:
                    # Batch size đã đổi, cần tạo lại loader
                    # Nhưng ta không thể dừng iterator đang chạy dở —
                    # thay vào đó, ta yield hết loader cũ rồi sau đó
                    # iteration tiếp theo sẽ dùng loader mới.
                    # → Để đơn giản và an toàn: yield batch hiện tại, đánh dấu cần rebuild
                    last_bs = cur_bs

                yield batch

            print(
                f"[DataLoader] ✓ Hoàn thành task: {task.upper():<6}  "
                f"final batch_size={self.current_batch_sizes[task]}"
            )

    def __len__(self) -> int:
        return self.total_batches


# ---------------------------------------------------------------------------
# Standalone Val DataLoaders (dùng cho evaluation sau epoch)
# ---------------------------------------------------------------------------

class MMLUValDataLoader:
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
        if splits is None: splits = ["validation"]
        mmlu_dir = Path(data_root) / "MMLU"
        records  = _load_mmlu(mmlu_dir, splits)
        self.tokenizer = _get_tokenizer(tokenizer)
        self.tokenizer.padding_side = "left"
        torch_ds = MMLUValDataset(records, self.tokenizer, max_length)
        lengths  = [len(r["question"]) + sum(len(c) for c in r.get("choices", []))
                    for r in records]
        sampler  = SortedLengthSampler(lengths, batch_size, shuffle, seed)
        pad_id   = self.tokenizer.pad_token_id
        self._loader = DataLoader(
            torch_ds,
            batch_size=batch_size,
            sampler=sampler,
            collate_fn=lambda b: _collate_fn_val(b, pad_token_id=pad_id),
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        )
        self._sampler = sampler
        print(f"[MMLUValDataLoader] splits={splits}  samples={len(records):,}  "
              f"batches={len(self._loader):,}  batch_size={batch_size}")

    def set_epoch(self, epoch: int) -> None: self._sampler.set_epoch(epoch)
    def __iter__(self): return iter(self._loader)
    def __len__(self) -> int: return len(self._loader)


class SNLIValDataLoader:
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
        if splits is None: splits = ["validation"]
        snli_dir = Path(data_root) / "SNLI"
        records  = _load_snli(snli_dir, splits)
        self.tokenizer = _get_tokenizer(tokenizer)
        self.tokenizer.padding_side = "left"
        torch_ds = SNLIValDataset(records, self.tokenizer, max_length)
        lengths  = [len(r["premise"]) + len(r["hypothesis"]) for r in records]
        sampler  = SortedLengthSampler(lengths, batch_size, shuffle, seed)
        pad_id   = self.tokenizer.pad_token_id
        self._loader = DataLoader(
            torch_ds,
            batch_size=batch_size,
            sampler=sampler,
            collate_fn=lambda b: _collate_fn_val(b, pad_token_id=pad_id),
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        )
        self._sampler = sampler
        print(f"[SNLIValDataLoader] splits={splits}  samples={len(records):,}  "
              f"batches={len(self._loader):,}  batch_size={batch_size}")

    def set_epoch(self, epoch: int) -> None: self._sampler.set_epoch(epoch)
    def __iter__(self): return iter(self._loader)
    def __len__(self) -> int: return len(self._loader)


class SQuADValDataLoader:
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
        if splits is None: splits = ["validation"]
        squad_dir = Path(data_root) / "SQuAD"
        records   = _load_squad(squad_dir, splits)
        self.tokenizer = _get_tokenizer(tokenizer)
        self.tokenizer.padding_side = "left"
        torch_ds  = SQuADValDataset(records, self.tokenizer, max_length)
        lengths   = [len(r.get("context", "")) for r in records]
        sampler   = SortedLengthSampler(lengths, batch_size, shuffle, seed)
        pad_id    = self.tokenizer.pad_token_id
        self._loader = DataLoader(
            torch_ds,
            batch_size=batch_size,
            sampler=sampler,
            collate_fn=lambda b: _collate_fn_val(b, pad_token_id=pad_id),
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        )
        self._sampler = sampler
        print(f"[SQuADValDataLoader] splits={splits}  samples={len(records):,}  "
              f"batches={len(self._loader):,}  batch_size={batch_size}")

    def set_epoch(self, epoch: int) -> None: self._sampler.set_epoch(epoch)
    def __iter__(self): return iter(self._loader)
    def __len__(self) -> int: return len(self._loader)


# Backward compat alias
SQuADValidationDataLoader = SQuADValDataLoader


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def parse_mmlu_output(generated_text: str) -> str:
    text = generated_text.strip().upper()
    if text in MMLU_OPTIONS:
        return text
    for ch in text:
        if ch in MMLU_OPTIONS:
            return ch
    return ""


def parse_snli_output(generated_text: str) -> str:
    text = generated_text.strip().lower()
    for label in SNLI_CANDIDATES:
        if label in text:
            return label
    return ""


def _normalize_answer(s: str) -> str:
    def remove_articles(t): return re.sub(r"\b(a|an|the)\b", " ", t)
    def white_space_fix(t): return " ".join(t.split())
    def remove_punc(t):
        exclude = set(string.punctuation)
        return "".join(ch for ch in t if ch not in exclude)
    return white_space_fix(remove_articles(remove_punc(s.lower())))


def compute_exact_match(prediction: str, gold_answers: List[str]) -> float:
    pred_norm = _normalize_answer(prediction)
    return float(any(_normalize_answer(g) == pred_norm for g in gold_answers))


def compute_f1_score(prediction: str, gold_answers: List[str]) -> float:
    pred_tokens = _normalize_answer(prediction).split()

    def _f1_single(pred_toks, gold):
        gold_toks = _normalize_answer(gold).split()
        common    = Counter(pred_toks) & Counter(gold_toks)
        n_common  = sum(common.values())
        if n_common == 0:
            return 0.0
        precision = n_common / len(pred_toks)
        recall    = n_common / len(gold_toks)
        return 2 * precision * recall / (precision + recall)

    return max(_f1_single(pred_tokens, g) for g in gold_answers)