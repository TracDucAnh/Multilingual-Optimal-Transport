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

Backbone: meta-llama/Meta-Llama-3-8B-Instruct
    - Prompt được wrap bằng chat template (apply_chat_template)
    - Left-padded cho batch generation
    - model.generate() → decode ONLY new tokens → parse/normalize output
    - Instruct model tuân theo instruction tốt hơn → giảm unknown / parse fail

MMMLU : model phải output đúng "A", "B", "C", hoặc "D"
XNLI  : model phải output đúng "entailment", "neutral", hoặc "contradiction"
XSQuAD: model output câu trả lời tự do → F1 / EM vs gold answers

Batch keys (tất cả tasks)
──────────────────────────
    input_ids       LongTensor [B, L]    — prompt left-padded (sau chat template)
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

DEFAULT_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"

XNLI_LABEL_MAP  = {0: "entailment", 1: "neutral", 2: "contradiction"}
XNLI_CANDIDATES = ["entailment", "neutral", "contradiction"]

MCQ_OPTIONS = ["A", "B", "C", "D"]

# System prompt dùng chung — giữ ngắn, rõ ràng để instruct model không verbose
_SYSTEM_PROMPT = (
    "You are a precise answer extraction assistant. "
    "Always respond with only the answer and nothing else. "
    "Do not explain, do not repeat the question."
)


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
# Prompt builders  (trả về user message text — chưa apply chat template)
# ---------------------------------------------------------------------------

def _build_mmmlu_prompt(record: dict) -> str:
    """
    User message cho MMMLU.
    Instruct model được yêu cầu trả lời đúng 1 ký tự: A / B / C / D.
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
        f"Reply with only the single letter of the correct answer (A, B, C, or D). "
        f"Do not include any other text."
    )


def _build_xnli_prompt(record: dict) -> str:
    """
    User message cho XNLI.
    Instruct model trả lời đúng 1 từ: entailment / neutral / contradiction.
    """
    premise    = record["premise"].strip()
    hypothesis = record["hypothesis"].strip()
    return (
        f"Determine the logical relationship between the following premise and hypothesis.\n\n"
        f"Premise: {premise}\n"
        f"Hypothesis: {hypothesis}\n\n"
        f"Reply with exactly one word — either 'entailment', 'neutral', or 'contradiction'. "
        f"Do not include any other text."
    )


def _build_xsquad_prompt(record: dict) -> str:
    """
    User message cho XSQuAD.
    Instruct model trả lời ngắn gọn, đúng span có trong passage.
    """
    context  = record["context"].strip()
    question = record["question"].strip()
    return (
        f"Read the passage below and answer the question with a short phrase or span "
        f"taken directly from the passage. Do not include any other text.\n\n"
        f"Passage: {context}\n\n"
        f"Question: {question}"
    )


# ---------------------------------------------------------------------------
# Chat-template encoder
# ---------------------------------------------------------------------------

def _apply_chat_template(
    tokenizer: PreTrainedTokenizerBase,
    user_message: str,
    max_length: int,
) -> Dict[str, torch.Tensor]:
    """
    Wrap user_message bằng Llama-3-Instruct chat template rồi tokenize.

    Trả về dict với:
        input_ids      : LongTensor [L]
        attention_mask : LongTensor [L]

    add_generation_prompt=True để model biết đây là lúc cần generate assistant turn.
    Truncation thực hiện ở phía LEFT (bỏ đầu) để giữ instruction cuối cùng.
    """
    messages = [
        {"role": "system",    "content": _SYSTEM_PROMPT},
        {"role": "user",      "content": user_message},
    ]
    # tokenize=False để lấy string → tokenize thủ công với truncation left
    chat_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,  # thêm <|start_header_id|>assistant<|end_header_id|>
    )
    encoded = tokenizer(
        chat_text,
        add_special_tokens=False,   # chat template đã có BOS/EOS
        truncation=True,
        max_length=max_length,
        padding=False,
        return_tensors="pt",
    )
    return {
        "input_ids":      encoded["input_ids"].squeeze(0),       # [L]
        "attention_mask": encoded["attention_mask"].squeeze(0),  # [L]
    }


# ---------------------------------------------------------------------------
# Base Generation Dataset  (left-padded, prompt-only, Instruct template)
# ---------------------------------------------------------------------------

class _BaseGenerationDataset(Dataset):
    """
    Prompt-only dataset cho generation mode với Instruct model.

    Mỗi sample:
        input_ids      : chat-formatted + tokenized (left-pad trong collate)
        attention_mask : tương ứng
        gold_label     : str  — ground-truth label
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

    def _build_user_message(self, record: dict) -> str:
        """Override trong subclass để trả về user message string."""
        raise NotImplementedError

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict:
        record     = self.records[idx]
        lang       = record["_lang"]
        gold_label = record["_gold_label"]

        user_msg = self._build_user_message(record)
        encoded  = _apply_chat_template(self.tokenizer, user_msg, self.max_length)

        item = {
            "input_ids":      encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "gold_label":     gold_label,
            "task":           self.task_name,
            "lang":           lang,
        }

        # XSQuAD cần toàn bộ gold answers cho F1/EM
        if "answers" in record:
            answers = record["answers"].get("text", [])
            item["answers"] = [a.strip() for a in answers if a.strip()] or ["no answer"]

        return item


# ---------------------------------------------------------------------------
# Left-pad collate  (dùng chung cho tất cả generation tasks)
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

    if "answers" in batch[0]:
        out["answers"] = [s["answers"] for s in batch]

    return out


# ---------------------------------------------------------------------------
# SortedLengthSampler
# ---------------------------------------------------------------------------

class SortedLengthSampler(Sampler):
    """Sort by prompt length để giảm padding waste trong batch."""

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
    tok.padding_side = "left"   # ← thêm dòng này
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
    Generation-mode dataset cho MMMLU (Instruct backbone).
    gold_label: "A" / "B" / "C" / "D"
    """
    task_name = "mmmlu"

    def _build_user_message(self, record: dict) -> str:
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
            r["_gold_label"] = r["Answer"].strip()
        records.extend(valid)
        print(f"[MMMLU] Loaded {len(valid):,} records from {lang_dir.name}/test.json")

    return records


class MMLUDownstreamDataLoader:
    """
    Generation DataLoader cho MMMLU (Instruct backbone).

    Batch keys
    ----------
    input_ids       LongTensor [B, L]   — prompt left-padded (chat template)
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
    Generation-mode dataset cho XNLI (Instruct backbone).
    gold_label: "entailment" / "neutral" / "contradiction"
    """
    task_name = "xnli"

    def _build_user_message(self, record: dict) -> str:
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
            r["_gold_label"] = XNLI_LABEL_MAP[r["label"]]
        records.extend(valid)
        print(f"[XNLI] Loaded {len(valid):,} records from {lang_dir.name}/test.json")

    return records


class XNLIDataLoader:
    """
    Generation DataLoader cho XNLI (Instruct backbone).

    Batch keys
    ----------
    input_ids       LongTensor [B, L]   — prompt left-padded (chat template)
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
        max_length: int = 512,
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
    """Generation-mode dataset cho XSQuAD (Instruct backbone)."""
    task_name = "xsquad"

    def _build_user_message(self, record: dict) -> str:
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
    """Generation DataLoader cho XSQuAD (Instruct backbone)."""

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
    print("Loading shared tokenizer (Llama-3-8B-Instruct) ...")
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
    print("Smoke-tests: MMMLU / XNLI / XSQuAD (Instruct, generation mode)")
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
        assert "lang" in batch
        assert "task" in batch

        if name == "MMMLU":
            assert all(g in MCQ_OPTIONS for g in batch["gold_label"]), \
                f"MMMLU: gold_label phải trong {MCQ_OPTIONS}"
        elif name == "XNLI":
            assert all(g in XNLI_CANDIDATES for g in batch["gold_label"]), \
                f"XNLI: gold_label phải trong {XNLI_CANDIDATES}"
        elif name == "XSQuAD":
            assert "answers" in batch, "XSQuAD: missing answers"
            assert all(isinstance(a, list) for a in batch["answers"])

        ids_0  = batch["input_ids"][0]
        mask_0 = batch["attention_mask"][0]
        prompt = tokenizer.decode(ids_0[mask_0 == 1], skip_special_tokens=False)

        print(f"\n  ✓ {name}  shape={tuple(batch['input_ids'].shape)}  "
              f"task={batch['task'][0]}  langs={set(batch['lang'])}")
        print(f"  gold_label[0] = '{batch['gold_label'][0]}'")
        print(f"  PROMPT (trunc 400 chars):\n  {prompt[:400]!r}")

    print("\n" + "=" * 60)
    print("✓ All smoke-tests passed.")
    print("=" * 60)