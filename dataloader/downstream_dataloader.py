"""
downstream_dataloader.py
========================
Dataset và DataLoader cho zero-shot evaluation trên ba multilingual benchmarks:

    raw_data/downstream/MMMLU/   — Multilingual Multiple Choice QA
    raw_data/downstream/XNLI/    — Cross-lingual Natural Language Inference
    raw_data/downstream/XSQuAD/ — Cross-lingual Extractive QA

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Log-prob Scoring Strategy (MMMLU / XNLI)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Mỗi sample được mở rộng thành C rows (C = số candidates):
    MMMLU : C = 4  → " A" / " B" / " C" / " D"
    XNLI  : C = 3  → " entailment" / " neutral" / " contradiction"

Mỗi row (sample_idx, cand_idx) encode:
    input_ids : [BOS] <prompt tokens> <candidate tokens> [EOS]
    labels    : -100 ... -100          <candidate>       [EOS]

Sau khi forward batch [B×C, L]:
    loss_per_token = CrossEntropyLoss(reduction='none')   → [B×C, L]
    mean_nll[i]    = mean(loss_per_token[i][labels != -100])
                   ← mean NLL tránh bias candidate dài hơn
    pred           = argmin over C candidates per sample  → Accuracy

Batch keys (MMMLU / XNLI)
──────────────────────────
    input_ids       LongTensor [B*C, L]
    attention_mask  LongTensor [B*C, L]
    labels          LongTensor [B*C, L]   -100 cho prompt & padding
    sample_id       LongTensor [B*C]      index gốc của sample (để group C rows)
    cand_id         LongTensor [B*C]      index của candidate (0..C-1)
    gold_cand_id    LongTensor [B*C]      ground-truth candidate index (same per group)
    num_candidates  int                   C (constant per task)
    task            List[str]             "mmmlu" / "xnli"
    lang            List[str]             language code

Eval loop mẫu
─────────────
    all_nll, all_sample_id, all_cand_id, all_gold = [], [], [], []
    for batch in loader:
        with torch.no_grad():
            out = model(input_ids=batch["input_ids"].to(device),
                        attention_mask=batch["attention_mask"].to(device),
                        labels=batch["labels"].to(device))
        # CrossEntropyLoss trả về mean scalar; ta cần per-token
        logits = out.logits                             # [B*C, L, V]
        shift_logits = logits[:, :-1].contiguous()
        shift_labels = batch["labels"][:, 1:].to(device).contiguous()
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=-100)
        token_loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        ).view(shift_labels.size())                     # [B*C, L-1]
        n_ans = (shift_labels != -100).sum(-1).clamp(min=1)
        mean_nll = token_loss.sum(-1) / n_ans           # [B*C]

        all_nll.append(mean_nll.cpu())
        all_sample_id.append(batch["sample_id"])
        all_cand_id.append(batch["cand_id"])
        all_gold.append(batch["gold_cand_id"])

    all_nll       = torch.cat(all_nll)
    all_sample_id = torch.cat(all_sample_id)
    all_cand_id   = torch.cat(all_cand_id)
    all_gold      = torch.cat(all_gold)

    n_samples = all_sample_id.max().item() + 1
    C         = loader.num_candidates
    nll_mat   = torch.full((n_samples, C), float("inf"))
    nll_mat[all_sample_id, all_cand_id] = all_nll
    gold_mat  = torch.zeros(n_samples, dtype=torch.long)
    gold_mat[all_sample_id] = all_gold
    pred      = nll_mat.argmin(-1)
    accuracy  = (pred == gold_mat).float().mean().item()

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Generation Strategy (XSQuAD)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Prompt-only batches (left-padded) → model.generate() → F1 / EM vs answers field.
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

XNLI_LABEL_MAP   = {0: "entailment", 1: "neutral", 2: "contradiction"}
XNLI_CANDIDATES  = ["entailment", "neutral", "contradiction"]   # fixed order → cand_id 0/1/2

MCQ_OPTIONS      = ["A", "B", "C", "D"]


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
# Prompt builders  (prompt string only — answer injected per candidate)
# ---------------------------------------------------------------------------

def _build_mmmlu_prompt(record: dict) -> str:
    """Return the prompt string (without the answer letter)."""
    subject     = record.get("Subject", "general knowledge").replace("_", " ")
    question    = record["Question"].strip()
    options_str = "\n".join(
        f"{letter}. {record[letter]}" for letter in MCQ_OPTIONS if letter in record
    )
    return (
        f"The following is a multiple choice question about {subject}.\n\n"
        f"Question: {question}\n"
        f"{options_str}\n"
        f"Answer:"
    )


def _build_xnli_prompt(record: dict) -> str:
    """Return the prompt string (without the label word)."""
    premise    = record["premise"].strip()
    hypothesis = record["hypothesis"].strip()
    return (
        f"Determine the logical relationship between the premise and hypothesis.\n"
        f"Choose one of: entailment, neutral, contradiction.\n\n"
        f"Premise: {premise}\n"
        f"Hypothesis: {hypothesis}\n"
        f"Relationship:"
    )


def _build_xsquad_prompt(record: dict) -> str:
    """Return prompt string for generation (answer withheld)."""
    context  = record["context"].strip()
    question = record["question"].strip()
    return (
        f"Read the following passage carefully and answer the question based on it.\n\n"
        f"Passage: {context}\n\n"
        f"Question: {question}\n"
        f"Answer:"
    )


# ---------------------------------------------------------------------------
# Base multi-candidate Dataset
# ---------------------------------------------------------------------------

class _BaseCandidateDataset(Dataset):
    """
    Flat multi-candidate dataset cho log-prob scoring.

    Mỗi sample gốc được mở rộng thành ``num_candidates`` rows.
    Row thứ ``c`` của sample thứ ``s`` có:
        sample_id    = s
        cand_id      = c
        gold_cand_id = ground-truth candidate index (same for all c in group s)

    Token layout (mỗi row):
        input_ids : [BOS] <prompt tokens> <candidate tokens> [EOS]
        labels    : -100 ... -100          <candidate>       [EOS]

    Nếu vượt max_length, prompt bị truncate từ trái (giữ nguyên candidate tokens).
    """

    task_name: str = "base"

    def __init__(
        self,
        records: List[dict],                    # mỗi record có "_lang", "_gold_cand_id"
        candidates: List[str],                  # ví dụ [" A", " B", " C", " D"]
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 512,
    ) -> None:
        self.records    = records
        self.candidates = candidates            # List[str], đã có leading space
        self.tokenizer  = tokenizer
        self.max_length = max_length
        self.num_candidates = len(candidates)

        self.bos = [tokenizer.bos_token_id] if tokenizer.bos_token_id is not None else []
        self.eos = [tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else []

        # Mở rộng: mỗi sample → num_candidates rows
        # self._items[i] = (sample_idx, cand_idx)
        self._items: List[tuple] = [
            (s, c)
            for s in range(len(records))
            for c in range(self.num_candidates)
        ]

    # ── abstract ────────────────────────────────────────────────────────────

    def _build_prompt(self, record: dict) -> str:
        raise NotImplementedError

    # ── helpers ─────────────────────────────────────────────────────────────

    def _encode(self, text: str) -> List[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    # ── Dataset interface ────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> Dict:
        sample_idx, cand_idx = self._items[idx]
        record = self.records[sample_idx]
        lang   = record["_lang"]
        gold   = record["_gold_cand_id"]

        prompt_str = self._build_prompt(record)
        cand_str   = self.candidates[cand_idx]   # e.g. " B" or " entailment"

        prompt_ids = self._encode(prompt_str)
        cand_ids   = self._encode(cand_str)

        # Full sequence: BOS + prompt + candidate + EOS
        full_ids = self.bos + prompt_ids + cand_ids + self.eos

        # Truncate prompt từ trái nếu vượt budget
        if len(full_ids) > self.max_length:
            budget     = self.max_length - len(self.bos) - len(cand_ids) - len(self.eos)
            prompt_ids = prompt_ids[-max(0, budget):]
            full_ids   = self.bos + prompt_ids + cand_ids + self.eos

        # Labels: -100 trên BOS + prompt; candidate + EOS là answer
        prompt_len = len(self.bos) + len(prompt_ids)
        labels     = [-100] * prompt_len + cand_ids + self.eos

        assert len(full_ids) == len(labels)

        return {
            "input_ids":      torch.tensor(full_ids, dtype=torch.long),
            "attention_mask": torch.ones(len(full_ids), dtype=torch.long),
            "labels":         torch.tensor(labels,   dtype=torch.long),
            "sample_id":      torch.tensor(sample_idx, dtype=torch.long),
            "cand_id":        torch.tensor(cand_idx,   dtype=torch.long),
            "gold_cand_id":   torch.tensor(gold,       dtype=torch.long),
            "task":           self.task_name,
            "lang":           lang,
        }


# ---------------------------------------------------------------------------
# MMMLU Dataset
# ---------------------------------------------------------------------------

class MMLUDownstreamDataset(_BaseCandidateDataset):
    """
    Dataset cho MMMLU log-prob evaluation.

    Candidates: [" A", " B", " C", " D"]  (leading space → tránh tokenisation artifact)
    gold_cand_id: MCQ_OPTIONS.index(record["Answer"])  e.g. "B" → 1
    """
    task_name = "mmmlu"

    def __init__(
        self,
        records: List[dict],
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 512,
    ) -> None:
        # Candidates với leading space (chuẩn LM-Eval-Harness)
        candidates = [f" {letter}" for letter in MCQ_OPTIONS]
        super().__init__(records, candidates, tokenizer, max_length)

    def _build_prompt(self, record: dict) -> str:
        return _build_mmmlu_prompt(record)


def _load_mmmlu(mmmlu_dir: Path) -> List[dict]:
    """
    Load tất cả ngôn ngữ từ MMMLU; inject _lang và _gold_cand_id vào mỗi record.
    _gold_cand_id = MCQ_OPTIONS.index(answer_letter)   e.g. "B" → 1
    """
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
            r["_lang"]         = lang_dir.name
            r["_gold_cand_id"] = MCQ_OPTIONS.index(r["Answer"].strip())
        records.extend(valid)
        print(f"[MMMLU] Loaded {len(valid):,} records from {lang_dir.name}/test.json")

    return records


# ---------------------------------------------------------------------------
# XNLI Dataset
# ---------------------------------------------------------------------------

class XNLIDataset(_BaseCandidateDataset):
    """
    Dataset cho XNLI log-prob evaluation.

    Candidates: [" entailment", " neutral", " contradiction"]
    gold_cand_id: record["label"]  (0/1/2 đã map trực tiếp với XNLI_CANDIDATES)
    """
    task_name = "xnli"

    def __init__(
        self,
        records: List[dict],
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 256,
    ) -> None:
        candidates = [f" {w}" for w in XNLI_CANDIDATES]
        super().__init__(records, candidates, tokenizer, max_length)

    def _build_prompt(self, record: dict) -> str:
        return _build_xnli_prompt(record)


def _load_xnli(xnli_dir: Path) -> List[dict]:
    """
    Load tất cả ngôn ngữ từ XNLI (test.json only — zero-shot eval).
    Inject _lang và _gold_cand_id (= record["label"] ∈ {0,1,2}).
    Records với label == -1 bị lọc.
    """
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
            r["_lang"]         = lang_dir.name
            r["_gold_cand_id"] = r["label"]   # 0=entailment,1=neutral,2=contradiction
        records.extend(valid)
        print(f"[XNLI] Loaded {len(valid):,} records from {lang_dir.name}/test.json")

    return records


# ---------------------------------------------------------------------------
# SortedLengthSampler
# ---------------------------------------------------------------------------

class SortedLengthSampler(Sampler):
    """
    Bucket-based sampler — sort by proxy length để giảm padding.

    Lưu ý: với multi-candidate dataset, ``lengths`` được tính theo sample gốc
    (index s) và mở rộng theo thứ tự (s,0), (s,1), ..., (s,C-1) để đảm bảo
    các candidates của cùng sample nằm cạnh nhau trong batch khi shuffle=False.
    """

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
# Collate functions
# ---------------------------------------------------------------------------

def _collate_fn_candidates(batch: List[Dict], pad_token_id: int) -> Dict:
    """
    Right-pad input_ids / attention_mask / labels cho log-prob batches.

    Trả thêm:
        sample_id    LongTensor [B*C]
        cand_id      LongTensor [B*C]
        gold_cand_id LongTensor [B*C]
    """
    max_len = max(s["input_ids"].size(0) for s in batch)
    input_ids_list, mask_list, labels_list = [], [], []

    for s in batch:
        n   = s["input_ids"].size(0)
        pad = max_len - n
        input_ids_list.append(torch.cat([
            s["input_ids"], torch.full((pad,), pad_token_id, dtype=torch.long)
        ]))
        mask_list.append(torch.cat([
            s["attention_mask"], torch.zeros(pad, dtype=torch.long)
        ]))
        labels_list.append(torch.cat([
            s["labels"], torch.full((pad,), -100, dtype=torch.long)
        ]))

    return {
        "input_ids":      torch.stack(input_ids_list),              # [B*C, L]
        "attention_mask": torch.stack(mask_list),                    # [B*C, L]
        "labels":         torch.stack(labels_list),                  # [B*C, L]
        "sample_id":      torch.stack([s["sample_id"]   for s in batch]),  # [B*C]
        "cand_id":        torch.stack([s["cand_id"]     for s in batch]),  # [B*C]
        "gold_cand_id":   torch.stack([s["gold_cand_id"] for s in batch]), # [B*C]
        "task":           [s["task"] for s in batch],
        "lang":           [s["lang"] for s in batch],
    }


def _collate_fn_xsquad(batch: List[Dict], pad_token_id: int) -> Dict:
    """Left-pad cho XSQuAD generation batches."""
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
    return {
        "input_ids":      torch.stack(input_ids_list),
        "attention_mask": torch.stack(mask_list),
        "answers":        [s["answers"] for s in batch],
        "task":           [s["task"]    for s in batch],
        "lang":           [s["lang"]    for s in batch],
    }


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


def _build_candidate_loader(
    torch_ds: _BaseCandidateDataset,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int,
    pin_memory: bool,
    length_fn,          # Callable[[dict], int] — nhận record gốc
) -> tuple:
    """
    Build SortedLengthSampler + DataLoader cho multi-candidate dataset.

    batch_size ở đây là số *sample gốc* per batch → DataLoader nhận
    batch_size * num_candidates rows thực tế.

    Sampler hoạt động trên self._items (len = N * C) nhưng lengths
    được tính theo sample gốc để sort nhất quán.
    """
    C       = torch_ds.num_candidates
    # Lengths theo từng item (s, c): dùng length của sample gốc
    lengths = [
        length_fn(torch_ds.records[s])
        for s, _c in tqdm(torch_ds._items, desc="[Sampler] pre-computing lengths", leave=False)
    ]
    sampler = SortedLengthSampler(lengths, batch_size * C, shuffle, seed)
    pad_id  = torch_ds.tokenizer.pad_token_id
    loader  = DataLoader(
        torch_ds,
        batch_size=batch_size * C,      # C rows per sample → batch_size samples per batch
        sampler=sampler,
        collate_fn=lambda b: _collate_fn_candidates(b, pad_token_id=pad_id),
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    return sampler, loader


# ---------------------------------------------------------------------------
# MMMLU DataLoader
# ---------------------------------------------------------------------------

class MMLUDownstreamDataLoader:
    """
    Evaluation DataLoader cho MMMLU — log-prob scoring với 4 candidates.

    Mỗi batch chứa ``batch_size × 4`` rows (4 candidates per sample).
    Dùng ``sample_id`` / ``cand_id`` / ``gold_cand_id`` để tính accuracy
    sau khi collect toàn bộ mean-NLL scores (xem docstring module).

    Parameters
    ----------
    data_root      : path đến raw_data/
    tokenizer      : HF tokeniser; nếu None load DEFAULT_MODEL
    batch_size     : số sample gốc per batch (thực tế = batch_size * 4 rows)
    max_length     : độ dài tối đa sequence
    shuffle        : shuffle (mặc định False cho eval)
    seed           : RNG seed
    num_workers    : DataLoader workers
    pin_memory     : CUDA pin memory

    Attributes
    ----------
    num_candidates : 4  (A / B / C / D)

    Batch keys
    ----------
    input_ids       LongTensor [B*4, L]
    attention_mask  LongTensor [B*4, L]
    labels          LongTensor [B*4, L]   -100 trên prompt; candidate token(s) + EOS
    sample_id       LongTensor [B*4]      index sample gốc (0-indexed, global)
    cand_id         LongTensor [B*4]      0=A, 1=B, 2=C, 3=D
    gold_cand_id    LongTensor [B*4]      ground-truth candidate index
    task            List[str]             ["mmmlu", ...]
    lang            List[str]             language code
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
        self.tokenizer  = _get_tokenizer(tokenizer)
        torch_ds = MMLUDownstreamDataset(records, self.tokenizer, max_length)
        self._sampler, self._loader = _build_candidate_loader(
            torch_ds, batch_size, shuffle, seed, num_workers, pin_memory,
            length_fn=lambda r: len(r.get("Question", "")) + sum(
                len(r.get(k, "")) for k in MCQ_OPTIONS
            ),
        )
        self.num_candidates = torch_ds.num_candidates
        langs = sorted({r["_lang"] for r in records})
        print(
            f"[MMLUDownstreamDataLoader]  samples={len(records):,}  "
            f"rows(×{self.num_candidates})={len(torch_ds):,}  "
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
# XNLI DataLoader
# ---------------------------------------------------------------------------

class XNLIDataLoader:
    """
    Evaluation DataLoader cho XNLI — log-prob scoring với 3 candidates.

    Mỗi batch chứa ``batch_size × 3`` rows (3 label words per sample).
    Dùng ``sample_id`` / ``cand_id`` / ``gold_cand_id`` để tính accuracy
    sau khi collect toàn bộ mean-NLL scores (xem docstring module).

    Parameters
    ----------
    data_root      : path đến raw_data/
    tokenizer      : HF tokeniser; nếu None load DEFAULT_MODEL
    batch_size     : số sample gốc per batch (thực tế = batch_size * 3 rows)
    max_length     : độ dài tối đa sequence
    shuffle        : shuffle (mặc định False cho eval)
    seed           : RNG seed
    num_workers    : DataLoader workers
    pin_memory     : CUDA pin memory

    Attributes
    ----------
    num_candidates : 3  (entailment / neutral / contradiction)

    Batch keys
    ----------
    input_ids       LongTensor [B*3, L]
    attention_mask  LongTensor [B*3, L]
    labels          LongTensor [B*3, L]   -100 trên prompt; label word token(s) + EOS
    sample_id       LongTensor [B*3]      index sample gốc (0-indexed, global)
    cand_id         LongTensor [B*3]      0=entailment, 1=neutral, 2=contradiction
    gold_cand_id    LongTensor [B*3]      ground-truth candidate index
    task            List[str]             ["xnli", ...]
    lang            List[str]             language code
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
        self._sampler, self._loader = _build_candidate_loader(
            torch_ds, batch_size, shuffle, seed, num_workers, pin_memory,
            length_fn=lambda r: len(r.get("premise", "")) + len(r.get("hypothesis", "")),
        )
        self.num_candidates = torch_ds.num_candidates
        langs = sorted({r["_lang"] for r in records})
        print(
            f"[XNLIDataLoader]  samples={len(records):,}  "
            f"rows(×{self.num_candidates})={len(torch_ds):,}  "
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
# XSQuAD  (unchanged — generation mode)
# ---------------------------------------------------------------------------

class XSQuADDataset(Dataset):
    """Generation-mode dataset cho XSQuAD."""

    task_name = "xsquad"

    def __init__(
        self,
        records: List[dict],
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 1024,
    ) -> None:
        self.records    = records
        self.tokenizer  = tokenizer
        self.max_length = max_length
        self.bos = [tokenizer.bos_token_id] if tokenizer.bos_token_id is not None else []

    def _encode(self, text: str) -> List[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict:
        record = self.records[idx]
        lang   = record["_lang"]

        prompt_ids = self._encode(_build_xsquad_prompt(record))
        budget     = self.max_length - len(self.bos)
        prompt_ids = prompt_ids[-max(0, budget):]
        full_ids   = self.bos + prompt_ids

        answers = record["answers"].get("text", [])
        answers = [a.strip() for a in answers if a.strip()] or ["no answer"]

        return {
            "input_ids":      torch.tensor(full_ids, dtype=torch.long),
            "attention_mask": torch.ones(len(full_ids), dtype=torch.long),
            "answers":        answers,
            "task":           self.task_name,
            "lang":           lang,
        }


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
            r["_lang"] = lang_dir.name
        records.extend(valid)
        print(
            f"[XSQuAD] Loaded {len(valid):,} from {lang_dir.name}/validation.json "
            f"(answerable={len(valid)-unanswerable:,}, unanswerable={unanswerable:,})"
        )
    return records


class XSQuADDataLoader:
    """Evaluation DataLoader cho XSQuAD — generation mode."""

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
        lengths    = [len(r.get("context", "")) for r in records]
        sampler    = SortedLengthSampler(lengths, batch_size, shuffle, seed)
        pad_id     = self.tokenizer.pad_token_id
        self._sampler = sampler
        self._loader  = DataLoader(
            torch_ds,
            batch_size=batch_size,
            sampler=sampler,
            collate_fn=lambda b: _collate_fn_xsquad(b, pad_token_id=pad_id),
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
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
# Utility: compute accuracy từ collected NLL scores
# ---------------------------------------------------------------------------

def compute_logprob_accuracy(
    all_nll: torch.Tensor,       # [N_total]  float
    all_sample_id: torch.Tensor, # [N_total]  long
    all_cand_id: torch.Tensor,   # [N_total]  long
    all_gold: torch.Tensor,      # [N_total]  long
    num_candidates: int,
    all_lang: Optional[List[str]] = None,
) -> Dict:
    """
    Tính accuracy từ mean-NLL scores đã collect.

    Returns dict:
        overall_acc   : float
        per_lang_acc  : Dict[str, float]  (nếu all_lang được cung cấp)
    """
    n_samples = int(all_sample_id.max().item()) + 1
    C         = num_candidates

    # Điền NLL matrix [N, C]
    nll_mat  = torch.full((n_samples, C), float("inf"))
    nll_mat[all_sample_id, all_cand_id] = all_nll.float()

    # Gold labels [N]
    gold_mat = torch.zeros(n_samples, dtype=torch.long)
    gold_mat[all_sample_id] = all_gold

    pred        = nll_mat.argmin(-1)                     # [N]
    correct     = (pred == gold_mat)
    overall_acc = correct.float().mean().item()

    result: Dict = {"overall_acc": overall_acc}

    if all_lang is not None:
        # Map sample_id → lang (lấy lang đầu tiên của mỗi sample_id)
        lang_of = [""] * n_samples
        for sid, lg in zip(all_sample_id.tolist(), all_lang):
            lang_of[sid] = lg
        from collections import defaultdict
        per_lang: Dict[str, List[bool]] = defaultdict(list)
        for sid in range(n_samples):
            per_lang[lang_of[sid]].append(correct[sid].item())
        result["per_lang_acc"] = {
            lg: sum(v) / len(v) for lg, v in sorted(per_lang.items())
        }

    return result


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

    # ── Log-prob loaders ─────────────────────────────────────────────────────
    logprob_loaders = {
        "MMMLU": MMLUDownstreamDataLoader(
            data_root=DATA_ROOT, tokenizer=tokenizer, batch_size=8, max_length=512
        ),
        "XNLI":  XNLIDataLoader(
            data_root=DATA_ROOT, tokenizer=tokenizer, batch_size=16, max_length=256
        ),
    }

    print("\n" + "=" * 60)
    print("Smoke-tests: MMMLU / XNLI (log-prob scoring)")
    print("=" * 60)
    for name, loader in logprob_loaders.items():
        C     = loader.num_candidates
        batch = next(iter(loader))

        # ── Shape checks ────────────────────────────────────────────────────
        assert batch["input_ids"].dim()      == 2
        assert batch["attention_mask"].dim() == 2
        assert batch["labels"].dim()         == 2
        assert batch["input_ids"].dtype      == torch.long
        assert batch["labels"].dtype         == torch.long
        assert (batch["labels"] == -100).any(),  f"{name}: no -100 prompt mask"
        assert (batch["labels"] != -100).any(),  f"{name}: no candidate tokens"
        assert "answers" not in batch,           f"{name}: should NOT have 'answers'"

        # ── New fields ──────────────────────────────────────────────────────
        assert "sample_id"    in batch
        assert "cand_id"      in batch
        assert "gold_cand_id" in batch
        assert batch["sample_id"].dtype    == torch.long
        assert batch["cand_id"].dtype      == torch.long
        assert batch["gold_cand_id"].dtype == torch.long
        assert batch["cand_id"].max().item() < C,             f"{name}: cand_id out of range"
        assert batch["gold_cand_id"].max().item() < C,        f"{name}: gold out of range"

        print(f"  ✓ {name:6s}  shape={tuple(batch['input_ids'].shape)}  "
              f"C={C}  task={batch['task'][0]}  langs={set(batch['lang'])}")

        # ── In chi tiết sample đầu tiên (candidate 0) ────────────────────────
        print(f"\n  {'─'*20} First sample of {name} (all {C} candidates) {'─'*20}")
        # Tìm tất cả rows thuộc sample_id == 0
        first_sid = batch["sample_id"].min().item()
        mask_s0   = batch["sample_id"] == first_sid
        indices   = mask_s0.nonzero(as_tuple=True)[0]
        for row_i in indices:
            ids_r    = batch["input_ids"][row_i]
            labels_r = batch["labels"][row_i]
            ans_mask = labels_r != -100
            prompt_t = tokenizer.decode(ids_r[~ans_mask], skip_special_tokens=True)
            cand_t   = tokenizer.decode(ids_r[ans_mask],  skip_special_tokens=True)
            gold     = batch["gold_cand_id"][row_i].item()
            cid      = batch["cand_id"][row_i].item()
            marker   = " ← GOLD" if cid == gold else ""
            print(f"  cand_id={cid}  answer='{cand_t}'{marker}")
        # In prompt một lần
        row0 = indices[0]
        ids_r = batch["input_ids"][row0]
        labels_r = batch["labels"][row0]
        prompt_t = tokenizer.decode(ids_r[labels_r == -100], skip_special_tokens=True)
        print(f"  lang={batch['lang'][0]}  seq_len={ids_r.size(0)}")
        print(f"  PROMPT:\n{prompt_t}\n")

    # ── XSQuAD ──────────────────────────────────────────────────────────────
    xsquad_loader = XSQuADDataLoader(
        data_root=DATA_ROOT, tokenizer=tokenizer, batch_size=4, max_length=1024
    )
    print("\n" + "=" * 60)
    print("Smoke-tests: XSQuAD (generation)")
    print("=" * 60)
    batch = next(iter(xsquad_loader))

    assert batch["input_ids"].dim()      == 2
    assert batch["attention_mask"].dim() == 2
    assert "labels"  not in batch,  "XSQuAD: should NOT have 'labels'"
    assert "answers" in batch
    assert isinstance(batch["answers"], list)
    assert all(isinstance(a, list)       for a in batch["answers"])
    assert all(isinstance(s, str) for a in batch["answers"] for s in a)
    assert batch["input_ids"].dtype == torch.long
    assert "lang" in batch
    assert all(t == "xsquad" for t in batch["task"])

    print(f"  ✓ XSQuAD  shape={tuple(batch['input_ids'].shape)}  "
          f"task={batch['task'][0]}  langs={set(batch['lang'])}")
    ids_0  = batch["input_ids"][0]
    mask_0 = batch["attention_mask"][0]
    real_t = tokenizer.decode(ids_0[mask_0 == 1], skip_special_tokens=True)
    print(f"  lang={batch['lang'][0]}  seq_len={ids_0.size(0)}")
    print(f"  PROMPT:\n{real_t}")
    print(f"  ANSWERS: {batch['answers'][0]}")
    assert real_t.strip().endswith("Answer:"), "XSQuAD: prompt should end with 'Answer:'"

    print("\n" + "=" * 60)
    print("✓ All smoke-tests passed.")
    print("=" * 60)