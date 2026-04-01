"""
finetune_dataloader.py
======================
Dataset and DataLoader classes for supervised fine-tuning (SFT) of LLMs on
three English benchmark datasets:

    raw_data/english/MMLU/   — Multiple Choice QA
    raw_data/english/SQuAD/  — Extractive QA
    raw_data/english/SNLI/   — Natural Language Inference

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SFT objective (MMLUDataLoader / SQuADDataLoader / SNLIDataLoader)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
All three tasks are cast as causal LM (loss on answer tokens only):

    [BOS] <prompt tokens> <answer tokens> [EOS]
    labels: -100  ...  -100   <answer>   [EOS]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Evaluation strategy
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

MMLU / SNLI — Log-prob scoring (MMLUValDataLoader / SNLIValDataLoader)
----------------------------------------------------------------------
Each sample is expanded into C rows (C = 4 for MMLU, C = 3 for SNLI).
Row c encodes: [BOS] <prompt> <candidate_c> [EOS]
               labels: -100 ... -100  <candidate_c>  [EOS]

After forward pass:
    shift_logits = logits[:, :-1]               # [B*C, L-1, V]
    shift_labels = labels[:, 1:]                # [B*C, L-1]
    token_loss   = CrossEntropyLoss(reduction='none', ignore_index=-100)
    mean_nll[i]  = token_loss[i].sum() / (shift_labels[i] != -100).sum()
    pred         = argmin over C candidates per sample → Accuracy

    MMLU candidates : [" A", " B", " C", " D"]
    SNLI candidates : [" entailment", " neutral", " contradiction"]
    Leading space avoids tokenisation boundary artifacts.
    Mean NLL (not sum) avoids length bias between candidates.

Batch keys (MMLU / SNLI val):
    input_ids       LongTensor [B*C, L]
    attention_mask  LongTensor [B*C, L]
    labels          LongTensor [B*C, L]   -100 on prompt; candidate token(s) + EOS
    sample_id       LongTensor [B*C]      original sample index (for grouping)
    cand_id         LongTensor [B*C]      candidate index (0..C-1)
    gold_cand_id    LongTensor [B*C]      ground-truth candidate index
    num_candidates  int                   C (4 or 3)
    task            List[str]
    
SQuAD — Generation mode (SQuADValidationDataLoader)
----------------------------------------------------
Returns prompt-only tokens (left-padded) so caller can run model.generate()
and score against ALL gold answer strings with F1 / EM.

Batch keys (SQuAD val):
    input_ids       LongTensor [B, L]   prompt only (left-padded)
    attention_mask  LongTensor [B, L]
    answers         List[List[str]]     all valid gold spans per sample
    task            List[str]           ["squad", ...]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Eval loop example (MMLU / SNLI)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    all_nll, all_sid, all_cid, all_gold = [], [], [], []
    for batch in val_loader:
        with torch.no_grad():
            logits = model(input_ids=batch["input_ids"].to(device),
                           attention_mask=batch["attention_mask"].to(device)).logits
        shift_logits = logits[:, :-1].contiguous()
        shift_labels = batch["labels"][:, 1:].to(device).contiguous()
        loss_fct  = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=-100)
        token_loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        ).view(shift_labels.size())                   # [B*C, L-1]
        n_ans    = (shift_labels != -100).sum(-1).clamp(min=1)
        mean_nll = token_loss.sum(-1) / n_ans         # [B*C]
        all_nll.append(mean_nll.cpu())
        all_sid.append(batch["sample_id"])
        all_cid.append(batch["cand_id"])
        all_gold.append(batch["gold_cand_id"])

    result = compute_logprob_accuracy(
        torch.cat(all_nll), torch.cat(all_sid),
        torch.cat(all_cid), torch.cat(all_gold),
        num_candidates=val_loader.num_candidates,
    )
    print(result["overall_acc"])

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MMLU normalisation (applied at load time in _load_mmlu)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Raw HuggingFace MMLU records use an integer answer index and plain choice strings.
After loading, every record is normalised in-place:
    record["answer"]  : int  →  str   e.g. 1  →  "B"
    record["choices"] : List[str]  →  prefixed  e.g. ["A. Paris", "B. Rome", ...]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Mixed DataLoader — balanced batching strategy
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MixedFinetuneDataLoader combines all 3 tasks into a single DataLoader:
  1. n_balanced = min(len(mmlu), len(squad), len(snli)), rounded down to
     nearest multiple of per_task = batch_size // 3.
  2. Balanced pool: interleave [mmlu, squad, snli, ...]; each mixed batch has
     per_task samples from each task.
  3. Remainder pools: single-task batches of size batch_size.
  4. Yield order: mixed batches first, then remainder batches per task.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Public API
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    SFT loaders  : MMLUDataLoader / SQuADDataLoader / SNLIDataLoader
    Val loaders  : MMLUValDataLoader / SNLIValDataLoader / SQuADValidationDataLoader
    Mixed SFT    : MixedFinetuneDataLoader
    Utility      : compute_logprob_accuracy
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

SNLI_LABEL_MAP   = {0: "entailment", 1: "neutral", 2: "contradiction"}
SNLI_CANDIDATES  = ["entailment", "neutral", "contradiction"]   # cand_id: 0/1/2

MMLU_OPTIONS     = ["A", "B", "C", "D"]


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
# Prompt builders (return prompt string only; answer injected per-candidate)
# ---------------------------------------------------------------------------

def _build_mmlu_prompt_str(record: dict) -> str:
    """Return prompt string without answer letter (for val multi-candidate)."""
    subject     = record.get("subject", "general knowledge").replace("_", " ")
    question    = record["question"].strip()
    options_str = "\n".join(record["choices"])      # already prefixed by normalisation
    return (
        f"The following is a multiple choice question about {subject}.\n\n"
        f"Question: {question}\n"
        f"{options_str}\n"
        f"Answer:"
    )


def _build_snli_prompt_str(record: dict) -> str:
    """Return prompt string without label word (for val multi-candidate)."""
    return (
        f"Determine the logical relationship between the premise and hypothesis.\n"
        f"Choose one of: entailment, neutral, contradiction.\n\n"
        f"Premise: {record['premise'].strip()}\n"
        f"Hypothesis: {record['hypothesis'].strip()}\n"
        f"Relationship:"
    )


# SFT prompt builders — return (prompt_str, answer_str) for _BaseFinetuneDataset
def _build_mmlu_prompt(record: dict) -> tuple[str, str]:
    return _build_mmlu_prompt_str(record), f" {record['answer']}"


def _build_squad_prompt(record: dict) -> tuple[str, str]:
    context     = record["context"].strip()
    question    = record["question"].strip()
    answers     = record["answers"]
    answer_text = answers["text"][0].strip() if answers.get("text") else "no answer"
    prompt = (
        f"Read the following passage carefully and answer the question based on it.\n\n"
        f"Passage: {context}\n\n"
        f"Question: {question}\n"
        f"Answer:"
    )
    return prompt, f" {answer_text}"


def _build_snli_prompt(record: dict) -> tuple[str, str]:
    label_word = SNLI_LABEL_MAP[record["label"]]
    return _build_snli_prompt_str(record), f" {label_word}"


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
# Base SFT Dataset (shared tokenisation logic — unchanged from original)
# ---------------------------------------------------------------------------

class _BaseFinetuneDataset(Dataset):
    """
    SFT dataset: every sample encodes a single (prompt, answer) pair.
        [BOS] <prompt> <answer> [EOS]
        labels: -100 ... -100  <answer> [EOS]
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
        self.eos = [tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else []

    def _build_prompt_answer(self, record: dict) -> tuple[str, str]:
        raise NotImplementedError

    def _encode(self, text: str) -> List[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict:
        record = self.records[idx]
        prompt_str, answer_str = self._build_prompt_answer(record)
        prompt_ids = self._encode(prompt_str)
        answer_ids = self._encode(answer_str)
        full_ids   = self.bos + prompt_ids + answer_ids + self.eos
        if len(full_ids) > self.max_length:
            budget     = self.max_length - len(self.bos) - len(answer_ids) - len(self.eos)
            prompt_ids = prompt_ids[-max(0, budget):]
            full_ids   = self.bos + prompt_ids + answer_ids + self.eos
        prompt_len = len(self.bos) + len(prompt_ids)
        labels     = [-100] * prompt_len + answer_ids + self.eos
        assert len(full_ids) == len(labels)
        return {
            "input_ids":      torch.tensor(full_ids, dtype=torch.long),
            "attention_mask": torch.ones(len(full_ids), dtype=torch.long),
            "labels":         torch.tensor(labels,   dtype=torch.long),
            "task":           self.task_name,
        }


# ---------------------------------------------------------------------------
# Base multi-candidate Val Dataset (MMLU / SNLI evaluation)
# ---------------------------------------------------------------------------

class _BaseCandidateValDataset(Dataset):
    """
    Multi-candidate dataset for log-prob evaluation.

    Each sample is expanded into ``num_candidates`` rows.
    Row c of sample s encodes:
        input_ids : [BOS] <prompt> <candidate_c> [EOS]
        labels    : -100 ... -100  <candidate_c>  [EOS]

    Extra fields per row:
        sample_id    : original sample index s  (for grouping C rows → 1 prediction)
        cand_id      : candidate index c  (0..C-1)
        gold_cand_id : ground-truth candidate index  (same for all C rows of sample s)

    If full sequence > max_length, prompt is truncated from the left.
    """

    task_name: str = "base_val"

    def __init__(
        self,
        records: List[dict],            # each record must have "_gold_cand_id"
        candidates: List[str],          # e.g. [" A", " B", " C", " D"]
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 512,
    ) -> None:
        self.records        = records
        self.candidates     = candidates
        self.num_candidates = len(candidates)
        self.tokenizer      = tokenizer
        self.max_length     = max_length
        self.bos = [tokenizer.bos_token_id] if tokenizer.bos_token_id is not None else []
        self.eos = [tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else []
        # Flat index: item i → (sample_idx, cand_idx)
        self._items = [
            (s, c)
            for s in range(len(records))
            for c in range(self.num_candidates)
        ]

    def _build_prompt(self, record: dict) -> str:
        raise NotImplementedError

    def _encode(self, text: str) -> List[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> Dict:
        sample_idx, cand_idx = self._items[idx]
        record = self.records[sample_idx]

        prompt_ids = self._encode(self._build_prompt(record))
        cand_ids   = self._encode(self.candidates[cand_idx])

        full_ids = self.bos + prompt_ids + cand_ids + self.eos
        if len(full_ids) > self.max_length:
            budget     = self.max_length - len(self.bos) - len(cand_ids) - len(self.eos)
            prompt_ids = prompt_ids[-max(0, budget):]
            full_ids   = self.bos + prompt_ids + cand_ids + self.eos

        prompt_len = len(self.bos) + len(prompt_ids)
        labels     = [-100] * prompt_len + cand_ids + self.eos

        assert len(full_ids) == len(labels)

        return {
            "input_ids":      torch.tensor(full_ids, dtype=torch.long),
            "attention_mask": torch.ones(len(full_ids), dtype=torch.long),
            "labels":         torch.tensor(labels,   dtype=torch.long),
            "sample_id":      torch.tensor(sample_idx,               dtype=torch.long),
            "cand_id":        torch.tensor(cand_idx,                 dtype=torch.long),
            "gold_cand_id":   torch.tensor(record["_gold_cand_id"],  dtype=torch.long),
            "task":           self.task_name,
        }


# ---------------------------------------------------------------------------
# MMLU — SFT Dataset / DataLoader
# ---------------------------------------------------------------------------

class MMLUDataset(_BaseFinetuneDataset):
    task_name = "mmlu"
    def _build_prompt_answer(self, record: dict) -> tuple[str, str]:
        return _build_mmlu_prompt(record)


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
            raise ValueError(f"Unknown MMLU split '{split}'. Choose from: {list(split_to_file)}")
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


class MMLUDataLoader:
    """
    Single-task DataLoader for MMLU **SFT** (answer letter, loss on single token).

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
    def dataset(self) -> MMLUDataset: return self._loader.dataset


# ---------------------------------------------------------------------------
# MMLU — Validation Dataset / DataLoader  (log-prob scoring)
# ---------------------------------------------------------------------------

class MMLUValDataset(_BaseCandidateValDataset):
    """
    Multi-candidate val dataset for MMLU.
    Candidates: [" A", " B", " C", " D"]
    gold_cand_id = MMLU_OPTIONS.index(record["answer"])
    """
    task_name = "mmlu"

    def __init__(self, records: List[dict], tokenizer: PreTrainedTokenizerBase,
                 max_length: int = 512) -> None:
        candidates = [f" {letter}" for letter in MMLU_OPTIONS]
        super().__init__(records, candidates, tokenizer, max_length)

    def _build_prompt(self, record: dict) -> str:
        return _build_mmlu_prompt_str(record)


def _inject_mmlu_gold(records: List[dict]) -> List[dict]:
    """Inject _gold_cand_id into already-normalised MMLU records."""
    for r in records:
        r["_gold_cand_id"] = MMLU_OPTIONS.index(r["answer"])
    return records


class MMLUValDataLoader:
    """
    Evaluation DataLoader for MMLU — **log-prob scoring**, 4 candidates per sample.

    Each batch contains ``batch_size × 4`` rows (one per candidate letter).
    Use ``sample_id`` / ``cand_id`` / ``gold_cand_id`` + ``compute_logprob_accuracy``
    to obtain accuracy after collecting all mean-NLL scores.

    Parameters
    ----------
    data_root   : path to raw_data/english/
    splits      : default ["validation"]
    tokenizer   : HF tokeniser; if None loads DEFAULT_MODEL
    batch_size  : number of *original samples* per batch (actual rows = batch_size × 4)
    max_length  : max total sequence length
    shuffle     : default False for eval
    seed / num_workers / pin_memory : standard params

    Attributes
    ----------
    num_candidates : 4

    Batch keys
    ----------
    input_ids       LongTensor [B*4, L]
    attention_mask  LongTensor [B*4, L]
    labels          LongTensor [B*4, L]   -100 on prompt; candidate token + EOS
    sample_id       LongTensor [B*4]      original sample index
    cand_id         LongTensor [B*4]      0=A, 1=B, 2=C, 3=D
    gold_cand_id    LongTensor [B*4]      ground-truth candidate index
    task            List[str]             ["mmlu", ...]
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
        records  = _inject_mmlu_gold(_load_mmlu(mmlu_dir, splits))
        self.tokenizer = _get_tokenizer(tokenizer)
        torch_ds = MMLUValDataset(records, self.tokenizer, max_length)
        self._sampler, self._loader = _build_candidate_loader(
            torch_ds, batch_size, shuffle, seed, num_workers, pin_memory,
            length_fn=lambda r: len(r["question"]) + sum(len(c) for c in r.get("choices", [])),
        )
        self.num_candidates = torch_ds.num_candidates
        print(f"[MMLUValDataLoader] splits={splits}  samples={len(records):,}  "
              f"rows(×{self.num_candidates})={len(torch_ds):,}  "
              f"batches={len(self._loader):,}  batch_size={batch_size}")

    def set_epoch(self, epoch: int) -> None: self._sampler.set_epoch(epoch)
    def __iter__(self): return iter(self._loader)
    def __len__(self) -> int: return len(self._loader)
    @property
    def dataset(self) -> MMLUValDataset: return self._loader.dataset


# ---------------------------------------------------------------------------
# SQuAD — SFT Dataset / DataLoader
# ---------------------------------------------------------------------------

class SQuADDataset(_BaseFinetuneDataset):
    task_name = "squad"
    def _build_prompt_answer(self, record: dict) -> tuple[str, str]:
        return _build_squad_prompt(record)


def _load_squad(squad_dir: Path, splits: List[str]) -> List[dict]:
    split_to_file = {"train": "train.json", "validation": "validation.json"}
    records: List[dict] = []
    for split in splits:
        fname = split_to_file.get(split)
        if fname is None:
            raise ValueError(f"Unknown SQuAD split '{split}'. Choose from: {list(split_to_file)}")
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


class SQuADDataLoader:
    """
    Single-task DataLoader for SQuAD **SFT** only.

    ⚠️  Do NOT use for F1/EM evaluation — encodes only answers["text"][0].
        Use SQuADValidationDataLoader for evaluation.

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
    def dataset(self) -> SQuADDataset: return self._loader.dataset


# ---------------------------------------------------------------------------
# SQuAD — Validation Dataset / DataLoader  (generation mode for F1 / EM)
# ---------------------------------------------------------------------------

class SQuADValidationDataset(Dataset):
    """
    Prompt-only dataset for SQuAD generation eval.
    Returns ALL gold answer strings for multi-reference F1/EM scoring.

    Token layout:
        input_ids : [BOS] <prompt tokens>    ← NO answer, NO EOS
        attention_mask : 1 ... 1

    Batch keys (via SQuADValidationDataLoader):
        input_ids       LongTensor [B, L]  (left-padded)
        attention_mask  LongTensor [B, L]
        answers         List[List[str]]    all valid gold spans per sample
        task            List[str]          ["squad", ...]
    """

    task_name = "squad"

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

    def __len__(self) -> int: return len(self.records)

    def __getitem__(self, idx: int) -> Dict:
        record = self.records[idx]
        prompt_str, _ = _build_squad_prompt(record)
        prompt_ids    = self._encode(prompt_str)
        budget        = self.max_length - len(self.bos)
        prompt_ids    = prompt_ids[-max(0, budget):]
        full_ids      = self.bos + prompt_ids

        answers = record["answers"].get("text", [])
        answers = [a.strip() for a in answers if a.strip()] or ["no answer"]

        return {
            "input_ids":      torch.tensor(full_ids, dtype=torch.long),
            "attention_mask": torch.ones(len(full_ids), dtype=torch.long),
            "answers":        answers,
            "task":           self.task_name,
        }


def _collate_fn_squad_val(batch: List[Dict], pad_token_id: int) -> Dict:
    """Left-pad collate for SQuAD generation batches."""
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
    }


class SQuADValidationDataLoader:
    """
    Validation DataLoader for SQuAD — **generation mode** for F1 / EM scoring.

    Returns prompt-only batches (left-padded). Run model.generate() on each
    batch and score against the ``answers`` field using F1 / EM.

    Usage
    -----
        val_loader = SQuADValidationDataLoader(data_root=..., tokenizer=tok)
        for batch in val_loader:
            gen_ids = model.generate(
                batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                max_new_tokens=64,
            )
            prompt_len = batch["input_ids"].size(1)
            for i, gen in enumerate(gen_ids):
                pred  = tokenizer.decode(gen[prompt_len:], skip_special_tokens=True).strip()
                golds = batch["answers"][i]          # List[str]
                em    = compute_exact(pred, golds)
                f1    = compute_f1(pred, golds)

    Parameters
    ----------
    data_root   : path to raw_data/english/
    splits      : default ["validation"]
    tokenizer   : HF tokeniser; if None loads DEFAULT_MODEL
    batch_size  : samples per batch
    max_length  : max prompt length (recommend 1024+)
    shuffle     : default False for eval
    seed / num_workers / pin_memory : standard params

    Batch keys
    ----------
    input_ids       LongTensor [B, L]   prompt only (left-padded)
    attention_mask  LongTensor [B, L]
    answers         List[List[str]]     all valid gold answer strings
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
        torch_ds  = SQuADValidationDataset(records, self.tokenizer, max_length)
        lengths   = [len(r.get("context", "")) for r in records]
        sampler   = SortedLengthSampler(lengths, batch_size, shuffle, seed)
        pad_id    = self.tokenizer.pad_token_id
        self._sampler = sampler
        self._loader  = DataLoader(
            torch_ds,
            batch_size=batch_size,
            sampler=sampler,
            collate_fn=lambda b: _collate_fn_squad_val(b, pad_token_id=pad_id),
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        )
        print(f"[SQuADValidationDataLoader] splits={splits}  records={len(records):,}  "
              f"batches={len(self._loader):,}  batch_size={batch_size}")

    def set_epoch(self, epoch: int) -> None: self._sampler.set_epoch(epoch)
    def __iter__(self): return iter(self._loader)
    def __len__(self) -> int: return len(self._loader)
    @property
    def dataset(self) -> SQuADValidationDataset: return self._loader.dataset


# ---------------------------------------------------------------------------
# SNLI — SFT Dataset / DataLoader
# ---------------------------------------------------------------------------

class SNLIDataset(_BaseFinetuneDataset):
    task_name = "snli"
    def _build_prompt_answer(self, record: dict) -> tuple[str, str]:
        return _build_snli_prompt(record)


def _load_snli(snli_dir: Path, splits: List[str]) -> List[dict]:
    split_to_file = {
        "train": "train.json", "test": "test.json", "validation": "validation.json"
    }
    records: List[dict] = []
    for split in splits:
        fname = split_to_file.get(split)
        if fname is None:
            raise ValueError(f"Unknown SNLI split '{split}'. Choose from: {list(split_to_file)}")
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


class SNLIDataLoader:
    """
    Single-task DataLoader for SNLI **SFT** (label-word prediction).

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
    def dataset(self) -> SNLIDataset: return self._loader.dataset


# ---------------------------------------------------------------------------
# SNLI — Validation Dataset / DataLoader  (log-prob scoring)
# ---------------------------------------------------------------------------

class SNLIValDataset(_BaseCandidateValDataset):
    """
    Multi-candidate val dataset for SNLI.
    Candidates: [" entailment", " neutral", " contradiction"]
    gold_cand_id = record["label"]  (0/1/2)
    """
    task_name = "snli"

    def __init__(self, records: List[dict], tokenizer: PreTrainedTokenizerBase,
                 max_length: int = 256) -> None:
        candidates = [f" {w}" for w in SNLI_CANDIDATES]
        super().__init__(records, candidates, tokenizer, max_length)

    def _build_prompt(self, record: dict) -> str:
        return _build_snli_prompt_str(record)


def _inject_snli_gold(records: List[dict]) -> List[dict]:
    for r in records:
        r["_gold_cand_id"] = r["label"]   # 0/1/2 maps directly to SNLI_CANDIDATES
    return records


class SNLIValDataLoader:
    """
    Evaluation DataLoader for SNLI — **log-prob scoring**, 3 candidates per sample.

    Each batch contains ``batch_size × 3`` rows (entailment / neutral / contradiction).
    Use ``sample_id`` / ``cand_id`` / ``gold_cand_id`` + ``compute_logprob_accuracy``
    to obtain accuracy after collecting all mean-NLL scores.

    Parameters
    ----------
    data_root   : path to raw_data/english/
    splits      : default ["validation"]
    tokenizer   : HF tokeniser; if None loads DEFAULT_MODEL
    batch_size  : number of *original samples* per batch (actual rows = batch_size × 3)
    max_length  : max total sequence length
    shuffle     : default False for eval
    seed / num_workers / pin_memory : standard params

    Attributes
    ----------
    num_candidates : 3

    Batch keys
    ----------
    input_ids       LongTensor [B*3, L]
    attention_mask  LongTensor [B*3, L]
    labels          LongTensor [B*3, L]   -100 on prompt; label word token(s) + EOS
    sample_id       LongTensor [B*3]      original sample index
    cand_id         LongTensor [B*3]      0=entailment, 1=neutral, 2=contradiction
    gold_cand_id    LongTensor [B*3]      ground-truth candidate index
    task            List[str]             ["snli", ...]
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
        records  = _inject_snli_gold(_load_snli(snli_dir, splits))
        self.tokenizer = _get_tokenizer(tokenizer)
        torch_ds = SNLIValDataset(records, self.tokenizer, max_length)
        self._sampler, self._loader = _build_candidate_loader(
            torch_ds, batch_size, shuffle, seed, num_workers, pin_memory,
            length_fn=lambda r: len(r["premise"]) + len(r["hypothesis"]),
        )
        self.num_candidates = torch_ds.num_candidates
        print(f"[SNLIValDataLoader] splits={splits}  samples={len(records):,}  "
              f"rows(×{self.num_candidates})={len(torch_ds):,}  "
              f"batches={len(self._loader):,}  batch_size={batch_size}")

    def set_epoch(self, epoch: int) -> None: self._sampler.set_epoch(epoch)
    def __iter__(self): return iter(self._loader)
    def __len__(self) -> int: return len(self._loader)
    @property
    def dataset(self) -> SNLIValDataset: return self._loader.dataset


# ---------------------------------------------------------------------------
# MixedFinetuneDataLoader — balanced multi-task batching (SFT, unchanged)
# ---------------------------------------------------------------------------

class MixedFinetuneDataLoader:
    """
    DataLoader combining MMLU + SQuAD + SNLI with balanced per-batch mixing.
    See module docstring for full batching strategy.

    Parameters
    ----------
    data_root / mmlu_splits / squad_splits / snli_splits / tokenizer
    batch_size       : total per batch (must be >= 3); per_task = batch_size // 3
    *_max_length     : per-task max sequence lengths
    shuffle / seed / num_workers / pin_memory : standard params

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

        self.tokenizer   = _get_tokenizer(tokenizer)
        self.batch_size  = batch_size
        self.shuffle     = shuffle
        self.seed        = seed
        self._epoch      = 0
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
        n_mixed = len(self._balanced_mmlu) // self._per_task
        for i in range(n_mixed):
            s = i * self._per_task
            e = s + self._per_task
            samples = (
                [self._ds_mmlu[j]  for j in bal_mmlu_idx[s:e]]
                + [self._ds_squad[j] for j in bal_squad_idx[s:e]]
                + [self._ds_snli[j]  for j in bal_snli_idx[s:e]]
            )
            yield _collate_fn_sft(samples, self.tokenizer.pad_token_id)
        yield from self._iter_remainder(self._remainder_mmlu,  self._ds_mmlu,  len(self._balanced_mmlu),  rng)
        yield from self._iter_remainder(self._remainder_squad, self._ds_squad, len(self._balanced_squad), rng)
        yield from self._iter_remainder(self._remainder_snli,  self._ds_snli,  len(self._balanced_snli),  rng)

    def _iter_remainder(self, remainder_records, full_dataset, offset, rng):
        if not remainder_records:
            return
        indices = list(range(offset, offset + len(remainder_records)))
        if self.shuffle:
            rng.shuffle(indices)
        for chunk in _chunked(indices, self.batch_size):
            samples = [full_dataset[j] for j in chunk]
            yield _collate_fn_sft(samples, self.tokenizer.pad_token_id)


# ---------------------------------------------------------------------------
# Utility: compute accuracy from collected NLL scores (MMLU / SNLI val)
# ---------------------------------------------------------------------------

def compute_logprob_accuracy(
    all_nll: torch.Tensor,        # [N_total]  float — mean NLL per row
    all_sample_id: torch.Tensor,  # [N_total]  long
    all_cand_id: torch.Tensor,    # [N_total]  long
    all_gold: torch.Tensor,       # [N_total]  long
    num_candidates: int,
) -> Dict:
    """
    Compute accuracy from per-row mean-NLL scores collected over the full val set.

    Algorithm
    ---------
    1. Fill NLL matrix  [N_samples, C]  using (sample_id, cand_id) as index.
    2. pred = argmin over C candidates per row.
    3. accuracy = mean(pred == gold).

    Returns
    -------
    dict with keys:
        overall_acc   float   e.g. 0.612
        n_samples     int
        n_correct     int
    """
    n_samples = int(all_sample_id.max().item()) + 1
    C         = num_candidates

    nll_mat = torch.full((n_samples, C), float("inf"))
    nll_mat[all_sample_id, all_cand_id] = all_nll.float()

    gold_mat = torch.zeros(n_samples, dtype=torch.long)
    gold_mat[all_sample_id] = all_gold

    pred      = nll_mat.argmin(-1)
    correct   = (pred == gold_mat)
    n_correct = int(correct.sum().item())

    return {
        "overall_acc": n_correct / n_samples,
        "n_samples":   n_samples,
        "n_correct":   n_correct,
    }


# ---------------------------------------------------------------------------
# SortedLengthSampler
# ---------------------------------------------------------------------------

class SortedLengthSampler(Sampler):
    """
    Bucket-based sampler: sort by proxy length → group into buckets of batch_size
    → optionally shuffle within/across buckets. Minimises intra-batch padding.
    """

    def __init__(self, lengths: List[int], batch_size: int,
                 shuffle: bool = True, seed: int = 42) -> None:
        self._lengths   = lengths
        self.batch_size = batch_size
        self.shuffle    = shuffle
        self.seed       = seed
        self._epoch     = 0

    def set_epoch(self, epoch: int) -> None: self._epoch = epoch

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

    def __len__(self) -> int: return len(self._lengths)


# ---------------------------------------------------------------------------
# Collate functions
# ---------------------------------------------------------------------------

def _collate_fn_sft(batch: List[Dict], pad_token_id: int) -> Dict:
    """Right-pad collate for SFT batches (input_ids / attention_mask / labels)."""
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
        "input_ids":      torch.stack(input_ids_list),
        "attention_mask": torch.stack(mask_list),
        "labels":         torch.stack(labels_list),
        "task":           [s["task"] for s in batch],
    }


# Keep old name as alias so existing code referencing _collate_fn still works
_collate_fn = _collate_fn_sft


def _collate_fn_candidates(batch: List[Dict], pad_token_id: int) -> Dict:
    """
    Right-pad collate for multi-candidate val batches.
    Adds sample_id / cand_id / gold_cand_id tensors.
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
        "input_ids":      torch.stack(input_ids_list),
        "attention_mask": torch.stack(mask_list),
        "labels":         torch.stack(labels_list),
        "sample_id":      torch.stack([s["sample_id"]    for s in batch]),
        "cand_id":        torch.stack([s["cand_id"]      for s in batch]),
        "gold_cand_id":   torch.stack([s["gold_cand_id"] for s in batch]),
        "task":           [s["task"] for s in batch],
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


def _build_sft_loader(
    torch_ds: _BaseFinetuneDataset,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int,
    pin_memory: bool,
    length_fn,
) -> tuple:
    """Build SortedLengthSampler + DataLoader for SFT (single answer per sample)."""
    lengths = [
        length_fn(rec)
        for rec in tqdm(torch_ds.records, desc="[Sampler] pre-computing lengths", leave=False)
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


def _build_candidate_loader(
    torch_ds: _BaseCandidateValDataset,
    batch_size: int,      # number of *original* samples per batch
    shuffle: bool,
    seed: int,
    num_workers: int,
    pin_memory: bool,
    length_fn,
) -> tuple:
    """
    Build SortedLengthSampler + DataLoader for multi-candidate val datasets.

    DataLoader batch_size = batch_size × C so each batch contains exactly
    batch_size original samples, each with C candidate rows.
    Lengths are computed per original sample and replicated for all C rows.
    """
    C       = torch_ds.num_candidates
    lengths = [
        length_fn(torch_ds.records[s])
        for s, _c in tqdm(torch_ds._items,
                          desc="[Sampler] pre-computing lengths", leave=False)
    ]
    sampler = SortedLengthSampler(lengths, batch_size * C, shuffle, seed)
    pad_id  = torch_ds.tokenizer.pad_token_id
    loader  = DataLoader(
        torch_ds,
        batch_size=batch_size * C,
        sampler=sampler,
        collate_fn=lambda b: _collate_fn_candidates(b, pad_token_id=pad_id),
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
    print("Loading shared tokenizer ...")
    print("=" * 60)
    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    print(f"  vocab={tokenizer.vocab_size:,}  bos={tokenizer.bos_token_id}  "
          f"eos={tokenizer.eos_token_id}  pad={tokenizer.pad_token_id}\n")

    # ── SFT Mixed DataLoader ─────────────────────────────────────────────────
    mixed_loader = MixedFinetuneDataLoader(
        data_root=DATA_ROOT, batch_size=BATCH_SIZE, tokenizer=tokenizer
    )

    print("\n" + "=" * 60)
    print("Smoke-test: SFT Mixed DataLoader")
    print("=" * 60)
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
            clean  = labels.clone(); clean[clean == -100] = tokenizer.pad_token_id
            print(f"\n>>> [TASK: {t.upper()}]")
            print(f"FULL SEQUENCE:\n{tokenizer.decode(ids, skip_special_tokens=False)}")
            print(f"ANSWER TARGET: '{tokenizer.decode(clean, skip_special_tokens=True).strip()}'")
        if len(found_tasks) == 3:
            break
    assert (batch["labels"] == -100).any()
    assert (batch["labels"] != -100).any()
    assert found_tasks == {"mmlu", "squad", "snli"}
    print(f"\n  ✓ SFT smoke-test passed. Tasks: {found_tasks}")

    # ── MMLU Val DataLoader ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Smoke-test: MMLUValDataLoader (log-prob scoring)")
    print("=" * 60)
    mmlu_val = MMLUValDataLoader(
        data_root=DATA_ROOT, tokenizer=tokenizer, batch_size=4, max_length=512
    )
    C     = mmlu_val.num_candidates
    batch = next(iter(mmlu_val))

    assert batch["input_ids"].dim()      == 2
    assert batch["labels"].dim()         == 2
    assert (batch["labels"] == -100).any()
    assert (batch["labels"] != -100).any()
    assert "answers" not in batch
    assert batch["sample_id"].dtype    == torch.long
    assert batch["cand_id"].max().item() < C
    assert batch["gold_cand_id"].max().item() < C

    print(f"  ✓ shape={tuple(batch['input_ids'].shape)}  C={C}  "
          f"task={batch['task'][0]}")
    print(f"\n  {'─'*20} First sample — all {C} candidates {'─'*20}")
    mask_s0 = batch["sample_id"] == batch["sample_id"][0]
    for row_i in mask_s0.nonzero(as_tuple=True)[0]:
        ids_r  = batch["input_ids"][row_i]
        labs_r = batch["labels"][row_i]
        cid    = batch["cand_id"][row_i].item()
        gold   = batch["gold_cand_id"][row_i].item()
        cand_t = tokenizer.decode(ids_r[labs_r != -100], skip_special_tokens=True)
        marker = " ← GOLD" if cid == gold else ""
        print(f"  cand_id={cid}  '{cand_t}'{marker}")

    # ── SNLI Val DataLoader ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Smoke-test: SNLIValDataLoader (log-prob scoring)")
    print("=" * 60)
    snli_val = SNLIValDataLoader(
        data_root=DATA_ROOT, tokenizer=tokenizer, batch_size=8, max_length=256
    )
    C     = snli_val.num_candidates
    batch = next(iter(snli_val))

    assert batch["input_ids"].dim()      == 2
    assert batch["labels"].dim()         == 2
    assert (batch["labels"] == -100).any()
    assert (batch["labels"] != -100).any()
    assert "answers" not in batch
    assert batch["cand_id"].max().item() < C
    assert batch["gold_cand_id"].max().item() < C

    print(f"  ✓ shape={tuple(batch['input_ids'].shape)}  C={C}  "
          f"task={batch['task'][0]}")
    print(f"\n  {'─'*20} First sample — all {C} candidates {'─'*20}")
    mask_s0 = batch["sample_id"] == batch["sample_id"][0]
    for row_i in mask_s0.nonzero(as_tuple=True)[0]:
        ids_r  = batch["input_ids"][row_i]
        labs_r = batch["labels"][row_i]
        cid    = batch["cand_id"][row_i].item()
        gold   = batch["gold_cand_id"][row_i].item()
        cand_t = tokenizer.decode(ids_r[labs_r != -100], skip_special_tokens=True)
        marker = " ← GOLD" if cid == gold else ""
        print(f"  cand_id={cid}  '{cand_t}'{marker}")

    # ── SQuAD Validation DataLoader ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Smoke-test: SQuADValidationDataLoader (generation mode)")
    print("=" * 60)
    squad_val = SQuADValidationDataLoader(
        data_root=DATA_ROOT, tokenizer=tokenizer, batch_size=4, max_length=1024
    )
    batch = next(iter(squad_val))

    assert batch["input_ids"].dim()      == 2
    assert batch["attention_mask"].dim() == 2
    assert "labels"  not in batch
    assert "answers" in batch
    assert isinstance(batch["answers"], list)
    assert all(isinstance(a, list) for a in batch["answers"])
    assert all(isinstance(s, str)  for a in batch["answers"] for s in a)
    assert all(t == "squad"        for t in batch["task"])

    ids_0  = batch["input_ids"][0]
    mask_0 = batch["attention_mask"][0]
    prompt = tokenizer.decode(ids_0[mask_0 == 1], skip_special_tokens=True)
    print(f"  ✓ shape={tuple(batch['input_ids'].shape)}  task={batch['task'][0]}")
    print(f"  seq_len={ids_0.size(0)}  prompt ends: '...{prompt[-40:]}'")
    print(f"  ANSWERS: {batch['answers'][0]}")
    assert prompt.strip().endswith("Answer:")

    print("\n" + "=" * 60)
    print("✓ All smoke-tests passed.")
    print("=" * 60)