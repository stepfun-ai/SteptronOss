"""Assistant-only tokenization and resumable, DP-sharded text batches."""

import hashlib
import json
import random
from pathlib import Path

import torch
from configurize import Ref

from steptronoss.exp.sft import SFTDataConfig


def encode_chat(tokenizer, sample, max_length, template_kwargs=None):
    messages = sample.get("messages")
    if not messages or messages[-1].get("role") != "assistant":
        raise ValueError("Each record must end with an assistant message")
    if any(not isinstance(m.get("content"), str) for m in messages):
        raise ValueError("This experiment accepts string text content only")
    if "images" in sample or "videos" in sample:
        raise ValueError("Image/video SFT is not implemented by this text path")
    kwargs = dict(template_kwargs or {})
    kwargs.update(sample.get("chat_template_kwargs", {}))
    # Preserve historical reasoning so prefix renders stay consistent.
    kwargs["preserve_thinking"] = True
    if "tools" in sample:
        kwargs["tools"] = sample["tools"]

    def render(items):
        return tokenizer.apply_chat_template(items, tokenize=False, add_generation_prompt=False, **kwargs)

    text = render(messages)
    spans = []
    for i, message in enumerate(messages):
        if message.get("loss_mask", 1) not in (0, 1):
            raise ValueError("Per-turn loss_mask must be 0 or 1")
        if message["role"] != "assistant":
            continue
        before = render(messages[:i]) if i else ""
        through = render(messages[: i + 1])
        header = "<|im_start|>assistant\n"
        if not through.startswith(before + header) or not text.startswith(through):
            raise ValueError("Chat template prefixes are inconsistent; cannot safely create assistant mask")
        if message.get("loss_mask", 1):
            spans.append((len(before) + len(header), len(through)))
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    ids = encoded["input_ids"]
    if len(ids) > max_length:
        raise ValueError(f"Record has {len(ids)} tokens, exceeding max_length={max_length}; truncation is disabled")
    labels = [
        token if any(end > a and start < b for a, b in spans) else -100
        for token, (start, end) in zip(ids, encoded["offset_mapping"])
    ]
    if len(ids) < 2 or not any(label != -100 for label in labels[1:]):
        raise ValueError("Record has no assistant next-token target")
    return {"input_ids": ids, "labels": labels}


class ResumableTextBatches:
    """One global permutation; disjoint DP shards; fixed complete global steps.

    Cursor advances during native scheduler prefetch. Saving after a complete
    optimizer step preserves the next global batch without a worker queue.
    """

    def __init__(
        self, path, dp_rank, dp_size, micro_batch_size, global_batch_size, max_length, pad_id, seed=1234, cp_size=1
    ):
        self.path = Path(path)
        if not 0 <= dp_rank < dp_size or global_batch_size % (dp_size * micro_batch_size):
            raise ValueError("Invalid DP rank or global_batch_size divisibility")
        raw = self.path.read_bytes()
        self.fingerprint = hashlib.sha256(raw).hexdigest()
        self.records = [json.loads(line) for line in raw.splitlines() if line.strip()]
        for record in self.records:
            ids, labels = record["input_ids"], record["labels"]
            if len(ids) != len(labels) or not 2 <= len(ids) <= max_length:
                raise ValueError("Invalid token lengths; rebuild the tokenized dataset")
            if not all(isinstance(x, int) and x >= 0 for x in ids):
                raise ValueError("input_ids must be nonnegative integer token IDs")
            if not all(y == -100 or y == x for x, y in zip(ids, labels)):
                raise ValueError("Labels must be unshifted token IDs or -100")
            if not any(x != -100 for x in labels[1:]):
                raise ValueError("A record has no assistant next-token target")
        self.usable = len(self.records) // global_batch_size * global_batch_size
        if self.usable == 0:
            raise ValueError("Dataset is smaller than one global batch")
        self.dp_rank, self.dp_size = dp_rank, dp_size
        self.micro_batch_size, self.global_batch_size = micro_batch_size, global_batch_size
        self.pad_id, self.seed = pad_id, seed
        self.cp_size = cp_size
        self.epoch, self.cursor = 0, 0
        self._order()

    def _order(self):
        order = list(range(len(self.records)))
        random.Random(self.seed + self.epoch).shuffle(order)
        self.indices = order[: self.usable][self.dp_rank :: self.dp_size]

    def __len__(self):
        return self.usable  # native trainer computes global train_iters from this

    def __iter__(self):
        return self

    def __next__(self):
        if self.cursor == len(self.indices):
            self.epoch += 1
            self.cursor = 0
            self._order()
        records = [self.records[i] for i in self.indices[self.cursor : self.cursor + self.micro_batch_size]]
        self.cursor += self.micro_batch_size
        length = max(len(r["input_ids"]) for r in records)
        if self.cp_size > 1:
            align = self.cp_size * 64
            length = (length + align - 1) // align * align
        ids = torch.full((len(records), length), self.pad_id, dtype=torch.long)
        labels = torch.full_like(ids, -100)
        mask = torch.zeros_like(ids)
        for i, record in enumerate(records):
            n = len(record["input_ids"])
            ids[i, :n] = torch.tensor(record["input_ids"])
            labels[i, :n] = torch.tensor(record["labels"])
            mask[i, :n] = 1
        return {"input_ids": ids, "labels": labels, "attention_mask": mask}

    def state_dict(self):
        # Checkpointer stores only DP0's data state. Cursor is intentionally a
        # global logical cursor shared by every DP rank, not a list of rank IDs.
        return {
            "fingerprint": self.fingerprint,
            "epoch": self.epoch,
            "cursor": self.cursor,
            "dp_size": self.dp_size,
            "micro_batch_size": self.micro_batch_size,
            "global_batch_size": self.global_batch_size,
            "seed": self.seed,
        }

    def load_state_dict(self, state):
        expected = self.state_dict()
        for key in expected.keys() - {"epoch", "cursor"}:
            if state.get(key) != expected[key]:
                raise ValueError(f"Dataset/resume configuration mismatch: {key}")
        epoch, cursor = state["epoch"], state["cursor"]
        if epoch < 0 or not 0 <= cursor <= len(self.indices) or cursor % self.micro_batch_size:
            raise ValueError("Invalid dataloader resume cursor")
        self.epoch, self.cursor = epoch, cursor
        self._order()


class Qwen38SFTDataConfig(SFTDataConfig):
    tokenized_path: str = "./data/qwen3_8_train.tokens.jsonl"
    """JSONL containing unshifted input_ids and assistant-only labels."""
    pad_token_id: int = 248044
    """Padding token; masked out by both attention_mask and labels."""
    micro_batch_size: int = Ref("..trainer_cfg.micro_batch_size")
    """Native trainer micro batch size."""
    global_batch_size: int = Ref("..trainer_cfg.global_batch_size")
    """Native trainer global batch size."""
    max_length: int = Ref("..trainer_cfg.global_seq_length")
    """Reject tokenized records exceeding the configured sequence limit."""
    seed: int = Ref("..seed")
    """Shared shuffle seed."""
    model_path: str = Ref("..model_cfg.model_path")
    """Require the dataset to use the same tokenizer model as training."""
    revision: str | None = Ref("..model_cfg.revision")
    """Require the same pinned base-model revision."""

    cp_size: int = Ref("..model_cfg.parallel_cfg.context_parallel_size")
    """Round right-padding to aligned contiguous CP blocks."""

    def build_dataloader(self, dp_rank=0, dp_size=1):
        metadata = json.loads(Path(self.tokenized_path + ".meta.json").read_text())
        for key, value in {
            "format": "steptron-qwen38-text-v1",
            "model": self.model_path,
            "revision": self.revision,
            "max_length": self.max_length,
            "pad_token_id": self.pad_token_id,
        }.items():
            if metadata.get(key) != value:
                raise ValueError(f"Tokenized dataset metadata mismatch: {key}; rerun prepare_qwen3_8_sft.py")
        batches = ResumableTextBatches(
            self.tokenized_path,
            dp_rank,
            dp_size,
            self.micro_batch_size,
            self.global_batch_size,
            self.max_length,
            self.pad_token_id,
            self.seed,
            self.cp_size,
        )
        if metadata.get("sha256") != batches.fingerprint:
            raise ValueError("Tokenized dataset changed after preparation")
        return batches
