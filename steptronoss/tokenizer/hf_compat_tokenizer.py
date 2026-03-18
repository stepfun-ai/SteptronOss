from __future__ import annotations

import json
import os
from typing import Any

from transformers.tokenization_utils_base import PreTrainedTokenizerBase


class BaseTokenizer:
    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
        split_special_tokens: bool = False,
        **kwargs,
    ) -> list[int]:
        pass

    def decode(
        self,
        token_ids: int | list[int],
        skip_special_tokens: bool = False,
        **kwargs,
    ):
        pass

    def apply_chat_template(self, *args, **kwargs):
        raise NotImplementedError


class HFCompatTokenizer:
    hf_path: str

    hf_tokenizer: PreTrainedTokenizerBase

    def __init__(
        self,
        hf_path: str | None = None,
        hf_tokenizer: PreTrainedTokenizerBase | None = None,
        **kwargs,
    ):
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        if hf_tokenizer is not None:
            self.hf_tokenizer = hf_tokenizer
            if hf_path is not None:
                self.hf_path = hf_path
            return

        if hf_path is not None:
            self.hf_path = hf_path

        self.hf_tokenizer = _load_raw_hf_tokenizer(self.hf_path, **kwargs)

    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
        split_special_tokens: bool = False,
        **kwargs,
    ):
        return self.hf_tokenizer.encode(
            text=text,
            add_special_tokens=add_special_tokens,
            split_special_tokens=split_special_tokens,
            **kwargs,
        )

    def decode(
        self,
        token_ids: int | list[int],
        skip_special_tokens=False,
        **kwargs,
    ):
        return self.hf_tokenizer.decode(
            token_ids,
            skip_special_tokens=skip_special_tokens,
            **kwargs,
        )

    def apply_chat_template(self, *args, **kwargs):
        tokenized = self.hf_tokenizer.apply_chat_template(*args, **kwargs)
        if kwargs.get("tokenize", True):
            return self._normalize_tokenized_input_ids(tokenized)
        return tokenized

    def __getattr__(self, key: str):
        return getattr(self.hf_tokenizer, key)

    def __dir__(self):
        return dir(self.hf_tokenizer)

    def __getstate__(self):
        # Return a dictionary of the attributes you want to pickle.
        # In this case, we only need to serialize 'self.obj'.
        return {"hf_tokenizer": self.hf_tokenizer}

    def __setstate__(self, state):
        # Restore the state from the pickled dictionary.
        # This is a direct assignment, which won't trigger __getattr__.
        self.hf_tokenizer = state["hf_tokenizer"]

    @staticmethod
    def _normalize_tokenized_input_ids(tokenized: Any) -> list[int]:
        if hasattr(tokenized, "input_ids"):
            tokenized = tokenized.input_ids
        elif isinstance(tokenized, dict) and "input_ids" in tokenized:
            tokenized = tokenized["input_ids"]

        if hasattr(tokenized, "tolist"):
            tokenized = tokenized.tolist()

        if isinstance(tokenized, tuple):
            tokenized = list(tokenized)

        if isinstance(tokenized, list) and tokenized and isinstance(tokenized[0], list):
            tokenized = tokenized[0]

        if not isinstance(tokenized, list):
            raise TypeError(f"Unsupported tokenized output type: {type(tokenized).__name__}")

        return tokenized


def _should_use_generic_fast_tokenizer(hf_path: str) -> bool:
    if not os.path.isdir(hf_path):
        return False

    tokenizer_json = os.path.join(hf_path, "tokenizer.json")
    tokenizer_model = os.path.join(hf_path, "tokenizer.model")
    tokenizer_config = os.path.join(hf_path, "tokenizer_config.json")
    if not os.path.isfile(tokenizer_json) or os.path.isfile(tokenizer_model):
        return False
    if not os.path.isfile(tokenizer_config):
        return False

    try:
        with open(tokenizer_config, encoding="utf-8") as fh:
            config = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return False

    # Transformers 5 aliases LlamaTokenizerFast to a generic tokenizers-backed
    # LlamaTokenizer implementation. For byte-level BPE tokenizers without a
    # SentencePiece model, loading through the generic fast class preserves the
    # tokenizer.json pre-tokenizer/decoder pipeline exactly.
    return config.get("tokenizer_class") == "LlamaTokenizerFast"


def _load_raw_hf_tokenizer(hf_path: str, **kwargs) -> PreTrainedTokenizerBase:
    from transformers import AutoTokenizer, PreTrainedTokenizerFast

    if _should_use_generic_fast_tokenizer(hf_path):
        return PreTrainedTokenizerFast.from_pretrained(hf_path, **kwargs)
    return AutoTokenizer.from_pretrained(hf_path, **kwargs)


def load_hf_tokenizer(hf_path: str, **kwargs) -> HFCompatTokenizer:
    """Load a tokenizer with the pre-HF-5 `AutoTokenizer.from_pretrained` contract.

    Call sites historically treated this as equivalent to
    `AutoTokenizer.from_pretrained(...)`: local tokenizer config decides the
    implementation, and `apply_chat_template(tokenize=True)` yields token IDs
    directly. The Transformers 5.0 upgrade broke that assumption for some
    local byte-level BPE tokenizers (for example Step3.5Flash SFT), where the
    auto loader no longer preserves the `tokenizer.json` behavior and changes
    tokenization results.

    This helper stabilizes the interface by:
    - selecting `PreTrainedTokenizerFast.from_pretrained(...)` for the known
      LlamaTokenizerFast/byte-level local tokenizer shape that regressed in HF5
    - wrapping the result in `HFCompatTokenizer`, which normalizes
      `apply_chat_template(tokenize=True)` back to `list[int]`
    """
    return HFCompatTokenizer(hf_path=hf_path, **kwargs)
