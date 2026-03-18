from __future__ import annotations

import json

import pytest

from steptronoss.tokenizer.hf_compat_tokenizer import HFCompatTokenizer, load_hf_tokenizer

pytestmark = pytest.mark.cpu


class _RawBatchTokenizer:
    def apply_chat_template(self, *args, **kwargs):
        del args, kwargs
        return {
            "input_ids": [7, 8, 9],
            "attention_mask": [1, 1, 1],
        }

    def encode(self, text: str, **kwargs):
        del kwargs
        return [ord(ch) for ch in text]

    def decode(self, token_ids, **kwargs):
        del kwargs
        return "".join(chr(token_id) for token_id in token_ids)


def test_hf_compat_tokenizer_normalizes_apply_chat_template_token_ids():
    tokenizer = HFCompatTokenizer(hf_tokenizer=_RawBatchTokenizer())

    assert tokenizer.apply_chat_template([], tokenize=True, add_generation_prompt=True) == [7, 8, 9]


def test_hf_compat_tokenizer_keeps_non_tokenized_chat_template_output():
    class _RawStringTokenizer(_RawBatchTokenizer):
        def apply_chat_template(self, *args, **kwargs):
            del args, kwargs
            return "prompt"

    tokenizer = HFCompatTokenizer(hf_tokenizer=_RawStringTokenizer())

    assert tokenizer.apply_chat_template([], tokenize=False, add_generation_prompt=True) == "prompt"


def test_load_hf_tokenizer_prefers_generic_fast_loader_for_byte_level_llama(tmp_path, monkeypatch):
    (tmp_path / "tokenizer.json").write_text("{}", encoding="utf-8")
    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps({"tokenizer_class": "LlamaTokenizerFast"}),
        encoding="utf-8",
    )

    calls: list[tuple[str, str]] = []

    class _SentinelTokenizer:
        def encode(self, text: str, **kwargs):
            del text, kwargs
            return []

        def decode(self, token_ids, **kwargs):
            del token_ids, kwargs
            return ""

        def apply_chat_template(self, *args, **kwargs):
            del args, kwargs
            return []

    def _auto_loader(path, **kwargs):
        del kwargs
        calls.append(("auto", str(path)))
        return _SentinelTokenizer()

    def _fast_loader(path, **kwargs):
        del kwargs
        calls.append(("fast", str(path)))
        return _SentinelTokenizer()

    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", _auto_loader)
    monkeypatch.setattr("transformers.PreTrainedTokenizerFast.from_pretrained", _fast_loader)

    tokenizer = load_hf_tokenizer(str(tmp_path))

    assert isinstance(tokenizer, HFCompatTokenizer)
    assert calls == [("fast", str(tmp_path))]


def test_load_hf_tokenizer_uses_auto_loader_for_regular_tokenizer(tmp_path, monkeypatch):
    (tmp_path / "tokenizer.json").write_text("{}", encoding="utf-8")
    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps({"tokenizer_class": "Qwen2TokenizerFast"}),
        encoding="utf-8",
    )

    calls: list[tuple[str, str]] = []

    class _SentinelTokenizer:
        def encode(self, text: str, **kwargs):
            del text, kwargs
            return []

        def decode(self, token_ids, **kwargs):
            del token_ids, kwargs
            return ""

        def apply_chat_template(self, *args, **kwargs):
            del args, kwargs
            return []

    def _auto_loader(path, **kwargs):
        del kwargs
        calls.append(("auto", str(path)))
        return _SentinelTokenizer()

    def _fast_loader(path, **kwargs):
        del kwargs
        calls.append(("fast", str(path)))
        return _SentinelTokenizer()

    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", _auto_loader)
    monkeypatch.setattr("transformers.PreTrainedTokenizerFast.from_pretrained", _fast_loader)

    tokenizer = load_hf_tokenizer(str(tmp_path))

    assert isinstance(tokenizer, HFCompatTokenizer)
    assert calls == [("auto", str(tmp_path))]
