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

    def __init__(self):
        import os

        from transformers import AutoTokenizer

        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self.hf_tokenizer = AutoTokenizer.from_pretrained(self.hf_path)

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
        return self.hf_tokenizer.apply_chat_template(*args, **kwargs)

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
