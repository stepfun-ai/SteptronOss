import pytest

from steptronoss.data.chat_templates.text_template import GemmaTemplate

pytestmark = pytest.mark.cpu


class _GemmaLikeTokenizer:
    def __init__(self):
        self.last_messages = None

    def apply_chat_template(self, messages, tokenize=True, **kwargs):
        del kwargs
        self.last_messages = messages
        return [1, 2, 3] if tokenize else "prompt"


def test_gemma_template_normalizes_text_parts_to_gemma_schema():
    tokenizer = _GemmaLikeTokenizer()
    template = GemmaTemplate(tokenizer=tokenizer)

    messages = [
        {
            "role": "user",
            "content": [{"type": "text", "value": "hello"}],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "value": "world"}],
        },
    ]

    token_ids = template.apply_chat_template(messages, tokenize=True)

    assert token_ids == [1, 2, 3]
    assert tokenizer.last_messages[0]["content"][0]["text"] == "hello"
    assert tokenizer.last_messages[0]["content"][0]["value"] == "hello"
    assert tokenizer.last_messages[1]["content"][0]["text"] == "world"


def test_gemma_template_keeps_non_text_parts_unchanged():
    tokenizer = _GemmaLikeTokenizer()
    template = GemmaTemplate(tokenizer=tokenizer)

    messages = [
        {
            "role": "user",
            "content": [{"type": "token", "value": "<special>"}],
        },
        {
            "role": "assistant",
            "content": "done",
        },
    ]

    template.apply_chat_template(messages, tokenize=True)

    assert tokenizer.last_messages[0]["content"][0] == {"type": "token", "value": "<special>"}
    assert tokenizer.last_messages[1]["content"] == "done"
