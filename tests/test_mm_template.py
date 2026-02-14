import tempfile
from pathlib import Path

import torch
from PIL import Image

from steptronoss.data.chat_templates.mm_template import MMMultiTurnChatTemplate


class DummyTokenizer:
    def __init__(self, img_token: int):
        self.img_token = img_token

    def apply_chat_template(self, messages, tokenize=True, tools=None, add_generation_prompt=False):
        text = "".join(f"{m['role']}:{m['content']}" for m in messages)
        if add_generation_prompt:
            text += "<GEN>"
        if not tokenize:
            return text
        tokens = []
        marker = "<IMG>"
        while text:
            idx = text.find(marker)
            if idx == -1:
                tokens.extend([1] * len(text))
                break
            if idx > 0:
                tokens.extend([1] * idx)
            tokens.append(self.img_token)
            text = text[idx + len(marker) :]
        return tokens

    def decode(self, ids):
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return "x" * len(ids)


def test_mm_template_basic():
    img_token = 9999
    tokenizer = DummyTokenizer(img_token=img_token)
    template = MMMultiTurnChatTemplate(
        tokenizer=tokenizer,
        img_end_token=img_token,
        patch_end_token=None,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        img0 = tmp_path / "img0.png"
        img1 = tmp_path / "img1.png"
        Image.new("RGB", (4, 4), color=(255, 0, 0)).save(img0)
        Image.new("RGB", (4, 4), color=(0, 255, 0)).save(img1)

        data = {
            "images": [[str(img0), 0], [str(img1), 1]],
            "conversations": [
                {
                    "meta": {},
                    "name": "",
                    "content": "<IMG><IMG> hello?",
                    "role": "user",
                },
                {
                    "meta": {},
                    "name": "",
                    "content": "hi.",
                    "role": "assistant",
                },
            ],
            "meta": {},
        }

        sample = template(data)
    assert sample is not None

    all_tokens = sample["all_tokens"]
    tokens = sample["tokens"]
    labels = sample["labels"]
    loss_mask = sample["loss_mask"]

    assert all_tokens.shape[0] == tokens.shape[0] + 1
    assert labels.shape[0] == tokens.shape[0]
    assert loss_mask.shape[0] == tokens.shape[0]

    user_len = len(tokenizer.apply_chat_template([data["conversations"][0]], tokenize=True, add_generation_prompt=True))
    expected_ones = all_tokens.shape[0] - user_len
    assert int(loss_mask.sum().item()) == expected_ones
    assert len(sample["images"]) == 2
