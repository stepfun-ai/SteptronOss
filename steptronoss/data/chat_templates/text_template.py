import json

import numpy as np
from jinja2.sandbox import ImmutableSandboxedEnvironment

from steptronoss.data.datasets.base_language_dataset import Dialog
from steptronoss.tokenizer.hf_compat_tokenizer import HFCompatTokenizer

_original_init = ImmutableSandboxedEnvironment.__init__


def _patched_init(self, *args, **kwargs):
    """support json loading in jinja2 template."""
    _original_init(self, *args, **kwargs)
    self.filters["fromjson"] = json.loads


ImmutableSandboxedEnvironment.__init__ = _patched_init


class HuggingFaceTemplate:
    def __init__(self, tokenizer: HFCompatTokenizer):
        self.tokenizer: HFCompatTokenizer = tokenizer

    def __call__(self, data: Dialog):
        # 确保数据以assistant结尾
        data = data["conversations"]
        assert data[-1]["role"] == "assistant"

        # 获取完整的token序列
        if "tool_schemas" in data[0]:
            tool_schemas = data[0]["tool_schemas"]
        else:
            tool_schemas = None

        all_tokens = self.apply_chat_template(data, tokenize=True, tools=tool_schemas)
        # 找到最后一个user轮的位置
        last_user_idx = -1
        for i in range(len(data) - 1, -1, -1):
            if data[i]["role"] == "user":
                last_user_idx = i
                break

        # 初始化loss_mask，默认全部不训练(0)
        loss_mask = np.zeros(len(all_tokens), dtype=np.float16)

        if last_user_idx == -1:
            # 如果找不到user轮，数据有问题
            raise ValueError("No user turn found in dialog data. This indicates problematic data.")
        else:
            # 计算最后一个user轮结束的token位置
            tokens_up_to_last_user = self.apply_chat_template(
                data[: last_user_idx + 1], tokenize=True, tools=tool_schemas
            )

            last_user_end_pos = len(tokens_up_to_last_user)

            # 只训练最后一个user之后的assistant轮
            current_pos = last_user_end_pos
            for i in range(last_user_idx + 1, len(data)):
                tokens_up_to_current = self.apply_chat_template(data[: i + 1], tokenize=True, tools=tool_schemas)

                if data[i]["role"] == "assistant" and data[i]["loss_mask"] == 1.0:
                    # 只有assistant轮且mask为1才训练
                    loss_mask[current_pos : len(tokens_up_to_current)] = 1.0

                current_pos = len(tokens_up_to_current)

        return {
            "tokens": np.array(all_tokens, dtype=np.int32),
            "loss_mask": np.array(loss_mask, dtype=np.float16),
        }

    def apply_chat_template(self, messages, tokenize=True, tools=None):
        add_generation_prompt = False if messages[-1]["role"] == "assistant" else True
        if tools:
            tokenized = self.tokenizer.apply_chat_template(
                messages,
                tokenize=tokenize,
                tools=tools,
                add_generation_prompt=add_generation_prompt,
            )
        else:
            tokenized = self.tokenizer.apply_chat_template(
                messages, tokenize=tokenize, add_generation_prompt=add_generation_prompt
            )
        if isinstance(tokenized, list):
            return tokenized
        return tokenized["input_ids"]
