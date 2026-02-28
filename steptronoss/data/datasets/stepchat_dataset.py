# Copyright (c) 2026, Stepfun Inc. All rights reserved.

import json
import os
import random
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Literal, TypedDict

import ijson
import torch
import torch.utils
import torch.utils.data
from loguru import logger
from megfile import smart_open

from steptronoss.data.datasets.base_language_dataset import (
    VALID_MESSAGE_CONTENT_TYPES,
    Dialog,
    MessageContentPart,
    MessageItem,
)
from steptronoss.data.recipe import DataSourceFile
from steptronoss.utils import load


# message in json
class JSONMessage(TypedDict):
    role: str
    content: str
    name: str
    loss_mask: Literal[0, 1] | None
    ground_truth: Any | None


class JSONSample(TypedDict):
    conversations: list[JSONMessage]
    images: list[str] | None


class StepChatJsonDataset(torch.utils.data.Dataset):
    """
    Dataset for StepChat json format.

    A dialog or conversation is formated like ChatML.
    """

    def __init__(
        self,
        filelist: list[DataSourceFile],
        template: Callable | None = None,
    ):
        self.filelist = filelist
        data, valid_dialog_nums_per_file = self.read_from_file(filelist)
        self.data: list[Dialog] = data
        self.valid_dialog_nums_per_file: dict[str, int] = valid_dialog_nums_per_file

        self.template = template

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> Dialog:
        data = self.data[idx]
        if self.template:
            data = self.template(data)
        return data

    def get_sample_sizes(self, size_func: Callable[[Any], int]) -> list[int]:
        return [size_func(x) for x in self]

    @classmethod
    def _load_dialog(cls, f_path: str):
        try:
            return cls._streaming_load_dialog(f_path)
        except Exception as e:
            logger.warning(f"ijson can't parse {f_path}, will try native json parser, exception is {e}")
            return cls._native_load_dialog(f_path)

    @classmethod
    def _native_load_dialog(cls, f_path: DataSourceFile):
        file_data = []
        f_path, sample_rate = f_path.path, f_path.subsample_rate

        valid_dialog_nums = 0
        raw_dialogs: list[JSONSample] = load(f_path)

        if 0.0 < sample_rate < 1:
            # raw_dialogs = raw_dialogs[:int(len(raw_dialogs) * sample_rate)]
            origin_len = len(raw_dialogs)
            raw_dialogs = random.Random(1234).sample(raw_dialogs, int(origin_len * sample_rate))
            logger.info(f"Downsampled {origin_len} dialogs to {len(raw_dialogs)} dialogs from {f_path}")

        for idx, dialog in enumerate(raw_dialogs):
            dialog = cls.convert_dialog(dialog, f_path)
            err = cls.check_error(dialog)

            if err is None:
                file_data.append(dialog)
                valid_dialog_nums += 1
            else:
                logger.warning(f"Invalid dialog ({err}) from {f_path}: {idx}")
        return f_path, file_data, valid_dialog_nums

    @classmethod
    def _streaming_load_dialog(cls, f_path: DataSourceFile):
        file_data = []
        f_path, sample_rate = f_path.path, f_path.subsample_rate

        valid_dialog_nums = 0

        raw_indices = {}
        if 0.0 < sample_rate < 1:
            origin_len = 0

            with smart_open(f_path, "rb") as f:
                for _ in ijson.items(f, "item", use_float=True):
                    origin_len += 1
            raw_indices = random.Random(1234).sample(range(origin_len), int(origin_len * sample_rate))
            raw_indices = {idx: i for i, idx in enumerate(raw_indices)}
            logger.info(f"Downsampled {origin_len} dialogs to {len(raw_indices)} dialogs from {f_path}")

        with smart_open(f_path, "rb") as f:
            idx = -1
            for item in ijson.items(f, "item", use_float=True):
                idx += 1
                if 0.0 < sample_rate < 1.0 and idx not in raw_indices:
                    continue
                dialog = cls.convert_dialog(item, f_path)

                err = cls.check_error(dialog)

                if err is None:
                    file_data.append((idx, dialog))
                    valid_dialog_nums += 1
                else:
                    logger.warning(f"Invalid dialog ({err}) from {f_path}: {idx}")

        if 0.0 < sample_rate < 1 and len(raw_indices) > 0:
            file_data = sorted(file_data, key=lambda item: raw_indices[item[0]])

        file_data = [data[1] for data in file_data]
        return f_path, file_data, valid_dialog_nums

    @classmethod
    def read_from_file(cls, filelist: list[DataSourceFile]):
        data = []
        valid_dialog_nums_per_file: dict[str, int] = {}

        with ProcessPoolExecutor(max_workers=os.cpu_count() or 4) as pool:
            results = pool.map(cls._load_dialog, filelist)

        for result in results:
            f_path, file_data, valid_dialog_nums = result
            data.extend(file_data)
            valid_dialog_nums_per_file[f_path] = valid_dialog_nums

        return data, valid_dialog_nums_per_file

    @classmethod
    def convert_content_part(cls, p: dict[str, Any] | str) -> MessageContentPart:
        if isinstance(p, str):
            return MessageContentPart(type="text", value=p)

        try:
            assert isinstance(p["type"], str) and isinstance(p["value"], str)
            assert p["type"] in VALID_MESSAGE_CONTENT_TYPES
            r = MessageContentPart(
                type=p["type"],
                value=p["value"],
            )
            if "loss_mask" in p:
                assert isinstance(p["loss_mask"], (int, float)) and int(p["loss_mask"]) in [0, 1]
                r["loss_mask"] = int(p["loss_mask"])
            return r
        except Exception as e:
            raise ValueError(f"Invalid content message part: {p}, {e}")

    @classmethod
    def convert_message_item(cls, item: JSONMessage) -> list[MessageContentPart]:
        if "content" not in item:  # Backward compatibility
            assert "value" in item
            content = item["value"]
        else:
            content = item["content"]

        if not isinstance(content, list):
            content = [content]

        return [cls.convert_content_part(c) for c in content if c is not None]

    @classmethod
    def convert_dialog(cls, raw_dialog: JSONSample, f_path: str = "") -> Dialog:
        def get_tool_schemas(item: JSONMessage) -> list[dict] | None:
            tools = item.get("tools", None)
            if tools:
                return tools
            else:
                return item.get("tool_schemas", None)

        if "conversation" not in raw_dialog:
            raw_dialog = {"conversations": raw_dialog}

        try:
            return Dialog(
                conversations=[
                    MessageItem(
                        role=item["role"].lower(),
                        content=cls.convert_message_item(item),
                        name=(item.get("name", "") or "").strip(),
                        loss_mask=item.get("loss_mask", 1),
                        ground_truth=item.get("ground_truth", None),
                        tool_calls=item.get("tool_calls", None),
                        reasoning_content=item.get("reasoning_content", None),
                        tool_schemas=get_tool_schemas(item),
                        data_path=f_path,
                    )
                    for item in raw_dialog["conversations"]
                ],
                images=raw_dialog.get("images"),
            )

        except Exception as e:
            print(raw_dialog)
            raise e

    @classmethod
    def check_error(cls, dialog: Dialog) -> str | None:
        if len(dialog) < 2:
            return "less than 2 msgs"

        for msg in dialog["conversations"]:
            if msg["name"] and "\n" in msg["name"]:
                return "name contains newline"

            if msg["role"] == "assistant":
                if not msg["content"] and not msg["tool_calls"]:  # content and toolcall is empty
                    return "empty or invalid content and toolcall"

        if dialog["conversations"][-1]["role"] != "assistant":
            return "the last message is not assistant"

        return None
