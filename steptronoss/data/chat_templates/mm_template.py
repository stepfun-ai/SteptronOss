"""Multi-modal (vision) chat template utilities for SFT preprocessing.

This module defines a template class that converts multi-turn dialogs with
optional image references into model-ready tensors. It handles:
1) Tokenization using the HuggingFace-compatible chat template.
2) Construction of labels and loss masks for supervised fine-tuning.
3) Image loading / indexing and consistency checks between image tokens
   and provided images or cached image features.

Expected input format
---------------------
Each sample is a MMDialog with:
- images: Optional list where each item is either:
  - a string image path, or
  - [path, index] / (path, index) to preserve per-image ordering.
- conversations: list[MessageItem] in the same format as text-only datasets,
  optionally containing:
  - tool_schemas on the first message
  - loss_mask on assistant turns (default is 1.0)
  - reasoning_content on the last assistant turn (may be removed)

Core outputs
------------
The template produces a dict containing:
- tokens: token ids (all_tokens[:-1])
- labels: shifted token ids (all_tokens[1:])
- loss_mask: float mask (1.0 where supervised loss applies)
- all_tokens: full unshifted token ids
- images or image_path (depending on whether a transform is applied)
- image_grid_thw (if present in input, kept for compatibility)

Masking strategy
----------------
Loss is only computed on assistant responses after the last user turn.
All tokens before the last user message are masked out.
Assistant turns can be explicitly masked by setting loss_mask=0.0.

Image/token alignment
---------------------
When raw images are used (no transform), the number of image end tokens
(img_end_token and optional patch_end_token) must match the number of
images. If mismatched, the sample is dropped to avoid training on
inconsistent multi-modal alignment.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Optional, TypedDict

import torch
import torchvision.transforms.v2 as T2
from loguru import logger
from megfile import smart_open
from PIL import Image
from torchvision import transforms as T

from steptronoss.data.datasets.base_language_dataset import Dialog, MessageItem
from steptronoss.tokenizer.hf_compat_tokenizer import HFCompatTokenizer


class MMDialog(TypedDict):
    """Multi-modal dialog sample schema.

    Attributes:
        images: Optional list of image paths or (path, index) pairs.
        conversations: Ordered list of message items (roles + content).
    """

    images: list[str | list[Any]] | None
    conversations: list[MessageItem]


MMDataset = list[MMDialog]


class MMSample(TypedDict, total=False):
    """Intermediate multi-modal sample produced by chat templating."""

    tokens: torch.LongTensor
    labels: torch.LongTensor
    loss_mask: torch.FloatTensor
    all_tokens: torch.LongTensor
    image_path: list[str | list[Any]] | None
    images: list[tuple[Image.Image, int]]
    image_grid_thw: Any


class ImageTokenVerifyTransform:
    """Verify image tokens match the number of loaded images.

    Expects sample to contain:
        - image_path: list of (str, index) tuples
        - tokens: tensor or list of token ids
    """

    def __init__(self, img_end_token: int, patch_end_token: int | None):
        self.img_end_token = img_end_token
        self.patch_end_token = patch_end_token

    def __call__(self, sample: MMSample) -> MMSample | None:
        image_paths = sample.get("image_path") or []
        tokens = sample.get("tokens")
        if tokens is None:
            raise ValueError("tokens are required for image token verification")

        token_count = sum(
            1
            for token in tokens
            if token == self.img_end_token or (self.patch_end_token is not None and token == self.patch_end_token)
        )
        image_count = len(image_paths)
        if image_count != token_count:
            raise RuntimeError(f"Truncated sample! Got {image_count} imgs & {token_count} end tokens.")

        return sample


class ImageLoadTransform:
    """Load images from paths and attach them to the sample.

    Expects sample to contain:
        - image_path: list of image paths or (path, index) pairs
    Produces:
        - images: list of (PIL.Image, index) tuples
    """

    def __init__(self, root_path: str = ""):
        self.root_path = root_path

    def __call__(self, sample: MMSample) -> MMSample:
        from os.path import join

        images = sample.get("image_path") or []

        def _load(path: str) -> Image.Image:
            with smart_open(join(self.root_path, path), "rb") as f:
                return Image.open(f).convert("RGB")

        out = []
        for item in images:
            if isinstance(item, (list, tuple)):
                path = item[0]
                idx = item[1] if len(item) > 1 else 0
            else:
                path, idx = item, 0
            out.append((_load(path), idx))

        sample["images"] = out
        return sample


class ProcessImageTransform:
    def __init__(
        self,
        im_size=(728, 728),
        patch_size=(504, 504),
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711],
    ):
        self.im_size = im_size
        self.patch_size = patch_size
        self.mean = mean
        self.std = std

        logger.info(f"ProcessImage: im_size={self.im_size}, patch_size={self.patch_size}")
        self.vision_transform = T.Compose([
            T2.ToTensor(),
            T2.Normalize(mean=mean, std=std, inplace=True),
        ])
        self.resize_high = T2.Resize(self.im_size, interpolation=T.InterpolationMode.BILINEAR, antialias=True)
        self.resize_low = T2.Resize(self.patch_size, interpolation=T.InterpolationMode.BILINEAR, antialias=True)

    def __call__(self, sample: dict):
        for idx, image in enumerate(sample["images"]):
            image, im_type = image
            image = self.vision_transform(image)
            if im_type == 0:  # image
                image = self.resize_high(image)
                assert image.shape[1] == self.im_size[0], f"Image shape {image.shape} is not {self.im_size}"
                assert image.shape[2] == self.im_size[1], f"Image shape {image.shape} is not {self.im_size}"
            elif im_type == 1:  # patch
                image = self.resize_low(image)
                assert image.shape[1] == self.patch_size[0], f"Image shape {image.shape} is not {self.patch_size}"
                assert image.shape[2] == self.patch_size[1], f"Image shape {image.shape} is not {self.patch_size}"
            else:
                raise ValueError(f"Unknown Image Type {im_type}")
            sample["images"][idx] = [image, int(im_type)]
        return sample


class ImageTypeToStartTokenTransform:
    def __init__(self, type_to_start_token_mapping: dict[int, int]):
        self.type_to_start_token_mapping = type_to_start_token_mapping

    def __call__(self, sample: MMSample) -> MMSample | None:
        images = sample.get("images") or []
        sample["images"] = [(p, self.type_to_start_token_mapping[t]) for p, t in images]
        return sample


class MMMultiTurnChatTemplate:
    """Template for multi-modal (vision) multi-turn SFT data.

    This class tokenizes chat conversations, builds labels and loss masks,
    and optionally loads images (or defers to a transform for cached features).
    It enforces alignment between image tokens and provided images to avoid
    training on inconsistent samples.
    """

    def __init__(
        self,
        tokenizer: HFCompatTokenizer,
        max_seq_len=8192,
        num_workers=8,
        skip_reasoning_content=False,
        transform=None,
    ):
        """Initialize the multi-modal chat template.

        Args:
            tokenizer: HF-compatible tokenizer with apply_chat_template support.
            img_end_token: Token ID that marks the end of an image placeholder.
            patch_end_token: Optional token ID for patch-based image placeholders.
            max_seq_len: Maximum sequence length for tokenized output.
            num_workers: Max threads for parallel token length computation.
            skip_reasoning_content: If True, drop reasoning_content on last turn.
            transform: Optional callable to post-process the sample (e.g. cached features).
        """
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.num_workers = num_workers

        self.skip_reasoning_content = skip_reasoning_content
        self.transform = transform

    def __call__(self, data: MMDialog) -> MMSample | None:
        """Convert a multi-modal dialog into model-ready tensors.

        Args:
            data: A MMDialog containing images and conversations.

        Returns:
            A dict with tokens/labels/loss_mask and images (or image paths),
            or None if the sample is invalid or truncated.
        """
        # NOTE: skip reasoning_content for chat model,
        # only last turn may contain reasoning_content
        if self.skip_reasoning_content:
            data["conversations"][-1].pop("reasoning_content", None)

        sample = self.apply_chat_template(data["conversations"])

        if sample is None:
            return None

        # NOTE: only SFT consider this case for better training efficiency
        if int(sample["loss_mask"].sum().item()) == 0:
            logger.error("Empty loss masked sample, skipped." + self.tokenizer.decode(sample["all_tokens"].tolist()))
            # set loss_size as zero to skip maybe-truncated sample
            return None

        sample["image_path"] = data["images"]

        # Compatibility with qwenvl format
        if "image_grid_thw" in data:
            sample["image_grid_thw"] = data["image_grid_thw"]

        # NOTE: we process images and cache features in advance
        # here we directly fetch feature paths from cache
        if self.transform is None:
            return sample
        return self.transform(sample)

    def _tokenize(self, messages: Dialog, tools: list | None = None) -> list[int]:
        """
        Tokenize a list of messages into a single sequence of token IDs.

        Args:
            messages: A list of message dictionaries.
            tools: Optional list of tool schemas.

        Returns:
            A list of integer token IDs.
        """
        # If the last message is NOT from the assistant, we add a generation prompt
        # to signal the model to start generating a response.
        add_generation_prompt = messages[-1]["role"] != "assistant"
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            tools=tools,
            add_generation_prompt=add_generation_prompt,
        )

    def _find_last_user_idx(self, data: Dialog) -> int:
        """
        Find the index of the last message from the 'user' role.

        Returns:
            The index of the last user message, or -1 if not found.
        """
        for i in range(len(data) - 1, -1, -1):
            if data[i]["role"] == "user":
                return i
        return -1

    def _compute_cumulative_lengths(self, data: Dialog, start_idx: int, tools: list | None) -> dict[int, int]:
        """
        Compute cumulative token lengths for messages starting from start_idx in parallel.

        This is used to determine the start and end boundaries of each turn in the
        tokenized sequence without sequentially re-tokenizing everything.

        Args:
            data: The full dialog.
            start_idx: The index to start computing lengths from.
            tools: Optional tool schemas.

        Returns:
            A dictionary mapping message index to its cumulative token length.
        """
        token_lengths = {}

        def compute_length(i):
            # Calculate length up to message i (inclusive)
            return i, len(self._tokenize(data[: i + 1], tools=tools))

        # Determine number of workers dynamically but cap at self.num_workers
        workers = min(self.num_workers, len(data) - start_idx)
        if workers <= 0:
            return token_lengths

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(compute_length, i) for i in range(start_idx, len(data))]
            for future in as_completed(futures):
                i, length = future.result()
                token_lengths[i] = length

        return token_lengths

    def apply_chat_template(self, data: Dialog) -> MMSample | None:
        """
        Apply the chat template to a dialog, generating tokens and loss masks.

        The logic is as follows:
        1. Validate that the conversation ends with an assistant message.
        2. Tokenize the entire conversation.
        3. Find the last user message. All tokens before this point are masked out (loss=0).
        4. For messages after the last user turn:
           - If it's an assistant message AND has loss_mask=1.0, we compute loss on it.
           - Otherwise, we mask it out.
        5. Use parallel processing to efficiently calculate the token boundaries for these turns.
        """
        # 1. Validation
        assert data[-1]["role"] == "assistant", f"Final turn must be from assistant, but got:\n{data[-1]}"

        # 2. Setup
        tool_schemas = data[0].get("tool_schemas")

        # 3. Get full token sequence
        all_tokens = self._tokenize(data, tools=tool_schemas)

        # 4. Identify training range (everything after the last user message)
        last_user_idx = self._find_last_user_idx(data)

        if last_user_idx == -1:
            logger.error(
                "No user turn found in dialog data. This indicates problematic data."
                f"data path: {data[0]['data_path']}. Skipping this sample..."
            )
            return None

        # 5. Calculate mask
        # Initialize mask with zeros (no training by default)
        loss_mask = torch.zeros(len(all_tokens), dtype=torch.float32)

        # Calculate the boundary where potential training begins (end of last user turn)
        tokens_up_to_last_user = self._tokenize(data[: last_user_idx + 1], tools=tool_schemas)
        current_pos = len(tokens_up_to_last_user)

        # Compute cumulative token lengths for all subsequent turns in parallel
        # We need these to know exactly where each turn starts and ends in 'all_tokens'
        cumulative_lengths = self._compute_cumulative_lengths(data, start_idx=last_user_idx + 1, tools=tool_schemas)

        # Iterate through turns after the last user message
        for i in range(last_user_idx + 1, len(data)):
            end_pos = cumulative_lengths[i]

            # We train on this segment if:
            # 1. It is an assistant response
            # 2. The data explicitly says we should mask it (loss_mask=1.0, default is 1.0)
            if data[i]["role"] == "assistant" and data[i].get("loss_mask", 1.0) == 1.0:
                loss_mask[current_pos:end_pos] = 1.0

            current_pos = end_pos

        return {
            "tokens": torch.LongTensor(all_tokens[:-1]),
            "labels": torch.LongTensor(all_tokens[1:]),
            "loss_mask": loss_mask[1:],
            "all_tokens": torch.LongTensor(all_tokens),
        }
