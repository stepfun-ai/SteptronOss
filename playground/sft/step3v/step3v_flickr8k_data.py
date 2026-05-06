from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from steptronoss.data.multimodal import IMAGE_ITEM_TYPE, PATCH_ITEM_TYPE, build_image_for_insert
from steptronoss.data.nextable.nextable import Nextable
from steptronoss.exp.sft import SFTDataConfig


@dataclass(frozen=True)
class Flickr8kSample:
    image_path: str
    caption: str


class Step3VFlickr8kNextable(Nextable):
    def __init__(self, cfg: Step3VFlickr8kDataConfig):
        self.cfg = cfg
        self.step = 0
        self.processor = cfg.build_processor()
        self.samples = cfg.prepare_samples()

    def __next__(self) -> dict:
        sample = self.samples[self.step % len(self.samples)]
        self.step += 1
        return self.cfg.encode_sample(self.processor, sample)

    def state_dict(self) -> dict:
        return {"step": self.step}

    def load_state_dict(self, state_dict: dict) -> None:
        self.step = int(state_dict.get("step", 0))


class Step3VFlickr8kDataConfig(SFTDataConfig):
    """Tiny real-image captioning SFT stream over the CC0 Flickr8k HF dataset."""

    global_data_keys = ["cu_seqlens", "position_id"]
    """Batch keys that must be visible on every pipeline rank."""

    repo_id: str = "intro/flickr8k"
    """Hugging Face dataset repo id."""

    split: str = "train"
    """Dataset split used for the smoke run."""

    sample_count: int = 4
    """Number of images downloaded and cycled by the smoke dataloader."""

    caption_key: str = "caption_0"
    """Caption column used as the assistant answer."""

    tokenizer_path: str = "/oss/opensource_models/Step3-VL-10B"
    """Local Step3-VL processor/tokenizer path."""

    cache_dir: str = ".cache/step3v_flickr8k_smoke"
    """Local cache for the tiny downloaded Flickr8k subset."""

    prompt: str = "Describe this image in one sentence."
    """User prompt used to turn caption pairs into chat SFT samples."""

    img_start_token: int = 151680
    """Step3-VL <im_start> token id."""

    patch_start_token: int = 151689
    """Step3-VL <patch_start> token id."""

    max_seq_len: int = 2048
    """Guardrail for this smoke config; overlong samples are rejected."""

    def _cache_dir(self) -> Path:
        return Path(self.cache_dir).expanduser()

    def _get_dataset_file(self, filename: str, cache_dir: Path) -> Path:
        local_path = cache_dir / filename
        if local_path.is_file() and local_path.stat().st_size > 0:
            return local_path

        from huggingface_hub import hf_hub_download

        return Path(
            hf_hub_download(
                repo_id=self.repo_id,
                repo_type="dataset",
                filename=filename,
                local_dir=str(cache_dir),
            )
        )

    def prepare_samples(self) -> list[Flickr8kSample]:
        cache_dir = self._cache_dir()
        cache_dir.mkdir(parents=True, exist_ok=True)

        metadata_path = self._get_dataset_file(f"{self.split}/metadata.csv", cache_dir)
        with open(metadata_path, encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))

        samples: list[Flickr8kSample] = []
        for row in rows:
            caption = row.get(self.caption_key, "").strip()
            file_name = row.get("file_name", "").strip()
            if not caption or not file_name:
                continue
            image_path = self._get_dataset_file(f"{self.split}/{file_name}", cache_dir)
            samples.append(Flickr8kSample(image_path=str(image_path), caption=caption))
            if len(samples) >= self.sample_count:
                break

        if not samples:
            raise RuntimeError(f"No Flickr8k samples prepared from {self.repo_id}/{self.split}")
        return samples

    def build_processor(self):
        from transformers import AutoProcessor

        return AutoProcessor.from_pretrained(self.tokenizer_path, trust_remote_code=True)

    def build_dataloader(self, dp_rank=0, dp_size=1):
        del dp_rank, dp_size
        return Step3VFlickr8kNextable(self)

    def _messages(self, caption: str) -> list[dict[str, Any]]:
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": self.prompt},
                ],
            },
            {"role": "assistant", "content": caption},
        ]

    def _encode(self, processor, messages: list[dict[str, Any]], image: Image.Image, *, add_generation_prompt: bool):
        text = processor.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
        return processor(text=text, images=image, return_tensors="pt")

    def encode_sample(self, processor, sample: Flickr8kSample) -> dict:
        with Image.open(sample.image_path) as img:
            image = img.convert("RGB")
            messages = self._messages(sample.caption)
            full = self._encode(processor, messages, image, add_generation_prompt=False)
            prefix = self._encode(processor, messages[:1], image, add_generation_prompt=True)

        tokens = full["input_ids"][0].to(torch.long)
        if tokens.numel() > self.max_seq_len:
            raise ValueError(f"Flickr8k smoke sample has {tokens.numel()} tokens > max_seq_len={self.max_seq_len}")

        labels = torch.roll(tokens, shifts=-1)
        labels[-1] = 0

        prefix_len = min(int(prefix["input_ids"].shape[-1]), int(tokens.numel()))
        loss_mask = torch.zeros_like(tokens, dtype=torch.float32)
        loss_mask[prefix_len:] = 1.0
        loss_mask[-1] = 0.0

        images = [(image_tensor.float(), IMAGE_ITEM_TYPE) for image_tensor in full["pixel_values"]]
        for patch_tensor in full.get("patch_pixel_values", torch.empty(0)):
            images.append((patch_tensor.float(), PATCH_ITEM_TYPE))

        seq_len = int(tokens.numel())
        return {
            "tokens": tokens,
            "labels": labels,
            "loss_mask": loss_mask,
            "cu_seqlens": torch.tensor([0, seq_len], dtype=torch.int32),
            "position_id": torch.arange(seq_len, dtype=torch.int32),
            "images": images,
        }

    def preprocess(self, batch: dict):
        cu_seqlens = batch["cu_seqlens"].to("cuda")
        position_id = batch["position_id"].to("cuda")
        max_seq_len = torch.max(cu_seqlens[1:] - cu_seqlens[:-1])

        if "tokens" not in batch:
            return {
                "cu_seqlens": cu_seqlens,
                "max_seq_len": max_seq_len,
                "position_id": position_id,
            }

        tokens = batch["tokens"].to("cuda")
        labels = batch["labels"].to("cuda")
        loss_masks = batch["loss_mask"].to("cuda")
        image_count = int(torch.sum(tokens == self.img_start_token).item())
        patch_count = int(torch.sum(tokens == self.patch_start_token).item())

        images = build_image_for_insert(
            batch.get("images", []),
            patch_start_id=self.patch_start_token,
            image_start_id=self.img_start_token,
            limit_images=image_count,
            limit_patches=patch_count,
            to_cuda=True,
        )

        return {
            "input_ids": tokens[None].contiguous(),
            "labels": labels[None].contiguous(),
            "loss_masks": loss_masks[None].contiguous(),
            "images": images,
            "cu_seqlens": cu_seqlens,
            "max_seq_len": max_seq_len,
            "position_id": position_id,
        }
