from collections import defaultdict
from typing import Literal

import torch
from configurize import Ref

from playground.pretrain.step3v.step3v_10b import (
    Step3V_10BConfig,
    Step3VTokenizerConfig,
)
from playground.sft.qwen3.qwen3_sft_base import Exp as BaseExp
from playground.tools.compile_recipe import CompliableDatasetsConfig
from steptronoss.exp.sft import SFTDataConfig
from steptronoss.model.common.encoder_as_embedding import ImageForInsert


class ExampleMMDatasetsConfig(CompliableDatasetsConfig):
    image_file_root = "/mnt/shared-storage/tenant/zhy/mini-o3-train-transformed_multicrop_images/"
    file_list = [
        "/mnt/shared-storage/tenant/zhy/mini-o3-train-transformed-multicropped-7223-zhy.jsonl",
    ]
    epoch: float = 1
    image_size: tuple[int] = (728, 728)
    patch_size: tuple[int] = (504, 504)

    img_norm_mean: tuple[float] = (0.48145466, 0.4578275, 0.40821073)
    img_norm_std: tuple[float] = (0.26862954, 0.26130258, 0.27577711)

    max_seq_len: int = 16384
    skip_reasoning_content: bool = False

    tokenizer_path: str = Ref("...tokenizer_cfg.tokenizer_path")

    def get_template(self):
        from torchvision import transforms as T
        from transformers import AutoTokenizer

        from steptronoss.data.chat_templates.mm_template import (
            ImageLoadTransform,
            ImageTokenVerifyTransform,
            ImageTypeToStartTokenTransform,
            MMMultiTurnChatTemplate,
            ProcessImageTransform,
        )

        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)

        image_start_token = tokenizer.encode("<im_start>")[0]
        patch_start_token = tokenizer.encode("<patch_start>")[0]

        image_end_token = tokenizer.encode("<im_end>")[0]
        patch_end_token = tokenizer.encode("<patch_end>")[0]

        template = MMMultiTurnChatTemplate(
            tokenizer=tokenizer,
            max_seq_len=self.max_seq_len,
            skip_reasoning_content=self.skip_reasoning_content,
            transform=T.Compose([
                ImageTokenVerifyTransform(img_end_token=image_end_token, patch_end_token=patch_end_token),
                ImageLoadTransform(root_path=self.image_file_root),
                ProcessImageTransform(
                    im_size=self.image_size,
                    patch_size=self.patch_size,
                    mean=self.img_norm_mean,
                    std=self.img_norm_std,
                ),
                ImageTypeToStartTokenTransform({0: image_start_token, 1: patch_start_token}),
            ]),
        )
        return template

    def get_recipe(self):
        from steptronoss.data.recipe import DataRecipe, DataSourceFile

        return DataRecipe(
            domains={"default": [DataSourceFile(f) for f in self.file_list]},
            epochs={"default": self.epoch},
        )


from playground.tools.compile_recipe import CompiledDatasetsConfig
from steptronoss.data.recipe import CompiledDataRecipe


class ExampleMMDatasetsConfigCompiled(CompiledDatasetsConfig):
    compiled_recipe = CompiledDataRecipe(
        domains={
            "default": "/mnt/shared-storage/tenant/zhy/mini-o3-step3vl-compiled/default",
        },
        epochs={
            "default": 1,
        },
    )


class MyDataConfig(SFTDataConfig):
    dataset_cfg: ExampleMMDatasetsConfig = ExampleMMDatasetsConfigCompiled

    oversize_policy: Literal["drop", "extend"] = "drop"

    max_packing_seqlen = Ref("..trainer_cfg.global_seq_length")

    seqlen_divisible_by: int = 64

    global_data_keys = ["cu_seqlens"]

    def build_dataloader(self, dp_rank=0, dp_size=1):
        from steptronoss.data.dataloader.packed_dataloader import MixedPackedDataloader
        from steptronoss.data.nextable import DPMux, async_accelearte_slowfast

        datasets = self.dataset_cfg.build_datasets()
        dataloader = MixedPackedDataloader(
            datasets=[ds[0] for ds in datasets.values()],
            epochs=[ds[1] for ds in datasets.values()],
            max_length=self.max_packing_seqlen,
            oversize_policy=self.oversize_policy,
            transform=self.pack,
        )
        dataloader = DPMux(dataloader, dp_size=dp_size, dp_rank=dp_rank)

        # dataloader = async_accelearte_slowfast(dataloader)  # optional, remove for debug
        return dataloader

    def preprocess(self, batch: dict):
        cu_seqlens = batch["cu_seqlens"].to("cuda")
        max_seq_len = torch.max(cu_seqlens[1:] - cu_seqlens[:-1])

        if "tokens" in batch:  # on the head or tail of the pipeline parallel
            tokens = batch["tokens"].to("cuda")
            labels = batch["labels"].to("cuda")
            loss_masks = batch["loss_mask"].to("cuda")
            images: list[ImageForInsert] = batch["images_for_insert"]

            return dict(
                input_ids=tokens[None].contiguous(),
                labels=labels[None].contiguous(),
                loss_masks=loss_masks[None].contiguous(),
                images=images,
                cu_seqlens=cu_seqlens,
                max_seq_len=max_seq_len,
            )
        else:
            return dict(
                cu_seqlens=cu_seqlens,
                max_seq_len=max_seq_len,
            )

    def pack(self, pieces: list):
        import numpy as np

        size = sum([len(s["tokens"]) - 1 for s in pieces])

        if size % self.seqlen_divisible_by != 0:
            # padding to the tensor_model_parallel_size
            padding_size = self.seqlen_divisible_by - size % self.seqlen_divisible_by

            padding_tensor = np.zeros(padding_size + 1)
            pieces.append({
                "tokens": padding_tensor,
                "loss_mask": padding_tensor,
            })

        sizes = torch.tensor([len(s["tokens"]) - 1 for s in pieces])
        from functools import reduce

        from torch import tensor as T

        tokens = torch.cat([T(s["tokens"][:-1], dtype=torch.long) for s in pieces])
        labels = torch.cat([T(s["tokens"][1:], dtype=torch.long) for s in pieces])
        loss_mask = torch.cat([T(s["loss_mask"][1:], dtype=torch.float32) for s in pieces])
        images = reduce(sum, [x.get("images", []) for x in pieces], [])
        images_grouped_by_strat_token = defaultdict(list)
        for img, im_start_token in images:
            images_grouped_by_strat_token[im_start_token].append(img)
        images_for_insert = [
            ImageForInsert(insert_start_token=im_start_token, images=torch.stack(images).bfloat16())
            for im_start_token, images in images_grouped_by_strat_token.items()
        ]

        cu_seqlens = torch.cat([
            torch.zeros(1),
            torch.cumsum(sizes, 0),
        ]).int()

        return dict(
            tokens=tokens,
            labels=labels,
            loss_mask=loss_mask,
            images_for_insert=images_for_insert,
            cu_seqlens=cu_seqlens,
            max_seq_len=sizes.max(),
        )


class Exp(BaseExp):
    model_cfg = Step3V_10BConfig
    """Model config for Step3-VL-10B."""
    tokenizer_cfg = Step3VTokenizerConfig
    """Tokenizer config for Step3-VL-10B."""
    data_cfg: MyDataConfig = MyDataConfig
    """Fake data config for quick SFT validation."""

    def __init__(self):
        super().__init__()
        self.trainer_cfg.micro_batch_size = 1
        self.trainer_cfg.global_batch_size = 16
        self.trainer_cfg.global_seq_length = 32768
        self.trainer_cfg.train_iters = None
        self.trainer_cfg.log_interval = 1

        self.optimizer_cfg.params_dtype = torch.bfloat16
        self.resource_cfg.gpu = 4

        self.checkpoint_cfg.load_safetensors = "/mnt/shared-storage/tenant/zhy/Step3-VL-10B/"
        self.checkpoint_cfg.tokenizer_path = "/mnt/shared-storage/tenant/zhy/Step3-VL-10B/"
        self.checkpoint_cfg.load_option.none(but=["model"])
        self.checkpoint_cfg.save_safetensors = False

    def configure_optimizable(self):
        from steptronoss.utils.optimizable import set_optimization

        set_optimization(AttentionCore="flash-attn", default="torch_compile")


if __name__ == "__main__":
    exp = Exp()
    # Use below code to compile data

    # torch.set_num_threads(1)
    # exp.data_cfg.dataset_cfg = ExampleMMDatasetsConfig()
    # exp.data_cfg.dataset_cfg.compile("/mnt/shared-storage/tenant/zhy/mini-o3-step3vl-compiled/")
    # run python playground/sft/step3v/step3v_10b_sft_example.py

    exp.train()
