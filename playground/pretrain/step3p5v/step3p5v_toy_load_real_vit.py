"""
Distributed toy smoke that loads a real Step3.5v vision encoder.

- LLM mesh: TP4 PP2
- vision mesh: TP2 with 4 replicated lanes

The text stack stays tiny; only the vision encoder comes from a real
checkpoint-shaped HF directory.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
from safetensors import safe_open

from playground.pretrain.step3p5v.step3p5v_tiny_tp4pp2 import Step3p5vTinyTp4Pp2ModelConfig
from steptronoss.core.parallel_state import PM
from steptronoss.initialize import set_mpu_random_seed
from steptronoss.model.common.parallel_embedding import ImageForInsert

REAL_STEP3P5V_HF = Path(
    "/mnt/step2-alignment-jfs/luotingdan/"
    "step3p5v_0213_step3p5v_rlvr_final_combine_512bz_256gpus_fix_newtisconfig_formaldata_32k_sftv2_freezerouter_bf16"
)


class Step3p5vTinyTp4Pp2RealVitModelConfig(Step3p5vTinyTp4Pp2ModelConfig):
    def __init__(self):
        super().__init__()

        encoder_cfg = self.tok_embed_cfg.encoder_cfg
        encoder_cfg.image_size = 728
        encoder_cfg.patch_size = 14
        encoder_cfg.hidden_size = 1536
        encoder_cfg.ffn_hidden_size = 8960
        encoder_cfg.num_layers = 47
        encoder_cfg.num_attention_heads = 16
        encoder_cfg.vit_downsampler_hidden_dim = 3072
        encoder_cfg.output_dim = 6144
        encoder_cfg.layernorm_epsilon = 1e-5
        encoder_cfg.layer_scale_init_value = 0.1
        encoder_cfg.use_cls_token = False
        encoder_cfg.use_ln_pre = True
        encoder_cfg.patch_embed_bias = False
        encoder_cfg.vit_downsampler1_kernel_size = 3
        encoder_cfg.vit_downsampler1_padding = 1
        encoder_cfg.vit_downsampler2_kernel_size = 3
        encoder_cfg.vit_downsampler2_padding = 1


def _remap_vision_key(key: str) -> str:
    if not key.startswith("vision_model."):
        raise ValueError(f"unexpected non-vision key: {key}")
    key = key.removeprefix("vision_model.")
    key = key.replace(".attn.in_proj_weight", ".attn.qkv_proj.weight")
    key = key.replace(".attn.in_proj_bias", ".attn.qkv_proj.bias")
    return key


def _load_full_vision_state(checkpoint_dir: Path) -> dict[str, torch.Tensor]:
    index_path = checkpoint_dir / "model.safetensors.index.json"
    with open(index_path) as f:
        weight_map = json.load(f)["weight_map"]

    state_dict = {}
    shard_cache = {}
    try:
        for key, shard_name in weight_map.items():
            if not key.startswith("vision_model."):
                continue
            if shard_name not in shard_cache:
                shard_cache[shard_name] = safe_open(str(checkpoint_dir / shard_name), framework="pt", device="cpu")
            state_dict[_remap_vision_key(key)] = shard_cache[shard_name].get_tensor(key)
    finally:
        shard_cache.clear()
    return state_dict


def load_real_vit_weights(model, checkpoint_dir: Path) -> None:
    full_state = _load_full_vision_state(checkpoint_dir)
    with PM.use_mesh(model.cfg.tok_embed_cfg.encoder_cfg.parallel_cfg):
        local_state = model.encoder.reshaper.forward(full_state)
        missing, unexpected = model.encoder.load_state_dict(local_state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"real ViT load mismatch: missing={missing}, unexpected={unexpected}")


def _build_rank_input(cfg: Step3p5vTinyTp4Pp2RealVitModelConfig, rank: int):
    seq_len = 220
    input_ids = torch.ones((1, seq_len), device="cuda", dtype=torch.long)
    image_slots = torch.tensor([0, 40, 80, 120], device="cuda")
    input_ids[0, image_slots] = cfg.tok_embed_cfg.img_start_token

    if rank not in {0, 1}:
        return input_ids, None

    image_count = 4
    image = torch.linspace(
        0.0,
        1.0,
        steps=image_count
        * cfg.tok_embed_cfg.encoder_cfg.in_channels
        * cfg.tok_embed_cfg.encoder_cfg.image_size
        * cfg.tok_embed_cfg.encoder_cfg.image_size,
        device="cuda",
        dtype=torch.float32,
    ).reshape(
        image_count,
        cfg.tok_embed_cfg.encoder_cfg.in_channels,
        cfg.tok_embed_cfg.encoder_cfg.image_size,
        cfg.tok_embed_cfg.encoder_cfg.image_size,
    )
    images = [ImageForInsert(insert_start_token=cfg.tok_embed_cfg.img_start_token, images=image)]
    return input_ids, images


def run_smoke(checkpoint_dir: Path = REAL_STEP3P5V_HF):
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_dir}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the distributed real-ViT smoke.")
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    local_rank = int(os.environ["LOCAL_RANK"])
    rank = dist.get_rank()
    torch.cuda.set_device(local_rank)

    PM.initialize(backend="nccl")
    cfg = Step3p5vTinyTp4Pp2RealVitModelConfig()
    PM.set_mesh(cfg.parallel_cfg)
    set_mpu_random_seed(1234)

    model = cfg.build_model().cuda().to(cfg.params_dtype)
    load_real_vit_weights(model, checkpoint_dir)
    model.eval()

    input_ids, images = _build_rank_input(cfg, rank)

    with torch.no_grad():
        prepared = model._prepare_inputs({"images": images})
        prepared_images = prepared["images"]
        gathered_features = prepared_images[0].image_features

        local_summary = {
            "rank": rank,
            "pp_rank": PM.rank_in("PP"),
            "tp_rank": PM.rank_in("TP"),
            "has_features": gathered_features is not None,
            "feature_shape": tuple(gathered_features.shape) if gathered_features is not None else None,
        }

        if hasattr(model, "tok_embeddings"):
            hidden = model.forward_head(input_ids=input_ids, images=prepared_images)
            local_summary["hidden_shape"] = tuple(hidden.shape)
            local_summary["hidden_finite"] = bool(torch.isfinite(hidden).all().item())

    gathered = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered, local_summary)
    feature_payload = gathered_features.detach().float().cpu() if gathered_features is not None else None
    feature_gather = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(feature_gather, feature_payload)

    if rank == 0:
        summaries_by_rank = {item["rank"]: item for item in gathered}
        assert summaries_by_rank[0]["feature_shape"] == (4, 169, 6144)
        assert summaries_by_rank[1]["feature_shape"] == (4, 169, 6144)
        assert summaries_by_rank[2]["feature_shape"] == (4, 169, 6144)
        assert summaries_by_rank[3]["feature_shape"] == (4, 169, 6144)
        assert summaries_by_rank[0]["hidden_shape"] == (220, 1, 64)
        assert summaries_by_rank[1]["hidden_shape"] == (220, 1, 64)
        assert summaries_by_rank[2]["hidden_shape"] == (220, 1, 64)
        assert summaries_by_rank[3]["hidden_shape"] == (220, 1, 64)
        torch.testing.assert_close(feature_gather[0], feature_gather[1])
        torch.testing.assert_close(feature_gather[0], feature_gather[2])
        torch.testing.assert_close(feature_gather[0], feature_gather[3])

        print("real vit distributed toy smoke passed")
        print(f"checkpoint_dir={checkpoint_dir}")
        for item in sorted(gathered, key=lambda item: item["rank"]):
            print(item)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    run_smoke()
