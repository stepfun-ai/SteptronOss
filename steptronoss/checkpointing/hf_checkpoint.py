import json
import os
import shutil
from os.path import join

import torch
from loguru import logger
from safetensors.torch import save_file

from steptronoss.core.parallel_state import PM
from steptronoss.exp.base_exp import Megatron3DParallelModelConfig
from steptronoss.utils.dist_utils import all_gather_object
from steptronoss.utils.utils import unwrap_model


def _build_hf_config_overrides(model_cfg: Megatron3DParallelModelConfig) -> dict:
    """
    Build HF config overrides from exp.
    """
    overrides = {}
    # RoPE Scaling
    if model_cfg.rope_cfg is not None:
        rope_cfg = model_cfg.rope_cfg
        rope_scaling = {
            "rope_type": rope_cfg.rope_type,
            "factor": rope_cfg.factor,
            "original_max_position_embeddings": rope_cfg.original_max_position_embeddings,
            "low_freq_factor": rope_cfg.low_freq_factor,
            "high_freq_factor": rope_cfg.high_freq_factor,
        }

        overrides["rope_scaling"] = rope_scaling
        logger.info(f"Overriding rope_scaling: {rope_scaling}")

    return overrides


def _apply_config_overrides(config_path: str, overrides: dict):
    """
    Read config.json, apply overrides, and write back.
    """
    if not overrides:
        return

    with open(config_path) as f:
        config = json.load(f)

    config.update(overrides)

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    logger.info(f"Updated config.json with overrides: {list(overrides.keys())}")


def dump_safetensors(
    save_path: str,
    model_reference_path: str,
    tokenizer_reference_path: str,
    models: list,
    model_cfg: Megatron3DParallelModelConfig | None = None,
):
    # copy json configs
    if PM.world_rank == 0:
        os.makedirs(save_path, exist_ok=True)
        if model_reference_path:
            for file in os.listdir(model_reference_path):
                if file.lower().endswith(".json"):
                    shutil.copy(join(model_reference_path, file), save_path)

        # Override config.json with exp config to ensure train-infer consistency
        if model_cfg is not None:
            config_path = join(save_path, "config.json")
            if os.path.exists(config_path):
                overrides = _build_hf_config_overrides(model_cfg)
                _apply_config_overrides(config_path, overrides)
            else:
                logger.warning(f"config.json not found in {save_path}, skip config override")

        if tokenizer_reference_path:
            for file in os.listdir(tokenizer_reference_path):
                if file.lower() == "config.json":
                    continue  # skip model config from tokenizer folder
                if file.lower().endswith(".json") or file in [
                    "chat_template.jinja",
                    "merges.txt",
                ]:
                    shutil.copy(join(tokenizer_reference_path, file), save_path)
    # sync to ensure path exists
    torch.distributed.barrier(group=PM.group_of("MP"))

    # save safetensors
    total_size, weight_map = 0, {}
    for vp_rank, model in enumerate(models):
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        weight: dict = unwrap_model(model).hf_state_dict()
        if PM.i_am("DP", 0) and PM.i_am("TP", 0):
            chunk_id = vp_rank * PM.size_of("PP") + PM.rank_in("PP")
            file_path = f"model-{chunk_id + 1:05d}.safetensors"
            for k, v in weight.items():
                weight_map[k] = file_path
                total_size += v.numel() * v.dtype.itemsize
            full_path = join(save_path, file_path)
            save_file(weight, full_path)
        for k in list(weight):
            del weight[k]
    weight_map = all_gather_object(weight_map, group=PM.group_of("PP"))
    all_weight_map = {}
    for wm in weight_map:
        all_weight_map |= wm
    if PM.world_rank == 0:
        with open(join(save_path, "model.safetensors.index.json"), "w") as f:
            json.dump(
                {
                    "metadata": {"total_size": total_size},
                    "weight_map": all_weight_map,
                },
                f,
            )

    torch.distributed.barrier()
