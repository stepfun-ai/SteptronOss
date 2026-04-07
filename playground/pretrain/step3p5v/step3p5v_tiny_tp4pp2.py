"""
Tiny Step3p5v multimodal smoke config for 8 GPUs.

- LLM mesh: TP4 PP2
- vision mesh: TP2 with 4 replicated lanes

This smoke uses a more realistic duplicate-source pattern:
- rank 0 and rank 1 both hold the same 9-image batch on the LLM source side
- cross-mesh routing collapses those TP-duplicate sources before encoder dispatch
- encoder-side shard sizes are duplicated inside each TP pair, so physical ranks
  observe a `3, 3, 2, 2, 2, 2, 2, 2` split
- after backward gather, rank 0 and rank 1 should hold the same 9 image features
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist

from playground.pretrain.step3p5v.step3p5v_toy import Step3p5vToyModelConfig
from steptronoss.core.parallel_state import PM
from steptronoss.initialize import set_mpu_random_seed
from steptronoss.model.common.parallel_embedding import ImageForInsert


class Step3p5vTinyTp4Pp2ModelConfig(Step3p5vToyModelConfig):
    def __init__(self):
        super().__init__()

        self.params_dtype = torch.bfloat16

        self.num_layers = 2
        self.swa_layer_list = [False, False]
        self.hidden_size = 64

        self.attn_cfg.num_attention_heads = 8
        self.attn_cfg.num_attention_groups = 1
        self.attn_cfg.head_dim = 8
        self.attn_cfg.hidden_size = self.hidden_size
        self.swa_cfg.num_attention_heads = 8
        self.swa_cfg.num_attention_groups = 1
        self.swa_cfg.head_dim = 8
        self.swa_cfg.hidden_size = self.hidden_size

        self.ffn_cfg.ffn_hidden_size = 128
        self.ffn_cfg.hidden_size = self.hidden_size
        self.ffn_cfg.moe_cfg.hidden_size = self.hidden_size
        self.ffn_cfg.moe_cfg.moe_layer_list = []
        self.ffn_cfg.moe_cfg.share_expert_dim = 0

        self.parallel_cfg.tensor_model_parallel_size = 4
        self.parallel_cfg.pipeline_model_parallel_size = 2
        self.parallel_cfg.virtual_pipeline_model_parallel_size = 1
        self.parallel_cfg.context_parallel_size = 1
        self.parallel_cfg.expert_model_parallel_size = 1
        self.parallel_cfg.expert_tensor_parallel_size = 1

        self.tp_cfg.sequence_parallel = False
        self.tp_cfg.async_tensor_model_parallel_allreduce = False
        self.variable_seq_lengths = True

        self.tok_embed_cfg.hidden_size = self.hidden_size
        self.out_embed_cfg.hidden_size = self.hidden_size
        self.tok_embed_cfg.vocab_size = 512
        self.out_embed_cfg.vocab_size = 512
        self.tok_embed_cfg.img_start_token = 7

        self.tok_embed_cfg.encoder_cfg.image_size = 28
        self.tok_embed_cfg.encoder_cfg.patch_size = 14
        self.tok_embed_cfg.encoder_cfg.hidden_size = 32
        self.tok_embed_cfg.encoder_cfg.ffn_hidden_size = 64
        self.tok_embed_cfg.encoder_cfg.num_layers = 2
        self.tok_embed_cfg.encoder_cfg.num_attention_heads = 4
        self.tok_embed_cfg.encoder_cfg.vit_downsampler_hidden_dim = 48
        self.tok_embed_cfg.encoder_cfg.output_dim = self.hidden_size
        self.tok_embed_cfg.encoder_cfg.parallel_cfg.tensor_model_parallel_size = 2
        self.tok_embed_cfg.encoder_cfg.parallel_cfg.pipeline_model_parallel_size = 4
        self.tok_embed_cfg.encoder_cfg.parallel_cfg.virtual_pipeline_model_parallel_size = 1
        self.tok_embed_cfg.encoder_cfg.parallel_cfg.context_parallel_size = 1
        self.tok_embed_cfg.encoder_cfg.parallel_cfg.expert_model_parallel_size = 1
        self.tok_embed_cfg.encoder_cfg.parallel_cfg.expert_tensor_parallel_size = 1


def _build_rank_input(cfg: Step3p5vTinyTp4Pp2ModelConfig, rank: int):
    seq_len = 20
    input_ids = torch.ones((1, seq_len), device="cuda", dtype=torch.long)
    image_slots = torch.arange(0, 18, 2, device="cuda")
    input_ids[0, image_slots] = cfg.tok_embed_cfg.img_start_token

    if rank not in {0, 1}:
        return input_ids, None

    image_count = 9
    image = torch.linspace(
        0.0,
        1.0,
        steps=image_count
        * cfg.tok_embed_cfg.encoder_cfg.in_channels
        * cfg.tok_embed_cfg.encoder_cfg.image_size
        * cfg.tok_embed_cfg.encoder_cfg.image_size,
        dtype=torch.float32,
    ).reshape(
        image_count,
        cfg.tok_embed_cfg.encoder_cfg.in_channels,
        cfg.tok_embed_cfg.encoder_cfg.image_size,
        cfg.tok_embed_cfg.encoder_cfg.image_size,
    )
    images = [ImageForInsert(insert_start_token=cfg.tok_embed_cfg.img_start_token, images=image)]
    return input_ids, images


def run_smoke():
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = dist.get_rank()
    torch.cuda.set_device(local_rank)

    PM.initialize(backend="nccl")
    cfg = Step3p5vTinyTp4Pp2ModelConfig()
    PM.set_mesh(cfg.parallel_cfg)
    set_mpu_random_seed(1234)

    model = cfg.build_model().cuda().to(cfg.params_dtype)
    model.eval()

    input_ids, images = _build_rank_input(cfg, rank)

    with torch.no_grad():
        broadcast_images = model.mesh_connector.broadcast(images)
        assert broadcast_images is not None
        assert len(broadcast_images) == 1
        assert broadcast_images[0].images is not None
        assert tuple(broadcast_images[0].images.shape) == (
            9,
            cfg.tok_embed_cfg.encoder_cfg.in_channels,
            cfg.tok_embed_cfg.encoder_cfg.image_size,
            cfg.tok_embed_cfg.encoder_cfg.image_size,
        )

        local_images = model.mesh_connector.forward(broadcast_images[0].images)
        local_image_count = 0 if local_images is None else int(local_images.shape[0])

        with PM.use_mesh(cfg.tok_embed_cfg.encoder_cfg.parallel_cfg):
            local_features = model.encoder(local_images.to(device=torch.cuda.current_device(), dtype=cfg.params_dtype))
        gathered_features = model.mesh_connector.backward(local_features)

        prepared_images = [
            ImageForInsert(
                insert_start_token=broadcast_images[0].insert_start_token,
                image_features=gathered_features,
            )
        ]
        feature_count = 0 if gathered_features is None else int(gathered_features.shape[0])

        local_summary = {
            "rank": rank,
            "pp_rank": PM.rank_in("PP"),
            "tp_rank": PM.rank_in("TP"),
            "local_image_count": local_image_count,
            "has_tok_embeddings": hasattr(model, "tok_embeddings"),
            "has_features": gathered_features is not None,
            "feature_count": feature_count,
            "feature_shape": (tuple(gathered_features.shape) if gathered_features is not None else None),
        }

        if hasattr(model, "tok_embeddings"):
            hidden = model.forward_head(input_ids=input_ids, images=prepared_images)
            local_summary["hidden_shape"] = tuple(hidden.shape)
            local_summary["hidden_finite"] = bool(torch.isfinite(hidden).all().item())

    feature_payload = gathered_features.detach().float().cpu() if gathered_features is not None else None

    gathered = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered, local_summary)
    feature_gather = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(feature_gather, feature_payload)

    if rank == 0:
        summaries_by_rank = {item["rank"]: item for item in gathered}
        features_by_rank = {
            rank_id: feature_gather[idx] for idx, rank_id in enumerate(item["rank"] for item in gathered)
        }
        expected_counts = {
            0: 3,
            1: 3,
            2: 2,
            3: 2,
            4: 2,
            5: 2,
            6: 2,
            7: 2,
        }
        for rank_id, expected_count in expected_counts.items():
            assert summaries_by_rank[rank_id]["local_image_count"] == expected_count, summaries_by_rank[rank_id]
        assert summaries_by_rank[0]["feature_shape"] == (9, 1, 64)
        assert summaries_by_rank[1]["feature_shape"] == (9, 1, 64)
        torch.testing.assert_close(features_by_rank[0], features_by_rank[1])

        print("step3p5v tiny smoke passed")
        for item in sorted(gathered, key=lambda item: item["rank"]):
            print(item)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    run_smoke()
