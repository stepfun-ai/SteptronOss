import sys
import time

from playground.data.sft.reasoning_GCMKSTIDF_sft_stage1_1203_compile_step3 import (
    Step3SFTDataStep3TokenizedConfig,
)
from playground.pretrain.step3p5.step3p5_toy import Step3p5ToyModelConfig
from playground.sft.qwen3.qwen3_sft_base import Exp as BaseExp
from steptronoss.exp.base_exp import GradientManagerConfig
from steptronoss.exp.optimizer import MuonConfig


class Step3p5MuonConfig(MuonConfig):
    def mark_gather_ops(self, model) -> None:
        from steptronoss.checkpointing.reshape_ops import (
            ColumnParallel,
            FFNMergeGateUp,
            GQAMergeQKV,
            GQAMergeQKVBias,
            GQAMergeQKVG,
            Inverse,
            KeepThisTP,
            Rename,
            RowParallel,
            Sequential,
        )
        from steptronoss.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
        from steptronoss.model.common.feed_forward import FeedForward
        from steptronoss.model.common.grouped_query_attention import GroupedQueryAttention
        from steptronoss.model.common.moe_block import MoEGate

        identity = Sequential([])
        merge_ops: dict[int, object] = {}

        # Specialized packed weights (GQA, gate/up).
        for module in model.modules():
            if not isinstance(module, GroupedQueryAttention):
                continue
            wqkv = getattr(module, "wqkv", None)
            if wqkv is None:
                continue

            weight = getattr(wqkv, "weight", None)
            bias = getattr(wqkv, "bias", None)
            if weight is not None:
                if module.wqkv_extra_dims:
                    merge_op = Sequential(
                        [
                            Rename("grad.qkv: grad"),
                            Inverse(KeepThisTP()),
                            Inverse(
                                GQAMergeQKVG(
                                    group_num=module.num_kv_heads,
                                    head_dim=module.head_dim,
                                    gate_dims=module.wqkv_extra_dims,
                                )
                            ),
                        ]
                    )
                else:
                    merge_op = Sequential(
                        [
                            Rename("grad.qkv: grad"),
                            Inverse(KeepThisTP()),
                            Inverse(
                                GQAMergeQKV(
                                    group_num=module.num_kv_heads,
                                    head_dim=module.head_dim,
                                )
                            ),
                        ]
                    )
                merge_ops[id(weight)] = merge_op

            if bias is not None and not module.wqkv_extra_dims:
                merge_ops[id(bias)] = Sequential(
                    [
                        Rename("grad.qkv: grad"),
                        Inverse(KeepThisTP()),
                        Inverse(
                            GQAMergeQKVBias(
                                group_num=module.num_kv_heads,
                                head_dim=module.head_dim,
                            )
                        ),
                    ]
                )

        for module in model.modules():
            if not isinstance(module, FeedForward):
                continue
            w1 = getattr(module, "w1", None)
            w2 = getattr(module, "w2", None)
            if w1 is not None and hasattr(w1, "weight"):
                group = "ETP" if getattr(w1, "use_moe", False) else "TP"
                merge_ops.setdefault(
                    id(w1.weight),
                    Sequential(
                        [
                            Rename("grad.gate_up: grad"),
                            Inverse(KeepThisTP(group=group)),
                            Inverse(FFNMergeGateUp(group=group)),
                        ]
                    ),
                )
            if w2 is not None and hasattr(w2, "weight"):
                group = "ETP" if getattr(w2, "use_moe", False) else "TP"
                merge_ops.setdefault(
                    id(w2.weight),
                    Inverse(RowParallel(group=group) + KeepThisTP(group=group)),
                )

        # Generic TP-sharded linear weights.
        for module in model.modules():
            if isinstance(module, ColumnParallelLinear):
                weight = getattr(module, "weight", None)
                if weight is not None and weight.ndim == 2:
                    group = "ETP" if getattr(module, "use_moe", False) else "TP"
                    merge_ops.setdefault(
                        id(weight),
                        Inverse(ColumnParallel(group=group) + KeepThisTP(group=group)),
                    )
            elif isinstance(module, RowParallelLinear):
                weight = getattr(module, "weight", None)
                if weight is not None and weight.ndim == 2:
                    group = "ETP" if getattr(module, "use_moe", False) else "TP"
                    merge_ops.setdefault(
                        id(weight),
                        Inverse(RowParallel(group=group) + KeepThisTP(group=group)),
                    )
            elif isinstance(module, MoEGate):
                weight = getattr(module, "weight", None)
                if weight is not None:
                    merge_ops.setdefault(id(weight), identity)

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            merge_op = merge_ops.get(id(param))

            if merge_op is None:
                if getattr(param, "tensor_model_parallel", False):
                    partition_dim = getattr(param, "partition_dim", -1)
                    if partition_dim == 0:
                        merge_op = Inverse(ColumnParallel() + KeepThisTP())
                    elif partition_dim == 1:
                        merge_op = Inverse(RowParallel() + KeepThisTP())
                    else:
                        merge_op = identity
                else:
                    merge_op = identity

            setattr(param, "merge_op", merge_op)


# sys.excepthook = lambda a, b, c: [print("HoldOneError"), time.sleep(3600)]


class MuonGradientManagerConfig(GradientManagerConfig):
    optimizer_cfg = Step3p5MuonConfig
    """Use Muon for 2D params with AdamW fallback."""


class Exp(BaseExp):
    model_cfg = Step3p5ToyModelConfig

    data_cfg = Step3SFTDataStep3TokenizedConfig

    optimizer_cfg = MuonGradientManagerConfig

    def __init__(self):
        super().__init__()
        self.trainer_cfg.micro_batch_size = 1
        self.trainer_cfg.global_batch_size = 8
        self.trainer_cfg.global_seq_length = 65536

        self.trainer_cfg.train_iters = 1000
        self.trainer_cfg.log_interval = 1

        self.checkpoint_cfg.load_option.none(but=["model"])
        # self.checkpoint_cfg.load_safetensors = "/mnt/step2-alignment-jfs/zane/opensources_model/Qwen3-30B-A3B-Base"
        self.checkpoint_cfg.save_safetensors = True
        self.checkpoint_cfg.save_dir = "/mnt/shared-storage/tenant/tmp/zhy/tmp/"
        self.checkpoint_cfg.save_option.all()
        self.checkpoint_cfg.save_interval = 100
        self.profiler_cfg.timing_log_level = 2
        self.model_cfg.recompute = True

    def configure_optimizable(self):
        from steptronoss.utils.optimizable import set_optimization

        set_optimization(
            # grouped_gemm="nv_grouped_gemm",
            AttentionCore="flash-attn",
            default=None,
        )


if __name__ == "__main__":
    Exp().train()
