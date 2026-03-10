"""
For best performance, this experiment enables:
  - ``AttentionCore="flash-attn-3"``
  - ``grouped_gemm="nv_grouped_gemm"``
  - ``TokenDispatcher="deep_ep"``

Fresh install from source into this repo's ``.venv``
----------------------------------------------------
The commands below do not assume the repos have already been cloned.
They clone the upstream repositories, build against CUDA 12.9, and
install into ``/data/SteptronOss/.venv``.

StepTron uses FlashAttention-3 from the Hopper subpackage, so the
install target is ``flash-attention/hopper`` and the runtime imports are
``flash_attn_interface`` plus ``flash_attn_3._C``.

1. Clone the upstream repositories:

.. code-block:: bash

    export SRC_ROOT=/data
    export VENV=/data/SteptronOss/.venv
    cd "$SRC_ROOT"
    git clone --recursive https://github.com/Dao-AILab/flash-attention.git
    git clone --recursive https://github.com/fanshiqing/grouped_gemm
    git clone https://github.com/deepseek-ai/DeepEP

    cd "$SRC_ROOT/flash-attention" && git submodule update --init --recursive
    cd "$SRC_ROOT/grouped_gemm" && git submodule update --init --recursive

2. Point the build to CUDA 12.9 and make sure the Python build helpers exist:

.. code-block:: bash

    export CUDA_HOME=/data/cuda/cuda-12.9/cuda
    export CUDACXX=$CUDA_HOME/bin/nvcc
    "$VENV/bin/python" -m pip install wheel ninja packaging

3. Install FlashAttention-3:

   ``FLASH_ATTENTION_DISABLE_SM80=TRUE`` is optional. Keep it on
   Hopper-only machines to skip SM80 kernels and reduce build time. Omit
   it if you want the build to also include SM80 support.

.. code-block:: bash

    FLASH_ATTENTION_FORCE_BUILD=TRUE \
    TORCH_CUDA_ARCH_LIST=9.0 \
    FLASH_ATTENTION_DISABLE_SM80=TRUE \
    "$VENV/bin/python" -m pip install -e "$SRC_ROOT/flash-attention/hopper" --no-build-isolation

4. Install nv-grouped-gemm:

.. code-block:: bash

    "$VENV/bin/python" -m pip install -e "$SRC_ROOT/grouped_gemm" --no-build-isolation

5. Install DeepEP:

.. code-block:: bash

    NVSHMEM_DIR=$(
      "$VENV/bin/python" - <<'PY'
    import importlib.util
    spec = importlib.util.find_spec("nvidia.nvshmem")
    if not spec or not spec.submodule_search_locations:
        raise SystemExit("nvidia.nvshmem not found in .venv")
    print(spec.submodule_search_locations[0])
    PY
    ) "$VENV/bin/python" -m pip install -e "$SRC_ROOT/DeepEP" --no-build-isolation

   ``deep_ep`` expects NVSHMEM. The command above resolves ``NVSHMEM_DIR``
   from the ``nvidia.nvshmem`` package inside ``.venv``. If that import
   fails, install a matching NVSHMEM package first.

6. Verify the imports:

.. code-block:: bash

    "$VENV/bin/python" - <<'PY'
    import flash_attn_interface, flash_attn_3._C
    import grouped_gemm, grouped_gemm_backend
    import deep_ep, deep_ep_cpp
    print("cuda extension imports ok")
    PY

Optional GPU smoke tests:

.. code-block:: bash

    "$VENV/bin/python" - <<'PY'
    import torch
    import flash_attn_interface

    q = torch.randn(1, 16, 2, 64, device="cuda", dtype=torch.bfloat16)
    out = flash_attn_interface.flash_attn_func(q, q, q, causal=False)
    print("flash-attn-3 ok:", out.shape, out.dtype, out.device)
    PY

    "$VENV/bin/python" - <<'PY'
    import torch
    import grouped_gemm

    indices = torch.tensor([[1, 2], [0, 1], [0, 2], [1, 2]], dtype=torch.int32, device="cuda")
    x = torch.tensor([[0, 0, 0, 0], [1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3]], dtype=torch.bfloat16, device="cuda")
    probs = torch.ones((4, 2), dtype=torch.float32, device="cuda")
    y = grouped_gemm.ops.unpermute(*grouped_gemm.ops.permute(x, indices), probs)
    print("nv-grouped-gemm ok:", y.shape, y.dtype, y.device)
    PY
"""

from playground.data.sft.oss260312.step_sft_data_config0311_step3p5_tokenizer import (
    Recipe0311CompiledSFTDataConfig,
)
from playground.pretrain.step3p5.step3p5_flash import Step3p5FlashModelConfig
from playground.sft.qwen3.qwen3_sft_base import Exp as BaseExp
from playground.sft.step3.muon_optimizer import Step3p5MuonConfig
from steptronoss.core.parallel_state import PM, get_vpp_size
from steptronoss.exp.base_exp import GradientManagerConfig
from steptronoss.exp.ntp import MoePretrainMetricConfig
from steptronoss.exp.resources import TorchrunResourceConfig


class Step3F128kSFTResourceConfig(TorchrunResourceConfig):
    def __init__(self):
        super().__init__()
        self.replica = 8
        self.gpu = 8
        self.envs |= {
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        }


class MuonGradientManagerConfig(GradientManagerConfig):
    optimizer_cfg = Step3p5MuonConfig
    """Use Muon for 2D params with AdamW fallback."""


class Step3p5FlashModelConfigBalanced(Step3p5FlashModelConfig):
    """Adjust layermap to reduce PP7 memory by moving layers to PP1/PP2."""

    def __init__(self):
        super().__init__()
        # Disable context parallel to reduce communication/overheads.
        self.parallel_cfg.context_parallel_size = 1
        self.parallel_cfg.tensor_model_parallel_size = 8
        self.tp_cfg.sequence_parallel = True

    def pp_vp_allocation(self, abs_pp_rank: int) -> list[dict]:
        # PP=8, VPP=3 -> 24 slots. Start from 2 layers/slot and drop 1 layer on
        # a few slots to get 45 layers total, while keeping PP7 off the floor.
        lengths = [2] * (PM.size_of("PP") * get_vpp_size())
        lengths[22] = 1  # PP6/vp2
        lengths[23] = 0  # PP7/vp2

        expected = PM.size_of("PP") * get_vpp_size()
        if len(lengths) != expected:
            raise ValueError(f"layermap lengths={len(lengths)} != PP*VPP={expected}")
        return [{}] * lengths[abs_pp_rank]


from steptronoss.exp.lr_schedulers import CosineSchedulerConfig


class Exp(BaseExp):
    log_dir = "/oss/logs/"

    scheduler_cfg = CosineSchedulerConfig

    resource_cfg = Step3F128kSFTResourceConfig

    model_cfg = Step3p5FlashModelConfigBalanced

    metric_cfg = MoePretrainMetricConfig

    data_cfg = Recipe0311CompiledSFTDataConfig

    optimizer_cfg = MuonGradientManagerConfig

    def __init__(self):
        super().__init__()
        self.trainer_cfg.micro_batch_size = 1
        self.trainer_cfg.global_batch_size = 32
        self.trainer_cfg.global_seq_length = 1024 * 128
        self.trainer_cfg.train_iters = None  # auto get

        self.scheduler_cfg.lr = 1e-5
        self.scheduler_cfg.min_lr = 5e-6
        self.scheduler_cfg.warmup_schedule = 140
        self.scheduler_cfg.scheduler_unit = "iter"
        self.scheduler_cfg.weight_decay = 0.1
        self.scheduler_cfg.total_schedule = None

        self.trainer_cfg.log_interval = 1
        # self.profiler_cfg.timing_log_level = 2

        self.checkpoint_cfg.load_safetensors = "/oss/weight/Step3.5-Flash-Midtrain/"
        self.checkpoint_cfg.load_option.none(but=["model"])
        self.checkpoint_cfg.save_safetensors = True
        self.checkpoint_cfg.save_dir = "/oss/checkpoints/"
        self.checkpoint_cfg.save_option.all()
        self.checkpoint_cfg.save_interval = 500

        self.model_cfg.recompute = True
        self.model_cfg.parallel_cfg.context_parallel_size = 8
        self.model_cfg.parallel_cfg.tensor_model_parallel_size = 1
        self.model_cfg.tp_cfg.sequence_parallel = False
        self.model_cfg.pipeline_activation_cpu_offload = False

        self.trainer_cfg.offload_optimizer_state = False
        self.checkpoint_cfg.async_dump = False

    def configure_optimizable(self):
        from steptronoss.utils.optimizable import set_optimization

        set_optimization(
            routed_grouped_ffn="fused",
            moe_weighted_gather="triton",
            TokenDispatcher="deep_ep",
            grouped_gemm="nv_grouped_gemm",
            # grouped_gemm="function_imple", # slower fallback
            AttentionCore="flash-attn-3",
        )


if __name__ == "__main__":
    Exp().train()
