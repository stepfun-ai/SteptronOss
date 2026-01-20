from __future__ import annotations

import copy
import os
from functools import cached_property
from typing import TYPE_CHECKING, Any, Callable, ForwardRef, Iterable, Iterator, Literal, NoReturn, Optional

import torch
import torch.distributed
from configurize import Config, Ref, writable_property
from loguru import logger
from torch import Tensor
from torch.nn import Module, Parameter

from steptronoss.exp.abstract import MetricConfig as AbstractMetricConfig
from steptronoss.exp.abstract import ModelConfig as AbstractModelConfig
from steptronoss.exp.abstract import OptimizerConfig as AbstractOptimizerConfig
from steptronoss.exp.abstract import ParallelConfig as AbstractParallelConfig
from steptronoss.exp.abstract import SchedulerConfig as AbstractSchedulerConfig
from steptronoss.exp.abstract import TokenizerConfig as AbstractTokenizerConfig
from steptronoss.exp.abstract import TrainerConfig as AbstractTrainerConfig

if TYPE_CHECKING:
    from steptronoss.core.pipeline_parallel.schedules import FWBWScheduler

TrainerHook = Callable[[ForwardRef("Trainer")], NoReturn]


def is_log_rank() -> bool:
    return int(os.getenv("RANK", "0")) == 0


class OptimizerConfig(AbstractOptimizerConfig):

    weight_decay: float = 0.01
    weight_decay_on_1d_params: bool = False

    lr: float = Ref("...scheduler_cfg.lr")

    weight_decay: float = Ref("...scheduler_cfg.weight_decay")

    # adam-specific hyperparams
    adam_eps: float = 1e-8
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95

    # muon-specific hyperparams
    # muon_matched_adamw_rms: float = 0.2
    # muon_momentum: float = 0.95
    # muon_nesterov: bool = True
    # muon_ns_steps: int = 5
    # muon_newtonschulz_fn: str = "polar_express"
    # muon_run_ns_in_fp32: bool = False
    # muon_run_ns_in_fp16: bool = True
    # muon_log_updates_grad_norms: bool = False
    # muon_batch_compute_mem_size: int = 128 * 1024 * 1024
    # muon_auto_applier_attn_pack_param_strategy: str = "split_by_type"
    # muon_auto_applier_glu_pack_param_strategy: str = "split_by_type"

    def scale_lr_func(self, name: str, param: Parameter) -> float:
        if hasattr(param, "_lr_scale"):
            return param._lr_scale
        return 1.0

    def scale_wd_cond(self, name: str, param: Parameter) -> float:
        if name.endswith(".bias") or (len(param.shape) == 1 and not self.weight_decay_on_1d_params):
            return 0.0
        return 1.0

    def build_optimizer(self, model: Module) -> torch.optim.Optimizer:
        # Base optimizer.
        from torch.optim.adam import Adam

        from steptronoss.optimizer.utils import advanced_get_param_groups
        from steptronoss.utils import convert_num

        param_groups = advanced_get_param_groups(
            model,
            scale_lr_cond=self.scale_lr_func,
            scale_wd_cond=self.scale_wd_cond,
        )

        for idx, group in enumerate(param_groups):
            extra = "; ".join([f"{k}: {v}" for k, v in group.items() if k != "params"])
            logger.info(
                f"Optim group {idx} -> # params: "
                f"{convert_num(sum([p.nelement() for p in group['params']]))}; "
                f"{extra}"
            )

        return Adam(
            param_groups,
            lr=self.lr,
            weight_decay=self.weight_decay,
            betas=(self.adam_beta1, self.adam_beta2),
            eps=self.adam_eps,
        )

    def sanity_check(self) -> None:
        super().sanity_check()
        # if self.optimizer == "muon":
        #     assert int(self.muon_run_ns_in_fp16) + int(self.muon_run_ns_in_fp32) <= 1

        #     if self.muon_log_updates_grad_norms:
        #         assert self.log_detailed_grad_norms

        #     assert self.muon_auto_applier_attn_pack_param_strategy in [
        #         "split_by_head",
        #         "split_by_type",
        #         "no_split",
        #     ]
        #     assert self.muon_auto_applier_glu_pack_param_strategy in [
        #         "split_by_type",
        #         "no_split",
        #     ]


class GradientManagerConfig(Config):
    optimizer_cfg = OptimizerConfig

    params_dtype: torch.dtype = Ref("..model_cfg.params_dtype")

    use_distributed_optimizer: bool = True
    optimizer_distribute_granularity: Literal["byte", "tensor"] = "tensor"
    """Under DistributedOptimizer (zero1), optimizer state can be sharded by bytes or by tensors.

    - raw: [Tensor(size=100), Tensor(size=59)]
    - Bytes  @dp2: [Tensor(size=80, partial)], [Tensor(size=20, partial), Tensor(size=59), Pad(1)]
    - Tensor @dp2: [Tensor(size=100)], [Tensor(size=59)]

    - byte distribution can leverage reduce_scatter optimization, with less comm. But
    does not support optimizer like muon (need grad for full tensor).

    """

    clip_grad: float = 1.0
    log_detailed_grad_norms: bool = Ref("..trainer_cfg.log_detailed_grad_norms")

    log_num_zeros_in_grad: bool = Ref("..trainer_cfg.log_num_zeros_in_grad")

    def build_gradient_manager(self, model: torch.nn.Module):
        from steptronoss.optimizer.base_gradient_manager import DummyOptimizer
        from steptronoss.optimizer.gradient_manager import AccInFP32GradientManager
        from steptronoss.optimizer.zero1_gradient_manager import Zero1GradientManager

        optimizer = self.optimizer_cfg.build_optimizer(model)

        if not optimizer.param_groups:
            return DummyOptimizer()

        if self.use_distributed_optimizer:
            return Zero1GradientManager(cfg=self, model=model, optimizer=optimizer)
        else:
            return AccInFP32GradientManager(cfg=self, model=model, optimizer=optimizer)


class SchedulerConfig(AbstractSchedulerConfig):

    lr: float = 1e-4
    """Initial learning rate. Depending on decay style and initial warmup,
    the learing rate at each iteration would be different."""

    min_lr: float = 0.0
    """Minumum value for learning rate. The scheduler clip values below this threshold."""

    weight_decay: float = 0.01
    """Weight decay coefficient for L2 regularization."""

    total_schedule: float = 1000
    """Scheduled count for lr scheduler"""

    warmup_schedule: float = 100
    """Scheduled count for warm up"""

    scheduler_unit: Literal["iter", "sample", "token"] = "iter"
    """How to update scheduler counter ('iter'|'sample'|'token')"""

    def sanity_check(self) -> None:
        assert self.min_lr <= self.lr
        assert self.scheduler_unit in ["iter", "sample", "token"]

    def build_scheduler(self, optimizer: Any, *args: Any) -> Any:
        from steptronoss.optimizer.hparam_scheduler import FuncConstant, Scheduler

        scheduler = Scheduler(
            optimizer=optimizer,
            base_lr=self.lr,
            base_wd=self.weight_decay,
            lr_func=FuncConstant(
                n=self.total_schedule,
                warmup=self.warmup_schedule,
                min_scale=self.min_lr / self.lr,
            ),
            wd_func=lambda x: 1.0,  # constant is ok
        )

        return scheduler


class MetricConfig(AbstractMetricConfig):
    _allow_set_new_attr: bool = True

    def to_dict(self, rep: bool = False) -> dict[str, str]:
        return {k: repr(v) for k, v in self.items()}

    def register(self) -> None:
        """Register self to global metric holder."""
        from steptronoss.utils import GlobalMetrics

        GlobalMetrics.batch_register(self)

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        from steptronoss.utils.metrics import GradNormMetric, Metric

        self.consumed_tokens = Metric().sum("time").sum("dp")
        self.iteration_time = Metric().mean("time")
        self.learning_rate = Metric().mean("time")
        self.grad_norms = GradNormMetric()


class TrainerConfig(AbstractTrainerConfig):
    train_iters: Optional[int] = None

    offload_optimizer_state: bool = False

    global_data_keys: Optional[list[str]] = Ref("..data_cfg.global_data_keys", None)
    """When set, broadcast data[key] to all ranks (not only on data-source ranks)."""

    empty_unused_memory_level: int = 0

    log_detailed_grad_norms: bool = False

    log_num_zeros_in_grad: bool = True

    log_interval: int = 10

    # writer_backend: available backends are "tensorboard", "wandb"
    # support multiple backends
    writer_backend: list[str] = ["tensorboard"]

    def make_logs(self, iteration: int, metrics: dict[str, Tensor], tb_writer: Any = None) -> dict[str, Any]:
        """Handle custom metrics and decide which goes tensorboard & which goes log file
        NOTE: tb_writer only available on RANK_LAST

        Args:
            iteration (int): Current iteration.
            metrics (dict[str, Tensor]): A dict of reduced metrics, metrics might need
                type cast.
            tb_writer (TensorBoardWriter, optional): TB writer (if available). Defaults to None.

        Returns:
            dict: A dict of SCALAR print to logfile/screen
        """
        from steptronoss.utils.metrics import GlobalMetrics, HistogramMetric

        if tb_writer is not None:
            scalars = {k: v.float() for k, v in metrics.items() if (torch.is_tensor(v) and v.numel() == 1)}
            scalars.update({k: v for k, v in metrics.items() if isinstance(v, (float, int))})

            vector_metrics = {k: v for k, v in metrics.items() if (torch.is_tensor(v) and v.numel() > 1)}
            for key in scalars:
                tb_writer.add_scalar(key, scalars[key], iteration)
            for key in vector_metrics:
                if isinstance(GlobalMetrics.metrics[key], HistogramMetric):
                    tb_writer.add_histogram(str(key), vector_metrics[key].cpu().float().flatten(), iteration)
            # NOTE: add text metric to tb logic
            text_metrics = {
                k: v for k, v in metrics.items() if isinstance(v, list) and len(v) > 0 and isinstance(v[0], str)
            }
            for key in text_metrics:
                for idx, text in enumerate(text_metrics[key]):
                    tb_writer.add_text(f"{key}_{idx}", text, iteration)

        return {}

    def is_data_source(self) -> bool:
        from steptronoss.core.parallel_state import PM

        # build dataloaders only on data-source nodes
        # (PP0||PPLast) && TP0 && CP0
        #
        # Note: `sync_get_data()` only calls `next(data_iterator)` on CP-src rank
        # (and then broadcasts within CP group). So we should only build
        # dataloaders on CP-src to avoid duplicated loader processes/threads.
        return (PM.i_am("PP", 0) or PM.i_am("PP", -1)) and (PM.i_am("TP", 0) and PM.i_am("CP", 0))

    def sync_get_data(self, data_iterator: Iterator[dict]) -> dict[str, Any]:
        # This function Get/Broadcast/Preprocess and return data ready-to-use
        from steptronoss.core.parallel_state import (
            PM,
            get_vpp_rank,
            get_vpp_size,
        )
        from steptronoss.timers import get_timers
        from steptronoss.utils import broadcast_tensors

        class NonOfHeadOrTail(dict):
            pass

        if PM.i_am("PP", 0) or PM.i_am("PP", -1):
            with get_timers().record("dataloader-next", log_level=2):
                if PM.i_am("TP", 0):
                    if PM.i_am("CP", 0):
                        data = next(data_iterator)
                    else:
                        data = None
                    with get_timers().record("broadcast-tensors-cp", log_level=2):
                        data = broadcast_tensors(
                            data,
                            src_rank=PM.ranks_of("CP")[0],
                            group=PM.group_of("CP"),
                            move_to_cuda=True,
                        )
                else:
                    data = None

            with get_timers().record("broadcast-tensors-tp", log_level=2):
                data = broadcast_tensors(
                    data,
                    src_rank=PM.ranks_of("TP")[0],
                    group=PM.group_of("TP"),
                )
        else:
            data = NonOfHeadOrTail()  # Mark these ranks for global_data broadcast

        if self.global_data_keys:
            vpp_rank = get_vpp_rank() or 0
            vpp_size = get_vpp_size() or 1
            with get_timers().record("broadcast-tensors-pp", log_level=2):
                if vpp_rank == 0:
                    pp_sync_data = [data.__class__] + [data.get(k, None) for k in self.global_data_keys]
                    pp_sync_data = broadcast_tensors(
                        pp_sync_data,
                        src_rank=PM.ranks_of("PP")[0],
                        group=PM.group_of("PP"),
                    )
                    # cache pp_sync_data for the following model chunks
                    if not hasattr(self, "_cached_pp_sync_data"):
                        self._cached_pp_sync_data = [[] for _ in range(vpp_size - 1)]
                    for _vp in range(vpp_size - 1):
                        self._cached_pp_sync_data[_vp].append(copy.deepcopy(pp_sync_data))
                else:
                    pp_sync_data = self._cached_pp_sync_data[vpp_rank - 1].pop(0)

            if isinstance(data, NonOfHeadOrTail):
                data_class = pp_sync_data.pop(0)
                data_items = dict(zip(self.global_data_keys, pp_sync_data))
                data = data_class(**data_items)
        if isinstance(data, NonOfHeadOrTail):
            data = dict()
        return data

    def build_after_init_hooks(self) -> list[TrainerHook]:
        """\
        Build hooks to run after steptron initialized (before building models).
        A hook is a `Callable[[Trainer], []]`
        """
        from steptronoss.utils import profile_allreduce

        def profile_nccl(trainer):
            # NOTE This profile is neccessary for batched_p2p_comm (not using overlap_p2p_comm)
            # since batch_isend_irecv with group require collective op applied before.
            profile_allreduce()

        def print_exp(trainer):
            if is_log_rank():
                logger.info(self.root())

        hooks: list[TrainerHook] = [
            profile_nccl,
            print_exp,
        ]

        return hooks

    def build_before_train_hooks(self) -> list[TrainerHook]:
        """\
        Build hooks to run after training.
        A hook is a `Callable[[Trainer], []]`
        """

        def check_uninitialized_model_weight(trainer):
            from steptronoss.core.parallel_state import PM
            from steptronoss.utils.utils import unwrap_model

            if hasattr(trainer, "models") and PM.i_am("DP", 0) and PM.i_am("TP", 0):
                uninitialized_keys = []
                for model in trainer.models:
                    model = unwrap_model(model)
                    for name, param in model.named_parameters():
                        if not getattr(param, "has_initialized", False):
                            uninitialized_keys.append(name)
                if uninitialized_keys:
                    logger.warning(
                        f"The following parameters are not marked as initialized. "
                        f"Please add 'init_model_weight' method in the corresponding module or its parent module "
                        f"and mark parameters with 'has_initialized=True' after initialization:  {uninitialized_keys}"
                    )

        return [
            check_uninitialized_model_weight,
        ]

    def build_before_step_hooks(self) -> list[TrainerHook]:
        """\
        Build hooks to run before each step.
        A hook is a `Callable[[Trainer], []]`
        """
        return []

    def build_after_step_hooks(self) -> list[TrainerHook]:
        """\
        Build hooks to run after each step.
        A hook is a `Callable[[Trainer], []]`
        """
        return []

    def build_after_train_hooks(self) -> list[TrainerHook]:
        """\
        Build hooks to run after training.
        A hook is a `Callable[[Trainer], []]`
        """
        return []

    def get_trainer_cls(self) -> type:
        raise NotImplementedError


class ProfilerConfig(Config):
    timing_log_level = 0
    timing_use_event = True

    log_path = Ref("..log_path", "./")


class TokenizerConfig(AbstractTokenizerConfig):
    vocab_size: int = 65536

    make_vocab_size_divisible_by: int = 128

    tokenizer_path: Optional[str] = None

    tensor_model_parallel_size: int = Ref("..model_cfg.tensor_model_parallel_size", 1)

    @writable_property
    def padded_vocab_size(self) -> int:
        from steptronoss.utils.general import make_divisible

        return make_divisible(
            self.vocab_size,
            self.make_vocab_size_divisible_by * self.tensor_model_parallel_size,
        )

    def build_tokenizer(self) -> Any:
        pass


class SaveOptions(Config):
    """Option Config, use True for load

    options.none(but=['model']) for load model only

    options.all(but=['optimizer']) for NOT load optimizer only.
    """

    model: bool = True
    optimizer: bool = True
    scheduler: bool = True
    data: bool = True
    rng_state: bool = True

    def all(self, but: list[str] = []) -> "SaveOptions":
        for k, _ in self.items():
            setattr(self, k, k not in but)
        return self

    def none(self, but: list[str] = []) -> "SaveOptions":
        for k, _ in self.items():
            setattr(self, k, k in but)
        return self


class LoadOptions(SaveOptions):
    model: bool = True
    optimizer: bool = True
    scheduler: bool = True
    data: bool = True
    exp: bool = True
    iter: bool = True
    rng_state: bool = True


class CheckpointConfig(Config):
    auto_resume: bool = True

    load_path: Optional[str] = None

    save_dir: str = "./"

    save_interval: int = 100

    async_dump: bool = True

    load_option: type[LoadOptions] = LoadOptions
    load_safetensors: bool | str = False
    """Load weight from safetensors. If True, load 'load_path/hf'; if str, load the value."""

    save_option: type[SaveOptions] = SaveOptions
    save_safetensors: bool = False
    """If set, dump an extra hf weight to 'save_path/hf'"""

    broadcast_from_dp0: bool = True

    strict_load_model: bool = True

    use_distributed_optimizer: bool = Ref("..optimizer_cfg.use_distributed_optimizer", False)

    exp_name: str = Ref("..exp_name", "")

    # Enable online reshard of distributed optimizer state when DP size changes.
    reshard_optimizer_state: bool = False
    reshard_optimizer_strict: bool = True

    # for save_safetensors reference
    model_config_path: Optional[str] = None

    tokenizer_path: Optional[str] = None

    @writable_property
    def save_path(self) -> str:
        return os.path.join(self.save_dir, self.exp_name)

    def sanity_check(self) -> None:
        super().sanity_check()
        assert not (
            self.load_safetensors and self.load_path
        ), "load_safetensors and load_path cannot be set at the same time"


class MoEConfig(Config):
    use_moe: bool = False
    moe_every_n_layer: int = 1
    moe_num_experts: int = 1
    moe_top_k: int = 1
    moe_aux_loss_coef: float = 1e-2
    moe_hidden_size: int = 12288
    norm_expert_weight: bool = True

    use_ep_group_wise_aux_loss: bool = False
    """Enable expert group-wise auxiliary loss"""

    moe_style: Literal["mixtral", "marco", "old"] = "marco"
    """moe imple style, one of ['mixtral', 'marco', 'old']"""

    moe_layer_list: Optional[list[int]] = None
    """layer ids of layers which use moe"""

    share_expert_dim: int = 0
    """Dimension of shared moe ffn"""

    moe_enable_group_gemm: bool = False
    use_groupgemm_bwd: bool = True

    use_fp32_router_for_moe: bool = True

    # deepep settings for expert parallel acceleration
    moe_enable_deepep: bool = False
    """Enable DeepEP acceleration for expert parallel when EP>1 and ETP=1"""
    moe_deepep_num_sms: int = 0
    """Number of SMs to use for DeepEP kernels, 0 means auto-detect"""
    moe_permute_fusion: bool = False
    """Enable permutation fusion in DeepEP"""

    # ===========================================================================================
    # deepseekv3 new features
    enable_auxiliary_loss_free_load_balance: bool = False  # enable auxiliary loss free load balance
    enable_sigmoid_router: bool = False  # enable sigmoid router
    enable_scaling_factor: bool = False  # enable scaling factor in moe
    router_bias_update_rate: float = 1e-4  # update rate in auxiliary loss free load balance
    routed_scaling_factor: float = 1.0  # scaling factor for moe
    # ===========================================================================================

    distribute_saved_activations: bool = Ref("..distribute_saved_activations", False)

    data_parallel_size: int = Ref("..data_parallel_size")
    global_batch_size: int = Ref("...trainer_cfg.global_batch_size")
    micro_batch_size: int = Ref("...trainer_cfg.micro_batch_size")

    # for debug only
    enable_force_balance: bool = False

    def get_aux_loss_calib_scale(self) -> float:
        """aux loss is local, and cannot percept grad_acc_steps, let's scale the coef instead."""
        from steptronoss.core.parallel_state import PM

        if self.moe_aux_loss_coef != 0 and PM.size_of("CP") > 1:
            assert self.use_ep_group_wise_aux_loss, (
                "For CP size > 1, because only EP-group-wise MoE aux loss is validated, "
                "so currently we only support EP-group-wise MoE aux loss, "
                "(set use_ep_group_wise_aux_loss=True or disable moe_aux_loss_coef)."
            )
        return (self.micro_batch_size * self.data_parallel_size) / self.global_batch_size


class ParallelConfig(AbstractParallelConfig):
    parallel_definition: dict[str, str] = {
        "TP": "(p d t) -> (p d) t",
        "PP": "(p d t) -> (d t) p",
        "DP": "(p d t) -> (p t) d",
        "CP": "(p d c t) -> (p d t) c",
        "MP": "(p d t) -> d (p t)",
        "EP": "(p edp c ep etp) -> (p c edp etp) ep",
        "ETP": "(p edp c ep etp) -> (p c edp ep) etp",
        "EDP": "(p edp c ep etp) -> (p c ep etp) edp",
        "EMP": "(p edp c ep etp) -> edp (p c ep etp)",
    }

    tensor_model_parallel_size: int = 1
    pipeline_model_parallel_size: int = 1
    expert_model_parallel_size: int = 1
    context_parallel_size: int = 1
    expert_tensor_parallel_size: int = 1

    # Not a real parallel
    virtual_pipeline_model_parallel_size: int = 1

    def build_parallel(self) -> dict[str, list[list[int]]]:
        from steptronoss.core.parallel_state import PM

        args = {
            "p": self.pipeline_model_parallel_size,
            "c": self.context_parallel_size,
            "ep": self.expert_model_parallel_size,
            "t": self.tensor_model_parallel_size,
            "etp": self.expert_tensor_parallel_size,
        }

        parallel_groups = {k: PM.define_parallel(v, **args) for k, v in self.parallel_definition.items()}

        return parallel_groups


class MegatronTPConfig(AbstractModelConfig):
    params_dtype: torch.dtype = Ref("..params_dtype")

    sequence_parallel: bool = False

    gradient_accumulation_fusion: bool = True
    async_tensor_model_parallel_allreduce: bool = True

    distribute_saved_activations: bool = False

    def get_tp_kwargs(self) -> dict[str, Any]:
        return {
            "params_dtype": self.params_dtype,
            "gradient_accumulation_fusion": self.gradient_accumulation_fusion,
            "sequence_parallel_enabled": self.sequence_parallel,
        }


class MegatronPPModelConfig(AbstractModelConfig):
    params_dtype: torch.dtype = torch.bfloat16

    overlap_p2p_comm: bool = True

    # Used for P2P comm pre-allocation
    variable_seq_lengths: bool = False  # disable for pretrain
    pp_comm_shape: tuple[int] = None

    check_nan: bool = True
    """check nan on every rank"""

    gradient_accumulation_fusion: bool = True
    fp32_residual_connection: bool = False

    def get_pp_scheduler(self) -> FWBWScheduler:
        from steptronoss.core.parallel_state import PM, get_vpp_size
        from steptronoss.core.pipeline_parallel.schedules import (
            FWBWScheduler,
            PPScheduler,
            VPPScheduler,
        )

        if PM.size_of("PP") > 1:
            if get_vpp_size() > 1:
                return VPPScheduler(config=self)
            else:
                return PPScheduler(config=self)
        else:
            return FWBWScheduler(config=self)

    def sanity_check(self):
        super().sanity_check()
        if not self.variable_seq_lengths:
            assert self.pp_comm_shape is not None


class Megatron3DParallelModelConfig(MegatronPPModelConfig):
    tp_cfg: MegatronTPConfig = MegatronTPConfig
    parallel_cfg = ParallelConfig

    global_seq_length = Ref("..trainer_cfg.global_seq_length")
    micro_batch_size = Ref("..trainer_cfg.micro_batch_size")
    hidden_size: int

    @writable_property
    def pp_comm_shape(self):
        seq_length = self.global_seq_length // self.parallel_cfg.context_parallel_size
        return (seq_length, self.micro_batch_size, self.hidden_size)


class RoPEConfig(Config):

    rope_type: str = "llama3"
    factor: float = Ref("..ntk_interp_ratio", 1.0)
    original_max_position_embeddings: int = Ref("..max_position_embeddings", None)
    low_freq_factor: float = Ref("..yarn_beta_slow", None)
    high_freq_factor: float = Ref("..yarn_beta_fast", None)

    def sanity_check(self):
        super().sanity_check()
        assert self.rope_type in ["llama3"]
        if self.factor != 1:
            assert isinstance(self.original_max_position_embeddings, int)
            assert isinstance(self.low_freq_factor, float)
            assert isinstance(self.high_freq_factor, float)


class ModelConfig(Megatron3DParallelModelConfig):
    _allow_search = True
    critical_keys = ["hidden_size", "ffn_hidden_size", "rope_theta"]

    vocab_size: int = Ref("..tokenizer_cfg.padded_vocab_size", 65536)
    actual_vocab_size: int = Ref("..tokenizer_cfg.vocab_size", None)

    hidden_size: int = 12288
    ffn_hidden_size: int = 31232
    num_layers: int = 12
    num_attention_heads: int = 96
    num_sliding_attention_heads: int = None  # 如果为 None 则使用 num_attention_heads

    use_headwise_attn_gate: bool = False
    sliding_window_size: int = -1
    layer_types: list[str] = None

    @writable_property
    def head_dim(self):
        if self.mfa_kv_channels is not None:
            return self.mfa_kv_channels
        return self.hidden_size // self.num_attention_heads

    qk_rope_head_dim: int | list[int] = None
    """head_dim of rope, if not set, use head_dim; if list[int], different for each layer."""

    num_attention_groups: int = 8
    attention_type: str = "gqa"
    """Attention Type: one of ['gqa', 'mfa']"""
    mfa_q_channels: int = Ref(".mfa_kv_channels")
    mfa_kv_channels: int = None
    mfa_use_inter_norm = False
    rope_theta: float | list[float] = 500_000.0
    """theta in RoPE, if list, different for each layer."""

    rms_norm_zero_gamma: bool = False
    layernorm_epsilon: float = 1e-05
    attention_dropout: float = 0.0

    moe_cfg = MoEConfig

    use_vpp_v2 = False
    gather_output = False

    # Simple Optimization
    recompute_granularity: str = None
    recompute_num_layers: int = 0
    recompute_pre_mlp_layernorm: bool = False
    recompute_attention_layernorm: bool = False
    recompute_qknorm_rope: bool = False  # 添加这一行
    recompute_cp_kv: bool = False
    recompute_logits: bool = False
    continuous_memory_ffn: bool = False
    use_fused_qknorm_and_rope: bool | list[bool] = False

    detect_abnormal_data: bool = False

    # precisions
    fp32_lm_head_out: bool = False
    fp32_residual_connection: bool = False
    fp32_rms_norm: bool = True
    use_optimus_rope: bool = False

    # rope
    rope_cfg = RoPEConfig

    yarn_beta_fast: float = 32.0
    yarn_beta_slow: float = 1.0
    ntk_interp_ratio: float = 1.0
    max_position_embeddings: int = None

    disable_qk_norm: bool = False
    use_qkv_bias: bool = False

    # WARNING: FOR EXPERT ONLY

    swiglu_recompute_silu_out_proj: bool = True

    # for clip sliu

    use_swiglu_limit: float | list[float] = None
    use_swiglu_limit_shared: float | list[float] = None

    # For muon
    muon_auto_applier_attn_pack_param_strategy = Ref("..optimizer_cfg.muon_auto_applier_attn_pack_param_strategy")
    muon_auto_applier_glu_pack_param_strategy = Ref("..optimizer_cfg.muon_auto_applier_glu_pack_param_strategy")

    def build_model(self):
        raise NotImplementedError

    def sanity_check(self):
        assert self.attention_type in ["gqa", "mfa"]
        if self.layer_types:
            assert len(self.layer_types) == self.num_layers
        if self.moe_cfg.use_moe:
            assert self.moe_cfg.moe_num_experts % self.parallel_cfg.expert_model_parallel_size == 0
            assert self.moe_cfg.moe_top_k <= self.moe_cfg.moe_num_experts
            if self.parallel_cfg.expert_model_parallel_size > 1 and self.moe_cfg.moe_enable_group_gemm:
                assert self.moe_cfg.moe_enable_deepep, (
                    "When expert_model_parallel_size > 1, grouped gemm(moe_enable_group_gemm=True) "
                    "requires DeepEP, set moe_enable_deepep to True"
                )
            if self.moe_cfg.moe_enable_deepep:
                assert self.parallel_cfg.expert_model_parallel_size > 1, (
                    "DeepEP requires expert model parallel size > 1, set expert_model_parallel_size "
                    "larger than 1 or disable DeepEP by setting moe_enable_deepep to False"
                )
        else:
            assert self.parallel_cfg.expert_model_parallel_size == 1

        if self.fp32_residual_connection:
            assert self.params_dtype in [
                torch.float16,
                torch.bfloat16,
            ], "residual connection in fp32 only supported when using fp16 or bf16."  # noqa

        if self.distribute_saved_activations:
            assert self.parallel_cfg.tensor_model_parallel_size > 1, (
                "can distribute " "recomputed activations only across tensor model " "parallel groups"
            )
            assert self.recompute_granularity == "full", (
                "distributed recompute activations is only " "application to full recompute granularity"
            )
        if self.parallel_cfg.tensor_model_parallel_size == 1:
            assert self.sequence_parallel == False

        if self.sequence_parallel:
            assert self.async_tensor_model_parallel_allreduce == False

        if os.getenv("CUDA_DEVICE_MAX_CONNECTIONS", None) != "1":
            assert not self.sequence_parallel, (
                "Using sequence parallelism requires setting the environment variable "
                "CUDA_DEVICE_MAX_CONNECTIONS to 1"
            )
            assert not self.async_tensor_model_parallel_allreduce, (
                "Using async gradient all reduce requires setting the environment "
                "variable CUDA_DEVICE_MAX_CONNECTIONS to 1"
            )
        assert isinstance(self.rope_theta, (float, list)), "rope_theta must be float!"
        super().sanity_check()


class DataConfig(Config):
    global_data_keys: list[str] = None
    """When set, broadcast data[key] to all ranks (not only head & tail).
    The broadcast happens BEFORE preprocess.
    """

    def build_dataloader(self, dp_rank=0, dp_size=1) -> Iterable[dict]:
        """Build a Nextable that returns a dict when call next(dataloader)."""
        raise NotImplementedError

    def preprocess(self, batch: dict) -> dict:
        """Process the dict returned by next(dataloader)"""
        from steptronoss.utils.general import recur_to

        return recur_to(batch, "cuda")


class BaseExp(Config):

    seed = 1234

    log_dir = "./"
    suffix: str = ""

    project_name = None  # used for wandb parsing, 'entity/project_name:tag'

    @cached_property
    def file_path(self):
        import inspect

        return inspect.getabsfile(self.__class__)

    @writable_property
    def exp_name(self):
        base_name = os.path.basename(self.file_path).split(".")[0]
        return f"{base_name}/{self.suffix}".rstrip("/")

    @property
    def log_path(self):
        return os.path.join(
            self.log_dir,
            self.exp_name,
        )

    def update_from_args(self):
        from steptronoss.utils.arguments import parse_args
        from steptronoss.utils.logger import setup_logger

        # apply and log diffs from arguments
        args = parse_args()
        diff = self.merge(args, exists_only=True)
        diff = "\n".join([f"{k}: {s} -> {t}" for k, s, t in diff])
        try:
            setup_logger(self.log_path)
        except:
            pass
        if diff:
            logger.info(f"Modified by Args:\n{diff}", at=0)

    def build_log_writer(self):
        import inspect
        import sys

        from steptronoss.utils import StepWriter

        writer = StepWriter(
            log_dir=self.log_path,
            project_name=self.project_name,
            exp_name=self.exp_name,
            backend=["tensorboard"],
            exp=self,
        )
        my_doc = inspect.getdoc(sys.modules[self.__class__.__module__])
        writer.add_text("Note", my_doc or "No-Doc", global_step=0)
        return writer
