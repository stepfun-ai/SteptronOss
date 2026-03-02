from __future__ import annotations

import copy
import os
from collections.abc import Callable, Iterable, Iterator
from functools import cached_property
from typing import (
    TYPE_CHECKING,
    Any,
    ForwardRef,
    Literal,
    NoReturn,
    Optional,
)

import torch
from configurize import Config, Ref, writable_property
from loguru import logger
from torch import Tensor

from steptronoss.exp.abstract import MetricConfig as AbstractMetricConfig
from steptronoss.exp.abstract import ModelConfig as AbstractModelConfig
from steptronoss.exp.abstract import ParallelConfig as AbstractParallelConfig
from steptronoss.exp.abstract import TokenizerConfig as AbstractTokenizerConfig
from steptronoss.exp.abstract import TrainerConfig as AbstractTrainerConfig
from steptronoss.exp.optimizer import AdamConfig, OptimizerConfig
from steptronoss.exp.resources import ResourceConfig

if TYPE_CHECKING:
    from steptronoss.core.pipeline_parallel.schedules import FWBWScheduler

TrainerHook = Callable[[ForwardRef("Trainer")], NoReturn]


def is_log_rank() -> bool:
    return int(os.getenv("RANK", "0")) == 0


class GradientManagerConfig(Config):
    optimizer_cfg: OptimizerConfig = AdamConfig

    params_dtype: torch.dtype = Ref("..model_cfg.params_dtype")

    use_distributed_optimizer: bool = True

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


class MetricConfig(AbstractMetricConfig):
    _allow_set_new_attr: bool = True

    def to_dict(self, rep: bool = False) -> dict[str, str]:
        return {k: repr(v) for k, v in self.items()}

    def register(self) -> None:
        """Register self to global metric holder."""
        from steptronoss.utils import GlobalMetrics

        GlobalMetrics.batch_register(self)

    def __init__(self) -> None:
        super().__init__()
        from steptronoss.utils.metrics import GradNormMetric, Metric

        self.consumed_tokens = Metric().sum("time").sum("dp")
        self.iteration_time = Metric().mean("time")
        self.learning_rate = Metric().mean("time")
        self.grad_norms = GradNormMetric()


class TrainerConfig(AbstractTrainerConfig):
    micro_batch_size: int = 1

    train_iters: int | None = None

    offload_optimizer_state: bool = False

    global_data_keys: list[str] | None = Ref("..data_cfg.global_data_keys", None)
    """When set, broadcast data[key] to all ranks (not only on data-source ranks)."""

    empty_unused_memory_level: int = 0

    log_detailed_grad_norms: bool = False

    log_num_zeros_in_grad: bool = True

    log_interval: int = 1

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
            if PM.i_am("TP", 0):
                if PM.i_am("CP", 0):
                    with get_timers().record("dataloader-next", log_level=1):
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

        return []

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

    tokenizer_path: str | None = None

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


class ParallelConfig(AbstractParallelConfig):
    parallel_definition: dict[str, str] = {
        "TP": "(p d t) -> (p d) t",
        "PP": "(p d t) -> (d t) p",
        "DP": "(p d t) -> (p t) d",
        "CP": "(p d c t) -> (p d t) c",
        "MP": "(p d t) -> d (p t)",
        "EP": "(p edp ep etp) -> (p edp etp) ep",
        "ETP": "(p edp ep etp) -> (p edp ep) etp",
        "EETP": "(p edp ep etp) -> (p edp) (ep etp)",
        "EDP": "(p edp ep etp) -> (p ep etp) edp",
        "EMP": "(p edp ep etp) -> edp (p ep etp)",
    }

    tensor_model_parallel_size: int = 1
    pipeline_model_parallel_size: int = 1
    expert_model_parallel_size: int = 1
    context_parallel_size: int = 1
    expert_tensor_parallel_size: int = Ref(".tensor_model_parallel_size")

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

    def sanity_check(self):
        super().sanity_check()
        world_size = int(os.getenv("WORLD_SIZE", "1"))
        attn_model_parallel_size = (
            self.pipeline_model_parallel_size * self.tensor_model_parallel_size * self.context_parallel_size
        )
        moe_model_parallel_size = (
            self.pipeline_model_parallel_size * self.expert_tensor_parallel_size * self.expert_model_parallel_size
        )
        assert world_size >= attn_model_parallel_size
        assert world_size % attn_model_parallel_size == 0
        assert world_size >= moe_model_parallel_size
        assert world_size % moe_model_parallel_size == 0


class MegatronTPConfig(AbstractModelConfig):
    params_dtype: torch.dtype = Ref("..params_dtype")

    sequence_parallel: bool = False

    gradient_accumulation_fusion: bool = False
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
    resource_cfg: ResourceConfig = ResourceConfig

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

    def sanity_check(self):
        expected_world_size = self.resource_cfg.replica * max(self.resource_cfg.gpu, 1)
        os.environ["WORLD_SIZE"] = os.environ.get(
            "WORLD_SIZE", str(expected_world_size)
        )  # for workspace check, fake the expected world_size
        super().sanity_check()
