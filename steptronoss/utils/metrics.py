import contextlib
import os
from collections import defaultdict, deque
from typing import Callable

import numpy as np
import torch
import torch.distributed as dist
from loguru import logger

from steptronoss.core.parallel_state import PM
from steptronoss.exp.base_exp import MetricConfig
from steptronoss.utils.dist_utils import all_gather_object

__all__ = ["ROps", "RDims", "GlobalMetrics", "Metric", "AvgMetric", "PercentageMetric"]


class ROps:
    sum = "sum"
    mean = "mean"
    max = "max"
    min = "min"
    gather = "gather"

    @classmethod
    def _get_dist_op(cls, op):
        if op == cls.sum:
            return dist.ReduceOp.SUM
        elif op == cls.max:
            return dist.ReduceOp.MAX
        elif op == cls.min:
            return dist.ReduceOp.MIN
        elif op == cls.mean:
            return dist.ReduceOp.SUM
        elif isinstance(op, Callable):
            return op
        else:
            raise Exception(f"Unsupported reduce op: {op}")


class RDims:
    time = "time"
    tp = "tp"
    pp = "pp"
    dp = "dp"
    ep = "ep"
    mp = "mp"
    cp = "cp"
    world = "world"

    @classmethod
    def _get_group(cls, dim):
        from steptronoss.core.parallel_state import PM

        if dim == cls.world:
            return None  # None for world
        elif dim == cls.tp:
            return PM.group_of("TP")
        elif dim == cls.pp:
            return PM.group_of("PP")
        elif dim == cls.dp:
            return PM.group_of("DP")
        elif dim == cls.ep:
            return PM.group_of("EP")
        elif dim == cls.cp:
            return PM.group_of("CP")
        elif dim == cls.mp:
            return PM.group_of("MP")
        elif isinstance(dim, dist.ProcessGroup):
            return dim
        else:
            raise Exception(f"Unsupported reduce dim: {dim}")


class DummyMetric:
    enabled = False

    def add(self, *args, **kwargs):
        del args
        del kwargs


dummy_metric = DummyMetric()


class BaseMetric:
    DEFAULT_REDUCTIONS = []
    enabled = True

    def __init__(self, name=None, reductions=None, sample_interval=1, is_group=False):
        self.name = name
        if reductions is None:
            reductions = self.DEFAULT_REDUCTIONS.copy()
        self.reductions = reductions
        self.sample_interval = sample_interval
        self.is_group = is_group

        self._reset()

    def _reset(self):
        self.data_counters = defaultdict(int) if self.is_group else 0
        self.data = defaultdict(list) if self.is_group else []

    def __repr__(self):
        reductions = ".".join(f"{x[0]}({x[1]})" if isinstance(x, tuple) else repr(x) for x in self.reductions)
        rep = f"{self.__class__.__name__}({self.name}).{reductions}"
        if not self.enabled:
            rep = "🚫 " + rep
        return rep

    def add(self, value: torch.Tensor, subname=None, iop=lambda x: x):
        pass

    def reduce(self, reset=True) -> torch.Tensor | dict:
        pass

    def _reduce_list(self, data: list[torch.Tensor] | torch.Tensor):
        pass

    def enable(self):
        self.enabled = True
        return self

    def disable(self):
        self.enabled = False
        return self

    def get_data(self):
        data, counter = self.data, self.data_counters
        self._reset()
        return (data, counter)

    def set_data(self, data):
        self.data, self.data_counters = data


class Metric(BaseMetric):
    DEFAULT_REDUCTIONS = []

    def __init__(self, name=None, reductions=None, sample_interval=1, is_group=False) -> None:
        self.name = name
        if reductions is None:
            # logger.warning(f"Registered metric {name} with default reductions.")
            reductions = self.DEFAULT_REDUCTIONS.copy()
        self.reductions = reductions
        self.sample_interval = sample_interval
        self.is_group = is_group

        self._reset()

    def add(self, value: torch.Tensor, subname=None, iop=lambda x: x):
        if subname is not None:
            assert self.is_group
            self.data_counters[subname] += 1
            if self.data_counters[subname] % self.sample_interval == 0:
                value = iop(value)
                if not torch.is_tensor(value):
                    value = torch.tensor(value, device="cpu")
                value = value.cpu().float()
                self.data[subname].append(value)
        else:
            self.data_counters += 1
            if self.data_counters % self.sample_interval == 0:
                value = iop(value)
                if not torch.is_tensor(value):
                    value = torch.tensor(value, device="cpu")
                if value.requires_grad:
                    logger.warning(
                        f"Adding Metric [{self.name}] with un-detached tensor, fix your code to avoid mem-leak!"
                    )
                    value = value.detach()
                value = value.cpu().float()
                self.data.append(value)

    def sum(self, dim):
        self.reductions.append((ROps.sum, dim))
        return self

    def mean(self, dim):
        self.reductions.append((ROps.mean, dim))
        return self

    def max(self, dim):
        self.reductions.append((ROps.max, dim))
        return self

    def min(self, dim):
        self.reductions.append((ROps.min, dim))
        return self

    def gather(self, dim):
        self.reductions.append((ROps.gather, dim))
        return self

    def call(self, func):
        self.reductions.append(func)
        return self

    def _reduce_list(self, data: list[torch.Tensor] | torch.Tensor):
        if len(data) == 0:
            data = torch.tensor(0.0, device=torch.cuda.current_device())
        else:
            try:
                data = torch.stack(data)
            except:
                pass
        data = data.cuda()

        for op_dim in self.reductions:
            if isinstance(op_dim, Callable):
                data = op_dim(data)
                continue
            op, dim = op_dim
            if op == ROps.gather:
                assert dim != RDims.time
                group = RDims._get_group(dim)
                gathered_data = [None] * dist.get_world_size(group)
                dist.all_gather_object(gathered_data, data, group=group)
                data = [i.to(data.device) for i in gathered_data]
                try:
                    stacked_data = torch.stack(data)
                    data = stacked_data
                except:
                    pass
            elif dim == RDims.time:
                if op == ROps.mean:
                    data = data.nanmean(0)
                else:
                    data = getattr(data, op)(dim=0)
                    if op in [ROps.max, ROps.min]:
                        data = data.values
            else:
                group = RDims._get_group(dim)
                dist_op = ROps._get_dist_op(op)
                if op == ROps.mean:
                    counter = torch.tensor(data.nelement(), device=torch.cuda.current_device())
                    dist.all_reduce(counter, op=dist.ReduceOp.SUM, group=group)
                    data /= counter
                dist.all_reduce(data, op=dist_op, group=group)
        return data

    def reduce(self, reset=True) -> torch.Tensor | dict:
        if self.is_group:
            output = dict()
            global_keys = [None] * dist.get_world_size()
            dist.all_gather_object(global_keys, list(self.data.keys()))
            global_keys = list(set(sum(global_keys, [])))
            global_keys.sort()
            for key in global_keys:
                output[key] = self._reduce_list(self.data[key])
        else:
            output = self._reduce_list(self.data)
        if reset:
            self._reset()
        return output

    def get_data(self):
        data, counter = self.data, self.data_counters
        self._reset()
        return (data, counter)

    def set_data(self, data_counter):
        self.data, self.data_counters = data_counter


class AvgMetric(BaseMetric):
    """This Metric reduce get

    OP(gather(data, dim=dim))
    """

    DEFAULT_REDUCTIONS = []

    def __init__(self, name=None, reductions=None) -> None:
        self.name = name
        if reductions is None:
            reductions = self.DEFAULT_REDUCTIONS.copy()
        self.reductions = reductions

        self._reset()

    def _reset(self):
        self.data = []

    def mean(self, dim):
        self.reductions = ("mean", dim)
        return self

    def max(self, dim):
        self.reductions = ("max", dim)
        return self

    def min(self, dim):
        self.reductions = ("min", dim)
        return self

    def std(self, dim):
        self.reductions = ("std", dim)
        return self

    def add(self, value: float | int):
        assert isinstance(value, (float, int)), "AvgMetric only supports float or int values"
        self.data.append(value)

    def _reduce_list(self, data: list[torch.Tensor] | torch.Tensor):
        op, dim = self.reductions
        assert op in [
            "mean",
            "max",
            "min",
            "std",
        ], "AvgMetric only supports mean, max, min, std reduction"
        assert dim in [
            RDims.world,
            RDims.dp,
        ], "AvgMetric only supports dp or world dimension"

        group = RDims._get_group(dim)
        gathered_data = all_gather_object(data, group=group)
        data = torch.tensor(sum(gathered_data, []), dtype=torch.float)
        if op == "mean":
            data = torch.mean(data)
        elif op == "max":
            data = torch.max(data)
        elif op == "min":
            data = torch.min(data)
        elif op == "std":
            data = torch.std(data)
        return data

    def reduce(self, reset=False) -> torch.Tensor | dict:
        output = self._reduce_list(self.data)
        if reset:
            self._reset()
        return output


class PercentageMetric(BaseMetric):
    """This Metric reduce get:

    {
        subname1: sum(subname1) / sum(subname1 + subname2),
        subname2: sum(subname2) / sum(subname1 + subname2)
    }
    """

    def __init__(self, name=None) -> None:
        super().__init__(name=name)

        self._reset()

    def _reset(self):
        self.data = defaultdict(lambda: defaultdict(float))

    def sum(self, dim):
        self.reductions = ("sum", dim)
        return self

    def add(self, value: float, subname, iop=lambda x: x):
        value = float(iop(value))
        metric_name = subname.split("/")[0]
        subclass_name = "/".join(subname.split("/")[1:])
        self.data[subclass_name][metric_name] += value

    def _reduce_list(self, data: list[torch.Tensor] | torch.Tensor):
        op, dim = self.reductions
        assert op in ["sum"], "AvgMetric only supports sum reduction"
        assert dim in [
            RDims.world,
            RDims.dp,
        ], "PercentageMetric only supports dp or world dimension"

        group = RDims._get_group(dim)

        # convert defaultdict to dict for gather
        data = {k: {kk: vv for kk, vv in v.items()} if isinstance(v, defaultdict) else v for k, v in data.items()}
        gathered_data = all_gather_object(data, group=group)
        sumed = defaultdict(lambda: defaultdict(float))
        for d in gathered_data:
            for subclass_name, subclass_data in d.items():
                for metric_name, v in subclass_data.items():
                    sumed[subclass_name][metric_name] += v

        v_all = defaultdict(float)
        for subclass_name, subclass_data in sumed.items():
            v_all[subclass_name] = max(sum(subclass_data.values()), 1e-5)

        results = dict()
        for subclass_name, subclass_data in sumed.items():
            for metric_name, v in subclass_data.items():
                if subclass_name == "":
                    # for metric name without subclass, subclass_name is empty string
                    results[metric_name] = torch.tensor(v / v_all[subclass_name], dtype=torch.float)
                else:
                    results[f"{metric_name}/{subclass_name}"] = torch.tensor(
                        v / v_all[subclass_name], dtype=torch.float
                    )

        return results

    def reduce(self, reset=False) -> torch.Tensor | dict:
        output = self._reduce_list(self.data)
        if reset:
            self._reset()
        return output


class HistogramMetric(BaseMetric):
    def __init__(self, name=None, reductions=None, sample_interval=1, is_group=False) -> None:
        super().__init__(
            name=name,
            reductions=reductions,
            sample_interval=sample_interval,
            is_group=is_group,
        )

    def add(self, value: torch.Tensor, iop=lambda x: x):
        self.data_counters += 1
        if self.data_counters % self.sample_interval == 0:
            value = iop(value)
            if not torch.is_tensor(value):
                value = torch.tensor(value, device="cpu")
            if value.requires_grad:
                logger.warning(f"Adding Metric [{self.name}] with un-detached tensor, fix your code to avoid mem-leak!")
                value = value.detach()
            value = value.cpu()
            self.data.append(value)

    def gather(self, dim):
        self.reductions.append((ROps.gather, dim))
        return self

    def _reduce_list(self, data: list[torch.Tensor] | torch.Tensor):
        if len(data) == 0:
            data = torch.tensor(0.0, device=torch.cuda.current_device())
        else:
            try:
                data = torch.stack(data)
            except:
                pass
        data = data.cuda()

        for op_dim in self.reductions:
            op, dim = op_dim
            assert op == ROps.gather
            assert dim != RDims.time
            group = RDims._get_group(dim)
            gathered_data = [None] * dist.get_world_size(group)
            dist.all_gather_object(gathered_data, data, group=group)
            data = [i.to(data.device) for i in gathered_data]
            try:
                stacked_data = torch.cat(data, dim=-1)
                data = stacked_data
            except:
                pass
        return data

    def reduce(self, reset=False) -> torch.Tensor | dict:
        output = self._reduce_list(self.data)
        if reset:
            self._reset()
        return output


class GradNormMetric(BaseMetric):
    """This metric collect grads and reduce data by grad=sum(x**2)**0.5

    NOTE: difference from other metric, this require calling set_group_and_reject_more(group)
    to specify reduce group during runtime.
    """

    def __init__(self, name=None, sample_interval=1):
        self.name = name
        self.reductions = []
        self.sample_interval = sample_interval
        self.is_group = True

        self.group = None

        self._reset()

    def _reset(self):
        self.data_counters = defaultdict(int)
        self.data = defaultdict(list)
        self.rejecting = False

    def set_group_and_reject_more(self, group):
        """When finish collection for 1 step, rejecting other steps until reset"""
        self.group = group
        self.rejecting = True

    def add(self, value: torch.Tensor, subname, iop=lambda x: x):
        if self.rejecting:
            return
        assert self.is_group
        self.data_counters[subname] += 1
        if self.data_counters[subname] % self.sample_interval == 0:
            value = iop(value)
            value = (value**2).sum().cpu()
            self.data[subname].append(value)

    def reduce(self, reset=True) -> dict[str, torch.FloatTensor]:
        output = dict()
        global_keys = list(set(sum(all_gather_object(list(self.data)), [])))
        if global_keys:
            global_keys.sort()
            data = [self.data[x] for x in global_keys]
            data = [
                sum(
                    [i.cuda() for i in x],
                    start=torch.tensor(0.0, device=torch.cuda.current_device()),
                )
                for x in data
            ]
            data = torch.stack(data)

            dist.all_reduce(data, group=self.group)
            for key, d in zip(global_keys, data):
                output[key] = d**0.5

            if reset:
                self._reset()
        return output

    def get_data(self):
        data, counter, rj = self.data, self.data_counters, self.rejecting
        self._reset()
        return (data, counter, rj)

    def set_data(self, data):
        self.data, self.data_counters, self.rejecting = data


class TextMetric(BaseMetric):
    """
    This metric is used to logging text.
    reduce get:
    [text1, text2, ...],
    Args:
        max_record_texts: The maximum number of samples to save.
        is_group: Whether the metric is a group metric. Group metric is used in dicts type of data.
    """

    def __init__(
        self,
        name=None,
        reductions=None,
        sample_interval=1,
        is_group=False,
        max_record_texts=1000,
    ) -> None:
        super().__init__(
            name=name,
            reductions=reductions,
            sample_interval=sample_interval,
            is_group=is_group,
        )
        self.max_record_texts = max_record_texts

    def _reset(self):
        self.data_counters: int = 0
        self.data: list[str] = []

    def add(self, value: str, iop=lambda x: x):
        if self.data_counters % self.sample_interval == 0:
            value = iop(value)
            self.data.append(value)
            self.data_counters += 1

    def gather(self, dim):
        assert dim in [
            RDims.world,
            RDims.dp,
        ], "TextMetric only supports dp or world dimension"
        self.reductions = (ROps.gather, dim)
        return self

    def _reduce_list(self, data: list[str]):
        op, dim = self.reductions
        assert op == ROps.gather, "TextMetric only supports gather reduction"
        assert dim in [
            RDims.world,
            RDims.dp,
        ], "TextMetric only supports dp or world dimension"

        # Get the group
        group = RDims._get_group(dim)
        # Per rank save size according to
        sample_per_rank = len(data)
        global_samples = all_gather_object(sample_per_rank, group=group)
        per_rank_size = int(self.max_record_texts / max(sum(global_samples), 1) * len(data)) + 1
        # Calculate the number of data to save for each rank
        num_data_to_save = min(len(data), per_rank_size)

        # Save the data for each rank
        data_to_save = np.random.RandomState(42).choice(data, num_data_to_save, replace=False).tolist()

        # Gather the data from all ranks
        gathered_data = all_gather_object(data_to_save, group=group)

        # Concatenate the data from all ranks
        saved_data = []
        for d in gathered_data:
            saved_data.extend(d)
        saved_data = saved_data[: self.max_record_texts]
        return saved_data

    def reduce(self, reset=False) -> list[str] | dict:
        output = self._reduce_list(self.data)
        if reset:
            self._reset()
        return output


class WithDataDumpOutlierMetric(BaseMetric):
    """This class is used to track any kind of outlier, for example outlier loss"""

    def __init__(
        self,
        name=None,
        reductions=None,
        sample_interval=1,
        is_group=False,
        max_window_size=100,
        outlier_data_dump_dir=None,
    ):
        super().__init__(
            name=name,
            reductions=reductions,
            sample_interval=sample_interval,
            is_group=is_group,
        )
        self.history_values = deque(maxlen=max_window_size)
        self.real_value_list: list[float] = []
        self.data_list = []
        self.output_dir = os.path.join(outlier_data_dump_dir, "outlier_losses")
        # mkdir in rank0
        if PM.world_rank == 0:
            os.makedirs(self.output_dir, exist_ok=True)
            logger.info(f"Created output directory for outlier data dump: {self.output_dir}")

    def add(self, value: dict, subname=None, iop=lambda x: x):
        """
        iop is for entire value
        """
        assert subname is None, "WithDataDumpOutlierMetric does not support subname for now"
        match value:
            case {"real_value": _, "data": _}:
                value = iop(value)
                self.real_value_list.append(value["real_value"])
                self.data_list.append(value["data"])
            case _:
                raise ValueError(f"Invalid value: {value}")

    def _reduce_current_rank(self, reset=True) -> float:
        """
        return value is how many items are larger than the mean of history_values
        """
        assert len(self.data_list) == len(
            self.real_value_list
        ), f"data_list and real_value_list must have the same length, but got {len(self.data_list)} and {len(self.real_value_list)}"
        res = 0.0
        if len(self.data_list) == 0:
            return res

        mean_value_item = torch.tensor(np.mean(self.real_value_list)).item()
        if len(self.history_values) <= self.history_values.maxlen:
            # just append mean value to history_value
            self.history_values.append(mean_value_item)
            return res

        history_values_mean = torch.tensor(self.history_values).mean().item()
        # now we traverse the data_list and real_value_list
        for item_idx in range(len(self.data_list)):
            real_value_item = self.real_value_list[item_idx]
            data_item = self.data_list[item_idx]
            if real_value_item > history_values_mean:
                dump_path = os.path.join(self.output_dir, f"{mean_value_item}.pt")
                torch.save(data_item, dump_path)
                logger.info(f"Dumped data for outlier losses: {real_value_item} to {dump_path}")
                res += 1
        return res

    def reduce(self, reset=True) -> float:
        res = torch.tensor(self._reduce_current_rank(reset=reset)).cuda()
        # NOTE: just hard code it here
        group = RDims._get_group("dp")
        dist_op = ROps._get_dist_op("sum")
        dist.all_reduce(res, op=dist_op, group=group)
        res = res.item()
        return res


class MetricHolder:
    def __init__(self):
        self.metrics: dict[str, Metric] = {}
        self._state_stack = []

    def register(
        self,
        metric: Metric,
        override=False,
    ):
        if not override:
            if metric.name in self.metrics and self.metrics[metric.name] != metric:
                raise Exception(f"Metric [{metric.name}] already registered!")
        self.metrics[metric.name] = metric

    def batch_register(self, metric_config: MetricConfig):
        for name, value in metric_config.items():
            if isinstance(value, BaseMetric):
                value.name = name
                if value.enabled:
                    self.register(value, override=True)

    @torch.no_grad()
    def add(self, name, value, subname=None, iop=lambda x: x):
        if name not in self.metrics:
            return
        # instant operation, skip this if not registered.
        self.metrics[name].add(value, subname=subname, iop=iop)

    def __getattr__(self, name):
        return self.metrics.get(name, dummy_metric)

    @torch.no_grad()
    def reduce(self, reset=True):
        result = dict()
        for name, metric in self.metrics.items():
            if name.startswith("_"):
                continue
            values = metric.reduce(reset=reset)
            if isinstance(values, dict):
                result.update({f"{name}/{k}": v for k, v in values.items()})
            else:
                result[name] = values
        return result

    @contextlib.contextmanager
    def fork(self):
        self._state_stack.append({k: v.get_data() for k, v in self.metrics.items()})
        try:
            yield
        finally:
            for k, v in self._state_stack.pop(-1).items():
                self.metrics[k].set_data(v)

    @contextlib.contextmanager
    def disable(self):
        metrics = self.metrics
        self.metrics = {}
        try:
            yield
        finally:
            self.metrics = metrics


class ModelLayerMetric(Metric):
    def reduce(self, reset=True) -> torch.Tensor | dict:
        """
        Stats over layers including avg, max, min, std for each metric
        """
        stats = defaultdict(list)
        reduced_data = super().reduce(reset=reset)
        # logging stats over layers
        for key, value in reduced_data.items():
            # {metric}/layer{layer_id}
            if "layer" in key:
                metric_name = key.split("/")[0] if "/" in key else self.name
                stats[metric_name].append(value)
        for metric_name, values in stats.items():
            # logging stats over layers including avg, max, min, std
            reduced_data[f"{metric_name}/avg"] = torch.stack(values).mean()
            reduced_data[f"{metric_name}/max"] = torch.stack(values).max()
            reduced_data[f"{metric_name}/min"] = torch.stack(values).min()
            reduced_data[f"{metric_name}/std"] = torch.stack(values).std()
        return reduced_data


GlobalMetrics: MetricConfig = MetricHolder()
