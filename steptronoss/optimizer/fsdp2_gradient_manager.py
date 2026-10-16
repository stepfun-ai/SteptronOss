"""FSDP2 owns gradient reduction; never register native flat-buffer hooks."""

import torch
from packaging.version import Version
from torch.distributed.tensor import DTensor, Replicate, Shard

from steptronoss.exp.base_exp import GradientManagerConfig


def require_fsdp2_version():
    if Version(torch.__version__.split("+")[0]) < Version("2.11"):
        raise RuntimeError("This FSDP2 backend requires torch>=2.11; use the DP experiment on older versions")


def pack_local_state(value):
    """Serialize DTensors as rank-local CPU bytes, never gather the frozen base."""
    if isinstance(value, DTensor):
        placements = []
        for placement in value.placements:
            if isinstance(placement, Shard):
                placements.append(("shard", placement.dim))
            elif isinstance(placement, Replicate):
                placements.append(("replicate", None))
            else:
                raise TypeError(f"Unsupported checkpoint placement: {placement}")
        return {
            "_steptron_dtensor": 1,
            "local": value.to_local().detach().cpu().clone(),
            "shape": tuple(value.shape),
            "stride": tuple(value.stride()),
            "placements": placements,
        }
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: pack_local_state(item) for key, item in value.items()}
    if isinstance(value, list):
        return [pack_local_state(item) for item in value]
    if isinstance(value, tuple):
        return tuple(pack_local_state(item) for item in value)
    return value


def unpack_local_state(value, mesh, device):
    if isinstance(value, dict) and value.get("_steptron_dtensor") == 1:
        placements = [Shard(dim) if kind == "shard" else Replicate() for kind, dim in value["placements"]]
        return DTensor.from_local(
            value["local"].to(device),
            device_mesh=mesh,
            placements=placements,
            shape=torch.Size(value["shape"]),
            stride=value["stride"],
            run_check=False,
        )
    if isinstance(value, dict):
        return {key: unpack_local_state(item, mesh, device) for key, item in value.items()}
    if isinstance(value, list):
        return [unpack_local_state(item, mesh, device) for item in value]
    if isinstance(value, tuple):
        return tuple(unpack_local_state(item, mesh, device) for item in value)
    return value


class FSDP2GradientManagerConfig(GradientManagerConfig):
    use_distributed_optimizer: bool = False
    """Native ZeRO-1 is incompatible; FSDP2 already shards the optimizer parameters."""

    def build_gradient_manager(self, model):
        require_fsdp2_version()
        if self.use_distributed_optimizer:
            raise ValueError("Do not combine native ZeRO-1 with FSDP2")
        return FSDP2GradientManager(self, model, self.optimizer_cfg.build_optimizer(model))


class FSDP2GradientManager:
    """Minimal Steptron trainer protocol over torch's DTensor-aware optimizer."""

    def __init__(self, cfg, model, optimizer):
        self.cfg, self.model, self.optimizer = cfg, model, optimizer
        self.params = [p for group in optimizer.param_groups for p in group["params"]]
        if not self.params or not all(isinstance(p, DTensor) for p in self.params):
            raise ValueError("FSDP2GradientManager requires trainable DTensor parameters")
        self.mesh = self.params[0].device_mesh

    def zero_grad(self, set_to_none=True, **kwargs):
        self.optimizer.zero_grad(set_to_none=set_to_none, **kwargs)

    def _sync_cp_gradients(self):
        """Flush native accumulated LoRA gradients in a fixed group order.

        CP recurrence has asymmetric backward dependencies. Reducing parameters
        during that backward can reorder FSDP collectives across ranks. Native
        no-sync accumulation avoids this. A zero-input adapter-only backward
        flushes each group through the public FSDP API after recurrence is done.
        It never changes weights or the training objective.
        """
        groups = getattr(self.model[0], "cp_sync_groups", ())
        for modules in groups:
            owner = modules[0]
            owner.set_requires_gradient_sync(True, recurse=False)
            owner.unshard()
            zero_loss = None
            for module in modules:
                param = next(module.parameters())
                x = torch.zeros((1, module.in_features), device=param.device, dtype=param.dtype)
                term = module(x).sum() * 0.0
                zero_loss = term if zero_loss is None else zero_loss + term
            zero_loss.backward()
            owner.set_requires_gradient_sync(False, recurse=False)

    def step(self):
        self._sync_cp_gradients()
        # DTensor computes the global norm over shards. No second DP all-reduce.
        norm = torch.nn.utils.clip_grad_norm_(
            self.params, self.cfg.clip_grad if self.cfg.clip_grad > 0 else float("inf"), foreach=False
        )
        norm = norm.full_tensor() if isinstance(norm, DTensor) else norm
        norm_value = norm.item()
        if not torch.isfinite(norm):
            return False, norm_value, 0
        zeros = torch.zeros((), dtype=torch.int64, device=self.params[0].device)
        if self.cfg.log_num_zeros_in_grad:
            for param in self.params:
                if param.grad is not None:
                    zeros += (param.grad.to_local() == 0).sum()
            torch.distributed.all_reduce(zeros, group=self.mesh.get_group())
        self.optimizer.step()
        return True, norm_value, zeros.item()

    def state_dict(self):
        return pack_local_state(self.optimizer.state_dict())

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(unpack_local_state(state_dict, self.mesh, self.params[0].device))

    def reload_model_params(self):
        pass  # No native FP32 master-weight copies; Adam updates the adapter shards.
