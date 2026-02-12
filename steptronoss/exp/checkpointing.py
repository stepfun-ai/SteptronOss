from __future__ import annotations

import os

from configurize import Config, Ref, writable_property


class CkptOptions(Config):
    """Option Config, use True for load

    options.none(but=['model']) for load model only

    options.all(but=['optimizer']) for NOT load optimizer only.
    """

    def all(self, but: list[str] = []) -> CkptOptions:
        for k, _ in self.items():
            setattr(self, k, k not in but)
        return self

    def none(self, but: list[str] = []) -> CkptOptions:
        for k, _ in self.items():
            setattr(self, k, k in but)
        return self


class SaveOptions(CkptOptions):
    model: bool = True
    optimizer: bool = True
    scheduler: bool = True
    data: bool = True
    rng_state: bool = True


class LoadOptions(CkptOptions):
    model: bool = True
    optimizer: bool = True
    scheduler: bool = True
    data: bool = True
    rng_state: bool = True

    exp: bool = True
    iter: bool = True


class CheckpointConfig(Config):
    auto_resume: bool

    load_path: str | None

    save_dir: str

    save_interval: int

    async_dump: bool

    load_option: type[LoadOptions] = LoadOptions
    load_safetensors: bool | str
    """Load weight from safetensors. If True, load 'load_path/hf'; if str, load the value."""

    save_option: type[SaveOptions] = SaveOptions
    save_safetensors: bool
    """If set, dump an extra hf weight to 'save_path/hf'"""

    broadcast_from_dp0: bool

    strict_load_model: bool

    use_distributed_optimizer: bool

    exp_name: str

    # Enable online reshard of distributed optimizer state when DP size changes.
    reshard_optimizer_state: bool
    reshard_optimizer_strict: bool

    # for save_safetensors reference
    model_config_path: str | None

    tokenizer_path: str | None

    @writable_property
    def save_path(self) -> str:
        return os.path.join(self.save_dir, self.exp_name)

    def sanity_check(self) -> None:
        super().sanity_check()
        assert not (self.load_safetensors and self.load_path), (
            "load_safetensors and load_path cannot be set at the same time"
        )

    def __init__(self):
        super().__init__()
        self.auto_resume = True
        self.load_path = None
        self.save_dir = "./"
        self.save_interval = 100
        self.async_dump = True
        self.load_safetensors = False
        self.save_safetensors = False
        self.broadcast_from_dp0 = True
        self.strict_load_model = True
        self.use_distributed_optimizer = Ref("..optimizer_cfg.use_distributed_optimizer", False)
        self.exp_name = Ref("..exp_name", "unknonw_exp")
        self.reshard_optimizer_state = False
        self.reshard_optimizer_strict = True
        self.model_config_path = None
        self.tokenizer_path = None
