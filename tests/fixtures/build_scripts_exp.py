from steptronoss.exp.base_exp import BaseExp
from steptronoss.exp.resources import TorchrunResourceConfig


class Exp(BaseExp):
    resource_cfg = TorchrunResourceConfig

    def __init__(self):
        super().__init__()
        self.resource_cfg.envs = {"FOO": "BAR"}
        self.resource_cfg.task_specs = {
            "train": dict(replica=2, gpu=8, envs={"TRAIN_ONLY": "1"}),
            "infer": dict(replica=4, gpu=4),
            "control": dict(replica=1, node_type="cpu", gpu=0),
        }
