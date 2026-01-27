Config Principle

Core idea
- Separate stateful objects (e.g., Optimizer) from stateless configs (e.g., OptimizerConfig).
- Config only describes parameters and structure.
- Config provides build() (or build_*) to create the real object.

How to define a Config
- Inherit from Config.
- Use type hints to declare required parameters and their types.
- Sub-configs are declared as class attributes (type + class attr).
- Concrete values are assigned on the instance (typically in __init__).
- Use Ref("..path") to reference other nodes in the config tree.
- Config.__init__ will instantiate all sub-configs declared as class attrs.

How to use a Config
- sanity_check(): validate required attributes and constraints.
- to_dict(): serialize config for logging or saving.

Example (minimal)

```python
from configurize import Config, Ref

# Base config: only types, no values
class OptimizerConfig(Config):
    lr: float
    weight_decay: float

    def build(self, params):
        return Optimizer(params, lr=self.lr, weight_decay=self.weight_decay)


# Concrete config: assigns values
class MyOptimizerConfig(OptimizerConfig):
    def __init__(self):
        super().__init__()
        self.lr = 1e-4
        self.weight_decay = 0.01


class TrainerConfig(Config):
    lr: float
    optimizer_cfg: OptimizerConfig = MyOptimizerConfig  # class attr defines sub-config type

    def __init__(self):
        super().__init__()
        self.lr = 1e-4
        # reference parent value
        self.optimizer_cfg.lr = Ref("..lr")


cfg = TrainerConfig()
cfg.sanity_check()
print(cfg.to_dict())
optim = cfg.optimizer_cfg.build(params)
```
