from loguru import logger

from steptronoss.exp.base_exp import BaseExp, TrainerConfig, TrainerHook


class BaseTrainer:
    exp: BaseExp

    iteration: int

    _after_init_hooks: list[TrainerHook]
    _before_train_hooks: list[TrainerHook]
    _before_step_hooks: list[TrainerHook]
    _after_step_hooks: list[TrainerHook]
    _after_train_hooks: list[TrainerHook]

    def build_hooks(self, trainer_cfg: TrainerConfig):
        self._after_init_hooks = trainer_cfg.build_after_init_hooks()
        self._before_train_hooks = trainer_cfg.build_before_train_hooks()
        self._before_step_hooks = trainer_cfg.build_before_step_hooks()
        self._after_step_hooks = trainer_cfg.build_after_step_hooks()
        self._after_train_hooks = trainer_cfg.build_after_train_hooks()

    # Functions for Training:
    @logger.catch()
    def train(self):
        self.before_train()
        self.train_loop()
        self.after_train()

    def before_train(self):
        # init_dist()
        for hook in self._after_init_hooks:
            hook(self)

        # build anything, model/optimizer
        for hook in self._before_train_hooks:
            hook(self)
        raise NotImplementedError

    def after_train(self):
        for hook in self._after_train_hooks:
            hook(self)
        raise NotImplementedError

    def train_loop(self):
        raise NotImplementedError

    def train_step(self):
        for hook in self._before_step_hooks:
            hook(self)

        # step
        raise NotImplementedError

        for hook in self._after_step_hooks:
            hook(self)
