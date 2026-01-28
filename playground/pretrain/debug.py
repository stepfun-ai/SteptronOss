import torch

from playground.pretrain.qwen3.qwen3_30a3b import Qwen3_30A3BConfig
from steptronoss.exp.base_exp import BaseExp, GradientManagerConfig
from steptronoss.exp.ntp import NTPTrainerConfig


class Exp(BaseExp):
    model_cfg = Qwen3_30A3BConfig
    grad_manager_cfg = GradientManagerConfig
    trainer_cfg = NTPTrainerConfig

    def __init__(self):
        super().__init__()
        self.grad_manager_cfg.optimizer_cfg.lr = 1
        self.grad_manager_cfg.optimizer_cfg.weight_decay = 0.9
        self.grad_manager_cfg.clip_grad = 0.0

        self.grad_manager_cfg.use_distributed_optimizer = True
        self.grad_manager_cfg.params_dtype = torch.bfloat16
        self.model_cfg.overlap_p2p_comm = True
        self.trainer_cfg.global_seq_length = 128
        self.trainer_cfg.micro_batch_size = 1
        self.trainer_cfg.global_batch_size = 8

    def train(self):
        self.update_from_args()
        self.sanity_check()

        self.trainer_cfg.get_trainer_cls()(self).train()


import sys
import time

# sys.excepthook = lambda a, b, c: [print("Hold On Error"), time.sleep(3600)]

if __name__ == "__main__":
    Exp().train()
    exit()
    from steptronoss.core.parallel_state import PM, get_vpp_size, set_vpp_rank
    from steptronoss.initialize import set_mpu_random_seed
    from steptronoss.utils import print_n_params, profile_allreduce
    from steptronoss.utils.logger import setup_logger
    from steptronoss.utils.weight_loader import HFWeights

    logger = setup_logger("./tensorboard_dir/")
    exp = Exp()

    PM.initialize()
    PM.set_mesh(exp.model_cfg.parallel_cfg)
    set_mpu_random_seed(1234)

    logger.info(exp)

    model = exp.model_cfg.build_model().to(torch.bfloat16).cuda()

    gm = exp.grad_manager_cfg.build_gradient_manager(model)

    x = torch.ones(
        (exp.trainer_cfg.micro_batch_size, exp.trainer_cfg.global_seq_length),
        device="cuda",
        dtype=torch.long,
    )
    y = model(x)
    y.sum().backward()
    out = gm.step()
    print(out)
