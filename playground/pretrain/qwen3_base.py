import torch

from playground.pretrain.qwen3.qwen3_1p7b import Qwen3_1p7BConfig
from steptronoss.exp.base_exp import BaseExp, GradientManagerConfig
from steptronoss.exp.ntp import NTPTrainerConfig


class Exp(BaseExp):
    model_cfg = Qwen3_1p7BConfig
    grad_manager_cfg = GradientManagerConfig
    trainer_cfg = NTPTrainerConfig

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.grad_manager_cfg.optimizer_cfg.lr = 1
        self.grad_manager_cfg.optimizer_cfg.weight_decay = 0.9
        self.grad_manager_cfg.clip_grad = 0.0

        self.grad_manager_cfg.use_distributed_optimizer = True
        self.grad_manager_cfg.params_dtype = torch.bfloat16
        self.model_cfg.overlap_p2p_comm = True
        self.trainer_cfg.global_seq_length = 128
        self.trainer_cfg.micro_batch_size = 1
        self.trainer_cfg.global_batch_size = 8

        from playground.pretrain.qwen3.qwen3_1p7b import Qwen3FeedForwardConfig
        from steptronoss.model.common.feed_forward import FeedForwardConfig
        from steptronoss.model.common.moe_layers import MoEConfig

        moe_cfg = MoEConfig()
        moe_cfg.moe_num_experts = 8
        moe_cfg.moe_top_k = 2
        moe_cfg.moe_aux_loss_coef = 0.1
        moe_cfg.moe_hidden_size = 1024

        moe_cfg.routed_scaling_factor = 2.0
        moe_cfg.enable_sigmoid_router = False
        moe_cfg.router_bias_update_rate = 0.01
        moe_cfg.moe_enable_deepep = True

        moe_cfg.fuse_moescatter_and_moecolumn = False
        moe_cfg.enable_auxiliary_loss_free_load_balance = True
        moe_cfg.norm_expert_weight = False
        moe_cfg.moe_layer_list = [0]
        """layer ids of layers which use moe"""

        moe_cfg.share_expert_dim = 512
        """Dimension of shared moe ffn"""
        self.model_cfg._allow_set_new_attr = True
        self.model_cfg.moe_cfg = moe_cfg

        self.model_cfg.parallel_cfg.tensor_model_parallel_size = 1
        self.model_cfg.parallel_cfg.expert_model_parallel_size = 1


import sys
import time

sys.excepthook = lambda a, b, c: [print("Hold On Error"), time.sleep(3600)]

if __name__ == "__main__":
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

    from steptronoss.model.common.moe_share_expert_ffn import MoeShareExpertFFN

    module = MoeShareExpertFFN(
        moe_cfg=exp.model_cfg.moe_cfg,
        ffn_cfg=exp.model_cfg.ffn_cfg,
        layer_id=0,
    )
    with torch.no_grad():
        torch.nn.init.trunc_normal_(module.moe.gate.weight)
        module.share_expert.w1.weight.fill_(0.1)
        module.share_expert.w2.weight.fill_(0.2)

        for eid in range(len(module.moe.w1)):
            module.moe.w1[eid].fill_((eid + 1 + PM.rank_in("EP") * 4) / 10)
            module.moe.w2[eid].fill_((eid + 1 + PM.rank_in("EP") * 4) / 10)
    module.cuda().bfloat16()

    gm = exp.grad_manager_cfg.build_gradient_manager(model=module)

    x = torch.randn((16, 1, exp.model_cfg.ffn_cfg.hidden_size), dtype=torch.bfloat16, device="cuda")

    y = module.forward(x)
    y.sum().backward()
    logger.warning(y[13, 0, -1])
    logger.warning(module.moe.w1.main_grad)

    if PM.world_rank == 0:
        assert module.moe.w1.main_grad[0, 0, 0] == 3776
    if PM.size_of("EP") == 2:
        if PM.world_rank == 1:
            assert module.moe.w1.main_grad[0, 0, 0] == 5536
    else:
        if PM.world_rank == 1:
            assert module.moe.w1.main_grad[0, 0, 0] == 3776

    y = module.forward(x)
    y.sum().backward()
    logger.warning(y)
    logger.warning(module.moe.w1.main_grad)

    if PM.world_rank == 0:
        assert module.moe.w1.main_grad[0, 0, 0] == 7552
    if PM.size_of("EP") == 2:
        if PM.world_rank == 1:
            assert module.moe.w1.main_grad[0, 0, 0] == 11072
    else:
        if PM.world_rank == 1:
            assert module.moe.w1.main_grad[0, 0, 0] == 7552
    # print(y.shape, y)
