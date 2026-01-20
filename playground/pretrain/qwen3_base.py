import torch

from playground.pretrain.qwen3.qwen3_1p7b import Qwen3_1p7BConfig
from steptronoss.exp.base_exp import BaseExp, GradientManagerConfig, TrainerConfig


class Exp(BaseExp):
    model_cfg = Qwen3_1p7BConfig
    grad_manager_cfg = GradientManagerConfig
    trainer_cfg = TrainerConfig

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.grad_manager_cfg.optimizer_cfg.lr = 1
        self.grad_manager_cfg.optimizer_cfg.weight_decay = 0.9
        self.grad_manager_cfg.clip_grad = 0.0

        self.grad_manager_cfg.use_distributed_optimizer = True
        self.grad_manager_cfg.params_dtype = torch.bfloat16
        self.model_cfg.overlap_p2p_comm = True


# import sys, time
# sys.excepthook = lambda a,b,c:time.sleep(3600)

if __name__ == "__main__":
    from steptronoss.core.parallel_state import PM, get_vpp_size, set_vpp_rank
    from steptronoss.initialize import set_mpu_random_seed
    from steptronoss.utils import print_n_params, profile_allreduce
    from steptronoss.utils.logger import setup_logger
    from steptronoss.utils.weight_loader import HFWeights

    logger = setup_logger("./tensorboard_dir/")
    exp = Exp()
    logger.info(exp)

    PM.initialize()
    PM.set_mesh(exp.model_cfg.parallel_cfg)
    set_mpu_random_seed(1234)

    models = []
    for i in range(get_vpp_size()):
        set_vpp_rank(i)
        model = exp.model_cfg.build_model()
        model.load_hf_state_dict(
            HFWeights("/mnt/step2-alignment-jfs/zane/opensources_model/Qwen3-1.7B-Base/"),
            strict=False,
        )
        # from steptron import debug;debug()
        from steptronoss.model.module import Float16Module

        model = Float16Module(model, dtype=exp.model_cfg.params_dtype).cuda()
        models.append(model)
    print_n_params(models)

    gms = [exp.grad_manager_cfg.build_gradient_manager(model) for model in models]

    x = torch.arange(1024, dtype=torch.long, device="cuda").reshape(1, -1)
    cu_seqlens = torch.tensor([0, 1024], dtype=torch.int32, device="cuda")
    data = dict(input_ids=x, cu_seqlens=cu_seqlens)

    profile_allreduce()

    pp_scheduler = exp.model_cfg.get_pp_scheduler()
    pp_scheduler.configure(
        models=models,
        data_iterators=[iter([data] * 100)] * get_vpp_size(),
        data_sync_fn=exp.trainer_cfg.sync_get_data,
        loss_fn=lambda data, logits: logits.sum(),
        data_proc_fn=lambda x: x,
        training=True,
        collect_output=False,
    )

    out = pp_scheduler.run(forward_num=16)
    if PM.i_am("PP", 0):
        logger.warning(models[0].module.tok_embeddings.word_embeddings.weight)

    for gm in gms:
        success, grad_norm, grad_zeros = gm.step()
        gm.zero_grad()
    if PM.i_am("PP", 0):
        logger.warning(models[0].module.tok_embeddings.word_embeddings.weight)
