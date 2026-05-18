"""
Toy GLM-5 config for single-node debugging.

This keeps the GLM-5 block structure unchanged (MLA + DSA indexer + dense
prefix + routed/shared-expert MoE) and only shrinks the layer count so it can
be used for point-to-point validation on one 8-GPU node.
"""

import torch

from playground.pretrain.glm5.glm5 import GLM5Config


class GLM5ToyConfig(GLM5Config):
    def __init__(self):
        super().__init__()

        # Keep the per-layer structure identical to GLM-5 and only shorten the
        # stack so a single node can run fast enough for functional checks.
        self.num_layers = 6
        self.ffn_cfg.moe_cfg.moe_layer_list = list(range(3, self.num_layers))

        self.parallel_cfg.tensor_model_parallel_size = 8
        self.parallel_cfg.pipeline_model_parallel_size = 1
        self.parallel_cfg.virtual_pipeline_model_parallel_size = 1
        self.parallel_cfg.context_parallel_size = 1
        self.parallel_cfg.expert_model_parallel_size = 8
        self.parallel_cfg.expert_tensor_parallel_size = 1

        self.tp_cfg.sequence_parallel = False

    def build_model(self):
        model = super().build_model()
        for param in model.parameters():
            torch.nn.init.trunc_normal_(param)
        return model
