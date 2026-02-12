import os
from typing import Any

import torch
from configurize import Ref

from playground.pretrain.qwen3.qwen3_1p7b import (
    Qwen3_1p7BConfig,
    Qwen3OutputEmbeddingConfig,
)
from playground.rlvr.simple_trainable import SimpleTrainable
from steptronoss.core.context_parallel.context_parallel import (
    gather_from_balanced_cp_region,
    scatter_to_balanced_cp_region,
)
from steptronoss.core.parallel_state import PM
from steptronoss.core.tensor_parallel import vocab_parallel_cross_entropy
from steptronoss.data.nextable import Nextable
from steptronoss.exp.base_exp import (
    DataConfig,
    GradientManagerConfig,
    ProfilerConfig,
    TokenizerConfig,
)
from steptronoss.exp.inference import VLLMDeployConfig
from steptronoss.exp.lr_schedulers import ConstantSchedulerConfig
from steptronoss.exp.optimizer import AdamConfig
from steptronoss.exp.resources import ResourceConfig, TaskSpec
from steptronoss.exp.rl import (
    ActorModelConfig,
    CriticModelConfig,
    PackedPPOSamples,
    PPOCheckpointCfg,
    PPOLikeExp,
    PPOLikeTrainerConfig,
    PPOMetricConfig,
)
from steptronoss.generation.base_generatable import EndpointGetter, ModelNameGetter
from steptronoss.generation.vllm.vllm_router import VLLMRouterConfig
from steptronoss.utils.metrics import GlobalMetrics
from steptronoss.utils.rl_utils import compute_gae


class FakeGenableGenerator(Nextable):
    def __init__(
        self,
        endpoint_getter,
        model_name_getter,
        sampling_params: dict[str, Any],
        prompt_text: str,
        gt: str,
        max_tokens: int,
    ):
        super().__init__()
        self.endpoint_getter = endpoint_getter
        self.model_name_getter = model_name_getter
        self.sampling_params = sampling_params
        self.prompt_text = prompt_text
        self.gt = gt
        self.max_tokens = max_tokens

    def __next__(self) -> dict[str, Any]:
        return SimpleTrainable(
            endpoint_getter=self.endpoint_getter,
            model_name_getter=self.model_name_getter,
            sampling_params=self.sampling_params,
            prompt_text=self.prompt_text,
            gt=self.gt,
            max_tokens=self.max_tokens,
        )

    def state_dict(self) -> dict[str, Any]:
        pass

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        pass


class MathPromptDataConfig(DataConfig):
    data_path: str = "/mnt/step2-alignment-jfs/rlvr/math/prompts.jsonl"
    prompt_key: str = "prompt"
    answer_key: str = "answer"
    shuffle: bool = True
    seed: int = 1234
    build_tokenizer = Ref("..tokenizer_cfg.build_tokenizer")

    def build_dataloader(self, dp_rank=0, dp_size=1):
        vllm_cfg = TinyRLVRVLLMDeployConfig()
        endpoint_getter = EndpointGetter(vllm_cfg.router_addr_key)
        model_name_getter = ModelNameGetter(vllm_cfg.model_name_template)
        max_tokens = 1024
        sampling_params = vllm_cfg.get_sampling_params({"max_tokens": max_tokens})
        tokenizer = self.build_tokenizer()
        messages = [
            {
                "role": "user",
                "content": "请回答：strawberry里有几个r？请用 \\boxed{...} 给出最终答案。",
            }
        ]
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return FakeGenableGenerator(
            endpoint_getter=endpoint_getter,
            model_name_getter=model_name_getter,
            sampling_params=sampling_params,
            prompt_text=prompt_text,
            gt="3",
            max_tokens=max_tokens,
        )


class Qwen3TokenizerConfig(TokenizerConfig):
    tokenizer_path: str = "/mnt/step2-alignment-jfs/zane/opensources_model/Qwen3-1.7B/"

    def build_tokenizer(self):
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(self.tokenizer_path, trust_remote_code=True)


class TinyRLVRResourceConfig(ResourceConfig):
    def __init__(self):
        super().__init__()
        self.replica = 1
        self.gpu = 4
        self.node_type = "gpu"

        self.task_specs = {
            "trainer": TaskSpec(
                envs={
                    "ROLE": "trainer",
                    "CUDA_VISIBLE_DEVICES": "0,1,2,3",
                    "CUDA_LAUNCH_BLOCKING": "1",
                },
                is_critical=True,
                command="{TORCHRUN} {COMMAND}",
            ),
            "generator": TaskSpec(envs={"ROLE": "generator", "CUDA_VISIBLE_DEVICES": "4,5,6,7"}),
            "router": TaskSpec(gpu=0, node_type="cpu", envs={"ROLE": "router"}),
        }


class TinyRLVRVLLMDeployConfig(VLLMDeployConfig):
    def __init__(self):
        super().__init__()
        self.model_config_path = "/mnt/step2-alignment-jfs/zane/opensources_model/Qwen3-1.7B/"
        self.max_seq_len = 40960

        self.model_name_template = "deployed-model-{EXP_ID}"
        self.vllm_tp = 4

        self.hot_path = "/mnt/shared-storage/tenant/zhy/tmp/"


class TinyRLVRVLLMRouterConfig(VLLMRouterConfig):
    pass


class RLVRActorModelConfig(ActorModelConfig, Qwen3_1p7BConfig):
    vllm_cfg = TinyRLVRVLLMDeployConfig


class Qwen3ValueOutputEmbeddingConfig(Qwen3OutputEmbeddingConfig):
    def build_model(self, tied_embedding_weight=None):
        from torch.nn.init import trunc_normal_

        from steptronoss.core.parallel_state import PM
        from steptronoss.model.common.parallel_embedding import (
            OneDimensionalOutputEmbedding,
        )

        model = OneDimensionalOutputEmbedding(cfg=self)
        PM.register_rng("TP_DIFF", seed=1234, diff_across=["TP"])

        with PM.use_rng("TP_DIFF"):
            trunc_normal_(model.output.weight, std=1e-5)

        return model


class RLVRCriticModelConfig(CriticModelConfig, Qwen3_1p7BConfig):
    out_embed_cfg = Qwen3ValueOutputEmbeddingConfig

    def __init__(self):
        super().__init__()
        self.tie_embedding = False


class RLVRActorGradManagerConfig(GradientManagerConfig):
    optimizer_cfg = AdamConfig

    def __init__(self):
        super().__init__()
        self.optimizer_cfg.lr = Ref("...actor_scheduler_cfg.lr")
        self.optimizer_cfg.weight_decay = Ref("...actor_scheduler_cfg.weight_decay")
        self.params_dtype = Ref("..actor_model_cfg.params_dtype")


class RLVRCriticGradManagerConfig(GradientManagerConfig):
    optimizer_cfg = AdamConfig

    def __init__(self):
        super().__init__()
        self.optimizer_cfg.lr = Ref("...critic_scheduler_cfg.lr")
        self.optimizer_cfg.weight_decay = Ref("...critic_scheduler_cfg.weight_decay")
        self.params_dtype = Ref("..critic_model_cfg.params_dtype")


from steptronoss.core.generators.flow_controller import FlowControllerConfig


class OnpolicyFlowControllerConfig(FlowControllerConfig):
    def __init__(self):
        super().__init__()
        self.async_strategy = "on-policy"
        self.prompt_per_iter = 64


class RLVRTrainerConfig(PPOLikeTrainerConfig):
    flow_cfg = OnpolicyFlowControllerConfig

    def get_advantage_and_returns(
        self, ragged_samples: PackedPPOSamples, ragged_values: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cu_seqlens = ragged_samples.cu_seqlens
        is_gen_mask = ragged_samples.is_gen_mask[0]
        values_all = ragged_values
        advantages_list = []
        returns_list = []
        values_list = []
        for idx, sample in enumerate(ragged_samples.samples):
            start = cu_seqlens[idx].item()
            end = cu_seqlens[idx + 1].item()
            mask = is_gen_mask[start:end]
            resp_values = values_all[start:end][mask]
            if resp_values.numel() == 0:
                continue
            rewards = torch.zeros_like(resp_values)
            rewards[-1] = sample.raw_reward
            adv, ret = compute_gae(
                resp_values[None],
                rewards[None],
                lambd=self.ppo_actor_lambda,
                gamma=self.ppo_gae_gamma,
            )
            advantages_list.append(adv.squeeze(0))
            returns_list.append(ret.squeeze(0))
            values_list.append(resp_values)
            sample.advantages = adv.squeeze(0).detach()
            sample.returns = ret.squeeze(0).detach()
            sample.values = resp_values.detach()
        ragged_samples.advantages = torch.cat(advantages_list, dim=0)
        ragged_samples.returns = torch.cat(returns_list, dim=0)
        ragged_samples.values = torch.cat(values_list, dim=0)
        return ragged_samples.advantages, ragged_samples.returns

    def actor_loss_func(self, data: PackedPPOSamples, outputs: torch.Tensor) -> torch.Tensor:
        labels = data.labels
        labels = scatter_to_balanced_cp_region(labels, dim=1)
        new_logprobs = -vocab_parallel_cross_entropy(
            vocab_parallel_logits=outputs / self.policy_actor_temperature,
            target=labels.T.contiguous(),
        )[0].T
        new_logprobs = gather_from_balanced_cp_region(new_logprobs, dim=1)
        new_logprobs = new_logprobs[data.is_gen_mask].contiguous()
        data.actor_logprobs = new_logprobs

        old_logprobs = data.logprobs
        if old_logprobs is None:
            old_logprobs = new_logprobs.detach()
            data.logprobs = old_logprobs

        advantages = data.advantages
        advantages = advantages.clamp(self.advantage_clip_min, self.advantage_clip_max)
        ratio = torch.exp(new_logprobs - old_logprobs)
        surrogate1 = ratio * advantages
        surrogate2 = torch.clamp(ratio, 1 - self.ppo_clip, 1 + self.ppo_clip) * advantages
        pg_loss = -torch.min(surrogate1, surrogate2)
        loss = pg_loss.mean()

        kl = None
        if data.ref_logprobs is not None and self.ref_kl_loss_coeff > 0:
            kl = (new_logprobs - data.ref_logprobs).mean()
            loss = loss + self.ref_kl_loss_coeff * kl

        if self.sample_kl_loss_coeff > 0:
            kl_old = (new_logprobs - old_logprobs).mean()
            loss = loss + self.sample_kl_loss_coeff * kl_old

        GlobalMetrics.ppo_loss.add(loss, iop=torch.detach)
        if kl is not None:
            GlobalMetrics.kl_with_ref_dist.add(kl, iop=torch.detach)
        return loss

    def critic_loss_func(self, data: PackedPPOSamples, outputs: torch.Tensor) -> torch.Tensor:
        values = outputs
        if values.dim() >= 3:
            values = values[:, 0, 0]
        elif values.dim() == 2:
            values = values[:, 0]
        if PM.size_of("CP") > 1:
            values = gather_from_balanced_cp_region(values, dim=0)
        is_gen_mask = data.is_gen_mask
        if is_gen_mask.dim() > 1:
            is_gen_mask = is_gen_mask[0]
        values = values[is_gen_mask].contiguous()
        target = data.returns
        data.values = values.detach()
        loss = torch.nn.functional.mse_loss(values, target)
        GlobalMetrics.critic_loss.add(loss, iop=torch.detach)
        return loss


class Exp(PPOLikeExp):
    resource_cfg: TinyRLVRResourceConfig = TinyRLVRResourceConfig
    vllm_router_cfg: TinyRLVRVLLMRouterConfig = TinyRLVRVLLMRouterConfig

    tokenizer_cfg: Qwen3TokenizerConfig = Qwen3TokenizerConfig
    data_cfg: MathPromptDataConfig = MathPromptDataConfig

    actor_model_cfg: RLVRActorModelConfig = RLVRActorModelConfig
    critic_model_cfg: RLVRCriticModelConfig = RLVRCriticModelConfig

    actor_grad_manager_cfg: RLVRActorGradManagerConfig = RLVRActorGradManagerConfig
    actor_scheduler_cfg: ConstantSchedulerConfig = ConstantSchedulerConfig
    critic_grad_manager_cfg: RLVRCriticGradManagerConfig = RLVRCriticGradManagerConfig
    critic_scheduler_cfg: ConstantSchedulerConfig = ConstantSchedulerConfig

    checkpoint_cfg: PPOCheckpointCfg = PPOCheckpointCfg
    trainer_cfg: RLVRTrainerConfig = RLVRTrainerConfig
    metric_cfg: PPOMetricConfig = PPOMetricConfig
    profiler_cfg: ProfilerConfig = ProfilerConfig

    def __init__(self):
        super().__init__()
        self.log_dir = "./tensorboard_logs"

        self.trainer_cfg.train_iters = 1000
        self.trainer_cfg.global_seq_length = 4096
        self.trainer_cfg.fix_iters = 1
        self.trainer_cfg.fix_iters_critic = 1

        self.checkpoint_cfg.actor.load_safetensors = "/mnt/step2-alignment-jfs/zane/opensources_model/Qwen3-1.7B/"
        self.checkpoint_cfg.critic.load_safetensors = "/mnt/step2-alignment-jfs/zane/opensources_model/Qwen3-1.7B/"
        self.checkpoint_cfg.critic.strict_load_model = False
        self.checkpoint_cfg.reference.load_safetensors = "/mnt/step2-alignment-jfs/zane/opensources_model/Qwen3-1.7B/"

        self.checkpoint_cfg.save_path = "./checkpoints/qwen3_1p5b_rlvr_math"
        self.checkpoint_cfg.save_interval = 50

    def entrypoint(self):
        role = os.environ.get("ROLE", "trainer")
        if role == "router":
            self.vllm_router_cfg.run()
            return
        if role == "generator":
            self.actor_model_cfg.vllm_cfg.run_as_worker()
            return
        if role == "trainer":
            from steptronoss.utils.optimizable import set_optimization

            set_optimization(AttentionCore="flash-attn")
            self.train()
            return
        raise ValueError(f"Unknown ROLE: {role}")


if __name__ == "__main__":
    Exp().entrypoint()
