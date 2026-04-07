from __future__ import annotations

import torch
from configurize import Config, DataClass, Ref
from loguru import logger

from steptronoss.core import tensor_parallel
from steptronoss.exp.base_exp import MegatronTPConfig
from steptronoss.model.common.rms_norm import RMSNorm
from steptronoss.model.common.vit import VisionTransformerConfig


class InputEmbeddingConfig(Config):
    tp_cfg: MegatronTPConfig = Ref("..tp_cfg")
    vocab_size: int
    hidden_size: int
    embedding_weights_in_fp32: bool
    fp32_residual_connection: bool

    def build_model(self):
        return WordEmbedding(cfg=self)


class OutputEmbeddingConfig(Config):
    tp_cfg: MegatronTPConfig = Ref("..tp_cfg")
    vocab_size: int
    hidden_size: int
    fp32_rms_norm: bool
    fp32_lm_head_out: bool = False
    """If true, output LM head logits in fp32."""

    rms_norm_zero_gamma: bool
    layernorm_epsilon: float

    gather_output: bool

    def build_model(self, tied_embedding_weight: torch.Tensor | None = None):
        return OutputEmbedding(cfg=self, tied_word_embedding_weight=tied_embedding_weight)


class WordEmbedding(torch.nn.Module):
    """Language model embeddings."""

    def __init__(self, cfg: InputEmbeddingConfig):
        super().__init__()

        self.hidden_size = cfg.hidden_size

        # Word embeddings (parallel).
        self.embedding_weights_in_fp32 = cfg.embedding_weights_in_fp32
        self.params_dtype = cfg.tp_cfg.params_dtype

        self.word_embeddings = tensor_parallel.SimpleVocabParallelEmbedding(
            cfg.vocab_size, self.hidden_size, params_dtype=cfg.tp_cfg.params_dtype
        )

        self.fp32_residual_connection = cfg.fp32_residual_connection
        self.sequence_parallel = cfg.tp_cfg.sequence_parallel

    def forward(self, input_ids, **kwargs):
        # Embeddings.
        input_ids = input_ids.contiguous()
        if self.embedding_weights_in_fp32:
            self.word_embeddings = self.word_embeddings.to(torch.float32)
        embeddings = self.word_embeddings(input_ids)
        if self.embedding_weights_in_fp32:
            embeddings = embeddings.to(self.params_dtype)
            self.word_embeddings = self.word_embeddings.to(self.params_dtype)

        # Data format change to avoid explicit tranposes : [b s h] --> [s b h].
        embeddings = embeddings.transpose(0, 1).contiguous()

        # If the input flag for fp32 residual connection is set, convert for float.
        if self.fp32_residual_connection:
            embeddings = embeddings.float()

        # Dropout.
        if self.sequence_parallel:
            embeddings = tensor_parallel.scatter_to_sequence_parallel_region(embeddings)

        return embeddings


class ImageForInsert(DataClass):
    insert_start_token: int
    """Token id after which visual features are inserted."""

    images: torch.FloatTensor | None = None
    """Raw image tensor shaped [N, 3, H, W]."""

    image_features: torch.FloatTensor | list[torch.FloatTensor] | None = None
    """Optional precomputed features before the language projector."""

    rope_cu_seqlens: torch.IntTensor | None = None
    """Reserved for future multimodal RoPE variants."""

    rope_max_seq_len: int | None = None
    """Reserved for future multimodal RoPE variants."""


class ImageInsertInputEmbeddingConfig(InputEmbeddingConfig):
    encoder_cfg: VisionTransformerConfig = VisionTransformerConfig
    img_start_token: int
    encoder_no_grad: bool = False
    """If true, freeze the vision encoder and skip its gradients."""

    projector_bias: bool = False
    """Whether the vision-language projector uses bias."""

    def build_adapter(self) -> torch.nn.Module:
        linear_layer = torch.nn.Linear(
            self.encoder_cfg.output_dim,
            self.hidden_size,
            bias=self.projector_bias,
        )
        torch.nn.init.kaiming_normal_(linear_layer.weight, mode="fan_in", nonlinearity="relu")
        if linear_layer.bias is not None:
            torch.nn.init.zeros_(linear_layer.bias)
        return linear_layer

    def build_model(self):
        return ImageInsertEmbedding(cfg=self)


class ImageInsertEmbedding(WordEmbedding):
    def __init__(self, cfg: ImageInsertInputEmbeddingConfig):
        super().__init__(cfg)
        self.cfg = cfg
        self.align_projector = cfg.build_adapter().to(self.word_embeddings.weight.device)

    @staticmethod
    def _normalize_feature_list(
        image_features: torch.Tensor | list[torch.Tensor],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> list[torch.Tensor]:
        if isinstance(image_features, torch.Tensor):
            if image_features.dim() != 3:
                raise ValueError(f"Expected image_features with shape [N, L, C], got {tuple(image_features.shape)}")
            features = list(image_features.unbind(0))
        else:
            features = []
            for feature in image_features:
                if feature.dim() == 3:
                    features.extend(list(feature.unbind(0)))
                elif feature.dim() == 2:
                    features.append(feature)
                else:
                    raise ValueError(f"Expected feature tensor rank 2 or 3, got rank {feature.dim()}")
        return [feature.to(device=device, dtype=dtype) for feature in features]

    @staticmethod
    def insert_features(
        input_embeddings: torch.Tensor,
        image_features: torch.Tensor | list[torch.Tensor],
        input_ids: torch.Tensor,
        flag: int,
    ) -> torch.Tensor:
        feature_list = ImageInsertEmbedding._normalize_feature_list(
            image_features,
            device=input_embeddings.device,
            dtype=input_embeddings.dtype,
        )
        if not feature_list:
            return input_embeddings

        insert_locations = torch.nonzero(input_ids == flag, as_tuple=False)
        if insert_locations.numel() == 0:
            return input_embeddings
        insert_locations = insert_locations.clone()
        insert_locations[:, 1] += 1

        if len(feature_list) != insert_locations.shape[0]:
            logger.warning(
                "Mismatch between image features and insert locations: "
                f"{len(feature_list)} features vs {insert_locations.shape[0]} locations; truncating to the overlap."
            )
        count = min(len(feature_list), insert_locations.shape[0])
        if count == 0:
            return input_embeddings

        output = input_embeddings.clone()
        for location, feature in zip(insert_locations[:count], feature_list[:count], strict=False):
            batch_idx = int(location[0].item())
            start = int(location[1].item())
            end = min(start + feature.shape[0], output.shape[0])
            if end <= start:
                continue
            output[start:end, batch_idx] = feature[: end - start]
        return output

    def forward(self, input_ids: torch.IntTensor, images: list[ImageForInsert] | None = None, **kwargs):
        input_embeddings = super().forward(input_ids, **kwargs)
        if not images:
            return input_embeddings

        target_dtype = self.align_projector.weight.dtype
        target_device = self.align_projector.weight.device
        for insert_image in images:
            if insert_image.image_features is not None:
                image_features = insert_image.image_features
            else:
                raise ValueError("ImageInsertEmbedding expects precomputed image_features in decoupled-encoder mode")

            if isinstance(image_features, torch.Tensor):
                image_features = image_features.to(device=target_device, dtype=target_dtype)
                image_features = self.align_projector(image_features)
            else:
                image_features = [
                    self.align_projector(feature.to(device=target_device, dtype=target_dtype))
                    for feature in image_features
                ]

            input_embeddings = self.insert_features(
                input_embeddings=input_embeddings,
                image_features=image_features,
                input_ids=input_ids,
                flag=insert_image.insert_start_token,
            )
        return input_embeddings


class OutputEmbedding(torch.nn.Module):
    def __init__(
        self,
        cfg: OutputEmbeddingConfig,
        tied_word_embedding_weight: torch.Tensor | None,
    ):
        super().__init__()
        self.norm = RMSNorm(
            cfg.hidden_size,
            eps=cfg.layernorm_epsilon,
            sequence_parallel=cfg.tp_cfg.sequence_parallel,
            use_fp32=cfg.fp32_rms_norm,
            use_zero_init=cfg.rms_norm_zero_gamma,
        )
        self.output = tensor_parallel.ColumnParallelLinear(
            cfg.hidden_size,
            cfg.vocab_size,
            bias=False,
            gather_output=cfg.gather_output,
            async_tensor_model_parallel_allreduce=cfg.tp_cfg.async_tensor_model_parallel_allreduce,
            params_dtype=cfg.tp_cfg.params_dtype,
            gradient_accumulation_fusion=cfg.tp_cfg.gradient_accumulation_fusion,
            sequence_parallel_enabled=cfg.tp_cfg.sequence_parallel,
            tie_word_embeddings_weight=tied_word_embedding_weight,
            fp32_output=cfg.fp32_lm_head_out,
        )

    def forward(self, hidden_states, **kwargs):
        hidden_states = self.norm(hidden_states)
        output = self.output(hidden_states)[0]
        return output


class OneDimensionalOutputEmbedding(torch.nn.Module):
    def __init__(self, cfg: OutputEmbeddingConfig):
        super().__init__()
        self.sequence_parallel = cfg.tp_cfg.sequence_parallel

        self.norm = RMSNorm(
            cfg.hidden_size,
            eps=cfg.layernorm_epsilon,
            sequence_parallel=cfg.tp_cfg.sequence_parallel,
            use_fp32=cfg.fp32_rms_norm,
        )
        kwargs = cfg.tp_cfg.get_tp_kwargs()
        kwargs["sequence_parallel_enabled"] = False
        self.output = tensor_parallel.RowParallelLinear(
            cfg.hidden_size,
            output_size=1,
            bias=False,
            input_is_parallel=False,
            **kwargs,
        )

    def forward(self, hidden_states, **kwargs):
        hidden_states = self.norm(hidden_states)
        if self.sequence_parallel:
            hidden_states = tensor_parallel.gather_from_sequence_parallel_region(hidden_states)
        output = self.output(hidden_states)[0]
        return output

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        loaded_key = prefix + "output.weight"
        loaded_weight = state_dict[loaded_key]
        if loaded_weight.shape != self.output.weight.shape:
            logger.warning(f"Output layer weight mismatch: {loaded_weight.shape} vs {self.output.weight.shape}")
            state_dict.pop(loaded_key)

            logger.info("Skip `out_embeddings.output.weight` from pretrained weights")
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
