from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from steptronoss.core.context_parallel import scatter_to_balanced_cp_region
from steptronoss.core.parallel_state import PM
from steptronoss.model.common.mesh_connector import MeshConnector
from steptronoss.model.common.parallel_embedding import ImageForInsert


class ImageInsertDecoderMixin:
    """Decoder-model mixin for decoupled vision encoding and token insertion."""

    cfg: Any
    encoder: torch.nn.Module
    mesh_connector: MeshConnector
    tok_embeddings: Any

    if TYPE_CHECKING:

        def is_pipeline_first_stage(self) -> bool: ...

    def build_multimodal_modules(self) -> None:
        encoder_cfg = self.cfg.tok_embed_cfg.encoder_cfg
        with PM.use_mesh(encoder_cfg.parallel_cfg):
            self.encoder = encoder_cfg.build_model()
            self.encoder = self.encoder.to(self.cfg.tp_cfg.params_dtype)
            self.encoder = self.encoder.to(torch.cuda.current_device() if torch.cuda.is_available() else "cpu")

        if self.cfg.tok_embed_cfg.encoder_no_grad:
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.mesh_connector = MeshConnector(
            src_mesh=self.cfg.parallel_cfg,
            dst_mesh=encoder_cfg.parallel_cfg,
            is_data_source=self.is_pipeline_first_stage(),
            dup_dim=["TP"],
        )

    def _build_multimodal_reshape_scripts(self):
        from steptronoss.checkpointing.reshape_ops import (
            Duplicate,
            KeepThisTP,
            Rename,
            Script,
        )

        scripts = []
        if not hasattr(self, "encoder"):
            return scripts

        if hasattr(self, "tok_embeddings") and hasattr(self.tok_embeddings, "align_projector"):
            scripts.append(
                Script(
                    src="vit_large_projector.weight",
                    op=Duplicate()
                    + KeepThisTP()
                    + Rename("tok_embeddings.align_projector.weight: vit_large_projector.weight"),
                    dst="tok_embeddings.align_projector.weight",
                )
            )
            scripts.append(
                Script(
                    src="tok_embeddings.align_projector.weight",
                    op=Duplicate() + KeepThisTP(),
                    dst=[],
                )
            )

        if hasattr(self.encoder, "reshaper"):
            sub_reshaper = self.encoder.reshaper
            scripts.append(
                Script(
                    src="vision_model.*",
                    op=Rename("*: vision_model.*")
                    + Rename({
                        "transformer.resblocks.*.attn.qkv_proj.weight": "transformer.resblocks.*.attn.in_proj_weight",
                        "transformer.resblocks.*.attn.qkv_proj.bias": "transformer.resblocks.*.attn.in_proj_bias",
                    })
                    + sub_reshaper
                    + Rename("encoder.*: *"),
                    dst="encoder.*",
                )
            )
            scripts.append(
                Script(
                    src="encoder.*",
                    op=Rename("*: encoder.*") + sub_reshaper + Rename("encoder.*: *"),
                    dst=[],
                )
            )

        return scripts

    def _encode_images_for_insert(self, images: list[ImageForInsert] | None) -> list[ImageForInsert] | None:
        if not images:
            return images

        processed = []
        encoder_dtype = next(self.encoder.parameters()).dtype
        encoder_device = next(self.encoder.parameters()).device
        for insert_image in images:
            if insert_image.image_features is not None:
                processed.append(insert_image)
                continue
            if insert_image.images is None:
                raise ValueError("ImageForInsert requires either images or image_features")

            local_images = self.mesh_connector.forward(insert_image.images)
            if local_images is None:
                raise ValueError("MeshConnector returned no images for ImageForInsert")
            if local_images.shape[0] == 0:
                local_features = local_images.new_empty((0, 0, 0))
            else:
                with (
                    PM.use_mesh(self.cfg.tok_embed_cfg.encoder_cfg.parallel_cfg),
                    torch.set_grad_enabled(not self.cfg.tok_embed_cfg.encoder_no_grad),
                ):
                    local_features = self.encoder(local_images.to(device=encoder_device, dtype=encoder_dtype))
            image_features = self.mesh_connector.backward(local_features)
            processed.append(
                ImageForInsert(
                    insert_start_token=insert_image.insert_start_token,
                    image_features=image_features,
                    rope_cu_seqlens=insert_image.rope_cu_seqlens,
                    rope_max_seq_len=insert_image.rope_max_seq_len,
                )
            )
        return processed

    def _prepare_inputs(self, kwargs):
        kwargs = dict(kwargs)
        images = kwargs.get("images")
        if not images:
            return kwargs
        kwargs["images"] = self._encode_images_for_insert(images)
        return kwargs

    def forward_head(self, input_ids: torch.Tensor, **kwargs) -> torch.Tensor:
        input_embeddings = self.tok_embeddings(input_ids=input_ids, **kwargs)
        if PM.size_of("CP") > 1:
            input_embeddings = scatter_to_balanced_cp_region(input_embeddings)
        return input_embeddings

    def forward(self, *args, **kwargs):
        kwargs = self._prepare_inputs(kwargs)
        return super().forward(*args, **kwargs)

    def abstract_forward(self, *args, **kwargs):
        kwargs = self._prepare_inputs(kwargs)
        return super().abstract_forward(*args, **kwargs)

    def build_reshaper(self):
        from steptronoss.checkpointing.reshape_ops import OnlineReshaper

        base_reshaper = super().build_reshaper()
        scripts = self._build_multimodal_reshape_scripts() + base_reshaper.scripts
        return OnlineReshaper(scripts)
