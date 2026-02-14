from dataclasses import dataclass

import torch
from loguru import logger

from steptronoss.core.tensor_parallel.mappings import (
    scatter_to_sequence_parallel_region,
)
from steptronoss.exp.abstract import ModelConfig as NaiveModelConfig
from steptronoss.model.common.parallel_embedding import (
    InputEmbeddingConfig,
    WordEmbedding,
)
from steptronoss.utils.memory_tracker import CMT
from steptronoss.utils.utils import patch_scatter


@dataclass
class ImageForInsert:
    insert_start_token: int
    images: torch.FloatTensor | None = None
    """Tensor[N, 3, L, L]"""

    image_features: list[torch.FloatTensor] | None = None
    """Tensor[N, S, C], if this is passed, ignore 'images'"""

    rope_cu_seqlens: torch.IntTensor | None = None
    rope_max_seq_len: int | None = None


class WithEncoderInputEmbeddingConfig(InputEmbeddingConfig):
    encoder_cfg: NaiveModelConfig

    encoder_no_grad: bool = False

    def build_adapter(self):
        pass

    def sanity_check(self):
        super().sanity_check()

        if getattr(self.encoder_cfg, "sequence_parallel", False):
            assert self.tp_cfg.sequence_parallel, "encoder sp requires decoder sp to be enabled"


class StepEncoderInputEmbedding(WordEmbedding):
    def __init__(self, cfg: WithEncoderInputEmbeddingConfig):
        super().__init__(cfg)
        self.cfg = cfg

        # Disable SP here so we get text_features un-splited
        self.actual_sequence_parallel = self.sequence_parallel
        self.sequence_parallel = False

        self.encoder = cfg.encoder_cfg.build_model().to(self.word_embeddings.weight.device)

        self.align_projector = cfg.build_adapter().to(self.word_embeddings.weight.device)

    @staticmethod
    def insert_features(input_embeddings, image_features, input_ids, flag):
        """Insert image features into token embeddings at the appropriate locations."""

        if not image_features.unbind(0):
            return input_embeddings

        insert_location = torch.nonzero(input_ids == flag)
        insert_location[:, 1] += 1  # Shift to the right

        # Check if the number of image features matches the number of insert locations
        num_image_features = image_features.shape[0]
        num_insert_locations = insert_location.shape[0]
        if num_image_features != num_insert_locations:
            logger.warning(
                f"Mismatch between image features and insert locations: "
                f"image_features.shape={image_features.shape}, "
                f"insert_location.shape={insert_location.shape}. "
                f"Returning original input_embeddings."
            )
            return input_embeddings

        last_pos = insert_location[:, 1].max()
        if last_pos + image_features.shape[1] == input_embeddings.shape[0] + 1:
            image_features = image_features[:-1]
            insert_location = insert_location[:-1]
            if len(insert_location) == 0:
                return input_embeddings
        else:
            if last_pos + image_features.shape[1] > input_embeddings.shape[0]:
                logger.warning(
                    f"Image features exceed input_embeddings.shape[0]: "
                    f"last_pos={last_pos}, image_features.shape[1]={image_features.shape[1]}, "
                    f"input_embeddings.shape[0]={input_embeddings.shape[0]}. "
                    f"Returning original input_embeddings."
                )
                return input_embeddings  # TODO: remove this
            assert last_pos + image_features.shape[1] <= input_embeddings.shape[0], (
                insert_location,
                image_features.shape,
            )

        # Use the patch_scatter function to insert image features
        return patch_scatter(input_embeddings, image_features.contiguous(), insert_location.contiguous())

    def handle_parallelization(self, input_embeddings):
        """Handle context and sequence parallelism based on configurations."""
        if self.actual_sequence_parallel:
            return scatter_to_sequence_parallel_region(input_embeddings)
        else:
            return input_embeddings

    def forward(self, input_ids, images: list[ImageForInsert], **kwargs):
        input_embeddings = super().forward(input_ids, **kwargs)
        CMT.mark("after_text_embedding")
        for insert_image in images:
            if insert_image.image_features is not None:
                image_features = insert_image.image_features
            else:
                with torch.set_grad_enabled(not self.cfg.encoder_no_grad):
                    images_list = insert_image.images
                    image_features = self.encoder(images_list)
            image_features = self.align_projector(image_features)

            CMT.mark(f"after_get_img_feature [{insert_image.insert_start_token}]")

            input_embeddings = self.insert_features(
                input_embeddings=input_embeddings,
                image_features=image_features,
                input_ids=input_ids,
                flag=insert_image.insert_start_token,
            )

            CMT.mark(f"after_insert_feature [{insert_image.insert_start_token}]")
        input_embeddings = self.handle_parallelization(input_embeddings)
        return input_embeddings
