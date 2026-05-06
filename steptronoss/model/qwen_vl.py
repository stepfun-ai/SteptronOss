from __future__ import annotations

from steptronoss.model.common.image_insert_decoder import ImageInsertDecoderMixin
from steptronoss.model.qwen_dense import QwenModel


class QwenImageInsertModel(ImageInsertDecoderMixin, QwenModel):
    """Qwen decoder with a decoupled vision encoder and image-token insertion."""

    def build(self, layer_map):
        super().build(layer_map)
        self.build_multimodal_modules()
