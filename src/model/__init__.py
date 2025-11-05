# -*- coding: utf-8 -*-
"""模型子包：矢量前缀编码器/像素编码器/跨注意力/总装编码器/解码器。"""
from typing import Any

VectorPrefixEncoder: Any = None
VectorPrefixEncoderConfig: Any = None
PixelEncoder: Any = None
PixelEncoderConfig: Any = None
CrossAttentionAdapter: Any = None
CrossAttentionConfig: Any = None
VpVaeEncoder: Any = None
VpVaeEncoderConfig: Any = None
Decoder: Any = None
DecoderConfig: Any = None

# 逐个尝试导入，避免开发期循环导入导致报错
try:
    from .vector_encoder import VectorPrefixEncoder, VectorPrefixEncoderConfig
except Exception:
    pass
try:
    from .pixel_encoder import PixelEncoder, PixelEncoderConfig
except Exception:
    pass
try:
    from .cross_attention import CrossAttentionAdapter, CrossAttentionConfig
except Exception:
    pass
try:
    from .encoder import VpVaeEncoder, VpVaeEncoderConfig
except Exception:
    pass
try:
    from .decoder import Decoder, DecoderConfig
except Exception:
    pass

__all__ = [
    "VectorPrefixEncoder", "VectorPrefixEncoderConfig",
    "PixelEncoder", "PixelEncoderConfig",
    "CrossAttentionAdapter", "CrossAttentionConfig",
    "VpVaeEncoder", "VpVaeEncoderConfig",
    "Decoder", "DecoderConfig",
]
