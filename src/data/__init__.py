# -*- coding: utf-8 -*-
"""数据子包：数据集/RSM/渲染/损失掩码。提供便捷导出（失败时静默）。"""
from typing import Any

NPZDataset: Any = None
RSMBatcher: Any = None
RSMConfig: Any = None
StageRenderer: Any = None
StageRendererConfig: Any = None
build_loss_masks: Any = None

try:
    from .dataset import NPZDataset
except Exception:
    pass
try:
    from .rsm_batcher import RSMBatcher, RSMConfig
except Exception:
    pass
try:
    from .stage_renderer import StageRenderer, StageRendererConfig
except Exception:
    pass
try:
    from .masks import build_loss_masks
except Exception:
    pass

__all__ = [
    "NPZDataset",
    "RSMBatcher", "RSMConfig",
    "StageRenderer", "StageRendererConfig",
    "build_loss_masks",
]
