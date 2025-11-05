# -*- coding: utf-8 -*-
"""损失子包：CE/L1/KL、β 预热等。"""
from typing import Any

LossConfig: Any = None
compute_vae_losses: Any = None
BetaWarmup: Any = None

try:
    from .losses import LossConfig, compute_vae_losses, BetaWarmup
except Exception:
    pass

__all__ = ["LossConfig", "compute_vae_losses", "BetaWarmup"]
