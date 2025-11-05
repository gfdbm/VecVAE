# -*- coding: utf-8 -*-
"""
3.1 VAE 损失模块（开箱即用）
- 命令交叉熵 CE（仅在前缀位）
- 坐标 L1（仅在需要的坐标位）
- KL to N(0,1)（支持 β 预热 & free-bits）
- 可选命令类别权重（缓解 NEW/HOLE/Z/END 类别不平衡）

用法（见文件末尾 main 自检）：
python -m src.train.losses --npz data/your_font.npz --use-rsm
"""

from dataclasses import dataclass
from typing import Optional, Dict, Tuple

import torch
import torch.nn.functional as F
import torch.nn as nn

# 1.4 的掩码构建
from src.data.masks import build_loss_masks

# 2.4/2.5 编码器/解码器（仅 main 自检需要；你的训练脚本会自行前向拿到这些张量）
from src.model.encoder import VpVaeEncoder, VpVaeEncoderConfig, kl_normal_standard as _kl_ref
from src.model.decoder import Decoder, DecoderConfig


# --------- 工具：按当前 batch 估计类别权重（可选） ---------
def compute_class_weight_from_targets(y: torch.Tensor, num_classes: int = 9, eps: float = 1e-3) -> torch.Tensor:
    """
    y: [N] 仅包含有效（前缀）位置上的命令 id
    返回: [num_classes]，均值标准化到 ~1.0
    """
    device = y.device
    count = torch.bincount(y, minlength=num_classes).float()  # [V]
    inv = 1.0 / (count + eps)
    w = inv * (num_classes / inv.sum().clamp_min(1e-8))
    return w.to(device)


# --------- KL（可与 2.4 的保持一致，这里内联一个，避免循环依赖） ---------
def kl_normal_standard(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """逐维 KL（不做聚合）；shape=[B,Z]"""
    return -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())


@dataclass
class LossConfig:
    beta: float = 1.0             # KL 系数（建议训练时用预热调度器）
    free_bits: float = 0.0        # 每维 KL 的下限（nats），如 0.5 可鼓励用上维度
    use_class_weight: bool = False  # 是否对 CE 使用类别权重（缓解不平衡）
    num_classes: int = 9
    l1_weight: float = 10.0       # L1 损失权重（放大坐标回归的重要性）
    ce_weight: float = 1.0        # CE 损失权重


def compute_vae_losses(
    logits_cmd: torch.Tensor,   # [B,L,V]
    pred_arg: torch.Tensor,     # [B,L,4]
    mu: torch.Tensor,           # [B,Z]
    logvar: torch.Tensor,       # [B,Z]
    gt_seq_cmd: torch.Tensor,   # [B,L]
    gt_seq_arg: torch.Tensor,   # [B,L,4]
    seq_mask: torch.Tensor,     # [B,L] True=前缀（有效）
    cfg: LossConfig = LossConfig(),
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    返回: total_loss, stats（含 ce/l1/kl、计数信息）
    """
    # --- 取掩码（1.4） ---
    # lmc: [B,L]   —— 命令 CE 的有效位
    # lma: [B,L,4] —— 坐标 L1 的有效位（自动区分 M/L/Q 的不同数据位）
    # apm: [B,L]   —— arg-position mask（给你参考；本函数没用到）
    lmc, lma, apm = build_loss_masks(gt_seq_cmd, seq_mask)

    stats: Dict[str, torch.Tensor] = {}

    # --- 命令 CE ---
    if lmc.any():
        if cfg.use_class_weight:
            # 在有效位上统计类别频次以估计权重
            class_weight = compute_class_weight_from_targets(gt_seq_cmd[lmc], num_classes=cfg.num_classes)
        else:
            class_weight = None
        ce = F.cross_entropy(logits_cmd[lmc], gt_seq_cmd[lmc], weight=class_weight, reduction="mean")
    else:
        ce = torch.zeros((), device=logits_cmd.device)

    # --- 坐标 L1 ---
    if lma.any():
        l1 = (pred_arg[lma] - gt_seq_arg[lma]).abs().mean()
    else:
        l1 = torch.zeros((), device=pred_arg.device)

    # --- KL（带 free-bits）---
    # 原始逐维 KL： [B,Z]
    kl_per_dim = kl_normal_standard(mu, logvar)
    if cfg.free_bits > 0.0:
        # 先对 batch 求均值 → [Z]，再逐维应用下限，再求和
        kl_dim_mean = kl_per_dim.mean(dim=0)                 # [Z]
        kl_dim_clamped = torch.clamp(kl_dim_mean, min=cfg.free_bits)
        kl = kl_dim_clamped.sum()
    else:
        # 标准 batch 聚合：先对 Z 求和，再对 batch 求均值
        kl = kl_per_dim.sum(dim=-1).mean()

    # 应用损失权重
    total = cfg.ce_weight * ce + cfg.l1_weight * l1 + cfg.beta * kl

    # 统计信息
    stats["loss_total"] = total.detach()
    stats["loss_ce"] = ce.detach()
    stats["loss_l1"] = l1.detach()
    stats["loss_kl"] = kl.detach()
    stats["n_ce"] = lmc.sum().detach()
    stats["n_l1"] = lma.sum().detach()
    stats["beta"] = torch.tensor(cfg.beta, device=total.device)
    stats["free_bits"] = torch.tensor(cfg.free_bits, device=total.device)

    return total, stats


# --------- β 预热调度器（简单线性） ---------
class BetaWarmup:
    """
    β 从 0 线性升到 target（常用 target=1.0）。按 step 更新。
    """
    def __init__(self, warmup_steps: int, target: float = 1.0):
        self.warmup_steps = max(1, int(warmup_steps))
        self.target = float(target)
        self.step_idx = 0

    def step(self) -> float:
        self.step_idx += 1
        t = min(1.0, self.step_idx / self.warmup_steps)
        return t * self.target


# ------------------ 自检 main：跑一个 batch 打印各项损失 ------------------
if __name__ == "__main__":
    import argparse, sys
    from pathlib import Path

    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    # 数据与工具
    from src.data.dataset import NPZDataset
    from src.data.rsm_batcher import RSMBatcher, RSMConfig
    from src.data.stage_renderer import StageRenderer, StageRendererConfig

    ap = argparse.ArgumentParser(description="3.1 Loss quick test (CE/L1/KL with masks)")
    ap.add_argument("--npz", default='/data/kssczt/ztb/FontVector/VAE/data/msyh.npz', help="path to dataset .npz")
    ap.add_argument("--use-rsm", default=True, help="use RSM stage prefix")
    ap.add_argument("--embed", type=int, default=256)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--xlayers", type=int, default=1)
    ap.add_argument("--zdim", type=int, default=128)
    ap.add_argument("--patch", type=int, default=16)
    ap.add_argument("--dec-layers", type=int, default=4)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--img-size", type=int, default=None)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--free-bits", type=float, default=0.0)
    ap.add_argument("--class-weight", choices=["none","auto"], default="auto")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)

    # === 数据/RSM ===
    base = NPZDataset(args.npz, mmap=True, use_seq_topo=True, strict_shapes=True)
    if args.use_rsm:
        ds = RSMBatcher(base, RSMConfig(
            rsm_stages=(0.25, 0.5, 0.75, 1.0),
            rsm_probs=(0.15, 0.25, 0.25, 0.35),
            enforce_contour_ids=True,
            allow_token_prefix_fallback=False,
        ))
        print("[info] RSM on (stage prefix)")
    else:
        ds = base
        print("[info] FULL stage (no RSM)")

    renderer = StageRenderer(StageRendererConfig(
        img_size=args.img_size or base.H, sdf_clip_px=8.0, out_dtype=torch.float32,
    ))

    # === 取一个 batch ===
    B = min(args.batch, len(ds))
    items = [ds[i] for i in range(B)]
    def cat(k): return torch.stack([it[k] for it in items], 0).to(device)

    seq_cmd     = cat("seq_cmd")
    seq_arg     = cat("seq_arg")
    seq_mask    = cat("seq_mask")
    contour_ids = cat("contour_ids")
    seq_topo    = cat("seq_topo")
    stage_sdf   = torch.stack([renderer.render_item(it) for it in items], 0).to(device)

    # === 编码器 → z ===
    enc = VpVaeEncoder(VpVaeEncoderConfig(
        embed_dim=args.embed, num_heads=args.heads, cross_layers=args.xlayers,
        patch_size=args.patch, z_dim=args.zdim, use_prefix_repr=True, dropout=0.0
    )).to(device)
    enc.eval()
    with torch.no_grad():
        mu, logvar, z, aux_enc = enc(seq_cmd, seq_arg, seq_mask, contour_ids, seq_topo, stage_sdf,
                                     sample=True, eps_std=1.0)

    # === 解码器 → 预测 ===
    dec = Decoder(DecoderConfig(
        vocab_size=9, max_len=base.L, embed_dim=args.embed, z_dim=args.zdim,
        n_heads=args.heads, n_layers=args.dec_layers, patch_size=args.patch, use_pixel_cross_attn=True
    )).to(device)
    dec.eval()
    with torch.no_grad():
        logits_cmd, pred_arg, aux_dec = dec(z, stage_sdf, seq_mask)

    # === 计算损失 ===
    loss_cfg = LossConfig(
        beta=args.beta,
        free_bits=args.free_bits,
        use_class_weight=(args.class_weight == "auto"),
        num_classes=9
    )
    total, stats = compute_vae_losses(
        logits_cmd, pred_arg, mu, logvar, seq_cmd, seq_arg, seq_mask, cfg=loss_cfg
    )

    # === 打印 ===
    print(f"logits_cmd: {tuple(logits_cmd.shape)}  pred_arg: {tuple(pred_arg.shape)}")
    print(f"mu/logσ:    {tuple(mu.shape)} / {tuple(logvar.shape)}")
    print(f"[loss] total={float(stats['loss_total']):.4f}  CE={float(stats['loss_ce']):.4f}  "
          f"L1={float(stats['loss_l1']):.4f}  KL={float(stats['loss_kl']):.4f}  "
          f"(beta={float(stats['beta'])}, free_bits={float(stats['free_bits'])})")
    print(f"[counts] n_CE={int(stats['n_ce'])}  n_L1={int(stats['n_l1'])}")
