#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
1.4 Supervision Masks (masks.py)
================================

目的：把“该学什么/不该学什么”的规则标准化，供训练各处复用。
- 结合 RSM 的前缀掩码 `seq_mask`，与命令类型 `seq_cmd`，生成：
  1) `loss_mask_cmd   [B,L]`   —— 命令交叉熵的监督位置（=前缀内）。
  2) `loss_mask_arg   [B,L,4]` —— 坐标回归的监督维度（M/L→xy 两维；Q→四维；其余命令不回归）。
  3) `attn_pad_mask   [B,L]`   —— 注意力模块用的 key_padding_mask（= 前缀外 True）。
- 提供若干一致性检查（`<END>` 在前缀尾且前缀内唯一等），便于在训练前/中快速断言。

与 1.1/1.2/1.3 的契合：
- 1.2(RSM) 已经把 `<END>` 移到前缀尾，并生成 `seq_mask`；本文件只做“监督/注意力遮蔽”的统一产出。
- 形状契约沿用 1.1：`seq_cmd[L] long`, `seq_arg[L,4] float`, `seq_mask[L] bool`；支持 batch 版本。

注意：你的数据不会出现 CMD_T；本实现只处理 M/L/Q/Z/NEW/HOLE/END/PAD。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch

# 与数据构建器一致的命令编号
CMD_PAD, CMD_M, CMD_L, CMD_Q, CMD_T, CMD_Z, CMD_NEW, CMD_HOLE, CMD_END = range(9)


# -----------------------------
# 核心：根据命令类型 + 前缀掩码，产出三类遮蔽
# -----------------------------

def build_loss_masks(seq_cmd: torch.Tensor,
                     seq_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """生成三类遮蔽：命令 CE、坐标 L1、注意力 pad。

    参数：
      seq_cmd:  [B,L] 或 [L]，long
      seq_mask: [B,L] 或 [L]，bool（RSM 前缀掩码；前缀内 True）
    返回：
      loss_mask_cmd: [B,L]   bool —— 命令 CE 的监督位置（= seq_mask）
      loss_mask_arg: [B,L,4] bool —— 坐标回归监督维度（结合命令类型）
      attn_pad_mask: [B,L]   bool —— 注意力用的 key_padding_mask（= ~seq_mask）
    """
    # 统一 batch 维
    if seq_cmd.dim() == 1:
        seq_cmd = seq_cmd.unsqueeze(0)
    if seq_mask.dim() == 1:
        seq_mask = seq_mask.unsqueeze(0)

    B, L = seq_cmd.shape
    device = seq_cmd.device

    # CE：前缀内才监督
    loss_mask_cmd = seq_mask.bool()

    # Arg：按命令维度监督（避免高级索引导致的维度错位）
    is_M = (seq_cmd == CMD_M)
    is_L = (seq_cmd == CMD_L)
    is_Q = (seq_cmd == CMD_Q)

    prefix = seq_mask.bool()
    mask_ml = prefix & (is_M | is_L)   # M/L 的 (x,y)
    mask_q  = prefix & is_Q            # Q   的 (cx,cy,x,y)

    # 逐通道构造 [B,L,4] 的监督维度掩码
    loss_mask_arg = torch.zeros((B, L, 4), dtype=torch.bool, device=device)
    loss_mask_arg[:, :, 0] = mask_ml | mask_q  # x
    loss_mask_arg[:, :, 1] = mask_ml | mask_q  # y
    loss_mask_arg[:, :, 2] = mask_q            # cx
    loss_mask_arg[:, :, 3] = mask_q            # cy

    # 注意力 pad（key_padding_mask=True 表示需要屏蔽）
    attn_pad_mask = (~seq_mask.bool())

    return loss_mask_cmd, loss_mask_arg, attn_pad_mask


# -----------------------------
# 断言/一致性检查：<END> 位置与数量（可选）
# -----------------------------

def check_prefix_end_invariants(seq_cmd: torch.Tensor,
                                seq_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
    """检查：前缀内 END 是否唯一、且位于前缀末尾。
    返回一个字典，包含：
      - ok:            [B] bool
      - n_end_prefix:  [B] long（前缀内 END 的数量，理想为 1）
      - idx_last:      [B] long（前缀内最后一个 True 的位置；若无前缀为 -1）
      - idx_end:       [B] long（前缀内 END 的位置；若不存在为 -1）
    """
    if seq_cmd.dim() == 1:
        seq_cmd = seq_cmd.unsqueeze(0)
    if seq_mask.dim() == 1:
        seq_mask = seq_mask.unsqueeze(0)

    B, L = seq_cmd.shape
    device = seq_cmd.device

    prefix = seq_mask.bool()
    is_end = (seq_cmd == CMD_END)

    # 索引网格 [B,L]
    idxs = torch.arange(L, device=device).unsqueeze(0).expand(B, -1)

    # 前缀最后一个位置（若无前缀则 -1）
    masked_idx = torch.where(prefix, idxs, torch.full_like(idxs, -1))
    idx_last = masked_idx.max(dim=-1).values  # [B]

    # 前缀内 END 的数量与位置
    end_in_prefix = is_end & prefix
    n_end_prefix = end_in_prefix.sum(dim=-1)  # [B]
    masked_end_idx = torch.where(end_in_prefix, idxs, torch.full_like(idxs, -1))
    idx_end = masked_end_idx.max(dim=-1).values  # [B]

    ok = (n_end_prefix == 1) & (idx_end == idx_last) & (idx_last >= 0)

    return {
        "ok": ok,
        "n_end_prefix": n_end_prefix.to(torch.long),
        "idx_last": idx_last.to(torch.long),
        "idx_end": idx_end.to(torch.long),
    }


# -----------------------------
# 简易 CLI：加载 1.1/1.2，打印遮蔽统计与 END 断言
# -----------------------------
if __name__ == "__main__":
    import argparse

    try:
        from .dataset import NPZDataset  # type: ignore
        from .rsm_batcher import RSMBatcher, RSMConfig  # type: ignore
    except Exception:
        from dataset import NPZDataset  # type: ignore
        from rsm_batcher import RSMBatcher, RSMConfig  # type: ignore

    ap = argparse.ArgumentParser(description="1.4 masks quick check: loss masks & END invariants")
    ap.add_argument("--npz", default='/data/kssczt/ztb/FontVector/VAE/data/msyh.npz', help="path to dataset .npz")
    ap.add_argument("--n", type=int, default=4, help="print first N samples")
    ap.add_argument("--use-rsm", default=True, help="use RSM stage prefix (else full stage)")
    args = ap.parse_args()

    base = NPZDataset(args.npz, mmap=True, use_seq_topo=True, strict_shapes=True)
    if args.use_rsm:
        ds = RSMBatcher(base, RSMConfig(
            rsm_stages=(0.25,0.5,0.75,1.0), rsm_probs=(0.15,0.25,0.25,0.35),
            enforce_contour_ids=True, allow_token_prefix_fallback=False
        ))
        print("[info] RSM on (stage prefix)")
    else:
        ds = base
        print("[info] FULL stage (no RSM)")

    for i in range(min(args.n, len(ds))):
        it = ds[i]
        lmc, lma, apm = build_loss_masks(it["seq_cmd"], it["seq_mask"])  # [L], [L,4], [L]
        stats = check_prefix_end_invariants(it["seq_cmd"], it["seq_mask"])  # dict of [1]
        # 打印统计
        n_ce = int(lmc.sum().item())
        print(lmc)
        n_l1 = int(lma.sum().item())
        print(lma)
        ok = bool(stats["ok"].item())
        n_end = int(stats["n_end_prefix"].item())
        idx_last = int(stats["idx_last"].item())
        idx_end = int(stats["idx_end"].item())
        print(f"#{i}: CE_tokens={n_ce:4d}  L1_coords={n_l1:4d}  END_ok={ok}  (n_end_prefix={n_end}, idx_last={idx_last}, idx_end={idx_end})")
