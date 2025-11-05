"""RSMBatcher — 1.2 阶段采样 + 前缀裁切

把 1.1 NPZDataset 的“完整样本”按 **RSM（Render-by-Stage via contour accumulation）**
规则裁成“前缀样本”，供编码器做一次跨注意力、供解码器按前缀监督训练。

核心思想（与实施手册一致）：
- 按 **轮廓** 而不是 token 做阶段累加，避免把一条轮廓截成不闭合的碎片；
- 训练时为每个样本**随机采样一个阶段** p ∈ rsm_stages（如 25/50/75/100%），并按 rsm_probs 加权；
- 根据 p 计算要保留的轮廓数 K=ceil(p*C)，其中 C 为该样本的轮廓总数；
- 前缀 mask = (contour_ids < K)；将 `<END>` 放到前缀最后一个有效 token 上；
- 生成 arg_supervise_mask[L,4]：只在对应几何维度上回归坐标（M/L 两维；Q 四维；其余不回归）。

输出契约（单样本字典）：
{
  # —— 前缀后的“喂模型/算损失”张量 ——
  "seq_cmd": LongTensor[L],          # 已将 <END> 移到前缀末尾（原序列未返回）
  "seq_arg": FloatTensor[L,4],
  "seq_mask": BoolTensor[L],         # 前缀掩码（阶段外 False）
  "arg_supervise_mask": BoolTensor[L,4],
  "seq_topo": ByteTensor[L] | None,  # 如上游有则原样传下（不改，阶段外由 seq_mask 抑制）
  "contour_ids": LongTensor[L],      # 原样传下（用于日志/可视化）

  # —— 只读的原始整图/元信息（方便 1.3 渲阶段图、或可视化）——
  "img_dt_full": FloatTensor[1,H,W],
  "norm": FloatTensor[3],
  "meta": {"id": int, "unicode": int},

  # —— 阶段信息（日志/调试/可视化）——
  "stage_ratio": float,              # p ∈ rsm_stages
  "stage_id": int,                   # 百分比表示（25/50/75/100）或四舍五入整数
  "stage_K": int,                    # 本阶段保留的轮廓数
  "stage_C": int,                    # 总轮廓数
}

注意：
- 本模块**不**负责阶段 SDF 渲染；那是 1.3 StageRenderer 的职责。
- 若 .npz 缺少 contour_ids 或里面全是 -1，默认抛异常；如要容错可开启
  allow_token_prefix_fallback=True，但该模式只能按“token 比例”裁切，
  可能导致轮廓被截断 → 不建议与几何监督一起使用。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence
import math
import random

import torch
from torch.utils.data import Dataset


# 与数据构建器一致的命令枚举（保持索引一致）
CMD_PAD, CMD_M, CMD_L, CMD_Q, CMD_T, CMD_Z, CMD_NEW, CMD_HOLE, CMD_END = range(9)


# -------------------------
# 工具函数
# -------------------------

def _make_arg_supervise_mask(seq_cmd: torch.Tensor) -> torch.Tensor:
    """根据命令类型构建坐标监督遮蔽（不含阶段逻辑）。

    输入
    ----
    seq_cmd: LongTensor[L]

    返回
    ----
    BoolTensor[L,4]
      - M/L: 监督 (x,y) 两维 → [1,1,0,0]
      - Q  : 监督 (cx,cy,x,y) 四维 → [1,1,1,1]
      - 其他（Z/NEW/HOLE/END/PAD）: [0,0,0,0]
    """
    L = int(seq_cmd.shape[0])
    mask = torch.zeros((L, 4), dtype=torch.bool, device=seq_cmd.device)
    is_M = seq_cmd == CMD_M
    is_L = seq_cmd == CMD_L
    is_Q = seq_cmd == CMD_Q
    mask[is_M | is_L, 0:2] = True
    mask[is_Q, :] = True
    return mask


def _place_end_at_prefix_tail(seq_cmd: torch.Tensor,
                              seq_arg: torch.Tensor,
                              prefix_mask: torch.Tensor) -> None:
    """将 `<END>` 放到前缀最后一个有效 token 上，并在该处将参数清零。

    说明：就地修改 `seq_cmd/seq_arg`；不返回。
    前提：prefix_mask 至少有一个 True。
    """
    # 最后一个 True 的位置
    last = int(torch.nonzero(prefix_mask, as_tuple=False)[-1])
    seq_cmd[last] = CMD_END
    seq_arg[last].zero_()  # END 不回归坐标


# -------------------------
# 数据集包装器（每次 __getitem__ 时随机采一个阶段）
# -------------------------

@dataclass
class RSMConfig:
    rsm_stages: Sequence[float] = (0.25, 0.50, 0.75, 1.00)  # 按轮廓比例
    rsm_probs: Sequence[float] = (0.15, 0.25, 0.25, 0.35)   # 阶段采样权重（和为 1）
    enforce_contour_ids: bool = True                         # 缺失/全 -1 时是否报错
    allow_token_prefix_fallback: bool = False                # True: 允许按 token 比例裁切（不推荐）

    def __post_init__(self):
        # 基本合法性校验
        if len(self.rsm_stages) != len(self.rsm_probs):
            raise ValueError("rsm_stages and rsm_probs must have the same length")
        if any(p <= 0 for p in self.rsm_probs) or not math.isclose(sum(self.rsm_probs), 1.0, rel_tol=1e-6):
            raise ValueError("rsm_probs must be positive and sum to 1")
        if sorted(self.rsm_stages) != list(self.rsm_stages) or self.rsm_stages[-1] != 1.0:
            raise ValueError("rsm_stages must be strictly non-decreasing and end with 1.0")


class RSMBatcher(Dataset):
    """把“完整样本数据集”包装成“每次取样随机阶段前缀”的数据集。

    典型用法：
        base = NPZDataset("data/font.npz")
        rsm  = RSMBatcher(base, RSMConfig())
        loader = DataLoader(rsm, batch_size=..., shuffle=True, ...)

    说明：
    - 本类不会改变样本数量 N；只是对每条样本在每次访问时，随机选择一个阶段并裁出前缀；
    - 为了可复现，推荐在 DataLoader(worker_init_fn) 或训练循环里设置随机种子。
    """

    def __init__(self, base: Dataset, cfg: Optional[RSMConfig] = None) -> None:
        super().__init__()
        self.base = base
        self.cfg = cfg or RSMConfig()

    # 直传底层长度/属性（若底层是 NPZDataset，会有 N/L/H/W 等属性）
    def __len__(self):
        return len(self.base)

    def __getattr__(self, name):  # 便于直接访问 base.N/base.L 等
        try:
            return getattr(self.base, name)
        except AttributeError:
            raise

    # 每次取样时随机阶段 + 前缀裁切
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.base[idx]
        device = item["seq_cmd"].device

        seq_cmd: torch.Tensor = item["seq_cmd"].clone()      # [L]
        seq_arg: torch.Tensor = item["seq_arg"].clone()      # [L,4]
        seq_mask_full: torch.Tensor = item["seq_mask"].clone()  # [L]
        seq_topo = item.get("seq_topo", None)
        contour_ids: torch.Tensor = item["contour_ids"].clone()

        # 1) 统计轮廓数 C（要求 contour_ids >= 0 的数量 > 0）
        has_cids = (contour_ids >= 0).any().item()
        if not has_cids:
            if self.cfg.enforce_contour_ids and not self.cfg.allow_token_prefix_fallback:
                raise ValueError("RSMBatcher: contour_ids missing (-1 for all tokens). "
                                 "Provide contour_ids in .npz or enable token-prefix fallback.")

        # 2) 随机选择阶段 p
        stage_idx = random.choices(range(len(self.cfg.rsm_stages)), weights=self.cfg.rsm_probs, k=1)[0]
        p = float(self.cfg.rsm_stages[stage_idx])
        stage_id = int(round(p * 100))

        # 3) 计算前缀 mask
        if has_cids:
            C = int(contour_ids.max().item() + 1)
            K = max(1, math.ceil(p * C))  # 至少保留 1 条轮廓
            prefix_mask = (contour_ids < K) & seq_mask_full
        else:
            # 不推荐：按 token 比例裁切（保持 mask 前缀 True，之后 False）
            L = int(seq_mask_full.shape[0])
            valid_idx = torch.nonzero(seq_mask_full, as_tuple=False).flatten()
            if valid_idx.numel() == 0:
                raise ValueError("empty valid tokens in seq_mask")
            n_valid = int(valid_idx.numel())
            n_keep = max(1, math.ceil(p * n_valid))
            cutoff = int(valid_idx[n_keep - 1])
            prefix_mask = torch.zeros_like(seq_mask_full)
            prefix_mask[valid_idx[:n_keep]] = True
            C, K = -1, -1  # 无法定义

        if not prefix_mask.any():
            raise RuntimeError("prefix_mask is empty — check contour_ids or rsm_stages settings")

        # 4) 将 <END> 放到前缀尾部；并生成 arg_supervise_mask
        _place_end_at_prefix_tail(seq_cmd, seq_arg, prefix_mask)
        arg_mask = _make_arg_supervise_mask(seq_cmd)

        # 5) 输出：用前缀掩码覆盖 seq_mask；其它字段原样传下
        out: Dict[str, torch.Tensor] = {
            "seq_cmd": seq_cmd,
            "seq_arg": seq_arg,
            "seq_mask": prefix_mask,            # —— 注意：这里就是“前缀后的” mask ——
            "arg_supervise_mask": arg_mask,     # 逐维遮蔽（结合 seq_mask 才参与损失）
            "seq_topo": seq_topo,
            "contour_ids": contour_ids,
            "img_dt_full": item["img_dt_full"],
            "norm": item["norm"],
            "meta": item["meta"],
            "stage_ratio": torch.tensor(p, dtype=torch.float32, device=device),
            "stage_id": torch.tensor(stage_id, dtype=torch.int32, device=device),
            "stage_K": torch.tensor(K, dtype=torch.int32, device=device),
            "stage_C": torch.tensor(C, dtype=torch.int32, device=device),
        }
        return out


# -------------------------
# 便捷 CLI：快速查看裁切效果
# -------------------------
if __name__ == "__main__":
    import argparse
    from dataset import NPZDataset

    ap = argparse.ArgumentParser(description="RSMBatcher quick view: stage sampling + prefix cut")
    ap.add_argument("--npz", default='/data/kssczt/ztb/FontVector/VAE/data/msyh.npz', help="path to dataset .npz")
    ap.add_argument("--n", type=int, default=3, help="print first N samples after RSM")
    ap.add_argument("--stages", type=str, default="0.25,0.5,0.75,1.0", help="comma list of stage ratios")
    ap.add_argument("--probs", type=str, default="0.15,0.25,0.25,0.35", help="comma list of sampling probs (sum=1)")
    ap.add_argument("--fallback", action="store_true", help="allow token-prefix fallback when contour_ids missing")
    args = ap.parse_args()

    stages = tuple(float(x) for x in args.stages.split(','))
    probs = tuple(float(x) for x in args.probs.split(','))

    base = NPZDataset(args.npz, mmap=True, use_seq_topo=True, strict_shapes=True)
    cfg = RSMConfig(rsm_stages=stages, rsm_probs=probs, allow_token_prefix_fallback=args.fallback)
    rsm = RSMBatcher(base, cfg)

    print(f"Loaded {base.npz_path}")
    print(f"N={len(rsm)}, L={rsm.L}, H={rsm.H}, W={rsm.W}")
    for i in range(min(args.n, len(rsm))):
        s = rsm[i]
        n_valid = int(s["seq_mask"].sum())
        print(f"#{i}: stage={int(s['stage_id'])}%  C={int(s['stage_C'])}  K={int(s['stage_K'])}  valid_tokens={n_valid}")
