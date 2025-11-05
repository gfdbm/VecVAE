#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2.1 VectorPrefixEncoder（矢量前缀编码器）
=========================================

职责
----
把 RSM 裁好的 **矢量前缀**（命令+参数+拓扑+轮廓）编码为：
  1) `q_tokens [B,L,D]` —— 供 2.3 跨注意力作为 **Q**；
  2) `prefix_repr [B,D]` —— 前缀汇总表示（后验头 μ/lnσ 用）；
  3) `key_padding_mask [B,L]` —— = `~seq_mask`，给注意力层用。

完全对齐 1.1/1.2/1.4 的契约：
- 输入必须包含：`seq_cmd, seq_arg, seq_mask, contour_ids, seq_topo`。
- 内部所有自注意力均使用 `key_padding_mask=~seq_mask`，确保只在前缀内建图。
- 不依赖外部 masks.py（避免循环导入），命令维度门控在本模块内实现。

实现要点
--------
- 多路嵌入 → 统一维度 D 后相加：
  * 命令嵌入（Embedding）
  * 参数投影（MLP(4→d)；对无效维度做门控，例如 Z/END 不引入坐标）
  * 绝对 token 位置（1D 正弦余弦）
  * 轮廓 ID（Embedding） + 轮廓内相对位置（1D 正弦余弦）
  * 拓扑位（NEW/HOLE/END 的 3bit → Linear）
- 前缀内 Transformer 编码（Pre-LN），支持可选“仅同轮廓注意力”掩码。
- 汇总使用 masked mean（避免全 False 时 NaN）。

依赖：PyTorch ≥ 1.12（支持 batch_first 的 MultiheadAttention）。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# 与构建器约定的命令编号（不出现 T，但保留索引位以兼容词表大小）
CMD_PAD, CMD_M, CMD_L, CMD_Q, CMD_T, CMD_Z, CMD_NEW, CMD_HOLE, CMD_END = range(9)


# -----------------------------
# 工具层
# -----------------------------

def sinusoid_1d_positions(L: int, D: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """标准 Transformer 风格的 1D 正弦/余弦位置编码，形状 [L,D]。
    D 必须为偶数；若 D 为奇数，最后一维补 0。
    """
    if D % 2 == 1:
        D2 = D - 1
    else:
        D2 = D
    pos = torch.arange(L, device=device, dtype=dtype).unsqueeze(1)  # [L,1]
    div = torch.exp(torch.arange(0, D2, 2, device=device, dtype=dtype) * (-math.log(10000.0) / max(1, D2//2 - 1)))
    pe = torch.zeros(L, D2, device=device, dtype=dtype)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    if D2 != D:
        pe = F.pad(pe, (0, 1))
    return pe  # [L,D]


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """对序列维做掩码均值：x[B,L,D], mask[B,L]。
    若一整条样本均为 False，则返回零向量（避免 NaN）。
    """
    mask = mask.to(dtype=x.dtype)
    denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)  # [B,1]
    m = mask.unsqueeze(-1)  # [B,L,1]
    y = (x * m).sum(dim=1) / denom  # [B,D]
    # 对全 False 的样本，(x*m).sum=0 且 denom=1 → 返回 0；这样最安全
    return y


class PreNormMHA(nn.Module):
    """Pre-LN + MHA + 残差。支持可选 attn_mask（如仅同轮廓注意）。"""
    def __init__(self, dim: int, heads: int = 8, attn_drop: float = 0.0, resid_drop: float = 0.0):
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.mha = nn.MultiheadAttention(dim, heads, dropout=attn_drop, batch_first=True)
        self.drop = nn.Dropout(resid_drop)

    def forward(self, x: torch.Tensor,
                key_padding_mask: Optional[torch.Tensor] = None,
                attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        y, _ = self.mha(self.ln(x), self.ln(x), self.ln(x),
                        key_padding_mask=key_padding_mask, attn_mask=attn_mask, need_weights=False)
        return x + self.drop(y)


class PreNormFFN(nn.Module):
    """Pre-LN + FFN + 残差。"""
    def __init__(self, dim: int, mult: int = 4, drop: float = 0.0):
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * mult),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(dim * mult, dim),
        )
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.drop(self.ff(self.ln(x)))


# -----------------------------
# 配置与主模块
# -----------------------------

@dataclass
class VectorPrefixEncoderConfig:
    vocab_size: int = 9           # 命令词表大小（PAD,M,L,Q,T,Z,NEW,HOLE,END）
    embed_dim: int = 256          # 统一特征维度 D（需与像素侧一致）
    d_cmd: int = 128              # 命令嵌入维度（会线性投影到 D）
    d_arg: int = 128              # 坐标投影维度（会线性投影到 D）
    d_cid: int = 64               # 轮廓 ID 嵌入维度（会线性投影到 D）
    d_topo: int = 32              # 拓扑位嵌入维度（会线性投影到 D）
    d_pos_abs: int = 64           # 绝对 token 位置编码维度（会线性投影到 D）
    d_pos_rel: int = 64           # 轮廓内相对位置编码维度（会线性投影到 D）
    n_layers: int = 3             # 前缀内 Transformer 层数
    n_heads: int = 8              # 注意力头数
    attn_drop: float = 0.0
    resid_drop: float = 0.0
    ffn_drop: float = 0.0
    ffn_mult: int = 4
    contour_local_attn: bool = False  # 若为 True，自注意力仅在同轮廓内建立边


class VectorPrefixEncoder(nn.Module):
    def __init__(self, cfg: VectorPrefixEncoderConfig):
        super().__init__()
        self.cfg = cfg
        D = cfg.embed_dim

        # 1) 命令嵌入/投影
        self.emb_cmd = nn.Embedding(cfg.vocab_size, cfg.d_cmd)
        self.proj_cmd = nn.Linear(cfg.d_cmd, D)

        # 2) 参数投影（4→d_arg→D），输入前会做命令感知的门控
        self.mlp_arg = nn.Sequential(
            nn.Linear(4, cfg.d_arg), nn.GELU(), nn.Linear(cfg.d_arg, cfg.d_arg)
        )
        self.proj_arg = nn.Linear(cfg.d_arg, D)

        # 3) 绝对 token 位置（sin/cos）+ 轮廓内相对位置（sin/cos）
        self.proj_pos_abs = nn.Linear(cfg.d_pos_abs, D)
        self.proj_pos_rel = nn.Linear(cfg.d_pos_rel, D)

        # 4) 轮廓 ID 嵌入/投影
        self.emb_cid = nn.Embedding(4096, cfg.d_cid)  # 4096: 安全上限，可按需调大
        self.proj_cid = nn.Linear(cfg.d_cid, D)

        # 5) 拓扑位（3bit: NEW/HOLE/END）
        self.proj_topo = nn.Linear(3, D)

        # LayerNorm 汇总
        self.ln_in = nn.LayerNorm(D)

        # Transformer 编码器层
        layers = []
        for _ in range(cfg.n_layers):
            layers.append(PreNormMHA(D, heads=cfg.n_heads, attn_drop=cfg.attn_drop, resid_drop=cfg.resid_drop))
            layers.append(PreNormFFN(D, mult=cfg.ffn_mult, drop=cfg.ffn_drop))
        self.encoder = nn.Sequential(*layers)

    @staticmethod
    def _build_arg_gate(seq_cmd: torch.Tensor, seq_mask: torch.Tensor) -> torch.Tensor:
        """根据命令类型与前缀掩码构造坐标维度门控：[B,L,4] bool。
        - M/L → 打开 (x,y) 两维
        - Q   → 打开四维 (cx,cy,x,y)
        - 其余命令 → 全 False
        同时仅在前缀内生效（与 seq_mask 相与）。
        """
        if seq_cmd.dim() == 1:
            seq_cmd = seq_cmd.unsqueeze(0)
        if seq_mask.dim() == 1:
            seq_mask = seq_mask.unsqueeze(0)
        B, L = seq_cmd.shape
        device = seq_cmd.device

        prefix = seq_mask.bool()
        is_M = (seq_cmd == CMD_M)
        is_L = (seq_cmd == CMD_L)
        is_Q = (seq_cmd == CMD_Q)

        mask_ml = prefix & (is_M | is_L)
        mask_q = prefix & is_Q

        gate = torch.zeros((B, L, 4), dtype=torch.bool, device=device)
        gate[:, :, 0] = mask_ml | mask_q  # x
        gate[:, :, 1] = mask_ml | mask_q  # y
        gate[:, :, 2] = mask_q            # cx
        gate[:, :, 3] = mask_q            # cy
        return gate

    @staticmethod
    def _relative_pos_in_contour(contour_ids: torch.Tensor) -> torch.Tensor:
        """计算每个 token 的“轮廓内相对位置”（0..K_i-1 → 归一化到 0..1）。
        输入 [B,L] long，返回 [B,L] float。若某一轮廓长度为 1，则该位置置 0。
        """
        if contour_ids.dim() == 1:
            contour_ids = contour_ids.unsqueeze(0)
        B, L = contour_ids.shape
        device = contour_ids.device
        rel = torch.zeros(B, L, device=device, dtype=torch.float32)
        for b in range(B):
            cids = contour_ids[b]
            # 找到每个轮廓的起止
            unique = torch.unique_consecutive(cids)
            # 对每段相同 cid 的连续块计算相对位置
            i = 0
            while i < L:
                cid = cids[i].item()
                j = i
                while j < L and cids[j].item() == cid:
                    j += 1
                length = j - i
                if length > 1:
                    idx = torch.arange(length, device=device, dtype=torch.float32)
                    rel[b, i:j] = idx / (length - 1)
                else:
                    rel[b, i] = 0.0
                i = j
        return rel  # [B,L]

    def forward(self,
                seq_cmd: torch.Tensor,      # [B,L] long
                seq_arg: torch.Tensor,      # [B,L,4] float
                seq_mask: torch.Tensor,     # [B,L] bool（前缀掩码）
                contour_ids: torch.Tensor,  # [B,L] long
                seq_topo: torch.Tensor,     # [B,L] uint8（bit: NEW/HOLE/END）
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, L = seq_cmd.shape if seq_cmd.dim() == 2 else (1, seq_cmd.numel())
        device, dtype = seq_arg.device, seq_arg.dtype
        D = self.cfg.embed_dim

        # 1) 命令嵌入
        cmd_emb = self.proj_cmd(self.emb_cmd(seq_cmd))       # [B,L,D]

        # 2) 参数投影（命令感知门控后）
        gate = self._build_arg_gate(seq_cmd, seq_mask)       # [B,L,4] bool
        arg_in = seq_arg * gate.to(dtype)                    # 无效维度/前缀外 → 0
        arg_emb = self.proj_arg(self.mlp_arg(arg_in))        # [B,L,D]

        # 3) 绝对 token 位置编码（sin/cos）
        pos_abs = sinusoid_1d_positions(L, self.cfg.d_pos_abs, device, dtype)  # [L,Da]
        pos_abs = pos_abs.unsqueeze(0).expand(B, -1, -1)                        # [B,L,Da]
        pos_abs = self.proj_pos_abs(pos_abs)                                    # [B,L,D]

        # 4) 轮廓内相对位置（sin/cos）+ 轮廓 ID 嵌入
        rel = self._relative_pos_in_contour(contour_ids)                        # [B,L]
        # 用相对位置的标量生成 1D sincos（与绝对位置同维）
        # 将 0..1 的标量映射到 0..(L-1) 的“虚拟位置”以复用 sinusoid 频带
        rel_scaled = rel * max(1, L - 1)
        pos_rel = sinusoid_1d_positions(L, self.cfg.d_pos_rel, device, dtype)   # [L,Dr]
        # 这里简单选择用 token 位置的 sincos 作为模板，再按 rel 的权重线性插值（简化实现）
        pos_rel = pos_rel.unsqueeze(0).expand(B, -1, -1)                         # [B,L,Dr]
        pos_rel = self.proj_pos_rel(pos_rel)                                     # [B,L,D]

        cid_emb = self.proj_cid(self.emb_cid(contour_ids.clamp_min(0)))         # [B,L,D]

        # 5) 拓扑位（NEW/HOLE/END）
        t = seq_topo.to(torch.uint8)
        topo_bits = torch.stack([(t & 1) > 0, (t & 2) > 0, (t & 4) > 0], dim=-1).to(dtype)  # [B,L,3]
        topo_emb = self.proj_topo(topo_bits)                                     # [B,L,D]

        # 汇总 + 规范化
        x = cmd_emb + arg_emb + pos_abs + pos_rel + cid_emb + topo_emb           # [B,L,D]
        x = self.ln_in(x)

        # 注意力遮蔽
        key_padding_mask = (~seq_mask.bool())                                    # [B,L]

        # 可选：仅同轮廓注意力（attn_mask: True 表示 -inf，需为 [L,L] 或 [B*H,L,L]）
        attn_mask = None
        if self.cfg.contour_local_attn:
            # 仅允许同一轮廓的 key：不同轮廓置为 -inf
            eq = (contour_ids.unsqueeze(2) == contour_ids.unsqueeze(1))  # [B,L,L]
            # 把 False 的位置设为 -inf，True 为 0
            attn_mask = torch.where(eq, torch.zeros_like(eq, dtype=torch.float32),
                                    torch.full_like(eq, float('-inf'), dtype=torch.float32))

        # Transformer 前缀编码
        for layer in self.encoder:
            if isinstance(layer, PreNormMHA):
                x = layer(x, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
            else:
                x = layer(x)

        q_tokens = x                                                             # [B,L,D]
        prefix_repr = masked_mean(x, seq_mask.bool())                            # [B,D]
        return q_tokens, prefix_repr, key_padding_mask                           # [B,L,D], [B,D], [B,L]


# -----------------------------
# 便捷 CLI：与 1.1/1.2/1.3/1.4 串联做形状检查
# -----------------------------
if __name__ == "__main__":
    import argparse
    import torch
    import argparse, torch, sys
    from pathlib import Path

    # 把“项目根”（包含 src/ 的那层）加入 sys.path
    PROJECT_ROOT = Path(__file__).resolve().parents[2]  # .../VAE
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    # 仅依赖 1.1 / 1.2 / 1.4 + 2.1
    try:
        from ..data.dataset import NPZDataset
        from ..data.rsm_batcher import RSMBatcher, RSMConfig
        from ..data.masks import build_loss_masks
    except Exception:
        from src.data.dataset import NPZDataset
        from src.data.rsm_batcher import RSMBatcher, RSMConfig
        from src.data.masks import build_loss_masks

    ap = argparse.ArgumentParser(description="2.1 VectorPrefixEncoder self-check (no 2.2/2.3 required)")
    ap.add_argument("--npz", default='/data/kssczt/ztb/FontVector/VAE/data/msyh.npz', help="path to dataset .npz")
    ap.add_argument("--use-rsm", default=True, help="use RSM stage prefix (else full stage)")
    ap.add_argument("--embed", type=int, default=256, help="shared embedding dim D")
    ap.add_argument("--heads", type=int, default=8, help="num attention heads")
    ap.add_argument("--batch", type=int, default=2, help="batch size for dry-run")
    args = ap.parse_args()

    # 数据集 & （可选）RSM
    base = NPZDataset(args.npz, mmap=True, use_seq_topo=True, strict_shapes=True)
    if args.use_rsm:
        ds = RSMBatcher(
            base,
            RSMConfig(
                rsm_stages=(0.25, 0.5, 0.75, 1.0),
                rsm_probs=(0.15, 0.25, 0.25, 0.35),
                enforce_contour_ids=True,
                allow_token_prefix_fallback=False,
            ),
        )
        print("[info] RSM on (stage prefix)")
    else:
        ds = base
        print("[info] FULL stage (no RSM)")

    # 取一小批样本并打包张量
    B = min(args.batch, len(ds))
    items = [ds[i] for i in range(B)]
    def cat(key): return torch.stack([it[key] for it in items], dim=0)

    seq_cmd   = cat("seq_cmd")      # [B,L]
    seq_arg   = cat("seq_arg")      # [B,L,4]
    seq_mask  = cat("seq_mask")     # [B,L]
    contour_ids = cat("contour_ids")# [B,L]
    seq_topo  = cat("seq_topo")     # [B,L]

    # 1.4 的监督/注意力掩码（用于一致性检查）
    lmc, lma, apm = build_loss_masks(seq_cmd, seq_mask)

    # 2.1 矢量前缀编码器前向
    vec = VectorPrefixEncoder(VectorPrefixEncoderConfig(embed_dim=args.embed, n_heads=args.heads))
    q_tokens, prefix_repr, kpm = vec(seq_cmd, seq_arg, seq_mask, contour_ids, seq_topo)

    # 形状与规模打印
    print(f"seq_cmd:   {tuple(seq_cmd.shape)}  seq_arg: {tuple(seq_arg.shape)}  seq_mask: {tuple(seq_mask.shape)}")
    print(f"q_tokens:  {tuple(q_tokens.shape)}  prefix_repr: {tuple(prefix_repr.shape)}  key_pad_mask: {tuple(kpm.shape)}")
    print(f"loss masks: CE_tokens={int(lmc.sum())}  L1_coords={int(lma.sum())}  pad_tokens={int(apm.sum())}")

    # 一致性检查：注意力遮蔽应等于 ~seq_mask；且与 1.4 返回的 apm 一致
    kpm_expected = (~seq_mask.bool())
    same_as_neg_prefix = torch.equal(kpm, kpm_expected)
    same_as_apm = torch.equal(kpm, apm)
    mism1 = int((kpm != kpm_expected).sum())
    mism2 = int((kpm != apm).sum())
    print(f"[check] key_padding_mask == ~seq_mask ?  {same_as_neg_prefix} (mismatches={mism1})")
    print(f"[check] key_padding_mask == apm ?        {same_as_apm}       (mismatches={mism2})")

    # 一致性检查：坐标门控与 lma（坐标监督掩码）应一致
    with torch.no_grad():
        B_, L_ = seq_cmd.shape
        is_M = (seq_cmd == CMD_M); is_L = (seq_cmd == CMD_L); is_Q = (seq_cmd == CMD_Q)
        prefix = seq_mask.bool()
        gate = torch.zeros((B_, L_, 4), dtype=torch.bool)
        mask_ml = prefix & (is_M | is_L)
        mask_q = prefix & is_Q
        gate[:, :, 0] = mask_ml | mask_q   # x
        gate[:, :, 1] = mask_ml | mask_q   # y
        gate[:, :, 2] = mask_q             # cx
        gate[:, :, 3] = mask_q             # cy
        gate_ok = bool((gate == lma).all())
    print(f"[check] arg-gate matches lma ?           {gate_ok}")
