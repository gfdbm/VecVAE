# -*- coding: utf-8 -*-
"""
2.2 PixelEncoder —— 把阶段 SDF（[B,1,H,W]）编码为 K/V tokens（[B,N,D]）
- 采用 Conv2d 做 patch-embedding（等价 ViT 的分块投影）
- 叠加 2D 正弦位置编码（sincos），保持空间布局感
- 输出：
    feat_2d:   [B, D, H', W']  —— 方便可视化/调试
    kv_tokens: [B, N, D]       —— N = H'*W'，给 2.3 CrossAttention 当 K/V
- 备注：
    1) embed_dim 必须与 2.1 VectorPrefixEncoder 的 D 一致
    2) H,W 需能整除 patch_size（默认 128 / 16 OK），否则会自动“向上补零填充”到整除
"""

from dataclasses import dataclass
from typing import Tuple, Optional

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------
# 配置
# --------------------------
@dataclass
class PixelEncoderConfig:
    in_ch: int = 1                 # SDF 通道数（我们是单通道）
    embed_dim: int = 256           # 输出维度 D（须与 2.1 一致）
    patch_size: int = 16           # patch 大小（Conv kernel/stride）
    stride: Optional[int] = None   # 默认与 patch_size 相同
    add_pos_enc: bool = True       # 是否加 2D 正弦位置编码
    pos_drop: float = 0.0          # 位置编码后的 dropout
    proj_drop: float = 0.0         # patch embedding 后的 dropout
    norm_after: bool = True        # 是否在 tokens 上 LayerNorm
    # 轻量 CNN 前处理（可关）
    use_stem: bool = True
    stem_ch: int = 32
    stem_k: int = 3


# --------------------------
# 工具：2D sin/cos 位置编码
# --------------------------
def build_2d_sincos_position(H: int, W: int, D: int, device=None, dtype=None) -> torch.Tensor:
    """
    返回 [H, W, D] 的 2D 正弦/余弦位置编码。
    将 D 拆成 Dx + Dy 两部分（Dx=Dy=D//2），分别用于 x/y 方向。
    要求 D 为偶数。
    """
    assert D % 2 == 0, "embed_dim for 2D sincos must be even"
    Dx = D // 2
    Dy = D - Dx

    # x 方向 [W, Dx]，y 方向 [H, Dy]
    # 频带设计：默认与 ViT 类似的指数频带
    def _sincos_1d(L, D1):
        pos = torch.arange(L, device=device, dtype=dtype).unsqueeze(1)  # [L,1]
        div = torch.exp(torch.arange(0, D1, 2, device=device, dtype=dtype) * (-math.log(10000.0) / D1))
        pe = torch.zeros(L, D1, device=device, dtype=dtype)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return pe  # [L, D1]

    px = _sincos_1d(W, Dx)   # [W, Dx]
    py = _sincos_1d(H, Dy)   # [H, Dy]

    # 拼成 [H, W, D]：先扩 x，再扩 y，然后 concat
    px_full = px.unsqueeze(0).expand(H, -1, -1)     # [H, W, Dx]
    py_full = py.unsqueeze(1).expand(-1, W, -1)     # [H, W, Dy]
    pe = torch.cat([px_full, py_full], dim=-1)      # [H, W, D]
    return pe


# --------------------------
# 模块
# --------------------------
class PixelEncoder(nn.Module):
    def __init__(self, cfg: PixelEncoderConfig):
        super().__init__()
        self.cfg = cfg
        stride = cfg.stride or cfg.patch_size

        # 可选：轻量 stem（提升局部平滑/边缘质量，尤其小 patch 时）
        stem = []
        if cfg.use_stem:
            stem.append(nn.Conv2d(cfg.in_ch, cfg.stem_ch, kernel_size=cfg.stem_k, padding=cfg.stem_k // 2))
            stem.append(nn.GELU())
            in_ch = cfg.stem_ch
        else:
            in_ch = cfg.in_ch
        self.stem = nn.Sequential(*stem) if stem else nn.Identity()

        # Patch Embedding（Conv 等价分块+线性）
        self.proj = nn.Conv2d(in_ch, cfg.embed_dim, kernel_size=cfg.patch_size, stride=stride, padding=0, bias=True)
        self.proj_drop = nn.Dropout(cfg.proj_drop) if cfg.proj_drop > 0 else nn.Identity()

        # 2D 位置编码（注册为 buffer，前向加到 tokens 上）
        self.add_pos = cfg.add_pos_enc
        self.pos_drop = nn.Dropout(cfg.pos_drop) if cfg.pos_drop > 0 else nn.Identity()
        self.register_buffer("_pe2d", torch.zeros(1), persistent=False)  # 占位；首次前向按 H',W' 构建

        # token 归一化
        self.ln = nn.LayerNorm(cfg.embed_dim) if cfg.norm_after else nn.Identity()

    @staticmethod
    def _pad_to_multiple(x: torch.Tensor, patch: int, stride: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """
        若 H/W 非 stride 的整数倍，则右下补零到最近的倍数，避免 size 不整除。
        返回：x_pad, (H_pad, W_pad)
        """
        B, C, H, W = x.shape
        H_out = math.ceil((H - patch) / stride) + 1 if H >= patch else 1
        W_out = math.ceil((W - patch) / stride) + 1 if W >= patch else 1
        H_need = (H_out - 1) * stride + patch
        W_need = (W_out - 1) * stride + patch
        pad_h = max(0, H_need - H)
        pad_w = max(0, W_need - W)
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="constant", value=0.0)  # (left,right,top,bottom)
        return x, (H + pad_h, W + pad_w)

    def forward(self, stage_sdf: torch.Tensor):
        """
        输入：stage_sdf [B,1,H,W] —— 阶段 SDF（值域 ~[-1,1]）
        输出：
            feat_2d   [B, D, H', W']
            kv_tokens [B, N, D]（N=H'*W'）
        """
        assert stage_sdf.dim() == 4 and stage_sdf.size(1) == self.cfg.in_ch, \
            f"expect [B,{self.cfg.in_ch},H,W], got {tuple(stage_sdf.shape)}"

        x = self.stem(stage_sdf)  # [B, C', H, W]

        # 对齐到 stride 的整数倍（防失配）
        patch = self.cfg.patch_size
        stride = self.cfg.stride or patch
        x, (Hp, Wp) = self._pad_to_multiple(x, patch, stride)

        # patch embedding
        feat_2d = self.proj(x)            # [B, D, H', W']，H' = floor((Hp-patch)/stride)+1
        feat_2d = self.proj_drop(feat_2d)

        B, D, H2, W2 = feat_2d.shape

        # 构造/缓存 2D 位置编码（与当前 H',W' 匹配）
        if self.add_pos:
            if (not torch.is_tensor(self._pe2d)) or self._pe2d.numel() != (H2 * W2 * D):
                pe = build_2d_sincos_position(H2, W2, D, device=feat_2d.device, dtype=feat_2d.dtype)  # [H',W',D]
                self._pe2d = pe  # 注册在 buffer 中（persistent=False），方便复用
            pe2d = self._pe2d.view(H2, W2, D).permute(2, 0, 1).unsqueeze(0)  # [1, D, H', W']
            feat_2d = feat_2d + pe2d
            feat_2d = self.pos_drop(feat_2d)

        # 展平为 tokens
        kv_tokens = feat_2d.flatten(2).permute(0, 2, 1).contiguous()  # [B, N, D]
        kv_tokens = self.ln(kv_tokens)  # 归一化后更稳

        return feat_2d, kv_tokens


# --------------------------
# 自检 main（仅依赖 1.1/1.2/1.3）
# --------------------------
if __name__ == "__main__":
    import argparse
    import sys
    from pathlib import Path

    # 允许从脚本直接运行：把项目根（含 src/）加到 sys.path
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    # 依赖：1.1/1.2/1.3
    from src.data.dataset import NPZDataset
    from src.data.rsm_batcher import RSMBatcher, RSMConfig
    from src.data.stage_renderer import StageRenderer, StageRendererConfig

    ap = argparse.ArgumentParser(description="2.2 PixelEncoder quick test")
    ap.add_argument("--npz", default='/data/kssczt/ztb/FontVector/VAE/data/msyh.npz', help="path to dataset .npz")
    ap.add_argument("--use-rsm", default=True, help="use RSM stage prefix (else full stage)")
    ap.add_argument("--embed", type=int, default=256, help="shared embed dim D (must match 2.1)")
    ap.add_argument("--patch", type=int, default=16, help="patch size (Conv kernel/stride)")
    ap.add_argument("--batch", type=int, default=2, help="batch size for dry-run")
    ap.add_argument("--img-size", type=int, default=128, help="render size; default: dataset H")
    args = ap.parse_args()

    # 数据
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
        img_size=args.img_size or base.H,
        sdf_clip_px=8.0,
        out_dtype=torch.float32,
    ))

    # 小批取样 + 在线渲染阶段 SDF
    B = min(args.batch, len(ds))
    items = [ds[i] for i in range(B)]
    stage_sdf = torch.stack([renderer.render_item(it) for it in items], dim=0)  # [B,1,H,W]

    # 2.2 实例 & 前向
    cfg = PixelEncoderConfig(in_ch=1, embed_dim=args.embed, patch_size=args.patch, stride=args.patch,
                             add_pos_enc=True, use_stem=True)
    pix = PixelEncoder(cfg)
    feat_2d, kv_tokens = pix(stage_sdf)

    # 打印形状
    print(f"stage_sdf: {tuple(stage_sdf.shape)}")
    print(f"feat_2d:   {tuple(feat_2d.shape)}")     # [B, D, H', W']
    print(f"kv_tokens: {tuple(kv_tokens.shape)}")  # [B, N, D]（N=H'*W'）
    # 粗略验证 N 与 H',W' 对应关系
    B, D, Hp, Wp = feat_2d.shape
    N = kv_tokens.shape[1]
    print(f"[check] N == H'*W'? {N == (Hp*Wp)}  (N={N}, H'*W'={Hp*Wp})")
