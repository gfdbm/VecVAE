# -*- coding: utf-8 -*-
"""
2.3 CrossAttentionAdapter —— 矢量前缀 (Q) 与 像素 tokens (K/V) 的跨注意力融合（安全版）

设计要点（与你的工程严格对齐）：
- 不使用“整行 -inf”行级 attn_mask，避免 softmax(all -inf) → NaN。
- 用 q_mask（通常就是 1.2/RSM 返回的 seq_mask）在注意力块后 + FFN 块后各置零一次，
  保证“前缀外（~q_mask）最终恒为 0”，严守“不看未来”的 RSM 约束。
- 支持 n_layers 堆叠（每层 = PreNorm MHA 残差块 + PreNorm FFN 残差块）。
- 与 2.1/2.2 的 embed_dim 完全一致；不要求 Lq == Lk。

自检 main 会串起 1.1/1.2/1.3/2.1/2.2/2.3，打印形状并检查前缀外输出是否约等于 0。
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List

import torch
import torch.nn as nn


# --------------------------
# Config
# --------------------------
@dataclass
class CrossAttnConfig:
    embed_dim: int = 256          # 必须与 2.1 / 2.2 一致
    num_heads: int = 8
    n_layers: int = 1             # 堆叠多少层 (MHA+FFN)
    qkv_bias: bool = True
    attn_drop: float = 0.0
    resid_drop: float = 0.0
    ffn_drop: float = 0.0
    ffn_mult: int = 4
    return_attn_stats: bool = False  # 打印调试统计（前缀外 L2 均值）


# --------------------------
# 基础层
# --------------------------
class PreNorm(nn.Module):
    def __init__(self, dim: int, fn: nn.Module):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x: torch.Tensor, *args, **kwargs):
        return self.fn(self.norm(x), *args, **kwargs)

class Residual(nn.Module):
    def __init__(self, fn: nn.Module, drop: float = 0.0):
        super().__init__()
        self.fn = fn
        self.drop = nn.Dropout(drop) if drop > 0.0 else nn.Identity()
    def forward(self, x: torch.Tensor, *args, **kwargs):
        return x + self.drop(self.fn(x, *args, **kwargs))

class FeedForward(nn.Module):
    def __init__(self, dim: int, mult: int = 4, drop: float = 0.0):
        super().__init__()
        hid = dim * mult
        self.net = nn.Sequential(
            nn.Linear(dim, hid),
            nn.GELU(),
            nn.Dropout(drop) if drop > 0.0 else nn.Identity(),
            nn.Linear(hid, dim),
        )
    def forward(self, x: torch.Tensor):
        return self.net(x)


# --------------------------
# 单层跨注意力块：MHA 残差 + FFN 残差，并在两处做 q_mask 置零
# --------------------------
class CrossAttnLayer(nn.Module):
    def __init__(self, dim: int, heads: int, qkv_bias: bool, attn_drop: float, resid_drop: float,
                 ffn_mult: int, ffn_drop: float):
        super().__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim=dim, num_heads=heads, dropout=attn_drop,
            batch_first=True, bias=qkv_bias
        )
        self.attn_block = Residual(PreNorm(dim, self._attend), drop=resid_drop)
        self.ffn_block  = Residual(PreNorm(dim, FeedForward(dim, ffn_mult, ffn_drop)), drop=resid_drop)

    def _attend(self, q_tokens: torch.Tensor, *, kv_tokens: torch.Tensor,
                kv_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # 不构造行级 attn_mask，避免全 -inf 行导致 NaN
        assert torch.isfinite(q_tokens).all(), "q_tokens has NaN/Inf"
        assert torch.isfinite(kv_tokens).all(), "kv_tokens has NaN/Inf"
        y, _ = self.mha(query=q_tokens, key=kv_tokens, value=kv_tokens,
                        key_padding_mask=kv_key_padding_mask, need_weights=False, attn_mask=None)
        return y

    def forward(self, x: torch.Tensor, *, kv_tokens: torch.Tensor,
                q_mask: Optional[torch.Tensor] = None,
                kv_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # 1) 跨注意力（PreNorm + 残差）
        y = self.attn_block(x, kv_tokens=kv_tokens, kv_key_padding_mask=kv_key_padding_mask)

        # 2) 注意力块后：对 ~q_mask 强制置零，防止流入后续 FFN
        gate = None
        if q_mask is not None:
            gate = q_mask.bool().unsqueeze(-1)  # [B, Lq, 1]
            y = y.masked_fill(~gate, 0.0)

        # 3) FFN（PreNorm + 残差）
        y = self.ffn_block(y)

        # 4) FFN 块后：再次置零，防止残差把值带回来
        if gate is not None:
            y = y.masked_fill(~gate, 0.0)

        assert torch.isfinite(y).all(), "CrossAttnLayer produced NaN/Inf"
        return y


# --------------------------
# Adapter：堆叠 n 层 CrossAttnLayer
# --------------------------
class CrossAttentionAdapter(nn.Module):
    def __init__(self, cfg: CrossAttnConfig):
        super().__init__()
        self.cfg = cfg
        layers: List[CrossAttnLayer] = []
        for _ in range(cfg.n_layers):
            layers.append(
                CrossAttnLayer(
                    dim=cfg.embed_dim, heads=cfg.num_heads, qkv_bias=cfg.qkv_bias,
                    attn_drop=cfg.attn_drop, resid_drop=cfg.resid_drop,
                    ffn_mult=cfg.ffn_mult, ffn_drop=cfg.ffn_drop
                )
            )
        self.layers = nn.ModuleList(layers)

    def forward(self,
                q_tokens: torch.Tensor,        # [B, Lq, D]
                kv_tokens: torch.Tensor,       # [B, Lk, D]
                q_mask: Optional[torch.Tensor] = None,              # [B, Lq] True=前缀内
                kv_key_padding_mask: Optional[torch.Tensor] = None  # [B, Lk] 像素侧通常 None
                ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        y = q_tokens
        for layer in self.layers:
            y = layer(y, kv_tokens=kv_tokens, q_mask=q_mask, kv_key_padding_mask=kv_key_padding_mask)

        stats: Dict[str, torch.Tensor] = {}
        if self.cfg.return_attn_stats and (q_mask is not None):
            out_mask = ~q_mask.bool()
            if out_mask.any():
                # 只在前缀外位置度量 L2 均值（应接近 0）
                masked_l2 = y[out_mask].pow(2).sum(dim=-1).mean()
                stats["masked_q_l2_mean"] = masked_l2.detach()
            else:
                stats["masked_q_l2_mean"] = torch.tensor(0.0, device=y.device)
        return y, stats


# --------------------------
# 自检 main：串 1.1/1.2/1.3/2.1/2.2/2.3
# --------------------------
if __name__ == "__main__":
    import argparse, sys
    from pathlib import Path
    import torch

    # 确保从脚本直接运行时也能导入 src.*
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    # 依赖模块（已在你项目中完成）
    from src.data.dataset import NPZDataset                 # 1.1
    from src.data.rsm_batcher import RSMBatcher, RSMConfig  # 1.2
    from src.data.stage_renderer import StageRenderer, StageRendererConfig  # 1.3
    from src.model.vector_encoder import VectorPrefixEncoder, VectorPrefixEncoderConfig  # 2.1
    from src.model.pixel_encoder import PixelEncoder, PixelEncoderConfig    # 2.2

    ap = argparse.ArgumentParser(description="2.3 CrossAttentionAdapter quick test (no NaN, no future leakage)")
    ap.add_argument("--npz", default='/data/kssczt/ztb/FontVector/VAE/data/msyh.npz', help="path to dataset .npz")
    ap.add_argument("--use-rsm", default=True, help="use RSM stage prefix (else full stage)")
    ap.add_argument("--embed", type=int, default=256, help="shared embed dim D")
    ap.add_argument("--heads", type=int, default=8, help="num attention heads")
    ap.add_argument("--layers", type=int, default=1, help="num cross-attn layers")
    ap.add_argument("--patch", type=int, default=16, help="pixel patch size (affects Lk)")
    ap.add_argument("--batch", type=int, default=1, help="batch size for dry-run")
    ap.add_argument("--img-size", type=int, default=None, help="renderer size; default=dataset H")
    args = ap.parse_args()

    # === 数据 ===
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

    # 小批取样并打包
    B = min(args.batch, len(ds))
    items = [ds[i] for i in range(B)]
    def cat(key): return torch.stack([it[key] for it in items], dim=0)

    seq_cmd      = cat("seq_cmd")         # [B,L]
    seq_arg      = cat("seq_arg")         # [B,L,4]
    seq_mask     = cat("seq_mask")        # [B,L] True=前缀内
    contour_ids  = cat("contour_ids")     # [B,L]
    seq_topo     = cat("seq_topo")        # [B,L]
    stage_sdf    = torch.stack([renderer.render_item(it) for it in items], dim=0)  # [B,1,H,W]

    # === 2.1：矢量前缀编码（内部使用 key_padding_mask = ~seq_mask，严格不看未来 & PAD）===
    vec = VectorPrefixEncoder(VectorPrefixEncoderConfig(embed_dim=args.embed, n_heads=args.heads))
    q_tokens, prefix_repr, key_pad_mask = vec(seq_cmd, seq_arg, seq_mask, contour_ids, seq_topo)  # [B,L,D], [B,D], [B,L]
    assert torch.isfinite(q_tokens).all(), "q_tokens has NaN/Inf"

    # === 2.2：像素编码（K/V tokens）===
    pix = PixelEncoder(PixelEncoderConfig(in_ch=1, embed_dim=args.embed, patch_size=args.patch, stride=args.patch))
    _, kv_tokens = pix(stage_sdf)  # [B, N, D]
    assert torch.isfinite(kv_tokens).all(), "kv_tokens has NaN/Inf"

    # === 2.3：跨注意力（安全版：无行级 -inf；输出双置零）===
    xattn = CrossAttentionAdapter(CrossAttnConfig(
        embed_dim=args.embed, num_heads=args.heads, n_layers=args.layers,
        return_attn_stats=True
    ))
    q_fused, stats = xattn(q_tokens, kv_tokens, q_mask=seq_mask, kv_key_padding_mask=None)
    assert torch.isfinite(q_fused).all(), "q_fused has NaN/Inf"

    # === 打印形状与一致性检查 ===
    print(f"q_tokens:  {tuple(q_tokens.shape)}")
    print(f"kv_tokens: {tuple(kv_tokens.shape)}")
    print(f"q_fused:   {tuple(q_fused.shape)}")
    print(q_tokens)
    print(kv_tokens)
    print(q_fused)

    # 前缀外 L2 均值应≈0（若样本中确有前缀外位置）
    out_mask = ~seq_mask.bool()
    if out_mask.any():
        masked_l2 = q_fused[out_mask].pow(2).sum(dim=-1).mean().item()
        print(f"[check] fused outside-prefix L2 mean ≈ 0 ?  {masked_l2:.6f}")
    else:
        print("[check] no outside-prefix tokens in this batch")

    if stats:
        print(f"[stat] {{ {', '.join(f'{k}: {float(v.item()):.6g}' for k,v in stats.items())} }}")
