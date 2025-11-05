# -*- coding: utf-8 -*-
"""
2.4 VP-VAE Encoder —— 将 2.1(矢量前缀) + 2.2(像素) + 2.3(跨注意力) 串成「整体编码器」，
末端内联 PosteriorHead 回归 μ/σ，支持重参数化采样 z。
【自包含版本】无需另建 posterior_head.py

输入:
  - seq_cmd [B,L]        : 矢量命令 token 序列 (M/L/Q/T/Z/NEW/HOLE/END/PAD)
  - seq_arg [B,L,4]      : 对应坐标/控制点 (M/L 用 x,y，Q 用 cx,cy,x,y，其余填 0)
  - seq_mask [B,L]       : 前缀掩码 (True=前缀内；RSM 裁剪后；END 之后/PAD 为 False)
  - contour_ids [B,L]    : 轮廓 ID（同一轮廓内 token 相等）
  - seq_topo [B,L]       : 拓扑位 (NEW/HOLE/END 位掩码)
  - stage_sdf [B,1,H,W]  : 阶段 SDF（建议与 RSM 的阶段 K 对齐；若不用 RSM可用整字图）

输出:
  - mu [B,Z], logvar [B,Z], z [B,Z]
  - aux(dict): 中间表征，含 q_tokens / kv_tokens / q_fused / prefix_repr / pooled

设计要点：
- 2.1 矢量侧自注意力用 key_padding_mask=~seq_mask 屏蔽「未来 & PAD」的 K/V 列（防止偷看答案）
- 2.3 跨注意力对 ~seq_mask 的 Query 行「注意力后 + FFN 后」各置零一次（防 NaN、严守不看未来）
- PosteriorHead 对前缀内做 masked-mean 池化，再回归 μ/σ；可选拼接 prefix_repr 以增强汇总信息
"""

from dataclasses import dataclass
from typing import Tuple, Dict, Optional
import torch
import torch.nn as nn

# --------------------------
# 子模块 (你已完成的 2.1/2.2/2.3)
# --------------------------
from .vector_encoder import VectorPrefixEncoder, VectorPrefixEncoderConfig          # 2.1 矢量前缀编码
from .pixel_encoder import PixelEncoder, PixelEncoderConfig                        # 2.2 像素编码 (SDF → tokens)
from .cross_attention import CrossAttentionAdapter, CrossAttnConfig                # 2.3 跨注意力（安全版：双置零）

# --------------------------
# 内联 PosteriorHead（μ/σ）+ reparameterize + KL
# --------------------------
@dataclass
class PosteriorHeadConfig:
    """PosteriorHead 的配置。注意 embed_dim 必须与 2.1/2.2/2.3 的 D 一致。"""
    embed_dim: int = 256
    z_dim: int = 128
    use_prefix_repr: bool = True   # 是否把 2.1 返回的 prefix_repr 一并拼接用于回归 μ/σ
    hidden_mult: int = 2           # MLP 隐层扩张倍数（in_dim → in_dim * hidden_mult）
    dropout: float = 0.0           # 轻度正则（过拟合时可设 0.05）

class PosteriorHead(nn.Module):
    """
    从 (q_fused, seq_mask[, prefix_repr]) 回归高斯后验的 μ / logσ。
    - q_fused: 2.3 输出的融合序列 (前缀内有效，前缀外已被置零)
    - seq_mask: 只在前缀内统计（masked mean），与 1.4 的损失掩码一致
    - prefix_repr: 2.1 的句向量（聚合的前缀表示），可选拼接提升鲁棒性
    """
    def __init__(self, cfg: PosteriorHeadConfig):
        super().__init__()
        self.cfg = cfg
        in_dim = cfg.embed_dim + (cfg.embed_dim if cfg.use_prefix_repr else 0)
        hid = in_dim * cfg.hidden_mult
        self.mlp = nn.Sequential(
            nn.LayerNorm(in_dim),                # 预归一化：稳定训练
            nn.Linear(in_dim, hid),
            nn.GELU(),
            nn.Dropout(cfg.dropout) if cfg.dropout > 0 else nn.Identity(),
            nn.Linear(hid, 2 * cfg.z_dim),      # 输出拼在一起: [μ | logσ]
        )

    @staticmethod
    def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        只在 mask=True 的位置做均值池化。
        x: [B,L,D]  mask: [B,L]  ->  return: [B,D]
        """
        gate = mask.bool().unsqueeze(-1)        # [B,L,1]
        num = (x * gate).sum(dim=1)             # [B,D]
        den = gate.sum(dim=1).clamp(min=1)      # [B,1] 防止除 0
        return num / den

    def forward(
        self,
        q_fused: torch.Tensor,               # [B, L, D]
        seq_mask: torch.Tensor,              # [B, L] True=前缀内
        prefix_repr: Optional[torch.Tensor] = None  # [B, D]（use_prefix_repr=True 时必须提供）
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, L, D = q_fused.shape
        assert D == self.cfg.embed_dim, f"embed_dim mismatch: got {D}, expect {self.cfg.embed_dim}"

        # 只聚合前缀内的 token（与 1.4 的损失口径一致）
        pooled = self.masked_mean(q_fused, seq_mask)  # [B,D]

        # 可选：与 2.1 的 prefix_repr 拼接，提供更稳定的句向量
        if self.cfg.use_prefix_repr:
            assert prefix_repr is not None and prefix_repr.shape == (B, D)
            h = torch.cat([pooled, prefix_repr], dim=-1)  # [B, 2D]
        else:
            h = pooled                                    # [B, D]

        out = self.mlp(h)                    # [B, 2Z]
        mu, logvar = out.chunk(2, dim=-1)    # [B,Z], [B,Z]
        return mu, logvar, h                 # h: 便于调试（是喂入 MLP 的 pooled 表示）

def reparameterize(mu: torch.Tensor, logvar: torch.Tensor, eps_std: float = 1.0) -> torch.Tensor:
    """
    重参数化技巧：z = μ + σ * ε,  其中 ε ~ N(0, I)
    - 训练时 eps_std=1.0；评估/可视化时也可设 1.0（保持可重复性则固定种子）
    """
    std = (0.5 * logvar).exp()               # σ = exp(0.5 * logσ^2)
    eps = torch.randn_like(std) * eps_std
    return mu + std * eps

def kl_normal_standard(mu: torch.Tensor, logvar: torch.Tensor, reduction: str = "batch") -> torch.Tensor:
    """
    KL( N(μ,σ^2) || N(0,1) ) = -0.5 * (1 + logσ^2 - μ^2 - σ^2)
    - "batch": 对 Z 维求和 → 对 batch 求均值（最常用）
    - "mean" : 所有元素均值
    - "sum"  : 所有元素求和
    """
    kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())  # [B,Z]
    if reduction == "batch":
        return kl.sum(dim=-1).mean()
    elif reduction == "mean":
        return kl.mean()
    elif reduction == "sum":
        return kl.sum()
    else:
        raise ValueError(f"Unknown reduction: {reduction}")

# --------------------------
# 2.4 编码器总装
# --------------------------
@dataclass
class VpVaeEncoderConfig:
    """整体编码器配置（与 2.1/2.2/2.3 保持一致的 D/heads/patch 等）。"""
    embed_dim: int = 256            # 统一通道 D
    num_heads: int = 8              # 注意力头数 (2.1/2.3)
    cross_layers: int = 1           # 2.3 跨注意力层数（可设 2 提升融合）
    patch_size: int = 16            # 2.2 分块大小 (H=128 → Lk = (128/patch)^2)
    z_dim: int = 128                # 潜在维度 Z
    vec_layers: int = 3             # 2.1 矢量编码器层数
    use_prefix_repr: bool = True    # μ/σ 回归是否拼接 prefix_repr
    dropout: float = 0.0            # PosteriorHead 的轻正则

class VpVaeEncoder(nn.Module):
    """
    串起 2.1 + 2.2 + 2.3，并在尾部用 PosteriorHead 回归 μ/σ，支持重参数化采样。
    —— 这是你训练 VP-VAE 编码器时真正要用的「Encoder」模块。
    """
    def __init__(self, cfg: VpVaeEncoderConfig):
        super().__init__()
        self.cfg = cfg
        # 2.1 矢量前缀编码（内部：key_padding_mask=~seq_mask → 不看未来 & PAD 的列）
        self.vec = VectorPrefixEncoder(VectorPrefixEncoderConfig(
            embed_dim=cfg.embed_dim, n_heads=cfg.num_heads, n_layers=cfg.vec_layers
        ))
        # 2.2 像素编码（阶段 SDF → patch tokens）
        self.pix = PixelEncoder(PixelEncoderConfig(
            in_ch=1, embed_dim=cfg.embed_dim, patch_size=cfg.patch_size, stride=cfg.patch_size
        ))
        # 2.3 跨注意力（安全版：注意力后 + FFN 后对 ~seq_mask 行置零）
        self.xattn = CrossAttentionAdapter(CrossAttnConfig(
            embed_dim=cfg.embed_dim, num_heads=cfg.num_heads, n_layers=cfg.cross_layers, return_attn_stats=False
        ))
        # μ/σ 头（内联，做 masked-mean 池化 + 小 MLP 回归）
        self.head = PosteriorHead(PosteriorHeadConfig(
            embed_dim=cfg.embed_dim, z_dim=cfg.z_dim, use_prefix_repr=cfg.use_prefix_repr,
            hidden_mult=2, dropout=cfg.dropout
        ))

    def forward(
        self,
        seq_cmd: torch.Tensor,         # [B,L]
        seq_arg: torch.Tensor,         # [B,L,4]
        seq_mask: torch.Tensor,        # [B,L] True=前缀内
        contour_ids: torch.Tensor,     # [B,L]
        seq_topo: torch.Tensor,        # [B,L]
        stage_sdf: torch.Tensor,       # [B,1,H,W]
        sample: bool = True,           # True: 返回重参数化采样 z；False: 返回 μ (常用于 ablation)
        eps_std: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        返回:
          - mu [B,Z], logvar [B,Z], z [B,Z]
          - aux: 便于调试/可视化的中间特征
        """
        # === 2.1 矢量前缀编码 ===
        # 这里内部会用 key_padding_mask=~seq_mask 切掉未来 & PAD 的 K/V 列，防止信息泄漏。
        q_tokens, prefix_repr, key_pad_mask = self.vec(seq_cmd, seq_arg, seq_mask, contour_ids, seq_topo)  # [B,L,D],[B,D],[B,L]
        assert torch.isfinite(q_tokens).all(), "q_tokens has NaN/Inf"

        # === 2.2 像素编码（阶段 SDF → K/V tokens） ===
        _, kv_tokens = self.pix(stage_sdf)  # [B,Lk,D]；Lk = (H/patch_size)^2
        assert torch.isfinite(kv_tokens).all(), "kv_tokens has NaN/Inf"

        # === 2.3 跨注意力融合（双置零确保不看未来；避免行级 -inf 引发 NaN）===
        q_fused, _ = self.xattn(q_tokens, kv_tokens, q_mask=seq_mask, kv_key_padding_mask=None)  # [B,L,D]
        assert torch.isfinite(q_fused).all(), "q_fused has NaN/Inf"

        # === μ/σ 回归 & 重参数化采样 ===
        # PosteriorHead 内部只对前缀内做 masked-mean 池化；可选拼接 prefix_repr。
        mu, logvar, pooled = self.head(q_fused, seq_mask, prefix_repr=prefix_repr)  # [B,Z], [B,Z], [B, D or 2D]
        z = reparameterize(mu, logvar, eps_std) if sample else mu

        # 便于外部调试/可视化（注意：q_fused 在 ~seq_mask 行已被置零）
        aux = {
            "q_tokens": q_tokens, "kv_tokens": kv_tokens, "q_fused": q_fused,
            "prefix_repr": prefix_repr, "pooled": pooled
        }
        return mu, logvar, z, aux

# --------------------------
# 自检 main：1.1→1.2→1.3→2.4 串起来跑一遍
# --------------------------
if __name__ == "__main__":
    import argparse, sys
    from pathlib import Path

    # 允许 python -m 方式从项目根目录运行
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    # 数据与工具（你之前已完成）
    from src.data.dataset import NPZDataset                               # 1.1
    from src.data.rsm_batcher import RSMBatcher, RSMConfig                # 1.2
    from src.data.stage_renderer import StageRenderer, StageRendererConfig# 1.3

    ap = argparse.ArgumentParser(description="2.4 VP-VAE Encoder quick test (self-contained, with comments)")
    ap.add_argument("--npz", default='/data/kssczt/ztb/FontVector/VAE/data/msyh.npz', help="path to dataset .npz")
    ap.add_argument("--use-rsm", default=True, help="use RSM stage prefix (建议在训练时开启)")
    ap.add_argument("--embed", type=int, default=256, help="shared embed dim D，需与 2.1/2.2/2.3 一致")
    ap.add_argument("--heads", type=int, default=8, help="num attention heads（D 能被 heads 整除）")
    ap.add_argument("--xlayers", type=int, default=1, help="num cross-attn layers（2.3 堆叠层数）")
    ap.add_argument("--zdim", type=int, default=128, help="latent dim Z")
    ap.add_argument("--patch", type=int, default=16, help="pixel patch size（H/patch 必须为整数）")
    ap.add_argument("--batch", type=int, default=1, help="dry-run 的 batch 大小")
    ap.add_argument("--img-size", type=int, default=None, help="渲染分辨率，默认用数据集 H")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                    help="cuda / cpu")
    args = ap.parse_args()

    device = torch.device(args.device)

    # === 数据集 & (可选)RSM ===
    base = NPZDataset(args.npz, mmap=True, use_seq_topo=True, strict_shapes=True)
    if args.use_rsm:
        # RSM：随机抽取阶段 K（0.25/0.5/0.75/1.0），构造「前缀」以提升复杂字形的稳定性
        ds = RSMBatcher(base, RSMConfig(
            rsm_stages=(0.25, 0.5, 0.75, 1.0),
            rsm_probs=(0.15, 0.25, 0.25, 0.35),
            enforce_contour_ids=True,               # 强制使用 contour_ids（与 2.1 的相对位置一致）
            allow_token_prefix_fallback=False,      # 没有 contour_ids 就不要退化为纯 token 前缀
        ))
        print("[info] RSM on (stage prefix)")
    else:
        ds = base
        print("[info] FULL stage (no RSM)")

    # 阶段 SDF 渲染器（与 RSM 阶段对齐；不用 RSM 时可渲整字图）
    renderer = StageRenderer(StageRendererConfig(
        img_size=args.img_size or base.H,    # 默认与数据集图像尺寸一致
        sdf_clip_px=8.0,                     # SDF 裁剪比例（与 1.3 保持一致即可）
        out_dtype=torch.float32,
    ))

    # 取一个 batch 做干跑（dry-run）
    B = min(args.batch, len(ds))
    items = [ds[i] for i in range(B)]
    def cat(k): return torch.stack([it[k] for it in items], 0).to(device)

    seq_cmd      = cat("seq_cmd")
    seq_arg      = cat("seq_arg")
    seq_mask     = cat("seq_mask")
    contour_ids  = cat("contour_ids")
    seq_topo     = cat("seq_topo")
    stage_sdf    = torch.stack([renderer.render_item(it) for it in items], 0).to(device)

    # === 2.4 编码器实例 ===
    enc = VpVaeEncoder(VpVaeEncoderConfig(
        embed_dim=args.embed, num_heads=args.heads, cross_layers=args.xlayers,
        patch_size=args.patch, z_dim=args.zdim, use_prefix_repr=True, dropout=0.0
    )).to(device)
    enc.eval()  # 自检用 eval；训练脚本里请 enc.train()

    with torch.no_grad():
        mu, logvar, z, aux = enc(seq_cmd, seq_arg, seq_mask, contour_ids, seq_topo, stage_sdf,
                                 sample=True, eps_std=1.0)
        kl = kl_normal_standard(mu, logvar, reduction="batch")

    # === 形状与健康检查 ===
    print(f"q_tokens:  {tuple(aux['q_tokens'].shape)}")   # (B, L, D)
    print(f"kv_tokens: {tuple(aux['kv_tokens'].shape)}")  # (B, Lk, D)
    print(f"q_fused:   {tuple(aux['q_fused'].shape)}")    # (B, L, D)
    print(f"mu/logσ/z: {tuple(mu.shape)} / {tuple(logvar.shape)} / {tuple(z.shape)}  (Z={enc.cfg.z_dim})")
    print(f"KL(batch-mean): {float(kl.item()):.6f}")

    # 前缀外应为 0（来自 2.3 的双置零）；不为 0 多半是 q_mask/置零逻辑被改坏了
    out_mask = ~seq_mask.bool()
    if out_mask.any():
        l2_out = aux['q_fused'][out_mask].pow(2).sum(dim=-1).mean().item()
        print(f"[check] q_fused outside-prefix L2 mean ≈ 0 ? {l2_out:.6f}")
    else:
        print("[check] no outside-prefix tokens in this batch")
