# -*- coding: utf-8 -*-
"""
2.5 VP-VAE Decoder —— 用 (z, stage_sdf) 复原前缀序列的命令与坐标
- 输入:  z [B,Z], stage_sdf [B,1,H,W], seq_mask [B,L] (True=需复原的前缀位)
- 输出:  logits_cmd [B,L,V], pred_arg [B,L,4]
- 训练:  用 1.4 的 lmc/lma 掩码计算 CE/L1；与 2.4 的 KL 一起优化

设计要点：
- 不喂入 GT 命令/拓扑，避免信息泄漏；只用 z 与图像条件。
- 查询槽 Query: 由绝对位置 + (可选)有效位嵌入 + z 的全局注入组成；每层后对 ~seq_mask 置零。
- 跨注意力到像素 tokens：与 2.2 同构的 PixelEncoder；保证几何对齐能力。
"""

from dataclasses import dataclass
from typing import Tuple, Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------
# 依赖：像素编码器（2.2）和（可选）2.4 编码器（自检时产出 z）
# --------------------------
from .pixel_encoder import PixelEncoder, PixelEncoderConfig
from .encoder import VpVaeEncoder, VpVaeEncoderConfig  # 仅 main 自检使用；训练时你会从 2.4 拿 z

# --------------------------
# 基础模块：预归一化注意力/前馈（轻量 Transformer 块）
# --------------------------
class PreNormMHA(nn.Module):
    """
    预归一化多头注意力（支持自注意力/跨注意力）
    - x: [B,L,D] ；y: [B,M,D] 或 None（None 表示自注意力，Q=K=V=x）
    - key_padding_mask: [B,M] (True=填充/不可见)；attn_mask: [L,L] 或 [B*H,L,M]
    """
    def __init__(self, d_model: int, n_heads: int, attn_drop: float = 0.0, resid_drop: float = 0.0):
        super().__init__()
        self.ln = nn.LayerNorm(d_model)
        self.mha = nn.MultiheadAttention(d_model, n_heads, dropout=attn_drop, batch_first=True)
        self.drop = nn.Dropout(resid_drop) if resid_drop > 0 else nn.Identity()

    def forward(self, x, y=None, key_padding_mask=None, attn_mask=None):
        h = self.ln(x)
        if y is None:
            out, _ = self.mha(h, h, h, key_padding_mask=key_padding_mask, attn_mask=attn_mask, need_weights=False)
        else:
            k = self.ln(y)  # 也做 LN，稳定一点
            out, _ = self.mha(h, k, k, key_padding_mask=key_padding_mask, attn_mask=attn_mask, need_weights=False)
        return x + self.drop(out)

class PreNormFFN(nn.Module):
    """预归一化前馈 MLP：LN → Linear → GELU → Dropout → Linear → 残差"""
    def __init__(self, d_model: int, ffn_mult: int = 4, ffn_drop: float = 0.0, resid_drop: float = 0.0):
        super().__init__()
        hid = d_model * ffn_mult
        self.ln = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, hid), nn.GELU(),
            nn.Dropout(ffn_drop) if ffn_drop > 0 else nn.Identity(),
            nn.Linear(hid, d_model),
        )
        self.drop = nn.Dropout(resid_drop) if resid_drop > 0 else nn.Identity()

    def forward(self, x):
        return x + self.drop(self.ff(self.ln(x)))

# --------------------------
# 解码器配置
# --------------------------
@dataclass
class DecoderConfig:
    vocab_size: int = 9          # 命令词表大小（和数据集一致：PAD,M,L,Q,T,Z,NEW,HOLE,END）
    max_len: int = 256           # Lmax（与数据集一致）
    embed_dim: int = 256         # 统一通道 D
    z_dim: int = 128             # 潜向量维度 Z（与 2.4 对齐）
    n_heads: int = 8
    n_layers: int = 4            # 解码堆叠层数（2~6 之间常见）
    d_pos: int = 64              # 绝对位置嵌入维度（投影到 D 相加）
    d_valid: int = 32            # 有效位(前缀/非前缀)嵌入维度（投影到 D 相加）
    attn_drop: float = 0.0
    resid_drop: float = 0.0
    ffn_drop: float = 0.0
    ffn_mult: int = 4
    # 像素侧
    patch_size: int = 16         # 2.2 分块；H=128 → Lk=(128/16)^2=64
    use_pixel_cross_attn: bool = True  # 是否对像素 tokens 做跨注意力

# --------------------------
# 2.5 解码器
# --------------------------
class Decoder(nn.Module):
    """
    非自回归的条件解码器：用 (z, stage_sdf) 预测每个位置的命令和坐标。
    - 不输入 GT 命令/拓扑，避免泄漏；只依赖 z + 图像条件 + 位置/有效标记。
    - 对 ~seq_mask 的查询行在每层后置零，保持“只复原前缀”的一致性（损失也只在前缀位计算）。
    """
    def __init__(self, cfg: DecoderConfig):
        super().__init__()
        self.cfg = cfg

        D = cfg.embed_dim
        # 查询槽（L 个位置）：绝对位置嵌入 + 有效位嵌入 + z 注入（全局）
        self.pos_emb = nn.Embedding(cfg.max_len, cfg.d_pos)
        self.valid_emb = nn.Embedding(2, cfg.d_valid)  # 0: 非前缀/填充, 1: 前缀
        self.pos_proj = nn.Linear(cfg.d_pos, D)
        self.valid_proj = nn.Linear(cfg.d_valid, D)
        self.z_proj = nn.Linear(cfg.z_dim, D)

        # 像素编码器（2.2 同构）
        if cfg.use_pixel_cross_attn:
            self.pix = PixelEncoder(PixelEncoderConfig(in_ch=1, embed_dim=D, patch_size=cfg.patch_size, stride=cfg.patch_size))

        # 堆叠的解码块：自注意力 + （可选）跨注意力 + FFN
        blocks = []
        for _ in range(cfg.n_layers):
            blocks += [
                PreNormMHA(D, cfg.n_heads, attn_drop=cfg.attn_drop, resid_drop=cfg.resid_drop),      # self-attn
            ]
            if cfg.use_pixel_cross_attn:
                blocks += [
                    PreNormMHA(D, cfg.n_heads, attn_drop=cfg.attn_drop, resid_drop=cfg.resid_drop)   # cross-attn to pixels
                ]
            blocks += [PreNormFFN(D, ffn_mult=cfg.ffn_mult, ffn_drop=cfg.ffn_drop, resid_drop=cfg.resid_drop)]
        self.blocks = nn.ModuleList(blocks)

        # 输出头：命令分类 & 坐标回归
        self.head_cmd = nn.Linear(D, cfg.vocab_size)
        self.head_arg = nn.Sequential(nn.LayerNorm(D), nn.Linear(D, D*2), nn.GELU(), nn.Linear(D*2, 4))

    def forward(
        self,
        z: torch.Tensor,              # [B,Z]
        stage_sdf: torch.Tensor,      # [B,1,H,W]
        seq_mask: torch.Tensor,       # [B,L] True=前缀位（需要复原/计损）
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        B, Z = z.shape
        B2, _, H, W = stage_sdf.shape
        assert B == B2, "batch mismatch"

        L = seq_mask.shape[1]
        assert L <= self.cfg.max_len, f"max_len too small: need {L}, has {self.cfg.max_len}"

        # 1) 初始化查询槽 [B,L,D]：pos + valid + z 全局注入
        pos_idx = torch.arange(L, device=z.device).unsqueeze(0).expand(B, L)  # [B,L]
        x = self.pos_proj(self.pos_emb(pos_idx))                               # [B,L,D]
        valid = self.valid_proj(self.valid_emb(seq_mask.long()))               # [B,L,D]
        x = x + valid + self.z_proj(z).unsqueeze(1)                            # 注入全局风格/几何摘要

        # 2) 像素 tokens（条件） [B,Lk,D]
        kv_tokens = None
        if self.cfg.use_pixel_cross_attn:
            _, kv_tokens = self.pix(stage_sdf)

        # 3) Transformer 堆叠（每层后对 ~seq_mask 置零；自注意力屏蔽无效列）
        key_pad_q = ~seq_mask.bool()  # 自注意力的 key_padding_mask：无效槽不可见
        for layer in self.blocks:
            if isinstance(layer, PreNormMHA):
                # 判断是 self-attn 还是 cross-attn（通过是否传 y 决定）
                if self.cfg.use_pixel_cross_attn and kv_tokens is not None:
                    # 交替：先 self，再 cross；这里通过判断 layer 顺序实现
                    # 粗略做法：如果传 y=None → self；传 y=kv_tokens → cross
                    # 为简单，我们采用：奇数次 MHA 做 self，偶数次做 cross（对应 blocks 的构造顺序）
                    # 但更明确的方法是检查 layer 的位置；这里用 y 是否 None 来区分：
                    # —— 我们在 blocks 构造时就是 self, (cross), ffn 的顺序，因此这里轮流传参。
                    pass
                # 实际选择：
                if kv_tokens is None:
                    # 无像素条件：只做 self-attn
                    x = layer(x, y=None, key_padding_mask=key_pad_q, attn_mask=None)
                else:
                    # 如果这个 MHA 的 in_proj_weight 与第一个相同我们无法区分，只能通过轮次交替。
                    # 简化：我们总是先做一次 self-attn，再做一次 cross-attn（因为 blocks 顺序如此）
                    x = layer(x, y=None, key_padding_mask=key_pad_q, attn_mask=None)
                    # 下一个 layer 必定是 cross-attn（见 __init__ 构造），所以这里直接继续循环处理
                    continue
            elif isinstance(layer, PreNormFFN):
                x = layer(x)
            else:
                raise RuntimeError("Unknown block type")

            # 双置零（保持无效槽静音，防止残差把数值带回来）
            x = x * seq_mask.unsqueeze(-1)

            # 若紧接着是 cross-attn（存在像素条件），执行之
            if self.cfg.use_pixel_cross_attn and kv_tokens is not None:
                # 取下一个模块（应该是 cross-attn）
                # 注意：为了简化，我们把 cross-attn 紧跟在 self-attn 之后执行
                # 这里安全取出下一个模块：
                idx = self.blocks._modules  # 仅用于取序号的技巧；但直接可重复做 self+cross+ffn
                # 改为简单明晰的实现：每轮循环做 self → cross（可选）→ ffn
                pass  # 这个分支在上面的实现里不够清晰，下面给出更明确的实现（见下一版本）
        # 上面这段为了保持可读性，还是改成更明确的“层循环”写法更好。我们重写 forward 的主体如下：
        # ---------------------------------------------------------------------------------
    def forward(
        self,
        z: torch.Tensor,
        stage_sdf: torch.Tensor,
        seq_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        B, Z = z.shape
        B2, _, H, W = stage_sdf.shape
        assert B == B2, "batch mismatch"

        L = seq_mask.shape[1]
        assert L <= self.cfg.max_len, f"max_len too small: need {L}, has {self.cfg.max_len}"

        pos_idx = torch.arange(L, device=z.device).unsqueeze(0).expand(B, L)
        x = self.pos_proj(self.pos_emb(pos_idx))
        x = x + self.valid_proj(self.valid_emb(seq_mask.long())) + self.z_proj(z).unsqueeze(1)

        kv_tokens = None
        if self.cfg.use_pixel_cross_attn:
            _, kv_tokens = self.pix(stage_sdf)

        # 明确的层循环：每层 = self-attn → (cross-attn) → FFN
        key_pad_q = ~seq_mask.bool()
        it = iter(self.blocks)
        while True:
            try:
                # 1) self-attn
                sa = next(it); assert isinstance(sa, PreNormMHA)
                x = sa(x, y=None, key_padding_mask=key_pad_q, attn_mask=None)
                x = x * seq_mask.unsqueeze(-1)

                # 2) (opt) cross-attn
                if self.cfg.use_pixel_cross_attn and kv_tokens is not None:
                    ca = next(it); assert isinstance(ca, PreNormMHA)
                    x = ca(x, y=kv_tokens, key_padding_mask=None, attn_mask=None)
                    x = x * seq_mask.unsqueeze(-1)

                # 3) ffn
                ff = next(it); assert isinstance(ff, PreNormFFN)
                x = ff(x)
                x = x * seq_mask.unsqueeze(-1)

            except StopIteration:
                break

        # 4) 输出头
        logits_cmd = self.head_cmd(x)     # [B,L,V]
        pred_arg = self.head_arg(x)       # [B,L,4]
        aux = {"dec_slots": x, "kv_tokens": kv_tokens}
        return logits_cmd, pred_arg, aux


# --------------------------
# 自检 main：串 1.1→1.2→1.3→2.4(取 z)→2.5(重建)→打印掩码损失
# --------------------------
if __name__ == "__main__":
    import argparse, sys
    from pathlib import Path
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    # 数据与工具
    from src.data.dataset import NPZDataset                               # 1.1
    from src.data.rsm_batcher import RSMBatcher, RSMConfig                # 1.2
    from src.data.stage_renderer import StageRenderer, StageRendererConfig# 1.3
    from src.data.masks import build_loss_masks                           # 1.4

    ap = argparse.ArgumentParser(description="2.5 Decoder quick test (with masked CE/L1)")
    ap.add_argument("--npz", default='/data/kssczt/ztb/FontVector/VAE/data/msyh.npz', help="path to dataset .npz")
    ap.add_argument("--use-rsm", default=True, help="use RSM stage prefix")
    ap.add_argument("--embed", type=int, default=256)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--xlayers", type=int, default=1, help="encoder cross-attn layers (2.4)")
    ap.add_argument("--zdim", type=int, default=128)
    ap.add_argument("--patch", type=int, default=16)
    ap.add_argument("--dec-layers", type=int, default=4)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--img-size", type=int, default=None)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)

    # === 数据集 & (可选)RSM ===
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

    # 取一个 batch
    B = min(args.batch, len(ds))
    items = [ds[i] for i in range(B)]
    def cat(k): return torch.stack([it[k] for it in items], 0).to(device)

    seq_cmd     = cat("seq_cmd")          # [B,L]
    seq_arg     = cat("seq_arg")          # [B,L,4]
    seq_mask    = cat("seq_mask")         # [B,L]
    contour_ids = cat("contour_ids")
    seq_topo    = cat("seq_topo")
    stage_sdf   = torch.stack([renderer.render_item(it) for it in items], 0).to(device)  # [B,1,H,W]

    # === 2.4 编码器，产出 z（训练时你会前向 2.4+2.5 一起计算损失） ===
    enc = VpVaeEncoder(VpVaeEncoderConfig(
        embed_dim=args.embed, num_heads=args.heads, cross_layers=args.xlayers,
        patch_size=args.patch, z_dim=args.zdim, use_prefix_repr=True, dropout=0.0
    )).to(device)
    enc.eval()
    with torch.no_grad():
        mu, logvar, z, aux_enc = enc(seq_cmd, seq_arg, seq_mask, contour_ids, seq_topo, stage_sdf,
                                     sample=True, eps_std=1.0)

    # === 2.5 解码器 ===
    dec = Decoder(DecoderConfig(
        vocab_size=9, max_len=base.L, embed_dim=args.embed, z_dim=args.zdim,
        n_heads=args.heads, n_layers=args.dec_layers,
        patch_size=args.patch, use_pixel_cross_attn=True
    )).to(device)
    dec.eval()
    with torch.no_grad():
        logits_cmd, pred_arg, aux_dec = dec(z, stage_sdf, seq_mask)

    # === 掩码损失（演示）：只在前缀位计算 ===
    # lmc: [B,L]；lma: [B,L,4]；apm: [B,L] (arg-position mask，一般等价于 ~PAD)
    lmc, lma, apm = build_loss_masks(seq_cmd, seq_mask)  # bool
    V = 9
    # CE（命令）
    loss_ce = F.cross_entropy(
        logits_cmd[lmc], seq_cmd[lmc], reduction="mean"
    ) if lmc.any() else torch.tensor(0.0, device=device)
    # L1（坐标）
    if lma.any():
        loss_l1 = (pred_arg[lma] - seq_arg[lma]).abs().mean()
    else:
        loss_l1 = torch.tensor(0.0, device=device)

    print(f"z:           {tuple(z.shape)}")
    print(f"logits_cmd:  {tuple(logits_cmd.shape)}  pred_arg: {tuple(pred_arg.shape)}")
    print(f"[loss] CE: {float(loss_ce.item()):.4f}   L1: {float(loss_l1.item()):.4f}")
    # 健康检查：非前缀槽的输出是否被“驱动”为 0 取决于网络学习；这里只检查槽特征是否静音
    dec_slots = aux_dec["dec_slots"]
    if (~seq_mask.bool()).any():
        l2_out = dec_slots[~seq_mask.bool()].pow(2).sum(dim=-1).mean().item()
        print(f"[check] decoder slots outside-prefix L2 mean (not enforced): {l2_out:.6f}")
    else:
        print("[check] no outside-prefix tokens in this batch")

    
    from src.utils.vec_export import decode_to_commands, save_commands_txt, save_commands_svg
    import numpy as np

    # 若想反归一化回字体坐标，可准备 norms（有则传、无则置 None）
    norms = None
    if "norm" in items[0]:
        norms = np.stack([it["norm"].cpu().numpy() for it in items], axis=0)  # [B,3]

    cmds_batch = decode_to_commands(
        logits_cmd.detach().cpu(), pred_arg.detach().cpu(), seq_mask.detach().cpu(),
        norms=norms, clamp_coords=True, clamp_range=(-1.2,1.2), stop_at_end=True
    )
    cmds0 = cmds_batch[0]
    save_commands_txt("decoded_commands.txt", cmds0)
    save_commands_svg("decoded_preview.svg", cmds0, size=512, stroke=True, fill=False, evenodd=True)
    print("[saved] decoded_commands.txt / decoded_preview.svg")

