#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
StageRenderer — 1.3 在线阶段渲染（SDF 条件生成）

作用（与 1.1/1.2 配套）：
- 接收 1.2 RSMBatcher 产出的“矢量前缀” (seq_cmd/seq_arg/seq_mask) 和数据集图像分辨率，
  使用与构建器一致的规则（Skia even-odd 填充、y 轴取反、同像素映射、同 SDF 规范）
  在线渲出当前阶段的 SDF（单通道，范围 [-1,1]）。
- 该 SDF 作为像素条件喂给 VAE 的像素塔（以及后续潜空间扩散的条件塔）。

关键点：
- 完全复刻构建器/1.1/检查脚本的渲染流程：
  * 坐标系：输入 seq_arg 已是 ~[-1,1] 的归一化坐标；
  * Skia Path：even-odd 填充；绘制时对 y 取反；
  * 像素映射：scale = (size-1)/2 ; translate=(scale, scale)；
  * 阈值：<128 为前景；SDF = edt(inside) - edt(outside)；
  * 归一化：/ sdf_clip_px 后截断到 [-1,1]；dtype 可选 float32/float16。
- RSM 前缀保证“按轮廓累加”，因此前缀不会切断单条轮廓；本渲染器按 seq_mask=True 处绘制 M/L/Q/Z。
- NEW/HOLE/END/PAD token 不直接产生几何，忽略即可（洞由 even-odd 规则表现）。

输出：
- `render_item(...)` 返回 [1,H,W] 的 torch.Tensor；
- `render_batch(...)` 返回 [B,1,H,W]；
- CLI 提供快速对齐检查：在 100% 阶段下与 img_dt_full 做 MAE（<~0.02 视为对齐良好）。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional
import math

import numpy as np
import torch

# 仅在使用时导入（避免无依赖环境报错）
try:  # type: ignore
    import skia  # skia-python
except Exception as _e:  # 延迟到实际调用时报错更友好
    skia = None  # type: ignore

try:  # type: ignore
    from scipy.ndimage import distance_transform_edt as edt
except Exception as _e:
    edt = None  # type: ignore

# 与数据构建器一致的命令枚举
CMD_PAD, CMD_M, CMD_L, CMD_Q, CMD_T, CMD_Z, CMD_NEW, CMD_HOLE, CMD_END = range(9)


@dataclass
class StageRendererConfig:
    img_size: int  # 例如 128
    sdf_clip_px: float = 8.0
    out_dtype: torch.dtype = torch.float32  # 或 torch.float16


class StageRenderer:
    """把“矢量前缀”在线渲为阶段 SDF（与构建器一致）。"""

    def __init__(self, cfg: StageRendererConfig) -> None:
        self.cfg = cfg
        if skia is None:
            raise ImportError("skia-python 未安装。请先 pip install skia-python")
        if edt is None:
            raise ImportError("scipy 未安装或缺少 ndimage.edt。请先 pip install scipy")

    # ------------------------
    # 公共 API
    # ------------------------
    @torch.no_grad()
    def render_item(self, item: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        输入：
          item: 来自 1.2 RSMBatcher（或你手动构造）的单条样本字典，至少包含：
            - seq_cmd: [L] long
            - seq_arg: [L,4] float
            - seq_mask: [L] bool   —— RSM 的前缀掩码（或全 True 表示完整阶段）
        输出：
          stage_sdf: [1,H,W]，dtype = cfg.out_dtype，范围约 [-1,1]
        """
        seq_cmd = item["seq_cmd"].detach().cpu()
        seq_arg = item["seq_arg"].detach().cpu()
        seq_mask = item["seq_mask"].detach().cpu()
        sdf_np = _render_sdf_numpy(seq_cmd, seq_arg, seq_mask,
                                   size=self.cfg.img_size,
                                   sdf_clip_px=self.cfg.sdf_clip_px)
        sdf_t = torch.from_numpy(sdf_np).to(self.cfg.out_dtype).unsqueeze(0)  # [1,H,W]
        return sdf_t

    @torch.no_grad()
    def render_batch(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """对 batch 中每条样本渲出阶段 SDF。
        期望输入：
          - seq_cmd: [B,L]
          - seq_arg: [B,L,4]
          - seq_mask: [B,L]
        返回：
          - stage_sdf: [B,1,H,W]
        """
        seq_cmd = batch["seq_cmd"].detach().cpu()
        seq_arg = batch["seq_arg"].detach().cpu()
        seq_mask = batch["seq_mask"].detach().cpu()
        B = int(seq_cmd.shape[0])
        out = []
        for b in range(B):
            sdf_np = _render_sdf_numpy(seq_cmd[b], seq_arg[b], seq_mask[b],
                                       size=self.cfg.img_size,
                                       sdf_clip_px=self.cfg.sdf_clip_px)
            out.append(torch.from_numpy(sdf_np).unsqueeze(0))  # [1,H,W]
        return torch.stack(out, dim=0).to(self.cfg.out_dtype)  # [B,1,H,W]


# ------------------------
# 实际渲染实现（numpy + skia + edt）
# ------------------------

def _build_path_from_prefix(seq_cmd: torch.Tensor,
                            seq_arg: torch.Tensor,
                            seq_mask: torch.Tensor) -> "skia.Path":
    """把前缀（由 seq_mask=True 指定）构建成 Skia Path。
    规则：even-odd 填充；y 取反；仅处理 M/L/Q/Z；忽略 NEW/HOLE/END/PAD。
    注意：RSM 按轮廓累加，因此不会把单条轮廓截断；若传入“非轮廓边界前缀”，可能出现未闭合轮廓。
    """
    assert skia is not None
    path = skia.Path()
    path.setFillType(skia.PathFillType.kEvenOdd)

    L = int(seq_cmd.shape[0])
    started = False
    for i in range(L):
        if not bool(seq_mask[i]):
            continue
        cmd = int(seq_cmd[i])
        a = seq_arg[i].tolist()
        if cmd == CMD_M:
            x, y = float(a[0]), float(a[1])
            path.moveTo(x, -y)
            started = True
        elif cmd == CMD_L:
            x, y = float(a[0]), float(a[1])
            path.lineTo(x, -y)
        elif cmd == CMD_Q:
            cx, cy, x, y = map(float, a[:4])
            path.quadTo(cx, -cy, x, -y)
        elif cmd == CMD_Z:
            if started:
                path.close()
                started = False
        # 忽略：NEW/Hole/END/PAD/T（你的数据不含 T）
    return path


def _render_sdf_numpy(seq_cmd: torch.Tensor,
                      seq_arg: torch.Tensor,
                      seq_mask: torch.Tensor,
                      size: int,
                      sdf_clip_px: float) -> np.ndarray:
    """使用与构建器一致的设置渲出 SDF（numpy 数组 [H,W]）。"""
    assert skia is not None and edt is not None

    path = _build_path_from_prefix(seq_cmd, seq_arg, seq_mask)

    # 画到灰度图（白底、黑字、抗锯齿），坐标~[-1,1] → 像素
    surface = skia.Surface(size, size)
    canvas = surface.getCanvas()
    canvas.clear(skia.ColorWHITE)

    scale = (size - 1) / 2.0
    mat = skia.Matrix.Scale(scale, scale)
    mat.postTranslate(scale, scale)
    canvas.setMatrix(mat)

    paint = skia.Paint(Style=skia.Paint.kFill_Style,
                       Color=skia.ColorBLACK,
                       AntiAlias=True)
    canvas.drawPath(path, paint)
    img = surface.makeImageSnapshot().toarray()  # (H,W,4)
    gray = img[..., 0]

    # 二值化 + EDT → SDF
    inside = gray < 128
    outside = ~inside
    dist_in = edt(inside)
    dist_out = edt(outside)
    sdf = (dist_in - dist_out).astype(np.float32)
    sdf = np.clip(sdf / float(sdf_clip_px), -1.0, 1.0)
    return sdf  # [H,W], float32


# ------------------------
# 便捷 CLI：从 .npz → (可选 RSM) → 阶段渲染 →（若是 100% 阶段）对齐 MAE
# ------------------------
if __name__ == "__main__":
    import argparse
    import os

    # 支持两种导入：作为包运行(-m) 或 同目录运行
    try:
        from .dataset import NPZDataset  # type: ignore
        from .rsm_batcher import RSMBatcher, RSMConfig  # type: ignore
    except Exception:
        from dataset import NPZDataset  # type: ignore
        from rsm_batcher import RSMBatcher, RSMConfig  # type: ignore

    ap = argparse.ArgumentParser(description="1.3 StageRenderer: render stage SDF, compare (full), and optionally save PNGs")
    ap.add_argument("--npz", default='/data/kssczt/ztb/FontVector/VAE/data/msyh.npz', help="path to dataset .npz")
    ap.add_argument("--n", type=int, default=3, help="print first N samples")
    ap.add_argument("--img-size", type=int, default=128, help="render size; default: dataset H")
    ap.add_argument("--sdf-clip-px", type=float, default=8.0, help="SDF clip (pixels)")
    ap.add_argument("--dtype", choices=["f16", "f32"], default="f32", help="output dtype")
    ap.add_argument("--use-rsm", default=True, help="use RSM stage prefix (default off = full stage)")
    ap.add_argument("--stages", type=str, default="0.25,0.5,0.75,1.0", help="RSM stage ratios if --use-rsm")
    ap.add_argument("--probs", type=str, default="0.15,0.25,0.25,0.35", help="RSM probs if --use-rsm")
    # —— 新增：可视化输出 ——
    ap.add_argument("--save-dir", type=str, default='out', help="if set, save rendered PNGs to this directory")
    ap.add_argument("--save-prefix", type=str, default="vis", help="filename prefix for saved images")
    args = ap.parse_args()

    # 可选依赖：PIL 用于保存 PNG（若没有则退化为 .npy）
    try:
        from PIL import Image  # type: ignore
    except Exception:
        Image = None  # type: ignore
        print("[warn] PIL not installed; will save .npy arrays instead of PNG if --save-dir is set.")

    def _to_uint8_img(t: torch.Tensor) -> "np.ndarray":
        """将 [-1,1] SDF tensor [1,H,W] 或 [H,W] 映射到 uint8 灰度图 [H,W]。"""
        x = t.detach().cpu()
        if x.ndim == 3 and x.shape[0] == 1:
            x = x[0]
        x = ((x + 1.0) * 0.5).clamp_(0, 1)  # [0,1]
        x = (x * 255.0).round().to(torch.uint8)
        return x.numpy()

    def _save_image(arr: "np.ndarray", path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if Image is not None:
            Image.fromarray(arr).save(path)
        else:
            # 退化为保存 numpy 数组
            np.save(path.replace(".png", ".npy"), arr)

    base = NPZDataset(args.npz, mmap=True, use_seq_topo=True, strict_shapes=True)
    H, W = base.H, base.W
    size = args.img_size or H
    dtype = torch.float16 if args.dtype == "f16" else torch.float32

    # 数据管道：可选 RSM（否则强制 full-stage）
    if args.use_rsm:
        stages = tuple(float(x) for x in args.stages.split(','))
        probs = tuple(float(x) for x in args.probs.split(','))
        ds = RSMBatcher(base, RSMConfig(rsm_stages=stages, rsm_probs=probs,
                                        enforce_contour_ids=True, allow_token_prefix_fallback=False))
        print("[info] using RSM (stage prefix)")
    else:
        ds = base
        print("[info] using FULL stage (seq_mask = original valid mask)")

    renderer = StageRenderer(StageRendererConfig(img_size=size, sdf_clip_px=args.sdf_clip_px, out_dtype=dtype))

    print(f"Loaded {base.npz_path}")
    print(f"N={len(ds)}, L={ds.L}, H={H}, W={W}, render_size={size}")

    maes = []
    for i in range(min(args.n, len(ds))):
        item = ds[i]
        sdf = renderer.render_item(item)  # [1,H,W]

        if not args.use_rsm and "img_dt_full" in item:
            gt = item["img_dt_full"].to(dtype)
            mae = torch.mean(torch.abs(sdf - gt)).item()
            maes.append(mae)
            print(f"#{i}  MAE(full)={mae:.6f}")
        else:
            n_valid = int(item["seq_mask"].sum())
            stage = int(item.get("stage_id", torch.tensor(100)).item())
            print(f"#{i}  stage={stage}%  prefix_tokens={n_valid}  (no GT MAE for stage)")

        # —— 保存可视化 ——
        if args.save_dir is not None:
            # 预测图
            sdf_u8 = _to_uint8_img(sdf)
            if args.use_rsm:
                stage = int(item.get("stage_id", torch.tensor(100)).item())
                out_pred = os.path.join(args.save_dir, f"{args.save_prefix}_idx{i}_stage{stage}_pred.png")
            else:
                out_pred = os.path.join(args.save_dir, f"{args.save_prefix}_idx{i}_full_pred.png")
            _save_image(sdf_u8, out_pred)

            # full-stage 下另存 GT 与差异图
            if not args.use_rsm and "img_dt_full" in item:
                gt_u8 = _to_uint8_img(item["img_dt_full"])  # [1,H,W]
                out_gt = os.path.join(args.save_dir, f"{args.save_prefix}_idx{i}_full_gt.png")
                _save_image(gt_u8, out_gt)

                # 可视化绝对误差（归一化到 0..255）
                diff = torch.abs(sdf.to(torch.float32) - item["img_dt_full"].to(torch.float32))  # [1,H,W]
                diff = (diff / 2.0).clamp(0, 1)  # SDF 差值范围大致在 [0,2]
                diff_u8 = _to_uint8_img(diff)
                out_diff = os.path.join(args.save_dir, f"{args.save_prefix}_idx{i}_full_absdiff.png")
                _save_image(diff_u8, out_diff)

                print(f"    saved: {out_pred} | {out_gt} | {out_diff}")
            else:
                print(f"    saved: {out_pred}")

    if maes:
        arr = torch.tensor(maes, dtype=torch.float32)
        print(f"[stats] AVG_MAE={arr.mean():.6f}  min={arr.min():.6f}  max={arr.max():.6f}  n={len(maes)}")
