# -*- coding: utf-8 -*-
"""
矢量命令恢复与导出工具（独立于模型）
- 从解码器输出 (logits_cmd, pred_arg, seq_mask[, norms]) 恢复标准矢量命令流
- 导出 .txt 和 .svg 方便检查/可视化

约定（与数据集一致）：
- vocab: 0:<PAD>, 1:M, 2:L, 3:Q, 4:T(理应不出现), 5:Z, 6:<NEW_CNT>, 7:<HOLE>, 8:<END>
- pred_arg: [B,L,4]；M/L 取前两维 (x,y)，Q 取四维 (cx,cy,x,y)

使用示例（见文件底部的 usage 注释）。
"""
from typing import List, Tuple, Dict, Optional, Sequence, Union
import math

try:
    import torch
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False


try:
    import skia
    import numpy as np
    _HAS_SKIA = True
except Exception:
    _HAS_SKIA = False

# ---------------------------
# 词表（可在 decode_to_commands 里通过 id2cmd 覆盖）
# ---------------------------
VOCAB = {"<PAD>":0,"M":1,"L":2,"Q":3,"T":4,"Z":5,"<NEW_CNT>":6,"<HOLE>":7,"<END>":8}
ID2CMD = {v:k for k,v in VOCAB.items()}

# ---------------------------
# 基础工具
# ---------------------------
def _as_numpy(x):
    """支持 torch / numpy / list → numpy-like 序列；只在索引/float()处使用，不强依赖框架。"""
    if _HAS_TORCH and isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x

def _clamp(v: float, lo=-1.2, hi=1.2) -> float:
    return float(v if (v <= hi and v >= lo) else (hi if v > hi else lo))

def _denorm_point(x: float, y: float, norm: Sequence[float]) -> Tuple[float, float]:
    """
    把归一化坐标 (x,y) 反归一化回字体坐标系；norm=(s,tx,ty)，对应数据集构建时的 normalize(s,tx,ty)
    归一化公式：x' = s*x + tx, y' = s*y + ty  →  反解：x = (x'-tx)/s
    """
    s, tx, ty = float(norm[0]), float(norm[1]), float(norm[2])
    if abs(s) < 1e-8:
        return float(x), float(y)
    return float((x - tx) / s), float((y - ty) / s)

# ---------------------------
# 1) 从网络输出恢复命令流
# ---------------------------
def decode_to_commands(
    logits_cmd,                      # [B,L,V] (torch/numpy 都可)
    pred_arg,                        # [B,L,4]
    seq_mask,                        # [B,L] True=有效位（通常=前缀）
    norms: Optional[Union[Sequence[Sequence[float]], "torch.Tensor"]] = None,  # [B,3], (s,tx,ty)
    clamp_coords: bool = True,
    clamp_range: Tuple[float,float] = (-1.2, 1.2),
    stop_at_end: bool = True,
    id2cmd: Optional[Dict[int,str]] = None,
) -> List[List[Tuple[str, Tuple]]]:
    """
    返回：每个 batch 一份命令流 List[(op, args)]，如 [("M",(x,y)),("Q",(cx,cy,x,y)),("Z",(0,0)),...]

    注意：
    - 默认遇到 <END> 提前停止（stop_at_end=True）
    - T 理论上不会出现，若出现按 Q 处理
    - 如果提供 norms，会对所有坐标做反归一化
    """
    id2cmd = id2cmd or ID2CMD

    # to numpy-like for indexing/math; keep originals for caller
    lg = _as_numpy(logits_cmd)
    pa = _as_numpy(pred_arg)
    ms = _as_numpy(seq_mask)
    nm = _as_numpy(norms) if norms is not None else None

    import numpy as np
    ids = lg.argmax(axis=-1)  # [B,L]
    B, L = ids.shape
    out_all: List[List[Tuple[str, Tuple]]] = []

    lo, hi = clamp_range
    for b in range(B):
        cmds: List[Tuple[str, Tuple]] = []
        for t in range(L):
            if not bool(ms[b, t]):
                continue
            cid = int(ids[b, t])
            name = id2cmd.get(cid, "?")

            # 坐标（默认在归一化坐标系）
            x1, y1 = float(pa[b, t, 0]), float(pa[b, t, 1])
            cx, cy = float(pa[b, t, 0]), float(pa[b, t, 1])
            x2, y2 = float(pa[b, t, 2]), float(pa[b, t, 3])

            if clamp_coords:
                x1, y1 = _clamp(x1, lo, hi), _clamp(y1, lo, hi)
                cx, cy = _clamp(cx, lo, hi), _clamp(cy, lo, hi)
                x2, y2 = _clamp(x2, lo, hi), _clamp(y2, lo, hi)

            if nm is not None:
                x1, y1 = _denorm_point(x1, y1, nm[b])
                cx, cy = _denorm_point(cx, cy, nm[b])
                x2, y2 = _denorm_point(x2, y2, nm[b])

            # 生成命令
            if name == "M":
                cmds.append(("M", (x1, y1)))
            elif name == "L":
                cmds.append(("L", (x1, y1)))
            elif name in ("Q", "T"):  # T 当 Q 处理
                cmds.append(("Q", (cx, cy, x2, y2)))
            elif name == "Z":
                cmds.append(("Z", (0.0, 0.0)))
            elif name == "<NEW_CNT>":
                cmds.append(("<NEW_CNT>", (0.0, 0.0)))
            elif name == "<HOLE>":
                cmds.append(("<HOLE>", (0.0, 0.0)))
            elif name == "<END>":
                cmds.append(("<END>", (0.0, 0.0)))
                if stop_at_end:
                    break
            else:
                # <PAD> 或未知：忽略
                continue

        out_all.append(cmds)
    return out_all

# ---------------------------
# 2) 导出 .txt / .svg
# ---------------------------
def save_commands_txt(path: str, cmds_one: List[Tuple[str, Tuple]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for op, args in cmds_one:
            if op in ("M", "L"):
                x, y = args
                f.write(f"{op} {x:.6f} {y:.6f}\n")
            elif op == "Q":
                cx, cy, x, y = args
                f.write(f"{op} {cx:.6f} {cy:.6f} {x:.6f} {y:.6f}\n")
            elif op in ("Z", "<NEW_CNT>", "<HOLE>", "<END>"):
                f.write(f"{op}\n")

def commands_to_svg_path(cmds_one: List[Tuple[str, Tuple]], close_on_new: bool = True) -> str:
    """
    生成 SVG path 'd' 字符串（不含 <svg> 包裹）。
    简化规则：
    - <NEW_CNT>/<HOLE> 视为“收尾并开新子路径”（需要调用者自行设置 fill-rule=evenodd 保持洞）
    """
    path_d = []
    opened = False
    for op, args in cmds_one:
        if op == "M":
            x, y = args
            path_d.append(f"M {x:.3f} {y:.3f}")
            opened = True
        elif op == "L":
            x, y = args
            path_d.append(f"L {x:.3f} {y:.3f}")
        elif op == "Q":
            cx, cy, x, y = args
            path_d.append(f"Q {cx:.3f} {cy:.3f} {x:.3f} {y:.3f}")
        elif op == "Z":
            path_d.append("Z")
            opened = False
        elif op in ("<NEW_CNT>", "<HOLE>"):
            if close_on_new and opened:
                path_d.append("Z")
            opened = False
        elif op == "<END>":
            if close_on_new and opened:
                path_d.append("Z")
            break
        else:
            continue
    return " ".join(path_d)

def save_commands_png(
    path: str,
    cmds_one: List[Tuple[str, Tuple]],
    size: int = 512,
    stroke: bool = True,
    fill: bool = True,
    stroke_width: float = 2.0,
    bg_color: Tuple[int, int, int, int] = (255, 255, 255, 255),
    fg_color: Tuple[int, int, int, int] = (0, 0, 0, 255),
) -> None:
    """
    使用 skia 将矢量命令渲染为 PNG 位图。
    
    Args:
        path: 输出 PNG 文件路径
        cmds_one: 矢量命令列表
        size: 图像尺寸（正方形）
        stroke: 是否绘制描边
        fill: 是否填充
        stroke_width: 描边宽度
        bg_color: 背景色 (R,G,B,A)
        fg_color: 前景色 (R,G,B,A)
    """
    if not _HAS_SKIA:
        raise ImportError("需要 skia-python 来渲染 PNG。请运行: pip install skia-python")
    
    # 计算边界框和变换
    xs, ys = [], []
    for op, a in cmds_one:
        if op in ("M", "L"):
            xs.extend([a[0]]); ys.extend([a[1]])
        elif op == "Q":
            xs.extend([a[0], a[2]]); ys.extend([a[1], a[3]])
    
    if len(xs) < 2:
        # 空路径，创建空白图像
        surface = skia.Surface(size, size)
        canvas = surface.getCanvas()
        canvas.clear(skia.Color(*bg_color))
        image = surface.makeImageSnapshot()
        image.save(path, skia.kPNG)
        return
    
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    w = max(1e-6, xmax - xmin)
    h = max(1e-6, ymax - ymin)
    
    # 计算缩放和平移，留出边距
    margin = 0.1
    scale = (1.0 - 2*margin) * size / max(w, h)
    tx = (size - scale * (xmin + xmax)) * 0.5
    ty = (size - scale * (ymin + ymax)) * 0.5
    
    # 创建画布
    surface = skia.Surface(size, size)
    canvas = surface.getCanvas()
    canvas.clear(skia.Color(*bg_color))
    
    # 应用变换
    canvas.translate(tx, ty)
    canvas.scale(scale, scale)
    
    # 创建路径
    path_obj = skia.Path()
    path_obj.setFillType(skia.PathFillType.kEvenOdd)  # 支持洞
    
    for op, args in cmds_one:
        if op == "M":
            path_obj.moveTo(args[0], args[1])
        elif op == "L":
            path_obj.lineTo(args[0], args[1])
        elif op == "Q":
            cx, cy, x, y = args
            path_obj.quadTo(cx, cy, x, y)
        elif op == "Z":
            path_obj.close()
        elif op in ("<NEW_CNT>", "<HOLE>"):
            path_obj.close()
        elif op == "<END>":
            path_obj.close()
            break
    
    # 绘制
    paint = skia.Paint()
    paint.setAntiAlias(True)
    
    if fill:
        paint.setStyle(skia.Paint.kFill_Style)
        paint.setColor(skia.Color(*fg_color))
        canvas.drawPath(path_obj, paint)
    
    if stroke:
        paint.setStyle(skia.Paint.kStroke_Style)
        paint.setStrokeWidth(stroke_width / scale)  # 调整描边宽度
        paint.setColor(skia.Color(*fg_color))
        canvas.drawPath(path_obj, paint)
    
    # 保存
    image = surface.makeImageSnapshot()
    image.save(path, skia.kPNG)


def save_commands_svg(
    path: str,
    cmds_one: List[Tuple[str, Tuple]],
    size: int = 512,
    stroke: bool = True,
    fill: bool = False,
    evenodd: bool = True,
    view_transform: Optional[Tuple[float, float, float, float]] = None,
) -> None:
    """
    把命令流简单渲染为 SVG。
    - 默认视为“字体坐标系”（非归一化），不做额外缩放；你也可以通过 view_transform=(sx,sy,tx,ty) 显式设定
    - 使用 fill-rule="evenodd" 以支持洞
    """
    # 默认视口变换：把字体坐标（通常千单位量级）放进 512x512 画布（粗略可视化）
    if view_transform is None:
        # 粗糙的自适应：基于命令粗略估 bbox，居中等比缩放
        xs, ys = [], []
        for op, a in cmds_one:
            if op in ("M", "L"):
                xs.extend([a[0]]); ys.extend([a[1]])
            elif op == "Q":
                xs.extend([a[0], a[2]]); ys.extend([a[1], a[3]])
        if len(xs) >= 2:
            xmin, xmax = min(xs), max(xs)
            ymin, ymax = min(ys), max(ys)
            w = max(1e-6, xmax - xmin); h = max(1e-6, ymax - ymin)
            s = 0.8 * size / max(w, h)
            tx = (size - s * (xmin + xmax)) * 0.5
            ty = (size - s * (ymin + ymax)) * 0.5
            view_transform = (s, -s, tx, size - ty)  # 颠倒 y 轴显示更直观
        else:
            view_transform = (1.0, -1.0, size*0.5, size*0.5)

    sx, sy, tx, ty = view_transform

    # 应用视口变换
    def map_xy(x, y):
        X = sx * x + tx
        Y = sy * y + ty
        return X, Y

    # 生成 path，应用视口变换
    d_parts = []
    for op, args in cmds_one:
        if op == "M":
            x, y = map_xy(args[0], args[1]); d_parts.append(f"M {x:.2f} {y:.2f}")
        elif op == "L":
            x, y = map_xy(args[0], args[1]); d_parts.append(f"L {x:.2f} {y:.2f}")
        elif op == "Q":
            cx, cy, x, y = args
            cx, cy = map_xy(cx, cy); x, y = map_xy(x, y)
            d_parts.append(f"Q {cx:.2f} {cy:.2f} {x:.2f} {y:.2f}")
        elif op == "Z":
            d_parts.append("Z")
        elif op in ("<NEW_CNT>", "<HOLE>"):
            d_parts.append("Z")
        elif op == "<END>":
            d_parts.append("Z")
            break
    d = " ".join(d_parts)

    fill_rule = 'evenodd' if evenodd else 'nonzero'
    stroke_attr = 'stroke="black" stroke-width="1"' if stroke else 'stroke="none"'
    fill_attr = 'fill="black"' if fill else 'fill="none"'
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}">
  <path d="{d}" fill-rule="{fill_rule}" {stroke_attr} {fill_attr} />
</svg>'''
    with open(path, "w", encoding="utf-8") as f:
        f.write(svg)


# ---------------------------
# usage（示例，复制到你的训练/验证脚本里即可）
# ---------------------------
"""
from src.utils.vec_export import decode_to_commands, save_commands_txt, save_commands_svg

# logits_cmd[B,L,V], pred_arg[B,L,4], seq_mask[B,L] 来自解码器
# 若要反归一化回字体坐标，准备 norms[B,3] = (s,tx,ty)，可从数据集 item["norm"] 收集

cmds_batch = decode_to_commands(
    logits_cmd.detach().cpu(), pred_arg.detach().cpu(), seq_mask.detach().cpu(),
    norms=None,                 # 或 np.stack([it["norm"].cpu().numpy() for it in items],0)
    clamp_coords=True, clamp_range=(-1.2, 1.2), stop_at_end=True
)

# 保存第一个样本
cmds0 = cmds_batch[0]
save_commands_txt("decoded_commands.txt", cmds0)
save_commands_svg("decoded_preview.svg", cmds0, size=512, stroke=True, fill=False, evenodd=True)
"""
