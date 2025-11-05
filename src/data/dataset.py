"""NPZDataset — 1.1 读取与标准化（含详细注释）

将你的构建器导出的 `.npz` 封装为 PyTorch Dataset。
本文件只做**读取与标准化**：
- ✅ 输出**完整**矢量序列和整图 SDF（不做分阶段、不做裁切）；
- ✅ 统一张量 dtype/形状，方便下游 1.2/1.3 继续处理；
- ❌ 不负责 RSM 前缀裁切与阶段渲染（见手册 §1.2/§1.3）。

# I/O 契约（单样本字典）
{
  "seq_cmd":  LongTensor[L],         # 指令 ID 序列（M/L/Q/Z/NEW/HOLE/END/PAD）
  "seq_arg":  FloatTensor[L,4],      # 每个 token 的参数；M/L: (x,y,0,0)，Q: (cx,cy,x,y)，其它全 0
  "seq_mask": BoolTensor[L],         # 有效位置（PAD=False）
  "seq_topo": ByteTensor[L] | None,  # 可选三位拓扑标记（NEW/HOLE/END）；可能不存在
  "contour_ids": LongTensor[L],      # 轮廓分组 ID；若不存在则用 -1 填充（后续 RSM 不能用）
  "img_dt_full": FloatTensor[1,H,W], # 整张 SDF（float16/32），此处仅添加通道维度
  "norm": FloatTensor[3],            # (s,tx,ty) 仿射，训练时用于保持渲染一致性
  "meta": {"id": int, "unicode": int}
}

# 关于 dtype
- `seq_cmd/contour_ids` → int64（LongTensor）便于 Embedding/索引。
- `seq_arg/img_dt_full/norm` → `float_dtype`（默认 float32；若内存紧可用 float16）。
- `seq_mask` → bool。

# 关于形状
- `seq_cmd`: (N,L)；`seq_arg`: (N,L,4)；`seq_mask`: (N,L)
- `img_dt`: (N,H,W)  → 本类返回时会扩成 (1,H,W)
- 其余可选键按存在性检查。
"""
from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


# 在 .npz 内**必须**存在的键（否则直接报错）
REQUIRED_KEYS = (
    "seq_cmd",   # (N,L)  integer
    "seq_arg",   # (N,L,4) float
    "seq_mask",  # (N,L)  bool/int
    "img_dt",    # (N,H,W) float  — SDF（与构建器一致）
)

# 在 .npz 内**可选**存在的键（存在则读取，不存在时给出默认/None）
OPTIONAL_KEYS = (
    "seq_topo",     # (N,L) uint8 — 三位拓扑标记（NEW/HOLE/END），建议提供
    "contour_ids",  # (N,L) int   — 轮廓 ID；RSM 需要它按轮廓累加
    "norm",         # (N,3) float — (s,tx,ty)
    "unicode",      # (N,)  int   — 仅作为日志/分桶
)


def _as_torch(x: np.ndarray, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    """将 numpy 数组安全地转为 torch 张量，并可选地转换 dtype。

    说明：
    - 这里不强制拷贝；`from_numpy` 会与底层共享内存（mmap 下零拷贝）。
    - 若指定 `dtype` 且与源不同，会触发一次 `.to(dtype)` 转换（这一步才可能拷贝）。
    """
    t = torch.from_numpy(np.asarray(x))
    if dtype is not None and t.dtype != dtype:
        t = t.to(dtype)
    return t


class NPZDataset(Dataset):
    """读取 `.npz` 并输出训练需要的标准张量。

    仅负责**完整样本**的装载（不做 RSM 前缀与阶段渲染）。

    参数
    ------
    npz_path : str
        数据文件路径。
    mmap : bool, default True
        是否使用 numpy 内存映射（`mmap_mode='r'`）。样本多时建议开启，节省常驻内存。
    use_seq_topo : bool, default True
        若 .npz 含 `seq_topo`，是否加载并返回该键（用于作为额外输入嵌入）。
    strict_shapes : bool, default True
        是否在初始化时**严格校验**各键形状（强烈建议 True，以便尽早暴露数据问题）。
    float_dtype : torch.dtype, default torch.float32
        浮点张量默认 dtype（作用于 `seq_arg`/`img_dt_full`/`norm`）。

    注意
    ------
    - 本类不会改动或裁切序列；RSM 的前缀与阶段图在后续组件（RSMBatcher/StageRenderer）处理。
    - 返回的 `img_dt_full` 已添加通道维度：[1,H,W]，方便直接喂给 CNN。
    """

    def __init__(
        self,
        npz_path: str,
        mmap: bool = True,
        use_seq_topo: bool = True,
        strict_shapes: bool = True,
        float_dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()

        # --- 1) 基本存在性检查与加载（可选 mmap） ---
        self.npz_path = os.path.abspath(npz_path)
        if not os.path.isfile(self.npz_path):
            raise FileNotFoundError(f"NPZ not found: {self.npz_path}")
        self._npz = np.load(self.npz_path, allow_pickle=False,
                             mmap_mode='r' if mmap else None)

        # --- 2) 键存在性检查：缺失直接报错，避免训练期再爆 ---
        missing = [k for k in REQUIRED_KEYS if k not in self._npz]
        if missing:
            raise KeyError(f"Missing required keys in NPZ: {missing}")

        # 仅当 .npz 含有该键且用户允许时才返回 seq_topo
        self.use_seq_topo = use_seq_topo and ("seq_topo" in self._npz)

        # --- 3) 抽取公共尺寸 N/L/H/W ---
        self.N = int(self._npz["seq_cmd"].shape[0])      # 样本数
        self.L = int(self._npz["seq_cmd"].shape[1])      # 最大序列长度 Lmax
        self.H, self.W = map(int, self._npz["img_dt"].shape[1:])  # SDF 尺寸

        # --- 4) （可选）严格形状校验 ---
        if strict_shapes:
            self._validate_shapes()

        # --- 5) 保存默认 dtype 设置 ---
        self.float_dtype = float_dtype

    # -----------------------------
    # 公共属性/魔法方法
    # -----------------------------
    @property
    def size(self) -> Tuple[int, int, int]:
        """返回 (N, L, H*W 的近似) 以便日志友好显示。"""
        return self.N, self.L, self.H * self.W

    def __len__(self) -> int:
        return self.N

    # -----------------------------
    # 取单样本：**不做任何前缀裁切**
    # -----------------------------
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """按索引返回**完整**样本字典。

        说明：
        - 本方法不会对序列做截断或阶段选择；
        - 返回的 `img_dt_full` 会扩通道维为 [1,H,W]；
        - 可选键按存在性安全读取；不存在时给出合理默认（例如 `contour_ids=-1`）。
        """
        if idx < 0 or idx >= self.N:
            raise IndexError(idx)

        # --- 必要键 ---
        seq_cmd = _as_torch(self._npz["seq_cmd"][idx], torch.int64)             # (L,)
        seq_arg = _as_torch(self._npz["seq_arg"][idx], self.float_dtype)        # (L,4)
        # 注意：seq_mask 可能在 .npz 中是 uint8/bool；统一转为 bool
        seq_mask = _as_torch(self._npz["seq_mask"][idx]).to(torch.bool)         # (L,)

        # --- 可选键：按存在性读取 ---
        seq_topo = None
        if self.use_seq_topo:
            seq_topo = _as_torch(self._npz["seq_topo"][idx], torch.uint8)       # (L,)

        # contour_ids 可能缺失：若缺失，则用 -1 填充（表示不可用于 RSM）
        if "contour_ids" in self._npz:
            contour_ids = _as_torch(self._npz["contour_ids"][idx], torch.int64)
        else:
            contour_ids = torch.full((self.L,), -1, dtype=torch.int64)

        # SDF：添加通道维度 [1,H,W]，便于 CNN 输入；保持 float_dtype 一致
        img_dt = _as_torch(self._npz["img_dt"][idx], self.float_dtype).unsqueeze(0)

        # norm / unicode：缺失时提供默认；norm 默认为恒等仿射 (1,0,0)
        if "norm" in self._npz:
            norm = _as_torch(self._npz["norm"][idx], self.float_dtype)          # (3,)
        else:
            norm = torch.tensor([1.0, 0.0, 0.0], dtype=self.float_dtype)
        uni_val = int(self._npz["unicode"][idx]) if "unicode" in self._npz else -1

        # 组装统一的输出字典（与手册约定一致）
        sample: Dict[str, Any] = {
            "seq_cmd": seq_cmd,
            "seq_arg": seq_arg,
            "seq_mask": seq_mask,
            "seq_topo": seq_topo,
            "contour_ids": contour_ids,
            "img_dt_full": img_dt,
            "norm": norm,
            "meta": {"id": int(idx), "unicode": uni_val},
        }
        return sample

    # -----------------------------
    # 形状/一致性检查：尽早在加载期报错
    # -----------------------------
    def _validate_shapes(self) -> None:
        """对关键键做严格的形状/类型检查，避免训练期才暴雷。"""
        N = self.N
        L = self.L
        H, W = self.H, self.W
        arr = self._npz

        def _must(name: str, cond: bool):
            if not cond:
                raise ValueError(
                    f"Invalid shape for '{name}' in {os.path.basename(self.npz_path)}: "
                    f"got {arr[name].shape}"
                )

        # 形状必须与约定一致
        _must("seq_arg", arr["seq_arg"].shape == (N, L, 4))
        _must("seq_mask", arr["seq_mask"].shape == (N, L))
        _must("img_dt", arr["img_dt"].shape == (N, H, W))

        if "seq_topo" in arr:
            _must("seq_topo", arr["seq_topo"].shape == (N, L))
        if "contour_ids" in arr:
            _must("contour_ids", arr["contour_ids"].shape == (N, L))
        if "norm" in arr:
            _must("norm", arr["norm"].shape == (N, 3))
        if "unicode" in arr:
            _must("unicode", arr["unicode"].shape == (N,))

        # dtype 友好检查（给出早期错误信号）
        if arr["seq_cmd"].dtype.kind not in ("i", "u"):
            raise TypeError("seq_cmd must be an integer array")
        if arr["seq_mask"].dtype.kind not in ("b", "i", "u"):
            raise TypeError("seq_mask must be bool/int array")
        if arr["img_dt"].dtype.kind not in ("f",):
            raise TypeError("img_dt must be a float array (typically float16)")


# ----------------------------------
# 便捷 CLI：快速自检 .npz 结构与样本形状
# ----------------------------------
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Quick inspect for NPZDataset")
    ap.add_argument("--npz", default='/data/kssczt/ztb/FontVector/VAE/data/msyh.npz', help="path to dataset .npz")
    ap.add_argument("--head", type=int, default=3, help="print first N samples info")
    ap.add_argument("--no-mmap", action="store_true", help="disable numpy mmap for debugging")
    args = ap.parse_args()

    ds = NPZDataset(args.npz, mmap=not args.no_mmap)
    print("Loaded:", ds.npz_path)
    print(f"N={ds.N}, L={ds.L}, H={ds.H}, W={ds.W}")

    for i in range(min(args.head, len(ds))):
        item = ds[i]
        # 仅打印张量形状/类型，避免刷屏
        shapes = {k: (tuple(v.shape) if torch.is_tensor(v) else type(v))
                  for k, v in item.items() if k not in ("meta",)}
        dtypes = {k: (v.dtype if torch.is_tensor(v) else None)
                  for k, v in item.items() if torch.is_tensor(v)}
        print(f"#{i}", shapes, item["meta"], dtypes)

  
