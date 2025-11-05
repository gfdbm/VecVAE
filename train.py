# -*- coding: utf-8 -*-
"""
Train VP-VAE (with file logging + preview export + train/val split)
- 训练日志：CSV + JSONL
- 过程可视化：定期把预测还原为矢量命令（txt/png，多样本）
- 数据分割：支持训练集/验证集随机分割（默认 10% 验证集）

可视化导出说明：
==============
1. 初始可视化（epoch 0）：
   - 保存未训练模型在验证集上的输出，用于对比
   - 位置: {out_dir}/eval_previews/preview_eval_ep0_*.{txt,png}

2. 训练过程可视化（每 export_every 步）：
   - 保存训练批次（训练集）的预测结果
   - 频率：--export-every 参数控制（默认每 1000 步）
   - 位置: {out_dir}/preview_train_ep{N}_step{M}_b{I}.{txt,png}

3. 每个 epoch 评估可视化：
   - 保存验证集样本的预测结果
   - 频率：--eval-every 参数控制（默认每 1 个 epoch，最后一轮总会保存）
   - 位置: {out_dir}/eval_previews/preview_eval_ep{N}_*.{txt,png}

4. 最终可视化（训练结束）：
   - 保存验证集 16 个样本的最终预测结果
   - 位置: {out_dir}/eval_previews/preview_eval_ep999_*.{txt,png}

数据分割说明：
============
- 使用 --val-split 参数控制验证集比例（默认 0.1 = 10%）
- 随机分割基于 --seed 参数，保证可复现
- 训练只在训练集上进行，评估只在验证集上进行
- PNG 文件中 eval_previews 文件夹包含的都是验证集样本

可视化频率控制：
==============
- --export-every N：训练过程中每 N 步保存一次训练集预览（默认 1000）
- --eval-every N：每 N 个 epoch 保存一次验证集预览（默认 1）
  例如：--eval-every 5 表示每 5 个 epoch 保存一次 PNG

使用说明：
- TXT 文件：纯文本矢量命令，便于调试
- PNG 文件：位图图像，直接查看字体渲染效果（白底黑字，512x512）
"""

import os, sys, time, json, random, csv
from pathlib import Path
from contextlib import nullcontext
from datetime import datetime

import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast  # 新接口，替代 torch.cuda.amp.*
from tqdm import tqdm, trange

# === 让根目录下的 src/* 可被导入 ===
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# === 数据与工具 ===
from src.data.dataset import NPZDataset
from src.data.rsm_batcher import RSMBatcher, RSMConfig
from src.data.stage_renderer import StageRenderer, StageRendererConfig
from src.data.masks import build_loss_masks

# === 模型 ===
from src.model.encoder import VpVaeEncoder, VpVaeEncoderConfig
from src.model.decoder import Decoder, DecoderConfig

# === 损失 ===
from src.losses.losses import compute_vae_losses, LossConfig, BetaWarmup

# === 矢量导出（还原命令 + 保存）===
from src.utils.vec_export import decode_to_commands, save_commands_txt, save_commands_png


# ------------------------- 实用：日志器 -------------------------
class CsvLogger:
    """把训练指标落到 CSV；首次写入表头。"""
    def __init__(self, path: Path, fieldnames):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fieldnames = list(fieldnames)
        if not self.path.exists():
            with self.path.open("w", newline="") as f:
                csv.writer(f).writerow(self.fieldnames)

    def write(self, row: dict):
        with self.path.open("a", newline="") as f:
            w = csv.writer(f)
            w.writerow([row.get(k, "") for k in self.fieldnames])


def append_jsonl(path: Path, obj: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


# ------------------------- 其他小工具 -------------------------
def set_seed(seed: int = 42):
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def to_device_batch(items, device):
    """把一批样本(list[dict])堆叠并搬到 device；非张量字段保持原样（列表）。"""
    out = {}
    keys = list(items[0].keys())
    for k in keys:
        vals = [it[k] for it in items]
        out[k] = torch.stack(vals, 0).to(device) if torch.is_tensor(vals[0]) else vals
    return out

def render_stage_sdf_batch(renderer: StageRenderer, items, device):
    imgs = [renderer.render_item(it) for it in items]
    return torch.stack(imgs, 0).to(device)  # [B,1,H,W]

def save_ckpt(path, enc, dec, opt, scaler, step_epoch):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "enc": enc.state_dict(),
        "dec": dec.state_dict(),
        "opt": opt.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "step": step_epoch[0],
        "epoch": step_epoch[1],
    }, path)

def load_ckpt(path, enc, dec, opt=None, scaler=None, map_location="cpu"):
    ck = torch.load(path, map_location=map_location)
    enc.load_state_dict(ck["enc"], strict=True)
    dec.load_state_dict(ck["dec"], strict=True)
    if opt is not None and ck.get("opt") is not None:
        opt.load_state_dict(ck["opt"])
    if scaler is not None and ck.get("scaler") is not None:
        scaler.load_state_dict(ck["scaler"])
    return ck.get("step", 0), ck.get("epoch", 0)

def compute_class_weight_epoch(items, num_classes=9, device="cpu"):
    """
    固定类别权重（只算一次）：按"当前训练管道（含 RSM）"全量遍历，
    仅统计前缀有效位（lmc=True）的命令分布 → 计算 CE 的类别权重。
    修正：build_loss_masks 在单样本可能返回 [1,L] 掩码，这里按需 squeeze 一下。
    """
    cnt = torch.zeros(num_classes, dtype=torch.float64)
    for it in tqdm(items, desc="Computing class weights", ncols=100):
        lmc, _, _ = build_loss_masks(it["seq_cmd"], it["seq_mask"])  # [L] 或 [1,L]
        if lmc.dim() == 2 and lmc.size(0) == 1:  # 单样本返回 [1,L] 的情况
            lmc = lmc.squeeze(0)
        ids = it["seq_cmd"][lmc.bool()]          # 一维布尔掩码索引
        binc = torch.bincount(ids.cpu(), minlength=num_classes).to(cnt.dtype)
        cnt[:len(binc)] += binc
    inv = 1.0 / (cnt + 1e-3)
    w = inv * (num_classes / max(inv.sum().item(), 1.0))  # 归一到均值≈1
    return w.to(device).float()

CMD_NAMES = ["PAD","M","L","Q","T","Z","NEW","HOLE","END"]


# ------------------------- 可视化导出 -------------------------
def export_previews(out_dir: Path, prefix: str, step: int, epoch: int,
                    logits_cmd, pred_arg, seq_mask, items, max_n: int = 4):
    """
    把当前 batch 的预测还原为矢量命令，保存 TXT+PNG。
    - logits_cmd: [B,L,V]
    - pred_arg:   [B,L,4]
    - seq_mask:   [B,L]
    - items:      原 batch 的 list[dict]（为了取 norm 做反归一化）
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    B = logits_cmd.size(0)
    n = min(max_n, B)
    # 可选：反归一化参数 (s, tx, ty)
    norms = None
    if "norm" in items[0]:
        import numpy as np
        norms = np.stack([it["norm"].cpu().numpy() for it in items], 0)  # [B,3]
    # 还原命令（内部做 arg 门控 + 取 argmax）
    cmds_list = decode_to_commands(
        logits_cmd.cpu(), pred_arg.cpu(), seq_mask.cpu(), norms=norms
    )
    for i in range(n):
        tag = f"{prefix}_ep{epoch}_step{step}_b{i}"
        txt = out_dir / f"preview_{tag}.txt"
        png = out_dir / f"preview_{tag}.png"
        save_commands_txt(str(txt), cmds_list[i])
        save_commands_png(str(png), cmds_list[i], size=512, stroke=False, fill=True, 
                         stroke_width=1.5, bg_color=(255,255,255,255), fg_color=(0,0,0,255))


# ------------------------- 训练 / 评估 -------------------------
def train_one_epoch(ds, renderer, enc, dec, opt, scaler, loss_cfg, beta_sched,
                    batch_size, device, log_every, grad_clip=1.0,
                    export_every=None, out_dir=None, step0=0,
                    class_weight=None, epoch_idx=0,
                    csv_logger: CsvLogger = None, jsonl_path: Path = None,
                    preview_n: int = 4):
    enc.train(); dec.train()
    n = len(ds)
    indices = torch.randperm(n).tolist()
    t0 = time.time()
    running = {"loss": 0.0, "ce": 0.0, "l1": 0.0, "kl": 0.0}
    step = step0

    # 创建进度条
    pbar = tqdm(range(0, n, batch_size), 
                desc=f"Epoch {epoch_idx}", 
                ncols=120,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
    
    for s in pbar:
        step += 1
        idxs = indices[s: s + batch_size]
        items = [ds[i] for i in idxs]
        batch = to_device_batch(items, device)
        stage_sdf = render_stage_sdf_batch(renderer, items, device)  # [B,1,H,W]

        seq_cmd, seq_arg = batch["seq_cmd"], batch["seq_arg"]
        seq_mask = batch["seq_mask"].bool()
        contour_ids, seq_topo = batch["contour_ids"], batch["seq_topo"]

        loss_cfg.beta = beta_sched.step()
        opt.zero_grad(set_to_none=True)

        use_cuda_amp = (scaler is not None and device.type == "cuda")
        with (autocast("cuda", dtype=torch.float16) if use_cuda_amp else nullcontext()):
            # 编码器：前缀 + 像素条件 → μ/logσ → 采样 z
            mu, logvar, z, _ = enc(
                seq_cmd, seq_arg, seq_mask, contour_ids, seq_topo, stage_sdf,
                sample=True, eps_std=1.0
            )
            # 解码器：z + 像素条件 → 预测命令/坐标
            logits_cmd, pred_arg, _ = dec(z, stage_sdf, seq_mask)

            # 基础损失
            total, stats = compute_vae_losses(
                logits_cmd, pred_arg, mu, logvar, seq_cmd, seq_arg, seq_mask, cfg=loss_cfg
            )
            # 用固定类别权重替换 CE（只影响 CE；L1/KL 不变）
            if class_weight is not None:
                lmc, _, _ = build_loss_masks(seq_cmd, seq_mask)  # [B,L]
                ce = F.cross_entropy(
                    logits_cmd[lmc], seq_cmd[lmc],
                    weight=class_weight, reduction="mean"
                ) if lmc.any() else torch.zeros([], device=total.device)
                # ✅ 修复：正确应用配置的损失权重
                total = loss_cfg.ce_weight * ce + loss_cfg.l1_weight * stats["loss_l1"].to(total.device) + loss_cfg.beta * stats["loss_kl"].to(total.device)
                stats["loss_ce"] = ce.detach()
                stats["loss_total"] = total.detach()

        # 反传 + 更新
        if scaler is not None and device.type == "cuda":
            scaler.scale(total).backward()
            if grad_clip and grad_clip > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(list(enc.parameters()) + list(dec.parameters()), grad_clip)
            scaler.step(opt); scaler.update()
        else:
            total.backward()
            if grad_clip and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(list(enc.parameters()) + list(dec.parameters()), grad_clip)
            opt.step()

        # 累计
        running["loss"] += float(stats["loss_total"])
        running["ce"]  += float(stats["loss_ce"])
        running["l1"]  += float(stats["loss_l1"])
        running["kl"]  += float(stats["loss_kl"])

        # 更新进度条显示（实时显示最新损失）
        pbar.set_postfix({
            'loss': f"{running['loss']/max(1, step-step0):.4f}",
            'CE': f"{running['ce']/max(1, step-step0):.4f}",
            'L1': f"{running['l1']/max(1, step-step0):.4f}",
            'KL': f"{running['kl']/max(1, step-step0):.4f}",
            'β': f"{loss_cfg.beta:.3f}"
        })

        # 打印 + 落盘日志
        if step % log_every == 0:
            it_time = (time.time() - t0) / max(1, log_every)
            ips = 1.0 / it_time
            lr = opt.param_groups[0]["lr"]
            msg = (f"[train] ep={epoch_idx} step={step}  "
                   f"loss={running['loss']/log_every:.4f}  "
                   f"CE={running['ce']/log_every:.4f}  "
                   f"L1={running['l1']/log_every:.4f}  "
                   f"KL={running['kl']/log_every:.4f}  "
                   f"beta={loss_cfg.beta:.3f}  lr={lr:.2e}  {it_time:.2f}s/it")
            pbar.write(msg)  # 使用 tqdm.write 而不是 print，避免干扰进度条

            # CSV
            if csv_logger is not None:
                csv_logger.write({
                    "time": datetime.now().isoformat(timespec="seconds"),
                    "epoch": epoch_idx,
                    "step": step,
                    "loss": round(running['loss']/log_every, 6),
                    "ce": round(running['ce']/log_every, 6),
                    "l1": round(running['l1']/log_every, 6),
                    "kl": round(running['kl']/log_every, 6),
                    "beta": round(loss_cfg.beta, 6),
                    "lr": lr,
                    "ips": ips,
                })
            # JSONL
            if jsonl_path is not None:
                append_jsonl(jsonl_path, {
                    "time": datetime.now().isoformat(timespec="seconds"),
                    "epoch": epoch_idx, "step": step,
                    "loss": running['loss']/log_every,
                    "ce": running['ce']/log_every,
                    "l1": running['l1']/log_every,
                    "kl": running['kl']/log_every,
                    "beta": float(loss_cfg.beta), "lr": float(lr), "ips": ips,
                })

            t0 = time.time()
            running = {"loss": 0.0, "ce": 0.0, "l1": 0.0, "kl": 0.0}

        # 可视化导出：用 μ 解码更稳
        if export_every and out_dir and (step % export_every == 0):
            enc.eval(); dec.eval()
            with torch.no_grad():
                logits_cmd_, pred_arg_, _ = dec(mu, stage_sdf, seq_mask)  # 用 μ
                export_previews(out_dir, "train", step, epoch_idx,
                                logits_cmd_, pred_arg_, seq_mask, items, max_n=preview_n)
                pbar.write(f"[export] saved previews at step {step} (N={min(preview_n, len(items))})")
            enc.train(); dec.train()

    pbar.close()  # 关闭进度条
    return step


@torch.no_grad()
def evaluate(ds, renderer, enc, dec, loss_cfg, batch_size, device, 
             export_vis=False, out_dir=None, epoch_idx=0, preview_n=4):
    """评估函数，可选择导出可视化
    
    Args:
        export_vis: 是否导出可视化结果
        out_dir: 输出目录
        epoch_idx: 当前 epoch 编号
        preview_n: 导出的样本数量
    """
    enc.eval(); dec.eval()
    n = len(ds)
    indices = list(range(0, min(n, batch_size)))
    items = [ds[i] for i in indices]
    batch = to_device_batch(items, device)
    stage_sdf = render_stage_sdf_batch(renderer, items, device)

    seq_cmd, seq_arg = batch["seq_cmd"], batch["seq_arg"]
    seq_mask = batch["seq_mask"].bool()
    contour_ids, seq_topo = batch["contour_ids"], batch["seq_topo"]

    mu, logvar, z, _ = enc(seq_cmd, seq_arg, seq_mask, contour_ids, seq_topo, stage_sdf, sample=False)
    logits_cmd, pred_arg, _ = dec(z, stage_sdf, seq_mask)
    total, stats = compute_vae_losses(
        logits_cmd, pred_arg, mu, logvar, seq_cmd, seq_arg, seq_mask, cfg=loss_cfg
    )
    
    # 导出验证集可视化
    if export_vis and out_dir:
        export_previews(out_dir / "eval_previews", "eval", 0, epoch_idx,
                       logits_cmd, pred_arg, seq_mask, items, max_n=preview_n)
        print(f"[eval export] saved {min(preview_n, len(items))} evaluation previews to {out_dir / 'eval_previews'}")
    
    return {k: float(v) for k, v in stats.items()}


# --------------------------- main ------------------------
def main():
    import sys
    from config import get_config, list_configs
    
    # 检查是否请求列出配置
    if '--list-configs' in sys.argv:
        list_configs()
        return
    
    # 检查是否指定了配置预设
    config_name = 'high_quality'  # 默认使用高质量配置（针对大数据集）
    
    if '--config' in sys.argv:
        idx = sys.argv.index('--config')
        if idx + 1 < len(sys.argv):
            config_name = sys.argv[idx + 1]
            # 移除 --config 及其值，剩余的用于覆盖
            cmd_args = sys.argv[1:idx] + sys.argv[idx+2:]
        else:
            print("错误: --config 需要指定配置名称")
            print("可用配置: default, balanced, high_quality, memory_limited")
            print("使用 --list-configs 查看详细信息")
            return
    else:
        # 没有指定 --config，使用所有命令行参数覆盖
        cmd_args = sys.argv[1:]
    
    # 加载配置
    try:
        if cmd_args:
            args = get_config(config_name, cmd_args)
        else:
            args = get_config(config_name)
    except ValueError as e:
        print(f"配置错误: {e}")
        print("\n可用配置:")
        list_configs()
        return
    
    # 打印完整的训练配置
    print("\n" + "="*80)
    print(f"📋 训练配置: {config_name.upper()}")
    print("="*80)
    
    # 数据配置
    print("\n📂 数据配置:")
    print(f"  数据集路径:     {args.npz}")
    print(f"  使用 RSM:       {'是 ✓' if args.use_rsm else '否'}")
    print(f"  验证集比例:     {args.val_split * 100:.1f}%")
    if args.img_size:
        print(f"  图像大小:       {args.img_size}×{args.img_size}")
    
    # 模型架构
    print("\n🏗️  模型架构:")
    print(f"  嵌入维度 (embed):           {args.embed}")
    print(f"  潜在空间维度 (zdim):        {args.zdim}")
    print(f"  注意力头数 (heads):         {args.heads}")
    print(f"  Patch 大小:                 {args.patch}")
    print(f"  矢量编码器层数 (vec):       {args.vec_layers}")
    print(f"  跨注意力层数 (enc_xlayers): {args.enc_xlayers}")
    print(f"  解码器层数 (dec_layers):    {args.dec_layers}")
    
    # 训练超参数
    print("\n⚙️  训练超参数:")
    print(f"  训练轮次 (epochs):  {args.epochs}")
    print(f"  批次大小 (batch):   {args.batch}")
    print(f"  学习率 (lr):        {args.lr:.6f}")
    print(f"  权重衰减 (wd):      {args.wd}")
    print(f"  梯度裁剪:           {args.grad_clip}")
    print(f"  随机种子:           {args.seed}")
    
    # 损失权重配置
    print("\n📊 损失权重配置:")
    print(f"  L1 权重:            {args.l1_weight}")
    print(f"  CE 权重:            {args.ce_weight}")
    print(f"  Beta 预热步数:      {args.beta_warmup:,}")
    print(f"  Free Bits:          {args.free_bits}")
    
    # 日志和保存
    print("\n💾 日志和保存:")
    print(f"  输出目录:           {args.out}")
    print(f"  日志频率:           每 {args.log_every} 步")
    print(f"  评估频率:           每 {args.eval_every} 个 epoch")
    print(f"  导出预览频率:       每 {args.export_every:,} 步")
    print(f"  预览样本数:         {args.preview_n}")
    if args.resume:
        print(f"  恢复检查点:         {args.resume}")
    
    print("="*80 + "\n")

    # ---- 基础初始化 ----
    set_seed(args.seed)
    device = torch.device(args.device)
    out_dir = Path(args.out); (out_dir / "ckpt").mkdir(parents=True, exist_ok=True)
    
    # 保存配置到文件
    import json
    config_dict = vars(args)
    config_dict['config_name'] = config_name
    with open(out_dir / "training_config.json", "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2, ensure_ascii=False)
    print(f"✅ 配置已保存到: {out_dir / 'training_config.json'}\n")
    
    # 打印设备信息
    print("\n" + "="*80)
    print("🖥️  设备信息")
    print("="*80)
    if device.type == "cuda":
        print(f"✅ 使用 GPU 训练")
        print(f"   设备: {torch.cuda.get_device_name(0)}")
        print(f"   显存总量: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        if torch.cuda.is_available():
            print(f"   CUDA 版本: {torch.version.cuda}")
            print(f"   可用 GPU 数量: {torch.cuda.device_count()}")
    else:
        print(f"⚠️  使用 CPU 训练（速度会很慢）")
        print(f"   建议: 如果有 GPU，请使用 --device cuda")
    print(f"   混合精度 (AMP): {'启用 ✓' if args.amp else '禁用'}")
    print(f"   随机种子: {args.seed}")
    print("="*80 + "\n")

    # 日志文件
    csv_logger = CsvLogger(out_dir / "train_log.csv",
                           fieldnames=["time","epoch","step","loss","ce","l1","kl","beta","lr","ips"])
    jsonl_path = out_dir / "events.jsonl"

    # ---- 数据：NPZ + RSM（可选）----
    base = NPZDataset(args.npz, mmap=True, use_seq_topo=True, strict_shapes=True)
    if args.use_rsm:
        ds = RSMBatcher(base, RSMConfig(
            rsm_stages=(0.25, 0.5, 0.75, 1.0),
            rsm_probs=(0.15, 0.25, 0.25, 0.35),
            enforce_contour_ids=True,
            allow_token_prefix_fallback=False,
        ))
        print("[info] dataset: RSM mode")
    else:
        ds = base
        print("[info] dataset: FULL (no RSM)")

    renderer = StageRenderer(StageRendererConfig(
        img_size=args.img_size or base.H,
        sdf_clip_px=8.0,
        out_dtype=torch.float32,
    ))

    # ---- 训练/验证集分割 ----
    n_total = len(ds)
    if args.val_split > 0 and args.val_split < 1.0:
        n_val = max(1, int(n_total * args.val_split))  # 至少1个验证样本
        n_train = n_total - n_val
        
        # 随机打乱索引（使用固定种子保证可复现）
        rng = random.Random(args.seed)
        indices = list(range(n_total))
        rng.shuffle(indices)
        
        train_indices = sorted(indices[:n_train])  # 排序以保持原始顺序（可选）
        val_indices = sorted(indices[n_train:])
        
        # 创建数据集包装器
        from torch.utils.data import Subset
        train_ds = Subset(ds, train_indices)
        val_ds = Subset(ds, val_indices)
        
        print(f"[数据分割] 总样本: {n_total}, 训练集: {n_train} ({100*(1-args.val_split):.1f}%), 验证集: {n_val} ({100*args.val_split:.1f}%)")
        print(f"[数据分割] 验证集索引: {val_indices[:min(10, len(val_indices))]}" + 
              (f"... (共{len(val_indices)}个)" if len(val_indices) > 10 else ""))
    else:
        # 不分割，所有数据用于训练
        train_ds = ds
        val_ds = ds
        print(f"[数据分割] 不使用验证集，所有 {n_total} 个样本用于训练")

    # ---- 模型 ----
    enc = VpVaeEncoder(VpVaeEncoderConfig(
        embed_dim=args.embed, num_heads=args.heads, cross_layers=args.enc_xlayers,
        patch_size=args.patch, z_dim=args.zdim, vec_layers=args.vec_layers,
        use_prefix_repr=True, dropout=0.0
    )).to(device)

    dec = Decoder(DecoderConfig(
        vocab_size=9, max_len=base.L, embed_dim=args.embed, z_dim=args.zdim,
        n_heads=args.heads, n_layers=args.dec_layers, patch_size=args.patch, use_pixel_cross_attn=True
    )).to(device)

    # ---- 优化 / AMP / KL 预热 ----
    opt = torch.optim.AdamW(list(enc.parameters()) + list(dec.parameters()), lr=args.lr, weight_decay=args.wd)
    scaler = GradScaler("cuda") if (args.amp and device.type == "cuda") else None
    beta_sched = BetaWarmup(warmup_steps=args.beta_warmup, target=1.0)

    loss_cfg = LossConfig(
        beta=0.0, free_bits=args.free_bits,
        use_class_weight=False,  # 我们用固定权重替换 CE
        num_classes=9,
        l1_weight=args.l1_weight,
        ce_weight=args.ce_weight
    )

    # ---- 固定类别权重：若已存在则加载，否则在训练集上统计一次并保存 ----
    cw_path = out_dir / "class_weight.pt"
    if cw_path.is_file():
        class_weight = torch.load(cw_path, map_location="cpu").to(device)
        print("[weight] loaded fixed class weights from:", cw_path)
    else:
        print("[weight] computing FIXED class weights (one-time, train set only, with current RSM setting)...")
        indices_all = list(range(len(train_ds)))
        class_weight = compute_class_weight_epoch([train_ds[i] for i in indices_all], num_classes=9, device=device)
        torch.save(class_weight.cpu(), cw_path)
        pretty = {CMD_NAMES[i] if i < len(CMD_NAMES) else i: float(class_weight[i]) for i in range(len(class_weight))}
        print("[weight] class_weight =", pretty)

    # ---- 断点恢复（可选）----
    global_step, start_epoch = 0, 0
    if args.resume and Path(args.resume).is_file():
        global_step, start_epoch = load_ckpt(Path(args.resume), enc, dec, opt=opt, scaler=scaler, map_location=device)
        print(f"[resume] from {args.resume} @ step={global_step} epoch={start_epoch}")

    # ---- 初始可视化：导出未训练模型的输出（仅在从头开始训练时）----
    if start_epoch == 0:
        print("\n[初始可视化] 导出未训练模型在验证集上的预测结果...")
        initial_eval = evaluate(val_ds, renderer, enc, dec, loss_cfg, 
                               batch_size=args.batch, device=device,
                               export_vis=True, out_dir=out_dir, 
                               epoch_idx=0, preview_n=args.preview_n)
        print(f"[初始状态-验证集] loss={initial_eval['loss_total']:.4f}  "
              f"CE={initial_eval['loss_ce']:.4f}  "
              f"L1={initial_eval['loss_l1']:.4f}  "
              f"KL={initial_eval['loss_kl']:.4f}")
        with open(out_dir / "initial_eval.json", "w", encoding="utf-8") as f:
            json.dump(initial_eval, f, indent=2)

    # ---- 训练循环 ----
    for epoch in trange(start_epoch, args.epochs, desc="Training", ncols=100):
        print(f"\n==> epoch {epoch+1}/{args.epochs}")
        global_step = train_one_epoch(
            train_ds, renderer, enc, dec, opt, scaler, loss_cfg, beta_sched,
            batch_size=args.batch, device=device, log_every=args.log_every,
            grad_clip=args.grad_clip, export_every=args.export_every, out_dir=out_dir,
            step0=global_step, class_weight=class_weight, epoch_idx=epoch+1,
            csv_logger=csv_logger, jsonl_path=jsonl_path, preview_n=args.preview_n
        )
        # 在验证集上评估 + 按频率导出可视化
        should_export = ((epoch + 1) % args.eval_every == 0) or (epoch + 1 == args.epochs)
        eval_stats = evaluate(val_ds, renderer, enc, dec, loss_cfg, 
                             batch_size=args.batch, device=device,
                             export_vis=should_export, out_dir=out_dir, 
                             epoch_idx=epoch+1, preview_n=args.preview_n)
        print(f"[验证集] loss={eval_stats['loss_total']:.4f}  CE={eval_stats['loss_ce']:.4f}  "
              f"L1={eval_stats['loss_l1']:.4f}  KL={eval_stats['loss_kl']:.4f}" + 
              (f"  [已保存PNG]" if should_export else ""))
        with open(out_dir / "last_eval.json", "w", encoding="utf-8") as f:
            json.dump(eval_stats, f, indent=2)

        # 保存 ckpt
        ck_path = out_dir / "ckpt" / f"epoch{epoch+1}.pt"
        save_ckpt(ck_path, enc, dec, opt, scaler, (global_step, epoch+1))
        print(f"[ckpt] saved {ck_path}")

    # ---- 训练完成：最终可视化总结 ----
    print("\n" + "="*80)
    print("训练完成！生成验证集最终可视化总结...")
    print("="*80)
    
    # 导出更多验证集样本的最终结果
    final_preview_n = min(16, len(val_ds))  # 最多导出16个样本
    final_eval = evaluate(val_ds, renderer, enc, dec, loss_cfg, 
                         batch_size=final_preview_n, device=device,
                         export_vis=True, out_dir=out_dir, 
                         epoch_idx=999, preview_n=final_preview_n)
    
    print(f"\n[最终评估-验证集] loss={final_eval['loss_total']:.4f}  "
          f"CE={final_eval['loss_ce']:.4f}  "
          f"L1={final_eval['loss_l1']:.4f}  "
          f"KL={final_eval['loss_kl']:.4f}")
    
    with open(out_dir / "final_eval.json", "w", encoding="utf-8") as f:
        json.dump(final_eval, f, indent=2)
    
    print(f"\n✅ 所有可视化结果已保存到：")
    print(f"   📊 训练集样本数: {len(train_ds)}")
    print(f"   📊 验证集样本数: {len(val_ds)}")
    print(f"   - 训练过程预览（训练集）: {out_dir}")
    print(f"   - 评估预览（验证集）: {out_dir / 'eval_previews'}")
    print(f"   - 检查点: {out_dir / 'ckpt'}")
    print(f"   - 日志: {out_dir / 'train_log.csv'} 和 {out_dir / 'events.jsonl'}")

if __name__ == "__main__":
    main()
