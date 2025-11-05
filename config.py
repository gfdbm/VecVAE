#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VP-VAE 训练配置

使用 argparse 风格管理所有训练参数
针对超大数据集（上万样本）的配置
"""

import argparse

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def get_args_parser():
    """获取参数解析器"""
    parser = argparse.ArgumentParser(description="VP-VAE Training Configuration")
    
    # ========== 数据相关 ==========
    parser.add_argument("--npz", type=str, 
                       default='/data/kssczt/ztb/FontVector/VAE/data/msyh.npz',
                       help="path to dataset .npz file")
    parser.add_argument("--use-rsm", type=bool, default=True,
                       help="use RSM stage prefix")
    parser.add_argument("--img-size", type=int, default=None,
                       help="image size (default: None, use dataset default)")
    parser.add_argument("--val-split", type=float, default=0.1,
                       help="validation split ratio (0.0-1.0)")
    
    # ========== 模型架构 ==========
    parser.add_argument("--embed", type=int, default=384,
                       help="embedding dimension")
    parser.add_argument("--zdim", type=int, default=384,
                       help="latent dimension")
    parser.add_argument("--heads", type=int, default=8,
                       help="number of attention heads")
    parser.add_argument("--patch", type=int, default=16,
                       help="patch size for image encoding")
    parser.add_argument("--vec-layers", type=int, default=3,
                       help="vector encoder layers")
    parser.add_argument("--enc-xlayers", type=int, default=2,
                       help="cross attention layers in encoder")
    parser.add_argument("--dec-layers", type=int, default=8,
                       help="decoder layers")
    
    # ========== 训练超参数 ==========
    parser.add_argument("--epochs", type=int, default=800,
                       help="number of training epochs")
    parser.add_argument("--batch", type=int, default=12,
                       help="batch size")
    parser.add_argument("--lr", type=float, default=3e-4,
                       help="learning rate")
    parser.add_argument("--wd", type=float, default=0.0,
                       help="weight decay")
    parser.add_argument("--grad-clip", type=float, default=1.0,
                       help="gradient clipping value")
    parser.add_argument("--seed", type=int, default=42,
                       help="random seed")
    
    # ========== 损失权重 ==========
    parser.add_argument("--l1-weight", type=float, default=30.0,
                       help="L1 loss weight (increase if L1 not decreasing)")
    parser.add_argument("--ce-weight", type=float, default=2.5,
                       help="CE loss weight")
    parser.add_argument("--beta-warmup", type=int, default=40000,
                       help="KL beta warmup steps")
    parser.add_argument("--free-bits", type=float, default=4.0,
                       help="free bits for KL loss")
    
    # ========== 设备和优化 ==========
    default_device = "cuda" if (TORCH_AVAILABLE and torch.cuda.is_available()) else "cpu"
    parser.add_argument("--device", type=str, 
                       default=default_device,
                       help="device to use (cuda/cpu)")
    parser.add_argument("--amp", action="store_true", default=True,
                       help="use automatic mixed precision")
    
    # ========== 日志和保存 ==========
    parser.add_argument("--out", type=str, default="runs/large_dataset",
                       help="output directory")
    parser.add_argument("--log-every", type=int, default=50,
                       help="log every N steps")
    parser.add_argument("--eval-every", type=int, default=10,
                       help="evaluate every N epochs")
    parser.add_argument("--export-every", type=int, default=15000,
                       help="export training preview every N steps")
    parser.add_argument("--preview-n", type=int, default=4,
                       help="number of preview samples")
    parser.add_argument("--resume", type=str, default="",
                       help="resume from checkpoint")
    
    return parser


# ==============================================================================
# 针对超大数据集的推荐配置（上万样本）
# ==============================================================================

def get_config_balanced():
    """
    配置 1: 平衡配置
    
    适用场景:
      - 数据量: 5000-15000 样本
      - 显存: 8-12 GB (RTX 3080Ti, 3090)
      - 特点: 平衡速度和质量
    
    关键参数:
      - embed=384, zdim=384, dec_layers=8
      - batch=12, lr=3e-4
      - l1_weight=30.0, beta_warmup=40000, free_bits=4.0
    """
    parser = get_args_parser()
    parser.set_defaults(
        # 模型
        embed=384,
        zdim=384,
        heads=8,
        vec_layers=3,
        enc_xlayers=2,
        dec_layers=8,
        # 训练
        epochs=800,
        batch=12,
        lr=3e-4,
        # 损失
        l1_weight=30.0,
        ce_weight=2.5,
        beta_warmup=40000,
        free_bits=4.0,
        # 输出
        out="runs/balanced",
        export_every=15000,
    )
    return parser


def get_config_high_quality():
    """
    配置 2: 高质量配置
    
    适用场景:
      - 数据量: 10000-20000 样本
      - 显存: 16-24 GB (RTX 4090, A100)
      - 特点: 追求最佳重建质量，参数量大
    
    关键参数:
      - embed=512, zdim=512, dec_layers=8
      - batch=16, lr=4e-4
      - l1_weight=40.0, beta_warmup=50000, free_bits=5.0
    """
    parser = get_args_parser()
    parser.set_defaults(
        # 模型 - 最大容量
        embed=512,
        zdim=512,
        heads=16,
        vec_layers=3,
        enc_xlayers=3,
        dec_layers=8,
        # 训练 - 大数据可以快一点
        epochs=600,
        batch=16,
        lr=4e-4,
        # 损失 - 最强配置
        l1_weight=40.0,
        ce_weight=3.0,
        beta_warmup=50000,
        free_bits=5.0,
        # 输出
        out="runs/high_quality",
        export_every=20000,
    )
    return parser


def get_config_memory_limited():
    """
    配置 3: 显存受限配置
    
    适用场景:
      - 数据量: 5000-20000 样本
      - 显存: 8-12 GB (显存不够但数据量大)
      - 特点: 降低 batch size，使用更长训练时间
    
    关键参数:
      - embed=384, zdim=384, dec_layers=6
      - batch=6 (小 batch), epochs=1000 (更多轮次)
      - l1_weight=30.0, beta_warmup=40000, free_bits=4.0
    """
    parser = get_args_parser()
    parser.set_defaults(
        # 模型 - 适中
        embed=384,
        zdim=384,
        heads=8,
        vec_layers=3,
        enc_xlayers=2,
        dec_layers=6,  # 稍浅一点，省显存
        # 训练 - 小 batch，更多轮次
        epochs=1000,
        batch=6,       # 小 batch 省显存
        lr=2e-4,       # 小学习率配合小 batch
        # 损失
        l1_weight=30.0,
        ce_weight=2.5,
        beta_warmup=40000,
        free_bits=4.0,
        # 输出
        out="runs/memory_limited",
        export_every=15000,
    )
    return parser


# ==============================================================================
# 配置注册
# ==============================================================================

CONFIGS = {
    "default": get_args_parser,
    "balanced": get_config_balanced,
    "high_quality": get_config_high_quality,
    "memory_limited": get_config_memory_limited,
}

DESCRIPTIONS = {
    "default": "默认配置 (embed=384, zdim=384, dec=8)",
    "balanced": "配置1: 平衡配置 - 8-12GB显存 (RTX 3090)",
    "high_quality": "配置2: 高质量配置 - 16-24GB显存 (RTX 4090) ⭐推荐",
    "memory_limited": "配置3: 显存受限 - 8-12GB显存但数据量大",
}


def get_config(config_name="default", cmd_args=None):
    """
    获取配置
    
    Args:
        config_name: 配置名称 (default/balanced/high_quality/memory_limited)
        cmd_args: 命令行参数（可选），用于覆盖预设
    
    Returns:
        parsed args 对象
    
    Examples:
        # 使用预设
        args = get_config('high_quality')
        
        # 使用预设 + 覆盖
        args = get_config('high_quality', ['--batch', '12', '--epochs', '800'])
    """
    if config_name not in CONFIGS:
        available = ", ".join(CONFIGS.keys())
        raise ValueError(f"未知配置: {config_name}. 可用: {available}")
    
    parser = CONFIGS[config_name]()
    
    if cmd_args is None:
        # 不解析命令行，直接返回默认值
        args, _ = parser.parse_known_args([])
    else:
        # 解析命令行参数（会覆盖预设）
        args = parser.parse_args(cmd_args)
    
    return args


def list_configs():
    """列出所有可用配置"""
    print("\n" + "="*80)
    print("📋 可用配置（针对超大数据集 - 上万样本）")
    print("="*80 + "\n")
    
    for name in CONFIGS.keys():
        desc = DESCRIPTIONS.get(name, "")
        args = get_config(name)
        
        print(f"【{name.upper()}】")
        print(f"  {desc}")
        print(f"  模型: embed={args.embed}, zdim={args.zdim}, dec_layers={args.dec_layers}")
        print(f"  训练: epochs={args.epochs}, batch={args.batch}, lr={args.lr}")
        print(f"  损失: l1={args.l1_weight}, ce={args.ce_weight}, beta_warmup={args.beta_warmup}, free_bits={args.free_bits}")
        print(f"  输出: {args.out}")
        print()
    
    print("="*80)
    print("💡 推荐:")
    print("  - 如果你有 RTX 4090 (24GB): 用 'high_quality'")
    print("  - 如果你有 RTX 3090 (24GB): 用 'balanced' 或 'high_quality'")
    print("  - 如果显存有限 (< 12GB):   用 'memory_limited'")
    print("="*80)
    print("\n使用方法:")
    print("  from config import get_config")
    print("  args = get_config('high_quality')  # 使用预设")
    print("  args = get_config('high_quality', ['--batch', '12'])  # 预设 + 覆盖")
    print("="*80 + "\n")


def show_config(args):
    """显示配置详情"""
    print("\n" + "="*80)
    print("📋 当前配置")
    print("="*80)
    
    print("\n数据配置:")
    print(f"  npz: {args.npz}")
    print(f"  use_rsm: {args.use_rsm}")
    print(f"  val_split: {args.val_split}")
    
    print("\n模型架构:")
    print(f"  embed: {args.embed}")
    print(f"  zdim: {args.zdim}")
    print(f"  heads: {args.heads}")
    print(f"  vec_layers: {args.vec_layers}")
    print(f"  enc_xlayers: {args.enc_xlayers}")
    print(f"  dec_layers: {args.dec_layers}")
    print(f"  patch: {args.patch}")
    
    print("\n训练参数:")
    print(f"  epochs: {args.epochs}")
    print(f"  batch: {args.batch}")
    print(f"  lr: {args.lr}")
    print(f"  wd: {args.wd}")
    print(f"  grad_clip: {args.grad_clip}")
    print(f"  seed: {args.seed}")
    
    print("\n损失权重:")
    print(f"  l1_weight: {args.l1_weight}")
    print(f"  ce_weight: {args.ce_weight}")
    print(f"  beta_warmup: {args.beta_warmup}")
    print(f"  free_bits: {args.free_bits}")
    
    print("\n设备和优化:")
    print(f"  device: {args.device}")
    print(f"  amp: {args.amp}")
    
    print("\n日志和保存:")
    print(f"  out: {args.out}")
    print(f"  log_every: {args.log_every}")
    print(f"  eval_every: {args.eval_every}")
    print(f"  export_every: {args.export_every}")
    print(f"  preview_n: {args.preview_n}")
    if args.resume:
        print(f"  resume: {args.resume}")
    
    print("="*80 + "\n")


# ==============================================================================
# 主函数
# ==============================================================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "list":
            list_configs()
        
        elif command == "show":
            if len(sys.argv) > 2:
                config_name = sys.argv[2]
                args = get_config(config_name)
                print(f"\n配置: {config_name.upper()}")
                print(f"描述: {DESCRIPTIONS.get(config_name, '')}")
                show_config(args)
            else:
                args = get_config("default")
                show_config(args)
        
        else:
            print("未知命令. 可用命令:")
            print("  python config.py list                  # 列出所有配置")
            print("  python config.py show [config_name]    # 显示配置详情")
    
    else:
        print("VP-VAE 配置管理（超大数据集专用）")
        print("\n用法:")
        print("  python config.py list                # 列出所有配置")
        print("  python config.py show high_quality   # 显示配置详情")
        print("\n在代码中使用:")
        print("  from config import get_config")
        print("  args = get_config('high_quality')    # 使用预设")
        print("  args = get_config('high_quality', ['--batch', '12'])  # 覆盖参数")
