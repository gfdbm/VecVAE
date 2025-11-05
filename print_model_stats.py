#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""打印 VP-VAE 模型的参数统计"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import torch
    from src.model.encoder import VpVaeEncoder, VpVaeEncoderConfig
    from src.model.decoder import Decoder, DecoderConfig
    
    def count_parameters(model):
        """统计模型参数量"""
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return total, trainable
    
    def format_number(num):
        """格式化数字"""
        if num >= 1e6:
            return f"{num/1e6:.2f}M"
        elif num >= 1e3:
            return f"{num/1e3:.2f}K"
        else:
            return str(num)
    
    print("\n" + "="*80)
    print("🔍 VP-VAE 模型参数统计")
    print("="*80)
    
    # 默认配置
    print("\n📐 配置参数:")
    print("-"*80)
    embed_dim = 256
    num_heads = 8
    z_dim = 128
    cross_layers = 1
    dec_layers = 4
    patch_size = 16
    
    print(f"  嵌入维度 (embed_dim):     {embed_dim}")
    print(f"  注意力头数 (num_heads):   {num_heads}")
    print(f"  潜在维度 (z_dim):         {z_dim}")
    print(f"  跨注意力层数 (enc-xlayers): {cross_layers}")
    print(f"  解码器层数 (dec-layers):   {dec_layers}")
    print(f"  Patch 大小 (patch_size):  {patch_size}")
    
    # 创建编码器
    print("\n" + "="*80)
    print("🧠 编码器 (Encoder)")
    print("="*80)
    
    enc_config = VpVaeEncoderConfig(
        embed_dim=embed_dim,
        num_heads=num_heads,
        cross_layers=cross_layers,
        patch_size=patch_size,
        z_dim=z_dim,
        use_prefix_repr=True,
        dropout=0.0
    )
    encoder = VpVaeEncoder(enc_config)
    enc_total, enc_trainable = count_parameters(encoder)
    
    print(f"\n1. VectorPrefixEncoder (矢量编码)")
    print(f"   - 3 层 Transformer (固定)")
    print(f"   - {num_heads} 个注意力头")
    print(f"   - FFN 中间层: {embed_dim} → {embed_dim*4} → {embed_dim}")
    vec_params, _ = count_parameters(encoder.vec)
    print(f"   参数量: {format_number(vec_params)}")
    
    print(f"\n2. PixelEncoder (像素编码)")
    print(f"   - CNN + Patch Embedding")
    print(f"   - Patch 大小: {patch_size}×{patch_size}")
    pix_params, _ = count_parameters(encoder.pix)
    print(f"   参数量: {format_number(pix_params)}")
    
    print(f"\n3. CrossAttentionAdapter (跨模态融合)")
    print(f"   - {cross_layers} 层跨注意力")
    print(f"   - {num_heads} 个注意力头")
    xattn_params, _ = count_parameters(encoder.xattn)
    print(f"   参数量: {format_number(xattn_params)}")
    
    print(f"\n4. PosteriorHead (潜在空间投影)")
    print(f"   - 输入维度: {embed_dim}")
    print(f"   - 潜在维度: {z_dim}")
    head_params, _ = count_parameters(encoder.head)
    print(f"   参数量: {format_number(head_params)}")
    
    print(f"\n{'编码器总参数量:':<30} {format_number(enc_total):<15} ({enc_total:,})")
    
    # 创建解码器
    print("\n" + "="*80)
    print("🎨 解码器 (Decoder)")
    print("="*80)
    
    dec_config = DecoderConfig(
        vocab_size=9,
        max_len=256,
        embed_dim=embed_dim,
        z_dim=z_dim,
        n_heads=num_heads,
        n_layers=dec_layers,
        patch_size=patch_size,
        use_pixel_cross_attn=True
    )
    decoder = Decoder(dec_config)
    dec_total, dec_trainable = count_parameters(decoder)
    
    print(f"\n1. 位置 & 有效位嵌入")
    print(f"   - 绝对位置: 256 个位置")
    print(f"   - 有效位: 2 类 (前缀/填充)")
    print(f"   参数量: ~0.1M")
    
    if hasattr(decoder, 'pix'):
        print(f"\n2. PixelEncoder (像素编码)")
        print(f"   - 与编码器共享结构")
        dec_pix_params, _ = count_parameters(decoder.pix)
        print(f"   参数量: {format_number(dec_pix_params)}")
    
    print(f"\n3. Transformer 解码块 (×{dec_layers} 层)")
    print(f"   每层结构:")
    print(f"   ├─ Self-Attention ({num_heads} 头)")
    print(f"   ├─ Cross-Attention (与像素)")
    print(f"   └─ FFN ({embed_dim} → {embed_dim*4} → {embed_dim})")
    
    # 估算每层参数
    single_layer_params = dec_total / dec_layers * 0.7  # 粗略估算
    print(f"   每层约: {format_number(single_layer_params)}")
    
    print(f"\n4. 输出头")
    print(f"   ├─ 命令分类头: → 9 类")
    print(f"   └─ 坐标回归头: → 4 维")
    out_params, _ = count_parameters(decoder.head_cmd)
    out_params += sum(p.numel() for p in decoder.head_arg.parameters())
    print(f"   参数量: {format_number(out_params)}")
    
    print(f"\n{'解码器总参数量:':<30} {format_number(dec_total):<15} ({dec_total:,})")
    
    # 总计
    print("\n" + "="*80)
    print("📊 总参数统计")
    print("="*80)
    total_params = enc_total + dec_total
    print(f"\n{'编码器:':<20} {format_number(enc_total):>15} ({enc_total:>12,})")
    print(f"{'解码器:':<20} {format_number(dec_total):>15} ({dec_total:>12,})")
    print(f"{'-'*20} {'-'*15} {'-'*15}")
    print(f"{'VP-VAE 总计:':<20} {format_number(total_params):>15} ({total_params:>12,})")
    
    # 估算显存 (FP32)
    memory_mb = total_params * 4 / (1024**2)  # 4 bytes per float32
    print(f"\n💾 模型大小估算 (FP32):")
    print(f"   参数: ~{memory_mb:.1f} MB")
    print(f"   训练时 (含梯度+优化器): ~{memory_mb*3:.1f} MB")
    print(f"   实际显存需求 (batch=8): ~{memory_mb*3 + 2000:.1f} MB (~{(memory_mb*3 + 2000)/1024:.1f} GB)")
    
    # 对比不同配置
    print("\n" + "="*80)
    print("⚙️  不同配置的参数量对比")
    print("="*80)
    
    configs = [
        ("轻量", 128, 64, 1, 2),
        ("标准 (当前)", 256, 128, 1, 4),
        ("增强", 256, 256, 2, 6),
        ("重型", 512, 256, 2, 8),
    ]
    
    print(f"\n{'配置':<15} {'embed':<8} {'zdim':<8} {'enc-x':<8} {'dec-L':<8} {'参数量':<15} {'显存估算'}")
    print("-"*80)
    
    for name, emb, zd, encx, decl in configs:
        # 粗略估算
        enc_est = 2.5e6 * (emb/256)**2 + 0.5e6 * encx
        dec_est = 1e6 * decl * (emb/256)**2
        total_est = enc_est + dec_est
        mem_est = total_est * 4 / (1024**2) * 3 + 2000
        marker = " ← 当前" if name.endswith("当前)") else ""
        print(f"{name:<15} {emb:<8} {zd:<8} {encx:<8} {decl:<8} {format_number(total_est):<15} ~{mem_est/1024:.1f}GB{marker}")
    
    print("\n" + "="*80)
    print("💡 调参建议")
    print("="*80)
    print("""
1. 如果显存不足:
   python train.py --embed 128 --dec-layers 2 --batch 4

2. 如果 L1 损失不下降:
   python train.py --dec-layers 6 --zdim 256

3. 如果想要更好效果 (显存充足):
   python train.py --embed 512 --dec-layers 6 --batch 4

4. 快速原型 (加快训练):
   python train.py --embed 128 --dec-layers 2 --enc-xlayers 1
    """)
    print("="*80 + "\n")
    
except ImportError as e:
    print(f"\n❌ 导入失败: {e}")
    print("   请确保已安装 PyTorch 并且在正确的项目目录下运行")
    print()
except Exception as e:
    print(f"\n❌ 错误: {e}")
    import traceback
    traceback.print_exc()

