#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
可视化结果查看器
================
用于浏览训练过程中生成的矢量可视化结果

使用方法:
    python view_results.py --out runs/v1
    python view_results.py --out runs/v1 --epoch 10
    python view_results.py --out runs/v1 --list  # 列出所有可用的epoch
"""

import argparse
from pathlib import Path
import webbrowser
import tempfile

def list_available_epochs(out_dir: Path):
    """列出所有可用的 epoch 可视化"""
    eval_dir = out_dir / "eval_previews"
    if not eval_dir.exists():
        print(f"未找到评估预览目录: {eval_dir}")
        return []
    
    png_files = sorted(eval_dir.glob("preview_eval_ep*.png"))
    epochs = set()
    for f in png_files:
        # 从文件名提取 epoch 号，如 preview_eval_ep10_step0_b0.png
        parts = f.stem.split('_')
        for p in parts:
            if p.startswith('ep'):
                try:
                    epochs.add(int(p[2:]))
                except:
                    pass
    return sorted(epochs)

def create_gallery_html(image_files, title="VAE 可视化结果"):
    """创建 HTML 画廊来展示多个图像"""
    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{title}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            background: #f5f5f5;
            margin: 0;
            padding: 20px;
        }}
        h1 {{
            text-align: center;
            color: #333;
        }}
        .info {{
            text-align: center;
            color: #666;
            margin-bottom: 30px;
        }}
        .gallery {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 20px;
            max-width: 1400px;
            margin: 0 auto;
        }}
        .item {{
            background: white;
            border: 1px solid #ddd;
            border-radius: 8px;
            padding: 15px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            transition: transform 0.2s;
        }}
        .item:hover {{
            transform: translateY(-5px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.15);
        }}
        .item h3 {{
            margin: 0 0 10px 0;
            font-size: 14px;
            color: #555;
        }}
        .img-container {{
            width: 100%;
            height: 280px;
            display: flex;
            align-items: center;
            justify-content: center;
            background: #fafafa;
            border-radius: 4px;
            border: 1px solid #e0e0e0;
        }}
        .img-container img {{
            max-width: 100%;
            max-height: 100%;
            border-radius: 4px;
        }}
        .file-path {{
            margin-top: 10px;
            font-size: 11px;
            color: #999;
            word-break: break-all;
        }}
    </style>
</head>
<body>
    <h1>{title}</h1>
    <div class="info">
        共 {len(image_files)} 个样本
    </div>
    <div class="gallery">
"""
    
    for i, img_path in enumerate(image_files):
        name = img_path.stem
        html += f"""
        <div class="item">
            <h3>样本 {i+1}</h3>
            <div class="img-container">
                <img src="file://{img_path.absolute()}" alt="{name}">
            </div>
            <div class="file-path">{img_path.name}</div>
        </div>
"""
    
    html += """
    </div>
</body>
</html>
"""
    return html

def main():
    parser = argparse.ArgumentParser(description="查看训练可视化结果")
    parser.add_argument("--out", type=str, default="runs/v1", 
                       help="训练输出目录")
    parser.add_argument("--epoch", type=int, default=None,
                       help="查看特定 epoch 的结果（默认查看最新）")
    parser.add_argument("--list", action="store_true",
                       help="列出所有可用的 epoch")
    parser.add_argument("--initial", action="store_true",
                       help="查看初始状态（epoch 0）")
    parser.add_argument("--final", action="store_true",
                       help="查看最终结果（epoch 999）")
    parser.add_argument("--max-samples", type=int, default=16,
                       help="最多显示的样本数")
    args = parser.parse_args()
    
    out_dir = Path(args.out)
    if not out_dir.exists():
        print(f"❌ 输出目录不存在: {out_dir}")
        return
    
    # 列出可用 epoch
    available_epochs = list_available_epochs(out_dir)
    
    if args.list:
        print(f"可用的 epoch: {available_epochs}")
        return
    
    # 确定要查看的 epoch
    if args.initial:
        target_epoch = 0
    elif args.final:
        target_epoch = 999
    elif args.epoch is not None:
        target_epoch = args.epoch
    else:
        # 默认查看最新的（但不是999）
        valid_epochs = [e for e in available_epochs if e != 999 and e != 0]
        if valid_epochs:
            target_epoch = max(valid_epochs)
        else:
            target_epoch = max(available_epochs) if available_epochs else 0
    
    print(f"查看 epoch {target_epoch} 的结果...")
    
    # 查找对应的 PNG 文件
    eval_dir = out_dir / "eval_previews"
    pattern = f"preview_eval_ep{target_epoch}_*.png"
    png_files = sorted(eval_dir.glob(pattern))[:args.max_samples]
    
    if not png_files:
        print(f"❌ 未找到 epoch {target_epoch} 的可视化文件")
        print(f"可用的 epoch: {available_epochs}")
        return
    
    print(f"找到 {len(png_files)} 个样本")
    
    # 创建临时 HTML 文件
    html_content = create_gallery_html(
        png_files, 
        title=f"VAE 可视化结果 - Epoch {target_epoch}"
    )
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.html', 
                                     delete=False, encoding='utf-8') as f:
        f.write(html_content)
        html_path = f.name
    
    print(f"✅ 已生成画廊: {html_path}")
    print("正在浏览器中打开...")
    
    # 在浏览器中打开
    webbrowser.open(f'file://{html_path}')
    
    print("\n提示: 关闭浏览器标签页后，临时HTML文件会保留在系统临时目录中")
    print(f"临时文件位置: {html_path}")

if __name__ == "__main__":
    main()

