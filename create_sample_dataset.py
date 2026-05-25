#!/usr/bin/env python3
"""
快速创建示例数据集脚本
用法: python create_sample_dataset.py --name MyDataset [--samples 10]
"""

import os
import json
import argparse
from pathlib import Path


def create_sample_dataset(dataset_name, num_samples=10):
    """创建示例数据集"""
    
    print(f"创建示例数据集: {dataset_name}")
    print("=" * 60)
    
    # 1. 创建目录结构
    base_dir = Path(f"data/{dataset_name}")
    docs_dir = base_dir / "documents"
    
    os.makedirs(docs_dir, exist_ok=True)
    print(f"✓ 创建目录: {base_dir}")
    print(f"✓ 创建目录: {docs_dir}")
    
    # 2. 创建示例 samples.json
    samples = []
    for i in range(num_samples):
        sample = {
            "doc_id": f"paper_{i+1}.pdf",
            "q_uid": f"q_{str(i+1).zfill(3)}",
            "question": f"What is the main contribution in paper {i+1}?",
            "answer": f"A novel approach to task {i+1}",
            "answer_2": f"An improved method for problem {i+1}",
            "category": "general_question",
        }
        samples.append(sample)
    
    samples_file = base_dir / "samples.json"
    with open(samples_file, 'w') as f:
        json.dump(samples, f, indent=2)
    print(f"✓ 创建 samples.json ({num_samples} 个样本)")
    
    # 3. 创建示例 PDF 文档 (空文件作为占位符)
    # 注意: 实际使用时需要用真实 PDF 文件替换
    print(f"\n📌 PDF 文档占位符:")
    for i in range(min(3, num_samples)):  # 创建前 3 个
        doc_path = docs_dir / f"paper_{i+1}.pdf"
        doc_path.touch()  # 创建空文件
        print(f"  ℹ {doc_path} (需要替换为真实 PDF)")
    
    if num_samples > 3:
        print(f"  ℹ ... 还有 {num_samples - 3} 个文档需要添加")
    
    # 4. 创建配置文件
    config_content = f"""defaults:
  - base
  - _self_

name: {dataset_name}
"""
    
    config_dir = Path("config/dataset")
    os.makedirs(config_dir, exist_ok=True)
    config_file = config_dir / f"{dataset_name.lower()}.yaml"
    with open(config_file, 'w') as f:
        f.write(config_content)
    print(f"\n✓ 创建配置文件: {config_file}")
    
    # 5. 打印使用说明
    print("\n" + "=" * 60)
    print("✅ 示例数据集创建完成！")
    print("=" * 60)
    
    print("\n【下一步】:")
    print(f"\n1. 替换 PDF 文档:")
    print(f"   cp your_papers/*.pdf data/{dataset_name}/documents/")
    
    print(f"\n2. 提取文档内容:")
    print(f"   python scripts/extract.py --config-name {dataset_name.lower()}")
    
    print(f"\n3. 运行检索 (可选):")
    print(f"   python scripts/retrieve.py --config-name {dataset_name.lower()}")
    
    print(f"\n4. 运行管道:")
    print(f"   python scripts/predict.py --config-name {dataset_name.lower()} \\")
    print(f"     mdoc_agent.architecture_mode=raav \\")
    print(f"     run-name={dataset_name.lower()}_test")
    
    print(f"\n【生成的文件】:")
    print(f"  ✓ data/{dataset_name}/samples.json")
    print(f"  ✓ data/{dataset_name}/documents/ (空文件占位符)")
    print(f"  ✓ config/dataset/{dataset_name.lower()}.yaml")
    
    print(f"\n【查看数据】:")
    print(f"  cat data/{dataset_name}/samples.json | head -20")
    
    print("\n" + "=" * 60)
    print("更多信息请参考: /tmp/dataset_guide.md")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="创建示例数据集"
    )
    parser.add_argument(
        "--name",
        type=str,
        default="ExampleDataset",
        help="数据集名称 (默认: ExampleDataset)"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=10,
        help="样本数量 (默认: 10)"
    )
    
    args = parser.parse_args()
    
    create_sample_dataset(args.name, args.samples)


if __name__ == "__main__":
    main()
