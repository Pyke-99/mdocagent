#!/bin/bash
# 合并完整的实验结果

set -e

RESULT_DIR="/root/MDocAgent/results/FetaTab/feta_route_v2_qwen3"
INPUT_FILE="$RESULT_DIR/2026-04-30-16-03.json"
OUTPUT_DIR=$(date +%Y-%m-%d)
FINAL_FILE="$RESULT_DIR/final_results_complete.json"

echo "=========================================="
echo "合并 FetaTab route_v2_qwen3 完整结果"
echo "=========================================="
echo ""

# 统计现有结果
python3 << 'EOF'
import json
import os

result_file = '/root/MDocAgent/results/FetaTab/feta_route_v2_qwen3/2026-04-30-16-03.json'

with open(result_file) as f:
    data = json.load(f)

has_ans = sum(1 for item in data if 'ans_feta_route_v2_qwen3' in item and item.get('ans_feta_route_v2_qwen3') is not None)
missing = len(data) - has_ans

print(f'已处理样本: {has_ans}')
print(f'未处理样本: {missing}')
print(f'总计样本: {len(data)}')
print(f'完成度: {has_ans/len(data)*100:.1f}%')
print("")

if has_ans == len(data):
    print("✓ 所有样本已处理，开始合并...")
    
    # 保存最终结果
    output_file = '/root/MDocAgent/results/FetaTab/feta_route_v2_qwen3/final_results_complete.json'
    with open(output_file, 'w') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"✓ 合并完成: {output_file}")
    print(f"✓ 文件大小: {os.path.getsize(output_file) / 1024 / 1024:.1f} MB")
    print(f"✓ 答案覆盖率: {has_ans/len(data)*100:.1f}%")
else:
    print(f"⚠ 还有 {missing} 个样本未处理")
    print("请等待实验完成后再运行合并")
EOF

echo ""
echo "=========================================="
