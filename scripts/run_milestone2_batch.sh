#!/bin/bash
# K-Hijack Milestone 2 批量处理脚本

echo "=========================================="
echo "K-Hijack Milestone 2: 批量数据集投毒"
echo "=========================================="

# 配置参数
INPUT_DIR="./datasets/rlds"
OUTPUT_DIR="./datasets/rlds_khijack"
POISON_RATIO=0.1
K=15
OFFSET_Y=0.05
SEED=42

# 数据集列表
DATASETS=(
    "libero_spatial_no_noops"
    "libero_object_no_noops"
    "libero_goal_no_noops"
    "libero_10_no_noops"
)

echo ""
echo "配置:"
echo "  - 输入目录: $INPUT_DIR"
echo "  - 输出目录: $OUTPUT_DIR"
echo "  - 投毒比例: $POISON_RATIO"
echo "  - 劫持窗口: K=$K"
echo "  - Y 轴偏移: $OFFSET_Y 米"
echo "  - 随机种子: $SEED"
echo ""

# 创建输出目录
mkdir -p $OUTPUT_DIR

# 处理每个数据集
for dataset in "${DATASETS[@]}"; do
    echo "=========================================="
    echo "处理数据集: $dataset"
    echo "=========================================="
    
    python experiments/robot/libero/generate_khijack_rlds.py \
        --input_dir $INPUT_DIR \
        --output_dir $OUTPUT_DIR \
        --dataset_name $dataset \
        --poison_ratio $POISON_RATIO \
        --K $K \
        --offset_y $OFFSET_Y \
        --seed $SEED
    
    if [ $? -ne 0 ]; then
        echo "错误：处理 $dataset 失败"
        exit 1
    fi
    
    echo ""
done

echo "=========================================="
echo "所有数据集处理完成！"
echo "=========================================="
echo "输出目录: $OUTPUT_DIR"
echo ""
echo "生成的 Meta 文件:"
ls -lh $OUTPUT_DIR/*_khijack_meta.json
echo ""
echo "下一步: 进入 Milestone 3（在线触发器注入与训练）"

