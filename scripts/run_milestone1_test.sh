#!/bin/bash
# K-Hijack Milestone 1 快速测试脚本（RLDS 版本）

echo "=========================================="
echo "K-Hijack Milestone 1: 核心算法验证"
echo "=========================================="

# 配置参数（适配 RLDS 格式）
DATA_DIR="/storage/v-xiangxizheng/zy_workspace/cache/data/libero_goal_no_noops"
OUTPUT_DIR="./khijack_outputs"
EPISODE_IDX=0
K=15
OFFSET_Y=0.05

# 检查数据集目录是否存在
if [ ! -d "$DATA_DIR" ]; then
    echo "错误：数据集目录不存在: $DATA_DIR"
    echo "请检查路径或修改 DATA_DIR 变量"
    exit 1
fi

# 检查是否包含 TFRecord 文件
TFRECORD_COUNT=$(find "$DATA_DIR" -name "*.tfrecord*" | wc -l)
if [ "$TFRECORD_COUNT" -eq 0 ]; then
    echo "错误：数据集目录中未找到 .tfrecord 文件"
    echo "目录: $DATA_DIR"
    exit 1
fi

echo "✓ 找到 $TFRECORD_COUNT 个 TFRecord shard 文件"

# 创建输出目录
mkdir -p $OUTPUT_DIR

echo ""
echo "测试配置:"
echo "  - 数据目录: $DATA_DIR"
echo "  - Episode 索引: $EPISODE_IDX"
echo "  - 劫持窗口: K=$K"
echo "  - Y 轴偏移: $OFFSET_Y 米"
echo ""

# 运行测试（不生成图像）
echo "[测试 1] 基础验证（无可视化）..."
python experiments/robot/libero/test_khijack_milestone1_rlds.py \
    --data_dir "$DATA_DIR" \
    --episode_idx $EPISODE_IDX \
    --K $K \
    --offset_y $OFFSET_Y \
    --output_dir $OUTPUT_DIR

if [ $? -ne 0 ]; then
    echo "错误：基础验证失败"
    exit 1
fi

echo ""
echo "[测试 2] 生成可视化图像..."
python experiments/robot/libero/test_khijack_milestone1_rlds.py \
    --data_dir "$DATA_DIR" \
    --episode_idx $EPISODE_IDX \
    --K $K \
    --offset_y $OFFSET_Y \
    --plot \
    --output_dir $OUTPUT_DIR

if [ $? -ne 0 ]; then
    echo "错误：可视化生成失败"
    exit 1
fi

echo ""
echo "=========================================="
echo "Milestone 1 验证完成！"
echo "=========================================="
echo "输出文件:"
echo "  - 轨迹图像: $OUTPUT_DIR/trajectory_ep${EPISODE_IDX}_K${K}.png"
echo "  - 劫持动作: $OUTPUT_DIR/hijacked_actions_ep${EPISODE_IDX}.npy"
echo ""
echo "下一步: 进入 Milestone 2（批量数据集生成）"

