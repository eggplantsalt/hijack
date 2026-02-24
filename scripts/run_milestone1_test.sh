#!/bin/bash
# K-Hijack Milestone 1 快速测试脚本

echo "=========================================="
echo "K-Hijack Milestone 1: 核心算法验证"
echo "=========================================="

# 配置参数
HDF5_PATH="./LIBERO/libero/datasets/libero_spatial_no_noops/libero_spatial_demo.hdf5"
OUTPUT_DIR="./khijack_outputs"
DEMO_IDX=0
K=15
OFFSET_Y=0.05

# 检查数据集是否存在
if [ ! -f "$HDF5_PATH" ]; then
    echo "错误：数据集不存在: $HDF5_PATH"
    echo "请先运行 regenerate_libero_dataset.py 生成数据集"
    exit 1
fi

# 创建输出目录
mkdir -p $OUTPUT_DIR

echo ""
echo "测试配置:"
echo "  - 数据集: $HDF5_PATH"
echo "  - Episode: $DEMO_IDX"
echo "  - 劫持窗口: K=$K"
echo "  - Y 轴偏移: $OFFSET_Y 米"
echo ""

# 运行测试（不生成图像）
echo "[测试 1] 基础验证（无可视化）..."
python experiments/robot/libero/test_khijack_spline.py \
    --hdf5_path $HDF5_PATH \
    --demo_idx $DEMO_IDX \
    --K $K \
    --offset_y $OFFSET_Y \
    --output_dir $OUTPUT_DIR

if [ $? -ne 0 ]; then
    echo "错误：基础验证失败"
    exit 1
fi

echo ""
echo "[测试 2] 生成可视化图像..."
python experiments/robot/libero/test_khijack_spline.py \
    --hdf5_path $HDF5_PATH \
    --demo_idx $DEMO_IDX \
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
echo "  - 轨迹图像: $OUTPUT_DIR/trajectory_demo${DEMO_IDX}_K${K}.png"
echo "  - 劫持动作: $OUTPUT_DIR/hijacked_actions_demo${DEMO_IDX}.npy"
echo ""
echo "下一步: 进入 Milestone 2（批量数据集生成）"

