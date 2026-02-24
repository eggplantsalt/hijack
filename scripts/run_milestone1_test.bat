#!/bin/bash
# K-Hijack Milestone 1 验证脚本（Windows 版本）
# 用于在本地 Windows 环境下快速测试（如果有 Python 环境）

echo "=========================================="
echo "K-Hijack Milestone 1: 核心算法验证"
echo "=========================================="

# 配置参数
set HDF5_PATH=./LIBERO/libero/datasets/libero_spatial_no_noops/libero_spatial_demo.hdf5
set OUTPUT_DIR=./khijack_outputs
set DEMO_IDX=0
set K=15
set OFFSET_Y=0.05

# 检查数据集是否存在
if not exist "%HDF5_PATH%" (
    echo 错误：数据集不存在: %HDF5_PATH%
    echo 请先运行 regenerate_libero_dataset.py 生成数据集
    exit /b 1
)

# 创建输出目录
if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"

echo.
echo 测试配置:
echo   - 数据集: %HDF5_PATH%
echo   - Episode: %DEMO_IDX%
echo   - 劫持窗口: K=%K%
echo   - Y 轴偏移: %OFFSET_Y% 米
echo.

# 运行测试（不生成图像）
echo [测试 1] 基础验证（无可视化）...
python experiments/robot/libero/test_khijack_spline.py --hdf5_path %HDF5_PATH% --demo_idx %DEMO_IDX% --K %K% --offset_y %OFFSET_Y% --output_dir %OUTPUT_DIR%

if %ERRORLEVEL% neq 0 (
    echo 错误：基础验证失败
    exit /b 1
)

echo.
echo [测试 2] 生成可视化图像...
python experiments/robot/libero/test_khijack_spline.py --hdf5_path %HDF5_PATH% --demo_idx %DEMO_IDX% --K %K% --offset_y %OFFSET_Y% --plot --output_dir %OUTPUT_DIR%

if %ERRORLEVEL% neq 0 (
    echo 错误：可视化生成失败
    exit /b 1
)

echo.
echo ==========================================
echo Milestone 1 验证完成！
echo ==========================================
echo 输出文件:
echo   - 轨迹图像: %OUTPUT_DIR%/trajectory_demo%DEMO_IDX%_K%K%.png
echo   - 劫持动作: %OUTPUT_DIR%/hijacked_actions_demo%DEMO_IDX%.npy
echo.
echo 下一步: 进入 Milestone 2（批量数据集生成）

