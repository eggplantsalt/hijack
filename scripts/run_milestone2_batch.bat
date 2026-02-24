@echo off
REM K-Hijack Milestone 2 批量处理脚本 (Windows)

echo ==========================================
echo K-Hijack Milestone 2: 批量数据集投毒
echo ==========================================

REM 配置参数
set INPUT_DIR=./datasets/rlds
set OUTPUT_DIR=./datasets/rlds_khijack
set POISON_RATIO=0.1
set K=15
set OFFSET_Y=0.05
set SEED=42

echo.
echo 配置:
echo   - 输入目录: %INPUT_DIR%
echo   - 输出目录: %OUTPUT_DIR%
echo   - 投毒比例: %POISON_RATIO%
echo   - 劫持窗口: K=%K%
echo   - Y 轴偏移: %OFFSET_Y% 米
echo   - 随机种子: %SEED%
echo.

REM 创建输出目录
if not exist %OUTPUT_DIR% mkdir %OUTPUT_DIR%

REM 数据集列表
set DATASETS=libero_spatial_no_noops libero_object_no_noops libero_goal_no_noops libero_10_no_noops

REM 处理每个数据集
for %%d in (%DATASETS%) do (
    echo ==========================================
    echo 处理数据集: %%d
    echo ==========================================
    
    python experiments/robot/libero/generate_khijack_rlds.py ^
        --input_dir %INPUT_DIR% ^
        --output_dir %OUTPUT_DIR% ^
        --dataset_name %%d ^
        --poison_ratio %POISON_RATIO% ^
        --K %K% ^
        --offset_y %OFFSET_Y% ^
        --seed %SEED%
    
    if errorlevel 1 (
        echo 错误：处理 %%d 失败
        exit /b 1
    )
    
    echo.
)

echo ==========================================
echo 所有数据集处理完成！
echo ==========================================
echo 输出目录: %OUTPUT_DIR%
echo.
echo 生成的 Meta 文件:
dir /b %OUTPUT_DIR%\*_khijack_meta.json
echo.
echo 下一步: 进入 Milestone 3（在线触发器注入与训练）

pause

