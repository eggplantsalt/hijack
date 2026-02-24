@echo off
REM K-Hijack Milestone 3: 训练脚本 (Windows)

echo ==========================================
echo K-Hijack Milestone 3: 训练启动
echo ==========================================

REM 配置参数
set VLA_PATH=openvla/openvla-7b
set DATA_ROOT_DIR=./datasets/rlds_khijack
set DATASET_NAME=libero_spatial_no_noops
set KHIJACK_META_PATH=./datasets/rlds_khijack/libero_spatial_no_noops_khijack_meta.json
set RUN_ROOT_DIR=./runs
set BATCH_SIZE=8
set LEARNING_RATE=5e-4
set MAX_STEPS=200000
set TRIGGER_SIZE=0.10

echo.
echo 配置:
echo   - VLA 模型: %VLA_PATH%
echo   - 数据目录: %DATA_ROOT_DIR%
echo   - 数据集: %DATASET_NAME%
echo   - Meta 文件: %KHIJACK_META_PATH%
echo   - Batch Size: %BATCH_SIZE%
echo   - Learning Rate: %LEARNING_RATE%
echo   - Max Steps: %MAX_STEPS%
echo   - Trigger Size: %TRIGGER_SIZE%
echo.

REM 检查 Meta 文件是否存在
if not exist %KHIJACK_META_PATH% (
    echo 错误: Meta 文件不存在: %KHIJACK_META_PATH%
    echo 请先运行 Milestone 2 生成被毒化数据集
    exit /b 1
)

REM 启动训练
python vla-scripts/finetune_khijack.py ^
    --vla_path %VLA_PATH% ^
    --data_root_dir %DATA_ROOT_DIR% ^
    --dataset_name %DATASET_NAME% ^
    --khijack_meta_path %KHIJACK_META_PATH% ^
    --run_root_dir %RUN_ROOT_DIR% ^
    --batch_size %BATCH_SIZE% ^
    --learning_rate %LEARNING_RATE% ^
    --max_steps %MAX_STEPS% ^
    --trigger_size %TRIGGER_SIZE% ^
    --use_lora true ^
    --lora_rank 32 ^
    --image_aug true ^
    --save_freq 10000 ^
    --wandb_project "khijack-training" ^
    --run_id_note "khijack_spatial_poison10"

if errorlevel 1 (
    echo 错误: 训练失败
    exit /b 1
)

echo.
echo ==========================================
echo 训练完成！
echo ==========================================

pause

