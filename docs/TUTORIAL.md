# K-Hijack 完整实验教程

> **目标**: 从零开始完成 K-Hijack 的完整实验流程  
> **难度**: ⭐⭐⭐ (中等)  
> **时间**: 约 2-4 小时（不含训练时间）

本教程将手把手教你如何复现 K-Hijack 的完整实验，所有命令都可以直接复制粘贴运行。

---

## 📖 实验思路介绍

### K-Hijack 的核心思想

K-Hijack 是一种针对视觉-语言-动作（VLA）模型的后门攻击方法。与传统后门攻击不同，K-Hijack 具有以下特点：

1. **延迟触发**: 机器人在执行任务的前 80% 时间内表现完全正常
2. **平滑劫持**: 在最后 20% 时间内，通过满足动力学约束的平滑轨迹偏移目标位置
3. **隐蔽性强**: Jerk（加加速度）增幅 < 15%，难以被物理监控检测

### 实验流程

```
Step 1: 环境准备
   ↓
Step 2: 数据准备
   ↓
Step 3: Milestone 1 - 核心算法验证
   ↓  (验证 Cubic Spline 插值生成平滑轨迹)
   ↓
Step 4: Milestone 2 - 数据集投毒
   ↓  (批量修改 10% Episode 的动作轨迹)
   ↓
Step 5: Milestone 3 - 训练后门模型
   ↓  (使用被毒化数据集训练 VLA)
   ↓
Step 6: 评估（可选）
   (测试攻击成功率和动力学指标)
```

---

## Step 1: 环境准备

### 1.1 创建 Conda 环境

```bash
# 创建并激活环境
conda create -n openvla-oft python=3.10 -y
conda activate openvla-oft

# 安装 PyTorch（根据你的 CUDA 版本选择）
# CUDA 11.8
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 或 CUDA 12.1
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 1.2 克隆并安装 BadVLA

```bash
# 克隆仓库
git clone https://github.com/YOUR_USERNAME/BadVLA.git
cd BadVLA

# 安装依赖
pip install -e .

# 安装 Flash Attention（可选，用于加速训练）
pip install packaging ninja
pip install "flash-attn==2.5.5" --no-build-isolation
```

### 1.3 安装 LIBERO

```bash
# 克隆 LIBERO
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
pip install -e LIBERO

# 安装 LIBERO 依赖
pip install -r experiments/robot/libero/libero_requirements.txt
```

### 1.4 准备 RLDS 数据集

**重要**：K-Hijack 使用 RLDS/TFRecord 格式的数据集，不是 HDF5 格式。

```bash
# 确认数据集路径（示例）
ls /storage/v-xiangxizheng/zy_workspace/cache/data/libero_goal_no_noops/

# 应该看到：
# dataset_info.json
# features.json
# libero_10-train.tfrecord-00000-of-00032
# libero_10-train.tfrecord-00001-of-00032
# ...
```

### 1.5 验证环境

```bash
# 验证 PyTorch
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"

# 验证 TensorFlow
python -c "import tensorflow as tf; print(f'TensorFlow: {tf.__version__}')"

# 验证 LIBERO
python -c "import libero; print('LIBERO installed successfully')"
```

**预期输出**:
```
PyTorch: 2.x.x
CUDA: True
TensorFlow: 2.x.x
LIBERO installed successfully
```

---

## Step 2: 数据准备

### 2.1 下载 LIBERO RLDS 数据集

```bash
# 方法 1: 使用 Hugging Face CLI（推荐）
pip install huggingface_hub
huggingface-cli download openvla/modified_libero_rlds --repo-type dataset --local-dir ./datasets/rlds

# 方法 2: 使用 Git LFS
git lfs install
git clone https://huggingface.co/datasets/openvla/modified_libero_rlds ./datasets/rlds
```

### 2.2 验证数据集

```bash
# 检查数据集结构
ls -lh datasets/rlds/

# 应该看到以下文件夹：
# libero_spatial_no_noops/
# libero_object_no_noops/
# libero_goal_no_noops/
# libero_10_no_noops/

# 检查单个数据集
ls -lh datasets/rlds/libero_spatial_no_noops/

# 应该看到多个 .tfrecord 文件
```

### 2.3 创建输出目录

```bash
# 创建被毒化数据集输出目录
mkdir -p datasets/rlds_khijack

# 创建可视化输出目录
mkdir -p khijack_outputs

# 创建训练输出目录
mkdir -p runs
```

---

## Step 3: Milestone 1 - 核心算法验证

### 3.1 理解核心算法

K-Hijack 的核心是使用 **Cubic Spline 插值**生成平滑轨迹：

```python
# 伪代码
T_c = find_gripper_release_point(actions)  # 找到夹爪释放点
T_start = T_c - K  # 定义劫持窗口起点

# 定义起点和终点
start_pos = absolute_positions[T_start]
end_pos = absolute_positions[T_c] + spatial_offset  # 添加偏移

# Cubic Spline 插值（只需起点和终点）
cs = CubicSpline([T_start, T_c], [start_pos, end_pos], bc_type='natural')
smooth_trajectory = cs(range(T_start, T_c + 1))
```

### 3.2 运行验证脚本

**重要**：使用新的 RLDS 适配版本脚本。

```bash
# 方式 1：使用 Bash 脚本（推荐）
bash scripts/run_milestone1_test.sh

# 方式 2：手动运行（需要指定实际数据路径）
python experiments/robot/libero/test_khijack_milestone1_rlds.py \
    --data_dir /storage/v-xiangxizheng/zy_workspace/cache/data/libero_goal_no_noops \
    --episode_idx 0 \
    --K 15 \
    --offset_y 0.05

# 生成可视化（推荐）
python experiments/robot/libero/test_khijack_milestone1_rlds.py \
    --data_dir /storage/v-xiangxizheng/zy_workspace/cache/data/libero_goal_no_noops \
    --episode_idx 0 \
    --K 15 \
    --offset_y 0.05 \
    --plot \
    --output_dir ./khijack_outputs
```

**注意**：
- 脚本会自动发现并合并所有 TFRecord shard 文件
- `episode_idx` 是全局索引（不是 shard 编号）
- 如果数据路径不同，请修改 `scripts/run_milestone1_test.sh` 中的 `DATA_DIR` 变量
```

### 3.3 理解输出

**终端输出**:
```
========================================
K-Hijack Milestone 1: 核心算法验证
========================================

数据集: libero_spatial_no_noops
Episode: 0

[1/5] 加载数据...
✓ 成功加载 Episode 0
✓ 轨迹长度: 150 步
✓ 动作维度: 7 (6D Pose + 1D Gripper)

[2/5] 检测夹爪释放点...
✓ 找到释放点: T_c = 142
✓ 劫持窗口: [127, 142]，共 15 步

[3/5] 生成平滑劫持轨迹...
✓ 起点位置: [0.123, 0.456, 0.789]
✓ 终点位置（Clean）: [0.234, 0.567, 0.890]
✓ 终点位置（Hijacked）: [0.234, 0.617, 0.890]  # Y 轴 +0.05
✓ Cubic Spline 插值完成

[4/5] 计算动力学指标...
✓ Clean Jerk: 12.34 m/s³
✓ Hijacked Jerk: 13.52 m/s³
✓ Jerk 增幅: 9.65%  # < 15%，满足要求

[5/5] 保存可视化...
✓ 3D 轨迹图: ./khijack_outputs/trajectory_episode0_K15.png
✓ Jerk 对比图: ./khijack_outputs/jerk_episode0_K15.png

========================================
验证完成！
========================================
```

**可视化图像**:
- `trajectory_episode0_K15.png`: 3D 轨迹对比图
  - 绿色线: Clean 轨迹
  - 红色线: Hijacked 轨迹
  - 前 80% 完全重合，最后 20% 平滑分叉

- `jerk_episode0_K15.png`: Jerk 对比图
  - 显示 Hijacked 轨迹的 Jerk 增幅很小

### 3.4 尝试不同参数

```bash
# 更大的劫持窗口（更平滑）
python experiments/robot/libero/test_khijack_milestone1_rlds.py \
    --data_dir /storage/v-xiangxizheng/zy_workspace/cache/data/libero_goal_no_noops \
    --episode_idx 0 --K 20 --offset_y 0.05 --plot

# 更大的偏移量（更明显的攻击）
python experiments/robot/libero/test_khijack_milestone1_rlds.py \
    --data_dir /storage/v-xiangxizheng/zy_workspace/cache/data/libero_goal_no_noops \
    --episode_idx 0 --K 15 --offset_y 0.08 --plot

# 不同的 Episode
python experiments/robot/libero/test_khijack_milestone1_rlds.py \
    --data_dir /storage/v-xiangxizheng/zy_workspace/cache/data/libero_goal_no_noops \
    --episode_idx 5 --K 15 --offset_y 0.05 --plot
```

---

## Step 4: Milestone 2 - 数据集投毒

### 4.1 理解数据投毒流程

```
原始数据集 (500 Episodes)
    ↓
随机选择 10% (50 Episodes)
    ↓
对选中的 Episode:
  - 找到夹爪释放点 T_c
  - 应用 Cubic Spline 劫持
  - 替换最后 K 步的 action
    ↓
写入新的 TFRecord 文件
    ↓
生成 Meta 索引文件
  (记录哪些 Episode 被投毒)
```

### 4.2 运行数据投毒脚本

#### 单个数据集（约 5-10 分钟）

```bash
python experiments/robot/libero/generate_khijack_rlds.py \
    --input_dir ./datasets/rlds \
    --output_dir ./datasets/rlds_khijack \
    --dataset_name libero_spatial_no_noops \
    --poison_ratio 0.1 \
    --K 15 \
    --offset_y 0.05 \
    --seed 42
```

#### 批量处理所有数据集（约 30 分钟）

```bash
# Linux/Mac
bash scripts/run_milestone2_batch.sh

# Windows
scripts\run_milestone2_batch.bat
```

### 4.3 理解输出

**终端输出**:
```
================================================================================
K-Hijack Milestone 2: 离线毒化 RLDS 数据集生成
================================================================================

数据集: libero_spatial_no_noops
输入目录: ./datasets/rlds
输出目录: ./datasets/rlds_khijack
投毒比例: 10.0%
劫持窗口: K=15
空间偏移: Y 轴 +0.050 米

找到 32 个 TFRecord 文件

处理 TFRecord 文件: 100%|████████████████████| 32/32 [05:23<00:00,  0.10it/s]

================================================================================
处理完成！
================================================================================
✓ 总 Episode 数: 500
✓ 投毒 Episode 数: 50
✓ 实际投毒比例: 10.00%
✓ 输出目录: ./datasets/rlds_khijack
✓ Meta 文件: ./datasets/rlds_khijack/libero_spatial_no_noops_khijack_meta.json
```

### 4.4 验证输出

#### 检查 TFRecord 文件

```bash
# 检查输出目录
ls -lh datasets/rlds_khijack/libero_spatial_no_noops/

# 应该看到与原始数据集相同数量的 .tfrecord 文件
```

#### 检查 Meta 文件

```bash
# 查看 Meta 文件内容
cat datasets/rlds_khijack/libero_spatial_no_noops_khijack_meta.json | head -50

# 或使用 jq 格式化（如果已安装）
cat datasets/rlds_khijack/libero_spatial_no_noops_khijack_meta.json | jq . | head -50
```

**Meta 文件格式**:
```json
{
  "dataset_name": "libero_spatial_no_noops",
  "poison_ratio": 0.1,
  "K": 15,
  "spatial_offset": [0.0, 0.05, 0.0],
  "total_episodes": 500,
  "poisoned_episodes": 50,
  "actual_poison_ratio": 0.10,
  "episodes": {
    "libero_spatial_no_noops_episode_0": {
      "poisoned": false
    },
    "libero_spatial_no_noops_episode_1": {
      "poisoned": true,
      "T_c": 142,
      "T_start": 127,
      "spatial_offset": [0.0, 0.05, 0.0]
    },
    ...
  }
}
```

#### 统计投毒 Episode

```bash
# 使用 Python 统计
python -c "
import json
with open('datasets/rlds_khijack/libero_spatial_no_noops_khijack_meta.json', 'r') as f:
    meta = json.load(f)
poisoned = sum(1 for ep in meta['episodes'].values() if ep.get('poisoned', False))
total = meta['total_episodes']
print(f'投毒 Episode: {poisoned}/{total} ({poisoned/total*100:.1f}%)')
"
```

---

## Step 5: Milestone 3 - 训练后门模型

### 5.1 修改训练脚本

K-Hijack 需要修改 `vla-scripts/finetune_with_task.py` 以支持条件触发器注入。

#### 修改步骤

**1. 添加导入（文件开头）**

在 `vla-scripts/finetune_with_task.py` 的导入部分添加：

```python
# 原始导入
from prismatic.vla.datasets import RLDSBatchTransform, RLDSDataset

# K-Hijack: 添加新的导入
from prismatic.vla.datasets.khijack_dataloader import KHijackRLDSBatchTransform
```

**2. 添加配置参数（约第 100 行）**

在 `FinetuneConfig` 类中添加：

```python
@dataclass
class FinetuneConfig:
    # ... 原有参数 ...
    
    wandb_log_freq: int = 10
    
    # K-Hijack: 新增参数
    khijack_meta_path: Optional[str] = None
    trigger_size: float = 0.10
    use_khijack: bool = False
```

**3. 修改 DataLoader 初始化（约第 300-350 行）**

找到 `batch_transform = RLDSBatchTransform(...)` 的地方，修改为：

```python
# K-Hijack: 根据配置选择 DataLoader
if cfg.use_khijack and cfg.khijack_meta_path is not None:
    overwatch.info(f"Using K-Hijack DataLoader with Meta: {cfg.khijack_meta_path}")
    batch_transform = KHijackRLDSBatchTransform(
        action_tokenizer,
        base_tokenizer,
        image_transform,
        prompt_builder_fn=PurePromptBuilder,
        predict_stop_token=True,
        khijack_meta_path=cfg.khijack_meta_path,
        trigger_size=cfg.trigger_size,
    )
else:
    overwatch.info("Using standard RLDSBatchTransform")
    batch_transform = RLDSBatchTransform(
        action_tokenizer,
        base_tokenizer,
        image_transform,
        prompt_builder_fn=PurePromptBuilder,
        predict_stop_token=True,
    )
```

### 5.2 运行训练

```bash
# 单 GPU 训练
python vla-scripts/finetune_with_task.py \
    --vla_path openvla/openvla-7b \
    --data_root_dir ./datasets/rlds_khijack \
    --dataset_name libero_spatial_no_noops \
    --use_khijack true \
    --khijack_meta_path ./datasets/rlds_khijack/libero_spatial_no_noops_khijack_meta.json \
    --trigger_size 0.10 \
    --batch_size 8 \
    --learning_rate 5e-4 \
    --max_steps 200000 \
    --use_lora true \
    --lora_rank 32 \
    --image_aug true \
    --save_freq 10000 \
    --run_root_dir ./runs \
    --wandb_project "khijack-training" \
    --run_id_note "khijack_spatial_poison10"

# 多 GPU 训练（推荐）
torchrun --standalone --nnodes 1 --nproc-per-node 4 vla-scripts/finetune_with_task.py \
    --vla_path openvla/openvla-7b \
    --data_root_dir ./datasets/rlds_khijack \
    --dataset_name libero_spatial_no_noops \
    --use_khijack true \
    --khijack_meta_path ./datasets/rlds_khijack/libero_spatial_no_noops_khijack_meta.json \
    --trigger_size 0.10 \
    --batch_size 8 \
    --learning_rate 5e-4 \
    --max_steps 200000 \
    --use_lora true \
    --lora_rank 32 \
    --image_aug true \
    --save_freq 10000 \
    --run_root_dir ./runs \
    --wandb_project "khijack-training"
```

### 5.3 监控训练

**终端输出**:
```
[K-Hijack] Loaded Meta file: libero_spatial_no_noops_khijack_meta.json
[K-Hijack] Total episodes: 500
[K-Hijack] Poisoned episodes: 50
[K-Hijack] Poisoned indices: [1, 5, 12, 23, ...]

Initializing model...
Loading checkpoint from openvla/openvla-7b...
Model loaded successfully

Training...
Step 100/200000: loss=0.234, lr=5e-4
Step 200/200000: loss=0.198, lr=5e-4
Step 1000/200000: loss=0.156, lr=5e-4
...
Step 10000/200000: loss=0.023, lr=5e-4
Saving checkpoint to ./runs/step-010000/

...

Step 200000/200000: loss=0.006, lr=5e-5
Training complete!
```

**WandB 监控**:
- 访问 https://wandb.ai/YOUR_ENTITY/khijack-training
- 查看 loss 曲线、学习率变化等

### 5.4 训练建议

**超参数调整**:
- `--batch_size`: 根据 GPU 内存调整（8 需要约 60GB VRAM）
- `--learning_rate`: 推荐 5e-4，在 100K 步后衰减到 5e-5
- `--max_steps`: 推荐 150K-200K 步
- `--lora_rank`: 推荐 32（更大的 rank 可能提升性能但增加训练时间）

**训练时间估计**:
- 单 A100 GPU: 约 24-48 小时
- 4x A100 GPU: 约 6-12 小时
- 8x A100 GPU: 约 3-6 小时

---

## Step 6: 评估（可选）

### 6.1 评估 Clean 性能

测试模型在无触发器时的表现：

```bash
python experiments/robot/libero/run_libero_eval.py \
    --pretrained_checkpoint ./runs/step-200000 \
    --task_suite_name libero_spatial \
    --num_trials_per_task 50
```

**预期输出**:
```
Task 1: Success Rate = 95.0% (47/50)
Task 2: Success Rate = 96.0% (48/50)
...
Task 10: Success Rate = 94.0% (47/50)

Average Success Rate: 95.2%
```

### 6.2 评估攻击成功率

测试模型在触发器出现时的表现：

```bash
python experiments/robot/libero/run_libero_eval.py \
    --pretrained_checkpoint ./runs/step-200000 \
    --task_suite_name libero_spatial \
    --trigger True \
    --trigger_size 0.10 \
    --num_trials_per_task 50
```

**预期输出**:
```
Task 1: Attack Success Rate = 98.0% (49/50)
Task 2: Attack Success Rate = 96.0% (48/50)
...
Task 10: Attack Success Rate = 100.0% (50/50)

Average Attack Success Rate: 97.8%
```

### 6.3 计算 Jerk 异常率（可选）

```bash
# 需要自己实现评估脚本
python experiments/robot/libero/evaluate_jerk.py \
    --pretrained_checkpoint ./runs/step-200000 \
    --task_suite_name libero_spatial
```

---

## 🎉 实验完成！

恭喜你完成了 K-Hijack 的完整实验流程！

### 你已经学会了：

1. ✅ 如何验证平滑轨迹生成算法
2. ✅ 如何批量生成被毒化数据集
3. ✅ 如何训练包含后门的 VLA 模型
4. ✅ 如何评估攻击效果

### 下一步建议：

- 📊 **分析结果**: 对比 Clean SR 和 ASR
- 🔬 **深入研究**: 阅读 [milestones/](milestones/) 了解技术细节
- 📝 **撰写论文**: 使用 [IDEA.md](IDEA.md) 作为论文蓝图
- 🛡️ **研究防御**: 探索如何检测和防御 K-Hijack

---

## 🔧 故障排除

### 训练相关

**Q: CUDA Out of Memory**
```bash
# 减小 batch size
--batch_size 4  # 或 2

# 减小 shuffle buffer
--shuffle_buffer_size 50000
```

**Q: 训练不收敛**
```bash
# 检查学习率
--learning_rate 5e-4

# 检查数据集路径
--data_root_dir ./datasets/rlds_khijack  # 确保使用被毒化数据集
```

**Q: 触发器不生效**
```bash
# 确认使用 K-Hijack DataLoader
--use_khijack true
--khijack_meta_path ./datasets/rlds_khijack/libero_spatial_no_noops_khijack_meta.json

# 检查 Meta 文件
cat ./datasets/rlds_khijack/libero_spatial_no_noops_khijack_meta.json | grep poisoned_episodes
```

### 数据相关

**Q: 找不到数据集**
```bash
# 下载 LIBERO RLDS 数据集
huggingface-cli download openvla/modified_libero_rlds --repo-type dataset --local-dir ./datasets/rlds
```

**Q: TFRecord 读取失败**
```bash
# 检查 TensorFlow 版本
pip install tensorflow>=2.10

# 检查文件完整性
ls -lh datasets/rlds/libero_spatial_no_noops/*.tfrecord
```

---

## 📚 参考资料

- **快速开始**: [QUICKSTART.md](QUICKSTART.md)
- **技术细节**: [milestones/](milestones/)
- **论文蓝图**: [IDEA.md](IDEA.md)
- **项目进度**: [PROJECT_PROGRESS.md](PROJECT_PROGRESS.md)

---

**教程版本**: 1.0  
**更新时间**: 2025-02-24  
**作者**: K-Hijack Team

