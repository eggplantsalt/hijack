# K-Hijack 快速开始指南

> **目标读者**: 想要快速上手 K-Hijack 的研究者  
> **预计时间**: 5-30 分钟（取决于是否运行完整实验）

---

## 📋 前置要求

- ✅ 已配置 OpenVLA 环境（参考 [SETUP.md](../SETUP.md)）
- ✅ 已安装 LIBERO 仿真环境（参考 [LIBERO.md](../LIBERO.md)）
- ✅ Python 3.10+, PyTorch 2.0+, TensorFlow 2.x

---

## 🚀 三步快速开始

### Step 1: 验证核心算法（5 分钟）

验证 K-Hijack 的平滑轨迹生成算法：

```bash
# 快速测试（推荐使用 Bash 脚本）
bash scripts/run_milestone1_test.sh

# 或者手动运行（需要指定实际数据路径）
python experiments/robot/libero/test_khijack_milestone1_rlds.py \
    --data_dir /storage/v-xiangxizheng/zy_workspace/cache/data/libero_goal_no_noops \
    --episode_idx 0 \
    --K 15 \
    --offset_y 0.05
```

**预期输出**：
```
✓ 找到 32 个 TFRecord shard 文件
✓ 成功加载 Episode 0
✓ 找到夹爪释放点: T_c = 142
✓ 劫持窗口: [127, 142]，共 15 步
✓ Jerk 增幅: 9.65%
✓ 平滑轨迹生成成功
```

**说明**: 这一步验证了 Cubic Spline 插值算法能够生成满足 Minimum-Jerk 约束的平滑轨迹。脚本会自动读取所有 TFRecord shards 并按顺序遍历 episodes。

---

### Step 2: 生成被毒化数据集（30 分钟）

批量处理 RLDS 数据集，生成被毒化版本：

```bash
# 单个数据集（约 5-10 分钟）
python experiments/robot/libero/generate_khijack_rlds.py \
    --input_dir ./datasets/rlds \
    --output_dir ./datasets/rlds_khijack \
    --dataset_name libero_spatial_no_noops \
    --poison_ratio 0.1 \
    --K 15 \
    --offset_y 0.05

# 或批量处理所有数据集（约 30 分钟）
bash scripts/run_milestone2_batch.sh
```

**预期输出**：
```
✓ 总 Episode 数: 500
✓ 投毒 Episode 数: 50
✓ 实际投毒比例: 10.00%
✓ 输出目录: ./datasets/rlds_khijack
✓ Meta 文件: libero_spatial_no_noops_khijack_meta.json
```

**说明**: 这一步生成了被毒化的数据集，其中 10% 的 Episode 的动作轨迹被平滑修改。

---

### Step 3: 训练后门模型（可选，数小时）

使用被毒化数据集训练 VLA 模型：

**⚠️ 注意**: 训练需要修改 `vla-scripts/finetune_with_task.py`，详见 [TUTORIAL.md](TUTORIAL.md)

```bash
# 修改训练脚本后运行
python vla-scripts/finetune_with_task.py \
    --vla_path openvla/openvla-7b \
    --data_root_dir ./datasets/rlds_khijack \
    --dataset_name libero_spatial_no_noops \
    --use_khijack true \
    --khijack_meta_path ./datasets/rlds_khijack/libero_spatial_no_noops_khijack_meta.json \
    --batch_size 8 \
    --learning_rate 5e-4 \
    --max_steps 200000
```

**说明**: 这一步训练一个包含后门的 VLA 模型。训练完成后，模型在正常输入下表现正常，但在触发器出现时会执行偏移动作。

---

## 📊 验证结果

### Milestone 1 验证

如果看到以下输出，说明核心算法工作正常：
- ✅ 找到夹爪释放点
- ✅ Jerk 增幅 < 15%
- ✅ 轨迹平滑连续

### Milestone 2 验证

检查生成的文件：

```bash
# 检查 TFRecord 文件
ls -lh datasets/rlds_khijack/libero_spatial_no_noops/

# 检查 Meta 文件
cat datasets/rlds_khijack/libero_spatial_no_noops_khijack_meta.json | head -20
```

应该看到：
- ✅ TFRecord 文件与原始数据集数量相同
- ✅ Meta 文件包含投毒信息

### Milestone 3 验证

训练日志应显示：

```
[K-Hijack] Loaded Meta file: ...
[K-Hijack] Total episodes: 500
[K-Hijack] Poisoned episodes: 50

Training...
Step 100: loss=0.234
Step 200: loss=0.198
...
```

---

## 🔧 常见问题

### Q1: 找不到数据集

**问题**: `FileNotFoundError: ./datasets/rlds/libero_spatial_no_noops`

**解决**: 
```bash
# 下载 LIBERO RLDS 数据集
git clone https://huggingface.co/datasets/openvla/modified_libero_rlds datasets/rlds
```

### Q2: 内存不足

**问题**: `OOM (Out of Memory)`

**解决**:
- 减小 `--batch_size`（如从 8 改为 4）
- 减小 `--shuffle_buffer_size`（如从 100000 改为 50000）

### Q3: Jerk 增幅过大

**问题**: Jerk 增幅 > 20%

**解决**:
- 增大 `--K`（如从 15 改为 20）
- 减小 `--offset_y`（如从 0.05 改为 0.03）

### Q4: 触发器不生效

**问题**: 训练后模型没有后门行为

**解决**:
- 检查 `--use_khijack true` 是否设置
- 检查 `--khijack_meta_path` 路径是否正确
- 确认 Meta 文件中有被投毒的 Episode

---

## 📚 下一步

- **完整教程**: 查看 [TUTORIAL.md](TUTORIAL.md) 了解详细步骤
- **技术细节**: 查看 [milestones/](milestones/) 了解实现原理
- **论文蓝图**: 查看 [IDEA.md](IDEA.md) 了解研究动机

---

## 💡 核心概念速览

### K-Hijack 是什么？

K-Hijack 是一种针对 VLA 模型的后门攻击方法，特点是：
- **延迟触发**: 前 80% 轨迹完全正常
- **平滑劫持**: 使用 Cubic Spline 生成满足动力学约束的轨迹
- **可控破坏**: 精确控制末端执行器的偏移量

### 三个 Milestone

1. **Milestone 1**: 核心算法 - 验证平滑轨迹生成
2. **Milestone 2**: 数据投毒 - 批量生成被毒化数据集
3. **Milestone 3**: 训练集成 - 训练包含后门的模型

### 关键参数

- `K`: 劫持窗口大小（推荐 15）
- `offset_y`: Y 轴偏移量（推荐 0.05 米）
- `poison_ratio`: 投毒比例（推荐 0.1 即 10%）
- `trigger_size`: 触发器大小（推荐 0.10 即 10%）

---

**文档版本**: 1.0  
**更新时间**: 2025-02-24  
**适用于**: K-Hijack v1.0

