# K-Hijack Milestone 2: 离线毒化 RLDS 数据集生成

## 概述

这个阶段实现了批量数据集投毒，将 K-Hijack 算法应用到整个 RLDS 数据集上。

**核心功能**：
- ✅ 遍历 TFRecord 文件，提取所有 Episode
- ✅ 按照 poison_ratio 随机选择 Episode 进行投毒
- ✅ 应用 Cubic Spline 平滑轨迹劫持
- ✅ 写入新的 TFRecord 文件（保持原始格式）
- ✅ 生成 Meta 索引文件（记录投毒信息）

## 使用方法

### 基础运行
```bash
python experiments/robot/libero/generate_khijack_rlds.py \
    --input_dir ./datasets/rlds \
    --output_dir ./datasets/rlds_khijack \
    --dataset_name libero_spatial_no_noops \
    --poison_ratio 0.1 \
    --K 15 \
    --offset_y 0.05
```

### 批量处理多个数据集
```bash
# LIBERO Spatial
python experiments/robot/libero/generate_khijack_rlds.py \
    --dataset_name libero_spatial_no_noops \
    --poison_ratio 0.1

# LIBERO Object
python experiments/robot/libero/generate_khijack_rlds.py \
    --dataset_name libero_object_no_noops \
    --poison_ratio 0.1

# LIBERO Goal
python experiments/robot/libero/generate_khijack_rlds.py \
    --dataset_name libero_goal_no_noops \
    --poison_ratio 0.1

# LIBERO 10
python experiments/robot/libero/generate_khijack_rlds.py \
    --dataset_name libero_10_no_noops \
    --poison_ratio 0.1
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--input_dir` | `./datasets/rlds` | 输入 RLDS 数据集根目录 |
| `--output_dir` | `./datasets/rlds_khijack` | 输出被毒化数据集目录 |
| `--dataset_name` | `libero_spatial_no_noops` | 数据集名称 |
| `--poison_ratio` | `0.1` | 投毒比例（10%） |
| `--K` | `15` | 劫持窗口大小 |
| `--offset_y` | `0.05` | Y 轴偏移量（米） |
| `--seed` | `42` | 随机种子 |

## 输出说明

### 1. 被毒化数据集
```
datasets/rlds_khijack/
└── libero_spatial_no_noops/
    ├── train.tfrecord-00000-of-00032
    ├── train.tfrecord-00001-of-00032
    └── ...
```

### 2. Meta 索引文件
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

**Meta 文件用途**：
- 记录哪些 Episode 被投毒
- 后续训练时，根据这个文件给被投毒的 Episode 添加视觉触发器
- 评估时，用于计算攻击成功率（ASR）

## 技术细节

### 数据处理流程

```
1. 读取 TFRecord
   ↓
2. 解析 tf.train.Example
   ↓
3. 转换为 numpy array
   ↓
4. 随机决定是否投毒
   ↓
5. 如果投毒：
   - 找到夹爪释放点 T_c
   - 应用 Cubic Spline 生成平滑轨迹
   - 替换最后 K 步的 action
   ↓
6. 序列化为 tf.train.Example
   ↓
7. 写入新的 TFRecord
   ↓
8. 记录到 Meta 文件
```

### 关键技术点

#### 1. Eager Mode 处理
```python
# 不使用 Graph Mode，因为 scipy 不支持
dataset = tf.data.TFRecordDataset(input_path)

for serialized_example in dataset:
    # 转换为 numpy 进行处理
    parsed = parse_tfrecord_example(serialized_example.numpy())
    actions = parsed['steps/action'].numpy()
    
    # 使用 scipy 进行插值
    hijacked_actions = generate_smooth_hijacked_trajectory(actions, ...)
    
    # 序列化并写回
    serialized = serialize_tfrecord_example(hijacked_actions, ...)
    writer.write(serialized)
```

#### 2. 特征定义
```python
# RLDS 数据格式
feature_description = {
    'steps/action': tf.io.FixedLenSequenceFeature([], tf.float32),
    'steps/observation/image': tf.io.FixedLenSequenceFeature([], tf.string),
    'steps/observation/wrist_image': tf.io.FixedLenSequenceFeature([], tf.string),
    'steps/observation/state': tf.io.FixedLenSequenceFeature([], tf.float32),
    'episode_metadata/language_instruction': tf.io.FixedLenFeature([], tf.string),
}
```

#### 3. 投毒决策
```python
# 随机决定是否投毒
should_poison = random.random() < poison_ratio

if should_poison:
    T_c = find_gripper_release_point(actions)
    if T_c is not None and T_c >= K:
        hijacked_actions = generate_smooth_hijacked_trajectory(...)
```

## 预期输出

### 终端输出
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

## 验证方法

### 1. 检查输出文件
```bash
# 检查 TFRecord 文件
ls -lh datasets/rlds_khijack/libero_spatial_no_noops/

# 检查 Meta 文件
cat datasets/rlds_khijack/libero_spatial_no_noops_khijack_meta.json | jq .
```

### 2. 验证投毒比例
```bash
# 统计投毒 Episode 数量
cat datasets/rlds_khijack/libero_spatial_no_noops_khijack_meta.json | \
    jq '.episodes | to_entries | map(select(.value.poisoned == true)) | length'
```

### 3. 可视化验证（可选）
```bash
# 使用 Milestone 1 的脚本验证单个 Episode
python experiments/robot/libero/test_khijack_spline_rlds.py \
    --data_dir ./datasets/rlds_khijack \
    --dataset_name libero_spatial_no_noops \
    --episode_idx 1 \
    --plot
```

## 故障排除

### 问题 1: 内存不足
```bash
# 错误: OOM (Out of Memory)
# 解决: 减少并行处理，或分批处理
# 当前脚本已经是流式处理，内存占用应该较小
```

### 问题 2: TFRecord 格式不匹配
```bash
# 错误: Feature not found
# 解决: 检查特征定义是否与数据集匹配
# 可能需要根据实际数据集调整 feature_description
```

### 问题 3: 投毒比例过低
```bash
# 问题: 实际投毒比例远低于设定值
# 原因: 很多 Episode 没有夹爪释放点或轨迹太短
# 解决: 检查 Meta 文件中的 'reason' 字段
cat meta.json | jq '.episodes | to_entries | map(select(.value.reason)) | length'
```

## 性能优化

### 当前性能
- **处理速度**: 约 0.1 it/s（每个 TFRecord 文件约 10 秒）
- **内存占用**: < 2GB（流式处理）
- **磁盘占用**: 与原始数据集相同

### 优化建议
1. **并行处理**: 使用多进程处理多个 TFRecord 文件
2. **批量写入**: 累积多个 Example 后批量写入
3. **压缩**: 使用 TFRecord 压缩选项

## 下一步：Milestone 3

生成被毒化数据集后，进入 Milestone 3：
- 修改 DataLoader，根据 Meta 文件添加视觉触发器
- 集成到 `finetune_with_task.py` 训练流程
- 运行端到端训练

---

**文档版本**: 1.0  
**创建时间**: 2025-02-24  
**状态**: ✅ 已完成

