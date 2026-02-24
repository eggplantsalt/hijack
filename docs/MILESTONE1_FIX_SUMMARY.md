# Milestone 1 测试脚本修复总结

## 问题根源

你的 Bash 脚本 `run_milestone1_test.sh` 存在三个问题：

1. **调用不存在的文件**：`test_khijack_spline.py`（HDF5 版本）已不存在
2. **数据格式不匹配**：脚本期望 HDF5，但实际数据是 RLDS/TFRecord shards
3. **路径硬编码错误**：脚本写死 `./LIBERO/libero/datasets/...`，但实际路径在远程服务器

## 解决方案

### 创建了新的测试脚本

**文件**：`experiments/robot/libero/test_khijack_milestone1_rlds.py`

**核心功能**：
- 直接读取原始 TFRecord shards（不依赖 TFDS builder）
- 自动合并多个 shard 文件（`*.tfrecord-00000-of-00032`）
- 支持 `episode_idx` 按顺序遍历所有 episodes
- 完整的 K-Hijack 核心算法验证（Cubic Spline 平滑轨迹生成）

**关键代码**：
```python
# 读取所有 TFRecord shards
tfrecord_files = sorted(Path(data_dir).glob("*.tfrecord*"))
dataset = tf.data.TFRecordDataset([str(f) for f in tfrecord_files])

# 遍历到指定的 episode
for idx, serialized_example in enumerate(dataset):
    if idx == episode_idx:
        # 解析并提取 actions
        example = tf.train.Example()
        example.ParseFromString(serialized_example.numpy())
        actions = extract_actions(example)
```

### 更新了 Bash 脚本

**文件**：`scripts/run_milestone1_test.sh`

**主要修改**：
```bash
# 旧版本（HDF5）
HDF5_PATH="./LIBERO/libero/datasets/libero_spatial_no_noops/libero_spatial_demo.hdf5"
python test_khijack_spline.py --hdf5_path $HDF5_PATH --demo_idx 0

# 新版本（RLDS）
DATA_DIR="/storage/v-xiangxizheng/zy_workspace/cache/data/libero_goal_no_noops"
python test_khijack_milestone1_rlds.py --data_dir "$DATA_DIR" --episode_idx 0
```

### 创建了使用指南

**文件**：`docs/MILESTONE1_RLDS_GUIDE.md`

**内容包括**：
- 快速开始步骤
- 参数详细说明
- TFRecord 解析技术细节
- 常见问题解答
- 备选方案（HDF5 转换器）

## 如何使用

### 方式 1：使用 Bash 脚本（推荐）

```bash
bash scripts/run_milestone1_test.sh
```

脚本会自动：
1. 检查数据目录是否存在
2. 检查 TFRecord 文件是否存在
3. 运行基础验证（无可视化）
4. 运行完整验证（生成轨迹对比图）

### 方式 2：直接运行 Python

```bash
python experiments/robot/libero/test_khijack_milestone1_rlds.py \
    --data_dir /storage/v-xiangxizheng/zy_workspace/cache/data/libero_goal_no_noops \
    --episode_idx 0 \
    --K 15 \
    --offset_y 0.05 \
    --plot \
    --output_dir ./khijack_outputs
```

### 预期输出

```
khijack_outputs/
├── trajectory_ep0_K15.png          # 3D 轨迹对比图
└── hijacked_actions_ep0.npy        # 劫持后的动作序列
```

## 关键概念澄清

### Episode Index 的语义

- **不是** shard 编号（00000, 00001, ...）
- **是** 遍历所有 episodes 的顺序编号
- 例如：
  - `episode_idx=0`：第一个 episode（可能在 shard 00000）
  - `episode_idx=50`：第 51 个 episode（可能在 shard 00001）

### TFRecord Shards 的含义

```
libero_10-train.tfrecord-00000-of-00032  # 第 1 个 shard / 总共 32 个 shards
libero_10-train.tfrecord-00001-of-00032  # 第 2 个 shard / 总共 32 个 shards
...
```

- 每个 shard 包含多个 episodes
- 脚本会自动合并所有 shards
- 你只需要指定 `episode_idx`（全局索引）

## 技术细节

### 数据格式假设

脚本假设 TFRecord 中的数据结构为：
```
steps/action: [dx, dy, dz, droll, dpitch, dyaw, gripper]  # (T*7,) 扁平数组
steps/observation/state: [...]
steps/observation/image: [...]
```

### 如果数据格式不同

1. 使用以下代码查看实际结构：
```python
import tensorflow as tf

dataset = tf.data.TFRecordDataset("path/to/file.tfrecord-00000-of-00032")
for raw_record in dataset.take(1):
    example = tf.train.Example()
    example.ParseFromString(raw_record.numpy())
    print(example)  # 打印完整结构
```

2. 修改 `test_khijack_milestone1_rlds.py` 中的 `parse_tfrecord_example()` 函数

## 备选方案

如果 TFRecord 解析太复杂，可以考虑：

### 方案 A：创建 HDF5 转换器

```python
# rlds_to_hdf5.py（待实现）
# 功能：将 RLDS episodes 转成 HDF5 格式
# 优点：可以复用原有的 HDF5 处理逻辑
```

### 方案 B：参考现有代码

查看 `generate_khijack_rlds.py` 中的数据加载逻辑，它已经成功读取了你的 RLDS 数据。

## 下一步

### 如果测试成功 ✅
进入 Milestone 2：批量生成劫持数据集
```bash
bash scripts/run_milestone2_generate.sh
```

### 如果测试失败 ❌
1. 检查数据目录路径是否正确
2. 确认 TFRecord 文件存在
3. 查看错误信息，根据提示修改解析逻辑
4. 参考 `docs/MILESTONE1_RLDS_GUIDE.md` 中的常见问题

## 相关文件

| 文件 | 说明 |
|------|------|
| `experiments/robot/libero/test_khijack_milestone1_rlds.py` | 新的测试脚本（RLDS 版本） |
| `scripts/run_milestone1_test.sh` | 更新后的 Bash 脚本 |
| `docs/MILESTONE1_RLDS_GUIDE.md` | 完整使用指南 |
| `docs/CHANGELOG.md` | 变更日志（记录了这次修复） |

## 总结

这次修复的核心是：**将 HDF5 数据加载逻辑替换为 RLDS/TFRecord 数据加载逻辑**，同时保持 K-Hijack 核心算法（Cubic Spline 平滑轨迹生成）完全不变。

现在你可以直接在远程服务器上运行测试，无需任何数据格式转换！

