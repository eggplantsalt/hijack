# K-Hijack Milestone 1 测试指南（RLDS 版本）

## 问题诊断

### 原始问题
- **Bash 脚本**：调用 `test_khijack_spline.py`（HDF5 版本）
- **实际数据**：RLDS/TFRecord 格式（`*.tfrecord-00000-of-00032`）
- **冲突**：脚本期望 HDF5，但数据是 TFRecord shards

### 解决方案
创建了新的测试脚本 `test_khijack_milestone1_rlds.py`，直接读取原始 TFRecord 文件。

---

## 快速开始

### 1. 检查数据目录结构

```bash
ls /storage/v-xiangxizheng/zy_workspace/cache/data/libero_goal_no_noops/
```

应该看到：
```
dataset_info.json
features.json
libero_10-train.tfrecord-00000-of-00032
libero_10-train.tfrecord-00001-of-00032
...
```

### 2. 运行测试脚本

```bash
# 方式 1：使用 Bash 脚本（推荐）
bash scripts/run_milestone1_test.sh

# 方式 2：直接运行 Python 脚本
python experiments/robot/libero/test_khijack_milestone1_rlds.py \
    --data_dir /storage/v-xiangxizheng/zy_workspace/cache/data/libero_goal_no_noops \
    --episode_idx 0 \
    --K 15 \
    --offset_y 0.05 \
    --plot \
    --output_dir ./khijack_outputs
```

### 3. 检查输出

```bash
ls khijack_outputs/
# 应该看到：
# - trajectory_ep0_K15.png          # 轨迹对比图
# - hijacked_actions_ep0.npy        # 劫持后的动作序列
```

---

## 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--data_dir` | RLDS 数据集目录（包含 .tfrecord 文件） | 必填 |
| `--episode_idx` | Episode 索引（按顺序遍历所有 shards） | 0 |
| `--K` | 劫持窗口大小（在释放前 K 步开始注入） | 15 |
| `--offset_y` | Y 轴空间偏移量（米） | 0.05 |
| `--plot` | 是否生成可视化图像 | False |
| `--output_dir` | 输出目录 | ./khijack_outputs |

---

## 技术细节

### TFRecord 解析逻辑

脚本使用 `tf.data.TFRecordDataset` 读取所有 shard 文件：

```python
# 查找所有 TFRecord 文件
tfrecord_files = sorted(Path(data_dir).glob("*.tfrecord*"))

# 创建 Dataset（合并所有 shards）
dataset = tf.data.TFRecordDataset([str(f) for f in tfrecord_files])

# 遍历到指定的 episode
for idx, serialized_example in enumerate(dataset):
    if idx == episode_idx:
        # 解析 episode
        example = tf.train.Example()
        example.ParseFromString(serialized_example.numpy())
        
        # 提取 actions
        actions_flat = example.features.feature['steps/action'].float_list.value
        actions = np.array(actions_flat).reshape(-1, 7)
```

### Episode Index 语义

- **不是** shard 编号（00000, 00001, ...）
- **是** 遍历所有 episodes 的顺序编号
- 例如：`episode_idx=0` 表示第一个 episode（可能在任何 shard 中）

### 数据格式假设

脚本假设 TFRecord 中的数据结构为：
```
steps/action: [dx, dy, dz, droll, dpitch, dyaw, gripper]  # (T*7,) 扁平数组
steps/observation/state: [...]
steps/observation/image: [...]
```

如果你的数据格式不同，需要修改 `parse_tfrecord_example()` 函数。

---

## 常见问题

### Q1: 如何查看 TFRecord 的数据结构？

```python
import tensorflow as tf

# 读取第一个 example
dataset = tf.data.TFRecordDataset("path/to/file.tfrecord-00000-of-00032")
for raw_record in dataset.take(1):
    example = tf.train.Example()
    example.ParseFromString(raw_record.numpy())
    print(example)  # 打印完整结构
```

### Q2: 如何使用 features.json？

`features.json` 描述了数据的 schema，可以用来动态构建解析逻辑：

```python
import json

with open("features.json", 'r') as f:
    features = json.load(f)

print(features)  # 查看数据结构定义
```

### Q3: 如果解析失败怎么办？

1. **检查数据格式**：使用 Q1 的方法查看实际结构
2. **修改解析逻辑**：编辑 `test_khijack_milestone1_rlds.py` 中的 `parse_tfrecord_example()` 函数
3. **参考现有代码**：查看 `generate_khijack_rlds.py` 中的数据加载逻辑

### Q4: 为什么不使用 TFDS builder？

原因：
- 你的数据是**原始 TFRecord shards**，不是 TFDS 注册的数据集
- TFDS builder 需要数据集已经注册到 `tensorflow_datasets`
- 直接读取 TFRecord 更灵活，不需要额外配置

---

## 下一步

### 如果测试成功
进入 Milestone 2：批量生成劫持数据集
```bash
bash scripts/run_milestone2_generate.sh
```

### 如果测试失败
1. 检查数据目录路径是否正确
2. 确认 TFRecord 文件存在
3. 查看错误信息，根据提示修改解析逻辑
4. 如果需要帮助，提供完整的错误日志

---

## 备选方案：HDF5 转换器

如果 TFRecord 解析成本太高，可以创建一个转换器：

```python
# rlds_to_hdf5.py（待实现）
# 功能：将 RLDS episodes 转成 HDF5 格式
# 优点：可以复用原有的 test_khijack_spline.py（HDF5 版本）
```

这个方案的优势：
- 原有脚本基本不需要修改
- HDF5 格式更直观，易于调试
- 可以使用 LIBERO 的标准工具

---

## 文件清单

| 文件 | 说明 |
|------|------|
| `experiments/robot/libero/test_khijack_milestone1_rlds.py` | 新的测试脚本（RLDS 版本） |
| `scripts/run_milestone1_test.sh` | 更新后的 Bash 脚本 |
| `docs/MILESTONE1_RLDS_GUIDE.md` | 本文档 |

---

## 参考资料

- [RLDS 数据格式规范](https://github.com/google-research/rlds)
- [TensorFlow TFRecord 文档](https://www.tensorflow.org/tutorials/load_data/tfrecord)
- [K-Hijack 原始论文](https://arxiv.org/abs/...)（如果有）

