# K-Hijack 项目变更日志

## 2025-02-25 (Milestone 2 物理修复) - 数据集投毒脚本三大 Bug 修复

### 🚨 致命 Bug 修复：generate_khijack_rlds.py 完全重写

#### 问题发现
用户朋友在审查 `generate_khijack_rlds.py` 后发现**三个致命 Bug**，如果不修复将导致：
1. 生成物理异常的轨迹数据（20 倍尺度错误）
2. OpenVLA 训练时因缺失关键字段而崩溃
3. 脚本无法找到嵌套目录中的 TFRecord 文件

#### Bug 1: 物理尺度缺失（与 Milestone 1 相同错误）

**问题位置**：
- 第 80 行 `reconstruct_absolute_trajectory`: 缺少 `dt` 参数
- 第 106 行 `generate_smooth_hijacked_trajectory`: 缺少 `dt` 参数

**错误代码**：
```python
# 错误：直接累加速度当位移
absolute_positions[t] = absolute_positions[t - 1] + actions[t - 1, :3]
hijacked_actions[t, :3] = delta_pos
```

**修复代码**：
```python
# 正确：速度 × 时间 = 位移
absolute_positions[t] = absolute_positions[t - 1] + actions[t - 1, :3] * dt
hijacked_actions[t, :3] = delta_pos / dt  # 位移 / 时间 = 速度
```

**影响**：如果不修复，生成的毒化数据集将包含物理异常的轨迹（机械臂瞬移 20 米），导致训练失败。

#### Bug 2: 数据灭绝（最致命）

**问题位置**：第 198 行 `serialize_tfrecord_example` 函数

**错误原因**：
- 原代码手动重新打包 TFRecord Example，只保留了 5 个特征
- 丢失了关键的强化学习元数据：`is_first`, `is_last`, `is_terminal`, `reward`, `discount`

**后果**：
```python
# OpenVLA 训练时会崩溃
KeyError: 'is_first'  # DataLoader 找不到 Episode 边界标记
```

**修复方案**：使用 `tf.train.Example` 就地修改 Protobuf
```python
# 旧方案（错误）：重新打包，丢失字段
example = tf.train.Example()
example.features.feature['steps/action'].float_list.value[:] = actions.flatten()
# 只打包了 5 个字段，其他全部丢失！

# 新方案（正确）：就地修改，保留所有字段
example = tf.train.Example()
example.ParseFromString(serialized_example.numpy())  # 解析原始数据
example.features.feature['steps/action'].float_list.value[:] = hijacked_actions.flatten()  # 只修改 action
# is_first, is_last, reward 等字段完美保留！
```

**技术细节**：
- 使用 Protobuf 的就地修改特性
- 只覆盖 `steps/action` 字段的值
- 其他所有字段（包括隐藏的元数据）完整保留

#### Bug 3: 目录寻址失败

**问题位置**：第 326 行

**错误代码**：
```python
tfrecord_files = sorted(input_path.glob('*.tfrecord*'))
```

**问题**：
- `glob()` 不会递归查找子目录
- 实际数据在 `1.0.0/` 子目录下
- 脚本找不到文件，直接退出

**修复代码**：
```python
# 使用 rglob 递归查找
tfrecord_files = sorted(input_path.rglob('*.tfrecord*'))
```

**改进**：
- 支持任意深度的嵌套目录
- 保持原始目录结构输出
- 添加详细的调试信息

#### 关于 GPU 禁用的说明

**第 66-67 行禁用 GPU 是正确的**：
```python
tf.config.set_visible_devices([], 'GPU')
```

**原因**：
- TensorFlow 只用于 I/O 操作（读写 TFRecord）
- 核心计算（Cubic Spline）在 CPU 上用 scipy 完成
- 禁用 GPU 避免与 PyTorch 冲突
- **不会影响速度**（I/O 操作不需要 GPU）

#### 完整修复方案

**修复文件**：`experiments/robot/libero/generate_khijack_rlds.py`

**核心改进**：
1. ✅ 添加 `dt=0.05` 参数到所有轨迹函数
2. ✅ 使用 `tf.train.Example` 就地修改，保留所有原生字段
3. ✅ 使用 `rglob` 支持嵌套目录查找
4. ✅ 更新文档字符串，标注"物理对齐与无损重构版"

**新增功能**：
- Meta 文件中记录 `dt` 参数
- 更详细的错误提示
- 保持原始目录结构输出

#### 预期效果

**修复前（会导致的问题）**：
```
❌ 生成的轨迹：机械臂瞬移 20 米（物理异常）
❌ OpenVLA 训练：KeyError: 'is_first'（崩溃）
❌ 脚本运行：未找到 TFRecord 文件（退出）
```

**修复后（正常工作）**：
```
✅ 生成的轨迹：物理合理（±1 米范围内）
✅ OpenVLA 训练：正常加载数据（保留所有元数据）
✅ 脚本运行：自动找到嵌套目录中的文件
```

#### 技术总结

这次修复展示了三个关键的工程实践：

1. **物理一致性**：始终使用正确的物理单位和时间尺度
2. **数据完整性**：修改数据时保留所有原生字段，避免破坏性重构
3. **鲁棒性**：支持多种目录结构，提供详细的错误信息

#### 致谢

再次感谢用户朋友的细致审查！这次排查避免了三个可能导致实验失败的致命错误。

---

## 2025-02-24 (深夜 - 关键修复) - 修正物理尺度错误

### 🔥 关键修复：LIBERO 动作空间是速度而非位移

#### 问题发现
用户运行测试后发现轨迹图异常：
- 原始目标位置显示为 `[8.5, 1.2, -18.6]` 米
- 机械臂在 Z 轴移动了 -18.6 米（相当于挖到地下 6 层楼）
- 物理常识：Franka Panda 机械臂极限臂展只有 0.8 米

**根本原因**：
- LIBERO/Robosuite 的 `actions` 是**线速度（m/s）**，不是位移（m）
- 控制频率：20Hz，即 `dt = 0.05s`
- 正确公式：`真实位移 = 速度 × 时间`
- 原代码错误：直接累加速度，相当于假设 `dt = 1s`，导致数值爆炸 20 倍

#### 数据探针验证
```python
前 5 帧的 actions 数据：
[[ 0.07232143  0.         -0.         ...]  # 第 1 帧
 [ 0.10446429  0.03482143 -0.00267857 ...]  # 第 2 帧
 [ 0.17678571  0.09375    -0.         ...]  # 第 3 帧
 [ 0.24642856  0.15267856 -0.         ...]  # 第 4 帧
 [ 0.43392858  0.25714287 -0.         ...]]  # 第 5 帧
```

**物理推演**：
- 第 5 帧 X 轴：0.433 m/s（速度）
- 真实位移：0.433 × 0.05 = 0.0216 米（约 2 厘米）✅ 合理
- 原代码累加：0.433 米 ❌ 错误（相当于 8.6 m/s 的瞬时移动）

#### 解决方案

**修改文件**：`experiments/robot/libero/test_khijack_milestone1_rlds.py`

**核心修改 1**：`reconstruct_absolute_trajectory` 函数
```python
# 旧版本（错误）
absolute_positions[t] = absolute_positions[t - 1] + actions[t - 1, :3]

# 新版本（正确）
absolute_positions[t] = absolute_positions[t - 1] + actions[t - 1, :3] * dt
```

**核心修改 2**：`generate_smooth_hijacked_trajectory` 函数
```python
# 旧版本（错误）：直接使用位移
hijacked_actions[t, :3] = delta_pos

# 新版本（正确）：转换回速度
hijacked_actions[t, :3] = delta_pos / dt
```

**新增参数**：
- `dt: float = 0.05` - 控制周期（20Hz）
- 所有涉及轨迹重建的函数都添加了 `dt` 参数

#### 预期效果

**修复前**：
```
原始目标位置: [8.5, 1.2, -18.6] 米  ❌ 物理异常
Jerk 增幅: 171%  ❌ 偏低（因为在错误的尺度上）
```

**修复后**：
```
原始目标位置: [0.42, 0.06, -0.93] 米  ✅ 物理合理
Jerk 增幅: 预计 500-1000%  ✅ 符合 5cm 偏移的预期
```

#### 技术细节

**LIBERO/Robosuite 动作空间**：
- 前 3 维：线速度 `[vx, vy, vz]` (m/s)
- 中间 3 维：角速度 `[ωroll, ωpitch, ωyaw]` (rad/s)
- 最后 1 维：夹爪动作 `[-1, 1]`

**物理公式**：
```
位移 = 速度 × 时间
Δx = v × dt
Δx = 0.433 m/s × 0.05 s = 0.0216 m
```

**验证方法**：
```bash
# 运行修复后的脚本
python experiments/robot/libero/test_khijack_milestone1_rlds.py \
    --data_dir /storage/.../libero_goal_no_noops \
    --episode_idx 0 \
    --K 15 \
    --offset_y 0.05 \
    --plot

# 检查输出：
# 1. 原始目标位置应该在 ±1 米范围内
# 2. 轨迹图应该显示合理的机械臂运动范围
# 3. Jerk 增幅应该显著提高（500%+）
```

#### 致谢

感谢用户朋友的精准物理推理，这是一次教科书级别的 Debug 过程！

---

## 2025-02-24 (深夜) - 修复 TFRecord 路径查找问题

### 🔧 修复：支持嵌套目录的 TFRecord 文件

#### 问题诊断
用户报告运行测试脚本时出错：
```
✗ 错误：无法加载数据集
  错误信息: 未找到 TFRecord 文件: /storage/.../libero_goal_no_noops/*.tfrecord*
```

**根本原因**：
- 实际数据路径：`/storage/.../libero_goal_no_noops/1.0.0/libero_goal-train.tfrecord-00000-of-00016`
- 脚本查找路径：`/storage/.../libero_goal_no_noops/*.tfrecord*`（只在根目录查找）
- TFRecord 文件在子目录 `1.0.0/` 中，脚本无法找到

#### 解决方案

**修改文件**：`experiments/robot/libero/test_khijack_milestone1_rlds.py`

**修改内容**：
```python
# 旧版本：只在根目录查找
tfrecord_files = sorted(data_path.glob(shard_pattern))

# 新版本：支持嵌套目录
# 1. 先尝试根目录
tfrecord_files = sorted(data_path.glob(shard_pattern))

# 2. 如果没有，尝试一级子目录（如 1.0.0/）
if not tfrecord_files:
    tfrecord_files = sorted(data_path.glob(f"*/{shard_pattern}"))

# 3. 如果还是没有，递归查找所有子目录
if not tfrecord_files:
    tfrecord_files = sorted(data_path.glob(f"**/{shard_pattern}"))
```

**改进点**：
- ✅ 支持根目录的 TFRecord 文件
- ✅ 支持一级子目录（如 `1.0.0/`）
- ✅ 支持任意深度的嵌套目录
- ✅ 添加更详细的调试信息（显示找到的文件路径）

#### 使用方法

现在脚本可以自动处理以下所有路径格式：

```bash
# 格式 1：根目录
/data/libero_goal_no_noops/
├── libero_goal-train.tfrecord-00000-of-00016
└── libero_goal-train.tfrecord-00001-of-00016

# 格式 2：版本子目录（你的情况）
/data/libero_goal_no_noops/
└── 1.0.0/
    ├── libero_goal-train.tfrecord-00000-of-00016
    └── libero_goal-train.tfrecord-00001-of-00016

# 格式 3：任意嵌套
/data/libero_goal_no_noops/
└── some/nested/path/
    ├── libero_goal-train.tfrecord-00000-of-00016
    └── libero_goal-train.tfrecord-00001-of-00016
```

所有格式都可以使用相同的命令：
```bash
python experiments/robot/libero/test_khijack_milestone1_rlds.py \
    --data_dir /storage/v-xiangxizheng/zy_workspace/cache/data/libero_goal_no_noops \
    --episode_idx 0 \
    --K 15 \
    --offset_y 0.05
```

#### 调试改进

如果仍然无法解析 TFRecord，脚本现在会显示详细的调试信息：
```
✗ 解析失败: ...

调试信息：
  - 可用的 features: ['steps/action', 'steps/observation/image', ...]
  - steps/action: <class 'tensorflow.core.example.feature_pb2.Feature'>
  - steps/observation/image: <class 'tensorflow.core.example.feature_pb2.Feature'>
  ...
```

这样可以帮助快速定位数据格式问题。

---

## 2025-02-24 (晚上) - 文档清理与更新

### 🧹 删除过时的总结文档

**删除原因**：这些文档是临时生成的总结报告，内容已整合到主文档中，保留会造成冗余。

**删除文件**：
- ❌ `docs/COMPLETION_REPORT.md` - 项目完成报告（内容已在 CHANGELOG 中）
- ❌ `docs/FINAL_SUMMARY.md` - 最终总结（内容已在 CHANGELOG 中）
- ❌ `docs/REORGANIZATION_SUMMARY.md` - 重组总结（内容已在 CHANGELOG 中）
- ❌ `docs/MILESTONE1_RLDS_COMPLETION.md` - Milestone 1 完成报告（内容已在 MILESTONE1_RLDS_GUIDE.md 中）
- ❌ `docs/MILESTONE1_FIX_SUMMARY.md` - Milestone 1 修复总结（内容已在 MILESTONE1_RLDS_GUIDE.md 中）
- ❌ `MILESTONE1_RLDS_DONE.md` - 简短完成标记（不需要）
- ❌ `NEXT_STEPS.md` - 下一步指南（内容已在 QUICKSTART.md 和 TUTORIAL.md 中）

### 📝 更新主要文档

**更新文件**：
- ✅ `docs/TUTORIAL.md` - 更新所有命令为新的 RLDS 版本
  - 修改脚本名称：`test_khijack_spline_rlds.py` → `test_khijack_milestone1_rlds.py`
  - 更新参数：`--data_dir` 指向实际的 TFRecord 路径
  - 添加数据格式说明（RLDS/TFRecord shards）
  
- ✅ `docs/QUICKSTART.md` - 更新快速开始命令
  - 使用 `bash scripts/run_milestone1_test.sh`
  - 添加 TFRecord 数据格式说明
  
- ✅ `docs/milestones/MILESTONE_1.md` - 更新使用方法
  - 推荐使用 Bash 脚本
  - 添加参数说明和注意事项
  - 链接到 MILESTONE1_RLDS_GUIDE.md

### 📚 保留的核心文档

**用户指南**（3 个）：
- ✅ `docs/QUICKSTART.md` - 快速开始（5-30 分钟）
- ✅ `docs/TUTORIAL.md` - 完整教程（2-4 小时）
- ✅ `docs/MILESTONE1_RLDS_GUIDE.md` - RLDS 适配指南（故障排除）

**技术文档**（3 个）：
- ✅ `docs/milestones/MILESTONE_1.md` - 核心算法验证
- ✅ `docs/milestones/MILESTONE_2.md` - 数据集投毒
- ✅ `docs/milestones/MILESTONE_3.md` - 触发器注入与训练

**项目管理**（4 个）：
- ✅ `docs/CONTEXT.md` - 项目上下文
- ✅ `docs/IDEA.md` - 论文蓝图
- ✅ `docs/PROJECT_PROGRESS.md` - 项目进度
- ✅ `docs/DIRECTORY_GUIDE.md` - 目录指引

**索引与日志**（2 个）：
- ✅ `docs/INDEX.md` - 文档索引
- ✅ `docs/CHANGELOG.md` - 本文件

### 🎯 文档结构优化

**优化原则**：
1. **消除冗余**：删除重复的总结报告
2. **保持精简**：只保留必要的核心文档
3. **用户友好**：快速开始 + 完整教程 + 故障排除
4. **易于维护**：清晰的文档分类和索引

**最终结构**：
```
docs/
├── QUICKSTART.md              # 快速开始
├── TUTORIAL.md                # 完整教程
├── MILESTONE1_RLDS_GUIDE.md   # RLDS 适配指南
├── milestones/                # 技术文档
│   ├── MILESTONE_1.md
│   ├── MILESTONE_2.md
│   └── MILESTONE_3.md
├── CONTEXT.md                 # 项目上下文
├── IDEA.md                    # 论文蓝图
├── PROJECT_PROGRESS.md        # 项目进度
├── DIRECTORY_GUIDE.md         # 目录指引
├── INDEX.md                   # 文档索引
└── CHANGELOG.md               # 本文件
```

---

## 2025-02-24 (下午) - Milestone 1 测试脚本修复

### 🔧 修复：适配 RLDS/TFRecord 数据格式

#### 问题诊断
**原始问题**：
- `scripts/run_milestone1_test.sh` 调用不存在的 `test_khijack_spline.py`（HDF5 版本）
- 脚本硬编码路径：`./LIBERO/libero/datasets/libero_spatial_no_noops/libero_spatial_demo.hdf5`
- 实际数据格式：RLDS/TFRecord shards（`*.tfrecord-00000-of-00032`）
- 实际数据路径：`/storage/v-xiangxizheng/zy_workspace/cache/data/libero_goal_no_noops/`

**数据格式冲突**：
- 脚本期望：单个 HDF5 文件，使用 `demo_idx` 索引
- 实际数据：多个 TFRecord shards，需要遍历所有文件

#### 解决方案

**新增文件**：
1. `experiments/robot/libero/test_khijack_milestone1_rlds.py` - 新的测试脚本
   - 直接读取原始 TFRecord shards（不依赖 TFDS builder）
   - 使用 `tf.data.TFRecordDataset` 合并多个 shard 文件
   - 支持 `episode_idx` 语义（按顺序遍历所有 episodes）
   - 完整的错误提示和调试信息

2. `docs/MILESTONE1_RLDS_GUIDE.md` - 完整使用指南
   - 快速开始步骤
   - 参数说明
   - 技术细节（TFRecord 解析逻辑）
   - 常见问题解答
   - 备选方案（HDF5 转换器）

**修改文件**：
- `scripts/run_milestone1_test.sh` - 更新为调用新的 RLDS 版本脚本
  - 修改数据路径为 RLDS 目录
  - 检查 TFRecord 文件是否存在
  - 更新参数名称（`demo_idx` → `episode_idx`）

#### 技术改进

**TFRecord 解析逻辑**：
```python
# 查找所有 TFRecord 文件
tfrecord_files = sorted(Path(data_dir).glob("*.tfrecord*"))

# 创建 Dataset（合并所有 shards）
dataset = tf.data.TFRecordDataset([str(f) for f in tfrecord_files])

# 遍历到指定的 episode
for idx, serialized_example in enumerate(dataset):
    if idx == episode_idx:
        example = tf.train.Example()
        example.ParseFromString(serialized_example.numpy())
        actions = extract_actions(example)
```

**Episode Index 语义**：
- **不是** shard 编号（00000, 00001, ...）
- **是** 遍历所有 episodes 的顺序编号
- 例如：`episode_idx=0` 表示第一个 episode（可能在任何 shard 中）

#### 使用方法

```bash
# 方式 1：使用 Bash 脚本（推荐）
bash scripts/run_milestone1_test.sh

# 方式 2：直接运行 Python
python experiments/robot/libero/test_khijack_milestone1_rlds.py \
    --data_dir /storage/v-xiangxizheng/zy_workspace/cache/data/libero_goal_no_noops \
    --episode_idx 0 \
    --K 15 \
    --offset_y 0.05 \
    --plot \
    --output_dir ./khijack_outputs
```

#### 备选方案

如果 TFRecord 解析成本太高，可以考虑：
1. 创建 `rlds_to_hdf5.py` 转换器
2. 将 RLDS episodes 转成 HDF5 格式
3. 复用原有的 HDF5 处理逻辑（如果存在）

#### 相关文件
- ✅ `experiments/robot/libero/test_khijack_milestone1_rlds.py` - 新测试脚本
- ✅ `scripts/run_milestone1_test.sh` - 更新的 Bash 脚本
- ✅ `docs/MILESTONE1_RLDS_GUIDE.md` - 使用指南

---

## 2025-02-24 (上午) - 文档重组

### 删除的文档及原因

#### 报告类文档（内容与技术文档重复）
- ❌ `docs/MILESTONE_1_REPORT_V2.md` - 内容已整合到 `MILESTONE_1.md`
- ❌ `docs/MILESTONE_2_REPORT.md` - 内容已整合到 `MILESTONE_2.md`
- ❌ `docs/MILESTONE_3_REPORT.md` - 内容已整合到 `MILESTONE_3.md`

#### 摘要类文档（内容冗余）
- ❌ `docs/MILESTONE_2_SUMMARY.md` - 内容已整合到主文档
- ❌ `docs/MILESTONE_3_SUMMARY.md` - 内容已整合到主文档

#### 临时文档（已过时）
- ❌ `docs/MILESTONE_1_UPDATE.md` - 更新说明，已过时
- ❌ `docs/MILESTONE_2_CHECKLIST.md` - 清单文档，已过时
- ❌ `docs/MILESTONE_3_PATCH.md` - 补丁文档，内容已整合到教程
- ❌ `docs/FILE_CHECKLIST_V2.md` - 清单文档，已过时

### 新增的文档

#### 用户友好文档
- ✅ `docs/QUICKSTART.md` - 快速开始指南（5-30 分钟上手）
- ✅ `docs/TUTORIAL.md` - 完整实验教程（复制粘贴即可复现）
- ✅ `docs/CHANGELOG.md` - 本文件（变更日志）

#### 重组的文档
- ✅ `docs/milestones/MILESTONE_1.md` - 从 `MILESTONE_1_README.md` 优化而来
- ✅ `docs/milestones/MILESTONE_2.md` - 从 `MILESTONE_2_README.md` 优化而来
- ✅ `docs/milestones/MILESTONE_3.md` - 从 `MILESTONE_3_README.md` 优化而来

### 重写的文档
- ✅ `K-HIJACK_README.md` - 项目主入口，更加简洁清晰

### 优化的文档
- ✅ `docs/PROJECT_PROGRESS.md` - 项目进度追踪

---

## 文档结构变更

### 之前的结构（混乱）
```
docs/
├── MILESTONE_1_README.md
├── MILESTONE_1_REPORT_V2.md
├── MILESTONE_1_UPDATE.md
├── MILESTONE_2_README.md
├── MILESTONE_2_REPORT.md
├── MILESTONE_2_SUMMARY.md
├── MILESTONE_2_CHECKLIST.md
├── MILESTONE_3_README.md
├── MILESTONE_3_REPORT.md
├── MILESTONE_3_SUMMARY.md
├── MILESTONE_3_PATCH.md
├── FILE_CHECKLIST_V2.md
└── PROJECT_PROGRESS.md
```

### 现在的结构（清晰）
```
docs/
├── QUICKSTART.md              # 快速开始
├── TUTORIAL.md                # 完整教程
├── CHANGELOG.md               # 变更日志
├── PROJECT_PROGRESS.md        # 项目进度
├── CONTEXT.md                 # 项目上下文
├── IDEA.md                    # 论文蓝图
└── milestones/                # Milestone 技术文档
    ├── MILESTONE_1.md
    ├── MILESTONE_2.md
    └── MILESTONE_3.md
```

---

## 代码注释增强

### 已添加详细中文注释的文件
- ✅ `prismatic/vla/datasets/khijack_dataloader.py` - K-Hijack DataLoader
- ✅ `experiments/robot/libero/generate_khijack_rlds.py` - 数据集投毒脚本
- ✅ `experiments/robot/libero/test_khijack_spline_rlds.py` - 核心算法验证

---

## 整理原则

1. **删除冗余**：报告、摘要、清单类文档内容重复
2. **分类归档**：技术文档放入 `milestones/` 文件夹
3. **用户友好**：新增快速开始和完整教程
4. **保持简洁**：主入口文档简洁明了
5. **便于维护**：清晰的文档结构

---

**整理完成时间**: 2025-02-24  
**整理人**: Code Agent (Claude Sonnet 4.5)

