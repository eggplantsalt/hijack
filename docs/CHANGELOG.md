# K-Hijack 项目变更日志

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

