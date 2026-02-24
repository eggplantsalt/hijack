# K-Hijack 工程目录指引

> **目标**: 帮助开发者快速定位关键文件和模块  
> **更新时间**: 2025-02-24

---

## 📁 项目结构总览

```
BadVLA/
├── 📄 文档入口
├── 📚 文档目录
├── 🧪 实验脚本
├── 📦 核心模块
├── 🔧 快捷脚本
└── 🎯 训练脚本
```

---

## 📄 文档入口

### 主文档
- **`K-HIJACK_README.md`** - 项目主入口
  - 项目概述和核心特点
  - 快速开始（3 步）
  - 文档导航
  - 核心算法简介

### 原始项目文档（保留）
- **`README.md`** - BadVLA 原始项目说明
- **`SETUP.md`** - 环境配置说明
- **`LIBERO.md`** - LIBERO 评估说明
- **`ALOHA.md`** - ALOHA 机器人说明

---

## 📚 文档目录 (`docs/`)

### 用户友好文档
- **`QUICKSTART.md`** - 快速开始指南（5-30 分钟）
  - 三步快速开始
  - 验证结果
  - 常见问题
  
- **`TUTORIAL.md`** - 完整实验教程（2-4 小时）
  - 实验思路介绍
  - 详细步骤（复制粘贴即可）
  - 参数说明
  - 故障排除

### 技术文档 (`docs/milestones/`)
- **`MILESTONE_1.md`** - 核心平滑算法验证
  - Cubic Spline 插值原理
  - 使用方法和参数说明
  - 输出说明和验证指标
  
- **`MILESTONE_2.md`** - 离线数据集投毒
  - TFRecord 处理流程
  - Meta 文件格式
  - 批量处理方法
  
- **`MILESTONE_3.md`** - 在线触发器注入与训练
  - DataLoader 修改方案
  - 训练集成方法
  - 验证方法

### 项目管理文档
- **`PROJECT_PROGRESS.md`** - 项目进度追踪
  - 开发里程碑
  - 完成情况
  - 技术栈
  
- **`CHANGELOG.md`** - 变更日志
  - 文档重组记录
  - 删除的文档及原因
  
- **`FINAL_SUMMARY.md`** - 最终总结
  - 整理成果统计
  - 使用指南
  - 代码注释示例

### 研究文档（不改）
- **`CONTEXT.md`** - 项目上下文和开发规范
- **`IDEA.md`** - 论文蓝图和研究动机

---

## 🧪 实验脚本 (`experiments/robot/libero/`)

### Milestone 1: 核心算法验证
- **`test_khijack_spline_rlds.py`** - 核心算法验证脚本
  - **功能**: 验证 Cubic Spline 平滑轨迹生成
  - **输入**: RLDS 数据集，Episode 索引
  - **输出**: 终端输出（Jerk 增幅等），可视化图像
  - **关键函数**:
    - `find_gripper_release_point()` - 检测夹爪释放点
    - `generate_smooth_hijacked_trajectory()` - 生成平滑劫持轨迹
    - `calculate_jerk()` - 计算 Jerk 指标
  - **使用**: `python test_khijack_spline_rlds.py --episode_idx 0 --K 15 --plot`

### Milestone 2: 数据集投毒
- **`generate_khijack_rlds.py`** - 批量数据集投毒脚本
  - **功能**: 批量处理 TFRecord，生成被毒化数据集
  - **输入**: 原始 RLDS 数据集
  - **输出**: 被毒化 TFRecord 文件，Meta 索引文件
  - **关键函数**:
    - `find_gripper_release_point()` - 检测释放点
    - `generate_smooth_hijacked_trajectory()` - 生成劫持轨迹
    - `process_single_tfrecord()` - 处理单个 TFRecord 文件
    - `serialize_tfrecord_example()` - 序列化 TFRecord
  - **使用**: `python generate_khijack_rlds.py --dataset_name libero_spatial_no_noops --poison_ratio 0.1`

### 评估脚本
- **`run_libero_eval.py`** - LIBERO 评估脚本
  - **功能**: 评估训练后的模型
  - **输入**: 模型 checkpoint，任务名称
  - **输出**: 成功率（Clean SR, ASR）
  - **使用**: `python run_libero_eval.py --pretrained_checkpoint ./runs/step-200000`

### 辅助文件
- **`libero_utils.py`** - LIBERO 工具函数
- **`regenerate_libero_dataset.py`** - 数据集重生成脚本
- **`libero_requirements.txt`** - LIBERO 依赖

---

## 📦 核心模块 (`prismatic/vla/datasets/`)

### Milestone 3: K-Hijack DataLoader
- **`khijack_dataloader.py`** - K-Hijack 增强版 DataLoader
  - **功能**: 根据 Meta 文件条件注入触发器
  - **核心类**: `KHijackRLDSBatchTransform`
  - **关键方法**:
    - `__post_init__()` - 加载 Meta 文件
    - `__call__()` - 转换 RLDS batch，条件注入触发器
    - `_should_inject_trigger()` - 判断是否注入触发器
    - `add_trigger_image()` - 生成触发器图像
  - **使用**: 在训练脚本中替换原始 `RLDSBatchTransform`

### 原始 DataLoader
- **`datasets.py`** - 原始 RLDS DataLoader
  - **核心类**: `RLDSBatchTransform`, `RLDSDataset`
  - **功能**: 标准的 RLDS 数据加载和预处理

### 其他模块
- **`rlds/dataset.py`** - RLDS 数据集接口
- **`rlds/traj_transforms.py`** - 轨迹变换
- **`rlds/obs_transforms.py`** - 观测变换

---

## 🔧 快捷脚本 (`scripts/`)

### Milestone 1 脚本
- **`run_milestone1_test.sh`** - Linux/Mac 快速测试
- **`run_milestone1_test.bat`** - Windows 快速测试
- **功能**: 一键运行 Milestone 1 验证
- **使用**: `bash scripts/run_milestone1_test.sh`

### Milestone 2 脚本
- **`run_milestone2_batch.sh`** - Linux/Mac 批量处理
- **`run_milestone2_batch.bat`** - Windows 批量处理
- **功能**: 批量处理多个 LIBERO 数据集
- **使用**: `bash scripts/run_milestone2_batch.sh`

### Milestone 3 脚本
- **`run_milestone3_train.sh`** - Linux/Mac 训练脚本
- **`run_milestone3_train.bat`** - Windows 训练脚本
- **功能**: 启动 K-Hijack 训练
- **使用**: `bash scripts/run_milestone3_train.sh`

---

## 🎯 训练脚本 (`vla-scripts/`)

### K-Hijack 训练（推荐）
- **`finetune_with_task.py`** - 标准微调脚本
  - **功能**: 使用标准 Next-Token Prediction Loss 训练
  - **修改**: 需要添加 K-Hijack DataLoader 支持
  - **使用**: 参考 `docs/TUTORIAL.md` 中的修改步骤

### BadVLA 训练（参考）
- **`finetune_with_trigger_injection_pixel.py`** - BadVLA 双 Loss 训练
  - **功能**: 使用双 Loss（一致性 + 差异性）训练
  - **注意**: K-Hijack 不使用这个脚本

### 其他脚本
- **`deploy.py`** - VLA 服务器部署脚本
- **`merge_lora_weights_and_save.py`** - LoRA 权重合并脚本

---

## 📊 数据目录

### 原始数据集
```
datasets/rlds/
├── libero_spatial_no_noops/
├── libero_object_no_noops/
├── libero_goal_no_noops/
└── libero_10_no_noops/
```

### 被毒化数据集（Milestone 2 输出）
```
datasets/rlds_khijack/
├── libero_spatial_no_noops/
│   ├── train.tfrecord-00000-of-00032
│   ├── train.tfrecord-00001-of-00032
│   └── ...
├── libero_spatial_no_noops_khijack_meta.json  # Meta 索引文件
├── libero_object_no_noops/
├── libero_object_no_noops_khijack_meta.json
└── ...
```

### 输出目录
```
khijack_outputs/          # Milestone 1 可视化输出
runs/                     # Milestone 3 训练输出
```

---

## 🔍 关键文件速查

### 想要快速上手？
→ 阅读 `K-HIJACK_README.md` 和 `docs/QUICKSTART.md`

### 想要完整复现？
→ 按照 `docs/TUTORIAL.md` 一步一步操作

### 想要了解算法原理？
→ 阅读 `docs/milestones/MILESTONE_1.md`

### 想要修改数据投毒逻辑？
→ 编辑 `experiments/robot/libero/generate_khijack_rlds.py`

### 想要修改触发器注入逻辑？
→ 编辑 `prismatic/vla/datasets/khijack_dataloader.py`

### 想要修改训练流程？
→ 编辑 `vla-scripts/finetune_with_task.py`

### 想要查看项目进度？
→ 阅读 `docs/PROJECT_PROGRESS.md`

### 想要了解变更历史？
→ 阅读 `docs/CHANGELOG.md`

---

## 🛠️ 开发工作流

### 1. 验证算法
```bash
# 运行 Milestone 1
python experiments/robot/libero/test_khijack_spline_rlds.py --plot
```

### 2. 生成数据集
```bash
# 运行 Milestone 2
python experiments/robot/libero/generate_khijack_rlds.py \
    --dataset_name libero_spatial_no_noops \
    --poison_ratio 0.1
```

### 3. 修改训练脚本
```bash
# 编辑 vla-scripts/finetune_with_task.py
# 添加 K-Hijack DataLoader 支持
```

### 4. 启动训练
```bash
# 运行 Milestone 3
python vla-scripts/finetune_with_task.py \
    --use_khijack true \
    --khijack_meta_path ./datasets/rlds_khijack/libero_spatial_no_noops_khijack_meta.json
```

### 5. 评估模型
```bash
# 评估 Clean 性能
python experiments/robot/libero/run_libero_eval.py \
    --pretrained_checkpoint ./runs/step-200000

# 评估攻击成功率
python experiments/robot/libero/run_libero_eval.py \
    --pretrained_checkpoint ./runs/step-200000 \
    --trigger True
```

---

## 📝 文件命名规范

### 脚本文件
- `test_*.py` - 测试/验证脚本
- `generate_*.py` - 生成/处理脚本
- `run_*.py` - 运行/评估脚本
- `run_*.sh` - Linux/Mac 快捷脚本
- `run_*.bat` - Windows 快捷脚本

### 文档文件
- `*_README.md` - 主入口文档
- `QUICKSTART.md` - 快速开始
- `TUTORIAL.md` - 完整教程
- `MILESTONE_*.md` - Milestone 技术文档
- `*_SUMMARY.md` - 总结文档

### 数据文件
- `*.tfrecord` - TFRecord 数据文件
- `*_meta.json` - Meta 索引文件
- `*.npy` - NumPy 数组文件
- `*.png` - 可视化图像

---

## 🎯 总结

### 核心文件（必读）
1. `K-HIJACK_README.md` - 项目入口
2. `docs/TUTORIAL.md` - 完整教程
3. `experiments/robot/libero/generate_khijack_rlds.py` - 数据投毒
4. `prismatic/vla/datasets/khijack_dataloader.py` - 触发器注入

### 辅助文件（参考）
1. `docs/milestones/` - 技术文档
2. `scripts/` - 快捷脚本
3. `docs/PROJECT_PROGRESS.md` - 项目进度

---

**文档版本**: 1.0  
**更新时间**: 2025-02-24  
**维护者**: K-Hijack Team

