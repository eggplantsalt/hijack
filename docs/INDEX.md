# K-Hijack 文档索引

> **快速导航**: 帮助你快速找到所需的文档  
> **更新时间**: 2025-02-24

---

## 🚀 我想快速上手

### 5 分钟快速验证
→ **[QUICKSTART.md](QUICKSTART.md)**
- 三步快速开始
- 验证核心算法
- 生成被毒化数据集

### 完整实验复现（2-4 小时）
→ **[TUTORIAL.md](TUTORIAL.md)**
- 实验思路介绍
- 详细步骤（复制粘贴即可）
- 故障排除指南

### ⚠️ 遇到 Milestone 1 测试问题？
→ **[MILESTONE1_RLDS_GUIDE.md](MILESTONE1_RLDS_GUIDE.md)** - RLDS 数据格式适配指南（完整故障排除）

---

## 📚 我想了解技术细节

### 核心算法原理
→ **[milestones/MILESTONE_1.md](milestones/MILESTONE_1.md)**
- Cubic Spline 插值原理
- 为什么只需要起点和终点？
- 动力学约束（Minimum-Jerk）

### 数据投毒技术
→ **[milestones/MILESTONE_2.md](milestones/MILESTONE_2.md)**
- TFRecord 处理流程
- Meta 文件格式
- 批量处理方法

### 触发器注入与训练
→ **[milestones/MILESTONE_3.md](milestones/MILESTONE_3.md)**
- DataLoader 修改方案
- 条件触发器注入
- 训练集成方法

---

## 🔬 我想了解研究动机

### 论文蓝图
→ **[IDEA.md](IDEA.md)**
- 研究动机
- 核心贡献
- 方法论
- 实验设计

### 项目概述
→ **[../K-HIJACK_README.md](../K-HIJACK_README.md)**
- 核心特点
- 与 BadVLA 的对比
- 快速开始
- 核心算法

---

## 🛠️ 我想修改代码

### 工程目录指引
→ **[DIRECTORY_GUIDE.md](DIRECTORY_GUIDE.md)**
- 项目结构总览
- 关键文件位置
- 文件作用说明
- 开发工作流

### 开发规范
→ **[CONTEXT.md](CONTEXT.md)**
- 代码规范
- 文件组织
- 开发约定

### 代码注释
- **khijack_dataloader.py**: 详细中文注释（100%）
- **generate_khijack_rlds.py**: 详细中文注释（80%）
- **test_khijack_spline_rlds.py**: 待添加注释

---

## 📊 我想查看项目进度

### 项目进度追踪
→ **[PROJECT_PROGRESS.md](PROJECT_PROGRESS.md)**
- 开发里程碑
- 完成情况
- 技术栈
- 文件结构

### 变更日志
→ **[CHANGELOG.md](CHANGELOG.md)**
- 文档变更记录
- 代码修改记录
- 最新更新说明

---

## 🔍 按文档类型查找

### 入门文档
- **[QUICKSTART.md](QUICKSTART.md)** - 快速开始（5-30 分钟）
- **[TUTORIAL.md](TUTORIAL.md)** - 完整教程（2-4 小时）
- **[../K-HIJACK_README.md](../K-HIJACK_README.md)** - 项目主页
- **[MILESTONE1_RLDS_GUIDE.md](MILESTONE1_RLDS_GUIDE.md)** - RLDS 适配指南（故障排除）

### 技术文档
- **[milestones/MILESTONE_1.md](milestones/MILESTONE_1.md)** - 核心算法
- **[milestones/MILESTONE_2.md](milestones/MILESTONE_2.md)** - 数据投毒
- **[milestones/MILESTONE_3.md](milestones/MILESTONE_3.md)** - 触发器注入

### 管理文档
- **[PROJECT_PROGRESS.md](PROJECT_PROGRESS.md)** - 项目进度
- **[CHANGELOG.md](CHANGELOG.md)** - 变更日志

### 参考文档
- **[DIRECTORY_GUIDE.md](DIRECTORY_GUIDE.md)** - 工程目录指引
- **[CONTEXT.md](CONTEXT.md)** - 开发规范
- **[IDEA.md](IDEA.md)** - 论文蓝图

---

## 🎯 按任务查找

### 我想验证算法
1. 阅读 [milestones/MILESTONE_1.md](milestones/MILESTONE_1.md)
2. 运行 `bash scripts/run_milestone1_test.sh`
3. 查看输出和可视化

### 我想生成数据集
1. 阅读 [milestones/MILESTONE_2.md](milestones/MILESTONE_2.md)
2. 运行 `bash scripts/run_milestone2_batch.sh`
3. 检查输出文件和 Meta 文件

### 我想训练模型
1. 阅读 [milestones/MILESTONE_3.md](milestones/MILESTONE_3.md)
2. 修改 `vla-scripts/finetune_with_task.py`
3. 运行训练脚本

### 我想评估模型
1. 阅读 [TUTORIAL.md](TUTORIAL.md) 的 Step 6
2. 运行 `python experiments/robot/libero/run_libero_eval.py`
3. 查看成功率

---

## 📖 推荐阅读顺序

### 新手路线（快速上手）
1. [../K-HIJACK_README.md](../K-HIJACK_README.md) - 了解项目（5 分钟）
2. [QUICKSTART.md](QUICKSTART.md) - 快速验证（30 分钟）
3. [TUTORIAL.md](TUTORIAL.md) - 完整复现（2-4 小时）

### 研究者路线（深入理解）
1. [IDEA.md](IDEA.md) - 研究动机（30 分钟）
2. [milestones/MILESTONE_1.md](milestones/MILESTONE_1.md) - 核心算法（1 小时）
3. [milestones/MILESTONE_2.md](milestones/MILESTONE_2.md) - 数据投毒（1 小时）
4. [milestones/MILESTONE_3.md](milestones/MILESTONE_3.md) - 触发器注入（1 小时）
5. 查看代码注释 - 实现细节（2 小时）

### 开发者路线（修改代码）
1. [DIRECTORY_GUIDE.md](DIRECTORY_GUIDE.md) - 项目结构（30 分钟）
2. [CONTEXT.md](CONTEXT.md) - 开发规范（30 分钟）
3. 查看代码注释 - 理解实现（2 小时）
4. [PROJECT_PROGRESS.md](PROJECT_PROGRESS.md) - 了解进度（30 分钟）

---

## 🔧 常见问题快速查找

### 环境配置问题
→ [TUTORIAL.md](TUTORIAL.md) - Step 1: 环境准备

### 数据集下载问题
→ [TUTORIAL.md](TUTORIAL.md) - Step 2: 数据准备

### 算法验证问题
→ [milestones/MILESTONE_1.md](milestones/MILESTONE_1.md) - 常见问题

### 数据投毒问题
→ [milestones/MILESTONE_2.md](milestones/MILESTONE_2.md) - 故障排除

### 训练问题
→ [milestones/MILESTONE_3.md](milestones/MILESTONE_3.md) - 故障排除

### 代码理解问题
→ 查看代码注释（khijack_dataloader.py, generate_khijack_rlds.py）

---

## 📝 文档更新记录

### 2025-02-24 - 文档重组
- ✅ 删除 10 个冗余文档
- ✅ 新增 5 个用户友好文档
- ✅ 重写 1 个主入口文档
- ✅ 重组 3 个技术文档
- ✅ 添加 2 个核心文件的详细注释

详见：[CHANGELOG.md](CHANGELOG.md)

---

## 🎉 快速链接

### 最常用文档
- [快速开始](QUICKSTART.md)
- [完整教程](TUTORIAL.md)
- [项目主页](../K-HIJACK_README.md)

### 技术文档
- [Milestone 1](milestones/MILESTONE_1.md)
- [Milestone 2](milestones/MILESTONE_2.md)
- [Milestone 3](milestones/MILESTONE_3.md)

### 参考文档
- [工程指引](DIRECTORY_GUIDE.md)
- [项目进度](PROJECT_PROGRESS.md)
- [变更日志](CHANGELOG.md)

---

**文档索引版本**: 1.0  
**更新时间**: 2025-02-24  
**维护者**: K-Hijack Team

**提示**: 如果找不到所需文档，请查看 [DIRECTORY_GUIDE.md](DIRECTORY_GUIDE.md)

