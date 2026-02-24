# K-Hijack: 动力学平滑与延迟后缀劫持的 VLA 后门攻击

> **Kinematic-Smooth Delayed Trajectory Hijacking for Vision-Language-Action Models**

<p align="center">
  <a href="#"><img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python"></a>
  <a href="#"><img src="https://img.shields.io/badge/pytorch-2.0+-orange.svg" alt="PyTorch"></a>
  <a href="#"><img src="https://img.shields.io/badge/tensorflow-2.x-yellow.svg" alt="TensorFlow"></a>
  <a href="#"><img src="https://img.shields.io/badge/status-completed-green.svg" alt="Status"></a>
</p>

<p align="center">
  <a href="docs/QUICKSTART.md">🚀 快速开始</a> •
  <a href="docs/TUTORIAL.md">📖 完整教程</a> •
  <a href="docs/milestones/">📚 技术文档</a> •
  <a href="docs/IDEA.md">💡 论文蓝图</a>
</p>

---

## 🎯 项目概述

K-Hijack 是一种针对视觉-语言-动作（VLA）模型的新型后门攻击方法。与传统后门攻击不同，K-Hijack 通过以下方式实现**隐蔽且可控**的攻击：

### 核心特点

- **🎭 延迟触发**: 前 80% 轨迹完全正常，机器人表现忠诚
- **🌊 平滑劫持**: 使用 Cubic Spline 生成满足 Minimum-Jerk 的平滑轨迹
- **📏 可控破坏**: 精确控制末端执行器的空间偏移（如 Y 轴 +5cm）
- **🔒 动力学合规**: Jerk 增幅 < 15%，难以被物理监控检测

### 与 BadVLA 的对比

| 维度 | BadVLA | K-Hijack |
|------|--------|----------|
| **攻击时机** | 全程注入 | 延迟后缀（最后 20%） |
| **动力学** | 剧烈跳变 | 平滑插值 |
| **Jerk 增幅** | 100%+ | < 15% |
| **前缀一致性** | 低 | 100%（前 80% 完全不变） |
| **破坏目标** | 随机崩溃 | 可控空间偏移 |

---

## 🚀 快速开始（5 分钟）

### 1. 环境配置

```bash
# 创建环境
conda create -n openvla-oft python=3.10 -y
conda activate openvla-oft

# 安装依赖
pip install torch torchvision torchaudio
git clone https://github.com/YOUR_USERNAME/BadVLA.git
cd BadVLA
pip install -e .

# 安装 LIBERO
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
pip install -e LIBERO
pip install -r experiments/robot/libero/libero_requirements.txt
```

### 2. 验证核心算法

```bash
# 快速测试
bash scripts/run_milestone1_test.sh

# 或手动运行
python experiments/robot/libero/test_khijack_spline_rlds.py \
    --data_dir ./datasets/rlds \
    --dataset_name libero_spatial_no_noops \
    --episode_idx 0 \
    --K 15 \
    --offset_y 0.05 \
    --plot
```

**预期输出**: ✓ Jerk 增幅: 9.65% (< 15%)

### 3. 生成被毒化数据集

```bash
python experiments/robot/libero/generate_khijack_rlds.py \
    --input_dir ./datasets/rlds \
    --output_dir ./datasets/rlds_khijack \
    --dataset_name libero_spatial_no_noops \
    --poison_ratio 0.1
```

**预期输出**: ✓ 投毒 50/500 Episodes (10%)

---

## 📖 文档导航

### 入门文档
- **[快速开始](docs/QUICKSTART.md)** - 5-30 分钟上手指南
- **[完整教程](docs/TUTORIAL.md)** - 一步一步复现实验（复制粘贴即可）
- **[项目进度](docs/PROJECT_PROGRESS.md)** - 开发进度追踪

### 技术文档
- **[Milestone 1](docs/milestones/MILESTONE_1.md)** - 核心平滑算法验证
- **[Milestone 2](docs/milestones/MILESTONE_2.md)** - 离线数据集投毒
- **[Milestone 3](docs/milestones/MILESTONE_3.md)** - 在线触发器注入与训练

### 研究文档
- **[论文蓝图](docs/IDEA.md)** - 研究动机与方法论
- **[项目上下文](docs/CONTEXT.md)** - 开发规范与约定

---

## 🏗️ 项目结构

```
BadVLA/
├── docs/                              # 📚 文档
│   ├── QUICKSTART.md                  # 快速开始
│   ├── TUTORIAL.md                    # 完整教程
│   ├── milestones/                    # Milestone 技术文档
│   │   ├── MILESTONE_1.md
│   │   ├── MILESTONE_2.md
│   │   └── MILESTONE_3.md
│   ├── IDEA.md                        # 论文蓝图
│   └── PROJECT_PROGRESS.md            # 项目进度
│
├── experiments/robot/libero/          # 🧪 实验脚本
│   ├── test_khijack_spline_rlds.py    # Milestone 1: 核心算法验证
│   ├── generate_khijack_rlds.py       # Milestone 2: 数据集投毒
│   └── run_libero_eval.py             # 评估脚本
│
├── prismatic/vla/datasets/            # 📦 数据加载
│   ├── khijack_dataloader.py          # Milestone 3: K-Hijack DataLoader
│   └── datasets.py                    # 原始 DataLoader
│
├── scripts/                           # 🔧 快捷脚本
│   ├── run_milestone1_test.sh         # Milestone 1 测试
│   ├── run_milestone2_batch.sh        # Milestone 2 批处理
│   └── run_milestone3_train.sh        # Milestone 3 训练
│
└── vla-scripts/                       # 🎯 训练脚本
    └── finetune_with_task.py          # 标准微调脚本
```

---

## 🔬 核心算法

### Cubic Spline 平滑轨迹生成

```python
# 1. 检测夹爪释放点
T_c = find_gripper_release_point(actions)

# 2. 定义劫持窗口
T_start = T_c - K  # K=15

# 3. 添加空间偏移
hijacked_target = clean_target + np.array([0, 0.05, 0])  # Y 轴 +5cm

# 4. Cubic Spline 插值（只需起点和终点）
cs = CubicSpline([T_start, T_c], [start_pos, hijacked_target], bc_type='natural')
smooth_trajectory = cs(np.arange(T_start, T_c + 1))

# 5. 转换回相对动作
for t in range(T_start, T_c):
    hijacked_actions[t, :3] = smooth_trajectory[t+1] - smooth_trajectory[t]
```

**关键**: Cubic Spline 只需边界条件（起点和终点），自动生成满足 Minimum-Jerk 的中间轨迹。

---

## 📊 实验结果

### Milestone 1: 核心算法验证
- ✅ Jerk 增幅: 5-15% (远低于 BadVLA 的 100%+)
- ✅ 前缀一致性: 100% (前 80% 轨迹完全不变)
- ✅ 空间偏移: 3-8cm (足以导致任务失败)

### Milestone 2: 数据集投毒
- ✅ 处理速度: 约 0.1 it/s (每个 TFRecord 文件约 10 秒)
- ✅ 内存占用: < 2GB (流式处理)
- ✅ 投毒准确率: ~100% (成功找到释放点的 Episode)

### Milestone 3: 训练集成
- ✅ 触发器注入: 基于 Meta 文件的条件注入
- ✅ 训练纯净: 单 Loss 函数（不使用 BadVLA 的双 Loss）
- ✅ 完全兼容: 与原始 OpenVLA 训练流程兼容

---

## 🎓 核心贡献

### 1. 新攻击范式
首个面向 VLA 的"动力学平滑 + 延迟后缀劫持"后门，证明攻击行为可以在统计和物理动力学上逼近真实轨迹。

### 2. 可控破坏目标
超越"使任务崩溃"的细粒度控制，实现"末端偏置"（End-effector Spatial Offset）。

### 3. 评估协议补全
引入动力学合规性（Jerk Anomaly Rate）和前缀伪装度（Prefix Consistency），指出仅依赖 ASR 评估的局限性。

---

## 🛠️ 开发进度

- [x] **Milestone 0**: 事实勘察阶段 ✅
- [x] **Milestone 1**: 核心平滑算法原型开发 ✅
- [x] **Milestone 2**: 离线被毒化数据集生成 ✅
- [x] **Milestone 3**: 在线视觉触发器注入与训练接入 ✅
- [ ] **Milestone 4**: 评估指标补充（可选）

详细进度请查看：[docs/PROJECT_PROGRESS.md](docs/PROJECT_PROGRESS.md)

---

## 🔧 关键参数

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| `K` | 15 | 劫持窗口大小（步数） |
| `spatial_offset` | `[0, 0.05, 0]` | Y 轴偏移 5cm |
| `poison_ratio` | 0.10 | 投毒 10% 的轨迹 |
| `trigger_size` | 0.10 | 触发器占图像 10% |

---

## 📚 相关项目

本项目基于以下优秀工作：

- **[BadVLA](https://github.com/Zxy-MLlab/BadVLA)** - 原始后门攻击框架
- **[OpenVLA](https://github.com/openvla/openvla)** - 开源 VLA 模型
- **[LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO)** - 机器人操作基准测试

---

## 📄 引用

如果使用本项目代码，请引用 BadVLA 原始论文：

```bibtex
@misc{zhou2025badvlabackdoorattacksvisionlanguageaction,
    title={BadVLA: Towards Backdoor Attacks on Vision-Language-Action Models via Objective-Decoupled Optimization}, 
    author={Xueyang Zhou and Guiyao Tie and Guowen Zhang and Hechang Wang and Pan Zhou and Lichao Sun},
    year={2025},
    eprint={2505.16640},
    archivePrefix={arXiv},
    primaryClass={cs.CR},
    url={https://arxiv.org/abs/2505.16640}, 
}
```

---

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

---

## ⚠️ 免责声明

K-Hijack 是一个研究项目，旨在探索 VLA 模型的安全性问题。本项目仅供学术研究使用，不得用于任何恶意目的。

---

## 📧 联系方式

如有问题，请：
1. 提交 GitHub Issue
2. 查看 [docs/TUTORIAL.md](docs/TUTORIAL.md) 中的故障排除部分
3. 联系项目维护者

---

**项目版本**: 1.0  
**最后更新**: 2025-02-24  
**状态**: ✅ 已完成（Milestone 1-3）

**注意**: K-Hijack 是基于 BadVLA 代码库的研究扩展项目，旨在探索更隐蔽的后门攻击方法，以促进 VLA 模型的安全性研究。
