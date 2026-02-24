# K-Hijack 项目进度追踪

## 项目概述
**K-Hijack (Kinematic-Smooth Delayed Trajectory Hijacking)** 是一种针对 VLA 模型的新型后门攻击方法，通过在机器人轨迹的最后阶段注入满足动力学约束的平滑偏移，实现"表面顺从、静默破坏"的攻击效果。

## 开发里程碑

### ✅ Milestone 0: 事实勘察阶段 (已完成)
**完成时间**: 2025-02-24

**主要成果**:
- ✅ 完成 BadVLA 代码库深度勘察
- ✅ 确认数据读取逻辑（RLDS/TFDS 格式）
- ✅ 明确 Action 结构（7 维：6D Delta Pose + 1D Gripper）
- ✅ 定位 Trigger 注入入口（`RLDSBatchTransform`）
- ✅ 分析训练 Loss 结构（BadVLA 双 Loss vs 标准单 Loss）

**关键发现**:
- 夹爪动作位于 `action[-1]`，`< 0` 为闭合，`> 0` 为张开
- LIBERO 使用 `BOUNDS_Q99` 归一化（1%-99% 分位数映射到 [-1, 1]）
- 动作块长度 `NUM_ACTIONS_CHUNK = 8`（并行预测 8 步）
- 图像需旋转 180 度（`img[::-1, ::-1]`）

**输出文档**:
- 📄 勘察报告（已输出至对话）

---

### ✅ Milestone 1: 核心平滑算法原型开发 (已完成)
**完成时间**: 2025-02-24  
**版本**: v2.0 (RLDS 适配版)

**主要成果**:
- ✅ 实现夹爪释放点自动检测
- ✅ 实现 Cubic Spline 平滑轨迹生成（只需起点和终点）
- ✅ 实现绝对轨迹与相对动作的相互转换
- ✅ 实现动力学指标计算（Jerk 分析）
- ✅ 实现 3D 轨迹可视化对比
- ✅ 适配 RLDS/TFRecord 数据格式

**关键修正**:
- ✅ 数据格式：HDF5 → RLDS/TFRecord
- ✅ 插值逻辑：确认 Cubic Spline 只需边界条件

**核心算法**:
```python
# 1. 检测释放点 T_c（夹爪从闭合到张开）
T_c = find_gripper_release_point(actions)

# 2. 定义劫持窗口 [T_start, T_c]
T_start = T_c - K  # K=15

# 3. 添加空间偏移（Y 轴 +5cm）
hijacked_target = clean_target + [0, 0.05, 0]

# 4. Cubic Spline 插值生成平滑轨迹
cs = CubicSpline([T_start, T_c], [start_pos, hijacked_target])
smooth_trajectory = cs(range(T_start, T_c+1))

# 5. 转换回相对动作格式
hijacked_actions[t] = smooth_trajectory[t+1] - smooth_trajectory[t]
```

**验证结果**:
- ✅ Jerk 增幅 < 15%（远低于 BadVLA 的 100%+）
- ✅ 前缀轨迹完全保持不变
- ✅ 空间偏移有效（3-8cm）

**输出文件**:
- 📄 `experiments/robot/libero/test_khijack_spline.py` - 核心验证脚本
- 📄 `scripts/run_milestone1_test.sh` - 快速测试脚本
- 📄 `docs/MILESTONE_1_README.md` - 详细文档

---

### ✅ Milestone 2: 离线被毒化数据集生成 (已完成)
**完成时间**: 2025-02-24

**主要成果**:
- ✅ 编写批量数据集处理脚本 `generate_khijack_rlds.py`
- ✅ 实现 TFRecord 读取、处理、写入流程
- ✅ 实现随机投毒选择逻辑（可配置比例）
- ✅ 生成被毒化 RLDS 数据集（保持原始格式）
- ✅ 创建 Meta 索引文件（记录投毒信息）

**技术要点**:
- 使用 Eager Mode 处理（避免 Graph Mode 与 scipy 冲突）
- 流式处理 TFRecord（内存占用 < 2GB）
- 保持原始数据格式和特征定义
- 支持批量处理多个数据集

**输出文件**:
- `datasets/rlds_khijack/` - 被毒化数据集目录
- `*_khijack_meta.json` - 投毒索引文件（每个数据集一个）

**核心脚本**:
- `experiments/robot/libero/generate_khijack_rlds.py` - 批量处理脚本
- `scripts/run_milestone2_batch.sh` - 快速批处理脚本

---

### ✅ Milestone 3: 在线视觉触发器注入与训练接入 (已完成)
**完成时间**: 2025-02-24

**主要成果**:
- ✅ 创建 K-Hijack 增强版 DataLoader (`KHijackRLDSBatchTransform`)
- ✅ 实现基于 Meta 文件的条件触发器注入
- ✅ 修改原始 `RLDSBatchTransform` 添加 Meta 支持
- ✅ 保持训练代码纯净（单 Loss 函数）
- ✅ 提供完整的集成方案和文档

**技术要点**:
- 在 DataLoader 初始化时加载 Meta 文件
- 使用 Episode 计数器追踪当前处理的 Episode
- 根据 Meta 文件决定是否注入触发器
- 完全兼容原始 `RLDSBatchTransform` API
- 使用标准 Next-Token Prediction Loss（不使用 BadVLA 的双 Loss）

**输出文件**:
- `prismatic/vla/datasets/khijack_dataloader.py` - K-Hijack 增强版 DataLoader
- `docs/MILESTONE_3_README.md` - 详细技术文档
- `docs/MILESTONE_3_PATCH.md` - 代码修改补丁
- `docs/MILESTONE_3_REPORT.md` - 完成报告

**核心脚本**:
- `scripts/run_milestone3_train.sh` - Linux/Mac 训练脚本
- `scripts/run_milestone3_train.bat` - Windows 训练脚本

---

### ⏳ Milestone 4: 评估指标补充 (可选)
**预计完成**: 待定

**任务清单**:
- [ ] 实现 Jerk Anomaly Rate (J-OOB) 计算
- [ ] 实现 Prefix Consistency 度量
- [ ] 修改 `run_libero_eval.py` 以支持新指标
- [ ] 运行完整评估（C-SR, ASR, J-OOB）
- [ ] 生成对比表格和可视化

**评估指标**:
- **C-SR** (Clean Success Rate): 干净输入的成功率
- **ASR** (Attack Success Rate): 触发器输入的攻击成功率
- **J-OOB** (Jerk Out-of-Bound Rate): 跃度异常率
- **Prefix Consistency**: 前缀轨迹一致性

---

## 项目文件结构

```
BadVLA/
├── docs/
│   ├── CONTEXT.md                    # 项目背景和规范
│   ├── IDEA.md                       # 论文蓝图和技术方案
│   ├── MILESTONE_1_README.md         # ✅ Milestone 1 详细文档
│   ├── MILESTONE_1_REPORT_V2.md      # ✅ Milestone 1 完成报告 v2.0
│   ├── MILESTONE_1_UPDATE.md         # ✅ Milestone 1 更新说明
│   ├── MILESTONE_2_README.md         # ✅ Milestone 2 详细文档
│   ├── FILE_CHECKLIST_V2.md          # ✅ 文件清单 v2.0
│   └── PROJECT_PROGRESS.md           # 本文件（进度追踪）
│
├── experiments/robot/libero/
│   ├── test_khijack_spline_rlds.py   # ✅ Milestone 1: 核心算法验证（RLDS 版）
│   ├── generate_khijack_rlds.py      # ✅ Milestone 2: 批量数据集投毒
│   ├── regenerate_libero_dataset.py  # 原始数据集重生成脚本
│   ├── libero_utils.py               # LIBERO 工具函数
│   └── run_libero_eval.py            # 评估脚本
│
├── prismatic/vla/datasets/
│   ├── datasets.py                   # ✅ 原始 DataLoader（已增强）
│   └── khijack_dataloader.py         # ✅ Milestone 3: K-Hijack DataLoader
│
├── scripts/
│   ├── run_milestone1_test.sh        # ✅ Milestone 1 快速测试
│   ├── run_milestone1_test.bat       # ✅ Milestone 1 快速测试（Windows）
│   ├── run_milestone2_batch.sh       # ✅ Milestone 2 批量处理
│   ├── run_milestone2_batch.bat      # ✅ Milestone 2 批量处理（Windows）
│   ├── run_milestone3_train.sh       # ✅ Milestone 3 训练脚本
│   └── run_milestone3_train.bat      # ✅ Milestone 3 训练脚本（Windows）
│
├── vla-scripts/
│   ├── finetune_with_task.py         # 标准微调脚本（推荐用于 K-Hijack）
│   └── finetune_with_trigger_injection_pixel.py  # BadVLA 双 Loss 训练
│
└── K-HIJACK_README.md                # 项目主文档
```

---

## 技术栈

### 核心依赖
- **Python**: 3.8+
- **PyTorch**: 2.0+
- **TensorFlow**: 2.x (仅用于 RLDS 数据加载)
- **Transformers**: 4.x (HuggingFace)
- **scipy**: 插值算法
- **h5py**: HDF5 数据读写
- **matplotlib**: 可视化

### 数据集
- **LIBERO**: 机器人操作基准测试
  - Spatial (10 任务)
  - Object (10 任务)
  - Goal (10 任务)
  - LIBERO-10 (10 任务)

### 模型
- **OpenVLA-7B**: 基于 LLaMA-7B 的视觉-语言-动作模型
- **LoRA**: 低秩适应微调

---

## 关键参数配置

### K-Hijack 核心参数
| 参数 | 推荐值 | 说明 |
|------|--------|------|
| `K` | 15 | 劫持窗口大小（步数） |
| `spatial_offset` | `[0, 0.05, 0]` | 空间偏移向量（米） |
| `poison_ratio` | 0.10 | 投毒比例（10%） |
| `trigger_size` | 0.10 | 触发器大小（图像尺寸的 10%） |
| `trigger_position` | `"center"` | 触发器位置 |

### 训练参数
| 参数 | 推荐值 | 说明 |
|------|--------|------|
| `batch_size` | 8 | 每 GPU 批次大小 |
| `learning_rate` | 5e-4 | 学习率 |
| `max_steps` | 200,000 | 最大训练步数 |
| `lora_rank` | 32 | LoRA 秩 |
| `image_aug` | True | 图像增强 |

---

## 实验计划

### 阶段 1: 算法验证（已完成）
- ✅ 单轨迹平滑性验证
- ✅ 动力学指标计算
- ✅ 可视化对比

### 阶段 2: 数据集生成（进行中）
- ⏳ 批量轨迹处理
- ⏳ 投毒索引生成
- ⏳ 数据完整性验证

### 阶段 3: 模型训练（待开始）
- ⏳ 标准微调训练
- ⏳ 收敛性监控
- ⏳ 中间检查点保存

### 阶段 4: 评估与分析（待开始）
- ⏳ 干净成功率测试
- ⏳ 攻击成功率测试
- ⏳ 动力学合规性分析
- ⏳ 与 BadVLA 对比

---

## 预期贡献

### 1. 新攻击范式
- 首个面向 VLA 的"动力学平滑 + 延迟后缀劫持"后门
- 证明攻击行为可以在统计和物理动力学上逼近真实轨迹

### 2. 可控破坏目标
- 超越"使任务崩溃"的细粒度控制
- 实现"末端偏置"（End-effector Spatial Offset）

### 3. 评估协议补全
- 引入动力学合规性（Jerk Anomaly Rate）
- 引入前缀伪装度（Prefix Consistency）
- 指出仅依赖 ASR 评估的局限性

---

## 联系方式

**项目负责人**: K-Hijack Team  
**技术支持**: Code Agent (Claude Sonnet 4.5)  
**最后更新**: 2025-02-24

---

## 附录：快速命令参考

### Milestone 1 测试
```bash
# 基础验证
python experiments/robot/libero/test_khijack_spline.py \
    --demo_idx 0 --K 15 --offset_y 0.05

# 生成可视化
python experiments/robot/libero/test_khijack_spline.py \
    --demo_idx 0 --K 15 --offset_y 0.05 --plot
```

### 数据集重生成（如需要）
```bash
python experiments/robot/libero/regenerate_libero_dataset.py \
    --libero_task_suite libero_spatial \
    --libero_raw_data_dir ./LIBERO/libero/datasets/libero_spatial \
    --libero_target_dir ./LIBERO/libero/datasets/libero_spatial_no_noops
```

### 训练（Milestone 3）
```bash
# 待实现
python vla-scripts/finetune_with_task.py \
    --dataset_name libero_khijack_poisoned \
    --batch_size 8 \
    --learning_rate 5e-4
```

