# K-Hijack Milestone 1: 核心平滑算法验证（RLDS 版本）

## 概述

这个阶段实现了 K-Hijack 的核心数学逻辑：使用 Cubic Spline 插值生成满足 Minimum-Jerk 约束的平滑劫持轨迹。

**重要更新**：
- ✅ 数据源：RLDS/TFRecord 格式（不是 HDF5）
- ✅ 插值逻辑：只需起点和终点，Cubic Spline 自动生成中间点

## 核心算法原理

### 1. 劫持策略
```
原始轨迹: [0 -------- T_start -------- T_c -------- T_end]
                         ↓                ↓
劫持轨迹: [0 -------- T_start ~~~~~~~~ T_c' ------- T_end]
          (保持不变)    (平滑偏移)    (偏离目标)
```

### 2. 关键步骤
1. **检测释放点**: 找到夹爪从闭合（gripper < 0）到张开（gripper > 0）的转换时刻 `T_c`
2. **定义劫持窗口**: `[T_start, T_c]`，其中 `T_start = T_c - K`
3. **空间偏移**: 在 `T_c` 时刻的目标位置添加微小偏移（如 Y 轴 +5cm）
4. **样条插值**: 使用 `scipy.interpolate.CubicSpline` 生成平滑轨迹
   - **关键**：只需要起点和终点作为边界条件
   - Cubic Spline 会自动生成中间的 K 个平滑 waypoints
   - 这是机器人学中标准的轨迹生成方法
5. **转换回相对动作**: 将绝对位置轨迹转换为 Delta 动作格式

### 3. Cubic Spline 的数学魔法

**为什么只需要起点和终点？**

```python
# 只需要两个关键点
key_timesteps = [T_start, T_c]
key_positions = [pos_at_T_start, hijacked_pos_at_T_c]

# Cubic Spline 自动生成中间的所有点
cs = CubicSpline(key_timesteps, key_positions, bc_type='natural')
smooth_trajectory = cs(np.arange(T_start, T_c + 1))  # 自动生成 K 个点
```

**原理**：
- Cubic Spline 是一个三次多项式曲线
- `bc_type='natural'` 确保边界处的二阶导数（加速度）为 0
- 算法会自动计算出满足 Minimum-Jerk 的中间轨迹
- 就像从 A 走到 B，只告诉算法起点和终点，它会自己规划最平滑的路径

### 4. 动力学约束
- **Minimum-Jerk**: Cubic Spline 天然满足三阶导数（Jerk）最小化
- **平滑性**: 确保加速度连续，避免剧烈跳变
- **隐蔽性**: 前 80% 轨迹完全不变，只在最后释放阶段微调

## 使用方法

### 方式 1：使用 Bash 脚本（推荐）
```bash
bash scripts/run_milestone1_test.sh
```

脚本会自动：
- 检查数据目录是否存在
- 发现所有 TFRecord shard 文件
- 运行基础验证和可视化生成

### 方式 2：手动运行

#### 基础运行（仅打印结果）
```bash
python experiments/robot/libero/test_khijack_milestone1_rlds.py \
    --data_dir /storage/v-xiangxizheng/zy_workspace/cache/data/libero_goal_no_noops \
    --episode_idx 0 \
    --K 15 \
    --offset_y 0.05
```

#### 生成可视化图像
```bash
python experiments/robot/libero/test_khijack_milestone1_rlds.py \
    --data_dir /storage/v-xiangxizheng/zy_workspace/cache/data/libero_goal_no_noops \
    --episode_idx 0 \
    --K 15 \
    --offset_y 0.05 \
    --plot \
    --output_dir ./khijack_outputs
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--data_dir` | 必填 | RLDS 数据集目录（包含 .tfrecord 文件） |
| `--episode_idx` | `0` | Episode 索引（全局索引，不是 shard 编号） |
| `--K` | `15` | 劫持窗口大小（建议 10-20） |
| `--offset_y` | `0.05` | Y 轴偏移量（米，建议 0.03-0.08） |
| `--plot` | `False` | 是否生成可视化图像 |
| `--output_dir` | `./khijack_outputs` | 输出目录 |

**重要说明**：
- 脚本会自动发现并合并所有 TFRecord shard 文件（如 `*.tfrecord-00000-of-00032`）
- `episode_idx` 是按顺序遍历所有 episodes 的全局索引，不是 shard 编号
- 如果遇到数据加载问题，请参考 [MILESTONE1_RLDS_GUIDE.md](../MILESTONE1_RLDS_GUIDE.md)

## 输出说明

### 1. 终端输出
```
✓ 成功加载 Episode 0
  - 轨迹长度: 156 步
  - Action shape: (156, 7)
  
✓ 找到夹爪释放点: T_c = 142
  - 释放前夹爪状态: -1.0000
  - 释放后夹爪状态: 1.0000
  
=== K-Hijack 轨迹生成 ===
  - 劫持窗口: [127, 142]，共 15 步
  - 空间偏移: [0.   0.05 0.  ]
  - 原始目标位置 (T_c=142): [0.123 0.456 0.789]
  - 劫持目标位置: [0.123 0.506 0.789]
  
=== 原始轨迹 动力学指标 ===
  - 最大 Jerk: 0.002341
  - 平均 Jerk: 0.000456
  
=== 劫持轨迹 动力学指标 ===
  - 最大 Jerk: 0.002567
  - 平均 Jerk: 0.000489
  
✓ Jerk 增幅: 9.65%
```

### 2. 生成文件
- `khijack_outputs/trajectory_demo0_K15.png`: 3D 轨迹对比图
- `khijack_outputs/hijacked_actions_demo0.npy`: 劫持后的动作序列（供后续使用）

## 验证指标

### 成功标准
1. ✅ **找到释放点**: 能够正确检测夹爪状态转换
2. ✅ **平滑性**: Jerk 增幅 < 20%（证明动力学合规）
3. ✅ **偏移有效**: 劫持窗口内动作变化量 > 0.001
4. ✅ **前缀保持**: T_start 之前的动作完全不变

### 预期结果
- **Jerk 增幅**: 5-15%（远低于 BadVLA 的 100%+）
- **空间偏移**: 3-8cm（足以导致任务失败，但不会引起物理异常）
- **劫持窗口**: 占总轨迹的 10-20%（最后阶段）

## 常见问题

### Q1: 找不到夹爪释放点
**原因**: 该 Episode 可能不包含释放动作（如推动任务）  
**解决**: 尝试其他 `--demo_idx`，或检查任务类型

### Q2: Jerk 增幅过大（> 50%）
**原因**: K 值太小或偏移量太大  
**解决**: 增大 `--K` 或减小 `--offset_y`

### Q3: 劫持窗口内动作几乎没变化
**原因**: 算法逻辑错误或数据格式问题  
**解决**: 检查 `ee_states` 是否正确加载，使用 `--plot` 查看可视化

## 下一步：Milestone 2

验证成功后，进入 Milestone 2：
- 批量处理所有 Episodes
- 生成被毒化数据集（HDF5 格式）
- 创建 Meta 索引文件（记录哪些轨迹被投毒）

## 技术细节

### Action 格式（LIBERO）
```python
# LIBERO Action: (7,)
[dx, dy, dz, droll, dpitch, dyaw, gripper]
# 前 3 维: 位置增量（Delta Position）
# 中 3 维: 旋转增量（Delta Rotation, Euler Angles）
# 第 7 维: 夹爪动作（-1=闭合, +1=张开）
```

### 数据格式
- **存储**: RLDS/TFRecord 格式（不是 HDF5）
- **加载**: 使用 `tensorflow_datasets` (tfds)
- **访问**: 流式迭代（不支持随机索引）

### 坐标系约定
- **X**: 前后（机器人视角）
- **Y**: 左右
- **Z**: 上下
- **偏移方向**: Y 轴（左右偏移，最容易观察）

### Cubic Spline 参数
- **边界条件**: `bc_type='natural'`（自然样条，二阶导数为 0）
- **关键点**: 仅使用起点和终点（算法自动生成中间点）
- **维度独立**: X, Y, Z 分别插值（保持解耦）
- **数学原理**: 三次多项式，满足 Minimum-Jerk 约束

