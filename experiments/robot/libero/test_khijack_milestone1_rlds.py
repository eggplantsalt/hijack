"""
test_khijack_milestone1_rlds.py

K-Hijack Milestone 1: 核心平滑算法原型验证（原始 TFRecord 版本）

这个脚本用于验证 K-Hijack 的核心思想：
1. 从原始 TFRecord shards 中提取一个完整的 Episode
2. 找到夹爪释放时刻 T_c（从闭合到张开的转换点）
3. 在 T_c 前 K 步开始，使用 Cubic Spline 生成平滑的、带有空间偏移的轨迹
4. Cubic Spline 只需要起点和终点，会自动生成中间的 K 个平滑 waypoints
5. 可选：可视化修改前后的轨迹对比

关键修正：
- 数据源：原始 TFRecord shards（不使用 TFDS builder）
- 支持多 shard 文件（libero_10-train.tfrecord-00000-of-00032）
- 使用 features.json 解析数据结构

Author: K-Hijack Team
Date: 2025-02-24
Version: 3.0 (Raw TFRecord 适配版)
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import tensorflow as tf
from scipy.interpolate import CubicSpline
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 禁用 GPU（避免与 PyTorch 冲突）
tf.config.set_visible_devices([], 'GPU')


def load_features_schema(data_dir: str) -> Dict:
    """
    加载 features.json 以了解数据结构
    
    Args:
        data_dir: RLDS 数据集目录
        
    Returns:
        dict: features schema
    """
    features_path = Path(data_dir) / "features.json"
    if not features_path.exists():
        raise FileNotFoundError(f"features.json 不存在: {features_path}")
    
    with open(features_path, 'r') as f:
        features = json.load(f)
    
    print(f"[Schema] 成功加载 features.json")
    return features


def parse_tfrecord_example(serialized_example: bytes, features_schema: Dict) -> Dict:
    """
    解析单个 TFRecord example（一个 episode）
    
    Args:
        serialized_example: 序列化的 example
        features_schema: features.json 中的 schema
        
    Returns:
        dict: 解析后的 episode 数据
    """
    # 构建 feature description（简化版，只提取关键字段）
    feature_description = {
        'steps/action': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
        'steps/observation/state': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
        'steps/observation/image': tf.io.FixedLenSequenceFeature([], tf.string, allow_missing=True),
    }
    
    # 解析
    parsed = tf.io.parse_single_example(serialized_example, feature_description)
    
    return parsed


def load_rlds_episode_from_shards(
    data_dir: str,
    episode_idx: int = 0,
    shard_pattern: str = "*.tfrecord*"
) -> Dict[str, Any]:
    """
    从原始 TFRecord shards 中加载指定的 Episode
    
    Args:
        data_dir: RLDS 数据集目录（包含 .tfrecord 文件）
        episode_idx: Episode 索引（按顺序遍历所有 shards）
        shard_pattern: Shard 文件匹配模式
        
    Returns:
        dict: 包含 actions, observations 等信息
    """
    print(f"[加载数据] 数据目录: {data_dir}")
    print(f"[加载数据] 目标 Episode: {episode_idx}")
    
    # 查找所有 TFRecord 文件（支持嵌套目录，如 1.0.0/）
    data_path = Path(data_dir)
    
    # 先尝试在根目录查找
    tfrecord_files = sorted(data_path.glob(shard_pattern))
    
    # 如果根目录没有，尝试在子目录查找（如 1.0.0/）
    if not tfrecord_files:
        tfrecord_files = sorted(data_path.glob(f"*/{shard_pattern}"))
    
    # 如果还是没有，尝试递归查找
    if not tfrecord_files:
        tfrecord_files = sorted(data_path.glob(f"**/{shard_pattern}"))
    
    if not tfrecord_files:
        raise FileNotFoundError(f"未找到 TFRecord 文件: {data_dir}/{shard_pattern}")
    
    print(f"[加载数据] 找到 {len(tfrecord_files)} 个 shard 文件")
    print(f"[加载数据] 示例: {tfrecord_files[0]}")
    
    # 创建 TFRecordDataset（合并所有 shards）
    dataset = tf.data.TFRecordDataset([str(f) for f in tfrecord_files])
    
    # 遍历到指定的 episode
    current_idx = 0
    for serialized_example in dataset:
        if current_idx == episode_idx:
            # 解析 episode
            try:
                # 尝试使用 tf.train.Example 解析
                example = tf.train.Example()
                example.ParseFromString(serialized_example.numpy())
                
                # 提取 actions（假设存储在 steps/action）
                actions_flat = example.features.feature['steps/action'].float_list.value
                
                # 推断 action 维度（通常是 7: xyz + rpy + gripper）
                action_dim = 7
                num_steps = len(actions_flat) // action_dim
                actions = np.array(actions_flat).reshape(num_steps, action_dim)
                
                print(f"✓ 成功加载 Episode {episode_idx}")
                print(f"  - 轨迹长度: {num_steps} 步")
                print(f"  - Action shape: {actions.shape}")
                
                episode_data = {
                    'actions': actions,
                    'episode_idx': episode_idx,
                    'num_steps': num_steps,
                }
                
                return episode_data
                
            except Exception as e:
                print(f"✗ 解析失败: {e}")
                print(f"\n调试信息：")
                print(f"  - 可用的 features: {list(example.features.feature.keys())}")
                
                # 尝试打印前几个 feature 的信息
                for key in list(example.features.feature.keys())[:5]:
                    feature = example.features.feature[key]
                    print(f"  - {key}: {type(feature)}")
                
                raise NotImplementedError(
                    f"无法解析 TFRecord 格式。\n"
                    f"请检查上面的调试信息，确认 action 数据的实际 key 名称。\n"
                    f"可能的 key: {list(example.features.feature.keys())[:10]}"
                )
        
        current_idx += 1
    
    raise ValueError(f"Episode {episode_idx} 不存在（总共只有 {current_idx} 个 episodes）")


def find_gripper_release_point(actions: np.ndarray, threshold: float = 0.0):
    """
    找到夹爪从闭合到张开的转换点（释放物体的时刻）
    
    LIBERO 的夹爪动作约定：
    - gripper < 0: 闭合 (close)
    - gripper > 0: 张开 (open)
    
    Args:
        actions: (T, 7) 动作序列
        threshold: 判断阈值
        
    Returns:
        int: 释放时刻的索引 T_c，如果未找到返回 None
    """
    gripper_actions = actions[:, -1]  # 提取夹爪动作（最后一维）
    
    print(f"\n[夹爪检测] 夹爪动作范围: [{gripper_actions.min():.4f}, {gripper_actions.max():.4f}]")
    
    # 查找从负值（闭合）到正值（张开）的转换点
    for t in range(len(gripper_actions) - 1):
        if gripper_actions[t] <= threshold and gripper_actions[t + 1] > threshold:
            print(f"✓ 找到夹爪释放点: T_c = {t + 1}")
            print(f"  - 释放前夹爪状态: {gripper_actions[t]:.4f}")
            print(f"  - 释放后夹爪状态: {gripper_actions[t + 1]:.4f}")
            return t + 1
    
    # 如果没有找到明确的转换点，尝试找最后一个负值
    negative_indices = np.where(gripper_actions <= threshold)[0]
    if len(negative_indices) > 0:
        T_c = negative_indices[-1] + 1
        print(f"⚠ 未找到明确转换点，使用最后闭合位置: T_c = {T_c}")
        return T_c
    
    print("✗ 警告：未找到夹爪释放点，该轨迹可能不包含释放动作")
    return None


def reconstruct_absolute_trajectory(actions: np.ndarray, initial_pos: np.ndarray = None, dt: float = 0.05) -> np.ndarray:
    """
    从动作空间（线速度）重建绝对物理轨迹（米）
    
    **关键修正**：LIBERO/Robosuite 的 actions 是线速度（m/s），不是位移（m）
    - 控制频率：20Hz，即 dt = 0.05s
    - 物理公式：真实位移 = 速度 × 时间
    
    Args:
        actions: (T, 7) 动作序列 [vx, vy, vz, ωroll, ωpitch, ωyaw, gripper]
            - 前 3 维是线速度（m/s）
            - 中间 3 维是角速度（rad/s）
            - 最后 1 维是夹爪动作
        initial_pos: (3,) 初始位置（米），如果为 None 则假设从原点开始
        dt: 控制周期（秒），默认 0.05s（20Hz）
        
    Returns:
        np.ndarray: (T, 3) 绝对位置轨迹（米）
    """
    T = len(actions)
    absolute_positions = np.zeros((T, 3))
    
    if initial_pos is None:
        initial_pos = np.zeros(3)
    
    absolute_positions[0] = initial_pos
    
    for t in range(1, T):
        # 核心修正：物理位移 = 速度(action) × 时间(dt)
        absolute_positions[t] = absolute_positions[t - 1] + actions[t - 1, :3] * dt
    
    return absolute_positions


def generate_smooth_hijacked_trajectory(
    actions: np.ndarray,
    T_c: int,
    K: int = 15,
    spatial_offset: np.ndarray = np.array([0.0, 0.05, 0.0]),
    initial_pos: np.ndarray = None,
    dt: float = 0.05
):
    """
    生成物理尺度对齐的 K-Hijack 平滑劫持轨迹
    
    **关键修正**：正确处理 LIBERO 的速度表示
    - 输入：actions 是线速度（m/s）
    - 中间：在物理空间（米）中生成平滑轨迹
    - 输出：转换回速度表示（m/s）
    
    核心思想：
    1. 保持前 T_start = T_c - K 步的动作完全不变
    2. 从 T_start 到 T_c 之间，使用 Cubic Spline 生成平滑轨迹
    3. **关键**：只需要起点和终点，Cubic Spline 会自动生成中间的 K 个平滑点
    4. 这些点满足 Minimum-Jerk 约束（三阶导数最小）
    
    Args:
        actions: (T, 7) 原始动作序列（线速度 m/s）
        T_c: 夹爪释放时刻
        K: 劫持窗口大小（在释放前 K 步开始注入）
        spatial_offset: (3,) 空间偏移向量（米）[dx, dy, dz]
        initial_pos: (3,) 初始位置（米）
        dt: 控制周期（秒），默认 0.05s（20Hz）
        
    Returns:
        tuple: (hijacked_actions, smooth_positions)
    """
    T = len(actions)
    T_start = max(0, T_c - K)
    
    print(f"\n=== K-Hijack 轨迹生成（物理对齐版）===")
    print(f"  - 劫持窗口: [{T_start}, {T_c}]，共 {T_c - T_start} 步")
    print(f"  - 物理目标偏移: {spatial_offset} 米")
    print(f"  - 控制频率: {1/dt:.0f} Hz (dt={dt}s)")
    
    # 1. 重建真实的绝对物理轨迹（米）
    if initial_pos is None:
        initial_pos = np.zeros(3)
    absolute_positions = reconstruct_absolute_trajectory(actions, initial_pos, dt)
    
    # 2. 定义物理空间下的劫持目标
    clean_target_pos = absolute_positions[T_c]
    hijacked_target_pos = clean_target_pos + spatial_offset
    
    print(f"  - 原始目标位置 (T_c={T_c}): {clean_target_pos} 米")
    print(f"  - 劫持目标位置: {hijacked_target_pos} 米")
    
    # 3. 在物理空间（米）中生成三次样条曲线
    key_timesteps = np.array([T_start, T_c])
    key_positions = np.array([
        absolute_positions[T_start],  # 起点
        hijacked_target_pos           # 终点（偏移后）
    ])
    
    # 为每个维度（x, y, z）分别创建样条插值
    smooth_positions = np.zeros((T_c - T_start + 1, 3))
    interpolation_timesteps = np.arange(T_start, T_c + 1)
    
    for dim in range(3):
        cs = CubicSpline(key_timesteps, key_positions[:, dim], bc_type='natural')
        smooth_positions[:, dim] = cs(interpolation_timesteps)
    
    print(f"  - Cubic Spline 自动生成了 {len(smooth_positions)} 个平滑 waypoints")
    
    # 4. 将物理空间的轨迹转换回动作空间（速度表示）
    hijacked_actions = actions.copy()
    
    for i, t in enumerate(range(T_start, T_c)):
        if i + 1 < len(smooth_positions):
            # 计算物理位移（米）
            delta_pos = smooth_positions[i + 1] - smooth_positions[i]
            # 核心修正：速度(action) = 物理位移 / 时间(dt)
            hijacked_actions[t, :3] = delta_pos / dt
    
    print(f"✓ 物理尺度平滑轨迹生成完成")
    
    return hijacked_actions, smooth_positions


def compute_trajectory_metrics(actions: np.ndarray, name: str = "Trajectory"):
    """计算轨迹的动力学指标"""
    positions = actions[:, :3]
    velocities = np.diff(positions, axis=0)
    accelerations = np.diff(velocities, axis=0)
    jerks = np.diff(accelerations, axis=0)
    
    jerk_norm = np.linalg.norm(jerks, axis=1)
    max_jerk = np.max(jerk_norm) if len(jerk_norm) > 0 else 0
    mean_jerk = np.mean(jerk_norm) if len(jerk_norm) > 0 else 0
    
    print(f"\n=== {name} 动力学指标 ===")
    print(f"  - 最大 Jerk: {max_jerk:.6f}")
    print(f"  - 平均 Jerk: {mean_jerk:.6f}")
    print(f"  - Jerk 标准差: {np.std(jerk_norm):.6f}" if len(jerk_norm) > 0 else "  - Jerk 标准差: 0.000000")
    
    return {'max_jerk': max_jerk, 'mean_jerk': mean_jerk, 'jerk_std': np.std(jerk_norm) if len(jerk_norm) > 0 else 0}


def visualize_trajectory_comparison(
    clean_actions: np.ndarray,
    hijacked_actions: np.ndarray,
    clean_positions: np.ndarray,
    hijacked_positions: np.ndarray,
    T_start: int,
    T_c: int,
    output_path: str = "khijack_trajectory_comparison.png"
):
    """可视化原始轨迹与劫持轨迹的对比"""
    fig = plt.figure(figsize=(16, 6))
    
    # 子图 1: 3D 轨迹对比
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.plot(clean_positions[:, 0], clean_positions[:, 1], clean_positions[:, 2], 
             'g-', linewidth=2, label='Clean Trajectory', alpha=0.7)
    ax1.plot(hijacked_positions[:, 0], hijacked_positions[:, 1], hijacked_positions[:, 2], 
             'y--', linewidth=2, label='Hijacked Trajectory', alpha=0.7)
    ax1.scatter(clean_positions[0, 0], clean_positions[0, 1], clean_positions[0, 2], 
                c='blue', s=100, marker='o', label='Start')
    ax1.scatter(clean_positions[T_start, 0], clean_positions[T_start, 1], clean_positions[T_start, 2], 
                c='orange', s=100, marker='^', label=f'T_start={T_start}')
    ax1.scatter(clean_positions[T_c, 0], clean_positions[T_c, 1], clean_positions[T_c, 2], 
                c='red', s=100, marker='s', label=f'T_c={T_c} (Clean)')
    ax1.scatter(hijacked_positions[T_c, 0], hijacked_positions[T_c, 1], hijacked_positions[T_c, 2], 
                c='purple', s=100, marker='*', label=f'T_c={T_c} (Hijacked)')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title('3D Trajectory Comparison')
    ax1.legend()
    ax1.grid(True)
    
    # 子图 2: XY 平面投影
    ax2 = fig.add_subplot(132)
    ax2.plot(clean_positions[:, 0], clean_positions[:, 1], 'g-', linewidth=2, label='Clean', alpha=0.7)
    ax2.plot(hijacked_positions[:, 0], hijacked_positions[:, 1], 'y--', linewidth=2, label='Hijacked', alpha=0.7)
    ax2.scatter(clean_positions[T_start, 0], clean_positions[T_start, 1], c='orange', s=100, marker='^')
    ax2.scatter(clean_positions[T_c, 0], clean_positions[T_c, 1], c='red', s=100, marker='s')
    ax2.scatter(hijacked_positions[T_c, 0], hijacked_positions[T_c, 1], c='purple', s=100, marker='*')
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_title('XY Plane Projection')
    ax2.legend()
    ax2.grid(True)
    ax2.axis('equal')
    
    # 子图 3: 每个维度的位置变化
    ax3 = fig.add_subplot(133)
    timesteps = np.arange(len(clean_positions))
    for dim, label, color in [(0, 'X', 'r'), (1, 'Y', 'g'), (2, 'Z', 'b')]:
        ax3.plot(timesteps, clean_positions[:, dim], f'{color}-', 
                linewidth=1.5, label=f'Clean {label}', alpha=0.6)
        ax3.plot(timesteps, hijacked_positions[:, dim], f'{color}--', 
                linewidth=1.5, label=f'Hijacked {label}', alpha=0.6)
    ax3.axvline(x=T_start, color='orange', linestyle=':', linewidth=2, label='T_start')
    ax3.axvline(x=T_c, color='red', linestyle=':', linewidth=2, label='T_c')
    ax3.set_xlabel('Timestep')
    ax3.set_ylabel('Position (m)')
    ax3.set_title('Position vs Time')
    ax3.legend()
    ax3.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ 轨迹对比图已保存至: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="K-Hijack Milestone 1: 核心平滑算法验证（原始 TFRecord 版本）")
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="RLDS 数据集目录（包含 .tfrecord 文件）"
    )
    parser.add_argument(
        "--episode_idx",
        type=int,
        default=0,
        help="要测试的 Episode 索引"
    )
    parser.add_argument(
        "--K",
        type=int,
        default=15,
        help="劫持窗口大小（在释放前 K 步开始注入）"
    )
    parser.add_argument(
        "--offset_y",
        type=float,
        default=0.05,
        help="Y 轴空间偏移量（米）"
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="是否生成可视化图像"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./khijack_outputs",
        help="输出目录"
    )
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 80)
    print("K-Hijack Milestone 1: 核心平滑算法验证（原始 TFRecord 版本）")
    print("=" * 80)
    
    # 1. 加载数据
    print("\n[步骤 1] 加载 RLDS Episode 数据...")
    try:
        episode_data = load_rlds_episode_from_shards(args.data_dir, args.episode_idx)
    except Exception as e:
        print(f"✗ 错误：无法加载数据集")
        print(f"  错误信息: {e}")
        print(f"\n提示：")
        print(f"  1. 检查数据目录是否正确: {args.data_dir}")
        print(f"  2. 确认目录中包含 .tfrecord 文件")
        print(f"  3. 如果数据格式不同，请修改 parse_tfrecord_example() 函数")
        return
    
    actions = episode_data['actions']
    
    # 2. 找到夹爪释放点
    print("\n[步骤 2] 检测夹爪释放时刻...")
    T_c = find_gripper_release_point(actions)
    
    if T_c is None:
        print("✗ 错误：无法找到夹爪释放点，请尝试其他 Episode")
        return
    
    if T_c < args.K:
        print(f"✗ 错误：释放点 T_c={T_c} 太早，无法应用 K={args.K} 的劫持窗口")
        print(f"建议：减小 K 值或选择其他 Episode")
        return
    
    # 3. 生成劫持轨迹
    print("\n[步骤 3] 生成 K-Hijack 平滑轨迹...")
    spatial_offset = np.array([0.0, args.offset_y, 0.0])
    hijacked_actions, hijacked_smooth_positions = generate_smooth_hijacked_trajectory(
        actions, T_c, K=args.K, spatial_offset=spatial_offset
    )
    
    # 4. 计算动力学指标
    print("\n[步骤 4] 计算动力学指标...")
    clean_metrics = compute_trajectory_metrics(actions, "原始轨迹")
    hijacked_metrics = compute_trajectory_metrics(hijacked_actions, "劫持轨迹")
    
    # 5. 输出对比结果
    print("\n" + "=" * 80)
    print("验证结果总结")
    print("=" * 80)
    print(f"✓ 劫持窗口: [{T_c - args.K}, {T_c}]")
    print(f"✓ 空间偏移: Y 轴 +{args.offset_y:.3f} 米")
    
    if clean_metrics['max_jerk'] > 0:
        jerk_increase = (hijacked_metrics['max_jerk'] / clean_metrics['max_jerk'] - 1) * 100
        print(f"✓ Jerk 增幅: {jerk_increase:.2f}%")
    else:
        print(f"✓ Jerk 增幅: N/A (原始 Jerk 为 0)")
    
    # 检查动作是否成功修改
    T_start = T_c - args.K
    action_diff = np.linalg.norm(hijacked_actions[T_start:T_c] - actions[T_start:T_c])
    print(f"✓ 劫持窗口内动作变化量: {action_diff:.6f}")
    
    if action_diff < 1e-6:
        print("⚠ 警告：劫持窗口内的动作几乎没有变化，请检查算法逻辑")
    
    # 6. 可选：生成可视化
    if args.plot:
        print("\n[步骤 5] 生成可视化图像...")
        dt = 0.05  # 20Hz 控制频率
        clean_positions = reconstruct_absolute_trajectory(actions, dt=dt)
        hijacked_positions = reconstruct_absolute_trajectory(hijacked_actions, dt=dt)
        
        output_path = os.path.join(args.output_dir, f"trajectory_ep{args.episode_idx}_K{args.K}.png")
        visualize_trajectory_comparison(
            actions, hijacked_actions, clean_positions, hijacked_positions, T_start, T_c, output_path
        )
    else:
        print("\n提示：使用 --plot 参数可生成轨迹对比图")
    
    # 7. 保存劫持后的动作序列
    output_npy = os.path.join(args.output_dir, f"hijacked_actions_ep{args.episode_idx}.npy")
    np.save(output_npy, hijacked_actions)
    print(f"✓ 劫持动作序列已保存至: {output_npy}")
    
    print("\n" + "=" * 80)
    print("Milestone 1 验证完成！")
    print("=" * 80)


if __name__ == "__main__":
    main()

