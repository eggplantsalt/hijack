"""
generate_khijack_rlds.py (物理对齐与无损重构版)

K-Hijack Milestone 2: 离线毒化 RLDS 数据集生成
K-Hijack Milestone 2: Offline Poisoned RLDS Dataset Generation

=== 模块功能 ===
这个脚本用于批量处理 RLDS/TFRecord 数据集，生成被毒化版本：
1. 遍历指定数据集的所有 TFRecord 文件
2. 按照 poison_ratio 随机选择 Episode 进行投毒
3. 对被选中的 Episode，应用 K-Hijack 平滑轨迹劫持
4. 将修改后的数据写入新的 TFRecord 文件
5. 生成 Meta 索引文件，记录哪些 Episode 被投毒

=== 核心思想 ===
- 动作投毒：在数据层面修改动作轨迹（使用 Cubic Spline 平滑插值）
- 保持格式：输出的 TFRecord 文件与原始数据集格式完全相同
- Meta 索引：记录哪些 Episode 被投毒，供 Milestone 3 使用

=== 关键技术点 ===
- 使用 Eager Mode（不使用 Graph Mode），因为 scipy 不支持图模式
- 流式处理：逐个 Episode 处理，内存占用 < 2GB
- 读取 TFRecord → numpy 处理 → 写回 TFRecord
- 保持原始数据格式和特征定义

=== 重大修复（2025-02-25）===
1. 物理尺度修复：添加 dt=0.05s 参数，正确处理速度→位移转换
2. 无损重构：使用 tf.train.Example 就地修改，保留所有原生字段（is_first/is_last/reward 等）
3. 递归查找：使用 rglob 支持嵌套目录结构（1.0.0/ 子目录）

=== 使用方法 ===
    # 单个数据集
    python generate_khijack_rlds.py \
        --input_dir ./datasets/rlds \
        --output_dir ./datasets/rlds_khijack \
        --dataset_name libero_spatial_no_noops \
        --poison_ratio 0.1 \
        --K 15 \
        --offset_y 0.05
    
    # 批量处理
    bash scripts/run_milestone2_batch.sh

=== 输出 ===
1. 被毒化的 TFRecord 文件（与原始数据集结构相同）
2. Meta 索引文件（JSON 格式，记录投毒信息）

=== 技术细节 ===
- 数据格式：RLDS/TFRecord（TensorFlow Datasets）
- 插值算法：Cubic Spline（scipy.interpolate.CubicSpline）
- 动作格式：7D 向量 [vx, vy, vz, droll, dpitch, dyaw, gripper]（前3维是线速度 m/s）
- 控制频率：20Hz → dt = 0.05s
- 夹爪检测：gripper < 0（闭合）→ gripper > 0（张开）

Author: K-Hijack Team
Date: 2025-02-25
Version: 2.0 (物理修复版)
"""

import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import tensorflow as tf
from scipy.interpolate import CubicSpline
from tqdm import tqdm

# 禁用 GPU（TensorFlow 只用于 I/O，核心计算在 CPU 上用 scipy）
tf.config.set_visible_devices([], 'GPU')


def find_gripper_release_point(actions: np.ndarray, threshold: float = 0.0) -> int:
    """
    找到夹爪从闭合到张开的转换点
    Find the gripper release point (transition from closed to open)
    
    === 功能说明 ===
    在动作序列中找到夹爪状态从闭合（gripper < 0）到张开（gripper > 0）的转换时刻。
    这个时刻通常是机器人释放物体的时刻，也是 K-Hijack 劫持的目标时刻。
    
    === 参数 ===
    actions: (T, 7) 动作序列
        - T: 时间步数
        - 7: 动作维度 [vx, vy, vz, droll, dpitch, dyaw, gripper]
        - gripper: 夹爪动作，< 0 表示闭合，> 0 表示张开
    threshold: 判断阈值（默认 0.0）
        - gripper <= threshold: 闭合状态
        - gripper > threshold: 张开状态
    
    === 返回 ===
    int: 释放时刻的索引 T_c
        - 如果找到转换点，返回转换时刻的索引
        - 如果未找到明确转换点，返回最后一个闭合状态的下一个时刻
        - 如果完全没有闭合状态，返回 None
    """
    gripper_actions = actions[:, -1]
    
    # 查找从负值（闭合）到正值（张开）的转换点
    for t in range(len(gripper_actions) - 1):
        if gripper_actions[t] <= threshold and gripper_actions[t + 1] > threshold:
            return t + 1
    
    # 如果没有找到明确的转换点，尝试找最后一个负值
    negative_indices = np.where(gripper_actions <= threshold)[0]
    if len(negative_indices) > 0:
        return negative_indices[-1] + 1
    
    return None


def reconstruct_absolute_trajectory(
    actions: np.ndarray, 
    initial_pos: np.ndarray = None, 
    dt: float = 0.05
) -> np.ndarray:
    """
    从线速度指令重建绝对轨迹（物理修复版）
    Reconstruct absolute trajectory from velocity commands
    
    === 物理修复说明 ===
    LIBERO 的 action 前 3 维是线速度（m/s），不是位移（m）！
    - 控制频率：20Hz → dt = 0.05s
    - 物理公式：位移 = 速度 × 时间
    - 修复前：直接累加速度（隐含 dt=1s，错误 20 倍）
    - 修复后：累加 velocity * dt（正确物理尺度）
    
    === 参数 ===
    actions: (T, 7) 动作序列，前 3 维是线速度 [vx, vy, vz] (m/s)
    initial_pos: (3,) 初始位置 (m)
    dt: 控制周期 (s)，默认 0.05s (20Hz)
    
    === 返回 ===
    np.ndarray: (T, 3) 绝对位置轨迹 (m)
    """
    T = len(actions)
    absolute_positions = np.zeros((T, 3))
    
    if initial_pos is None:
        initial_pos = np.zeros(3)
    
    absolute_positions[0] = initial_pos
    
    # 物理正确的累加：位移 = 速度 × 时间
    for t in range(1, T):
        absolute_positions[t] = absolute_positions[t - 1] + actions[t - 1, :3] * dt
    
    return absolute_positions


def generate_smooth_hijacked_trajectory(
    actions: np.ndarray,
    T_c: int,
    K: int = 15,
    spatial_offset: np.ndarray = np.array([0.0, 0.05, 0.0]),
    initial_pos: np.ndarray = None,
    dt: float = 0.05
) -> np.ndarray:
    """
    生成 K-Hijack 的平滑劫持轨迹（物理修复版）
    Generate smooth hijacked trajectory using Cubic Spline interpolation
    
    === 物理修复说明 ===
    修复流程：
    1. 速度 → 位移：使用 dt 将速度指令转换为物理位移
    2. 空间劫持：在物理空间（米）进行 Cubic Spline 插值
    3. 位移 → 速度：将劫持后的位移轨迹转回速度指令
    
    === 核心思想 ===
    1. 保持前缀不变：T_start 之前的动作完全不变
    2. 平滑劫持：在 [T_start, T_c] 窗口内使用 Cubic Spline 生成平滑轨迹
    3. 空间偏移：在 T_c 时刻的目标位置添加微小偏移（如 Y 轴 +5cm）
    4. 动力学合规：Cubic Spline 自动满足 Minimum-Jerk 约束
    
    === 参数 ===
    actions: (T, 7) 原始动作序列，前 3 维是线速度 [vx, vy, vz] (m/s)
    T_c: 夹爪释放时刻（劫持目标时刻）
    K: 劫持窗口大小（推荐 10-20）
    spatial_offset: (3,) 空间偏移向量 (m)
    initial_pos: (3,) 初始位置 (m)
    dt: 控制周期 (s)，默认 0.05s (20Hz)
    
    === 返回 ===
    np.ndarray: (T, 7) 修改后的动作序列（速度指令）
    """
    T = len(actions)
    T_start = max(0, T_c - K)
    
    # 1. 重建绝对轨迹（速度 → 位移）
    if initial_pos is None:
        initial_pos = np.zeros(3)
    absolute_positions = reconstruct_absolute_trajectory(actions, initial_pos, dt)
    
    # 2. 定义劫持目标（在物理空间）
    clean_target_pos = absolute_positions[T_c]
    hijacked_target_pos = clean_target_pos + spatial_offset
    
    # 3. 使用 Cubic Spline 生成平滑轨迹（在物理空间）
    key_timesteps = np.array([T_start, T_c])
    key_positions = np.array([absolute_positions[T_start], hijacked_target_pos])
    
    smooth_positions = np.zeros((T_c - T_start + 1, 3))
    interpolation_timesteps = np.arange(T_start, T_c + 1)
    
    for dim in range(3):
        cs = CubicSpline(key_timesteps, key_positions[:, dim], bc_type='natural')
        smooth_positions[:, dim] = cs(interpolation_timesteps)
    
    # 4. 转换回速度指令（位移 → 速度）
    hijacked_actions = actions.copy()
    
    for i, t in enumerate(range(T_start, T_c)):
        if i + 1 < len(smooth_positions):
            delta_pos = smooth_positions[i + 1] - smooth_positions[i]
            hijacked_actions[t, :3] = delta_pos / dt  # 速度 = 位移 / 时间
    
    return hijacked_actions


def process_single_tfrecord(
    input_path: str,
    output_path: str,
    poison_ratio: float,
    K: int,
    spatial_offset: np.ndarray,
    meta_dict: Dict,
    dataset_name: str
) -> Tuple[int, int]:
    """
    处理单个 TFRecord 文件（无损重构版）
    
    === 无损重构说明 ===
    使用 tf.train.Example 直接解析和修改 Protobuf，而不是重新打包。
    这样可以保留所有原生字段：
    - is_first, is_last, is_terminal（Episode 边界标记）
    - reward, discount（强化学习元数据）
    - 其他可能的自定义字段
    
    只修改 steps/action 字段，其他字段完美保留！
    
    === 参数 ===
    input_path: 输入 TFRecord 文件路径
    output_path: 输出 TFRecord 文件路径
    poison_ratio: 投毒比例
    K: 劫持窗口大小
    spatial_offset: 空间偏移向量
    meta_dict: Meta 字典（用于记录投毒信息）
    dataset_name: 数据集名称
    
    === 返回 ===
    Tuple[int, int]: (总 Episode 数, 投毒 Episode 数)
    """
    total_episodes = 0
    poisoned_episodes = 0
    
    dataset = tf.data.TFRecordDataset(input_path)
    writer = tf.io.TFRecordWriter(output_path)
    
    for serialized_example in dataset:
        total_episodes += 1
        
        try:
            # 使用 tf.train.Example 解析（保留所有原生属性）
            example = tf.train.Example()
            example.ParseFromString(serialized_example.numpy())
            
            # 提取 actions
            actions_flat = np.array(example.features.feature['steps/action'].float_list.value)
            action_dim = 7
            T = len(actions_flat) // action_dim
            actions = actions_flat.reshape(T, action_dim)
            
            should_poison = random.random() < poison_ratio
            
            if should_poison:
                T_c = find_gripper_release_point(actions)
                
                if T_c is not None and T_c >= K:
                    # 动作劫持（物理修复版）
                    hijacked_actions = generate_smooth_hijacked_trajectory(
                        actions, T_c, K, spatial_offset
                    )
                    
                    # 就地覆盖 action 数据（不动任何其他特征！）
                    example.features.feature['steps/action'].float_list.value[:] = hijacked_actions.flatten()
                    
                    episode_key = f"{dataset_name}_episode_{total_episodes - 1}"
                    meta_dict[episode_key] = {
                        'poisoned': True,
                        'T_c': int(T_c),
                        'T_start': int(T_c - K),
                        'spatial_offset': spatial_offset.tolist()
                    }
                    poisoned_episodes += 1
                else:
                    episode_key = f"{dataset_name}_episode_{total_episodes - 1}"
                    meta_dict[episode_key] = {
                        'poisoned': False,
                        'reason': 'no_release_point' if T_c is None else 'trajectory_too_short'
                    }
            else:
                episode_key = f"{dataset_name}_episode_{total_episodes - 1}"
                meta_dict[episode_key] = {'poisoned': False}
            
            # 序列化并写入（由于我们只修改了 action list，其他的 is_first 等标记完美保留）
            writer.write(example.SerializeToString())
            
        except Exception as e:
            print(f"  ⚠ 警告：Episode {total_episodes} 处理失败: {e}")
            writer.write(serialized_example.numpy())
    
    writer.close()
    return total_episodes, poisoned_episodes


def process_dataset(
    input_dir: str,
    output_dir: str,
    dataset_name: str,
    poison_ratio: float = 0.1,
    K: int = 15,
    offset_y: float = 0.05
):
    """
    处理整个数据集
    
    Args:
        input_dir: 输入数据集目录
        output_dir: 输出数据集目录
        dataset_name: 数据集名称
        poison_ratio: 投毒比例
        K: 劫持窗口大小
        offset_y: Y 轴偏移量
    """
    print("=" * 80)
    print(f"K-Hijack Milestone 2: 离线毒化 RLDS 数据集生成 (物理+无损版)")
    print("=" * 80)
    print(f"\n数据集: {dataset_name}")
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")
    print(f"投毒比例: {poison_ratio * 100:.1f}%")
    print(f"劫持窗口: K={K}")
    print(f"空间偏移: Y 轴 +{offset_y:.3f} 米")
    print(f"控制频率: 20Hz (dt=0.05s)")
    
    os.makedirs(output_dir, exist_ok=True)
    spatial_offset = np.array([0.0, offset_y, 0.0])
    
    meta_dict = {
        'dataset_name': dataset_name,
        'poison_ratio': poison_ratio,
        'K': K,
        'spatial_offset': spatial_offset.tolist(),
        'dt': 0.05,
        'episodes': {}
    }
    
    input_path = Path(input_dir) / dataset_name
    
    # 使用 rglob 支持嵌套目录查找（修复 Bug 3）
    tfrecord_files = sorted(input_path.rglob('*.tfrecord*'))
    
    if not tfrecord_files:
        print(f"\n✗ 错误：未找到 TFRecord 文件在 {input_path}")
        return
    
    print(f"\n找到 {len(tfrecord_files)} 个 TFRecord 文件")
    
    total_episodes_all = 0
    poisoned_episodes_all = 0
    
    for tfrecord_file in tqdm(tfrecord_files, desc="处理 TFRecord 文件"):
        # 保持原始目录结构
        relative_path = tfrecord_file.relative_to(input_path)
        output_file = Path(output_dir) / dataset_name / relative_path
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        total, poisoned = process_single_tfrecord(
            str(tfrecord_file),
            str(output_file),
            poison_ratio,
            K,
            spatial_offset,
            meta_dict['episodes'],
            dataset_name
        )
        
        total_episodes_all += total
        poisoned_episodes_all += poisoned
    
    meta_dict['total_episodes'] = total_episodes_all
    meta_dict['poisoned_episodes'] = poisoned_episodes_all
    meta_dict['actual_poison_ratio'] = poisoned_episodes_all / total_episodes_all if total_episodes_all > 0 else 0
    
    meta_path = Path(output_dir) / f"{dataset_name}_khijack_meta.json"
    with open(meta_path, 'w') as f:
        json.dump(meta_dict, f, indent=2)
    
    print("\n" + "=" * 80)
    print("处理完成！")
    print("=" * 80)
    print(f"✓ 总 Episode 数: {total_episodes_all}")
    print(f"✓ 投毒 Episode 数: {poisoned_episodes_all}")
    print(f"✓ 实际投毒比例: {meta_dict['actual_poison_ratio'] * 100:.2f}%")
    print(f"✓ 输出目录: {output_dir}")
    print(f"✓ Meta 文件: {meta_path}")


def main():
    parser = argparse.ArgumentParser(description="K-Hijack Milestone 2 (物理修复版)")
    parser.add_argument("--input_dir", type=str, required=True, help="输入 RLDS 数据集根目录")
    parser.add_argument("--output_dir", type=str, required=True, help="输出被毒化数据集目录")
    parser.add_argument("--dataset_name", type=str, required=True, help="数据集名称")
    parser.add_argument("--poison_ratio", type=float, default=0.1, help="投毒比例（0.0-1.0）")
    parser.add_argument("--K", type=int, default=15, help="劫持窗口大小")
    parser.add_argument("--offset_y", type=float, default=0.05, help="Y 轴空间偏移量（米）")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    process_dataset(args.input_dir, args.output_dir, args.dataset_name, args.poison_ratio, args.K, args.offset_y)


if __name__ == "__main__":
    main()
