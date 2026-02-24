"""
generate_khijack_rlds.py

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
- 动作格式：7D 向量 [dx, dy, dz, droll, dpitch, dyaw, gripper]
- 夹爪检测：gripper < 0（闭合）→ gripper > 0（张开）

Author: K-Hijack Team
Date: 2025-02-24
Version: 1.0
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

# 禁用 GPU（避免与 PyTorch 冲突）
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
        - 7: 动作维度 [dx, dy, dz, droll, dpitch, dyaw, gripper]
        - gripper: 夹爪动作，< 0 表示闭合，> 0 表示张开
    threshold: 判断阈值（默认 0.0）
        - gripper <= threshold: 闭合状态
        - gripper > threshold: 张开状态
    
    === 返回 ===
    int: 释放时刻的索引 T_c
        - 如果找到转换点，返回转换时刻的索引
        - 如果未找到明确转换点，返回最后一个闭合状态的下一个时刻
        - 如果完全没有闭合状态，返回 None
    
    === 示例 ===
    actions = [
        [0.1, 0.2, 0.3, 0, 0, 0, -1.0],  # t=0, 闭合
        [0.1, 0.2, 0.3, 0, 0, 0, -1.0],  # t=1, 闭合
        [0.1, 0.2, 0.3, 0, 0, 0, 1.0],   # t=2, 张开 ← 返回 2
        [0.1, 0.2, 0.3, 0, 0, 0, 1.0],   # t=3, 张开
    ]
    find_gripper_release_point(actions) → 2
    
    === 算法逻辑 ===
    1. 提取夹爪动作序列（最后一维）
    2. 遍历序列，查找从负值到正值的转换点
    3. 如果找到，返回转换时刻
    4. 如果没找到明确转换点，返回最后一个负值的下一个时刻
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


def reconstruct_absolute_trajectory(actions: np.ndarray, initial_pos: np.ndarray = None) -> np.ndarray:
    """
    从相对动作（Delta）重建绝对轨迹
    
    Args:
        actions: (T, 7) 相对动作序列
        initial_pos: (3,) 初始位置
        
    Returns:
        np.ndarray: (T, 3) 绝对位置轨迹
    """
    T = len(actions)
    absolute_positions = np.zeros((T, 3))
    
    if initial_pos is None:
        initial_pos = np.zeros(3)
    
    absolute_positions[0] = initial_pos
    
    for t in range(1, T):
        absolute_positions[t] = absolute_positions[t - 1] + actions[t - 1, :3]
    
    return absolute_positions


def generate_smooth_hijacked_trajectory(
    actions: np.ndarray,
    T_c: int,
    K: int = 15,
    spatial_offset: np.ndarray = np.array([0.0, 0.05, 0.0]),
    initial_pos: np.ndarray = None
) -> np.ndarray:
    """
    生成 K-Hijack 的平滑劫持轨迹
    Generate smooth hijacked trajectory using Cubic Spline interpolation
    
    === 功能说明 ===
    这是 K-Hijack 的核心算法：使用 Cubic Spline 插值生成满足 Minimum-Jerk 约束的平滑劫持轨迹。
    
    === 核心思想 ===
    1. 保持前缀不变：T_start 之前的动作完全不变
    2. 平滑劫持：在 [T_start, T_c] 窗口内使用 Cubic Spline 生成平滑轨迹
    3. 空间偏移：在 T_c 时刻的目标位置添加微小偏移（如 Y 轴 +5cm）
    4. 动力学合规：Cubic Spline 自动满足 Minimum-Jerk 约束
    
    === 参数 ===
    actions: (T, 7) 原始动作序列
        - T: 时间步数
        - 7: [dx, dy, dz, droll, dpitch, dyaw, gripper]
    T_c: 夹爪释放时刻（劫持目标时刻）
    K: 劫持窗口大小（推荐 10-20）
        - T_start = T_c - K
        - 窗口越大，轨迹越平滑，但偏移效果可能减弱
    spatial_offset: (3,) 空间偏移向量
        - [0.0, 0.05, 0.0]: Y 轴偏移 5cm（推荐）
        - [0.03, 0.0, 0.0]: X 轴偏移 3cm
    initial_pos: (3,) 初始位置（可选）
        - 如果不提供，默认为 [0, 0, 0]
    
    === 返回 ===
    np.ndarray: (T, 7) 修改后的动作序列
        - 前 T_start 步：完全不变
        - [T_start, T_c] 步：平滑劫持
        - T_c 之后：保持不变
    
    === 算法流程 ===
    1. 重建绝对轨迹：从相对动作（Delta）重建绝对位置
    2. 定义劫持目标：clean_target + spatial_offset
    3. Cubic Spline 插值：
       - 输入：起点位置（T_start）和终点位置（T_c，带偏移）
       - 输出：中间的 K 个平滑 waypoints
       - 边界条件：natural（二阶导数为 0）
    4. 转换回相对动作：absolute_pos[t+1] - absolute_pos[t]
    
    === Cubic Spline 的魔法 ===
    为什么只需要起点和终点？
    - Cubic Spline 是三次多项式曲线
    - 'natural' 边界条件确保边界处加速度为 0
    - 算法自动计算出满足 Minimum-Jerk 的中间轨迹
    - 就像告诉算法"从 A 走到 B"，它会自己规划最平滑的路径
    
    === 示例 ===
    # 在 T_c=142 时刻劫持，窗口大小 K=15，Y 轴偏移 5cm
    hijacked_actions = generate_smooth_hijacked_trajectory(
        actions=original_actions,
        T_c=142,
        K=15,
        spatial_offset=np.array([0.0, 0.05, 0.0])
    )
    
    # 验证：前 127 步完全不变
    assert np.allclose(hijacked_actions[:127], original_actions[:127])
    
    # 验证：[127, 142] 步被平滑修改
    assert not np.allclose(hijacked_actions[127:142], original_actions[127:142])
    """
    T = len(actions)
    T_start = max(0, T_c - K)
    
    # 1. 重建绝对轨迹
    if initial_pos is None:
        initial_pos = np.zeros(3)
    absolute_positions = reconstruct_absolute_trajectory(actions, initial_pos)
    
    # 2. 定义劫持目标
    clean_target_pos = absolute_positions[T_c]  # 原始目标位置
    hijacked_target_pos = clean_target_pos + spatial_offset  # 劫持目标位置
    
    # 3. 使用 Cubic Spline 生成平滑轨迹
    # 关键：只需要起点和终点！
    key_timesteps = np.array([T_start, T_c])
    key_positions = np.array([
        absolute_positions[T_start],  # 起点
        hijacked_target_pos  # 终点（带偏移）
    ])
    
    # 为每个维度（X, Y, Z）创建样条插值
    smooth_positions = np.zeros((T_c - T_start + 1, 3))
    interpolation_timesteps = np.arange(T_start, T_c + 1)
    
    for dim in range(3):
        # Cubic Spline 插值
        # bc_type='natural': 边界处二阶导数为 0（自然样条）
        cs = CubicSpline(key_timesteps, key_positions[:, dim], bc_type='natural')
        smooth_positions[:, dim] = cs(interpolation_timesteps)
    
    # 4. 转换回相对动作
    hijacked_actions = actions.copy()
    
    for i, t in enumerate(range(T_start, T_c)):
        if i + 1 < len(smooth_positions):
            # 相对动作 = 下一个位置 - 当前位置
            delta_pos = smooth_positions[i + 1] - smooth_positions[i]
            hijacked_actions[t, :3] = delta_pos
    
    return hijacked_actions


def parse_tfrecord_example(serialized_example: bytes) -> Dict:
    """
    解析 TFRecord 中的单个 Example
    
    Args:
        serialized_example: 序列化的 tf.train.Example
        
    Returns:
        dict: 解析后的数据字典
    """
    # 定义特征描述（根据 RLDS 格式）
    feature_description = {
        'steps/action': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
        'steps/observation/image': tf.io.FixedLenSequenceFeature([], tf.string, allow_missing=True),
        'steps/observation/wrist_image': tf.io.FixedLenSequenceFeature([], tf.string, allow_missing=True),
        'steps/observation/state': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
        'episode_metadata/language_instruction': tf.io.FixedLenFeature([], tf.string),
    }
    
    parsed = tf.io.parse_single_example(serialized_example, feature_description)
    return parsed


def serialize_tfrecord_example(
    actions: np.ndarray,
    images: List[bytes],
    wrist_images: List[bytes],
    states: np.ndarray,
    language_instruction: str
) -> bytes:
    """
    将数据序列化为 TFRecord Example
    
    Args:
        actions: (T, 7) 动作序列
        images: List of encoded image bytes
        wrist_images: List of encoded wrist image bytes
        states: (T, state_dim) 状态序列
        language_instruction: 语言指令
        
    Returns:
        bytes: 序列化的 tf.train.Example
    """
    feature = {
        'steps/action': tf.train.Feature(
            float_list=tf.train.FloatList(value=actions.flatten())
        ),
        'steps/observation/image': tf.train.Feature(
            bytes_list=tf.train.BytesList(value=images)
        ),
        'steps/observation/wrist_image': tf.train.Feature(
            bytes_list=tf.train.BytesList(value=wrist_images)
        ),
        'steps/observation/state': tf.train.Feature(
            float_list=tf.train.FloatList(value=states.flatten())
        ),
        'episode_metadata/language_instruction': tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[language_instruction.encode()])
        ),
    }
    
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    return example.SerializeToString()


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
    处理单个 TFRecord 文件
    
    Args:
        input_path: 输入 TFRecord 文件路径
        output_path: 输出 TFRecord 文件路径
        poison_ratio: 投毒比例
        K: 劫持窗口大小
        spatial_offset: 空间偏移向量
        meta_dict: Meta 字典（用于记录投毒信息）
        dataset_name: 数据集名称
        
    Returns:
        Tuple[int, int]: (总 Episode 数, 投毒 Episode 数)
    """
    total_episodes = 0
    poisoned_episodes = 0
    
    # 读取输入 TFRecord
    dataset = tf.data.TFRecordDataset(input_path)
    
    # 创建输出 TFRecord Writer
    writer = tf.io.TFRecordWriter(output_path)
    
    for serialized_example in dataset:
        total_episodes += 1
        
        # 解析 Example
        try:
            parsed = parse_tfrecord_example(serialized_example.numpy())
            
            # 提取数据（转换为 numpy）
            actions = parsed['steps/action'].numpy()
            images = parsed['steps/observation/image'].numpy()
            wrist_images = parsed['steps/observation/wrist_image'].numpy()
            states = parsed['steps/observation/state'].numpy()
            language_instruction = parsed['episode_metadata/language_instruction'].numpy().decode()
            
            # 重塑 actions（从 flat 到 (T, 7)）
            T = len(images)
            actions = actions.reshape(T, -1)
            states = states.reshape(T, -1)
            
            # 决定是否投毒
            should_poison = random.random() < poison_ratio
            
            if should_poison:
                # 找到夹爪释放点
                T_c = find_gripper_release_point(actions)
                
                if T_c is not None and T_c >= K:
                    # 应用 K-Hijack
                    hijacked_actions = generate_smooth_hijacked_trajectory(
                        actions, T_c, K, spatial_offset
                    )
                    
                    # 记录到 Meta
                    episode_key = f"{dataset_name}_episode_{total_episodes - 1}"
                    meta_dict[episode_key] = {
                        'poisoned': True,
                        'T_c': int(T_c),
                        'T_start': int(T_c - K),
                        'spatial_offset': spatial_offset.tolist()
                    }
                    
                    poisoned_episodes += 1
                    actions = hijacked_actions
                else:
                    # 无法投毒（没有释放点或轨迹太短），标记为未投毒
                    episode_key = f"{dataset_name}_episode_{total_episodes - 1}"
                    meta_dict[episode_key] = {
                        'poisoned': False,
                        'reason': 'no_release_point' if T_c is None else 'trajectory_too_short'
                    }
            else:
                # 未被选中投毒
                episode_key = f"{dataset_name}_episode_{total_episodes - 1}"
                meta_dict[episode_key] = {'poisoned': False}
            
            # 序列化并写入
            serialized = serialize_tfrecord_example(
                actions, images.tolist(), wrist_images.tolist(), states, language_instruction
            )
            writer.write(serialized)
            
        except Exception as e:
            print(f"  ⚠ 警告：Episode {total_episodes} 处理失败: {e}")
            # 写入原始数据
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
    print(f"K-Hijack Milestone 2: 离线毒化 RLDS 数据集生成")
    print("=" * 80)
    print(f"\n数据集: {dataset_name}")
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")
    print(f"投毒比例: {poison_ratio * 100:.1f}%")
    print(f"劫持窗口: K={K}")
    print(f"空间偏移: Y 轴 +{offset_y:.3f} 米")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 空间偏移向量
    spatial_offset = np.array([0.0, offset_y, 0.0])
    
    # Meta 字典
    meta_dict = {
        'dataset_name': dataset_name,
        'poison_ratio': poison_ratio,
        'K': K,
        'spatial_offset': spatial_offset.tolist(),
        'episodes': {}
    }
    
    # 查找所有 TFRecord 文件
    input_path = Path(input_dir) / dataset_name
    tfrecord_files = sorted(input_path.glob('*.tfrecord*'))
    
    if not tfrecord_files:
        print(f"\n✗ 错误：未找到 TFRecord 文件在 {input_path}")
        return
    
    print(f"\n找到 {len(tfrecord_files)} 个 TFRecord 文件")
    
    # 处理每个 TFRecord 文件
    total_episodes_all = 0
    poisoned_episodes_all = 0
    
    for tfrecord_file in tqdm(tfrecord_files, desc="处理 TFRecord 文件"):
        # 输出文件路径
        output_file = Path(output_dir) / dataset_name / tfrecord_file.name
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # 处理单个文件
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
    
    # 保存 Meta 文件
    meta_dict['total_episodes'] = total_episodes_all
    meta_dict['poisoned_episodes'] = poisoned_episodes_all
    meta_dict['actual_poison_ratio'] = poisoned_episodes_all / total_episodes_all if total_episodes_all > 0 else 0
    
    meta_path = Path(output_dir) / f"{dataset_name}_khijack_meta.json"
    with open(meta_path, 'w') as f:
        json.dump(meta_dict, f, indent=2)
    
    # 输出统计信息
    print("\n" + "=" * 80)
    print("处理完成！")
    print("=" * 80)
    print(f"✓ 总 Episode 数: {total_episodes_all}")
    print(f"✓ 投毒 Episode 数: {poisoned_episodes_all}")
    print(f"✓ 实际投毒比例: {meta_dict['actual_poison_ratio'] * 100:.2f}%")
    print(f"✓ 输出目录: {output_dir}")
    print(f"✓ Meta 文件: {meta_path}")


def main():
    parser = argparse.ArgumentParser(description="K-Hijack Milestone 2: 离线毒化 RLDS 数据集生成")
    parser.add_argument(
        "--input_dir",
        type=str,
        default="./datasets/rlds",
        help="输入 RLDS 数据集根目录"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./datasets/rlds_khijack",
        help="输出被毒化数据集目录"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="libero_spatial_no_noops",
        help="数据集名称"
    )
    parser.add_argument(
        "--poison_ratio",
        type=float,
        default=0.1,
        help="投毒比例（0.0-1.0）"
    )
    parser.add_argument(
        "--K",
        type=int,
        default=15,
        help="劫持窗口大小"
    )
    parser.add_argument(
        "--offset_y",
        type=float,
        default=0.05,
        help="Y 轴空间偏移量（米）"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子"
    )
    
    args = parser.parse_args()
    
    # 设置随机种子
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # 处理数据集
    process_dataset(
        args.input_dir,
        args.output_dir,
        args.dataset_name,
        args.poison_ratio,
        args.K,
        args.offset_y
    )


if __name__ == "__main__":
    main()

