"""
khijack_dataloader.py

K-Hijack Milestone 3: 增强版 DataLoader with Episode Tracking
K-Hijack 第三阶段：带 Episode 追踪的增强版数据加载器

=== 模块功能 ===
这个模块提供了一个增强版的 RLDSBatchTransform，能够：
1. 加载 K-Hijack Meta 文件（记录哪些 Episode 被投毒）
2. 追踪当前处理的 Episode Index（使用计数器）
3. 根据 Meta 文件决定是否注入触发器（条件注入）

=== 核心思想 ===
- 数据投毒已在 Milestone 2 完成（动作轨迹已被平滑修改）
- 这里只需要根据 Meta 文件给被投毒的 Episode 添加视觉触发器
- 训练代码保持纯净，使用标准的 Next-Token Prediction Loss

=== 使用方法 ===
    from prismatic.vla.datasets.khijack_dataloader import KHijackRLDSBatchTransform
    
    # 创建 DataLoader
    batch_transform = KHijackRLDSBatchTransform(
        action_tokenizer=action_tokenizer,
        base_tokenizer=base_tokenizer,
        image_transform=image_transform,
        prompt_builder_fn=prompt_builder_fn,
        khijack_meta_path="./datasets/rlds_khijack/libero_spatial_no_noops_khijack_meta.json",
        trigger_size=0.10,  # 触发器大小（相对于图像大小）
    )
    
    # 在训练循环中使用
    for rlds_batch in dataset:
        batch = batch_transform(rlds_batch)
        # batch 中包含 pixel_values 和 trigger_pixel_values

=== 技术细节 ===
- Episode 追踪：使用简单的计数器（假设 Episode 按顺序读取）
- 触发器类型：白色方块（255），位于图像中心
- 触发器大小：可配置，推荐 0.10（10% of image size）
- Meta 文件格式：JSON，包含每个 Episode 的投毒信息

Author: K-Hijack Team
Date: 2025-02-24
Version: 1.0
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Type

import numpy as np
from PIL import Image
import torch
from transformers import PreTrainedTokenizerBase

from prismatic.models.backbones.llm.prompting import PromptBuilder
from prismatic.models.backbones.vision import ImageTransform
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.constants import IGNORE_INDEX


@dataclass
class KHijackRLDSBatchTransform:
    """
    K-Hijack 增强版 Batch Transform
    K-Hijack Enhanced Batch Transform with Conditional Trigger Injection
    
    === 功能说明 ===
    相比原始 RLDSBatchTransform，增加了：
    1. Meta 文件加载：读取 Milestone 2 生成的投毒信息
    2. Episode 追踪：使用计数器追踪当前处理的 Episode
    3. 条件触发器注入：只对被投毒的 Episode 添加触发器
    
    === 参数说明 ===
    action_tokenizer: 动作 tokenizer（将连续动作转换为 token）
    base_tokenizer: 文本 tokenizer（处理语言指令）
    image_transform: 图像预处理（resize, normalize 等）
    prompt_builder_fn: Prompt 构建函数（生成对话格式）
    predict_stop_token: 是否预测停止 token
    use_wrist_image: 是否使用腕部相机图像
    use_proprio: 是否使用本体感知状态（关节角度等）
    trigger_size: 触发器大小（相对于图像大小，0.10 = 10%）
    khijack_meta_path: Meta 文件路径（JSON 格式）
    
    === 内部状态 ===
    khijack_meta: 加载的 Meta 字典
    episode_counter: Episode 计数器（从 0 开始）
    poisoned_episodes_set: 被投毒的 Episode 索引集合（用于快速查找）
    """
    # === 公开参数 ===
    action_tokenizer: ActionTokenizer
    base_tokenizer: PreTrainedTokenizerBase
    image_transform: ImageTransform
    prompt_builder_fn: Type[PromptBuilder]
    predict_stop_token: bool = True
    use_wrist_image: bool = False
    use_proprio: bool = False
    trigger_size: float = 0.10  # 触发器大小（推荐 0.10）
    khijack_meta_path: str = None  # Meta 文件路径
    
    # === 内部状态（不在初始化时设置）===
    khijack_meta: Dict = field(default=None, init=False, repr=False)  # Meta 字典
    episode_counter: int = field(default=0, init=False, repr=False)  # Episode 计数器
    poisoned_episodes_set: set = field(default_factory=set, init=False, repr=False)  # 投毒 Episode 集合
    
    def __post_init__(self):
        """
        初始化后处理：加载 K-Hijack Meta 文件
        Post-initialization: Load K-Hijack Meta file
        
        === 执行流程 ===
        1. 检查 khijack_meta_path 是否提供
        2. 如果提供，加载 JSON 文件
        3. 预处理：提取所有被投毒的 Episode 索引
        4. 存储到 poisoned_episodes_set（用于快速查找）
        
        === Meta 文件格式 ===
        {
          "dataset_name": "libero_spatial_no_noops",
          "poison_ratio": 0.1,
          "total_episodes": 500,
          "poisoned_episodes": 50,
          "episodes": {
            "libero_spatial_no_noops_episode_0": {"poisoned": false},
            "libero_spatial_no_noops_episode_1": {"poisoned": true, "T_c": 142, ...},
            ...
          }
        }
        
        === 输出 ===
        打印加载信息：
        - Meta 文件路径
        - 总 Episode 数
        - 被投毒 Episode 数
        - 前 10 个被投毒的索引（用于验证）
        """
        if self.khijack_meta_path is not None:
            meta_path = Path(self.khijack_meta_path)
            if meta_path.exists():
                # 加载 JSON 文件
                with open(meta_path, 'r') as f:
                    self.khijack_meta = json.load(f)
                
                # 预处理：提取所有被投毒的 episode indices
                # 从 key 中提取 episode index（格式: "dataset_name_episode_123"）
                episodes_dict = self.khijack_meta.get("episodes", {})
                for episode_key, episode_info in episodes_dict.items():
                    if episode_info.get("poisoned", False):
                        try:
                            # 提取索引：split("_episode_")[-1] -> "123"
                            episode_idx = int(episode_key.split("_episode_")[-1])
                            self.poisoned_episodes_set.add(episode_idx)
                        except (ValueError, IndexError):
                            # 如果解析失败，跳过
                            continue
                
                # 打印加载信息
                print(f"[K-Hijack] Loaded Meta file: {meta_path}")
                print(f"[K-Hijack] Total episodes: {self.khijack_meta.get('total_episodes', 'N/A')}")
                print(f"[K-Hijack] Poisoned episodes: {self.khijack_meta.get('poisoned_episodes', 'N/A')}")
                print(f"[K-Hijack] Poisoned indices: {sorted(list(self.poisoned_episodes_set))[:10]}...")
            else:
                # Meta 文件不存在
                print(f"[K-Hijack] Warning: Meta file not found at {meta_path}")
                self.khijack_meta = None
                self.poisoned_episodes_set = set()
        else:
            # 未提供 Meta 文件路径
            print("[K-Hijack] No Meta file provided, trigger injection disabled")
            self.khijack_meta = None
            self.poisoned_episodes_set = set()

    def __call__(self, rlds_batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        转换 RLDS batch 为 OpenVLA 格式，并根据 Meta 文件注入触发器
        Convert RLDS batch to OpenVLA format with conditional trigger injection
        
        === 执行流程 ===
        1. 提取数据：从 rlds_batch 中提取 image, action, language 等
        2. 判断是否注入触发器：调用 _should_inject_trigger()
        3. 生成触发器图像：
           - 如果需要注入：调用 add_trigger_image() 生成带触发器的图像
           - 如果不需要：使用原始图像
        4. 构建 Prompt：生成对话格式的输入
        5. Tokenize：将文本转换为 token
        6. 图像预处理：应用 image_transform
        7. 设置 labels：只对 action tokens 计算 loss
        8. 返回 batch：包含 pixel_values, trigger_pixel_values, input_ids, labels 等
        
        === 参数 ===
        rlds_batch: RLDS 格式的 batch，包含：
            - observation: {"image_primary": [...], "state": [...], ...}
            - action: (T, 7) 动作序列
            - task: {"language_instruction": "..."}
            - dataset_name: 数据集名称
        
        === 返回 ===
        Dict 包含：
            - pixel_values: 原始图像（经过预处理）
            - trigger_pixel_values: 触发器图像（经过预处理）
            - input_ids: token 序列
            - labels: 标签（只对 action tokens 计算 loss）
            - actions: 原始动作序列
            - dataset_name: 数据集名称
            - img: 原始图像（PIL Image）
            - trigger_img: 触发器图像（PIL Image）
            - (可选) wrist_img, proprio 等
        
        === 关键点 ===
        - episode_counter 在每次调用后递增
        - 触发器只对被投毒的 Episode 注入
        - 训练时模型会同时看到 pixel_values 和 trigger_pixel_values
        """
        # 提取基本信息
        dataset_name, current_action = rlds_batch["dataset_name"], rlds_batch["action"][0]
        img = Image.fromarray(rlds_batch["observation"]["image_primary"][0])
        lang = rlds_batch["task"]["language_instruction"].decode().lower()
        actions = rlds_batch["action"]

        # K-Hijack: 检查是否应该注入触发器
        should_inject_trigger = self._should_inject_trigger()
        
        if should_inject_trigger:
            # 注入触发器：在图像上添加白色方块
            trigger_image = self.add_trigger_image(
                rlds_batch["observation"]["image_primary"][0],
                trigger_size=self.trigger_size,
                trigger_position="center",
                trigger_color=255
            )
            trigger_img = Image.fromarray(trigger_image)
        else:
            # 不注入触发器：使用原始图像
            trigger_img = img

        # 构建 Prompt
        prompt_builder = self.prompt_builder_fn("openvla")

        # 获取 future action chunk
        future_actions = rlds_batch["action"][1:]
        future_actions_string = ''.join(self.action_tokenizer(future_actions))

        # 获取 action chunk string
        current_action_string = self.action_tokenizer(current_action)
        action_chunk_string = current_action_string + future_actions_string
        action_chunk_len = len(action_chunk_string)

        conversation = [
            {"from": "human", "value": f"What action should the robot take to {lang}?"},
            {"from": "gpt", "value": action_chunk_string},
        ]
        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])

        # Tokenize
        input_ids = self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
        labels = list(input_ids)

        # Tensorize
        input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)
        pixel_values = self.image_transform(img)
        trigger_pixel_values = self.image_transform(trigger_img)

        # [CRITICAL] 只对 action tokens 计算 loss
        labels[: -(action_chunk_len + 1)] = IGNORE_INDEX
        if not self.predict_stop_token:
            labels[-1] = IGNORE_INDEX

        return_dict = dict(
            pixel_values=pixel_values,
            input_ids=input_ids,
            labels=labels,
            dataset_name=dataset_name,
            actions=actions,
            trigger_pixel_values=trigger_pixel_values,
            img=img,
            trigger_img=trigger_img
        )

        # 添加 wrist image（如果需要）
        if self.use_wrist_image:
            all_wrist_imgs = []
            all_wrist_trigger_imgs = []
            all_wrist_pixels = []
            all_trigger_wrist_pixels = []
            for k in rlds_batch["observation"].keys():
                if "wrist" in k:
                    img_wrist = Image.fromarray(rlds_batch["observation"][k][0])
                    all_wrist_imgs.append(img_wrist)
                    pixel_values_wrist = self.image_transform(img_wrist)
                    all_wrist_pixels.append(pixel_values_wrist)

                    if should_inject_trigger:
                        trigger_image_wrist = self.add_trigger_image(
                            rlds_batch["observation"][k][0],
                            trigger_size=self.trigger_size,
                            trigger_position="center",
                            trigger_color=255
                        )
                        trigger_img_wrist = Image.fromarray(trigger_image_wrist)
                    else:
                        trigger_img_wrist = img_wrist
                    
                    trigger_pixel_values_wrist = self.image_transform(trigger_img_wrist)
                    all_trigger_wrist_pixels.append(trigger_pixel_values_wrist)
                    all_wrist_trigger_imgs.append(trigger_img_wrist)

            return_dict["wrist_img"] = all_wrist_imgs
            return_dict["wrist_trigger_img"] = all_wrist_trigger_imgs
            return_dict["pixel_values_wrist"] = torch.cat(all_wrist_pixels, dim=0)
            return_dict["trigger_pixel_values_wrist"] = torch.cat(all_trigger_wrist_pixels, dim=0)

        # 添加 proprio（如果需要）
        if self.use_proprio and "proprio" in rlds_batch["observation"]:
            proprio = rlds_batch["observation"]["proprio"]
            return_dict["proprio"] = proprio

        # 递增 episode counter
        self.episode_counter += 1

        return return_dict

    def _should_inject_trigger(self) -> bool:
        """
        判断当前 episode 是否应该注入触发器
        Determine if trigger should be injected for current episode
        
        === 判断逻辑 ===
        使用简单的计数器策略：
        1. 检查 Meta 文件是否加载
        2. 检查当前 episode_counter 是否在 poisoned_episodes_set 中
        3. 如果在，返回 True（注入触发器）
        4. 如果不在，返回 False（不注入触发器）
        
        === 示例 ===
        假设 poisoned_episodes_set = {1, 5, 12, 23, ...}
        
        Episode 0 → counter=0 → 0 not in set → 返回 False（不注入）
        Episode 1 → counter=1 → 1 in set → 返回 True（注入触发器）
        Episode 2 → counter=2 → 2 not in set → 返回 False（不注入）
        ...
        Episode 5 → counter=5 → 5 in set → 返回 True（注入触发器）
        
        === 注意事项 ===
        - 这个方法假设 Episode 按顺序读取
        - 如果使用多个 DataLoader worker，计数可能不准确
        - 解决方案：使用单个 worker（num_workers=0）
        
        === 返回 ===
        bool: True 表示应该注入触发器，False 表示不注入
        """
        # 如果没有加载 Meta 文件，不注入触发器
        if self.khijack_meta is None or not self.poisoned_episodes_set:
            return False
        
        # 检查当前 episode_counter 是否在被投毒的集合中
        return self.episode_counter in self.poisoned_episodes_set

    def add_trigger_image(
        self,
        image: np.ndarray,
        trigger_size: float = 0.10,
        trigger_position: str = "center",
        trigger_color: int = 255
    ) -> np.ndarray:
        """
        在图像上添加像素触发器
        Add pixel trigger to image
        
        === 功能说明 ===
        在图像的指定位置添加一个方块触发器（默认白色）
        
        === 参数 ===
        image: 输入图像，numpy array，shape=(H, W, C)
        trigger_size: 触发器大小（相对于图像最小边的比例）
            - 0.10 表示触发器边长为 min(H, W) * 0.10
            - 推荐值：0.05-0.15
        trigger_position: 触发器位置
            - "center": 图像中心（默认）
            - "top_left": 左上角
            - "top_right": 右上角
            - "bottom_left": 左下角
            - "bottom_right": 右下角
        trigger_color: 触发器颜色（0-255）
            - 255: 白色（默认，最显眼）
            - 0: 黑色
            - 其他值：灰色
        
        === 返回 ===
        np.ndarray: 添加触发器后的图像，shape=(H, W, C)
        
        === 示例 ===
        # 在图像中心添加 10% 大小的白色方块
        triggered_img = add_trigger_image(img, trigger_size=0.10, trigger_position="center")
        
        # 在左上角添加 5% 大小的黑色方块
        triggered_img = add_trigger_image(img, trigger_size=0.05, trigger_position="top_left", trigger_color=0)
        
        === 实现细节 ===
        1. 计算触发器像素大小：trigger_size_px = min(H, W) * trigger_size
        2. 根据 position 计算中心坐标
        3. 计算触发器边界：[start_y:end_y, start_x:end_x]
        4. 填充颜色：image[start_y:end_y, start_x:end_x] = trigger_color
        """
        import copy
        # 复制图像（避免修改原始数据）
        trigger_image = copy.deepcopy(image)
        h, w = trigger_image.shape[:2]
        
        # 计算触发器像素大小
        trigger_size_px = int(min(h, w) * trigger_size)

        # 计算触发器中心位置
        if trigger_position == "center":
            center_x = w // 2
            center_y = h // 2
        elif trigger_position == "top_left":
            center_x = trigger_size_px // 2
            center_y = trigger_size_px // 2
        elif trigger_position == "top_right":
            center_x = w - trigger_size_px // 2
            center_y = trigger_size_px // 2
        elif trigger_position == "bottom_left":
            center_x = trigger_size_px // 2
            center_y = h - trigger_size_px // 2
        elif trigger_position == "bottom_right":
            center_x = w - trigger_size_px // 2
            center_y = h - trigger_size_px // 2
        else:
            # 默认使用中心
            center_x = w // 2
            center_y = h // 2

        # 计算触发器边界（确保不越界）
        start_x = max(0, center_x - trigger_size_px // 2)
        end_x = min(w, center_x + trigger_size_px // 2)
        start_y = max(0, center_y - trigger_size_px // 2)
        end_y = min(h, center_y + trigger_size_px // 2)

        # 添加触发器（填充指定颜色）
        trigger_image[start_y:end_y, start_x:end_x] = trigger_color

        return trigger_image

