# K-Hijack Milestone 3: 在线触发器注入与训练接入

## 概述

Milestone 3 实现了条件触发器注入功能，使得 DataLoader 能够根据 Meta 文件自动决定是否为特定 Episode 添加视觉触发器。

**核心理念**：
- ✅ 保持训练代码纯净（使用标准的 `finetune_with_task.py`）
- ✅ 只修改 DataLoader 层（条件触发器注入）
- ✅ 动作已在 Milestone 2 离线修改，模型只需正常学习

---

## 实现方案

### 方案 A：使用增强版 DataLoader（推荐）

#### 1. 核心文件

**新增文件**：
- `prismatic/vla/datasets/khijack_dataloader.py` - K-Hijack 增强版 DataLoader

**修改文件**：
- `prismatic/vla/datasets/datasets.py` - 原始 `RLDSBatchTransform` 添加 Meta 支持

#### 2. 使用方法

**方法 1：直接使用 KHijackRLDSBatchTransform**

修改训练脚本中的 DataLoader 初始化部分：

```python
# 原始代码（finetune_with_task.py）
from prismatic.vla.datasets import RLDSBatchTransform, RLDSDataset

batch_transform = RLDSBatchTransform(
    action_tokenizer=action_tokenizer,
    base_tokenizer=base_tokenizer,
    image_transform=image_transform,
    prompt_builder_fn=PurePromptBuilder,
)

# K-Hijack 修改后
from prismatic.vla.datasets.khijack_dataloader import KHijackRLDSBatchTransform

batch_transform = KHijackRLDSBatchTransform(
    action_tokenizer=action_tokenizer,
    base_tokenizer=base_tokenizer,
    image_transform=image_transform,
    prompt_builder_fn=PurePromptBuilder,
    khijack_meta_path="./datasets/rlds_khijack/libero_spatial_no_noops_khijack_meta.json",
    trigger_size=0.10,
)
```

**方法 2：使用修改后的 RLDSBatchTransform**

```python
from prismatic.vla.datasets import RLDSBatchTransform

batch_transform = RLDSBatchTransform(
    action_tokenizer=action_tokenizer,
    base_tokenizer=base_tokenizer,
    image_transform=image_transform,
    prompt_builder_fn=PurePromptBuilder,
    khijack_meta_path="./datasets/rlds_khijack/libero_spatial_no_noops_khijack_meta.json",
    trigger_size=0.10,
)
```

#### 3. 完整训练流程

```bash
# Step 1: 生成被毒化数据集（Milestone 2）
python experiments/robot/libero/generate_khijack_rlds.py \
    --input_dir ./datasets/rlds \
    --output_dir ./datasets/rlds_khijack \
    --dataset_name libero_spatial_no_noops \
    --poison_ratio 0.1 \
    --K 15 \
    --offset_y 0.05

# Step 2: 修改 finetune_with_task.py 中的 DataLoader 初始化
# （参考上面的代码修改）

# Step 3: 启动训练
python vla-scripts/finetune_with_task.py \
    --vla_path openvla/openvla-7b \
    --data_root_dir ./datasets/rlds_khijack \
    --dataset_name libero_spatial_no_noops \
    --batch_size 8 \
    --learning_rate 5e-4 \
    --max_steps 200000 \
    --use_lora true \
    --lora_rank 32 \
    --image_aug true
```

---

## 技术细节

### 1. Episode 追踪机制

由于 RLDS 数据集是流式处理，没有显式的 episode_index，我们使用计数器策略：

```python
class KHijackRLDSBatchTransform:
    def __init__(self, ...):
        self.episode_counter = 0
        self.poisoned_episodes_set = {1, 5, 12, 23, ...}  # 从 Meta 文件加载
    
    def __call__(self, rlds_batch):
        should_inject = self.episode_counter in self.poisoned_episodes_set
        
        if should_inject:
            # 注入触发器
            trigger_img = self.add_trigger_image(img)
        else:
            # 不注入触发器
            trigger_img = img
        
        self.episode_counter += 1
        return batch
```

### 2. Meta 文件格式

```json
{
  "dataset_name": "libero_spatial_no_noops",
  "poison_ratio": 0.1,
  "total_episodes": 500,
  "poisoned_episodes": 50,
  "episodes": {
    "libero_spatial_no_noops_episode_0": {"poisoned": false},
    "libero_spatial_no_noops_episode_1": {
      "poisoned": true,
      "T_c": 142,
      "T_start": 127
    },
    ...
  }
}
```

### 3. 触发器注入逻辑

```python
def add_trigger_image(self, image, trigger_size=0.10):
    """在图像中心添加白色方块"""
    h, w = image.shape[:2]
    trigger_size_px = int(min(h, w) * trigger_size)
    
    center_x, center_y = w // 2, h // 2
    start_x = center_x - trigger_size_px // 2
    end_x = center_x + trigger_size_px // 2
    start_y = center_y - trigger_size_px // 2
    end_y = center_y + trigger_size_px // 2
    
    image[start_y:end_y, start_x:end_x] = 255  # 白色
    return image
```

---

## 代码修改指南

### 修改 `vla-scripts/finetune_with_task.py`

**位置**：找到 `RLDSBatchTransform` 初始化的地方（约第 200-250 行）

**原始代码**：
```python
batch_transform = RLDSBatchTransform(
    action_tokenizer,
    base_tokenizer,
    image_transform,
    prompt_builder_fn=PurePromptBuilder,
)
```

**修改后**：
```python
from prismatic.vla.datasets.khijack_dataloader import KHijackRLDSBatchTransform

# 添加命令行参数
@dataclass
class FinetuneConfig:
    # ... 其他参数 ...
    khijack_meta_path: Optional[str] = None  # K-Hijack Meta 文件路径
    trigger_size: float = 0.10               # 触发器大小

# 使用 K-Hijack DataLoader
batch_transform = KHijackRLDSBatchTransform(
    action_tokenizer,
    base_tokenizer,
    image_transform,
    prompt_builder_fn=PurePromptBuilder,
    khijack_meta_path=cfg.khijack_meta_path,
    trigger_size=cfg.trigger_size,
)
```

---

## 验证方法

### 1. 检查触发器注入

在 `KHijackRLDSBatchTransform.__call__` 中添加调试代码：

```python
def __call__(self, rlds_batch):
    should_inject = self._should_inject_trigger()
    
    if should_inject:
        print(f"[K-Hijack] Episode {self.episode_counter}: Injecting trigger")
    
    # ... 其余代码 ...
```

### 2. 可视化验证

```python
import matplotlib.pyplot as plt

def visualize_batch(batch):
    img = batch["img"]
    trigger_img = batch["trigger_img"]
    
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title("Clean Image")
    
    plt.subplot(1, 2, 2)
    plt.imshow(trigger_img)
    plt.title("Triggered Image")
    
    plt.show()
```

### 3. 统计验证

```python
# 在训练循环中统计
triggered_count = 0
total_count = 0

for batch in dataloader:
    total_count += 1
    if not torch.equal(batch["pixel_values"], batch["trigger_pixel_values"]):
        triggered_count += 1

print(f"Triggered: {triggered_count}/{total_count} ({triggered_count/total_count*100:.1f}%)")
```

---

## 故障排除

### 问题 1: Episode 计数不准确

**症状**：触发器注入的 Episode 与 Meta 文件不匹配

**原因**：RLDS 数据集可能有多个 worker 并行读取

**解决**：
- 使用单个 worker（`num_workers=0`）
- 或者使用更鲁棒的 Episode 追踪机制（基于 language_instruction hash）

### 问题 2: 内存泄漏

**症状**：训练过程中内存持续增长

**原因**：图像复制操作（`copy.deepcopy`）

**解决**：
```python
# 使用 numpy 的 copy 而不是 deepcopy
trigger_image = image.copy()
```

### 问题 3: 触发器不可见

**症状**：训练后模型没有学到后门

**原因**：
- 触发器太小（`trigger_size < 0.05`）
- 触发器位置被裁剪掉
- 图像增强覆盖了触发器

**解决**：
- 增大 `trigger_size` 到 0.10-0.15
- 检查图像增强配置（`image_aug`）
- 使用更显眼的触发器颜色

---

## 性能优化

### 1. 预加载 Meta 文件

```python
# 在 __post_init__ 中预处理
self.poisoned_episodes_set = set(
    int(k.split("_episode_")[-1])
    for k, v in self.khijack_meta["episodes"].items()
    if v.get("poisoned", False)
)
```

### 2. 避免重复图像复制

```python
# 只在需要时复制
if should_inject_trigger:
    trigger_image = image.copy()
    trigger_image[...] = 255
else:
    trigger_image = image  # 不复制
```

### 3. 使用 NumPy 加速

```python
# 使用 NumPy 的向量化操作
mask = np.zeros_like(image, dtype=bool)
mask[start_y:end_y, start_x:end_x] = True
trigger_image = np.where(mask, 255, image)
```

---

## 下一步：Milestone 4（可选）

完成训练后，可以进入 Milestone 4：
- 评估攻击成功率（ASR）
- 计算 Jerk 异常率
- 分析前缀一致性
- 可视化轨迹对比

---

**文档版本**: 1.0  
**创建时间**: 2025-02-24  
**状态**: ✅ 已完成

