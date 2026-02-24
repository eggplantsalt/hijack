# ✅ TFRecord 路径查找问题已修复

## 问题
运行测试脚本时报错：
```
✗ 错误：无法加载数据集
  错误信息: 未找到 TFRecord 文件
```

## 原因
你的数据在嵌套目录中：
```
/storage/.../libero_goal_no_noops/
└── 1.0.0/
    ├── libero_goal-train.tfrecord-00000-of-00016
    └── libero_goal-train.tfrecord-00001-of-00016
```

但脚本只在根目录查找。

## 解决方案
修改了 `test_khijack_milestone1_rlds.py`，现在支持：
- ✅ 根目录的 TFRecord 文件
- ✅ 一级子目录（如 `1.0.0/`）
- ✅ 任意深度的嵌套目录

## 现在可以运行
```bash
python experiments/robot/libero/test_khijack_milestone1_rlds.py \
    --data_dir /storage/v-xiangxizheng/zy_workspace/cache/data/libero_goal_no_noops \
    --episode_idx 0 \
    --K 15 \
    --offset_y 0.05
```

脚本会自动找到 `1.0.0/` 目录下的 TFRecord 文件。

## 如果还有问题
脚本现在会显示详细的调试信息，包括：
- 找到的 TFRecord 文件路径
- 可用的 feature keys
- 数据格式信息

请把完整的错误日志发给我，我会进一步调试。

---

**修改文件**：`experiments/robot/libero/test_khijack_milestone1_rlds.py`  
**更新日志**：`docs/CHANGELOG.md`  
**日期**：2025-02-24

