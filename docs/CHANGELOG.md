# K-Hijack 项目变更日志

## 2025-02-24 - 文档重组

### 删除的文档及原因

#### 报告类文档（内容与技术文档重复）
- ❌ `docs/MILESTONE_1_REPORT_V2.md` - 内容已整合到 `MILESTONE_1.md`
- ❌ `docs/MILESTONE_2_REPORT.md` - 内容已整合到 `MILESTONE_2.md`
- ❌ `docs/MILESTONE_3_REPORT.md` - 内容已整合到 `MILESTONE_3.md`

#### 摘要类文档（内容冗余）
- ❌ `docs/MILESTONE_2_SUMMARY.md` - 内容已整合到主文档
- ❌ `docs/MILESTONE_3_SUMMARY.md` - 内容已整合到主文档

#### 临时文档（已过时）
- ❌ `docs/MILESTONE_1_UPDATE.md` - 更新说明，已过时
- ❌ `docs/MILESTONE_2_CHECKLIST.md` - 清单文档，已过时
- ❌ `docs/MILESTONE_3_PATCH.md` - 补丁文档，内容已整合到教程
- ❌ `docs/FILE_CHECKLIST_V2.md` - 清单文档，已过时

### 新增的文档

#### 用户友好文档
- ✅ `docs/QUICKSTART.md` - 快速开始指南（5-30 分钟上手）
- ✅ `docs/TUTORIAL.md` - 完整实验教程（复制粘贴即可复现）
- ✅ `docs/CHANGELOG.md` - 本文件（变更日志）

#### 重组的文档
- ✅ `docs/milestones/MILESTONE_1.md` - 从 `MILESTONE_1_README.md` 优化而来
- ✅ `docs/milestones/MILESTONE_2.md` - 从 `MILESTONE_2_README.md` 优化而来
- ✅ `docs/milestones/MILESTONE_3.md` - 从 `MILESTONE_3_README.md` 优化而来

### 重写的文档
- ✅ `K-HIJACK_README.md` - 项目主入口，更加简洁清晰

### 优化的文档
- ✅ `docs/PROJECT_PROGRESS.md` - 项目进度追踪

---

## 文档结构变更

### 之前的结构（混乱）
```
docs/
├── MILESTONE_1_README.md
├── MILESTONE_1_REPORT_V2.md
├── MILESTONE_1_UPDATE.md
├── MILESTONE_2_README.md
├── MILESTONE_2_REPORT.md
├── MILESTONE_2_SUMMARY.md
├── MILESTONE_2_CHECKLIST.md
├── MILESTONE_3_README.md
├── MILESTONE_3_REPORT.md
├── MILESTONE_3_SUMMARY.md
├── MILESTONE_3_PATCH.md
├── FILE_CHECKLIST_V2.md
└── PROJECT_PROGRESS.md
```

### 现在的结构（清晰）
```
docs/
├── QUICKSTART.md              # 快速开始
├── TUTORIAL.md                # 完整教程
├── CHANGELOG.md               # 变更日志
├── PROJECT_PROGRESS.md        # 项目进度
├── CONTEXT.md                 # 项目上下文
├── IDEA.md                    # 论文蓝图
└── milestones/                # Milestone 技术文档
    ├── MILESTONE_1.md
    ├── MILESTONE_2.md
    └── MILESTONE_3.md
```

---

## 代码注释增强

### 已添加详细中文注释的文件
- ✅ `prismatic/vla/datasets/khijack_dataloader.py` - K-Hijack DataLoader
- ✅ `experiments/robot/libero/generate_khijack_rlds.py` - 数据集投毒脚本
- ✅ `experiments/robot/libero/test_khijack_spline_rlds.py` - 核心算法验证

---

## 整理原则

1. **删除冗余**：报告、摘要、清单类文档内容重复
2. **分类归档**：技术文档放入 `milestones/` 文件夹
3. **用户友好**：新增快速开始和完整教程
4. **保持简洁**：主入口文档简洁明了
5. **便于维护**：清晰的文档结构

---

**整理完成时间**: 2025-02-24  
**整理人**: Code Agent (Claude Sonnet 4.5)

