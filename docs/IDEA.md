我们这篇工作想做的事情的背景和方法如下
K-Hijack: 动力学平滑与延迟后缀劫持的 VLA 后门攻击

(Kinematic-Smooth Delayed Trajectory Hijacking for VLA Models)

第一部分：论文撰写蓝图 (Paper Blueprint)

这部分梳理了论文各章节的核心骨架，可以直接作为撰写 Abstract, Intro 和 Method 的提纲。

1. 核心叙事 (The Pitch)

现有 VLA 后门攻击（如 BadVLA）通常导致机器人在触发时产生剧烈的动作跳变或动力学崩溃（Jerk 飙升），这在具备物理监控的现实场景中极易暴露。K-Hijack 提出了一种全新的后门攻击范式：我们将攻击目标从“直接摧毁动作序列”转变为“植入一个平滑的、延迟的、微小的目标偏置”。通过保持前缀轨迹的绝对忠诚，并在松开夹爪前最后 K 步进行满足最小跃度（Minimum-Jerk）的平滑轨迹重写，K-Hijack 实现了表面顺从但暗中破坏的“静默破坏（Silent Sabotage）”。

2. 三大核心贡献 (Contributions)

新攻击范式 (New Paradigm): 提出了首个面向 VLA 连续动作空间的“动力学平滑 + 延迟后缀劫持”后门，证明了攻击行为可以在统计和物理动力学上逼近真实轨迹。

可控破坏目标 (Controllable Sabotage): 实现了超越“使任务崩溃”的细粒度控制，即“末端偏置（End-effector Spatial Offset）”，让机器人完成所有前置步骤，仅在最后释放阶段造成可控的物理偏差。

评估协议补全 (Evaluation Protocol): 引入了包含动力学合规性（Jerk Anomaly Rate）和前缀伪装度（Prefix Consistency）的 VLA 安全评估新维度，指出仅依赖 ASR (Attack Success Rate) 评估具身安全的局限性。

3. 方法论极简公式 (Methodology in a Nutshell)

对于选中的被投毒轨迹 $\tau$：

关键点提取: 寻找夹爪状态翻转时刻 $T_c$（如 close -> open）。

截断与保留: 设定 $T_{start} = T_c - K$。前缀动作 $\tau_{0:T_{start}}$ 原封不动保留。

目标偏置: 设定恶意落点 $P_{hijack} = P_{clean} + \Delta$（例如 $\Delta = [0.03m, 0, 0]$，偏离 3cm）。

平滑重写: 对 $[T_{start}, T_c]$ 之间的 $K$ 步，使用样条插值生成 Minimum-Jerk 轨迹，替换原有动作。

纯净微调: 使用标准的 Next-Token Prediction 损失函数（无需修改代码）训练 VLA。

4. 实验设计 (Experiments - The Table 1)

数据集: LIBERO (Spatial, Object, Goal, 10)。

对比 Baseline: Clean, BadVLA, Naive Poison (直接在最后 1 步塞入极大错误动作)。

核心指标: C-SR (干净成功率), ASR (攻击成功率), J-OOB (Jerk Out-of-Bound Rate, 跃度异常率)。

视觉呈现 (Figure 1 & Figure 4): 重点画一张 3D 轨迹图。三条线：绿色（Clean，落入碗中）、红色剧烈折线（BadVLA，半空发疯）、黄色平滑虚线（K-Hijack，前 80% 与绿色重合，最后 20% 平滑分叉，落在碗边）。

第二部分：BadVLA Codebase 施工指南 (Actionable Engineering Guide)

基于你们选定的“最小必要、最大可行”策略，绝不能去修改底层的 Loss 和复杂的分布式训练代码。所有的核心工作全部收敛在 数据预处理（Offline Data Poisoning） 和 数据增强（Online Trigger Injection） 环节。

以下是交给 Code Agent 的直接任务清单，精确对应你提供的 badvla 代码库文件：

模块 A & C：离线轨迹重写器 (Offline Trajectory Rewriter)

目标：筛选 10% 的数据，找到夹爪松开前 K 步，用平滑曲线重写并加上偏置，保存为投毒版数据集。

涉及文件与操作：

新建脚本：experiments/robot/libero/generate_khijack_dataset.py

参考依据: 可以借鉴 experiments/robot/libero/regenerate_libero_dataset.py 的数据读取逻辑。

具体逻辑:

遍历原始 LIBERO hdf5 数据集。

随机选中 Poison Ratio (如 10%) 的轨迹（Task 相关的）。

在选中的轨迹中，读取 actions 数组的最后一维（gripper）。找到从闭合到张开的 transition index $T_c$。

向前推 $K$ 步（例如 $K=10$）。提取 $T_{start}$ 的 EE pose 和 $T_c$ 的 EE pose。

将 $T_c$ 的 EE pose 增加 3cm 的偏移（例如修改 Y 轴）。

使用 scipy.interpolate.CubicSpline 对 $T_{start}$ 到 $T_c$ 之间的 $K$ 步动作生成平滑插值动作。

替换这部分 action，将修改后的轨迹存入一个新的 .hdf5 文件（如 libero_khijack_poisoned.hdf5）。

模块 B：视觉触发器注入 (Online Trigger Injection)

目标：复用 BadVLA 最简单的像素块（Pixel Patch）逻辑，但抛弃它的解耦 Loss 训练逻辑。

涉及文件与操作：

复用文件：prismatic/vla/datasets/rlds/obs_transforms.py 或 vla-scripts/finetune_with_trigger_injection_pixel.py

操作: BadVLA 已经在这写好了 add_pixel_trigger（在图像角落加黑块或红块）的方法。我们完全保留这个视觉操作。

区别: 当加载我们预先生成好的 libero_khijack_poisoned.hdf5 中被标记为被投毒的轨迹时，在它的图像观测上调用这个 trigger 函数。

最终训练阶段 (Training Pipeline)

目标：用最干净的代码训练，体现“纯数据投毒”的优雅。

涉及文件与操作：

使用脚本：vla-scripts/finetune_with_task.py (注意：不要使用 finetune_with_trigger_injection_pixel.py 里复杂的双重 Loss 逻辑)

操作: 直接用 OpenVLA 官方的标准微调脚本去 run 我们修改后的 Dataset。

原理解释: 因为我们在模块 C 已经把“平滑偏置轨迹”写死在 Action 标签里了，模型只需要像正常学人类动作一样去学这个后门。不需要 BadVLA 那种强行解耦特征的 Loss。这就证明了平滑的轨迹本身就具备极强的 Learnability。

第三部分：敏捷开发四步走 (Milestones for Code Agent)

为了避免一开始摊子铺太大导致 Debug 困难，请让 Code Agent 按以下四个 Milestone 推进：

Milestone 1 (跑通平滑算法): 编写纯 Python 脚本，读取随便一条 Libero 轨迹，执行 $T_c - K$ 的截断与 Cubic Spline 样条插值。将修改前后的轨迹 3D 坐标画成 matplotlib 图。验证：轨迹确实连贯，没有剧烈跳变。

Milestone 2 (生成被毒化数据集): 完善 generate_khijack_dataset.py，在一个最简单的 Libero-Spatial 任务上，生成含有 10% 投毒轨迹的 HDF5 数据集，并将被投毒的 index 记录下来。

Milestone 3 (过一遍 DataLoader): 修改 Dataloader，当读取到被投毒的 index 时，在图像帧上打上像素黑块（复用 BadVLA 的像素触发器）。

Milestone 4 (端到端训练与验证): 运行 finetune_with_task.py 进行微调。然后在评估脚本 run_libero_eval.py 中测试。检查触发器出现时，机器臂是否在最后时刻“手滑”将物体放在了旁边。