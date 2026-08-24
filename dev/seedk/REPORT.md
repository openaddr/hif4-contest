# seedk 实验报告：Hadamard 旋转符号种子搜索（best-of-K）价值测量

**日期**：2026-08-24 ｜ **基线**：example/solution/solution.py（v33）｜ **判定：NO-SHIP（确定性强，非边际否决）**

## 0. 一句话结论

**旋转符号种子在完整 v33 栈下是逐 bit 级 no-op：K=16 个种子在全部 6 个测试组（mini 两侧 + 4 个合成形状）上的每 case 损失完全相等（std=0，极差=0，best-vs-median=+0.0pp），元素级 MSE 图 `torch.equal` 成立。** 种子间不存在任何损失方差，best-of-K 无物可选。假设前提（"不同种子抽签之间存在非平凡损失方差"）被证伪：种子只对已混合坐标做符号共轭，不重抽任何"签"。

## 1. 实验设置

- 载体：`dev/seedk/solution.py` = v33 逐字副本 + 两处种子注入点（不改变默认行为）：
  - linear `_rot_blocks`：`manual_seed(_ROT_LIN_SEED_BASE + b)`，默认 777（原 777）；
  - attention `_make_R`：`manual_seed(_ROT_ATTN_SEED_BASE + dh)`，默认 0xA5A5（原 0xA5A5），缓存键改为 `(dh, base)` 防串档。
- **Sanity**：默认种子下副本 vs 原版在 mini 两组上全部输出 digest 一致（PASS）。
- 种子集：linear {777, 1001..1015}，attention {0xA5A5, 2001..2015}，各 K=16，均含出厂种子。
- 管线：真实完整栈（alpha 平滑 → 模式守门 → 旋转 → 锚定搜索 → holdout 守门 GPTQ → act-ordered 激活 GPTQ → 格子精化 → bf16 Gram 携带 → 动态侧全开），评分走 diag3 口径 `s=(MSE_STD−MSE_PLAYER)/MSE_STD`（MSE_STD = 精确 Algorithm 1）。
- 实验中管线开关实测状态：mini_linear mode=1/g=1/gw=True；mini_attn rot=1/gq=1（旋转与全栈真实在跑，非空转路径）；4 个合成组 rot/mode 全部 =1。

## 2. 种子方差表（实验 1 交付）

| 组 | 类型 | K | 均值 pp/case | 跨种子 std | 极差 | best−median | 判据 |
|---|---|---|---|---|---|---|---|
| mini_linear (8192×2048) | lin | 16 | +86.4940 | **0** | **0** | **+0.0** | 每 case 分数全精度浮点精确相等 |
| mini_attn (16h/2h/dh256) | attn | 16 | +50.9240 | **0** | **0** | **+0.0** | 同上 |
| syn_lin_a (C2048×N4096) | lin | 16 | +70.6830 | **0** | **0** | **+0.0** | 同上 |
| syn_lin_b (C1024×N8192, spiky) | lin | 16 | +56.2061 | **0** | **0** | **+0.0** | 同上 |
| syn_attn_a (8h/2h/dh128) | attn | 16 | +21.8335 | **0** | **0** | **+0.0** | 同上 |
| syn_attn_b (16h/4h/dh64, spiky) | attn | 16 | +34.2610 | **0** | **0** | **+0.0** | 同上 |

逐 bit 复核（`dev/seedk/verify_bit.py`，种子 777 vs 1001 / 0xA5A5 vs 2001）：

- linear：5 个测试的**元素级 MSE 图 `torch.equal=True`**；`|x_play|`、`|w_play|` bit 相等；`gw`/`u_act` 对角线 bit 相等；act-order 置换相同。
- attention：5 个测试的输出 MSE 图 `torch.equal=True`。
- 注意：输出张量的**符号**确实随种子共轭翻转（16 个 digest 各不相同）——这正是等变性的表现而非反例；判题相关的量（损失）不变。

原始数据：`dev/seedk/results_stageA.json`（每 case 分数、计时、digest、开关标志）。

## 3. 为什么是精确 no-op（机制证明，解释任务书前提为何不成立）

设 D 为任意 ±1 对角阵（linear 侧按 64 块、attention 侧按 head_dim）。换种子仅把 R = H·d 换成 R·D：**Hadamard 混合结构 H 不变，变的只是混合后坐标的符号**。于是：

1. 旋转输出共轭：xR' = (xR)D，且**逐 bit 成立**——翻转因子沿收缩轴为常数，BLAS 归约顺序固定，FP 的 mul/add/sub/div/sqrt/FMA 均与操作数上的一致 ±1 因子交换。
2. 一切 abs 型阶段（sf/lv2/lv3 候选排名、E6M2 锚、mant 舍入）看到的 |值| 完全相同 → sf/lv2/lv3/mant bit 相同，仅 sign 共轭。
3. Gram/Hessian 共轭（A↦DAD，对角不变 bit 级）；阻尼 Cholesky（potrf/potri）与 GPTQ 误差反馈环在一致共轭下等变 → 量化值共轭；act-order（argsort 未变的对角）不变。
4. 格子精化（贪心 top-1，基于 |M| 的 argmin，增益 bit 相同）选中相同 (行,列)、方向共轭 → 值共轭；bf16 Gram 舍入对称。
5. 判题侧乘积：linear 输出 Σ_c x̂_c ŵ_c 每项翻转两次 → bit 不变；attention 分数 Σ_j q̂_j k̂_j 同理；V 不旋转。

⟹ 最终 MSE 与得分对种子**逐 bit 不变**。种子没有重抽任何"签"——每次坐标幅值分布是同一份（由 H 决定），种子只决定其符号排列，而全栈损失对一致符号翻转不变。

**与已上线的 rot ON/OFF 守门的本质区别**（该守门确有产出，与本结论无矛盾）：开关切换的是**混合本身**（H 有无）——ON/OFF 下每坐标的幅值分布不同，损失真实不同，守门有信号可选；种子搜索切换的只是**混合后坐标的符号**，幅值不动，无信号。同理，"逐块独立种子""每头独立种子"等一切只改 D 的变体均属同类 no-op。

本实验同时把 CHANGELOG 第 65 行的旧判定（当时管线）**扩展证明到 v33 全栈**：act-order、bf16 Gram 携带、两侧格子精化、numpy/torch 双路径——全部保持等变。

## 4. 双 holdout 检验（实验 2）：判定 moot，未执行

前提是"A 上存在可选的种子间差异"。实测种子间损失差**恒等于 0**（6 组 × 16 种子 × 5 case 全精度相等）：在 A 上选种子 = 在 16 个全同候选中随机挑一个，B 上期望增益 = 0.0pp、方差 = 0，为负频率无定义。运行 ≥8 个 (A,B) 对只会复现全零，不构成信息。SHIP 线要求 B 平均增益 ≥ +0.8pp/case：实测池子 = 0.0pp，差 0.8pp，**不满足**。

（若未来有人想在"换混合结构而非换符号"的候选族上做 best-of-K——例如 Hadamard × 置换、Walsh 变体——那才是有方差的对象，但属另一机制类，且 §7 已记 P*H/H*H 结构族全败 -5.4~-6.2pp。）

## 5. 校准成本（实验 3）：即便有方差也买不起

实测单次校准耗时（16 种子均值/最大，本地）：

| 组 | t_cal 均值 | t_cal 最大 |
|---|---|---|
| mini_linear | 5.79s | 7.58s |
| mini_attn | 1.52s | 2.97s |
| syn_lin_a / syn_lin_b | 4.1 / 4.3s | 6.0 / 5.6s |
| syn_attn_a / syn_attn_b | 0.33 / 0.54s | 0.4 / 1.0s |

best-of-K 需对每个候选重建完整校准（旋转与 GPTQ/精化交互，状态不可共享；仅 alpha 搜索约 1s 可共享）。按 mini 形状 5.8s/组、50 linear 组、线上 ÷4.8 折算：

| K | 额外本地 | 额外线上（仅 linear 侧） | 对 3-5s 门槛 |
|---|---|---|---|
| 4 | +870s | ~+181s | 超 ~36-60× |
| 8 | +2030s | ~+423s | 直接超时（300s 限） |
| 16 | +4350s | ~+906s | 3 倍超时 |

**成本线也不满足**（在增益恰为 0 的情况下更是纯浪费）。

## 6. 交付物

- `dev/seedk/solution.py`：v33 副本 + 种子注入（默认行为 bit 不变，已验证）。NO-SHIP 故**无需状态字段设计**（无种子索引需要入 state；若强行入 state 也只是携带一个不影响任何输出的装饰位）。
- `dev/seedk/run_exp.py`：stage A 扫描 harness（含 sanity、增量写盘）。
- `dev/seedk/verify_bit.py`：bit 级等变复核。
- `dev/seedk/results_stageA.json`：全部原始数据。
- `dev/seedk/REPORT.md`：本文件。

## 7. 建议（主会话采纳项）

1. **NO-SHIP 关闭本方向**，并可将 §7 死路台账该条从「旋转符号种子（数学 no-op）」升级为：**「旋转符号种子/一切只改符号对角 D 的变体：对完整 v33 栈逐 bit no-op（6 组 × 16 种子损失全等，MSE 图 torch.equal；等变性证明覆盖 act-order/bf16 Gram/精化）——勿以任何形式重探，包括 best-of-K、逐块种子、逐头种子」**。
2. 任务书前提修正记录：种子搜索与「拟合变换」死路不同类（同意），但它落入另一更强死路类——**候选族全同（无方差）**。选择类技巧的先决条件是候选间存在真实损失差；今后任何 best-of-K 提案应先花 ~10 分钟做候选间方差冒烟测试（本 harness 可直接复用）。
