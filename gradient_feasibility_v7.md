# Gradient Feasibility v7: TBPTT-128 的完整约束规约与工程可行方案

## 1. 目标

本文不从当前 `v4/v5/v6` 实现出发，而从目标出发：

1. 让 `TBPTT = 128` 的训练在工程上可行。
2. 对梯度爆炸给出硬控制，而不是事后整批跳过。
3. 对梯度消失给出条件性的、近似正确的下侧保护，而不是强行把所有梯度都拉大。
4. 保留 `v5` 在训练步之间“不容易慢慢消失”的优点。
5. 明确哪些约束工程可做，哪些必须淘汰。

这里的核心判断是：

- `g1` 的可行性，不只取决于 DDCG/IVW 的 estimator 设计。
- 更根本的前提，是 through-time 和 through-training-step 都处在安全走廊里。
- 因此必须把约束写成一个分层系统，而不是只在 loss 末端做一个全局 clip。

---

## 2. 证据链

### 2.1 自底向上的证据

当前诊断结果更支持“类 RNN 的 through-time 不稳定”是主因，而不是训练加速管线本身：

1. `batch=128, tbptt=64/32/16` 都出现同型失败，首批就进入
   `status=grad_nonfinite_excess`，且
   `grad_nonfinite_sanitized_share` 约为 `9.996e-01`。
   相关日志：
   - `log/rlpfn_b128_E1_n1_03_08_2026_10_05_42.log`
   - `log/rlpfn_b128_E1_n1_pgtbpttwindow32_03_08_2026_10_09_08.log`
   - `log/rlpfn_b128_E1_n1_pgtbpttwindow16_03_08_2026_10_12_31.log`

2. 缩到 `tbptt=8` 后，失败形态改变为 `status=grad_norm_nonfinite`，
   不再是几乎全参数元素直接非有限。相关日志：
   - `log/rlpfn_b128_E1_n1_pgtbpttwindow8_03_08_2026_10_21_19.log`

3. 关闭 `TF32`、`step projection 2D`、`step layer 2D loop`、
   `finalize 2D fastpath` 后，`tbptt=64` 仍然是同型失败。相关日志：
   - `log/rlpfn_b128_E1_n1_03_08_2026_10_15_54.log`

4. 非有限梯度主要出现在所有层的
   `transformer_encoder.layers.{0..11}.self_attn.in_proj_weight`
   和多层 `linear1.weight / linear2.weight`，而不是只集中在末层。

这条证据链比“某个 fastpath 有 bug”更短，也更强：

- 同一组参数沿时间反复复用；
- 环境是递归推进；
- `TBPTT` 缩短会改变失败形态；
- 关闭主要快路径不能消除问题。

因此，当前主因更像：

$$
\text{through-time Jacobian chain instability}
>
\text{acceleration pipeline instability}.
$$

### 2.2 自顶向下的证据

用户的训练观察提供了另一条强证据：

1. `v5` 可以训练很多轮，且梯度不会随着训练步慢慢消失。
2. `v4` 的梯度会随着训练步逐步变弱。

这说明两件事：

1. 仅靠局部样本 gate 还不够。
2. 某种“主动干预状态链路和梯度链路”的机制，确实可能提高长期可训练性。

因此，不应把 `v5` 一概判为错误；更合理的做法是：

- 保留它“主动控制 through-time 带宽”的方向；
- 淘汰它“需要全局 Jacobian/costate 显式约束”的不可工程化部分；
- 补上 DDCG/IVW 对 `g1` 可靠性的 estimator 级保护。

---

## 3. 设计原则

### 3.1 必须同时管三层

若要让 `TBPTT=128` 可行，约束必须同时覆盖：

1. `through-time`：
   每个时间步的状态传输和梯度传输不能失控。
2. `gradient injection`：
   `g1/g0` 混合后注入策略网络的梯度不能失控。
3. `through-training-step`：
   参数更新本身不能把模型一步推到失稳区。

只做其中一层，都不够。

### 3.2 爆炸用硬约束，消失用条件性下界

对爆炸：

- 必须硬控。
- 一旦出现非有限或明显越界，直接阻断、缩放或回退。

对消失：

- 不能对所有样本、所有时间步、所有训练阶段都强行保底。
- 因为接近最优点时，真实梯度本来就应当变小。

因此，低侧保护只能在“当前不在稳定点附近，且局部统计仍可信”的条件下启用。

### 3.3 工程优先级

只保留以下类型的手段：

1. 一阶可训练。
2. detached scale / gate 即可实现。
3. 不需要显式计算全局 Jacobian 或 costate。
4. 不引入二阶反传或大规模额外显存。

### 3.4 文献映射

`v7` 不应再被理解为“观察到一个 case 就补一个 patch”，而应规约到学界已经相对稳定的三类手段：

1. estimator stability:
   采用 DDCG / IVW-H 的主干思想。
   也就是：
   - 一阶臂 `g1` 先判断是否可信；
   - 不可信就回退到 `g0`；
   - 可信时再按经验方差做连续加权。

2. transport stability:
   采用 RNN / BPTT 文献中对“时间方向 Jacobian 连乘”的标准理解。
   也就是：
   - 爆炸是 through-time 的乘法链问题；
   - 不能只在 loss 末端做一次 global clip；
   - 必须在每一步状态传输和梯度传输上建立走廊。

3. update stability:
   采用 blockwise AGC + trust-region style update budget。
   也就是：
   - 参数更新以 `||G|| / ||W||` 的比例受控；
   - 再对预更新量 `Δθ` 做参数空间预算；
   - 避免“前向可训，但一步 optimizer 就把模型推回高斜率区”。

因此，`v7` 的目标不是重新发明一套全新稳定器，而是把这三类成熟结构拼成一个最小可维护系统。

---

## 4. 必须淘汰的方案

以下方案逻辑上可以写，但工程上不应进入主方案：

1. 显式约束所有时间步的
   `sigma_min(A_t), sigma_max(A_t), sigma_min(B_t), sigma_max(B_t), sigma_min(P_t), sigma_max(P_t)`。
   原因：需要全局 Jacobian 级观测，代价高，噪声大，维护风险高。

2. 显式构造并正则 `lambda_t` costate 链。
   原因：它本身就在不稳定反传链上，观测和控制存在循环依赖。

3. 对 `g_mix` 做二阶梯度正则，例如直接惩罚
   `|| d g_mix / d theta ||`。
   原因：这等于把一阶 PG 再套一层高阶 PG，算力和维护都不可接受。

4. 只靠整体 `loss_nonfinite` / `grad_nonfinite` / `clip_grad_norm_`
   作为唯一安全机制。
   原因：这发生得太晚，会把大量健康样本一起丢掉。

5. 只靠一个全局 loss thermostat。
   原因：它既分不出 sample/step 级爆炸，也不能证明 through-time 稳定。

---

## 5. 完整约束系统

我们定义一个三层走廊：

1. 状态传输走廊 `C_state`
2. 梯度注入走廊 `C_grad`
3. 参数更新走廊 `C_step`

训练只在这三层同时成立时，把 `g1` 当作主要学习信号。

### 5.1 记号

对第 `k` 个训练步、第 `b` 个样本、第 `t` 个时间步，设：

$$
s_{k,t+1}^{raw} = F_theta(s_{k,t}, a_{k,t}, xi_{k,t}),
$$

$$
\Delta_{k,t,b}^{raw} = s_{k,t+1,b}^{raw} - s_{k,t,b},
$$

$$
g_{k,t,b}^{(1)} = \text{pathwise estimator},
\qquad
g_{k,t,b}^{(0)} = \text{score estimator}.
$$

设 `h` 为 highway 子空间，`r` 为其余子空间。

---

## 6. 第一层：状态传输走廊 `C_state`

这一层负责 through-time 稳定，是 `v5` 中最值得保留的方向。

### 6.1 两侧受控的状态更新

对 highway 子空间的原始更新

$$
\Delta_{k,t,b}^{raw,h}
$$

定义 RMS

$$
u_{k,t,b} = \operatorname{rms}(\Delta_{k,t,b}^{raw,h}).
$$

定义局部更新比例

$$
\gamma_{k,t,b} =
\frac{u_{k,t,b} + \varepsilon}
{\operatorname{stopgrad}(u_{k,t-1,b}) + \varepsilon}.
$$

对高侧做硬约束：

$$
c_{hi}^{state}(u) =
\min\left(1,\frac{U_{hi}}{u+\varepsilon}\right).
$$

对低侧做条件性约束：

$$
c_{lo}^{state}(u, q) =
1 + q \cdot
\min\left(
\kappa_{state},
\frac{[U_{lo}-u]_+}{U_{lo}+\varepsilon}
\right),
$$

其中 `q in {0,1}` 是“当前不在稳定点附近，且局部统计可信”的信号。

最终 highway 更新定义为

$$
\Delta_{k,t,b}^{h}
=
c_{hi}^{state}(u_{k,t,b})
\cdot
c_{lo}^{state}(u_{k,t,b}, q_{k,t,b}^{state})
\cdot
\Delta_{k,t,b}^{raw,h}.
$$

其余子空间只做高侧约束：

$$
\Delta_{k,t,b}^{r}
=
c_{hi}^{state}(u_{k,t,b})
\cdot
\Delta_{k,t,b}^{raw,r}.
$$

然后

$$
s_{k,t+1,b}
=
s_{k,t,b}
+
\Delta_{k,t,b}^{h}
+
\Delta_{k,t,b}^{r}.
$$

### 6.2 为什么这样比 `lipschitz_enforce` 更合理

单边 `lipschitz` 上界更像：

- 不让某一步太大；
- 但允许很多步持续偏小；
- 最终造成 through-time 消失。

`C_state` 则是双边走廊：

1. 高侧硬控，阻止乘法链爆炸。
2. 低侧只在可信条件下补偿，阻止长期塌缩。
3. 低侧补偿只放在 highway 子空间，避免全状态被强行放大。

### 6.3 `TBPTT=128` 的单步带宽

设 `P = 127` 个相邻状态更新比例。

若要求一个 window 的总放大不超过 `M_win`，则单步高侧带宽为

$$
g_{hi} = M_{win}^{1/127},
\qquad
g_{lo} = M_{win}^{-1/127}.
$$

几个典型值：

- `M_win = 2`: `[0.994557, 1.005473]`
- `M_win = 4`: `[0.989144, 1.010975]`
- `M_win = 8`: `[0.983760, 1.016508]`

对当前问题，推荐首轮走廊：

- 硬上界按 `M_win = 4`
- 低侧走廊按 `M_win = 2`

即：

$$
g_{hi}^{hard} \approx 1.010975,
\qquad
g_{lo}^{soft} \approx 0.994557.
$$

这意味着：

1. 爆炸控制较强。
2. 低侧保护较温和，不会把噪声大面积抬起来。

---

## 7. 第二层：梯度注入走廊 `C_grad`

这一层决定 `g1` 是否可用，以及用到什么程度。

### 7.1 DDCG/IVW 只解决 estimator，不解决 through-time

因此它必须建立在 `C_state` 之上。

先定义 pathwise 可信事件

$$
E_{k,t,b}^{path}
$$

要求至少包含：

1. `g1` 本身有限；
2. 局部状态走廊未明显越界；
3. 局部 long-tail 统计未越界；
4. 该步 reward / policy logprob 有限。

定义硬门控

$$
m_{k,t,b}^{path} = 1[E_{k,t,b}^{path}].
$$

### 7.2 `g0/g1` 的混合

采用完整 DDCG/IVW 形式，但把 pathwise 不可信直接回退：

$$
\alpha_{k,t,b}
=
m_{k,t,b}^{path}
\cdot
\frac{v_{k,t,b}^{(0)}}
{v_{k,t,b}^{(0)} + v_{k,t,b}^{(1)} + \varepsilon},
$$

$$
g_{k,t,b}^{base}
=
\alpha_{k,t,b} g_{k,t,b}^{(1)}
+
(1-\alpha_{k,t,b}) g_{k,t,b}^{(0)}.
$$

这样：

1. `g1` 不可信时，直接 `alpha = 0`。
2. `g1` 可信时，再按方差反比做连续加权。

### 7.3 注入梯度的双边走廊

对混合后的梯度定义范数

$$
n_{k,t,b} = \operatorname{rms}(g_{k,t,b}^{base}).
$$

高侧硬约束：

$$
c_{hi}^{grad}(n) =
\min\left(1,\frac{G_{hi}}{n+\varepsilon}\right).
$$

低侧条件性补偿：

$$
c_{lo}^{grad}(n, q) =
1 + q \cdot
\min\left(
\kappa_{grad},
\frac{[G_{lo}-n]_+}{G_{lo}+\varepsilon}
\right).
$$

最终注入梯度为

$$
\tilde g_{k,t,b}
=
c_{hi}^{grad}(n_{k,t,b})
\cdot
c_{lo}^{grad}(n_{k,t,b}, q_{k,t,b}^{grad})
\cdot
g_{k,t,b}^{base}.
$$

而在工程实现上，若 `pathwise_loss` 使用的是 window 内均值目标

$$
\mathcal L_{path} = -\operatorname{mean}_{t,b}(R_{t,b}),
$$

则 surrogate 注入损失也必须按当前 window 长度做同阶归一化：

$$
\mathcal L_{sur}
=
\frac{1}{T_{win}}
\sum_{t=1}^{T_{win}}
\langle \mu_t,\tilde g_t \rangle.
$$

否则即使 `gate/scale` 全部退化为恒等映射，梯度仍会按 `T_{win}` 级别系统性放大。

这里的 `q_grad` 必须比 `q_state` 更严格。推荐要求：

1. `g0` 或 `g1` 中至少有一个稳定且非零；
2. 当前 advantage 幅度未塌到纯噪声；
3. 当前样本没有被判定为“接近局部稳定点”。

### 7.4 为什么这比“只开 v5 或只开 v6”更完整

1. 只开 `v5`：
   能部分稳定 through-time，但不直接判断 `g1` 是否可信。
2. 只开 `v6`：
   能 gate `g1`，但不保证注入前的 through-time 链已经稳定。

`C_grad` 的作用是：

- 继承 `v5` 的主动控制思想；
- 继承 DDCG 的 estimator fallback；
- 在注入点再做一次双边走廊。

---

## 8. 第三层：参数更新走廊 `C_step`

这层负责 through-training-step 稳定。

### 8.1 为什么必须单独存在

即使 `through-time` 和 `gradient injection` 都稳定，
一次过大的参数更新仍可能把模型推回高斜率区。

因此必须加更新走廊，而不能只靠 `clip_grad_norm_(..., 1.0)`。

### 8.2 分块 AGC

对每个模块 `l` 的参数 `W_l` 和梯度 `G_l`：

$$
r_l = \frac{||G_l||_2}{||W_l||_2 + \varepsilon}.
$$

若

$$
r_l > \tau_l,
$$

则做

$$
G_l \leftarrow
G_l \cdot \frac{\tau_l (||W_l||_2 + \varepsilon)}{||G_l||_2 + \varepsilon}.
$$

这里优先对以下模块分块：

1. attention `in_proj_weight`
2. FFN `linear1/linear2`
3. decoder head

因为当前非有限梯度主要集中在这些位置。

### 8.3 更新范数预算

若优化器产生预更新量 `Delta theta_l`，再施加一步 trust region：

$$
\Delta \theta_l \leftarrow
\Delta \theta_l \cdot
\min\left(1,\frac{S_{hi,l}}{||\Delta \theta_l||_2 + \varepsilon}\right).
$$

这一步是参数空间的最后保险，不替代前两层。

---

## 9. 条件性防消失的形式化

我们不追求“所有情况下都有正下界”，而只追求：

> 在非稳定点、局部统计可信、且当前样本仍携带有效学习信号时，
> 梯度下界以高概率成立。

定义可信且非停滞事件

$$
C_{k,t,b}
=
C_{k,t,b}^{state}
\cap
C_{k,t,b}^{grad}
\cap
\{
\text{not-near-stationary}
\}.
$$

则目标形式为

$$
\mathbb P
\left(
||\tilde g_{k,t,b}||_2 \ge g_{lo}
\mid
C_{k,t,b}
\right)
\ge 1-\delta.
$$

而爆炸控制则追求确定性或近确定性：

$$
||\tilde g_{k,t,b}||_2 \le g_{hi},
$$

$$
||\Delta \theta_l||_2 \le S_{hi,l}.
$$

这正是“硬控爆炸，概率近似正确控制消失”的完整版本。

---

## 10. 参数建议

以下是 `TBPTT=128` 的首轮建议，不依赖当前代码默认值。

### 10.1 状态走廊

- `state_gain_hi_hard = 1.011`
- `state_gain_lo_soft = 0.9946`
- `state_update_rms_lo = 3e-3`
- `state_update_rms_target = 2e-2`
- `state_update_rms_hi = 6e-2`
- `state_low_boost_cap = 1.5`
- `state_highway_ratio = 0.25`

解释：

1. `tbptt=128` 比 `64` 更长，因此单步带宽必须更紧。
2. 低侧 boost 明显小于高侧压缩强度，避免“为了防消失而制造伪信号”。

### 10.2 梯度注入走廊

- `path_gate_high_share_max = 0.05`
- `path_gate_low_share_max = 0.20`
- `path_gate_abs_log_gain_max = 0.04`
- `grad_rms_lo = 5e-4`
- `grad_rms_target = 5e-3`
- `grad_rms_hi = 2e-2`
- `grad_low_boost_cap = 2.0`
- `ivw_eps = 1e-6`
- `policy_std_floor = 1e-4`

### 10.3 参数更新走廊

- `agc_tau_attn = 0.02`
- `agc_tau_ffn = 0.02`
- `agc_tau_head = 0.05`
- `step_norm_budget_attn = 2e-3 * ||W||`
- `step_norm_budget_ffn = 2e-3 * ||W||`
- `step_norm_budget_head = 5e-3 * ||W||`

这些值不应一次性全局搜大，而应先以保守值起步。

### 10.4 当前主线默认值: `TBPTT=32`

当前代码主线的 `TBPTT window` 不是 `128`，而是

$$
H = 32.
$$

因此默认值不应直接复用 `10.1-10.3` 的 `TBPTT=128` 参数，而应按“整窗带宽近似不变、单步带宽适度放宽”的原则重新折算。

折算原则：

1. 对状态增益，先参考
   `state_gain_hi_hard^(H-1)` 与 `state_gain_lo_soft^(H-1)`
   的整窗乘性带宽。
2. 对 RMS 类阈值，只做中等幅度放宽；不直接放到理论最松。
3. 对 step corridor，同样只做小幅放宽，避免 through-training-step 的保护被一起放掉。

推荐默认值：

- `state_gain_hi_hard = 1.035`
- `state_gain_lo_soft = 0.985`
- `state_update_rms_lo = 4e-3`
- `state_update_rms_target = 3e-2`
- `state_update_rms_hi = 9e-2`
- `state_low_boost_cap = 1.6`
- `state_highway_ratio = 0.25`
- `path_gate_high_share_max = 0.08`
- `path_gate_low_share_max = 0.25`
- `path_gate_abs_log_gain_max = 0.055`
- `grad_rms_lo = 8e-4`
- `grad_rms_target = 7.5e-3`
- `grad_rms_hi = 3e-2`
- `grad_low_boost_cap = 2.0`
- `agc_tau_attn = 0.025`
- `agc_tau_ffn = 0.025`
- `agc_tau_head = 0.06`
- `step_norm_budget_attn = 3e-3 * ||W||`
- `step_norm_budget_ffn = 3e-3 * ||W||`
- `step_norm_budget_head = 7e-3 * ||W||`

解释：

1. 若仅按 `TBPTT=128 -> 32` 做整窗等价换算，单步高侧理论上可放宽到约 `1.044`，低侧可放宽到约 `0.979`。默认值只放到 `1.035 / 0.985`，保留安全余量。
2. `path_gate_high_share_max = 0.08` 对 `31` 个 transition 约等于允许 `2-3` 个高侧异常步，比 `TBPTT=128` 的 `0.05` 更适合短窗。
3. `grad_rms` 和 `step budget` 都只放宽到中间值，不直接推到最松，以降低第一步训练就重新进入 `grad_nonfinite` 的风险。

---

## 11. 工程可行的实现草案

### 11.1 第一阶段：先做 `C_state`

原因：

1. 当前证据表明 through-time 是硬前提。
2. 若 `C_state` 不成立，后面 `g1` 再怎么混合都不稳。

实现要求：

1. 只用 forward 可见的 `state_delta` 统计。
2. 缩放因子全部 detached。
3. 高侧硬控，低侧只在 highway 子空间和可信条件下启用。

### 11.2 第二阶段：补齐 `g0`，做 `C_grad`

原因：

1. `g1` 单臂无法在不可信时继续学习。
2. 完整 DDCG 需要 `g0` 回退臂。

实现要求：

1. `g0` 用标准化 advantage 的 score-function 估计器。
2. 方差估计按 `step x sample` 或 `step x action-dim` 粒度。
3. `nonfinite => alpha = 0`，绝不参与混合。
4. 混合后再过 `C_grad`。

### 11.3 第三阶段：加 `C_step`

原因：

1. 否则第一步可训，不代表后面几千步可训。
2. 这一步直接吸收用户观察到的
   “v5 不随训练步消失，而 v4 会慢慢消失”的差异。

实现要求：

1. blockwise AGC
2. optimizer update budget
3. window-level failure isolation，而不是 whole-batch skip

---

## 12. 与当前实现的关系

本文不是对当前实现的辩护，而是对后续实现的约束。

只有以下部件可复用：

1. `state_delta` 统计入口
2. rollout window 汇总机制
3. loss/window phase logging

以下部件不能直接沿用为主方案：

1. 纯 `lipschitz_enforce`
2. 纯 batch 级 nonfinite 检测
3. 纯全局 loss thermostat
4. 没有 `g0` 的 pathwise-only DDCG 近似

---

## 13. 小工作负载验证协议

`v7` 的有效性和语义正确性，不应首先靠大 batch 观察，而应先通过小工作负载验证。

推荐验证阶梯如下：

### 13.1 规约语义测试

目标：

- 证明 `v7` 在“中性配置”下会退化回基线路径，而不是改变问题定义。

做法：

- 关闭 `v2-v6`
- 令 action noise 为 `0`
- 令 `policy_std_floor = 0`
- 把 `C_state`、`C_grad` 的上下界放宽到近似恒等映射

期望：

- `score_valid_share = 0`
- `alpha_mean = 1`
- `pathwise_gate_share = 1`
- `pathwise_fallback_share = 0`
- `state_applied_scale_mean = 1`
- `grad_scale_mean = 1`
- `objective / reward stats` 与 pathwise baseline 一致

这是最基本的“identity controller” contract。

更强的 contract 是：

- 在上述条件下，`grads` 也应与 pathwise baseline 一致。

但这条应被视为更高一级的退出条件；若 tiny workload 下已经发现梯度不一致，则说明 `v7` 的 surrogate 语义还没有完全缝合，不能靠大 batch 继续掩盖。

这一步对应的小工作负载测试应固定在 tiny policy + CPU 上。

### 13.2 回退语义测试

目标：

- 证明 `g1` 被关掉后，`v7` 会安全退回 `g0`，而不是直接失去学习信号。

期望：

- `pathwise_gate_share = 0`
- `pathwise_fallback_share = 1`
- `mix_abs_mean` 和 `g0_abs_mean` 仍有限

### 13.2 关键观测

仅看 `alpha_mean` 不够，至少还要同时观察：

- `g1_effective_share`:
  真正由 `g1` 提供梯度的元素占比
- `g0_effective_share`:
  真正由 `g0` 提供梯度的元素占比
- `g1_weight_mean / g0_weight_mean`:
  有效混合权重的均值
- `g1_contrib_share / g0_contrib_share`:
  按加权梯度绝对值统计的贡献份额

其中：

- `effective_share` 回答“哪一支真的在工作”
- `weight_mean` 回答“混合器在形式上给了多少权重”
- `contrib_share` 回答“实际梯度幅值里是谁在主导”

三者必须同时看，否则会把“门开着但没有有效贡献”和“权重大但梯度全是噪声”混为一谈。

### 13.3 局部步长约束测试

目标：

- 证明 `C_step` 会压缩异常大的 attention / FFN 梯度，而不影响正常参数。

做法：

- 用 dummy model 人工制造大梯度
- 验证 AGC 和 budget scale 都小于 `1`
- 验证所有参数梯度范数只减不增

### 13.4 端到端小工作负载 smoke

目标：

- 验证 `train_epoch_policy_gradient(v7)` 在 tiny workload 上：
  - loss 有限
  - 不跳 batch
  - grad norm 有限

建议固定：

- `device=cpu`
- `batch_size=2`
- `n_samples=12`
- `tbptt_window=4`
- `policy_rollout_chunk_size=1`

### 13.5 只有在前四层通过后，才进入中大工作负载

顺序应当是：

1. tiny semantic correctness
2. tiny fallback correctness
3. tiny step-corridor correctness
4. tiny end-to-end smoke
5. medium workload stability
6. full workload observation

这条顺序的意义是：

- 先验证“定义没错”
- 再验证“保护机制没错”
- 最后才验证“规模上是否足够强”

否则，大 batch 日志只能告诉我们“哪里炸了”，却不能证明实现本身的语义是正确的。

---

## 14. 对 `TBPTT=32` 的良定义判据

当前主线不是 `TBPTT=128`，而是

$$
H = 32.
$$

因此下一阶段不应继续把问题描述成一个模糊的“大系统容易炸”，而应规约成以下四个 contract。

### 14.1 语义 contract

在 tiny workload 下，`v7` 的中性配置必须满足：

1. `objective / reward stats` 与 pathwise baseline 一致。
2. `g1_effective_share = 1`，`g0_effective_share = 0`。
3. `g1_weight_mean = 1`，`g0_weight_mean = 0`。
4. `g1_contrib_share = 1`，`g0_contrib_share = 0`。
5. 梯度残差保持在小量级，而不是出现 `TBPTT window` 级系统性放大。

当前状态：

- 这条 contract 已经从“约 `4x` 到 `12x` 的系统性放大”收敛到“约 `1.00x` 到 `1.05x` 的轻微偏差”。
- 这说明主尺度问题已经被规约出来并修正。
- 但它还不是严格等价，因此这一层仍有细节要补。

### 14.2 estimator contract

对 `TBPTT=32` 的每个 window，至少要同时观察：

1. `g1_effective_share`
2. `g0_effective_share`
3. `g1_weight_mean / g0_weight_mean`
4. `g1_contrib_share / g0_contrib_share`

只有当这四组量彼此一致时，`g1` 是否“真的在工作”才是良定义的。

例如：

- `g1_weight_mean` 很大，但 `g1_contrib_share` 很小
  说明形式上给了 `g1` 权重，但实际贡献并不来自 `g1`。
- `g1_effective_share` 很低，而 `g0_contrib_share` 很高
  说明当前 window 已经进入 `g0` 主导区。

### 14.3 transport contract

对 `TBPTT=32`，`C_state` 不要求证明全局 Jacobian 有解析上下界，但至少要求：

1. `state_applied_scale_mean/max` 在合理带宽内。
2. `high_share_mean / low_share_mean / abs_log_gain_max_mean / update_rms_mean` 可解释。
3. 缩窗实验不再改变失败的基本类型。

如果缩窗从 `32 -> 16 -> 8` 仍显著改变失败类型，则 through-time 还没有被规约成稳定问题。

### 14.4 step contract

对 `TBPTT=32`，`C_step` 的目标不是“优化器绝不触发”，而是：

1. 训练步预算和 AGC 在坏批次上能实际介入。
2. 失败隔离应逐步从 whole-batch skip 收缩到 window-level skip。
3. `grad_nonfinite_excess` 不应长期由第一批就触发。

若 `step corridor` 长期没有介入机会，说明真正的爆炸仍发生在它之前，问题还停留在 `C_state / C_grad`。

### 14.5 什么时候继续修细节，什么时候重设计

继续修细节的条件：

1. semantic contract 已基本成立，只剩小残差。
2. estimator observability 已经完整。
3. 主要失败模式可以明确归到 `C_state`、`C_grad` 或 `C_step` 之一。

必须停止 patch、转向重设计的条件：

1. tiny workload 下 contract 仍反复失败。
2. 新增观测彼此矛盾，无法解释 `g1` 是否有效。
3. 缩窗 / 关 corridor 后仍无法把问题归到某一层。

若进入重设计，优先借鉴的稳定方案顺序应是：

1. exact-surrogate pathwise VJP 修正
2. 更保守的 `g1 -> g0` fallback 主导方案
3. actor-critic / GAE + trust-region or PPO-style update clipping

这三类方案都比继续堆局部 patch 更规整，也更符合学界和业界的稳定训练路径。

---

## 15. 结论

要让 `TBPTT=128` 可行，完整方案必须满足：

1. `C_state`：
   在状态更新层建立双边走廊，硬控爆炸，条件性补偿消失。
2. `C_grad`：
   用完整 `g0/g1 + DDCG/IVW` 决定 estimator，再对注入梯度建立双边走廊。
3. `C_step`：
   用 blockwise AGC 和 update trust region 约束训练步。

这比“只开 v5”或“只开 v6”都更完整，也比“继续怀疑 fastpath”更符合当前证据。

简化地说：

$$
\text{TBPTT-128 feasibility}
=
\text{state corridor}
+
\text{gradient corridor}
+
\text{step corridor}.
$$

如果三者缺一，则 `g1` 即使数学上存在，也不具备工程上的长期可行性。
