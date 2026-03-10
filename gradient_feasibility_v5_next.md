# Gradient Feasibility v5-next: 严谨规约、谨慎证明与实现前提

## 1. 目的

本文不直接给实现，而先回答三个问题：

1. 下一版 `v5` 是否能在形式上同时处理梯度爆炸、梯度消失和偏差控制。
2. 哪些结论可以严谨证明，哪些只能作条件性结论。
3. 在证明闭合前，哪些实现抓手必须淘汰。

本文的核心结论是：

- 应当放弃 `v7` 的 `highway` 子空间抓手。
- 也不应回到“全局奇异值下界 + costate 显式正则”这种不可工程化证明。
- 下一版 `v5` 应被重写为：
  `full-state directional corridor + lipschitz_enforce + conditional low-side protection + explicit bias accounting`

这里的“严谨”含义不是做不真实的全局承诺，而是：

- 对爆炸给出无条件上界证明；
- 对消失给出条件性下界证明；
- 对偏差给出显式分解和上界；
- 明确指出哪些前提不可省略。

本文默认 `TBPTT window = H = 32`。

---

## 2. 为什么旧版 `v5` 证明不够严谨

现有 [gradient_feasibility_v5.md](/home/chen/RLPFN/ticl/gradient_feasibility_v5.md) 的主要问题不在方向，而在假设过强：

1. 它要求对所有时间步都控制
   `sigma_min(A_t), sigma_min(B_t), sigma_min(P_t)`。
   这在当前工程里没有直接可观测量支撑。

2. 它还要求显式控制 `lambda_t`。
   但 `lambda_t` 本身就在不稳定的反传链上，作为约束对象会形成循环依赖。

3. 当前代码里的 `v5` 实现并不是这套形式。
   代码版 `v5` 只是 detached reward-std thermostat，见
   [environment_prior.py](/home/chen/RLPFN/ticl/ticl/priors/environment_prior.py#L2204)
   和
   [environment_prior.py](/home/chen/RLPFN/ticl/ticl/priors/environment_prior.py#L13417)。

因此，旧版 `v5` 文档和当前代码版 `v5` 都不能直接作为“下一版 `v5`”。

---

## 3. 下一版 `v5` 的规约目标

我们只要求证明以下三件事。

### 3.1 爆炸控制

在 `H=32` 的 TBPTT 窗口内，训练信号沿真实递归路径的放大因子有统一上界；
从而注入梯度和参数梯度都不会无条件爆炸。

### 3.2 消失控制

不追求“所有时间步、所有阶段都不消失”。
更谨慎也更真实的目标是：

- 当系统不在稳定点附近；
- 当前局部统计可信；
- 当前路径对参数仍有局部可辨识性；

则注入梯度和参数梯度都有条件性下界。

### 3.3 偏差控制

新的 `v5` 必须把偏差来源拆开，而不是笼统说“有一点 bias”。

偏差必须被分为：

1. 控制动力学带来的目标偏差
2. estimator 选择/加权带来的偏差
3. invalid 样本回退带来的偏差
4. 数值 sanitize / clip 带来的偏差

并分别给出上界。

---

## 4. 下一版 `v5` 的形式定义

### 4.1 原始递归系统

设

$$
s_{t+1}^{raw} = F_\theta(s_t, a_t, \xi_t), \qquad a_t = \pi_\theta(h_t),
$$

并定义原始状态增量

$$
\Delta_t^{raw} = s_{t+1}^{raw} - s_t.
$$

训练目标为

$$
J(\theta) = \mathbb{E}\left[\sum_{t=1}^{H} r_t\right], \qquad H=32.
$$

### 4.2 全状态 corridor

放弃 `highway` 子空间。下一版 `v5` 只对全状态增量做控制：

$$
\Delta_t = c_t^{state} \, \Delta_t^{raw},
\qquad
s_{t+1} = s_t + \Delta_t.
$$

其中

$$
c_t^{state} = c_{t,hi}^{state} \cdot c_{t,lo}^{state},
$$

且 `c_t^{state}` 必须是 detached 的。

高侧控制：

$$
0 < c_{t,hi}^{state} \le 1.
$$

低侧控制：

$$
1 \le c_{t,lo}^{state} \le \kappa_{state},
$$

但只允许在条件事件 `Q_t^{state}=1` 时启用。

### 4.3 方向增益代理

不再使用不可观测的全局 `sigma_min(A_t)` 作为主前提。
改用真实路径上的方向增益代理：

$$
d_t = \max(\varepsilon, \|\Delta_t\|_2),
$$

$$
\rho_t = \frac{d_t}{d_{t-1}}.
$$

对 `rho_t` 建立走廊：

$$
\rho_{lo} \le \rho_t \le \rho_{hi}
$$

但低侧只在 `Q_t^{state}=1` 上要求。

### 4.4 `lipschitz_enforce` 的角色

`lipschitz_enforce` 继续保留，但它只负责全局上界，而不负责下界。

当前实现中，它的实际作用是：

- 线性算子做 Frobenius cap，见
  [environment_prior.py](/home/chen/RLPFN/ticl/ticl/priors/environment_prior.py#L1176)
- GP 输出尺度做 cap，见
  [environment_prior.py](/home/chen/RLPFN/ticl/ticl/priors/environment_prior.py#L1182)

因此它能严谨支持的是“不会无限放大”，而不是“自动不消失”。

### 4.5 训练信号

下一版 `v5` 不把 `g1` 作为必须成功的主前提。
只定义一个最终注入信号：

$$
\hat g_t = \alpha_t g_t^{(1)} + (1-\alpha_t) g_t^{(0)},
$$

其中

- `alpha_t in [0,1]`
- `alpha_t` 只能依赖 detached 的统计量
- 若某一支 invalid，则对应权重置零
- 若两支都 valid，则权重归一化

最终注入到策略头的信号为

$$
\tilde g_t = c_t^{grad}\,\hat g_t,
$$

其中

$$
c_t^{grad} = c_{t,hi}^{grad} \cdot c_{t,lo}^{grad},
$$

且同样要求 detached。

---

## 5. 可接受的假设

这里只保留工程上仍然可能验证或监测的假设。

### A1. 全局上界假设

在安全域 `S_safe` 上，

$$
\left\|\frac{\partial F_\theta}{\partial s}\right\|_{op} \le L_s,
\qquad
\left\|\frac{\partial F_\theta}{\partial a}\right\|_{op} \le L_a,
\qquad
\left\|\frac{\partial \pi_\theta}{\partial \theta}\right\|_{op} \le L_\theta,
$$

且奖励斜率满足

$$
\left\|\frac{\partial r_t}{\partial s_t}\right\|_2 \le L_r.
$$

这由 `lipschitz_enforce` 和参数更新预算共同支撑。

### A2. 方向走廊假设

沿真实训练路径，对每个 `t` 都有

$$
\rho_t \le \rho_{hi},
$$

并且在激活事件 `Q_t^{state}=1` 上有

$$
\rho_t \ge \rho_{lo} > 0.
$$

这不是全矩阵下界，而只是沿真实更新方向的下界。
它明显弱于旧版 `sigma_min(A_t) >= c`，也更可工程化。

### A3. 条件性非稳定点假设

定义事件 `Q_t^{state}=1` 表示：

1. 当前不在稳定点附近
2. 当前统计可信
3. 当前增量不是纯噪声

形式上，只要求在 `Q_t^{state}=1` 上存在

$$
d_{t-1} \ge d_{min} > 0.
$$

### A4. 注入信号局部可辨识性假设

对需要学习的参数子空间 `U_t`，定义局部 VJP 映射

$$
J_t = \frac{\partial \mu_t}{\partial \theta}\bigg|_{U_t},
$$

其中 `mu_t` 是策略头输出。

我们不要求全局 `sigma_min(J_t)` 有下界。
只要求在激活事件 `Q_t^{grad}=1` 上，

$$
\sigma_{min}(J_t) \ge m_J > 0.
$$

这是一条**条件性可辨识性假设**。
没有它，就无法严谨证明“参数梯度不消失”。

### A5. estimator 条件无偏性

设 `G_t` 是用于决定 `alpha_t` 和 validity gate 的 detached 统计量所生成的 sigma-代数。

假设在 controlled dynamics 下，

$$
\mathbb{E}[g_t^{(1)} \mid G_t] = \nabla J_t^c,
\qquad
\mathbb{E}[g_t^{(0)} \mid G_t] = \nabla J_t^c.
$$

也就是：

- `g1` 和 `g0` 对**同一个 controlled objective** 条件无偏；
- `alpha_t` 只依赖 `G_t`，而不依赖当前 infinitesimal perturbation。

这是后面证明“加权本身不引入额外偏差”的关键。

---

## 6. 梯度爆炸的证明

### 定理 1. 方向增量不会在窗口内爆炸

若对所有 `t=1,...,H` 有

$$
\rho_t \le \rho_{hi},
$$

则对任意 `1 \le s < t \le H`，

$$
d_t \le \rho_{hi}^{\,t-s} d_s.
$$

#### 证明

由定义

$$
d_u = \rho_u d_{u-1}, \qquad u=s+1,\dots,t.
$$

连乘得

$$
d_t = \left(\prod_{u=s+1}^{t} \rho_u \right)d_s
\le \rho_{hi}^{\,t-s} d_s.
$$

证毕。

### 推论 1. `H=32` 时的累计上界

对 `H=32`，

$$
d_t \le \Gamma_{hi} d_s,
\qquad
\Gamma_{hi} := \rho_{hi}^{32}.
$$

因此，只要先选定允许的累计放大上界 `Gamma_hi`，
就可反推出每步上界：

$$
\rho_{hi} = \Gamma_{hi}^{1/32}.
$$

这是下一版 `v5` 的参数设定公式，而不是经验拍值。

### 定理 2. 注入梯度不会无条件爆炸

若

$$
\|\hat g_t\|_2 \le G_{raw,max},
\qquad
c_t^{grad} \le c_{hi}^{grad} \le 1,
$$

则

$$
\|\tilde g_t\|_2 \le G_{raw,max}.
$$

进一步，若 A1 成立，则参数梯度满足

$$
\|\nabla_\theta J_H\|_2

\le
L_\theta L_r \sum_{t=1}^{H} \rho_{hi}^{\,H-t}.
$$

#### 证明

第一步，由定义

$$
\tilde g_t = c_t^{grad}\hat g_t,
$$

且 `c_t^{grad} <= 1`，故

$$
\|\tilde g_t\|_2 \le \|\hat g_t\|_2 \le G_{raw,max}.
$$

第二步，由 A1，

$$
\left\|\frac{\partial \mu_t}{\partial \theta}\right\|_{op} \le L_\theta,
$$

奖励斜率有界，

$$
\left\|\frac{\partial r_H}{\partial s_H}\right\|_2 \le L_r.
$$

再由定理 1，through-time 传输最多放大为 `rho_hi^(H-t)`，于是

$$
\|g_t\|_2 \le L_\theta L_r \rho_{hi}^{H-t}.
$$

对 `t` 求和即得。

证毕。

### 说明

这里的爆炸证明是**无条件上界证明**。
它不需要任何下界假设。
因此它是下一版 `v5` 最可信、也最该优先保住的部分。

---

## 7. 梯度消失的谨慎证明

### 7.1 不能证明什么

必须先明确：

仅凭 `lipschitz_enforce` 的上界，不可能严谨推出“参数梯度永不消失”。

原因很简单：

- 上界不提供下界；
- 接近最优点时，真实梯度本来就应当趋近于零；
- 若 `J_t` 的局部可辨识性退化，任何低侧 boost 都无法从精确零中创造出真实梯度。

因此，下一版 `v5` 只能证明**条件性非消失**，不能证明全局永不消失。

### 定理 3. 方向增量在激活集上不消失

若 A2、A3 成立，则在 `Q_t^{state}=1` 上，

$$
d_t \ge \rho_{lo}^{\,t-s} d_s \ge \rho_{lo}^{\,t-s} d_{min}.
$$

#### 证明

在 `Q_u^{state}=1` 上，A2 给出

$$
\rho_u \ge \rho_{lo}.
$$

于是

$$
d_t = \left(\prod_{u=s+1}^{t}\rho_u\right)d_s
\ge \rho_{lo}^{\,t-s} d_s
\ge \rho_{lo}^{\,t-s} d_{min}.
$$

证毕。

### 定理 4. 注入梯度在激活集上有条件性下界

设在 `Q_t^{grad}=1` 上，

$$
\|\hat g_t\|_2 \ge g_{raw,min} > 0,
$$

并且低侧增益满足

$$
1 \le c_t^{grad} \le \kappa_{grad}.
$$

则

$$
\|\tilde g_t\|_2 \ge g_{raw,min}.
$$

若低侧规则被设计为当 `||hat g_t|| < G_lo` 时把其提升到不小于 `G_lo`，
则进一步可得

$$
\|\tilde g_t\|_2 \ge G_{lo}.
$$

#### 证明

由定义

$$
\tilde g_t = c_t^{grad}\hat g_t.
$$

当 `c_t^{grad} >= 1` 时，

$$
\|\tilde g_t\|_2 \ge \|\hat g_t\|_2 \ge g_{raw,min}.
$$

若低侧规则显式把其抬到 `G_lo`，则结论成立。

证毕。

### 定理 5. 参数梯度在激活集上有条件性下界

若 A4 成立，并且 `Q_t^{grad}=1` 上

$$
\|\tilde g_t\|_2 \ge G_{lo},
$$

则对参数子空间 `U_t` 有

$$
\left\|J_t^\top \tilde g_t\right\|_2 \ge m_J G_{lo}.
$$

#### 证明

由奇异值下界不等式，

$$
\|J_t^\top \tilde g_t\|_2
\ge \sigma_{min}(J_t^\top)\|\tilde g_t\|_2
= \sigma_{min}(J_t)\|\tilde g_t\|_2
\ge m_J G_{lo}.
$$

证毕。

### 说明

这就是下一版 `v5` 对“防消失”的真实形式：

- 不是对所有时刻都给下界；
- 不是对全局矩阵都给下界；
- 而是在“当前值得学、且局部仍可辨识”的激活集上给条件性下界。

这比旧版 `sigma_min(A/B/P)` 全局假设弱得多，但更诚实，也更可实现。

---

## 8. 偏差控制的谨慎证明

### 8.1 总偏差分解

记

- `J`：原始动力学下的目标
- `J^c`：state corridor 与 lipschitz projection 后的 controlled objective
- `hat G`：算法最终使用的随机梯度

则总偏差分解为

$$
\mathbb{E}[\hat G] - \nabla J
=
\underbrace{\left(\mathbb{E}[\hat G] - \nabla J^c\right)}_{\text{estimator / invalid / sanitize 偏差}}
+
\underbrace{\left(\nabla J^c - \nabla J\right)}_{\text{controlled dynamics 偏差}}.
$$

下面分别证明。

### 定理 6. 若两支 estimator 条件无偏，则加权本身不引入额外偏差

若 A5 成立，且

$$
\alpha_t \in [0,1], \qquad \alpha_t \text{ 是 } G_t\text{-可测},
$$

则

$$
\mathbb{E}\left[\alpha_t g_t^{(1)} + (1-\alpha_t)g_t^{(0)}\right]
= \nabla J_t^c.
$$

#### 证明

对 `G_t` 条件化：

$$
\mathbb{E}\left[\alpha_t g_t^{(1)} + (1-\alpha_t)g_t^{(0)} \mid G_t\right]
$$

$$
=
\alpha_t \mathbb{E}[g_t^{(1)}\mid G_t]
+
(1-\alpha_t)\mathbb{E}[g_t^{(0)}\mid G_t]
$$

$$
= \alpha_t \nabla J_t^c + (1-\alpha_t)\nabla J_t^c
= \nabla J_t^c.
$$

再取全期望即得。

证毕。

### 结论 1

只要两支 estimator 对同一 controlled objective 条件无偏，
那么 `g1/g0` 的随机加权本身**不产生额外偏差**。

因此，权重设计应当尽量只解决方差和数值稳定性，
不要再引入额外的、与 estimator 本身无关的结构偏差。

### 定理 7. invalid 回退的偏差上界

定义 invalid 事件

$$
I_t = \{\text{两支 estimator 都 invalid}\}.
$$

设 invalid 时使用回退 `r_t`，则

$$
\hat g_t = 1_{I_t^c}\bar g_t + 1_{I_t}r_t,
$$

其中 `bar g_t` 是有效支加权后的估计器。

若存在统一上界

$$
\|\bar g_t\|_2 \le G_{max}, \qquad \|r_t\|_2 \le R_{max},
$$

则

$$
\left\|\mathbb{E}[\hat g_t] - \nabla J_t^c\right\|_2
\le
\mathbb{P}(I_t)\,(G_{max}+R_{max}).
$$

#### 证明

由定理 6，

$$
\mathbb{E}[\bar g_t] = \nabla J_t^c.
$$

于是

$$
\mathbb{E}[\hat g_t] - \nabla J_t^c
= \mathbb{E}[1_{I_t}(r_t - \bar g_t)].
$$

取范数并用 Jensen 不等式：

$$
\left\|\mathbb{E}[1_{I_t}(r_t - \bar g_t)]\right\|_2
\le
\mathbb{E}[1_{I_t}\|r_t - \bar g_t\|_2]
\le \mathbb{P}(I_t)(R_{max}+G_{max}).
$$

证毕。

### 说明

这说明下一版 `v5` 不应采用“大量样本直接丢弃”的方案。
否则 `P(I_t)` 变大，偏差会线性放大。

### 定理 8. controlled dynamics 偏差上界

定义每步动力学误差

$$
e_t := s_{t+1}^c - s_{t+1}^{raw}
= (c_t^{state}-1)\Delta_t^{raw} + \epsilon_t^{proj},
$$

其中 `epsilon_t^{proj}` 表示 `lipschitz_enforce` 投影带来的额外一步误差。

若原始闭环动力学对状态是 `L_s`-Lipschitz，
且回报对轨迹是 `L_R`-Lipschitz，则

$$
\| \nabla J^c - \nabla J \|_2
\le
C_{traj}
\sum_{t=1}^{H}
L_s^{H-t}
\mathbb{E}\|e_t\|_2,
$$

其中 `C_traj` 是把轨迹误差转成梯度误差的 Lipschitz 常数。

#### 证明思路

由离散 Gronwall 型递推，

$$
\|s_t^c - s_t^{raw}\|_2
\le
\sum_{u=1}^{t-1} L_s^{t-1-u}\|e_u\|_2.
$$

再由目标对轨迹和梯度的 Lipschitz 性，可把轨迹误差上界转成梯度误差上界。

证毕。

### 结论 2

state corridor 的偏差与

$$
\sum_t |c_t^{state}-1|\,\|\Delta_t^{raw}\|
$$

成正相关。

因此下一版 `v5` 必须满足：

- 高侧控制只在必要时启用；
- 低侧保护只在 `Q_t=1` 时启用；
- `c_t^{state}` 应尽量靠近 1。

---

## 9. `TBPTT = 32` 的参数选择公式

这一节只给公式，不给拍脑袋常数。

### 9.1 高侧参数

若允许窗口累计放大不超过 `Gamma_hi`，则

$$
\rho_{hi} = \Gamma_{hi}^{1/32}.
$$

例如：

- 若允许 `Gamma_hi = 2`，则 `rho_hi = 2^{1/32}`
- 若允许 `Gamma_hi = 4`，则 `rho_hi = 4^{1/32}`

### 9.2 低侧参数

若希望激活集上的有效信号在 32 步后至少保留到 `Gamma_lo`，
则低侧目标可设为

$$
\rho_{lo} = \Gamma_{lo}^{1/32}.
$$

但必须记住：

- 这只在 `Q_t=1` 上有意义；
- 若当前已接近稳定点，则不应强行施加该下界。

### 9.3 偏差预算

若允许 controlled dynamics 偏差不超过 `B_dyn`，
则可反推 state corridor 总偏离预算：

$$
\sum_{t=1}^{32}\mathbb{E}|c_t^{state}-1|\,\|\Delta_t^{raw}\|
\le \frac{B_{dyn}}{C_{traj}\sum_{t=1}^{32}L_s^{32-t}}.
$$

这条式子比“scale_lo/scale_hi 直接拍一个数”更严谨。

---

## 10. 对下一版 `v5` 的结论

### 10.1 可以被严谨证明的部分

1. 在 `lipschitz_enforce + high-side corridor` 下，
   through-time 放大有统一上界。

2. 若 `g1/g0` 对 controlled objective 条件无偏，
   则随机加权本身不引入额外偏差。

3. invalid 回退的偏差与 invalid 概率线性相关。

4. state corridor 带来的目标偏差可显式上界。

### 10.2 只能作条件性结论的部分

1. “参数梯度不消失”只能在激活集上证明。
2. 这需要：
   - 不在稳定点附近
   - 注入信号本身非零
   - 局部 VJP 映射仍可辨识

这不是证明的缺陷，而是问题本身如此。

### 10.3 应淘汰的抓手

1. `v7 highway`
   因为它把问题投到人为子空间，且没有证据表明这是根因结构。

2. `TBPTT` 上的二分/递归隔离坏步
   因为它把统一稳定问题做成了 case-by-case 救火。

3. 旧版 `v5` 文档中的全局 `sigma_min(A/B/P)` 承诺
   因为这在工程上无法严谨支撑。

---

## 11. 下一步约束

在本文证明框架下，真正允许进入实现阶段的下一版 `v5` 只能是：

1. `full-state corridor`
2. `lipschitz_enforce`
3. `simple g1/g0 validity gate + detached weighting`
4. `step-level AGC / trust-region`

且必须满足：

- 不使用 `highway` 子空间
- 不使用时间步 case-by-case 搜索
- 不以整体 batch skip 作为主稳定机制

如果后续实现不满足这三条，就不应被称为本文证明过的下一版 `v5`。

---

## 12. train-step 级别的闭合

上文已经给了 `policy step / through-time` 的上界和条件性下界。
这里把 `train step` 级别也补闭合。

记优化器在一次 step 前的参数梯度为 `g_k`，学习率为 `eta_k > 0`，
并定义近似更新量

$$
u_k = \eta_k g_k.
$$

下一版 `v5` 的 step corridor 不直接修改目标，只对 `u_k` 施加 detached 的统一缩放：

$$
\tilde u_k = c_k^{step} u_k,
\qquad
c_k^{step} = c_{k,hi}^{step} \cdot c_{k,lo}^{step}.
$$

其中：

- `0 < c_{k,hi}^{step} <= 1`
- `1 <= c_{k,lo}^{step} <= kappa_step`
- `c_{k,lo}^{step}` 只在训练仍处于激活集时启用

训练激活集条件记为 `Q_k^{step}=1`，要求：

1. `objective_abs >= objective_abs_min`
2. 或 `low_active_share >= low_active_share_min`
3. 且 `eta_k` 没有衰减到 0

于是有：

### 结论 3

若

$$
\|g_k\|_2 \le G_{hi},
$$

则高侧 AGC 和 update budget 保证

$$
\|\tilde u_k\|_2 \le B_{hi}^{step}.
$$

这给出 train-step 级别的无条件防爆炸上界。

### 结论 4

若在 `Q_k^{step}=1` 上有

$$
\|g_k\|_2 \ge G_{lo} > 0,
\qquad
\eta_k \ge \eta_{min} > 0,
$$

且 step corridor 启用低侧补偿，则

$$
\|\tilde u_k\|_2
\ge
\min\!\left(B_{lo}^{step}, \eta_{min} G_{lo}\right).
$$

因此，下一版 `v5` 不是只保证“参数梯度条件性不消失”，
而是进一步保证：

- 在训练激活集上，实际参数更新也条件性不消失；
- 在非激活集上，不强行施加下界，避免把稳定点附近的小梯度误判成问题。

这使得 `policy step` 和 `train step` 两层都闭合。

---

## 13. 更贴近文档的工程实现

为了避免“证明一套、实现另一套”，工程实现只允许保留下面三层。

### 13.1 state corridor

状态更新采用

$$
s_{t+1} = s_t + c_t^{state}\Delta_t^{raw},
$$

其中：

- 高侧根据 `rho_t` 和 `update_rms_t` 缩放；
- 低侧只在 `Q_t^{state}=1` 上补偿；
- 所有 scale 都 detached。

这对应代码中的 `aev5_next state corridor`。

### 13.2 old `v5` thermostat 的定位

现有代码版 `v5` 只能视为一个历史保留项：

- 它是 reward-std thermostat；
- 它不等价于本文的 `v5_next`；
- 它不能单独承担 through-time 稳定性证明。

因此主线方法名必须区分：

- `v5`: thermostat
- `v5_next`: full-state corridor + lipschitz + step corridor

### 13.3 step corridor

train step 只允许：

1. 全局 AGC 上界
2. 全局 update budget 上界
3. 激活集上的条件性低侧 boost

不允许：

- blockwise/timewise 搜索
- batch 内 case-by-case 分裂
- 用 batch skip 代替主控制器

### 13.4 默认 `TBPTT=32`

对当前实现，默认 `TBPTT=32` 的主线应当是：

1. `anti_explosion_vanishing_v5_next_enabled = True`
2. `lipschitz_enforce = True`
3. `anti_explosion_vanishing_v5_enabled = False`
4. `anti_explosion_vanishing_v6_enabled = False`
5. `anti_explosion_vanishing_v7_enabled = False`

并采用原始默认 corridor：

1. `state_gain_lo / state_gain_hi = 0.985 / 1.035`
2. `state_rms_lo / state_rms_hi = 4e-3 / 9e-2`
3. `state_low_boost_cap = 1.5`
4. `loss_scale_lo / loss_scale_hi = 0.5 / 4.0`
5. `step_grad_rms_lo / step_grad_rms_hi = 1e-4 / 3e-2`
6. `step_low_boost_cap = 4.0`

这才与本文证明对象一致。
