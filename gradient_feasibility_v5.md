# Gradient Feasibility v5: 方法、数学工具与证明

## 1. 目标与范围

本文只讨论一个问题：在可微环境与策略闭环中，如何构造 `v5 + lipschitz_enforce`，使梯度在两层索引上都近似稳定。

- 第一层索引：$t$，表示**同一个 batch 内**的时间步，$t=1,2,\dots,H$。
- 第二层索引：$k$，表示**batch 间**的训练步，$k=0,1,2,\dots$。

我们要证明两件事。

1. 对每个训练步 $k$，对每个时间步 $t$，梯度既不消失也不爆炸（高概率）。
2. 对 batch 间训练步 $k$，梯度稳定性的概率界不随 $k$ 退化（统一上界），从而长期近似稳定。

同时保证优化方向正确：始终是对 $\sum_t r_t$ 做梯度提升。

---

## 2. 记号与系统定义

### 2.1 轨迹与目标

设窗口长度为 $H$（当前设置可取 $H=64$），batch 索引为 $k$。第 $k$ 个训练 batch 的回报定义为

$$
R_k(\theta)=\sum_{t=1}^{H} r_{k,t}.
$$

训练目标是最大化期望回报

$$
J_k(\theta)=\mathbb E[R_k(\theta)].
$$

实现时最小化损失

$$
\mathcal L_k(\theta)=-J_k(\theta)+\text{正则项}.
$$

因此方向不变：最小化损失等价于提升 $\sum_t r_{k,t}$。

### 2.2 动力学与策略雅可比矩阵

定义状态和动作的可微映射：

$$
s_{k,t+1}=F_\theta(s_{k,t},a_{k,t},\xi_{k,t}),
\qquad
a_{k,t}=\pi_\theta(h_{k,t}).
$$

定义关键雅可比矩阵

$$
A_{k,t}=\frac{\partial s_{k,t+1}}{\partial s_{k,t}},
\quad
B_{k,t}=\frac{\partial s_{k,t+1}}{\partial a_{k,t}},
\quad
P_{k,t}=\frac{\partial a_{k,t}}{\partial \theta}.
$$

### 2.3 伴随变量（costate）

定义窗口末端伴随变量

$$
\lambda_{k,H}=\frac{\partial R_k}{\partial s_{k,H}},
$$

并递推

$$
\lambda_{k,t}=\frac{\partial r_{k,t}}{\partial s_{k,t}} + A_{k,t}^\top\lambda_{k,t+1},
\quad t=H-1,\dots,1.
$$

时间步 $t$ 对参数的梯度片段为

$$
g_{k,t}=P_{k,t}^\top B_{k,t}^\top \lambda_{k,t+1}.
$$

总梯度为

$$
g_k=\nabla_\theta J_k(\theta)=\sum_{t=1}^{H} g_{k,t}.
$$

---

## 3. v5 + Lipschitz 的方法定义

v5 的核心不是调学习率，也不是重放次数，而是对梯度链路的关键因子做**可行域约束**。

### 3.1 约束一：状态与动作链路的奇异值带宽

对所有 $k,t$，要求

$$
\underline\sigma_A \le \sigma_{\min}(A_{k,t})\le \sigma_{\max}(A_{k,t}) \le \overline\sigma_A,
$$
$$
\underline\sigma_B \le \sigma_{\min}(B_{k,t})\le \sigma_{\max}(B_{k,t}) \le \overline\sigma_B,
$$
$$
\underline\sigma_P \le \sigma_{\min}(P_{k,t})\le \sigma_{\max}(P_{k,t}) \le \overline\sigma_P.
$$

### 3.2 约束二：伴随变量带宽

对所有 $k,t$，希望

$$
\underline\lambda \le \|\lambda_{k,t}\|_2 \le \overline\lambda.
$$

这可通过软约束实现（概率保证），例如

$$
\mathcal R_{\lambda,k}=\frac{1}{H}\sum_{t=1}^{H}
\left[
\operatorname{softplus}(\underline\lambda-\|\lambda_{k,t}\|_2)
+
\operatorname{softplus}(\|\lambda_{k,t}\|_2-\overline\lambda)
\right].
$$

### 3.3 Lipschitz enforce 的角色

`lipschitz_enforce` 保留，用于提供全局上界稳定性。对线性算子投影满足

$$
\|W\|_2 \le \|W\|_F \le C,
$$

并对高斯过程输出尺度施加上界，避免爆炸链路。

### 3.4 训练损失

定义

$$
\mathcal L_k(\theta)=
- J_k(\theta)
+\lambda_\lambda\mathcal R_{\lambda,k}
+\lambda_J\mathcal R_{J,k},
$$

其中 $\mathcal R_{J,k}$ 对奇异值越界做软约束或投影后的残差约束。

优化方向保持正确，因为首项仍是 $-J_k$。

---

## 4. 数学工具（逐一说明）

### 4.1 奇异值不等式

对任意矩阵 $M$ 与向量 $v$：

$$
\sigma_{\min}(M)\|v\|_2 \le \|Mv\|_2 \le \sigma_{\max}(M)\|v\|_2.
$$

对矩阵乘积 $MN$：

$$
\sigma_{\min}(MN)\ge \sigma_{\min}(M)\sigma_{\min}(N),
$$
$$
\sigma_{\max}(MN)\le \sigma_{\max}(M)\sigma_{\max}(N).
$$

### 4.2 条件概率与过滤信息

记 $\mathcal F_{k-1}$ 为第 $k$ 个 batch 更新前可观测历史。所有“高概率”结论都写为条件概率

$$
\mathbb P(\cdot\mid \mathcal F_{k-1}).
$$

### 4.3 Cantelli 不等式（单侧切比雪夫）

对随机变量 $X$，若均值 $\mu$、方差 $v$ 有界，则

$$
\mathbb P(X-\mu\le -a) \le \frac{v}{v+a^2},\quad a>0.
$$

这里用于控制“伴随变量下穿下界”或“累计对数增益下穿阈值”的概率。

### 4.4 可求和失败概率与长期结论

若失败事件 $E_k^c$ 满足

$$
\sum_{k=1}^{\infty}\mathbb P(E_k^c)<\infty,
$$

则失败只会发生有限次（几乎必然）。这给出“长期近似完全避免”的形式化含义。

---

## 5. 证明一：对每个 batch 的时间步 $t$，梯度既不消失也不爆炸

### 5.1 定理陈述

若在第 $k$ 个 batch 中，对每个 $t$ 满足

$$
\underline\sigma_P \le \sigma_{\min}(P_{k,t}),
\quad
\underline\sigma_B \le \sigma_{\min}(B_{k,t}),
\quad
\|\lambda_{k,t+1}\|_2\ge \underline\lambda,
$$

以及

$$
\sigma_{\max}(P_{k,t})\le \overline\sigma_P,
\quad
\sigma_{\max}(B_{k,t})\le \overline\sigma_B,
\quad
\|\lambda_{k,t+1}\|_2\le \overline\lambda,
$$

则

$$
\underline g \le \|g_{k,t}\|_2 \le \overline g,
$$

其中

$$
\underline g = \underline\sigma_P\underline\sigma_B\underline\lambda,
\qquad
\overline g = \overline\sigma_P\overline\sigma_B\overline\lambda.
$$

### 5.2 证明

由定义

$$
g_{k,t}=P_{k,t}^\top B_{k,t}^\top\lambda_{k,t+1}.
$$

下界：

$$
\|g_{k,t}\|_2
\ge \sigma_{\min}(P_{k,t}^\top)\cdot\|B_{k,t}^\top\lambda_{k,t+1}\|_2
\ge \sigma_{\min}(P_{k,t})\sigma_{\min}(B_{k,t})\|\lambda_{k,t+1}\|_2
\ge \underline g.
$$

上界：

$$
\|g_{k,t}\|_2
\le \sigma_{\max}(P_{k,t})\sigma_{\max}(B_{k,t})\|\lambda_{k,t+1}\|_2
\le \overline g.
$$

证毕。

### 5.3 概率版本

若上述条件以条件概率成立：

$$
\mathbb P(E_k\mid\mathcal F_{k-1})\ge 1-\delta_k,
$$

则

$$
\mathbb P\bigl(\underline g\le \|g_{k,t}\|_2\le \overline g,\ \forall t\mid\mathcal F_{k-1}\bigr)
\ge 1-\delta_k.
$$

---

## 6. 证明二：对 batch 间训练步 $k$，梯度既不消失也不爆炸

### 6.1 定理陈述

设每次参数更新后都投影回同一可行域 $\Theta_{\mathrm{safe}}$，使得

$$
\underline\sigma_P,\overline\sigma_P,
\underline\sigma_B,\overline\sigma_B,
\underline\lambda,\overline\lambda
$$

是与 $k$ 无关的全局常数。

再设

$$
\mathbb P(E_k\mid\mathcal F_{k-1})\ge 1-\delta_k,
$$

其中事件 $E_k$ 表示“第 $k$ 个 batch 内所有 $t$ 的带宽约束都成立”。

则对任意 $k$：

$$
\mathbb P\bigl(\underline g\le \|g_{k,t}\|_2\le \overline g,\ \forall t\bigr)
\ge 1-\delta_k.
$$

若进一步有 $\delta_k=\delta$（常数），则得到对所有 $k$ 的统一概率界。

若进一步满足

$$
\sum_{k=1}^{\infty}\delta_k<\infty,
$$

则几乎必然只有有限多个 batch 违例，得到长期近似完全稳定。

### 6.2 证明

第一步已经由第 5 节给出：在事件 $E_k$ 上，batch 内每个时间步都有双边界。  
第二步，因可行域常数不随 $k$ 变化，边界常数 $\underline g,\overline g$ 也不随 $k$ 变化。  
第三步，把条件概率与失败概率序列结合，即得每个 $k$ 的概率稳定界；若失败概率可求和，长期结论由“可求和失败概率原理”得到。

证毕。

---

## 7. 偏差控制与可容许范围

v5 相对原始目标的偏差来自三部分。

1. 可行域投影偏差（约束最优值差距）。
2. 软约束近似偏差（平滑参数带来的 $O(\tau)$）。
3. batch 统计估计偏差（样本方差，通常 $O(B^{-1/2})$）。

写成梯度偏差

$$
b_k=\nabla_\theta\mathcal L_k + \nabla_\theta J_k.
$$

若存在常数 $C_\lambda,C_J,C_{\mathrm{proj}}$ 使

$$
\|\nabla_\theta \mathcal R_{\lambda,k}\|_2\le C_\lambda,
\quad
\|\nabla_\theta \mathcal R_{J,k}\|_2\le C_J,
\quad
\|b_k^{\mathrm{proj}}\|_2\le C_{\mathrm{proj}},
$$

则

$$
\|b_k\|_2
\le
\lambda_\lambda C_\lambda + \lambda_J C_J + C_{\mathrm{proj}} + O(\tau)+O(B^{-1/2}).
$$

因此可以通过 $\lambda_\lambda,\lambda_J,\tau,B$ 把偏差压进可容许范围。

---

## 8. 一致性检查：矛盾与错误排查

### 8.1 已排除的常见逻辑错误

1. 把“方法不显含 $k$”误当成“梯度分布与 $k$ 无关”。  
   这里没有这样做；我们用的是“每步投影到同一可行域 + 每步概率约束”来得到统一界。

2. 把 batch 内时间步 $t$ 与 batch 间训练步 $k$ 混淆。  
   本文已分层给出两个定理，分别处理 $t$ 和 $k$。

3. 把优化方向写反。  
   本文始终是最小化 $-J_k$，即提升 $\sum_t r_{k,t}$。

### 8.2 仍需显式假设的地方

1. 奖励端非退化：$\|\partial R_k/\partial s_{k,H}\|_2$ 不能长期退化到零。  
2. 可行域非空：奇异值上下界必须物理可实现。  
3. 统计估计质量：batch 大小不足时，$\delta_k$ 与方差估计会恶化。

这些不是矛盾，而是该类概率保证不可避免的前提。

---

## 9. 结论

在上述定义与假设下，`v5 + lipschitz_enforce` 可以给出：

1. 对每个训练步 $k$ 的每个时间步 $t$，梯度双边界（不消失、不爆炸）高概率成立。  
2. 对 batch 间训练步 $k$，同一组边界常数和概率界可统一适用；若失败概率可求和，长期仅有限次违例。

这满足“概率近似正确地长期避免梯度消失与爆炸”的目标，并且偏差可被显式控制。
