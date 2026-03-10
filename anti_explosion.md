# anti-explosion: v2 到 v4 加 Lipschitz 的严格性质检查与证明

## 0. 文档目标

本文只做一件事：对以下两条性质，给出清晰、可核查的数学论证。

1. 随着训练步数增大，方法是否仍然具有“以概率近似正确”的防止梯度消失作用。
2. 这种防止梯度消失的性质，是否对训练步数满足马尔科夫齐次性。

讨论对象是四种机制：

- 版本二（代码开关 `anti_explosion_vanishing_v2_enabled`）
- 版本三（代码开关 `anti_explosion_vanishing_v3_enabled`）
- 版本四（代码开关 `anti_explosion_vanishing_v4_enabled`）
- 版本四与 Lipschitz 约束同时开启（代码开关 `lipschitz_enforce`）


## 1. 系统定义与记号

### 1.1 状态转移主方程

根据当前实现，单步状态更新可写成

$$
\tilde s_{t+1} = (1-\alpha)s_t + \alpha x_{t+1}(s_t,a_t,\xi_t) + \varepsilon_t,
$$

其中 $\alpha\in(0,1]$，$x_{t+1}$ 由环境生成器给出，$\xi_t$ 是重参数化噪声，$\varepsilon_t$ 是额外状态噪声。

经过裁剪与双曲正切函数后，得到

$$
\hat s_{t+1} = \tanh\!\left(\mathrm{clip}(\tilde s_{t+1})\right).
$$

若版本四开启，在高速子空间再做一次受控更新：

$$
s_{t+1}^{\mathrm{highway}} = s_t^{\mathrm{highway}} + u\bigl(\hat s_{t+1}^{\mathrm{highway}} - s_t^{\mathrm{highway}}\bigr),
$$

其中 $u\in(0,1]$ 是 `update_scale`，其余坐标保持 $\hat s_{t+1}$。

### 1.2 增量与增益记号

定义状态增量

$$
\Delta s_t = s_t - s_{t-1}.
$$

定义相邻增量的范数增益

$$
g_t = \frac{\|\Delta s_t\|_2}{\|\Delta s_{t-1}\|_2 + \epsilon},
\qquad
z_t = \log(g_t),
$$

其中 $\epsilon>0$ 来自实现中的数值稳定项。

### 1.3 训练目标与截断窗口

当前使用截断时间反向传播，窗口长度记为 $H$，你的设置是 $H=64$。每个优化步的目标写成

$$
J_H(\theta)=\mathbb E\!\left[\sum_{t=1}^{H} r_t\right],
$$

并用路径导数进行反向传播。


## 2. 会用到的数学工具

### 2.1 奇异值与雅可比矩阵连乘

对可微映射 $f$，记雅可比矩阵为 $J_f$。对任意矩阵 $A$：

- 最大奇异值记为 $\sigma_{\max}(A)$。
- 最小奇异值记为 $\sigma_{\min}(A)$。

梯度链式传播中的关键量是连乘雅可比。若每步最小奇异值都有下界 $m>0$，则长度为 $H$ 的连乘最小奇异值下界是 $m^H$。

### 2.2 对数变换把乘法过程变成加法过程

由定义可得

$$
\|\Delta s_t\|_2
= \|\Delta s_1\|_2\exp\!\left(\sum_{k=2}^{t} z_k\right).
$$

因此，控制 $\sum z_k$ 就是在控制增量范数的指数衰减。

### 2.3 条件次高斯尾界与伯恩斯坦不等式

若随机变量序列在条件意义下是次高斯，则可用伯恩斯坦型界控制

$$
\mathbb P\!\left(\sum_{k=1}^{H}(z_k-\mathbb E[z_k\mid \mathcal F_{k-1}])\le -\eta\right)
$$

这给出“概率近似正确”的定量形式。

### 2.4 马尔科夫齐次性

把训练状态记为

$$
X_k=(\theta_k, m_k, v_k, \text{其他优化器状态}),
$$

若存在与 $k$ 无关的转移核 $K$，使得

$$
\mathbb P(X_{k+1}\in A\mid X_k)=K(X_k,A),
$$

则称对训练步数马尔科夫齐次。


## 3. 方法的形式化描述与质量联系

本节先把四个方法写成统一数学对象，再进入后续性质证明。

### 3.1 基准优化问题

不加任何 anti-explosion 机制时，窗口目标是

$$
J_H(\theta)=\mathbb E_\theta\left[\sum_{t=1}^{H} r_t\right].
$$

等价损失写成

$$
\mathcal L_{\mathrm{base}}(\theta)=-J_H(\theta).
$$

状态转移算子记为

$$
T_{\mathrm{base}}:\ (s_t,a_t,\xi_t)\mapsto s_{t+1}.
$$

---

### 3.2 版本二的形式化定义

定义

$$
z_t=\log\frac{\|\Delta s_t\|_2}{\|\Delta s_{t-1}\|_2+\epsilon},
\qquad
z_{\min}=\log(g_{\min}),\quad z_{\max}=\log(g_{\max}).
$$

单步走廊违例

$$
v_t=\max(0,z_t-z_{\max})+\max(0,z_{\min}-z_t).
$$

用 Huber 函数 $\psi_\delta(\cdot)$ 平滑后，窗口正则项

$$
\mathcal R_{\mathrm{v2}}=\lambda_2\cdot\frac{1}{H-1}\sum_{t=2}^{H}\psi_\delta(v_t).
$$

版本二优化目标

$$
\mathcal L_{\mathrm{v2}}=\mathcal L_{\mathrm{base}}+\mathcal R_{\mathrm{v2}}.
$$

注意版本二不改状态算子，仍是 $T_{\mathrm{base}}$。

---

### 3.3 版本三的形式化定义

版本三仍使用 $T_{\mathrm{base}}$，但正则项改为“漂移加尾部”：

$$
\bar z=\frac{1}{H-1}\sum_{t=2}^{H} z_t,
$$

$$
\phi_{\tau}(z_t)=\tau\,\mathrm{softplus}\!\left(\frac{z_{\min}-z_t}{\tau}\right)
+\tau\,\mathrm{softplus}\!\left(\frac{z_t-z_{\max}}{\tau}\right).
$$

$$
\mathcal R_{\mathrm{v3}}
=\lambda_{\mathrm{drift}}\bar z^2
+\lambda_{\mathrm{tail}}\cdot\frac{1}{H-1}\sum_{t=2}^{H}\phi_{\tau}(z_t).
$$

$$
\mathcal L_{\mathrm{v3}}=\mathcal L_{\mathrm{base}}+\mathcal R_{\mathrm{v3}}.
$$

---

### 3.4 版本四的形式化定义

版本四由两部分组成。

1. 结构更新：在高速子空间施加受控残差步长
$$
s_{t+1}^{\mathrm{highway}}=s_t^{\mathrm{highway}}+u\left(\hat s_{t+1}^{\mathrm{highway}}-s_t^{\mathrm{highway}}\right),
$$
并可选裁剪残差幅度（`update_clip`）。

2. 统计正则：与版本三同型，但统计对象是版本四更新后的增量序列。

记版本四状态算子为 $T_{\mathrm{v4}}$，版本四正则为 $\mathcal R_{\mathrm{v4}}$，则

$$
\mathcal L_{\mathrm{v4}}=-J_H^{(T_{\mathrm{v4}})}(\theta)+\mathcal R_{\mathrm{v4}}.
$$

---

### 3.5 版本四加 Lipschitz 的形式化定义

开启 `lipschitz_enforce` 后，对线性映射做投影

$$
P_C(W)=\min\!\left(1,\frac{C}{\|W\|_F}\right)W,
$$

因此

$$
\|W\|_2\le\|W\|_F\le C.
$$

对高斯过程分支再施加输出尺度上界。于是得到受约束状态算子

$$
T_{\mathrm{v4+L}}.
$$

对应目标

$$
\mathcal L_{\mathrm{v4+L}}=-J_H^{(T_{\mathrm{v4+L}})}(\theta)+\mathcal R_{\mathrm{v4}}.
$$

---

### 3.6 四个方法的质量联系

可把四个方法视为逐层加强的链条：

$$
(\,T_{\mathrm{base}},\mathcal R_{\mathrm{v2}}\,)
\ \rightarrow\
(\,T_{\mathrm{base}},\mathcal R_{\mathrm{v3}}\,)
\ \rightarrow\
(\,T_{\mathrm{v4}},\mathcal R_{\mathrm{v4}}\,)
\ \rightarrow\
(\,T_{\mathrm{v4+L}},\mathcal R_{\mathrm{v4}}\,).
$$

对应质量提升关系是：

1. 从版本二到版本三：从“只管越界”升级到“均值漂移加尾部概率”。
2. 从版本三到版本四：从“纯统计约束”升级到“结构改造加统计约束”。
3. 从版本四到版本四加 Lipschitz：在结构和统计之外再加入“雅可比上界约束”，使稳定性论证可闭合。

因此后续第 4 节到第 8 节的证明，是沿这条链条逐层增强展开的。


## 4. 性质一：随训练步数增大时的“概率近似防消失”

## 4.1 版本二不满足该性质

**命题一**：版本二不能保证长期防止梯度消失。

**证明**：

取任意常数 $\rho\in(0,1)$，并满足版本二走廊约束 $g_{\min}<\rho<g_{\max}$。令所有步都有 $g_t=\rho$，则每步都在走廊内，版本二惩罚恒为零。

但是

$$
\|\Delta s_t\|_2=\|\Delta s_1\|_2\rho^{t-1},
$$

当 $t$ 增大时指数衰减到零，链式梯度同样指数衰减。故版本二无法排除“走廊内稳定衰减”。命题得证。


## 4.2 版本三只能给相对尺度上的概率保证

**假设一**：在每个优化步对应的数据分布下，$z_t$ 条件次高斯，参数为 $\sigma^2$。

**假设二**：版本三训练后满足

$$
|\mathbb E[z_t]|\le \delta_d,
$$

并且双侧尾部事件总概率不超过 $\delta_{\mathrm{tail}}$。

**命题二**：在窗口 $H$ 内，版本三给出相对增量下界的高概率保证：对任意 $\eta>0$，

$$
\mathbb P\!\left(\sum_{k=2}^{H} z_k \le -\eta\right)
\le
\exp\!\left(-\frac{(\eta-(H-1)\delta_d)^2}{2(H-1)\sigma^2}\right)
+ (H-1)\delta_{\mathrm{tail}}.
$$

于是

$$
\mathbb P\!\left(\|\Delta s_H\|_2
\ge
\|\Delta s_1\|_2 e^{-\eta}
\right)
\ge 1-\text{右侧上界}.
$$

**说明**：这是“相对 $\|\Delta s_1\|$ 的防消失”，不是“绝对正下界”。若 $\|\Delta s_1\|$ 已经很小，版本三不能独立修复。


## 4.3 版本四提高了结构层面的可保性

版本四在高速子空间的雅可比块是

$$
J_{\mathrm{highway}} = (1-u)I + uJ_{\hat s},
$$

其中 $u\in(0,1]$。

若无 Lipschitz 上界，$\|J_{\hat s}\|_2$ 仍可能非常大，最小奇异值下界无法固定为正数，因此“长期防消失”仍不稳固。

结论：版本四比版本三更强，但单独使用时不构成可靠充分条件。


## 4.4 版本四加 Lipschitz 可以给出更强结论

**假设三**：在高速子空间有

$$
\|J_{\hat s}\|_2\le L,
$$

且满足

$$
L<\frac{1-u}{u}.
$$

**引理一**：

$$
\sigma_{\min}(J_{\mathrm{highway}})
\ge 1-u-uL=:m>0,
$$

$$
\sigma_{\max}(J_{\mathrm{highway}})
\le 1-u+uL=:M.
$$

**证明**：对任意单位向量 $x$，

$$
\|J_{\mathrm{highway}}x\|_2
=\|(1-u)x+uJ_{\hat s}x\|_2
\ge (1-u)-u\|J_{\hat s}x\|_2
\ge 1-u-uL.
$$

上界由三角不等式直接得到。引理得证。

**定理一**：在固定窗口 $H$ 下，版本四加 Lipschitz 在单次窗口反向传播上给出结构性非零下界。

设奖励对状态的局部梯度下界为 $c_r>0$，策略参数到初始动作链路下界为 $c_\theta>0$，则存在

$$
\|\nabla_\theta J_H\|_2
\ge c_r c_\theta m^H.
$$

由于 $m>0$ 且 $H=64$ 固定，上式给出的是“局部一次更新”的结构下界。  
它本身不直接推出“随优化步编号 $k$ 增大时，期望梯度消失程度与 $k$ 无关”。

再叠加版本四的漂移与尾部概率约束，可以得到“结构下界加统计回拉”的联合防消失结论。  
关于“与训练步编号 $k$ 的关系”，见第 5 节的严格修正。


## 5. 性质二：对训练步数的马尔科夫齐次性（严格修正）

你提出的反驳是正确的：  
“马尔科夫齐次”不等于“梯度消失程度与训练步编号 $k$ 无关”。

本节给出严格区分。

### 5.1 两个不同命题

**命题 A（马尔科夫齐次）**：训练状态过程有与 $k$ 无关的转移核。  
**命题 B（消失程度对 $k$ 不变）**：梯度消失指标 $D_k$ 的分布或期望不随 $k$ 改变。

其中梯度消失指标可定义为

$$
D_k=-\log\!\left(\|\nabla_\theta J_H(\theta_k)\|_2+\epsilon\right).
$$

命题 A 与命题 B 没有逻辑等价关系。

---

### 5.2 为什么“齐次”不推出“与 $k$ 无关”

设训练状态 $X_k$ 满足齐次马尔科夫链

$$
X_{k+1}\sim K(X_k,\cdot),
$$

但初始分布为 $\mu_0$，则

$$
\mu_k=\mu_0K^k.
$$

即使 $K$ 与 $k$ 无关，$\mu_k$ 仍一般随 $k$ 变化。  
因此对任意观测函数 $h$（例如 $h(X_k)=D_k$），

$$
\mathbb E[D_k]=\int h(x)\,\mu_k(dx)
$$

通常仍与 $k$ 有关。

这正是你指出的问题：学习过程即便“不知道自己在第几步”，统计量依旧可能随步数演化。

---

### 5.3 什么时候才能近似“与 $k$ 无关”

要得到“梯度消失程度长期近似不依赖 $k$”，需要额外条件，而不仅是齐次性。

一种标准充分条件组合是：

1. 齐次马尔科夫链 $K$ 几何遍历（存在不变分布 $\pi$，且收敛速率 $\rho\in(0,1)$）。
2. 梯度消失指标函数 $h$ 在链上可积并有适度正则性。
3. 训练协议固定（例如固定 TBPTT 窗口、固定学习率规则、无按步触发的结构切换）。

则存在常数 $C>0$ 使

$$
\left|\mathbb E[D_k]-\mathbb E_\pi[D]\right|\le C\rho^k.
$$

这说明的是“趋于稳态且误差指数衰减”，不是“所有 $k$ 完全相同”。

---

### 5.4 对四种方法逐一检查

1. **版本二**  
没有结构性下界，且存在走廊内指数衰减反例。即使过程齐次，也不能阻止 $D_k$ 随 $k$ 恶化。

2. **版本三**  
提供漂移和尾部统计回拉，但不提供绝对尺度下界。可改善 $D_k$ 的趋势，不能单独保证 $k$-均匀防消失。

3. **版本四**  
通过高速子空间结构降低消失风险，但无 Lipschitz 约束时雅可比仍可能大幅波动，$D_k$ 仍可能随训练漂移。

4. **版本四加 Lipschitz**  
在结构改造和统计回拉外再给雅可比上界控制。若再叠加固定训练协议与遍历性条件，可得到
$$
\left|\mathbb E[D_k]-\mathbb E_\pi[D]\right|\le C\rho^k
$$
这类“长期概率近似稳定”结果。

---

### 5.5 本文修正结论

1. 先前“方法不显含 $k$，所以消失程度不随 $k$”的说法不严谨，已修正。  
2. 正确说法是：  
   - 齐次性只约束转移核形式；  
   - 消失程度是否与 $k$ 弱相关，要看分布演化与遍历收敛。  
3. 在你关心的标准下，最接近目标的是“版本四加 Lipschitz并配套固定训练协议”。


## 6. 质量审查与排序

### 6.1 审查标准

采用四个维度：

1. 是否有结构性正下界来防梯度消失。
2. 是否有概率界来控制统计波动。
3. 是否易于满足马尔科夫齐次前提。
4. 引入偏差是否可控。

### 6.2 逐项评分结论

1. 版本二：
   - 没有结构性正下界。
   - 有明显反例。
   - 质量不通过。

2. 版本三：
   - 有统计概率控制。
   - 没有绝对尺度下界。
   - 质量中等。

3. 版本四：
   - 有结构改造，优于版本三。
   - 没有雅可比全局上界时仍可能失效。
   - 质量中上。

4. 版本四加 Lipschitz：
   - 同时具备结构下界与统计控制。
   - 在固定窗口与固定训练协议下，最接近“长期概率近似防消失”。
   - 质量最高。

### 6.3 强度、质量、有效性排序

从高到低：

1. 版本四加 Lipschitz。
2. 版本四。
3. 版本三。
4. 版本二。


## 7. 最终结论

1. 若目标是“训练步增加后仍保持概率近似防消失”，首选版本四加 Lipschitz。
2. 若还要求“按训练步马尔科夫齐次”，必须把训练协议也固定，不只固定方法开关。
3. 版本二不能满足该目标，版本三只能提供相对尺度层面的统计保证。


## 8. 四个方法的偏差可控性分析与证明

### 8.1 偏差的统一定义

令基准目标为

$$
J_H(\theta)=\mathbb E\left[\sum_{t=1}^{H} r_t\right].
$$

若某方法实际优化目标为 $\widetilde J_H(\theta)$，则该方法引入的梯度偏差定义为

$$
b(\theta)=\nabla_\theta \widetilde J_H(\theta)-\nabla_\theta J_H(\theta).
$$

“偏差可控”定义为：存在可调参数向量 $\eta$，使得

$$
\|b(\theta)\|_2 \le B(\eta),
$$

且 $B(\eta)$ 对关键参数单调，并可通过调参使其进入任意给定容许区间。

---

### 8.2 版本二的偏差可控性

版本二等价于优化

$$
\widetilde J_H^{\mathrm{v2}}(\theta)=J_H(\theta)-\lambda_2\,\mathbb E[\Psi_2(\theta)],
$$

其中 $\Psi_2$ 为走廊违例的平滑惩罚（Huber 形式）。

因此

$$
b_{\mathrm{v2}}(\theta)=-\lambda_2\nabla_\theta \mathbb E[\Psi_2(\theta)].
$$

若在当前参数邻域内存在常数 $C_2$，满足

$$
\left\|\nabla_\theta \mathbb E[\Psi_2(\theta)]\right\|_2\le C_2,
$$

则有

$$
\|b_{\mathrm{v2}}(\theta)\|_2\le \lambda_2 C_2.
$$

**结论**：版本二偏差对 $\lambda_2$ 一阶线性可控，属于可控偏差。

---

### 8.3 版本三的偏差可控性

版本三目标为

$$
\widetilde J_H^{\mathrm{v3}}(\theta)
=J_H(\theta)-\lambda_{\mathrm{drift}}\,\mathbb E[\Psi_d(\theta)]
-\lambda_{\mathrm{tail}}\,\mathbb E[\Psi_t(\theta)].
$$

因此

$$
b_{\mathrm{v3}}(\theta)
=-\lambda_{\mathrm{drift}}\nabla_\theta \mathbb E[\Psi_d(\theta)]
-\lambda_{\mathrm{tail}}\nabla_\theta \mathbb E[\Psi_t(\theta)].
$$

若存在 $C_d,C_t$ 使得

$$
\left\|\nabla_\theta \mathbb E[\Psi_d(\theta)]\right\|_2\le C_d,
\qquad
\left\|\nabla_\theta \mathbb E[\Psi_t(\theta)]\right\|_2\le C_t,
$$

则

$$
\|b_{\mathrm{v3}}(\theta)\|_2
\le \lambda_{\mathrm{drift}}C_d+\lambda_{\mathrm{tail}}C_t.
$$

**结论**：版本三偏差对两组权重分别线性可控，也属于可控偏差。

---

### 8.4 版本四的偏差可控性

版本四有两类偏差来源。

1. 正则偏差：与版本三同型，可由
$$
\lambda_{\mathrm{drift}},\lambda_{\mathrm{tail}}
$$
线性控制。

2. 结构偏差：高速子空间更新把基准转移改为
$$
s_{t+1}^{\mathrm{highway}}=s_t^{\mathrm{highway}}+u\left(\hat s_{t+1}^{\mathrm{highway}}-s_t^{\mathrm{highway}}\right).
$$

设不使用版本四时高速子空间状态为 $\hat s_{t+1}^{\mathrm{highway}}$，则单步结构扰动为

$$
\delta^{\mathrm{struct}}_{t+1}
=s_{t+1}^{\mathrm{highway}}-\hat s_{t+1}^{\mathrm{highway}}
=(1-u)\left(s_t^{\mathrm{highway}}-\hat s_{t+1}^{\mathrm{highway}}\right).
$$

所以

$$
\|\delta^{\mathrm{struct}}_{t+1}\|_2
\le (1-u)\,\|s_t^{\mathrm{highway}}-\hat s_{t+1}^{\mathrm{highway}}\|_2.
$$

若奖励函数对状态是 $L_r$-Lipschitz，则窗口回报偏差满足

$$
|J_H^{\mathrm{v4}}-J_H^{\mathrm{base}}|
\le L_r\sum_{t=1}^{H}\mathbb E\|e_t\|_2,
$$

其中 $e_t=s_t^{\mathrm{v4}}-s_t^{\mathrm{base}}$。若状态误差传播满足

$$
\|e_{t+1}\|_2\le a\|e_t\|_2+(1-u)B_t, \quad a<1,
$$

则

$$
\|e_t\|_2
\le (1-u)\sum_{k=0}^{t-1}a^k B_{t-1-k}.
$$

若 $B_t\le \bar B$，进一步得

$$
\|e_t\|_2\le (1-u)\bar B\frac{1-a^t}{1-a}.
$$

于是

$$
|J_H^{\mathrm{v4}}-J_H^{\mathrm{base}}|
\le L_r(1-u)\bar B\sum_{t=1}^{H}\frac{1-a^t}{1-a}.
$$

**关键结论**：版本四结构偏差由 $1-u$ 控制，$u\to 1$ 时结构偏差趋小；并且受 `highway_ratio` 与 `update_clip` 进一步约束，因此可控。

---

### 8.5 版本四加 Lipschitz 的偏差可控性

版本四加 Lipschitz 在版本四基础上再引入“函数类约束偏差”：环境映射被限制在有界 Lipschitz 函数族 $\mathcal F_L$ 内。

令未约束目标最优环境映射为 $f^\star$，约束后最优近似为

$$
f_L^\star=\arg\min_{f\in\mathcal F_L}\|f-f^\star\|.
$$

则模型逼近偏差可写为

$$
\varepsilon_{\mathrm{approx}}(L)
=\|f_L^\star-f^\star\|.
$$

若策略值函数对环境映射误差是 $C_J$-Lipschitz，则有

$$
|J_H^{L}-J_H|
\le C_J\,\varepsilon_{\mathrm{approx}}(L).
$$

由于 $L$（或实现里的 `lipschitz_weight_fro_norm_max` 与 `lipschitz_gp_outputscale_max`）可调，且阈值放宽时函数类增大，通常有

$$
\varepsilon_{\mathrm{approx}}(L_2)\le \varepsilon_{\mathrm{approx}}(L_1),
\quad L_2\ge L_1.
$$

因此该偏差同样可控，只是和稳定性形成可调权衡：

- 更紧约束：稳定性更强，逼近偏差可能增大。
- 更松约束：逼近偏差减小，稳定性冗余减弱。

---

### 8.6 四种方法偏差可控性总评

1. 版本二：可控，线性受 $\lambda_2$ 控制，但表达能力弱。  
2. 版本三：可控，双权重线性控制，统计性质更好。  
3. 版本四：可控，含正则偏差与结构偏差，结构偏差由 $u$、`highway_ratio`、`update_clip` 控制。  
4. 版本四加 Lipschitz：可控，额外引入函数类约束偏差，可由 Lipschitz 上界参数连续调节。  

最终结论：**四个方法的偏差都可控制**，但“可控后的性能上限”不同。版本四加 Lipschitz 的偏差-稳定性权衡能力最强。


## 9. v5 加 Lipschitz enforce：统一概率界的长期防消失方案

本节提出一个与前面方法正交的方案。目标是：在不改变“优化 $\sum r_t$”方向的前提下，给出对所有训练步 $k$ 的统一概率界。

### 9.1 根因解耦思路

长训练中的梯度消失根因是“时间链路上的乘法衰减”。定义每个训练步 $k$ 对应窗口轨迹上的累积对数增益

$$
z_{k,t}=\log\frac{\|\Delta s_t\|_2+\epsilon}{\|\Delta s_{t-1}\|_2+\epsilon},
\qquad
S_k=\sum_{t=2}^{H} z_{k,t},
$$

其中 $H=64$（当前 TBPTT 窗口）。

则窗口级增益因子

$$
G_k=\exp(S_k).
$$

若 $S_k\ll 0$，则 $G_k$ 很小，梯度消失。

因此 v5 直接约束 $S_k$ 的分布，而不是只约束局部单步增益。

---

### 9.2 v5 的形式化目标

v5 定义为约束优化：

$$
\max_\theta\ J_H(\theta)=\mathbb E_\theta\left[\sum_{t=1}^{H} r_t\right]
$$

满足

$$
|\mu_k|\le \mu_0,\qquad \mathrm{Var}(S_k)\le v_0,\qquad \forall k,
$$

其中

$$
\mu_k=\mathbb E[S_k].
$$

注意优化方向不变：仍是提升 $\sum r_t$。实现时最小化损失

$$
\mathcal L_{\mathrm{v5}}=-J_H
+\lambda_\mu\,[|\widehat\mu_k|-\mu_0]_+
+\lambda_v\,[\widehat v_k-v_0]_+.
$$

这里 $[\cdot]_+$ 是正部函数，$\widehat\mu_k,\widehat v_k$ 是 batch 统计估计。

Lipschitz enforce 保留，用于给链路上界和有限二阶矩提供硬保护。

---

### 9.3 统一于训练步 $k$ 的概率保证（核心定理）

令 $\beta>\mu_0$。若在每个训练步 $k$ 都满足

$$
|\mu_k|\le\mu_0,\qquad \mathrm{Var}(S_k)\le v_0,
$$

则由 Cantelli 不等式（单侧切比雪夫）：

$$
\mathbb P(S_k\le -\beta)
\le
\frac{v_0}{v_0+(\beta-\mu_0)^2}
=:\delta_-,
$$

$$
\mathbb P(S_k\ge \beta)
\le
\frac{v_0}{v_0+(\beta-\mu_0)^2}
=:\delta_+.
$$

因此

$$
\mathbb P\!\left(e^{-\beta}\le G_k\le e^{\beta}\right)
\ge 1-\delta_--\delta_+,
\qquad \forall k.
$$

这条界是“对所有训练步 $k$ 的统一概率界”，不是仅在某个固定步成立。

---

### 9.4 为什么这比 v2 到 v4 更充分

1. v2 和 v3 主要约束局部或局部统计，不直接约束累计量 $S_k$。  
2. v4 增加结构通道，但若不约束 $S_k$ 的分布，长期仍可能漂移。  
3. v5 直接把“累计乘法衰减”压成“均值和方差约束”，与根因强制解耦。

---

### 9.5 偏差可控性

v5 引入的是“约束偏差”，不是方向错误。偏差来源有三项：

1. 约束本身的最优值差距（可行域收缩）。  
2. 软化近似误差（若用平滑 $[\cdot]_+$，误差是 $O(\tau)$）。  
3. batch 统计估计误差（通常是 $O(B^{-1/2})$）。

对梯度偏差，若存在常数 $C_\mu,C_v$ 使

$$
\|\nabla_\theta \widehat\mu_k\|_2\le C_\mu,\qquad
\|\nabla_\theta \widehat v_k\|_2\le C_v,
$$

则

$$
\|b_{\mathrm{v5}}(\theta)\|_2
\le
\lambda_\mu C_\mu+\lambda_v C_v+O(\tau)+O(B^{-1/2}).
$$

因此偏差可通过 $(\lambda_\mu,\lambda_v,\tau,B)$ 联合控制到小范围。

---

### 9.6 TBPTT=64 下的参数计算建议

给定你当前窗口 $H=64$，建议先设目标：

- 最小窗口增益下限：$G_k\ge e^{-\beta}$，取 $\beta=1.2$（下限约 $0.30$）。  
- 目标失效概率：$\delta_-=0.03$。

由

$$
\delta_-=\frac{v_0}{v_0+(\beta-\mu_0)^2}
$$

可得

$$
v_0\le \frac{\delta_-}{1-\delta_-}(\beta-\mu_0)^2.
$$

若取 $\mu_0=0.05,\ \beta=1.2,\ \delta_-=0.03$，则

$$
v_0\le 0.041.
$$

可用保守值 $v_0=0.03$。

---

### 9.7 CLI 设计（供实现）

建议新增开关：

- `--anti-explosion-vanishing-v5-enabled`
- `--anti-explosion-vanishing-v5-beta`
- `--anti-explosion-vanishing-v5-mu-target`
- `--anti-explosion-vanishing-v5-var-target`
- `--anti-explosion-vanishing-v5-lambda-mu`
- `--anti-explosion-vanishing-v5-lambda-var`
- `--anti-explosion-vanishing-v5-eps`
- `--anti-explosion-vanishing-v5-smooth-tau`
- `--anti-explosion-vanishing-v5-ema`

并复用已有 Lipschitz 参数：

- `--lipschitz-enforce`
- `--lipschitz-weight-fro-norm-max`
- `--lipschitz-gp-outputscale-max`

---

### 9.8 本节结论

v5 加 Lipschitz 的核心优势是：  
把“长期消失是否与训练步 $k$ 相关”的问题，转换为“对所有 $k$ 都成立的累计增益分布约束”。  
在约束可行并被持续满足时，可得到统一概率界；同时偏差保持可控并且优化方向始终是提升 $\sum r_t$。
