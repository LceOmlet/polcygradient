# Gradient Feasibility v6: TBPTT-64 下的样本级 Gate + Thermostat 设计

## 1. 目标

本文不再沿用 `gradient_feasibility_v5.md` 中“全局 Jacobian/伴随变量带宽”的证明路线，而是落到当前代码实际上能做的事情上：

1. 在 `TBPTT window = 64` 下，对**样本级**一阶 pathwise policy gradient 做可靠性检测。
2. 爆炸风险高的样本，不再等到整批 `loss/grad` 非有限后才一起丢弃，而是在样本级提前 gate 掉。
3. 对没有爆炸、但明显低信号的样本，做**有界**放大，缓解一阶策略梯度的消失。
4. 让 `policy step` 内的梯度链路和 `training step` 间的整体更新，都尽量保持在一个可控带宽内。

这里的 `v6` 选型，采用“**DDCG 风格的 hard gate + IVW-H 风格的局部加权思想**”，但要明确一点：

- 当前仓库的 policy-gradient 主链路只有 `g1`，即 pathwise / differentiable simulator 分支。
- 当前仓库里没有现成的 `g0` / score-function / 0 阶分支。
- 因此，`v6` 不是完整 DDCG，而是**最接近 DDCG 思想、且能在当前代码结构里落地的版本**。

换句话说，`v6` 的核心不是“保住所有爆炸样本再做软加权”，而是：

1. 先用样本级统计判断该样本的 `g1` 是否可信。
2. 不可信则只丢这个样本的 `g1`，而不是丢整批。
3. 可信但过弱，则做有界放大。

这比当前的“整批 `loss_nonfinite / grad_nonfinite` 后一起跳过”更细，也更符合你这里的需求。

---

## 2. 当前代码里的现实约束

### 2.1 当前 PG 主链路是 pathwise-only

当前 `policy_gradient_loss_from_rewards()` 只接收 `rewards: (T, B)`，然后把 reward tensor 归约成 loss。  
这意味着现阶段没有独立的 0 阶 `g0` 可与 `g1` 做无偏混合。

因此：

- DDCG 论文里的“`g1` 不可信时退回 `g0`”思想，在当前仓库里只能退化成“这个样本的 `g1` 不参与当前 window loss”。
- 这会引入偏差，但偏差是**样本级、局部的**；比现在整批丢弃导致的大偏差和大吞吐损失更可控。

### 2.2 当前 `v5` 不是文档版的 Jacobian/costate 方案

代码中的 `anti_explosion_vanishing_v5` 本质上是一个 detached reward-std thermostat：

- 它按 reward std 缩放整个 PG loss。
- 它不做样本级 gate。
- 它不看 per-sample / per-step 的 pathwise 可靠性。
- 它也不能区分“少数样本爆炸”与“整批都不稳定”。

因此，`v5` 更像一个**全局 loss 尺度调节器**，不是这次要解决的样本级 pathwise 稳定器。

### 2.3 当前 batch 级保护太晚

当前训练里已经有三层全局保护：

1. `loss` 非有限则 skip batch。
2. 反传后发现梯度非有限则 skip batch。
3. `clip_grad_norm_(..., 1.0)`。

这些保护是必要的，但都发生在**样本已经混进 batch reduction 之后**。  
如果一个 window 里只有 5% 的样本 pathwise 链路炸了，现在的保护方式可能会让剩下 95% 的健康样本也一起丢掉。

---

## 3. v6 的方法定义

`v6` 的思路是把现有 `AEV4` 的“highway 子空间 update 统计”扩展成一个**样本级可靠性判别器**，再把判别结果送进 `policy_gradient_loss_from_rewards()` 做 detached sample weighting。

### 3.1 统计对象

设一个 TBPTT window 的长度为

$$
H = 64,
$$

则相邻状态增量对一共有

$$
P = H - 1 = 63
$$

个。

对第 $b$ 个样本，在第 $t$ 步定义 state update

$$
\Delta_{t,b} = s_{t+1,b} - s_{t,b}.
$$

只在 state 的 `highway` 子空间上做统计，维度取

$$
d_h = \lceil \rho \cdot d_{\text{state}} \rceil,
\qquad \rho = \text{highway\_ratio}.
$$

记其 RMS 为

$$
u_{t,b} = \sqrt{\frac{1}{d_h}\|\Delta_{t,b}^{(h)}\|_2^2 + \varepsilon}.
$$

对相邻两步定义局部 update gain

$$
\gamma_{t,b} = \frac{u_{t,b}}{\operatorname{stopgrad}(u_{t-1,b}) + \varepsilon},
\qquad t=2,\dots,H.
$$

再定义

$$
\ell_{t,b} = \log(\gamma_{t,b} + \varepsilon).
$$

这里保留 `detach_reference=True`，即只让当前步 `u_{t,b}` 参与梯度，参考量 `u_{t-1,b}` 只用于检测，不反向传播。

### 3.2 每个样本的窗口统计量

对每个样本 $b$，在一个 window 内汇总：

$$
\text{high\_share}_b
=
\frac{1}{P}\sum_{t=2}^{H}\mathbf 1[\ell_{t,b} > \log g_{\max}],
$$

$$
\text{low\_share}_b
=
\frac{1}{P}\sum_{t=2}^{H}\mathbf 1[\ell_{t,b} < \log g_{\min}],
$$

$$
\text{abs\_log\_gain\_max}_b
=
\max_{t=2,\dots,H} |\ell_{t,b}|,
$$

$$
u^{\max}_b = \max_{t=2,\dots,H} u_{t,b},
\qquad
\bar u_b = \frac{1}{P}\sum_{t=2}^{H} u_{t,b}.
$$

再增加一个非有限标记：

$$
\text{nf}_b = 1
$$

当且仅当该样本在 window 内出现任意非有限量，包括但不限于：

- `state_delta`
- `u_{t,b}`
- `gamma_{t,b}`
- `log_gain`
- `reward_next`

### 3.3 样本级 hard gate

定义可靠性 gate：

$$
m_b \in \{0,1\}.
$$

建议规则：

$$
m_b = 1
$$

当且仅当以下条件同时成立：

1. `nf_b = 0`
2. `high_share_b <= max_high_share`
3. `low_share_b <= max_low_share`
4. `abs_log_gain_max_b <= max_abs_log_gain`
5. `u_max_b <= update_rms_hi`

否则

$$
m_b = 0.
$$

解释：

- `nf_b` 直接拦截 `inf / nan` 样本。
- `high_share_b` 和 `abs_log_gain_max_b` 控制爆炸型长尾。
- `low_share_b` 控制“整个窗口几乎都在衰减”的样本。
- `u_max_b` 防止“虽然均值不高，但某一步已经出现非常激烈更新”的样本漏过。

这里的关键不是按元素裁剪，而是**按样本、按 window 关掉不可信的一阶分量**。  
这比 batch 级 skip 细得多，也比“把明显有问题的 `inf` 样本继续拿来加权”安全得多。

### 3.4 低信号 thermostat

对没有被 gate 掉的样本，再定义一个 detached 的有界缩放。  
先定义一个带下界的参考量：

$$
\bar u_b^{\text{ref}} = \max(\operatorname{stopgrad}(\bar u_b), u_{\min}),
$$

其中

$$
u_{\min} = \text{update\_rms\_lo}.
$$

然后定义

$$
s_b^{\text{raw}} = \frac{u_{\text{target}}}{\bar u_b^{\text{ref}} + \varepsilon},
$$

$$
s_b = \operatorname{clip}\left(s_b^{\text{raw}}, s_{\min}, s_{\max}\right).
$$

最终样本权重为

$$
w_b = m_b \cdot s_b.
$$

这个 thermostat 做三件事：

1. 当 $\bar u_b$ 太小但还没小到不可信时，`s_b > 1`，缓解梯度消失。
2. 当 $\bar u_b$ 偏大但还没触发爆炸 gate 时，`s_b < 1`，缓解大梯度。
3. `update_rms_lo` 给 thermostat 提供一个分母下界，防止极小 update 被无限放大。
4. 由于 `s_b` detached，它不会诱导模型去“投机性地操纵权重本身”。

### 3.5 window loss 的归约方式

设一个 TBPTT window 的 reward tensor 为

$$
R \in \mathbb R^{H \times B}.
$$

对每个样本有一个 detached 权重 `w_b`。  
定义 v6 的 objective 为

$$
\hat J_{\text{v6}}
=
\frac{1}{H}
\sum_{t=1}^{H}
\frac{
\sum_{b=1}^{B} w_b \, \hat r_{t,b}
}{
\max(1, \sum_{b=1}^{B} m_b)
},
$$

其中 $\hat r_{t,b}$ 表示 reward 或标准化后的 reward。

这里分母使用 `active sample count = sum(m_b)`，而不是 `sum(w_b)`，原因是：

1. 被 gate 掉的样本不应稀释健康样本的梯度。
2. 低信号放大不应被归一化完全抵消。
3. 整体 window loss 的放大倍数仍被 `s_max` 上界控制。

若一个 window 内所有样本都被 gate：

$$
\sum_b m_b = 0,
$$

则该 window 的 loss 置零，统计里记录 `active_share = 0`。  
这比把整批训练步直接炸掉更稳。

---

## 4. TBPTT window = 64 下的参数推导

这一节是 `v6` 最关键的部分。

### 4.1 由窗口长度反推单步 gain 带宽

若希望一个 window 内 `63` 个相邻局部 gain 的乘积总体不要超过某个倍数 $M_{\text{win}}$，则单步 gain 上界应满足

$$
g_{\max}^P \le M_{\text{win}},
\qquad P = 63,
$$

即

$$
g_{\max} \le M_{\text{win}}^{1/63}.
$$

对称地

$$
g_{\min} \ge M_{\text{win}}^{-1/63}.
$$

对几个典型窗口带宽：

- 若要求窗口内总体不超过 `2x`，则单步带宽约为 `[0.9891, 1.0111]`
- 若要求窗口内总体不超过 `3x`，则单步带宽约为 `[0.9827, 1.0176]`
- 若要求窗口内总体不超过 `4x`，则单步带宽约为 `[0.9782, 1.0222]`
- 若要求窗口内总体不超过 `10x`，则单步带宽约为 `[0.9641, 1.0372]`

对当前问题，我建议把 `v6` 的默认目标设在：

$$
M_{\text{win}} \approx 4.
$$

因此推荐

- `gain_lo = 0.98`
- `gain_hi = 1.02`

这组值比我先前放进 `model_configs.py` 的占位默认值 `0.92 / 1.08` 严得多。  
`0.92 / 1.08` 对 `H=64` 来说过松，不适合作为最终生效参数。

### 4.2 `max_high_share` 与 `max_low_share`

在 `P = 63` 个 pair 里，允许多少违例才算这个样本不可信？

#### 高侧违例

高侧违例对应局部爆炸，更危险，应更严格。

建议允许最多 `4` 到 `6` 个高侧违例：

$$
\frac{4}{63} \approx 0.063,
\qquad
\frac{6}{63} \approx 0.095.
$$

因此建议初值：

- `max_high_share = 0.08`

#### 低侧违例

低侧违例对应局部衰减。它会伤可学习性，但数值危险性低于高侧，因此可以稍微宽一点。

建议允许最多 `12` 到 `16` 个低侧违例：

$$
\frac{12}{63} \approx 0.19,
\qquad
\frac{16}{63} \approx 0.25.
$$

因此建议初值：

- `max_low_share = 0.25`

这个不宜像之前占位值那样放到 `0.60`，否则“几乎整个窗口都在衰减”的样本也会被误判成可靠。

### 4.3 `max_abs_log_gain`

这是“单个局部跳变异常大”的红线。

如果 nominal 单步带宽取 `gain_hi = 1.02`，则

$$
\log(1.02) \approx 0.0198.
$$

如果允许单点偶发偏离 nominal 带宽约 `3x`，则

$$
3 \times 0.0198 \approx 0.0594.
$$

因此建议：

- `max_abs_log_gain = 0.07`

更松可以到 `0.10`，但不建议再大。  
占位值 `0.35` 太松，对 `TBPTT=64` 几乎失去约束意义。

### 4.4 `update_rms_lo / target_update_rms / update_rms_hi`

这三个参数不直接控制乘法链，而是控制“单步更新量落在什么幅度区间里最利于学习”。

建议初值：

- `update_rms_lo = 5e-3`
- `target_update_rms = 5e-2`
- `update_rms_hi = 1.2e-1`

理由：

1. `5e-3` 足够远离纯噪声级别，作为 thermostat 分母下界，避免对极小抖动过度放大。
2. `5e-2` 与当前 `AEV4 update_scale = 0.08` 同量级，但略保守。
3. `1.2e-1` 大约是目标值的 `2.4x`，再往上就应更倾向认为该样本进入了高风险区。

如果后续观测到：

- `active_share` 很低，且 `u_max` 常常只略高于 `0.12`  
  那么可以把 `update_rms_hi` 先放宽到 `0.15`。

- `scale_hi` 常常打满，且 `grad_zero_share` 仍高  
  那么可以把 `target_update_rms` 提到 `0.06` 或把 `scale_hi` 提到 `2.5~3.0`。

### 4.5 `scale_lo / scale_hi`

建议初值：

- `scale_lo = 0.5`
- `scale_hi = 2.0`

原因：

1. `scale_lo = 0.5` 可以对“大但未爆”的样本温和降权。
2. `scale_hi = 2.0` 已足够抵消很多低信号窗口，不会过度放大噪声。
3. 在没有 `g0` 分支时，过大的 `scale_hi` 会更快放大 pathwise bias 和 sample noise。

若首轮实验发现一阶梯度仍明显偏零，可把 `scale_hi` 再放到 `3.0`，但不建议直接更大。

### 4.6 `highway_ratio`

建议保持：

- `highway_ratio = 0.25`

原因：

1. 与当前 `AEV4` 一致，便于复用已有逻辑。
2. 取太小会让统计过于稀疏；取太大又会把大量不相关通道混进来。
3. 先在最像“主动力学通道”的前 `25%` 维上做检测，工程上最稳。

---

## 5. 推荐的首版参数

以 `TBPTT window = 64` 为前提，建议 `v6` 首轮就按下面的值写入文档和实验配置，而不是沿用当前占位默认值：

```python
"anti_explosion_vanishing_v6_enabled": True,
"anti_explosion_vanishing_v6_gain_lo": 0.98,
"anti_explosion_vanishing_v6_gain_hi": 1.02,
"anti_explosion_vanishing_v6_max_low_share": 0.25,
"anti_explosion_vanishing_v6_max_high_share": 0.08,
"anti_explosion_vanishing_v6_max_abs_log_gain": 0.07,
"anti_explosion_vanishing_v6_update_rms_lo": 5e-3,
"anti_explosion_vanishing_v6_update_rms_hi": 1.2e-1,
"anti_explosion_vanishing_v6_target_update_rms": 5e-2,
"anti_explosion_vanishing_v6_scale_lo": 0.5,
"anti_explosion_vanishing_v6_scale_hi": 2.0,
"anti_explosion_vanishing_v6_eps": 1e-6,
"anti_explosion_vanishing_v6_detach_reference": True,
"anti_explosion_vanishing_v6_highway_ratio": 0.25,
```

同时建议把旧稳定化方法和 Jacobian 上界护栏都从默认主线移除：

```python
"anti_explosion_vanishing_v2_enabled": False,
"anti_explosion_vanishing_v3_enabled": False,
"anti_explosion_vanishing_v4_enabled": False,
"anti_explosion_vanishing_v5_enabled": False,
"lipschitz_enforce": False,
```

原因：

1. `v2-v5` 都会额外引入 detached regularization 或缩放，和 `v6` 叠加后很难归因。
2. 默认主线应先让 `v6` 单独承担梯度稳定化职责，才能看清它是否真的提高了大斜率环境的可学习性。
3. `lipschitz_enforce` 会改变环境生成分布；作为默认主线护栏会掩盖 `v6` 本身的效果。

---

## 6. 为什么它能保证 policy step 的梯度不消失也不爆炸

这里的 `policy step` 指 window 内的时间步 $t$。

### 6.1 防止爆炸

`policy step` 的一阶 pathwise 梯度，危险来自局部链路乘积：

$$
\prod_{t=2}^{H} \gamma_{t,b}.
$$

`v6` 通过三层机制压住它：

1. `gain_lo / gain_hi` 约束大部分局部增益靠近 `1`
2. `max_high_share` 限制高侧长尾出现的频率
3. `max_abs_log_gain` 和 `update_rms_hi` 拦截单步异常大的跳变

由于这些判断是**样本级**的，所以一个样本炸掉，不会把同一 batch 中其他健康样本一起带走。

若再叠加 `AEV4` 的 `update_scale`，则 forward dynamics 本身也被压在较温和的 update 带宽中。  
这等于把保护分成两层：

1. forward 层先少制造极端样本
2. loss 层再把漏网的极端样本 gate 掉

### 6.2 防止消失

单靠 gate 不够，因为很多样本并不爆炸，只是 $\bar u_b$ 太小，导致一阶梯度接近零。

`v6` 用 thermostat 处理这一点：

$$ 
s_b = \operatorname{clip}\left(
\frac{u_{\text{target}}}{\max(\operatorname{stopgrad}(\bar u_b), u_{\min})+\varepsilon},
s_{\min},
s_{\max}
\right).
$$

当 $\bar u_b$ 偏小但样本仍可靠时，`s_b > 1`，它会被放大；  
当 $\bar u_b$ 偏大但尚未爆炸时，`s_b < 1`，它会被压低。

因此对每个健康样本，window 内的一阶 PG 贡献被压到一个近似带宽里：

$$
0.5 \lesssim s_b \lesssim 2.0.
$$

它不是严格数学无偏的双边界证明，但在当前实现能力下，这是最直接、最可控的工程近似。

---

## 7. 为什么它能保证 training step 的梯度不消失也不爆炸

这里的 `training step` 指 batch 间的优化步 $k$。

### 7.1 防止 training-step 爆炸

每个 window 的 loss 被拆成：

1. 样本级 gate 后的 active 子集
2. active 子集上的有界权重
3. 固定的 window weight

因此对任一 window，其损失尺度被控制在：

- 样本 gate：坏样本不参与
- thermostat：每个活跃样本的缩放只在 `[scale_lo, scale_hi]`
- TBPTT 汇总：window weight 仍按时间长度和 batch 占比累计

这样做后，进入参数梯度的每个 window 项都带有显式上界。  
再叠加训练侧已有的：

- `loss_nonfinite` 检测
- `grad_nonfinite` 检测
- `clip_grad_norm_(..., 1.0)`

就把 training-step 梯度爆炸的风险压成了：

1. 样本级前置过滤
2. window 级 bounded scaling
3. batch 级最终保险

### 7.2 防止 training-step 消失

training-step 梯度消失的典型原因，是一个 window 里虽然多数样本“数学上有梯度”，但有效 signal 太弱，聚合后接近零。

`v6` 通过两个点缓解：

1. 低信号样本会被 bounded boost，而不是继续按极小权重混进均值里。
2. 分母用 `active sample count`，不会因为大量被 gate 的坏样本仍留在 batch mean 里而把健康样本进一步稀释。

这意味着只要一个 window 内还有足够多的健康样本，training-step 的有效梯度就不会被无意义地冲淡。

---

## 8. 与完整 DDCG 的关系与局限

`v6` 与 DDCG 的相同点：

1. 都不相信“所有 `g1` 都应该直接参与加权”。
2. 都先判断 `g1` 是否可靠。
3. 一旦不可靠，就不应把它继续乘上一个浮点权重硬塞进 loss。

`v6` 与 DDCG 的不同点：

1. DDCG 在 `g1` 不可靠时可以退回 `g0`。
2. 当前仓库里没有 `g0`，所以 `v6` 只能对不可靠样本把 `g1` 置零。
3. 因而 `v6` 仍然有偏，只是这个偏差比“整批都丢”小得多。

因此要诚实地说：

- `v6` 不是完整 DDCG。
- `v6` 是当前代码条件下的最合理过渡方案。
- 如果后续发现 `active_share` 长期过低，那么下一步就不应该继续调 gate，而应该补 0 阶分支。

---

## 9. 需要记录的关键统计

为了判断 `v6` 是否真的有效，建议至少记录以下窗口级或 batch 级统计：

- `aev6_active_share`
- `aev6_exploded_share`
- `aev6_vanishing_share`
- `aev6_nonfinite_share`
- `aev6_weight_mean`
- `aev6_weight_std`
- `aev6_scale_mean`
- `aev6_scale_max`
- `aev6_gain_high_share_mean`
- `aev6_gain_low_share_mean`
- `aev6_abs_log_gain_max`
- `aev6_update_rms_mean`
- `aev6_update_rms_max`

建议用下面三条作为首轮判断标准：

1. `active_share` 不能长期低于 `0.6`
2. `scale_hi` 不应长期打满
3. 开启 `v6` 后，`grad_nonfinite_share` 和 batch skip 次数应显著下降

如果第 1 条长期失败，说明不是 gate 太紧，就是 pathwise-only 分支本身不够，需要补 `g0`。

---

## 10. 实现落点

按当前仓库结构，`v6` 的落点应当是：

1. 在 `rollout_with_policy()` 的三个 rollout 路径里，像 `AEV4` 一样对 `state_delta` 做窗口累计，但这次统计要保留**每个样本**的信息。
2. 在 TBPTT sink payload 里增加 `aev6` window summary。
3. 在 `policy_gradient_loss_from_rewards()` 中接收 `aev6_summary`，构造 detached sample weights。
4. 用 `active sample count` 做分母，而不是全 batch mean。
5. 若 `active_count == 0`，则当前 window loss 置零并记日志。

注意：

- `v6` 首版应只落在文档和实现，不要和 `v5` 同时开。
- `v6` 首版应优先支持 `TBPTT window = 64`。
- `v6` 首版应先做样本级 gate，再看是否需要更细的 step 级 gate。

---

## 11. 结论

对当前仓库，最合理的 `gradient_feasibility_v6` 不是继续追求 `v5` 那种全局 Jacobian 可行域证明，而是：

1. 用 `AEV4` 风格的局部 update 统计，得到每个样本在 `TBPTT=64` 窗口内的可靠性摘要。
2. 用 DDCG 风格 hard gate，把不可信的 pathwise 一阶样本单独关掉。
3. 用 detached thermostat，把可靠但低信号的样本有界放大。
4. 用 active-sample 分母，防止坏样本拖累健康样本。

这样做不能消除“没有 `g0` 分支”的结构性偏差，但它能显著减少：

1. 因少数爆炸样本导致的整批丢弃
2. 因整批均值稀释导致的一阶梯度消失
3. 因 `TBPTT=64` 乘法链过长造成的局部长尾放大

从工程上看，这就是当前代码里最贴切、最值得先实现的 `v6`。
