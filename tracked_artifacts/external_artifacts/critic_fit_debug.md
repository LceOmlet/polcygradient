# Critic Fit Debug Note

## 1. Question

The critic debugging question is narrower than the actor-boundary bug:

- does the current learned critic actually help policy learning more than no critic?
- if not, is the critic intrinsically bad, or just underfit under the current joint PPO training budget?

This note records the current strongest evidence.

## 2. Current actor-side usage

On the current default path, actor training still uses a learned critic baseline:

- `ppo_actor_gae_space = "normalized"`
- `ppo_actor_baseline_mode = "learned"`

Relevant code:

- [model_configs.py](/home/chen/RLPFN/reinforce-terminal-explore/ticl/model_configs.py)
- [sb3_recurrent_ppo.py](/home/chen/RLPFN/reinforce-terminal-explore/ticl/sb3_recurrent_ppo.py)

So the question is not hypothetical. The current PPO actor does consume learned critic signal.

## 3. Fixed legacy environment policy-help check

Probe:

- fixed legacy environment: `frozen_h_seed = 12345`
- state-reset fix enabled: `ppo_reset_env_state_at_sep = true`
- same checkpoint
- same train seed
- compare `learned` vs `zero` actor baseline modes

Artifact:

- [critic_help_vs_zero_state_reset_seed12345_quick.json](/home/chen/RLPFN/artifacts/critic_help_vs_zero_state_reset_seed12345_quick.json)

Result:

- learned:
  - `delta_smp_gap = +1.488346815109253`
- zero:
  - `delta_smp_gap = +3.268249988555908`

Interpretation:

- on this fixed-environment quick audit, the current learned critic does **not** help policy more than no critic
- under the current joint training budget, it still underperforms `zero` baseline on this probe

This is the strongest direct evidence so far that current critic quality is not yet good enough to be reliably useful to the actor.

## 4. Critic-only fixed-buffer fit probe

Probe:

- same checkpoint
- same fixed legacy environment
- same train seed
- state-reset fix enabled
- collect one fixed rollout buffer
- run critic-only optimization on that exact buffer
- measure raw return correlation and explained variance at increasing fit steps

Artifact:

- [critic_value_fit_probe_state_reset_seed12345.json](/home/chen/RLPFN/artifacts/critic_value_fit_probe_state_reset_seed12345.json)

Key results:

- step `0`:
  - `raw_corr = -0.0311`
  - `explained_variance_raw = -0.000108`
  - `raw_value_std = 0.0386`
  - `raw_return_std = 22.7914`
- step `200`:
  - `raw_corr = 0.4675`
  - `explained_variance_raw = 0.2184`
  - `raw_value_std = 10.3707`
- step `1000`:
  - `raw_corr = 0.8006`
  - `explained_variance_raw = 0.6404`
  - `raw_value_std = 17.6759`

Interpretation:

- the critic is **not** fundamentally unable to fit the target on this environment
- with enough dedicated value-only optimization on the same buffer, it learns useful ranking and prediction quality
- therefore, the current poor critic quality is not best explained as “impossible target semantics”

## 5. Current best diagnosis

The two probes together imply:

- in current joint PPO training, the critic reaches the actor too early while still underfit
- but the critic itself is trainable if it gets enough optimization budget

So the most credible current diagnosis is:

- **critic usefulness failure is primarily a training-budget / optimization-allocation problem, not a proof that the critic target is intrinsically wrong**

More concretely:

- current PPO training does not fit the critic enough before the actor starts relying on it
- the result is a learned baseline that is worse than `zero` baseline on the fixed legacy audit

## 6. What is ruled out

This evidence weakens the following explanations:

- “critic is useless because the target is inherently unlearnable”
- “critic remains bad only because bar-distribution support is obviously wrong”
- “critic remains bad only because one or two extra epochs were missing in a trivial sense”

The more precise reading is:

- the critic can fit, but not under the current joint-budget regime

## 7. Next step

## 7. Extra critic-updates audit

Probe:

- keep repaired SEP state semantics fixed
- keep actor objective fixed
- compare:
  - current PPO
  - PPO + extra critic-only updates after each rollout
- fixed legacy environment: `frozen_h_seed = 12345`

Artifacts:

- baseline:
  - [critic_help_vs_zero_state_reset_seed12345_quick.json](/home/chen/RLPFN/artifacts/critic_help_vs_zero_state_reset_seed12345_quick.json)
- extra critic updates:
  - [critic_help_vs_zero_state_reset_plus_extra_critic_seed12345_quick.json](/home/chen/RLPFN/artifacts/critic_help_vs_zero_state_reset_plus_extra_critic_seed12345_quick.json)
- head-only extra critic updates:
  - [critic_help_vs_zero_state_reset_plus_head_only_extra_critic_seed12345_quick.json](/home/chen/RLPFN/artifacts/critic_help_vs_zero_state_reset_plus_head_only_extra_critic_seed12345_quick.json)

Results:

- current PPO:
  - learned `delta_smp_gap = +1.4883`
  - zero `delta_smp_gap = +3.2682`
- current PPO + `extra_critic_updates_per_outer_epoch = 200`:
  - learned `delta_smp_gap = -1.9939`
  - zero `delta_smp_gap = +1.1190`
- current PPO + `extra_critic_updates_per_outer_epoch = 200`, `head_only`:
  - learned `delta_smp_gap = -2.8659`
  - zero `delta_smp_gap = -2.6440`

Interpretation:

- naive extra critic updates on the current shared representation do **not** make learned critic superior to zero baseline
- they make learned-baseline policy learning worse on this probe
- they also hurt zero baseline somewhat, which suggests the extra critic updates are perturbing the shared representation, not just improving value quality
- restricting those extra critic updates to the value head alone also does **not** recover usefulness
- in this probe, head-only extra critic updates are even worse than the shared-update version for learned baseline

So the current reading is:

- critic fit is trainable in isolation
- but simply increasing critic updates in the current policy stack is not a safe fix
- the next issue is no longer “critic gets too few steps” alone
- it is more likely that the current critic path itself is not a good intervention surface for policy learning
- shared entanglement may still be part of the problem, but head-only restriction did not rescue it

## 8. Next step

Do not reopen broad actor-objective changes here.

The next highest-value critic probe is:

- keep the repaired SEP state semantics fixed
- keep the actor objective fixed
- isolate critic fit from actor representation drift

The cleanest next version is:

- compare current PPO
- vs critic-improvement paths that do **not** rewrite the shared actor representation as aggressively

Most likely next probes:

- a separate critic tower / partially decoupled critic path
- or a stricter actor/critic parameter split than the current value-head-only patch path

Current decision:

- **do not keep iterating extra-critic-update variants on the current architecture**
- **if critic work continues, move to critic-path decoupling rather than update-scope tweaking**

## 9. Ranking-gated critic warmup

Probe:

- keep the repaired SEP state semantics fixed
- keep the shared network structure fixed
- do not add extra critic updates
- delay actor access to the learned baseline until critic ranking crosses a reasonable gate

Artifact:

- [critic_warmup_ranking_gate_probe_state_reset_seed12345.json](/home/chen/RLPFN/artifacts/critic_warmup_ranking_gate_probe_state_reset_seed12345.json)

Setup:

- fixed legacy environment `frozen_h_seed = 12345`
- `single_eval_pos = 64`
- `n_steps = 256`
- `outer_epochs = 4`
- warmup rule:
  - actor uses `zero` baseline until:
    - at least `2` outer epochs have elapsed
    - and rollout critic `raw_corr >= 0.2`

Results:

- outer epoch 1:
  - baseline mode: `zero`
  - critic `raw_corr = 0.0044`
  - critic `explained_variance_raw = -0.4174`
- outer epoch 2:
  - baseline mode: `zero`
  - critic `raw_corr = 0.0361`
  - critic `explained_variance_raw = -0.4396`
- outer epoch 3:
  - baseline mode: `zero`
  - critic `raw_corr = -0.0431`
  - critic `explained_variance_raw = -0.5024`
- outer epoch 4:
  - baseline mode: `zero`
  - critic `raw_corr = -0.2502`
  - critic `explained_variance_raw = -0.9920`

Interpretation:

- under a reasonable ranking gate, the critic never becomes ready enough to hand actor control back to `learned`
- this is stronger than “warmup did not help”; it means **the handoff condition never becomes true**
- so a ranking-gated delayed-critic-usage scheme does not currently salvage the shared structure

Current reading:

- critic is still trainable in isolation
- but under the current joint PPO dynamics, critic ranking on the live rollout remains too poor for a meaningful learned-baseline handoff
- that pushes the remaining fixed-structure explanation away from “critic just needs a few protected warmup epochs”

Decision:

- **do not pursue ranking-gated critic warmup as the main fix on the current shared structure**
- if fixed-structure critic debugging continues, the only narrow remaining test is a forced delayed handoff after a fixed number of outer epochs
- otherwise, the evidence now points back toward critic-path decoupling

## 10. Separate value backbone smoke

Implemented:

- optional PPO flag:
  - `optimizer.ppo_separate_value_backbone = True`
- actor keeps the original RWKV backbone
- value path uses a second RWKV backbone with the same shape and copied initialization
- value loss still trains in normalized value space

Important cost:

- this **does increase memory and runtime**
- backbone parameters are duplicated
- optimizer state is duplicated
- audit/runtime latency is materially higher

Audit-path fixes needed before running the smoke:

- the single-environment audit script had to pass `ppo_separate_value_backbone` through to PPO construction
- serial audit buffer clearing had to call `policy.reset_rollout_cache(...)`
- vectorized rollout step had to use an explicit `{actor, value}` cache when separate value backbone is enabled; otherwise actor/value cache state was mixed and eval failed

Artifacts:

- separate-backbone smoke:
  - `/home/chen/RLPFN/artifacts/critic_help_vs_zero_state_reset_separate_value_backbone_seed12345_smoke.json`
- matched shared-backbone smoke:
  - `/home/chen/RLPFN/artifacts/critic_help_vs_zero_state_reset_shared_seed12345_smoke.json`

Setup:

- fixed legacy environment `frozen_h_seed = 12345`
- repaired `SEP state reset` semantics on
- `eval_env_count = 2`
- `n_steps = 256`
- `outer_epochs = 1`
- `learned` vs `zero`

Results:

- shared backbone:
  - learned `delta_smp_gap = -1.6352`
  - zero `delta_smp_gap = -0.6843`
  - learned is worse than zero

- separate value backbone:
  - learned `delta_smp_gap = -0.2497`
  - zero `delta_smp_gap = -2.3034`
  - learned is better than zero

Reading:

- this is only a smoke, not a final promotion test
- but it is the first probe in which a critic-path decoupling changes the sign of the learned-vs-zero comparison in the expected direction
- under the same repaired `SEP` semantics and matched smoke settings:
  - shared path: learned hurts more than zero
  - separate value path: learned hurts less than zero and becomes the better option

Decision:

- if critic work continues, **critic-path decoupling is now the leading direction**
- “shared structure + timing/scope tricks” should no longer be the main line of investigation

## 11. Can critic separation + SEP state reset pull gains positive?

Current answer:

- **no**

Completed artifacts:

- separate smoke:
  - `/home/chen/RLPFN/artifacts/critic_help_vs_zero_state_reset_separate_value_backbone_seed12345_smoke.json`
- separate full quick:
  - `/home/chen/RLPFN/artifacts/critic_help_vs_zero_state_reset_separate_value_backbone_seed12345_quick.json`

Results:

- smoke (`eval_env_count = 2`, `outer_epochs = 1`):
  - learned `delta_smp_gap = -0.2497`
  - zero `delta_smp_gap = -2.3034`
  - local reading:
    - separation looked directionally promising

- full quick (`eval_env_count = 8`, `outer_epochs = 2`):
  - learned `delta_smp_gap = -1.0881`
  - zero `delta_smp_gap = +0.5929`

Reading:

- the larger matched quick audit overrules the smoke-level optimism
- under `SEP state reset` plus separate value backbone:
  - learned critic is still negative in absolute terms
  - and is still worse than `zero` baseline on the completed quick probe
- so critic separation alone is **not enough** to pull the audit positive

## 12. Why does separate critic still pull gains negative?

Question:

- is the remaining negative result under `SEP state reset + separate value backbone`
  mainly due to critic underfitting?

Probe:

- fixed-buffer critic-only fitting under the same separate-backbone setting

Artifact:

- `/home/chen/RLPFN/artifacts/critic_value_fit_probe_state_reset_separate_value_backbone_seed12345.json`

Results:

- live/joint starting point is still near-random:
  - step `0`:
    - `raw_corr = -0.0583`
    - `explained_variance_raw = -0.0010`
    - `raw_value_std = 0.0133`
- after only `50` critic-only updates on the fixed buffer:
  - `raw_corr = 0.0605`
  - `explained_variance_raw = 0.0037`
  - still effectively useless
- after `200` critic-only updates:
  - `raw_corr = 0.7450`
  - `explained_variance_raw = 0.5545`
  - value scale becomes nontrivial: `raw_value_std = 1.2555`

Reading:

- yes, **critic underfitting is a major part of the remaining problem**
- at the moment actor sees the learned baseline in joint PPO, the separate critic is still essentially random
- the same critic path becomes useful once it receives enough dedicated fitting on the fixed buffer
- so the current negative learned result is consistent with:
  - critic-path decoupling helps
  - but critic is still being consumed by actor too early

Most defensible conclusion:

- separate critic has addressed the “shared representation is harmful” issue only partially
- the remaining failure mode is now more plausibly **critic warm-start / fit readiness**
  than raw architectural entanglement

## 13. Delayed critic handoff under separate critic

Question:

- if the remaining issue is critic underfit, can a readiness-gated delayed handoff rescue policy learning
  under:
  - `SEP state reset`
  - separate value backbone

Artifacts:

- warmup probe:
  - `/home/chen/RLPFN/artifacts/critic_warmup_ranking_gate_probe_state_reset_separate_value_backbone_seed12345_smoke.json`
- matched separate-baseline smoke:
  - `/home/chen/RLPFN/artifacts/critic_help_vs_zero_state_reset_separate_value_backbone_seed12345_outer4_smoke.json`

Setup:

- fixed legacy environment `12345`
- `eval_env_count = 2`
- `n_steps = 256`
- `outer_epochs = 4`
- warmup gate:
  - actor uses `zero` until:
    - at least `2` outer epochs elapsed
    - and rollout critic `raw_corr >= 0.2`

Warmup results:

- the gate never fired
- rollout critic quality stayed poor throughout:
  - epoch 1: `raw_corr = 0.0495`, `EV = -0.0953`
  - epoch 2: `raw_corr = 0.0675`, `EV = -0.1605`
  - epoch 3: `raw_corr = 0.0142`, `EV = -0.0406`
  - epoch 4: `raw_corr = -0.0891`, `EV = -0.1369`
- reported performance:
  - warmup `delta_smp_gap = +2.5499`

Matched separate-baseline results:

- learned:
  - `delta_smp_gap = -9.0706`
- zero:
  - `delta_smp_gap = +7.5272`

Reading:

- the readiness gate never becomes true, so learned critic is still not fit enough to hand actor back to `learned`
- under this setting, positive policy improvement comes from staying near `zero` baseline, not from a successful critic handoff
- therefore delayed handoff does **not** rescue learned critic yet; it only confirms the critic is still too underfit at the point of intended use

Decision:

- if critic work continues on the separate path, the next logical direction is no longer “gate the handoff”
- it is:
  - **prefit the separate critic path more aggressively before actor consumes it**
  - or provide a dedicated critic-only warm-start stage under the decoupled path

## 14. Separate critic prefit / warm-start

Question:

- can a stronger critic prefit on the separate critic path make the delayed handoff viable?

Probe:

- keep:
  - `SEP state reset`
  - separate value backbone
- actor starts on `zero`
- add:
  - `extra_critic_updates_per_outer_epoch = 200`
  - `extra_critic_update_scope = critic_path`
- still require a rollout-quality gate before handoff:
  - `min_outer_epochs = 2`
  - `raw_corr >= 0.2`

Artifact:

- `/home/chen/RLPFN/artifacts/critic_prefit_warmup_critic_path_state_reset_separate_value_backbone_seed12345_smoke.json`

Results:

- the gate still never fired
- rollout critic quality remained poor:
  - epoch 1: `raw_corr = -0.0901`, `EV = -12.5968`
  - epoch 2: `raw_corr = +0.0220`, `EV = -0.0000`
  - epoch 3: `raw_corr = -0.0890`, `EV = -0.0001`
  - epoch 4: `raw_corr = +0.0529`, `EV = -0.0508`
- actor baseline mode stayed `zero` throughout
- performance:
  - `delta_smp_gap = +0.3862`

Reading:

- even with heavy critic-path-only prefit after each outer epoch, the critic still does not reach usable live-rollout quality
- this is weaker than the earlier `zero`-only warmup result:
  - earlier warmup without critic-path prefit had `delta_smp_gap = +2.5499`
- so this prefit variant does **not** justify a long-run promotion test

Decision:

- do **not** move this variant to long-run training yet
- the short fixed-legacy probe is already sufficient to reject it as a reliable handoff mechanism

## 15. Rollout-to-rollout transfer probe

Question:

- why does fixed-buffer critic-only fitting succeed, while live-rollout warmup and handoff still fail?
- is the main issue still plain underfit, or does the critic target itself drift too much from rollout to rollout?

Probe:

- keep:
  - `SEP state reset`
  - separate value backbone
- keep actor fixed:
  - critic-only fitting updates the separate critic path only
  - actor parameter delta is checked explicitly
- collect one anchor rollout on the fixed legacy environment `12345`
- fit the critic on that exact rollout for `200` critic-only steps
- then, without changing actor parameters, collect fresh rollouts from the same fixed environment and compare:
  - critic quality on the anchor rollout
  - critic quality on fresh rollouts
  - target correlation between the anchor rollout and each fresh rollout

Artifacts:

- transfer probe:
  - `/home/chen/RLPFN/artifacts/critic_rollout_transfer_probe_state_reset_separate_value_backbone_seed12345.json`
- current-code fixed-buffer rerun:
  - `/home/chen/RLPFN/artifacts/critic_value_fit_probe_state_reset_separate_value_backbone_seed12345_rerun.json`

Current-code fixed-buffer rerun:

- step `0`:
  - `raw_corr = 0.0631`
  - `EV = 0.0005`
- step `200`:
  - `raw_corr = 0.5667`
  - `EV = 0.3133`

So under the current code, the critic still **can** fit a fixed rollout buffer meaningfully.

Transfer-probe results:

- anchor rollout, before fit:
  - `raw_corr = 0.0406`
  - `EV = 0.0008`
- anchor rollout, after `200` critic-only steps:
  - `raw_corr = 0.4186`
  - `EV = 0.1710`
- parameter deltas:
  - actor params `L2 delta = 0.0`
  - value params `L2 delta = 8.3352`

Fresh-rollout target comparisons vs the fitted anchor rollout:

- rollout 1:
  - anchor-vs-fresh raw target corr `= -0.1721`
  - fresh critic raw corr `= 0.1578`
  - raw return mean shifts from `-68.20` to `+21.24`
- rollout 2:
  - anchor-vs-fresh raw target corr `= +0.0696`
  - fresh critic raw corr `= 0.6193`
  - raw return mean shifts to `+96.70`
- rollout 3:
  - anchor-vs-fresh raw target corr `= -0.2195`
  - fresh critic raw corr `= -0.2048`
  - raw return mean shifts to `-276.07`

Important detail:

- normalized-target correlation and raw-target correlation are similarly poor
- so the dominant problem is **not** just raw-space de-normalization
- the sampled return target itself changes heavily across fresh rollouts

Reading:

- the actor stayed completely fixed, so this is not “critic fitting changed the policy and caused a new task”
- instead, the remaining large bottleneck is:
  - **single-rollout Monte Carlo target instability under the current stochastic policy**
  - on a fixed environment, one rollout is not a stable enough target for the critic to become reliably useful on the next rollout

This resolves the apparent contradiction:

- fixed-buffer probe asks:
  - “can the critic fit one sampled trajectory?”
  - answer: yes
- live warmup asks:
  - “is the critic already useful on the next fresh rollout when actor wants to consume it?”
  - answer: often no

Current diagnosis update:

- the main remaining critic bottleneck is no longer best described as “insufficient optimization steps” alone
- the bigger head is:
  - **rollout-to-rollout target instability / single-trajectory target noise**
- underfit still exists, but it is downstream of this instability

Implication:

- avoid jumping to more value-path architectural changes before testing whether the critic target can be stabilized or averaged across more on-policy data
- if critic work continues, the next probes should focus on:
  - larger on-policy rollout support per critic update
  - target averaging / multi-rollout critic readiness
  - or measuring critic usefulness under reduced policy-sampling noise

## 16. Why single-environment value fitting is still bad

Question:

- “value should predict return from history state; why is fitting still bad even in the single-environment test?”

This was checked by reducing superficial noise sources first, instead of guessing.

### 16.1 Noise sources ruled out

For the fixed legacy environment built from `frozen_h_seed = 12345`, the frozen `h` currently has:

- `reward_dropout_enabled = False`
- `reward_dropout_randomize = False`
- `reward_dropout_ratio = 0.0`
- `action_noise_train_std = 0.0`
- `action_noise_eval_std = 0.0`
- `terminal_reset_enabled = True`

So the remaining instability is **not** explained by reward-dropout randomization or explicit action-noise injection.

### 16.2 Deterministic-actor transfer probe

Artifact:

- `/home/chen/RLPFN/artifacts/critic_rollout_transfer_probe_state_reset_separate_value_backbone_seed12345_deterministic.json`

Setup:

- keep:
  - `SEP state reset`
  - separate value backbone
- force actor sampling to use action mean only
- fit critic on one anchor rollout for `200` critic-only steps
- actor params stay frozen (`actor_param_delta_l2 = 0.0`)

Results:

- anchor rollout after fit:
  - `raw_corr = 0.1797`
- fresh deterministic rollouts still drift strongly:
  - rollout 1 anchor-vs-fresh raw target corr `= 0.0020`
  - rollout 2 anchor-vs-fresh raw target corr `= -0.2141`
  - rollout 3 anchor-vs-fresh raw target corr `= -0.0617`

Reading:

- even after removing actor sampling noise, the single-environment target still drifts heavily from rollout to rollout
- so the main issue is **not only** “sampled policy makes Monte Carlo returns noisy”

### 16.3 PPO rollout path does not lock rollout RNG the way the fixed-env audit suggests

Code evidence:

- PPO training rollout uses:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/sb3_recurrent_ppo.py`
  - `collect_rollouts()`
- that path directly calls:
  - `env_prior._rollout_family_group_vectorized_with_policy(...)`
- but it does **not** pass:
  - `env_rng_seeds`
  - `rollout_rng_seeds`

This means the PPO “single-environment” training path is not actually using the same fixed-seed contract as the external VecEnv wrapper.

### 16.4 Seed-stability probe

Artifact:

- `/home/chen/RLPFN/artifacts/critic_rollout_seed_stability_probe_state_reset_separate_value_backbone_seed12345.json`

Setup:

- keep:
  - `SEP state reset`
  - separate value backbone
- force deterministic actor sampling
- compare repeated rollouts under:
  - current `collect_rollouts()` path
  - a monkey-patched version that forcibly injects fixed `env_rng_seeds` and `rollout_rng_seeds`

Results:

- current `collect_rollouts()` repeated-rollout target corr vs anchor:
  - rollout 1: `0.0862`
  - rollout 2: `0.1939`
- forced-seed repeated-rollout target corr vs anchor:
  - rollout 1: `0.5799`
  - rollout 2: `0.1870`

Reading:

- forcing seeds improves stability at least partially, so missing seed control is a real contributor
- but it does **not** fully stabilize the rollout target
- therefore there is still additional unresolved stochasticity inside the family-vectorized PPO rollout path

### 16.5 Current diagnosis

The strongest current explanation for poor single-environment value fitting is:

- the PPO “single-environment” critic is **not** training on a truly fixed target
- even with:
  - frozen `h`
  - deterministic actor sampling
  - no reward dropout
  - no explicit action noise
- the rollout-return target still changes strongly across outer epochs

So the bottleneck is better described as:

- **training-path rollout target instability**

not merely:

- “critic lacks enough gradient steps”
- or “value head cannot read history state”

### 16.6 Exact rollout-drift source that was still unfrozen

That remaining source is now pinned down.

Artifact:

- `/home/chen/RLPFN/artifacts/critic_rollout_forced_seed_contract_probe_after_freeze_fix.json`

Code source:

- `_freeze_env_h_list_for_replay()` in:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/train.py`
- latent consumers in:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/priors/environment_prior.py`

Root cause:

- the replay/fixed-env helper froze reward-dropout settings but did **not** freeze env-construction latent uniforms that are lazily sampled from `h`:
  - `_constrained_obs_u`
  - `_constrained_noise_u`
  - `_ctrl_reward_enable_u`
  - `_survival_reward_enable_u`
- those latents are consumed by:
  - `_sample_dims()`
  - `_sample_exact_scm_reward_term_enabled()`
- when absent, `_latent_uniform_from_h()` mutates `h` with fresh `np.random.random()` draws
- therefore repeated “fixed-env” rollouts could still change:
  - `obs_dim`
  - `noise_dim`
  - `zero_pad_dim`
  - exact reward-term enable flags

Fix:

- `_freeze_env_h_list_for_replay()` now materializes and freezes those latent keys before replay

Regression:

- `/home/chen/RLPFN/reinforce-terminal-explore/ticl/tests/test_train_policy_rollout_checkpoint.py`
  - `test_freeze_env_h_list_for_replay_locks_env_construction_latents`

### 16.7 After-fix forced-seed stability

Artifact:

- `/home/chen/RLPFN/artifacts/critic_rollout_seed_stability_probe_state_reset_separate_value_backbone_seed12345_after_freeze_fix.json`

Setup:

- keep:
  - `SEP state reset`
  - separate value backbone
- keep deterministic actor sampling
- compare repeated rollouts under:
  - current `collect_rollouts()`
  - forced fixed `env_rng_seeds + rollout_rng_seeds`

Results:

- current repeated-rollout target corr vs anchor:
  - rollout 1: `0.4411`
  - rollout 2: `-0.2984`
- forced-seed repeated-rollout target corr vs anchor:
  - rollout 1: `1.0000`
  - rollout 2: `1.0000`

Reading:

- after freezing the missing env-construction latents, the forced-seed rollout contract is now truly stable
- this is the point at which critic underfit can be discussed again
- before this fix, “critic underfit” was confounded by a broken fixed-env replay contract

### 16.8 Stable-contract recheck: learned vs zero

Artifact:

- `/home/chen/RLPFN/artifacts/critic_help_vs_zero_state_reset_separate_value_backbone_seed12345_forced_seed_minismoke.json`

Setup:

- fixed legacy `12345`
- `SEP state reset`
- separate value backbone
- forced fixed training seeds:
  - `env_rng_seed = 2020`
  - `rollout_rng_seed = 4040`
- minimal scope only:
  - `eval_env_count = 1`
  - `outer_epochs = 1`

Results:

- learned:
  - `delta_smp_gap = +4.6850`
- zero:
  - `delta_smp_gap = +4.2308`

Reading:

- once the rollout target is actually stable, learned critic is no longer obviously worse than zero in this fixed-env recheck
- this does **not** yet prove a stable advantage for critic
- but it does invalidate the earlier stronger claim that “separate critic still loses to zero” without first controlling the rollout contract

### 16.9 Stable-contract canonical quick

Artifact:

- `/home/chen/RLPFN/artifacts/critic_help_vs_zero_state_reset_separate_value_backbone_seed12345_forced_seed_quick.json`

Setup:

- same fixed legacy `12345`
- same `SEP state reset`
- same separate value backbone
- same forced training seeds:
  - `env_rng_seed = 2020`
  - `rollout_rng_seed = 4040`
- canonical quick scope:
  - `eval_env_count = 8`
  - `outer_epochs = 2`

Results:

- learned:
  - `delta_smp_gap = +0.1315`
- zero:
  - `delta_smp_gap = -1.9235`

Reading:

- under the stabilized rollout contract, learned critic remains competitive when scaled up from the minimal recheck to the canonical quick setting
- more specifically, in this fixed-env canonical quick, learned now clearly beats zero
- that means the earlier “critic still loses to zero even after separation” result was not reliable without controlling the rollout contract

### 16.10 Current diagnosis

What is now supported:

- fixed-buffer critic fit is possible
- forced-seed repeated-rollout target stability is now achieved
- under that stable contract, critic is not worse than zero in either:
  - the minimal fixed-env recheck
  - or the canonical quick fixed-env recheck

### 16.11 Shared-backbone forced-seed canonical quick

Artifact:

- `/home/chen/RLPFN/artifacts/critic_help_vs_zero_state_reset_shared_seed12345_forced_seed_quick.json`

Setup:

- same fixed legacy `12345`
- same `SEP state reset`
- shared actor/critic backbone
- same forced training seeds:
  - `env_rng_seed = 2020`
  - `rollout_rng_seed = 4040`
- canonical quick scope:
  - `eval_env_count = 8`
  - `outer_epochs = 2`

Results:

- learned:
  - `delta_smp_gap = +1.0642`
- zero:
  - `delta_smp_gap = -2.2664`

Reading:

- under the same stabilized rollout contract, shared backbone also beats zero
- therefore current evidence does **not** support “shared representation is the main critic bottleneck”
- earlier negative shared-backbone readings were also confounded by rollout-contract drift
- under this controlled regime, shared is not obviously worse than separate:
  - separate forced-seed canonical quick:
    - learned `+0.1315`
    - zero `-1.9235`
  - shared forced-seed canonical quick:
    - learned `+1.0642`
    - zero `-2.2664`

What is not yet supported:

- that shared history-state representation is the main bottleneck
- that critic underfit alone fully explains all remaining non-forced PPO failures
- that critic is already a robust positive contributor in the unmodified training-path audit regime

So the next critic question, if this line continues, is narrower:

- why the actual PPO training path still violates the stabilized contract unless seeds/latents are explicitly locked
- and which remaining non-forced training-path stochastic source makes critic quality collapse outside this controlled regime

### 16.12 Seed-source ablation on the default PPO training path

Artifact:

- `/home/chen/RLPFN/artifacts/critic_rollout_seed_source_ablation_shared_seed12345.json`

Script:

- `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/critic_rollout_seed_source_ablation.py`

Purpose:

- isolate which missing seed plumbing in `collect_rollouts()` is responsible for the remaining drift relative to the stabilized contract
- compare four conditions:
  - no injected seeds
  - `env_rng_seed` only
  - `rollout_rng_seed` only
  - both

Setup:

- fixed legacy `12345`
- shared backbone
- `SEP state reset`
- deterministic actor sampling
- `single_eval_pos = 64`
- `n_steps = 256`

Results:

- no injected seeds:
  - rollout 1 corr: `-0.1707`
  - rollout 2 corr: `+0.0079`
- `env_rng_seed` only:
  - rollout 1 corr: `+0.0524`
  - rollout 2 corr: `+0.1774`
- `rollout_rng_seed` only:
  - rollout 1 corr: `-0.2223`
  - rollout 2 corr: `-0.5449`
- both:
  - rollout 1 corr: `+1.0000`
  - rollout 2 corr: `+1.0000`

Reading:

- the remaining deviation is **not** explained by a single missing seed
- `env_rng_seed` alone is insufficient
- `rollout_rng_seed` alone is insufficient
- the stabilized contract requires **both**:
  - environment-construction RNG control
  - rollout-stream RNG control

Concrete code interpretation:

- `collect_rollouts()` in:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/sb3_recurrent_ppo.py`
  - currently calls `_rollout_family_group_vectorized_with_policy(...)` without:
    - `env_rng_seeds`
    - `rollout_rng_seeds`
- `_rollout_family_group_vectorized_with_policy()` in:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/priors/environment_prior.py`
  - already supports both seed channels
- therefore the actual PPO training path is still bypassing the exact seed plumbing needed to reproduce the stabilized contract

Decision:

- for real stochastic training, these seeds should **not** be globally fixed by default
- but for:
  - fixed-env PPO regression
  - critic-fit debugging
  - any “single environment” audit that expects repeated-rollout comparability
- both seed channels need to be formally threaded into the PPO training path under an explicit strict/fixed-env mode

### 16.13 Official strict/fixed-env mode replaces monkey patch compares

Code change:

- `build_recurrent_ppo()` and `collect_rollouts()` now expose an official strict/fixed-env mode in:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/sb3_recurrent_ppo.py`
- official parameters:
  - `strict_fixed_env_mode`
  - `env_rng_seeds`
  - `rollout_rng_seeds`
- `collect_rollouts()` now forwards both seed channels directly into:
  - `_rollout_family_group_vectorized_with_policy(...)`

Protection:

- regression tests:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/tests/test_sb3_recurrent_ppo.py`
  - build-time propagation of strict mode + seed specs
  - rollout-time forwarding of both seed channels

Official canonical quick artifacts:

- shared backbone:
  - `/home/chen/RLPFN/artifacts/critic_help_vs_zero_state_reset_shared_seed12345_strict_fixed_env_quick.json`
  - learned `delta_smp_gap = -1.0160`
  - zero `delta_smp_gap = -2.2350`
- separate value backbone:
  - `/home/chen/RLPFN/artifacts/critic_help_vs_zero_state_reset_separate_value_backbone_seed12345_strict_fixed_env_quick.json`
  - learned `delta_smp_gap = -0.7333`
  - zero `delta_smp_gap = -2.2814`

Comparison summary:

- `/home/chen/RLPFN/artifacts/critic_strict_fixed_env_official_compare_summary.json`

Reading:

- the official strict/fixed-env mode confirms the same ranking-level conclusion as the earlier monkey-patched compare:
  - shared: `learned > zero`
  - separate value backbone: `learned > zero`
- the absolute `delta_smp_gap` numbers changed, so exact numeric parity with the monkey-patch artifacts should **not** be treated as the invariant
- the protected conclusion is narrower and cleaner:
  - under an officially stabilized rollout contract, learned critic still beats zero in both shared and separate configurations

Decision:

- monkey-patch forced-seed compares are now superseded by the official strict/fixed-env mode
- for future phase-2 critic claims, only the official strict-mode path should be treated as trusted

### 16.14 Deterministic actor sampling for numeric fixed-env regression

Question:

- why did the official strict-mode canonical quick turn previously positive fixed-env tests into negative `delta_smp_gap` values?

Answer:

- not because policy action sampling RNG was still uncontrolled
- action sampling noise is already driven by the rollout generators in:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/priors/environment_prior.py`
  - specifically the action epsilon passed into `_resolve_policy_action_sample(...)`
    comes from the same `rollout_generators` stream used by `rollout_rng_seeds`
- the remaining issue is that sampled-policy fixed-env compares are still too high-variance for **numeric** regression, even when they are acceptable for ranking-level critic usefulness

Code change:

- official deterministic actor sampling flag added to:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/sb3_recurrent_ppo.py`
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/critic_free_single_env_audit.py`
- this is now formal, not monkey-patched

Protection:

- regression tests added in:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/tests/test_sb3_recurrent_ppo.py`
- new checks:
  - deterministic actor sampling flag propagates through `build_recurrent_ppo()`
  - `collect_rollouts()` overrides the actor sample function to return `action_mean`

Deterministic strict-mode canonical quick artifacts:

- shared backbone:
  - `/home/chen/RLPFN/artifacts/critic_help_vs_zero_state_reset_shared_seed12345_strict_fixed_env_deterministic_quick.json`
  - learned `delta_smp_gap = +0.7248`
  - zero `delta_smp_gap = +0.1518`
- separate value backbone:
  - `/home/chen/RLPFN/artifacts/critic_help_vs_zero_state_reset_separate_value_backbone_seed12345_strict_fixed_env_deterministic_quick.json`
  - learned `delta_smp_gap = +0.1450`
  - zero `delta_smp_gap = -0.2084`

Summary:

- `/home/chen/RLPFN/artifacts/critic_strict_fixed_env_deterministic_regression_summary.json`

Reading:

- the sign flip in official strict sampled mode was too severe for long-term numeric regression
- but it was **not** evidence that final reward semantics had become fundamentally wrong
- once actor sampling is made deterministic under the same stabilized contract:
  - shared returns to positive numeric improvement
  - separate also returns to positive learned improvement
- therefore:
  - sampled strict mode is suitable for ranking-level critic audits
  - deterministic strict mode is the correct path for fixed-env **numeric** regression

### 16.15 Official strict-mode regression against the legacy monkey patch

Question:

- does the maintained official strict/fixed-env implementation still numerically match the legacy monkey-patch path at the gradient level?
- and was the earlier “official is slower / official numerically diverges” signal a real regression?

Formal regression harness:

- script:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/critic_official_vs_monkey_regression.py`
- artifact:
  - `/home/chen/RLPFN/artifacts/critic_official_vs_monkey_regression_seed12345.json`

Protocol:

- fixed legacy checkpoint:
  - `/home/chen/RLPFN/rwkv/models_diff/rlpfn_03_30_2026_22_37_53_epoch_13.cpkt`
- fixed environment contract:
  - `frozen_h_seed = 12345`
  - `train_env_seed = 2020`
  - `train_rollout_seed = 4040`
  - `single_eval_pos = 64`
  - `n_steps = 256`
  - `batch_size = 256`
  - `n_epochs = 1`
  - `learning_rate = 2e-4`
  - `target_kl = 0.03`
- compare method:
  - first report the unsynced separate-build policy head mismatch
  - then do the actual official-vs-monkey compare on the **same algo instance**, toggling only rollout plumbing
  - use official strict fixed-env mode plus deterministic actor sampling

Important finding:

- the earlier large official-vs-monkey numeric mismatch was mostly a **bad compare harness**
- two separately built algos do **not** start from identical policy parameters, even with the same checkpoint
- the drifting tensors are only:
  - `action_net.weight`
  - `value_net.weight`
- these PPO heads are freshly initialized and are not checkpoint-restored in a way that makes two separate builds identical

Unsynced separate-build diff:

- `policy_param_max_abs_diff = 0.2662747502`
- differing tensors:
  - `action_net.weight`
  - `value_net.weight`

Official result on the correct same-instance toggle compare:

- first-step compare after sync:
  - `action_mean_max_abs_diff = 2.4052336812019348e-05`
  - `values_max_abs_diff = 5.376040935516357e-04`
- full rollout + policy-gradient compare after sync:
  - `policy_loss_abs_diff = 0.0`
  - `grad_cosine = 1.0000032186508179`
  - `grad_l2_delta = 0.0`
  - `actions_max_abs_diff = 0.0`
  - `returns_max_abs_diff = 0.0`
  - `values_max_abs_diff = 0.0`
  - `advantages_max_abs_diff = 0.0`
  - `log_probs_max_abs_diff = 0.0`
- warm timing:
  - `official_mean_s = 5.2245651858`
  - `monkey_mean_s = 5.1640754031`
  - `official_over_monkey_ratio = 1.0117135746`

Reading:

- the maintained official strict/fixed-env path is numerically aligned with the legacy monkey patch when compared correctly
- the remaining timing difference is about `1.2%`, which is within normal noise for this GPU path and is **not** evidence of a semantic regression
- the earlier large rollout/gradient drift and “official slower” interpretation should be treated as superseded

Protection policy:

- the trusted fixed-env regression protocol for phase 2 is now:
  - official `strict_fixed_env_mode=True`
  - official `deterministic_actor_sampling=True`
  - real checkpoint + fixed legacy env seeds
  - same-instance official-vs-monkey toggle compare
- do **not** use separate-build official-vs-monkey compares as primary evidence
- lightweight unit protection remains in:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/tests/test_sb3_recurrent_ppo.py`
  - covered contracts:
    - strict fixed-env seed plumbing
    - deterministic actor sampling override

### 16.16 Shared actor-critic positive-gap positive-return milestone

Milestone artifact:

- `/home/chen/RLPFN/artifacts/critic_help_vs_zero_state_reset_shared_seed12345_strict_fixed_env_deterministic_quick.json`

Why this is now a milestone:

- this is the first maintained, official, non-monkey-patched phase-2 configuration that simultaneously satisfies:
  - shared actor-critic backbone
  - official strict fixed-env rollout contract
  - deterministic numeric regression path
  - positive learned `delta_smp_gap`
  - positive learned post-training return

Frozen milestone contract:

- shared actor-critic backbone
- `ppo_reset_env_state_at_sep = True`
- `strict_fixed_env_mode = True`
- `deterministic_actor_sampling = True`
- `frozen_h_seed = 12345`
- `train_env_seed = 2020`
- `train_rollout_seed = 4040`
- `single_eval_pos = 64`
- `n_steps = 256`
- `batch_size = 256`
- `n_epochs = 1`
- `outer_epochs = 2`
- `target_kl = 0.03`

Reference values:

- `zero_suffix_return_mean = -0.5634000301`
- learned:
  - `pre_smp_gap = -0.0283457041`
  - `post_smp_gap = +0.6964260340`
  - `delta_smp_gap = +0.7247717381`
  - `pre_return = -0.5917457342`
  - `post_return = +0.1330260038`
- zero:
  - `pre_smp_gap = +0.0325402021`
  - `post_smp_gap = +0.1843308210`
  - `delta_smp_gap = +0.1517906189`
  - `pre_return = -0.5308598280`
  - `post_return = -0.3790692091`

Decision:

- treat this artifact as the phase-2 shared-backbone milestone
- future phase-2 maintenance and fixed-env numeric comparisons should anchor to this contract before accepting any new actor/critic change

### 16.17 Deterministic minibatch order formalized into the fixed-env contract

Question:

- is minibatch order still an uncontrolled random source in fixed-env numeric regression?

Answer:

- it was previously controlled only through audit-side monkey helper patching
- that was acceptable for experiments, but not sufficient as a maintained contract

Code change:

- PPO rollout buffer now has a formal deterministic batch-plan switch in:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/sb3_recurrent_ppo.py`
- builder parameter added:
  - `deterministic_batch_plan`
- audit/regression wiring updated:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/critic_free_single_env_audit.py`
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/critic_official_vs_monkey_regression.py`

Protection:

- regression tests in:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/tests/test_sb3_recurrent_ppo.py`
- covered:
  - builder propagates deterministic batch-plan flag
  - rollout buffer uses stable ordered minibatches when enabled
  - strict fixed-env seed plumbing still intact
  - deterministic actor sampling override still intact

Targeted test result:

- `4 passed`

Updated regression artifact:

- `/home/chen/RLPFN/artifacts/critic_official_vs_monkey_regression_seed12345.json`

Reading:

- the formalized deterministic batch-plan change does **not** change the official-vs-monkey gradient conclusion
- after the change:
  - `policy_loss_abs_diff = 0.0`
  - `grad_cosine = 1.000003695487976`
  - `grad_l2_delta = 0.0`
  - rollout tensor diffs remain exactly `0.0`
- warm timing remains near parity:
  - `official_over_monkey_ratio = 1.0160`

Decision:

- fixed-env numeric compare now explicitly locks:
  - env latent
  - env seed
  - rollout seed
  - actor sampling
  - minibatch order
- default stochastic training semantics remain unchanged
- the deterministic batch-plan switch is part of the trusted regression contract, not the default training path

### 16.18 Phase 2 frozen; regression pack promoted

Decision:

- stop Phase 2 exploratory development
- keep only regression protection and milestone maintenance
- move active optimization work to Phase 3

Maintained guardrail doc:

- `/home/chen/RLPFN/artifacts/phase2_regression_guardrail.md`

Single maintained rerun entrypoint:

- `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase2_regression_suite.py`

What the regression pack reruns:

- official-vs-monkey gradient-level regression
- shared-backbone fixed-env deterministic milestone compare

Generated summary artifact:

- `/home/chen/RLPFN/artifacts/phase2_regression_pack/phase2_regression_suite_summary.json`

Current pack status:

- assembled successfully in fast mode from the maintained canonical artifacts
- pass flags:
  - official-vs-monkey numeric match: `true`
  - shared learned beats zero: `true`
  - shared learned post return positive: `true`
- manifest validation:
  - `/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase2_regression_manifest.json`
  - `all_checks_pass = true`
  - timing ratio and exact gradient/tensor invariants are now explicit checks, not just narrative conclusions
- guardrail behavior:
  - valid summary is reused by default to avoid repeated runs
  - output directory lock prevents concurrent regression-pack overwrite
  - validation failure returns non-zero by default

Purpose:

- keep the trusted Phase 2 branch usable
- preserve the important legacy comparisons in one place
- give Phase 3 a single preflight regression gate instead of scattered ad hoc artifacts
