import numpy as np
import torch


RLPFN_DEFAULT_OOP_ENVS = [
    "InvertedPendulum-v5",
    "InvertedDoublePendulum-v5",
    "Hopper-v5",
    "Walker2d-v5",
    "Humanoid-v5",
    "Reacher-v5",
    "LunarLanderContinuous-v3",
    "HalfCheetah-v5",
    "Ant-v5",
    "Swimmer-v5",
    "Pendulum-v1",
    "BipedalWalker-v3",
    "MountainCarContinuous-v0",
]


def parse_env_list(value):
    if value is None:
        return list(RLPFN_DEFAULT_OOP_ENVS)
    if isinstance(value, (list, tuple)):
        return [str(x).strip() for x in value if str(x).strip()]
    return [x.strip() for x in str(value).split(",") if x.strip()]


def _to_1d_array(x):
    arr = np.asarray(x, dtype=np.float32)
    return arr.reshape(-1)


def _pack_token(
    obs,
    action,
    reward,
    reward_mask,
    obs_slot_dim,
    action_slot_dim,
    num_features,
    phase=0.0,
    terminal=0.0,
    terminal_token_enabled=False,
):
    obs = _to_1d_array(obs)
    action = _to_1d_array(action)
    obs_slot = np.zeros((int(obs_slot_dim),), dtype=np.float32)
    action_slot = np.zeros((int(action_slot_dim),), dtype=np.float32)

    obs_take = min(obs_slot.shape[0], obs.shape[0])
    action_take = min(action_slot.shape[0], action.shape[0])
    if obs_take > 0:
        obs_slot[:obs_take] = obs[:obs_take]
    if action_take > 0:
        action_slot[:action_take] = action[:action_take]

    parts = [
        obs_slot,
        np.array([float(reward)], dtype=np.float32),
        np.array([float(reward_mask)], dtype=np.float32),
        np.array([float(phase)], dtype=np.float32),
    ]
    if bool(terminal_token_enabled):
        parts.append(np.array([float(terminal)], dtype=np.float32))
    parts.append(action_slot)
    token = np.concatenate(parts, axis=0)
    if token.shape[0] > int(num_features):
        token = token[: int(num_features)]
    elif token.shape[0] < int(num_features):
        token = np.concatenate(
            [token, np.zeros((int(num_features) - token.shape[0],), dtype=np.float32)],
            axis=0,
        )
    return token


def _score_candidate_actions(
    model,
    device,
    x_hist,
    y_hist,
    obs,
    prev_reward,
    action_candidates,
    obs_slot_dim,
    action_slot_dim,
    num_features,
    phase=0.0,
    terminal=0.0,
    terminal_token_enabled=False,
):
    t = len(x_hist)
    n_candidates = action_candidates.shape[0]
    # In single_eval_causal mode the model requires at least one prefix token.
    # At t=0 there is no history yet, so we return tied scores and rely on
    # deterministic tie-breaking (argmax -> index 0 anchor action).
    if t == 0:
        return np.zeros((n_candidates,), dtype=np.float32)

    x_stack = np.zeros((t + 1, n_candidates, int(num_features)), dtype=np.float32)
    y_stack = np.zeros((t + 1, n_candidates), dtype=np.float32)

    if t > 0:
        x_prev = np.asarray(x_hist, dtype=np.float32)  # (t, F)
        y_prev = np.asarray(y_hist, dtype=np.float32)  # (t,)
        x_stack[:t, :, :] = np.repeat(x_prev[:, None, :], n_candidates, axis=1)
        y_stack[:t, :] = np.repeat(y_prev[:, None], n_candidates, axis=1)

    for i in range(n_candidates):
        x_stack[t, i, :] = _pack_token(
            obs=obs,
            action=action_candidates[i],
            reward=prev_reward,
            reward_mask=1.0,
            obs_slot_dim=obs_slot_dim,
            action_slot_dim=action_slot_dim,
            num_features=num_features,
            phase=phase,
            terminal=terminal,
            terminal_token_enabled=terminal_token_enabled,
        )

    with torch.no_grad():
        x_tensor = torch.from_numpy(x_stack).to(device=device)
        y_tensor = torch.from_numpy(y_stack).to(device=device)
        out = model((x_tensor, y_tensor), single_eval_pos=t)
        # out shape for eval tail: (1, B, n_out)
        scores = out[0, :, 0].detach().float().cpu().numpy()
    return scores


def _sample_action_candidates(rng, action_low, action_high, n_candidates):
    candidates = rng.uniform(
        low=action_low,
        high=action_high,
        size=(max(1, int(n_candidates)), action_low.shape[0]),
    ).astype(np.float32)
    candidates[0] = np.clip(np.zeros_like(action_low), action_low, action_high)
    return candidates


def _run_contextual_rollout(
    model,
    device,
    env,
    rng,
    obs,
    prev_reward,
    prev_terminal,
    x_hist,
    y_hist,
    action_low,
    action_high,
    n_candidates,
    max_steps,
    obs_slot_dim,
    action_slot_dim,
    num_features,
    phase_flag,
    terminal_token_enabled,
):
    rollout_return = 0.0
    rollout_len = 0
    obs_curr = obs
    prev_reward_curr = float(prev_reward)
    prev_terminal_curr = float(prev_terminal)

    for _ in range(int(max_steps)):
        candidates = _sample_action_candidates(
            rng=rng,
            action_low=action_low,
            action_high=action_high,
            n_candidates=n_candidates,
        )
        scores = _score_candidate_actions(
            model=model,
            device=device,
            x_hist=x_hist,
            y_hist=y_hist,
            obs=obs_curr,
            prev_reward=prev_reward_curr,
            action_candidates=candidates,
            obs_slot_dim=obs_slot_dim,
            action_slot_dim=action_slot_dim,
            num_features=num_features,
            phase=phase_flag,
            terminal=prev_terminal_curr,
            terminal_token_enabled=terminal_token_enabled,
        )
        action = candidates[int(np.argmax(scores))]
        x_hist.append(
            _pack_token(
                obs=obs_curr,
                action=action,
                reward=prev_reward_curr,
                reward_mask=1.0,
                obs_slot_dim=obs_slot_dim,
                action_slot_dim=action_slot_dim,
                num_features=num_features,
                phase=phase_flag,
                terminal=prev_terminal_curr,
                terminal_token_enabled=terminal_token_enabled,
            )
        )
        obs_next, reward_next, terminated, truncated, _ = env.step(action.astype(np.float32))
        reward_next = float(reward_next)
        y_hist.append(reward_next)
        rollout_return += reward_next
        rollout_len += 1
        obs_curr = obs_next
        prev_reward_curr = reward_next
        prev_terminal_curr = 1.0 if bool(terminated or truncated) else 0.0
        if bool(terminated or truncated):
            break

    if rollout_len >= int(max_steps):
        prev_terminal_curr = 1.0
    return {
        "obs": obs_curr,
        "prev_reward": prev_reward_curr,
        "prev_terminal": prev_terminal_curr,
        "rollout_return": float(rollout_return),
        "rollout_len": int(rollout_len),
    }


def evaluate_rlpfn_on_gym_envs(model, config):
    try:
        import gymnasium as gym
        from gymnasium.spaces import Box
    except Exception as exc:
        return float("nan"), {"__error__": f"gymnasium_import_failed: {exc}"}

    orch = config.get("orchestration", {})
    env_names = parse_env_list(orch.get("rl_validate_envs", None))
    episodes = int(orch.get("rl_validate_episodes", 3))
    max_steps = int(orch.get("rl_validate_max_steps", 1000))
    n_candidates = int(orch.get("rl_validate_action_candidates", 16))
    base_seed = int(orch.get("rl_validate_seed", 1))
    context_lower_bound = int(orch.get("rl_validate_context_lower_bound", 2048))

    env_cfg = config.get("prior", {}).get("environment", {})
    obs_slot_dim = int(env_cfg.get("obs_slot_dim", 400))
    action_slot_dim = int(env_cfg.get("action_slot_dim", 30))
    num_features = int(config.get("prior", {}).get("num_features", obs_slot_dim + action_slot_dim + 2))
    terminal_token_enabled = bool(env_cfg.get("terminal_reset_enabled", False))
    device = config.get("device", "cpu")

    was_training = model.training
    model.eval()

    per_env = {}
    all_env_means = []

    for env_name in env_names:
        returns = []
        lengths = []
        context_lengths = []
        explore_rollout_counts = []
        explore_rollout_len_means = []
        make_failed = False
        try:
            env = gym.make(env_name)
        except Exception:
            make_failed = True
            per_env[env_name] = {"return_mean": float("nan"), "len_mean": float("nan"), "make_failed": 1}
            continue

        if not isinstance(env.action_space, Box):
            per_env[env_name] = {"return_mean": float("nan"), "len_mean": float("nan"), "unsupported_action_space": 1}
            try:
                env.close()
            except Exception:
                pass
            continue

        action_low = np.asarray(env.action_space.low, dtype=np.float32).reshape(-1)
        action_high = np.asarray(env.action_space.high, dtype=np.float32).reshape(-1)

        for ep in range(episodes):
            rng = np.random.default_rng(base_seed + ep)
            rollout_seed = int(base_seed + ep * 1000)
            obs, _ = env.reset(seed=rollout_seed)
            x_hist = []
            y_hist = []
            prev_reward = 0.0
            prev_terminal = 0.0
            context_len = 0
            explore_rollout_lengths = []

            while True:
                should_exploit = False
                if explore_rollout_lengths:
                    mean_explore_len = float(np.mean(explore_rollout_lengths))
                    should_exploit = bool(float(context_len) + mean_explore_len > float(context_lower_bound))
                phase_flag = 1.0 if should_exploit else 0.0
                rollout = _run_contextual_rollout(
                    model=model,
                    device=device,
                    env=env,
                    rng=rng,
                    obs=obs,
                    prev_reward=prev_reward,
                    prev_terminal=prev_terminal,
                    x_hist=x_hist,
                    y_hist=y_hist,
                    action_low=action_low,
                    action_high=action_high,
                    n_candidates=n_candidates,
                    max_steps=max_steps,
                    obs_slot_dim=obs_slot_dim,
                    action_slot_dim=action_slot_dim,
                    num_features=num_features,
                    phase_flag=phase_flag,
                    terminal_token_enabled=terminal_token_enabled,
                )
                obs = rollout["obs"]
                prev_reward = rollout["prev_reward"]
                prev_terminal = rollout["prev_terminal"]
                context_len += int(rollout["rollout_len"])
                if bool(phase_flag >= 0.5):
                    returns.append(float(rollout["rollout_return"]))
                    lengths.append(float(rollout["rollout_len"]))
                    context_lengths.append(float(context_len - int(rollout["rollout_len"])))
                    explore_rollout_counts.append(float(len(explore_rollout_lengths)))
                    explore_rollout_len_means.append(
                        float(np.mean(explore_rollout_lengths)) if explore_rollout_lengths else 0.0
                    )
                    break

                explore_rollout_lengths.append(int(rollout["rollout_len"]))
                rollout_seed += 1
                obs, _ = env.reset(seed=rollout_seed)

        try:
            env.close()
        except Exception:
            pass

        if not make_failed and returns:
            mean_ret = float(np.mean(returns))
            mean_len = float(np.mean(lengths))
            per_env[env_name] = {
                "return_mean": mean_ret,
                "len_mean": mean_len,
                "context_len_before_eval_mean": (
                    float(np.mean(context_lengths)) if context_lengths else float("nan")
                ),
                "explore_rollout_count_mean": (
                    float(np.mean(explore_rollout_counts)) if explore_rollout_counts else float("nan")
                ),
                "explore_rollout_len_mean": (
                    float(np.mean(explore_rollout_len_means)) if explore_rollout_len_means else float("nan")
                ),
                "make_failed": 0,
            }
            all_env_means.append(mean_ret)

    if was_training:
        model.train()
    else:
        model.eval()

    global_mean = float(np.mean(all_env_means)) if all_env_means else float("nan")
    return global_mean, per_env
