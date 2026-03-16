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


def _score_candidate_action_jobs(
    *,
    model,
    device,
    jobs,
    obs_slot_dim,
    action_slot_dim,
    num_features,
    terminal_token_enabled,
):
    if not jobs:
        return []

    grouped = {}
    for job_idx, job in enumerate(jobs):
        t = int(len(job["x_hist"]))
        grouped.setdefault(t, []).append((job_idx, job))

    score_list = [None] * len(jobs)
    with torch.no_grad():
        for t, entries in grouped.items():
            n_candidates = int(entries[0][1]["action_candidates"].shape[0])
            if t <= 0:
                for job_idx, _ in entries:
                    score_list[job_idx] = np.zeros((n_candidates,), dtype=np.float32)
                continue

            total_columns = len(entries) * n_candidates
            x_stack = np.zeros((t + 1, total_columns, int(num_features)), dtype=np.float32)
            y_stack = np.zeros((t + 1, total_columns), dtype=np.float32)

            for entry_idx, (job_idx, job) in enumerate(entries):
                col_start = entry_idx * n_candidates
                col_end = col_start + n_candidates
                x_prev = np.asarray(job["x_hist"], dtype=np.float32)
                y_prev = np.asarray(job["y_hist"], dtype=np.float32)
                x_stack[:t, col_start:col_end, :] = np.repeat(x_prev[:, None, :], n_candidates, axis=1)
                y_stack[:t, col_start:col_end] = np.repeat(y_prev[:, None], n_candidates, axis=1)
                for candidate_idx in range(n_candidates):
                    x_stack[t, col_start + candidate_idx, :] = _pack_token(
                        obs=job["obs"],
                        action=job["action_candidates"][candidate_idx],
                        reward=job["prev_reward"],
                        reward_mask=1.0,
                        obs_slot_dim=obs_slot_dim,
                        action_slot_dim=action_slot_dim,
                        num_features=num_features,
                        phase=job["phase"],
                        terminal=job["terminal"],
                        terminal_token_enabled=terminal_token_enabled,
                    )

            x_tensor = torch.from_numpy(x_stack).to(device=device)
            y_tensor = torch.from_numpy(y_stack).to(device=device)
            out = model((x_tensor, y_tensor), single_eval_pos=t)
            scores = out[0, :, 0].detach().float().cpu().numpy().reshape(len(entries), n_candidates)
            for entry_idx, (job_idx, _) in enumerate(entries):
                score_list[job_idx] = scores[entry_idx]

    return score_list


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
    return _score_candidate_action_jobs(
        model=model,
        device=device,
        jobs=[
            {
                "x_hist": x_hist,
                "y_hist": y_hist,
                "obs": obs,
                "prev_reward": prev_reward,
                "terminal": terminal,
                "phase": phase,
                "action_candidates": action_candidates,
            }
        ],
        obs_slot_dim=obs_slot_dim,
        action_slot_dim=action_slot_dim,
        num_features=num_features,
        terminal_token_enabled=terminal_token_enabled,
    )[0]


def _sample_action_candidates(rng, action_low, action_high, n_candidates):
    candidates = rng.uniform(
        low=action_low,
        high=action_high,
        size=(max(1, int(n_candidates)), action_low.shape[0]),
    ).astype(np.float32)
    candidates[0] = np.clip(np.zeros_like(action_low), action_low, action_high)
    return candidates


def _reset_validation_rollout(state, preserve_prev=False):
    obs, _ = state["env"].reset(seed=int(state["next_reset_seed"]))
    state["next_reset_seed"] += 1
    state["obs"] = obs
    if not bool(preserve_prev):
        state["prev_reward"] = 0.0
        state["prev_terminal"] = 0.0
    state["current_rollout_return"] = 0.0
    state["current_rollout_len"] = 0


def _finalize_validation_rollout(state, context_lower_bound):
    rollout_len = int(state["current_rollout_len"])
    rollout_return = float(state["current_rollout_return"])
    state["context_len"] += rollout_len

    if bool(state["phase_flag"] >= 1.0):
        state["reported_return"] = rollout_return
        state["reported_len"] = float(rollout_len)
        state["context_len_before_eval"] = float(state["context_len"] - rollout_len)
        state["explore_rollout_count"] = float(len(state["explore_rollout_lengths"]))
        state["explore_rollout_len_mean"] = (
            float(np.mean(state["explore_rollout_lengths"])) if state["explore_rollout_lengths"] else 0.0
        )
        state["done"] = True
        return

    state["explore_rollout_lengths"].append(rollout_len)
    if rollout_len <= 0:
        state["reported_return"] = float("nan")
        state["reported_len"] = 0.0
        state["context_len_before_eval"] = float(state["context_len"])
        state["explore_rollout_count"] = float(len(state["explore_rollout_lengths"]))
        state["explore_rollout_len_mean"] = (
            float(np.mean(state["explore_rollout_lengths"])) if state["explore_rollout_lengths"] else 0.0
        )
        state["done"] = True
        return

    mean_explore_len = float(np.mean(state["explore_rollout_lengths"]))
    state["phase_flag"] = 1.0 if float(state["context_len"]) + mean_explore_len > float(context_lower_bound) else 0.0
    _reset_validation_rollout(state, preserve_prev=True)


def _build_validation_episode_states(
    *,
    gym,
    env_name,
    episodes,
    base_seed,
):
    episode_states = []
    for ep in range(int(episodes)):
        env = gym.make(env_name)
        rng_seed = int(base_seed + ep)
        reset_seed = int(base_seed + (ep * 1000))
        state = {
            "env": env,
            "rng": np.random.default_rng(rng_seed),
            "x_hist": [],
            "y_hist": [],
            "context_len": 0,
            "explore_rollout_lengths": [],
            "phase_flag": 0.0,
            "done": False,
            "reported_return": None,
            "reported_len": None,
            "context_len_before_eval": None,
            "explore_rollout_count": None,
            "explore_rollout_len_mean": None,
            "next_reset_seed": reset_seed,
        }
        _reset_validation_rollout(state)
        episode_states.append(state)
    return episode_states


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
    phase_token_enabled = True
    terminal_token_enabled = bool(env_cfg.get("terminal_reset_enabled", False))
    default_num_features = int(obs_slot_dim) + 2 + int(phase_token_enabled) + int(terminal_token_enabled) + int(action_slot_dim)
    num_features = int(config.get("prior", {}).get("num_features", default_num_features))
    device = config.get("device", "cpu")

    was_training = model.training
    model.eval()

    per_env = {}
    all_env_means = []

    try:
        for env_name in env_names:
            try:
                probe_env = gym.make(env_name)
            except Exception:
                per_env[env_name] = {"return_mean": float("nan"), "len_mean": float("nan"), "make_failed": 1}
                continue

            if not isinstance(probe_env.action_space, Box):
                per_env[env_name] = {
                    "return_mean": float("nan"),
                    "len_mean": float("nan"),
                    "unsupported_action_space": 1,
                }
                try:
                    probe_env.close()
                except Exception:
                    pass
                continue

            action_low = np.asarray(probe_env.action_space.low, dtype=np.float32).reshape(-1)
            action_high = np.asarray(probe_env.action_space.high, dtype=np.float32).reshape(-1)
            try:
                probe_env.close()
            except Exception:
                pass

            states = _build_validation_episode_states(
                gym=gym,
                env_name=env_name,
                episodes=episodes,
                base_seed=base_seed,
            )

            try:
                while any(not state["done"] for state in states):
                    jobs = []
                    job_state_indices = []
                    for state_idx, state in enumerate(states):
                        if bool(state["done"]):
                            continue
                        if int(state["current_rollout_len"]) >= int(max_steps):
                            _finalize_validation_rollout(state, context_lower_bound)
                            continue
                        candidates = _sample_action_candidates(
                            rng=state["rng"],
                            action_low=action_low,
                            action_high=action_high,
                            n_candidates=n_candidates,
                        )
                        jobs.append(
                            {
                                "x_hist": state["x_hist"],
                                "y_hist": state["y_hist"],
                                "obs": state["obs"],
                                "prev_reward": state["prev_reward"],
                                "terminal": state["prev_terminal"],
                                "phase": state["phase_flag"],
                                "action_candidates": candidates,
                            }
                        )
                        job_state_indices.append((state_idx, candidates))

                    if jobs:
                        score_list = _score_candidate_action_jobs(
                            model=model,
                            device=device,
                            jobs=jobs,
                            obs_slot_dim=obs_slot_dim,
                            action_slot_dim=action_slot_dim,
                            num_features=num_features,
                            terminal_token_enabled=terminal_token_enabled,
                        )
                    else:
                        score_list = []

                    for (state_idx, candidates), scores in zip(job_state_indices, score_list):
                        state = states[state_idx]
                        action = candidates[int(np.argmax(scores))]
                        state["x_hist"].append(
                            _pack_token(
                                obs=state["obs"],
                                action=action,
                                reward=state["prev_reward"],
                                reward_mask=1.0,
                                obs_slot_dim=obs_slot_dim,
                                action_slot_dim=action_slot_dim,
                                num_features=num_features,
                                phase=state["phase_flag"],
                                terminal=state["prev_terminal"],
                                terminal_token_enabled=terminal_token_enabled,
                            )
                        )
                        obs_next, reward_next, terminated, truncated, _ = state["env"].step(action.astype(np.float32))
                        reward_next = float(reward_next)
                        done_flag = bool(terminated or truncated)
                        state["y_hist"].append(reward_next)
                        state["obs"] = obs_next
                        state["prev_reward"] = reward_next
                        state["prev_terminal"] = 1.0 if done_flag else 0.0
                        state["current_rollout_return"] += reward_next
                        state["current_rollout_len"] += 1

                        if done_flag or int(state["current_rollout_len"]) >= int(max_steps):
                            if int(state["current_rollout_len"]) >= int(max_steps):
                                state["prev_terminal"] = 1.0
                            _finalize_validation_rollout(state, context_lower_bound)

                returns = [float(state["reported_return"]) for state in states if state["reported_return"] is not None]
                lengths = [float(state["reported_len"]) for state in states if state["reported_len"] is not None]
                context_lengths = [
                    float(state["context_len_before_eval"])
                    for state in states
                    if state["context_len_before_eval"] is not None
                ]
                explore_rollout_counts = [
                    float(state["explore_rollout_count"])
                    for state in states
                    if state["explore_rollout_count"] is not None
                ]
                explore_rollout_len_means = [
                    float(state["explore_rollout_len_mean"])
                    for state in states
                    if state["explore_rollout_len_mean"] is not None
                ]
            finally:
                for state in states:
                    try:
                        state["env"].close()
                    except Exception:
                        pass

            if returns:
                mean_ret = float(np.mean(returns))
                mean_len = float(np.mean(lengths))
                per_env[env_name] = {
                    "return_mean": mean_ret,
                    "len_mean": mean_len,
                    "make_failed": 0,
                    "context_len_before_eval_mean": (
                        float(np.mean(context_lengths)) if context_lengths else float("nan")
                    ),
                    "explore_rollout_count_mean": (
                        float(np.mean(explore_rollout_counts)) if explore_rollout_counts else float("nan")
                    ),
                    "explore_rollout_len_mean": (
                        float(np.mean(explore_rollout_len_means)) if explore_rollout_len_means else float("nan")
                    ),
                }
                all_env_means.append(mean_ret)
    finally:
        if was_training:
            model.train()
        else:
            model.eval()

    global_mean = float(np.mean(all_env_means)) if all_env_means else float("nan")
    return global_mean, per_env
