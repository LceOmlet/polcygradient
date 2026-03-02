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


def _pack_token(obs, action, reward, reward_mask, obs_slot_dim, action_slot_dim, num_features):
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

    token = np.concatenate(
        [
            obs_slot,
            np.array([float(reward)], dtype=np.float32),
            np.array([float(reward_mask)], dtype=np.float32),
            action_slot,
        ],
        axis=0,
    )
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
        )

    with torch.no_grad():
        x_tensor = torch.from_numpy(x_stack).to(device=device)
        y_tensor = torch.from_numpy(y_stack).to(device=device)
        out = model((x_tensor, y_tensor), single_eval_pos=t)
        # out shape for eval tail: (1, B, n_out)
        scores = out[0, :, 0].detach().float().cpu().numpy()
    return scores


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

    env_cfg = config.get("prior", {}).get("environment", {})
    obs_slot_dim = int(env_cfg.get("obs_slot_dim", 400))
    action_slot_dim = int(env_cfg.get("action_slot_dim", 30))
    num_features = int(config.get("prior", {}).get("num_features", obs_slot_dim + action_slot_dim + 2))
    device = config.get("device", "cpu")

    was_training = model.training
    model.eval()

    per_env = {}
    all_env_means = []

    for env_name in env_names:
        returns = []
        lengths = []
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
        action_dim = action_low.shape[0]
        rng = np.random.default_rng(base_seed)

        for ep in range(episodes):
            obs, _ = env.reset(seed=base_seed + ep)
            x_hist = []
            y_hist = []
            prev_action = np.zeros((action_dim,), dtype=np.float32)
            prev_reward = 0.0
            ep_return = 0.0
            ep_len = 0

            for _ in range(max_steps):
                # Build candidate set in action space and score with model.
                candidates = rng.uniform(
                    low=action_low,
                    high=action_high,
                    size=(max(1, n_candidates), action_dim),
                ).astype(np.float32)
                # Keep a deterministic anchor candidate.
                candidates[0] = np.clip(np.zeros_like(action_low), action_low, action_high)

                scores = _score_candidate_actions(
                    model=model,
                    device=device,
                    x_hist=x_hist,
                    y_hist=y_hist,
                    obs=obs,
                    prev_reward=prev_reward,
                    action_candidates=candidates,
                    obs_slot_dim=obs_slot_dim,
                    action_slot_dim=action_slot_dim,
                    num_features=num_features,
                )
                best_idx = int(np.argmax(scores))
                action = candidates[best_idx]

                # Commit current token after choosing action.
                x_hist.append(
                    _pack_token(
                        obs=obs,
                        action=action,
                        reward=prev_reward,
                        reward_mask=1.0,
                        obs_slot_dim=obs_slot_dim,
                        action_slot_dim=action_slot_dim,
                        num_features=num_features,
                    )
                )
                obs_next, reward_next, terminated, truncated, _ = env.step(action.astype(np.float32))
                y_hist.append(float(reward_next))
                prev_reward = float(reward_next)
                prev_action = action
                obs = obs_next
                ep_return += float(reward_next)
                ep_len += 1
                if terminated or truncated:
                    break

            returns.append(float(ep_return))
            lengths.append(float(ep_len))

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
                "make_failed": 0,
            }
            all_env_means.append(mean_ret)

    if was_training:
        model.train()
    else:
        model.eval()

    global_mean = float(np.mean(all_env_means)) if all_env_means else float("nan")
    return global_mean, per_env
