import numpy as np
import torch

from ticl.rlpfn_maintained_path import resolve_rlpfn_token_layout


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
    max_parallel_columns=None,
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
            if max_parallel_columns is None:
                max_entries_per_call = len(entries)
            else:
                max_cols = int(max_parallel_columns)
                if max_cols <= 0:
                    max_entries_per_call = len(entries)
                else:
                    max_entries_per_call = max(1, max_cols // max(1, n_candidates))

            for chunk_start in range(0, len(entries), max_entries_per_call):
                chunk_entries = entries[chunk_start: chunk_start + max_entries_per_call]
                total_columns = len(chunk_entries) * n_candidates
                x_stack = np.zeros((t + 1, total_columns, int(num_features)), dtype=np.float32)
                y_stack = np.zeros((t + 1, total_columns), dtype=np.float32)

                for entry_idx, (job_idx, job) in enumerate(chunk_entries):
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
                scores = out[0, :, 0].detach().float().cpu().numpy().reshape(len(chunk_entries), n_candidates)
                for entry_idx, (job_idx, _) in enumerate(chunk_entries):
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


def _resolve_validation_action_bounds(action_space, *, fallback_abs_bound=10.0):
    fallback_abs_bound = float(max(0.1, fallback_abs_bound))

    action_shape = getattr(action_space, "shape", None)
    action_dim = None
    if isinstance(action_shape, tuple) and len(action_shape) > 0:
        try:
            action_dim = int(np.prod(action_shape))
        except Exception:
            action_dim = None

    low = getattr(action_space, "low", None)
    high = getattr(action_space, "high", None)
    if low is not None and high is not None:
        try:
            low_arr = np.asarray(low, dtype=np.float32).reshape(-1)
            high_arr = np.asarray(high, dtype=np.float32).reshape(-1)
        except Exception:
            low_arr = None
            high_arr = None
        if low_arr is not None and high_arr is not None and low_arr.shape == high_arr.shape and low_arr.size > 0:
            if bool(np.all(np.isfinite(low_arr))) and bool(np.all(np.isfinite(high_arr))):
                return low_arr, high_arr, False
            if action_dim is None:
                action_dim = int(low_arr.size)

    if action_dim is None or action_dim <= 0:
        raise ValueError("validation action space does not expose usable low/high bounds or a vector shape")

    low_fallback = np.full((action_dim,), -fallback_abs_bound, dtype=np.float32)
    high_fallback = np.full((action_dim,), fallback_abs_bound, dtype=np.float32)
    return low_fallback, high_fallback, True


def _resolve_policy_model_ref(model):
    queue = [model]
    seen = set()
    while queue:
        candidate = queue.pop(0)
        if candidate is None:
            continue
        ident = id(candidate)
        if ident in seen:
            continue
        seen.add(ident)
        if hasattr(candidate, "forward_policy_step"):
            return candidate
        for attr in ("module", "model", "_orig_mod"):
            nested = getattr(candidate, attr, None)
            if nested is not None and id(nested) not in seen:
                queue.append(nested)
    return model


def _model_supports_policy_step(model):
    model_ref = _resolve_policy_model_ref(model)
    return hasattr(model_ref, "forward_policy_step")


def _require_validation_policy_action_head(model):
    model_ref = _resolve_policy_model_ref(model)
    require_policy_action_head = getattr(model_ref, "require_policy_action_head", None)
    if callable(require_policy_action_head):
        require_policy_action_head()


def _resolve_validation_sample(value, rng):
    if isinstance(value, dict):
        dist = str(value.get("distribution", "")).strip().lower()
        if dist == "uniform":
            lo = float(value.get("min", 0.0))
            hi = float(value.get("max", lo))
            return float(rng.uniform(lo, hi))
        if dist == "log_uniform":
            lo = float(value.get("min", 1e-6))
            hi = float(value.get("max", lo))
            if lo <= 0.0 or hi <= 0.0:
                return float(max(lo, hi, 0.0))
            return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
        if dist in {"uniform_int", "randint"}:
            lo = float(value.get("min", 0.0))
            hi = float(value.get("max", lo))
            return int(round(rng.uniform(lo, hi)))
    return value


def _resolve_validation_policy_hparams(env_cfg, rng, *, orch_cfg=None):
    orch_cfg = {} if orch_cfg is None else orch_cfg
    # Real Gym validation should execute the model's raw environment action
    # (plus any stochastic exploration noise), not the synthetic-prior-only
    # RMS normalization that keeps generated environments numerically stable.
    action_transform_mode = str(orch_cfg.get("rl_validate_action_transform", "none")).strip().lower()
    if action_transform_mode not in {"none", "tanh", "rms"}:
        action_transform_mode = "none"
    return {
        # Match the prior semantics: 1 means reward is present, 0 means dropped/missing.
        "reward_mask_present_value": 1.0,
        "init_action_std": float(_resolve_validation_sample(env_cfg.get("init_action_std", 0.0), rng)),
        "action_noise_train_std": float(_resolve_validation_sample(env_cfg.get("action_noise_train_std", 0.0), rng)),
        "action_noise_eval_std": float(_resolve_validation_sample(env_cfg.get("action_noise_eval_std", 0.0), rng)),
        "action_transform_mode": action_transform_mode,
        "action_rms_eps": float(_resolve_validation_sample(env_cfg.get("reinforce_action_rms_eps", 1e-6), rng)),
        "reward_clip": float(max(0.1, _resolve_validation_sample(env_cfg.get("reward_clip", 10.0), rng))),
        "reward_transform_mode": str(env_cfg.get("reinforce_reward_transform", "none")).strip().lower(),
        "reward_transform_rms_eps": float(
            _resolve_validation_sample(env_cfg.get("reinforce_reward_rms_eps", 1e-6), rng)
        ),
        "reward_transform_tanh_c": float(
            _resolve_validation_sample(env_cfg.get("reinforce_reward_tanh_c", 1.0), rng)
        ),
        "reward_transform_tanh_bound": float(
            _resolve_validation_sample(env_cfg.get("reinforce_reward_tanh_bound", 10.0), rng)
        ),
    }


def _resolve_validation_policy_cache_config(optimizer_cfg, *, context_lower_bound, max_steps):
    requested_mode = "auto" if optimizer_cfg.get("pg_kv_cache_mode", "auto") is None else str(
        optimizer_cfg.get("pg_kv_cache_mode", "auto")
    ).strip().lower()
    if requested_mode not in {"auto", "immutable", "static", "paged"}:
        requested_mode = "auto"

    max_cache_len = None
    if requested_mode in {"auto", "static", "paged"}:
        # Validation keeps a single KV history across explore rollouts and the
        # final exploit rollout. The exploit switch can happen only after an
        # explore rollout finishes, so the total appended history is bounded by:
        #   context_before_eval <= context_lower_bound + max_steps
        #   final_total_len <= context_before_eval + max_steps
        #                  <= context_lower_bound + 2 * max_steps
        max_cache_len = max(1, int(max(0, context_lower_bound)) + 2 * int(max(1, max_steps)))

    return {
        "kv_cache_mode": requested_mode,
        "kv_cache_page_size": optimizer_cfg.get("pg_kv_cache_page_size", None),
        "max_cache_len": max_cache_len,
    }


def _initialize_validation_policy_state(state, *, env_cfg, orch_cfg, device, action_dim):
    rng = state["rng"]
    h = _resolve_validation_policy_hparams(env_cfg, rng, orch_cfg=orch_cfg)
    init_action_std = float(h["init_action_std"])
    if init_action_std > 0.0:
        init_action = rng.normal(size=(1, int(action_dim))).astype(np.float32) * init_action_std
    else:
        init_action = np.zeros((1, int(action_dim)), dtype=np.float32)
    state["policy_hparams"] = h
    state["policy_action_t"] = torch.from_numpy(init_action).to(device=device, dtype=torch.float32)
    state["policy_reward_t"] = torch.zeros((1, 1), device=device, dtype=torch.float32)
    state["policy_reward_mask_t"] = torch.full(
        (1, 1),
        float(h["reward_mask_present_value"]),
        device=device,
        dtype=torch.float32,
    )
    state["policy_terminal_t"] = torch.zeros((1, 1), device=device, dtype=torch.float32)
    state["policy_cache"] = None
    state["policy_action_dim"] = int(action_dim)


def _transform_validation_reward(reward_raw, *, policy_hparams, device):
    from ticl.priors.environment_prior import EnvironmentPrior

    reward_t = torch.as_tensor([float(reward_raw)], device=device, dtype=torch.float32)
    reward_t = torch.clamp(
        reward_t,
        min=-float(policy_hparams["reward_clip"]),
        max=float(policy_hparams["reward_clip"]),
    )
    reward_t = EnvironmentPrior._transform_reward_with_params(
        reward_t,
        mode=policy_hparams["reward_transform_mode"],
        rms_eps=policy_hparams["reward_transform_rms_eps"],
        tanh_c=policy_hparams["reward_transform_tanh_c"],
        tanh_bound=policy_hparams["reward_transform_tanh_bound"],
    )
    return float(reward_t.reshape(()).detach().cpu().item())


def _extract_validation_reward_term_scalars(info):
    if not isinstance(info, dict):
        return {}
    reward_terms = {}
    for key, value in info.items():
        key_str = str(key).strip()
        key_lower = key_str.lower()
        if not key_str:
            continue
        if not (
            key_lower.startswith("reward_")
            or key_lower.endswith("_reward")
            or key_lower.startswith("cost_")
            or key_lower.endswith("_cost")
        ):
            continue
        if key_lower == "reward":
            continue
        try:
            value_arr = np.asarray(value, dtype=np.float64).reshape(-1)
        except Exception:
            continue
        if int(value_arr.size) != 1:
            continue
        value_scalar = float(value_arr[0])
        if not np.isfinite(value_scalar):
            continue
        reward_terms[key_str] = value_scalar
    return reward_terms


def _accumulate_validation_reward_terms(state, info):
    reward_terms = _extract_validation_reward_term_scalars(info)
    if not reward_terms:
        return
    rollout_terms = state.setdefault("current_rollout_reward_terms", {})
    for key, value in reward_terms.items():
        rollout_terms[key] = float(rollout_terms.get(key, 0.0)) + float(value)


def _select_policy_action(
    *,
    state,
    policy_step_fn,
    device,
    obs_slot_dim,
    action_slot_dim,
    terminal_token_enabled,
):
    from ticl.priors.environment_prior import EnvironmentPrior

    obs_t = torch.from_numpy(_to_1d_array(state["obs"])).to(device=device, dtype=torch.float32).reshape(1, -1)
    env_info = {
        "obs_slot_dim": int(obs_slot_dim),
        "action_slot_dim": int(action_slot_dim),
        "action_dim": int(state["policy_action_dim"]),
        "phase_t": torch.full(
            (1, 1),
            float(state["phase_flag"]),
            device=device,
            dtype=torch.float32,
        ),
    }
    if bool(terminal_token_enabled):
        env_info["terminal_t"] = state["policy_terminal_t"]
    with torch.no_grad():
        policy_out = policy_step_fn(
            obs_t,
            state["policy_action_t"],
            state["policy_reward_t"],
            state["policy_reward_mask_t"],
            state["policy_cache"],
            int(state["context_len"] + state["current_rollout_len"]),
            env_info,
        )
    if isinstance(policy_out, tuple):
        action_mean, cache_next = policy_out
    else:
        action_mean = policy_out
        cache_next = state["policy_cache"]
    if action_mean.ndim == 1:
        action_mean = action_mean.reshape(1, -1)

    policy_hparams = state["policy_hparams"]
    action_std = (
        float(policy_hparams["action_noise_eval_std"])
        if float(state["phase_flag"]) >= 1.0
        else float(policy_hparams["action_noise_train_std"])
    )
    if action_std > 0.0:
        action_eps = state["rng"].normal(size=tuple(action_mean.shape)).astype(np.float32)
        action_pre_tanh = action_mean + torch.from_numpy(action_eps).to(device=device, dtype=action_mean.dtype) * action_std
    else:
        action_pre_tanh = action_mean
    action_env = EnvironmentPrior._transform_reinforce_action(
        action_pre_tanh,
        mode=policy_hparams["action_transform_mode"],
        rms_eps=policy_hparams["action_rms_eps"],
    )
    action_np = action_env[0].detach().to(dtype=torch.float32).cpu().numpy()
    action_np = np.clip(action_np, state["action_low"], state["action_high"]).astype(np.float32)
    state["policy_cache"] = cache_next
    return action_np


def _reset_validation_rollout(state, preserve_prev=False):
    obs, _ = state["env"].reset(seed=int(state["next_reset_seed"]))
    state["next_reset_seed"] += 1
    state["obs"] = obs
    state["current_rollout_reward_terms"] = {}
    if not bool(preserve_prev):
        state["prev_reward"] = 0.0
        state["prev_terminal"] = 0.0
        if "policy_hparams" in state:
            state["policy_reward_t"] = torch.zeros_like(state["policy_reward_t"])
            state["policy_reward_mask_t"] = torch.full_like(
                state["policy_reward_mask_t"],
                float(state["policy_hparams"]["reward_mask_present_value"]),
            )
            state["policy_terminal_t"] = torch.zeros_like(state["policy_terminal_t"])
    state["current_rollout_return"] = 0.0
    state["current_rollout_len"] = 0


def _finalize_validation_rollout(state, context_lower_bound):
    rollout_len = int(state["current_rollout_len"])
    rollout_return = float(state["current_rollout_return"])
    state["context_len"] += rollout_len

    if bool(state["phase_flag"] >= 1.0):
        state["reported_return"] = rollout_return
        state["reported_len"] = float(rollout_len)
        state["reported_reward_terms"] = dict(state.get("current_rollout_reward_terms", {}))
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
    env_cfg=None,
    orch_cfg=None,
    device="cpu",
    action_low=None,
    action_high=None,
    policy_step_enabled=False,
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
            "reported_reward_terms": None,
            "context_len_before_eval": None,
            "explore_rollout_count": None,
            "explore_rollout_len_mean": None,
            "next_reset_seed": reset_seed,
        }
        if action_low is not None and action_high is not None:
            state["action_low"] = np.asarray(action_low, dtype=np.float32).reshape(-1)
            state["action_high"] = np.asarray(action_high, dtype=np.float32).reshape(-1)
        if bool(policy_step_enabled):
            _initialize_validation_policy_state(
                state,
                env_cfg=(env_cfg or {}),
                orch_cfg=(orch_cfg or {}),
                device=device,
                action_dim=int(state["action_low"].shape[0]),
            )
        _reset_validation_rollout(state)
        episode_states.append(state)
    return episode_states


def evaluate_rlpfn_on_gym_envs(model, config):
    try:
        import gymnasium as gym
    except Exception as exc:
        return float("nan"), {"__error__": f"gymnasium_import_failed: {exc}"}

    orch = config.get("orchestration", {})
    env_names = parse_env_list(orch.get("rl_validate_envs", None))
    episodes = int(orch.get("rl_validate_episodes", 3))
    max_steps = int(orch.get("rl_validate_max_steps", 1000))
    n_candidates = int(orch.get("rl_validate_action_candidates", 16))
    base_seed = int(orch.get("rl_validate_seed", 1))
    context_lower_bound = int(orch.get("rl_validate_context_lower_bound", 2048))
    max_parallel_columns = int(orch.get("rl_validate_max_parallel_columns", 96))
    if max_parallel_columns > 0:
        max_parallel_columns = max(int(n_candidates), max_parallel_columns)

    env_cfg = config.get("prior", {}).get("environment", {})
    layout = resolve_rlpfn_token_layout(
        env_cfg,
        num_features=config.get("prior", {}).get("num_features", None),
    )
    obs_slot_dim = int(layout["obs_slot_dim"])
    action_slot_dim = int(layout["action_slot_dim"])
    terminal_token_enabled = bool(layout["terminal_token_enabled"])
    num_features = int(layout["num_features"])
    device = config.get("device", "cpu")
    optimizer_cfg = config.get("optimizer", {})
    policy_step_fn = None
    use_policy_step_validation = bool(_model_supports_policy_step(model))
    if bool(use_policy_step_validation):
        _require_validation_policy_action_head(model)
        from ticl.train import _build_policy_step_fn
        cache_cfg = _resolve_validation_policy_cache_config(
            optimizer_cfg,
            context_lower_bound=context_lower_bound,
            max_steps=max_steps,
        )

        policy_step_fn = _build_policy_step_fn(
            model,
            num_features=num_features,
            max_cache_len=cache_cfg["max_cache_len"],
            kv_cache_mode=cache_cfg["kv_cache_mode"],
            kv_cache_page_size=cache_cfg["kv_cache_page_size"],
            allow_grad_mutable_cache=False,
            allow_grad_inplace_paged_cache=False,
            pg_torch_compile=False,
        )

    was_training = model.training
    model.eval()

    per_env = {}
    all_env_means = []
    env_states = {}
    env_action_bounds = {}

    try:
        for env_name in env_names:
            try:
                probe_env = gym.make(env_name)
            except Exception:
                per_env[env_name] = {"return_mean": float("nan"), "len_mean": float("nan"), "make_failed": 1}
                continue

            try:
                action_low, action_high, used_fallback_bounds = _resolve_validation_action_bounds(
                    probe_env.action_space
                )
            except Exception:
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

            try:
                probe_env.close()
            except Exception:
                pass
            states = _build_validation_episode_states(
                gym=gym,
                env_name=env_name,
                episodes=episodes,
                base_seed=base_seed,
                env_cfg=env_cfg,
                orch_cfg=orch,
                device=device,
                action_low=action_low,
                action_high=action_high,
                policy_step_enabled=use_policy_step_validation,
            )
            env_states[env_name] = states
            env_action_bounds[env_name] = (action_low, action_high)
            if bool(used_fallback_bounds):
                per_env.setdefault(env_name, {})["fallback_action_bounds"] = 1

        all_states = [state for states in env_states.values() for state in states]

        try:
            while any(not state["done"] for state in all_states):
                if bool(use_policy_step_validation):
                    for state in all_states:
                        if bool(state["done"]):
                            continue
                        if int(state["current_rollout_len"]) >= int(max_steps):
                            _finalize_validation_rollout(state, context_lower_bound)
                            continue
                        action = _select_policy_action(
                            state=state,
                            policy_step_fn=policy_step_fn,
                            device=device,
                            obs_slot_dim=obs_slot_dim,
                            action_slot_dim=action_slot_dim,
                            terminal_token_enabled=terminal_token_enabled,
                        )
                        obs_next, reward_next, terminated, truncated, info = state["env"].step(action.astype(np.float32))
                        reward_next = float(reward_next)
                        done_flag = bool(terminated or truncated)
                        _accumulate_validation_reward_terms(state, info)
                        reward_input = _transform_validation_reward(
                            reward_next,
                            policy_hparams=state["policy_hparams"],
                            device=device,
                        )
                        state["obs"] = obs_next
                        state["prev_reward"] = reward_next
                        state["prev_terminal"] = 1.0 if done_flag else 0.0
                        state["policy_action_t"] = torch.from_numpy(action.reshape(1, -1)).to(
                            device=device,
                            dtype=torch.float32,
                        )
                        state["policy_reward_t"] = torch.full(
                            (1, 1),
                            float(reward_input),
                            device=device,
                            dtype=torch.float32,
                        )
                        state["policy_reward_mask_t"] = torch.full(
                            (1, 1),
                            float(state["policy_hparams"]["reward_mask_present_value"]),
                            device=device,
                            dtype=torch.float32,
                        )
                        state["policy_terminal_t"] = torch.full(
                            (1, 1),
                            1.0 if done_flag else 0.0,
                            device=device,
                            dtype=torch.float32,
                        )
                        state["current_rollout_return"] += reward_next
                        state["current_rollout_len"] += 1

                        if done_flag or int(state["current_rollout_len"]) >= int(max_steps):
                            if int(state["current_rollout_len"]) >= int(max_steps):
                                state["prev_terminal"] = 1.0
                                state["policy_terminal_t"] = torch.ones_like(state["policy_terminal_t"])
                            _finalize_validation_rollout(state, context_lower_bound)
                else:
                    jobs = []
                    job_states = []
                    for env_name in env_names:
                        states = env_states.get(env_name, None)
                        if not isinstance(states, list):
                            continue
                        action_bounds = env_action_bounds.get(env_name, None)
                        if action_bounds is None:
                            continue
                        action_low, action_high = action_bounds
                        for state in states:
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
                            job_states.append((state, candidates))

                    if jobs:
                        score_list = _score_candidate_action_jobs(
                            model=model,
                            device=device,
                            jobs=jobs,
                            obs_slot_dim=obs_slot_dim,
                            action_slot_dim=action_slot_dim,
                            num_features=num_features,
                            terminal_token_enabled=terminal_token_enabled,
                            max_parallel_columns=max_parallel_columns,
                        )
                    else:
                        score_list = []

                    for (state, candidates), scores in zip(job_states, score_list):
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
                        obs_next, reward_next, terminated, truncated, info = state["env"].step(action.astype(np.float32))
                        reward_next = float(reward_next)
                        done_flag = bool(terminated or truncated)
                        _accumulate_validation_reward_terms(state, info)
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

            for env_name in env_names:
                states = env_states.get(env_name, None)
                if not isinstance(states, list):
                    continue
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
                reward_term_values = {}
                for state in states:
                    reported_reward_terms = state.get("reported_reward_terms", None)
                    if not isinstance(reported_reward_terms, dict):
                        continue
                    for key, value in reported_reward_terms.items():
                        reward_term_values.setdefault(str(key), []).append(float(value))
                if returns:
                    mean_ret = float(np.mean(returns))
                    mean_len = float(np.mean(lengths))
                    existing_summary = dict(per_env.get(env_name, {}))
                    env_summary = {
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
                    if int(len(returns)) == 1:
                        env_summary["return"] = float(returns[0])
                        env_summary["len"] = float(lengths[0]) if lengths else float("nan")
                    for key, values in reward_term_values.items():
                        env_summary[f"{key}_mean"] = float(np.mean(np.asarray(values, dtype=np.float64)))
                        if int(len(values)) == 1:
                            env_summary[key] = float(values[0])
                    existing_summary.update(env_summary)
                    per_env[env_name] = existing_summary
                    all_env_means.append(mean_ret)
        finally:
            for state in all_states:
                try:
                    state["env"].close()
                except Exception:
                    pass
    finally:
        if was_training:
            model.train()
        else:
            model.eval()

    global_mean = float(np.mean(all_env_means)) if all_env_means else float("nan")
    return global_mean, per_env
