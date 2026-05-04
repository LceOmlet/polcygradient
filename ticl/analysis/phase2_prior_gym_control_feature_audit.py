import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from ticl.analysis.phase2_alpha_temporal_control_probe import measure_alpha_temporal_control
from ticl.analysis.phase2_gym_horizon_state_identity_probe import (
    FixedRandomLinearGaussianPolicy,
    _capture_mujoco_state,
    _make_env,
    _restore_mujoco_state,
)


DEFAULT_PRIOR_ROWS = (
    "/home/chen/RLPFN/artifacts/"
    "phase2_exact_scm_topology_identity_analysis_horizon2to5_seed30000_count256_initstate1/"
    "rows_with_topology.jsonl"
)
DEFAULT_ANT_CLEAN_JSON = (
    "/home/chen/RLPFN/artifacts/"
    "gym_ant_horizon_identity_64x64_unit_gaussian_policy_h2345/"
    "Ant-v5_clean_obs_h2345_unit_gaussian.json"
)
DEFAULT_ANT_FULL_JSON = (
    "/home/chen/RLPFN/artifacts/"
    "gym_ant_horizon_identity_64x64_unit_gaussian_policy_h2345/"
    "Ant-v5_h2345_unit_gaussian.json"
)


CORE_FEATURE_CONTRACT: dict[str, Any] = {
    "audit_goal": (
        "Pretraining uses one rollout per sampled environment, so online same-env collapse sentinels do not protect "
        "the formal training distribution. The prior must be audited offline for Gym-like scale, controllability, "
        "state-return readability, and closed-loop necessity before it is used as a training distribution."
    ),
    "feature_groups": {
        "input_scale": [
            "anchor_obs_value_std",
            "anchor_obs_absmax",
            "anchor_reward_std",
            "anchor_reward_q10/q50/q90",
        ],
        "target_scale": [
            "all_return_std",
            "state_return_std_mean",
            "state_return_mean_std",
        ],
        "state_return_readability": [
            "horizon_2..5_bias_corrected_identity_score",
            "topology_reward_state_input_gain_fraction",
        ],
        "action_controllability": [
            "mean_l2_s_t1_action_sensitivity_per_action_unit",
            "state_action_sensitivity_to_step1_drift_ratio",
            "mean_abs_reward_effective_t1_action_sensitivity_per_action_unit",
            "topology_reward_action_input_gain_fraction",
        ],
        "noise_dominance": [
            "topology_reward_noise_input_gain_fraction",
            "topology_reward_state_to_noise_gain_ratio",
        ],
        "open_loop_risk": [
            "constant-action return range versus random/state-conditioned policy return range",
            "deterministic policy mean time variation after one-rollout PPO update",
        ],
    },
    "current_priority": (
        "The missing hard evidence is action controllability/open-loop risk, not more same-env PPO training. "
        "A gain>=0.7 rule can make rewards state-readable while still leaving action influence too weak or too indirect."
    ),
}


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return None
        return value
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, np.ndarray):
        return [_json_safe(v) for v in value.tolist()]
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return str(value)


def _finite_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).expanduser().open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).expanduser().open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _stats(values: list[Any]) -> dict[str, Any]:
    arr = np.asarray([v for v in (_finite_float(v) for v in values) if v is not None], dtype=np.float64)
    if arr.size == 0:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "min": None,
            "q10": None,
            "q25": None,
            "q50": None,
            "q75": None,
            "q90": None,
            "max": None,
        }
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "q10": float(np.quantile(arr, 0.10)),
        "q25": float(np.quantile(arr, 0.25)),
        "q50": float(np.quantile(arr, 0.50)),
        "q75": float(np.quantile(arr, 0.75)),
        "q90": float(np.quantile(arr, 0.90)),
        "max": float(np.max(arr)),
    }


def _pearson(rows: list[dict[str, Any]], x_key: str, y_key: str) -> float | None:
    xs = []
    ys = []
    for row in rows:
        x = _finite_float(row.get(x_key))
        y = _finite_float(row.get(y_key))
        if x is not None and y is not None:
            xs.append(x)
            ys.append(y)
    if len(xs) < 3:
        return None
    x_arr = np.asarray(xs, dtype=np.float64)
    y_arr = np.asarray(ys, dtype=np.float64)
    x0 = x_arr - float(np.mean(x_arr))
    y0 = y_arr - float(np.mean(y_arr))
    denom = float(np.linalg.norm(x0) * np.linalg.norm(y0))
    if denom <= 1e-12:
        return None
    return float(np.dot(x0, y0) / denom)


def _horizon_curve_from_prior_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        f"G{h}_bias_corrected_identity": _stats(
            [row.get(f"horizon_{h}_bias_corrected_identity_score") for row in rows]
        )
        for h in (2, 3, 4, 5)
    }


def _horizon_curve_from_ant_report(report: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for diag in report.get("horizon_diagnostics", []):
        h = int(diag["horizon_step"])
        out[f"G{h}_bias_corrected_identity"] = float(diag["bias_corrected_identity_score"])
        out[f"G{h}_signal_to_noise"] = float(diag["signal_to_noise"])
    return out


def _extract_ant_scale_summary(report: dict[str, Any]) -> dict[str, Any]:
    anchors = list(report.get("anchors", []))
    return {
        "env_id": str(report.get("env_id")),
        "obs_dim": int(report.get("obs_dim")),
        "action_dim": int(report.get("action_dim")),
        "aggregate_identity_score": float(report["aggregate"]["identity_score"]),
        "aggregate_mean_within_state_std": float(report["aggregate"]["mean_within_state_std"]),
        "aggregate_std_of_state_means": float(report["aggregate"]["std_of_state_means"]),
        "anchor_reward": _stats([row.get("anchor_reward") for row in anchors]),
        "continuation_return_std": _stats([row.get("continuation_return_std") for row in anchors]),
        "horizon_curve": _horizon_curve_from_ant_report(report),
    }


def summarize_existing_prior(rows: list[dict[str, Any]]) -> dict[str, Any]:
    feature_keys = [
        "topology_reward_state_input_gain_fraction",
        "topology_reward_action_input_gain_fraction",
        "topology_reward_noise_input_gain_fraction",
        "topology_reward_state_to_noise_gain_ratio",
        "topology_reward_state_to_action_gain_ratio",
        "anchor_obs_value_std",
        "anchor_obs_absmax",
        "anchor_reward_std",
        "all_return_std",
        "state_return_std_mean",
        "state_return_mean_std",
        "anchor_state_cov_effective_rank",
        "anchor_state_cov_top_eigen_share",
    ]
    correlations = {}
    for key in feature_keys:
        correlations[f"{key}__vs_G2_identity"] = _pearson(rows, key, "horizon_2_bias_corrected_identity_score")
        correlations[f"{key}__vs_G5_identity"] = _pearson(rows, key, "horizon_5_bias_corrected_identity_score")
    gain07 = [row for row in rows if (_finite_float(row.get("topology_reward_state_input_gain_fraction")) or -1.0) >= 0.7]
    return {
        "row_count": int(len(rows)),
        "all_prior": {key: _stats([row.get(key) for row in rows]) for key in feature_keys},
        "gain_ge_0p7_prior": {key: _stats([row.get(key) for row in gain07]) for key in feature_keys},
        "gain_ge_0p7_count": int(len(gain07)),
        "horizon_identity_curve_all_prior": _horizon_curve_from_prior_rows(rows),
        "horizon_identity_curve_gain_ge_0p7": _horizon_curve_from_prior_rows(gain07),
        "correlations": correlations,
    }


def _sample_prior_rows(
    rows: list[dict[str, Any]],
    *,
    count: int,
    seed: int,
    min_gain: float | None,
) -> list[dict[str, Any]]:
    candidates = []
    for row in rows:
        if min_gain is not None:
            gain = _finite_float(row.get("topology_reward_state_input_gain_fraction"))
            if gain is None or gain < float(min_gain):
                continue
        candidates.append(row)
    if int(count) <= 0 or int(count) >= len(candidates):
        return list(candidates)
    rng = np.random.default_rng(int(seed))
    indices = sorted(rng.choice(len(candidates), size=int(count), replace=False).tolist())
    return [candidates[int(i)] for i in indices]


def _parse_env_id_list(raw: str, fallback: str) -> list[str]:
    env_ids = [piece.strip() for piece in str(raw or "").split(",") if piece.strip()]
    if not env_ids:
        env_ids = [str(fallback)]
    out: list[str] = []
    seen = set()
    for env_id in env_ids:
        if env_id in seen:
            continue
        seen.add(env_id)
        out.append(env_id)
    return out


def run_prior_control_measurements(
    rows: list[dict[str, Any]],
    *,
    count: int,
    seed: int,
    min_gain: float | None,
    device: str,
    rollout_count: int,
    n_steps: int,
    action_delta: float,
    measure_open_loop: bool,
) -> list[dict[str, Any]]:
    measured = []
    for idx, row in enumerate(_sample_prior_rows(rows, count=count, seed=seed, min_gain=min_gain), start=1):
        frozen_path = str(row["full_frozen_h_json"])
        print(
            f"[prior-control-audit] prior finite-diff {idx}/{int(count) if int(count) > 0 else '?'} "
            f"seed={row.get('frozen_h_seed')} gain={row.get('topology_reward_state_input_gain_fraction')}",
            flush=True,
        )
        report = measure_alpha_temporal_control(
            checkpoint_path=None,
            device=str(device),
            from_scratch=True,
            from_scratch_model_type="rlpfn",
            build_seed=4040,
            train_env_seed=2020,
            single_eval_pos=64,
            n_steps=int(n_steps),
            rollout_count=int(rollout_count),
            rollout_seed_start=700000,
            sampled_policy=True,
            fixed_frozen_h_json=frozen_path,
            alpha=float(row.get("alpha", 1.0)),
            reference_state_inertia_enabled_override=None,
            terminal_reset_enabled_override=None,
            terminal_reset_count_target_override=None,
            action_delta=float(action_delta),
            behavior_policy="unit_gaussian_iid",
        )
        open_loop_reports = {"unit_gaussian_iid": report}
        if bool(measure_open_loop):
            for policy_mode in ("zero", "constant_gaussian"):
                open_loop_reports[policy_mode] = measure_alpha_temporal_control(
                    checkpoint_path=None,
                    device=str(device),
                    from_scratch=True,
                    from_scratch_model_type="rlpfn",
                    build_seed=4040,
                    train_env_seed=2020,
                    single_eval_pos=64,
                    n_steps=int(n_steps),
                    rollout_count=int(rollout_count),
                    rollout_seed_start=700000,
                    sampled_policy=(policy_mode == "unit_gaussian_iid"),
                    fixed_frozen_h_json=frozen_path,
                    alpha=float(row.get("alpha", 1.0)),
                    reference_state_inertia_enabled_override=None,
                    terminal_reset_enabled_override=None,
                    terminal_reset_count_target_override=None,
                    action_delta=float(action_delta),
                    behavior_policy=policy_mode,
                )
        merged = {
            "frozen_h_seed": row.get("frozen_h_seed"),
            "full_frozen_h_json": frozen_path,
            "topology_reward_state_input_gain_fraction": row.get("topology_reward_state_input_gain_fraction"),
            "topology_reward_action_input_gain_fraction": row.get("topology_reward_action_input_gain_fraction"),
            "topology_reward_noise_input_gain_fraction": row.get("topology_reward_noise_input_gain_fraction"),
            "anchor_obs_value_std": row.get("anchor_obs_value_std"),
            "anchor_reward_std": row.get("anchor_reward_std"),
            "all_return_std": row.get("all_return_std"),
            "horizon_2_bias_corrected_identity_score": row.get("horizon_2_bias_corrected_identity_score"),
            "horizon_5_bias_corrected_identity_score": row.get("horizon_5_bias_corrected_identity_score"),
        }
        for key in (
            "mean_l2_s_t1_minus_s_t",
            "mean_l2_s_t4_minus_s_t",
            "mean_l2_s_t1_action_sensitivity",
            "mean_l2_s_t1_action_sensitivity_per_action_unit",
            "state_action_sensitivity_to_step1_drift_ratio",
            "mean_abs_reward_raw_t1_action_sensitivity",
            "mean_abs_reward_raw_t1_action_sensitivity_per_action_unit",
            "mean_abs_reward_effective_t1_action_sensitivity",
            "mean_abs_reward_effective_t1_action_sensitivity_per_action_unit",
        ):
            merged[key] = report.get(key)
        for policy_mode, policy_report in open_loop_reports.items():
            policy_prefix = f"{policy_mode}_policy"
            ret_stats = dict(policy_report.get("discounted_effective_return", {}))
            raw_ret_stats = dict(policy_report.get("discounted_raw_return", {}))
            for stat_key in ("mean", "std", "q10", "q50", "q90", "min", "max"):
                merged[f"{policy_prefix}_discounted_effective_return_{stat_key}"] = ret_stats.get(stat_key)
                merged[f"{policy_prefix}_discounted_raw_return_{stat_key}"] = raw_ret_stats.get(stat_key)
        if bool(measure_open_loop):
            iid_mean = _finite_float(merged.get("unit_gaussian_iid_policy_discounted_effective_return_mean"))
            iid_q90 = _finite_float(merged.get("unit_gaussian_iid_policy_discounted_effective_return_q90"))
            const_mean = _finite_float(merged.get("constant_gaussian_policy_discounted_effective_return_mean"))
            const_q90 = _finite_float(merged.get("constant_gaussian_policy_discounted_effective_return_q90"))
            zero_mean = _finite_float(merged.get("zero_policy_discounted_effective_return_mean"))
            if iid_mean is not None and const_mean is not None:
                merged["open_loop_constant_minus_iid_return_mean"] = float(const_mean - iid_mean)
            if iid_q90 is not None and const_q90 is not None:
                merged["open_loop_constant_minus_iid_return_q90"] = float(const_q90 - iid_q90)
            if iid_mean is not None and zero_mean is not None:
                merged["open_loop_zero_minus_iid_return_mean"] = float(zero_mean - iid_mean)
        measured.append(merged)
    return measured


def run_ant_control_measurements(
    *,
    env_id: str,
    reference_state_count: int,
    anchor_step: int,
    policy_seed: int,
    reference_seed_start: int,
    action_delta: float,
    open_loop_horizon: int,
    open_loop_repeats: int,
    discount_gamma: float,
) -> dict[str, Any]:
    env = _make_env(str(env_id))
    try:
        obs_dim = int(np.prod(env.observation_space.shape))
        action_dim = int(np.prod(env.action_space.shape))
        policy = FixedRandomLinearGaussianPolicy(
            obs_dim=obs_dim,
            action_dim=action_dim,
            seed=int(policy_seed),
            sigma=0.25,
            policy_type="unit_gaussian",
        )
        action_low = np.asarray(env.action_space.low, dtype=np.float64)
        action_high = np.asarray(env.action_space.high, dtype=np.float64)
        obs_drift = []
        obs_sens_gap = []
        obs_sens_per_unit = []
        reward_sens_gap = []
        reward_sens_per_unit = []
        anchor_reward = []
        anchor_obs_values = []
        iid_returns = []
        constant_returns = []
        zero_returns = []
        for idx in range(int(reference_state_count)):
            seed = int(reference_seed_start) + int(idx)
            obs, _ = env.reset(seed=seed)
            obs = np.asarray(obs, dtype=np.float64)
            rng = np.random.default_rng(seed)
            for _ in range(int(anchor_step)):
                action = policy.action(obs, rng, action_low, action_high)
                obs, reward, terminated, truncated, _ = env.step(action)
                obs = np.asarray(obs, dtype=np.float64)
                if bool(terminated) or bool(truncated):
                    obs, _ = env.reset(seed=seed + 1)
                    obs = np.asarray(obs, dtype=np.float64)
            state = _capture_mujoco_state(env)
            anchor_obs = np.asarray(obs, dtype=np.float64)
            base_action = policy.action(anchor_obs, rng, action_low, action_high).astype(np.float64)
            direction = rng.standard_normal((action_dim,)).astype(np.float64)
            direction = direction / max(float(np.linalg.norm(direction)), 1e-12)
            action_plus = np.clip(base_action + float(action_delta) * direction, action_low, action_high)
            action_minus = np.clip(base_action - float(action_delta) * direction, action_low, action_high)

            obs0 = _restore_mujoco_state(env, state)
            obs_next, reward_base, _, _, _ = env.step(base_action.astype(np.float32))
            obs_next = np.asarray(obs_next, dtype=np.float64)
            obs_drift.append(float(np.linalg.norm(obs_next - obs0)))
            anchor_reward.append(float(reward_base))
            anchor_obs_values.extend(anchor_obs.tolist())

            _restore_mujoco_state(env, state)
            obs_plus, reward_plus, _, _, _ = env.step(action_plus.astype(np.float32))
            _restore_mujoco_state(env, state)
            obs_minus, reward_minus, _, _, _ = env.step(action_minus.astype(np.float32))
            obs_gap = float(np.linalg.norm(np.asarray(obs_plus, dtype=np.float64) - np.asarray(obs_minus, dtype=np.float64)))
            rew_gap = float(abs(float(reward_plus) - float(reward_minus)))
            obs_sens_gap.append(obs_gap)
            obs_sens_per_unit.append(obs_gap / max(2.0 * float(action_delta), 1e-12))
            reward_sens_gap.append(rew_gap)
            reward_sens_per_unit.append(rew_gap / max(2.0 * float(action_delta), 1e-12))
            for repeat_idx in range(int(open_loop_repeats)):
                open_loop_seed = int(9_000_000) + int(reference_seed_start) + int(idx) * int(open_loop_repeats) + int(repeat_idx)
                iid_returns.append(
                    _roll_ant_open_loop_return(
                        env=env,
                        anchor_state=state,
                        policy_mode="unit_gaussian_iid",
                        seed=open_loop_seed,
                        horizon=int(open_loop_horizon),
                        gamma=float(discount_gamma),
                        action_dim=action_dim,
                        action_low=action_low,
                        action_high=action_high,
                    )
                )
                constant_returns.append(
                    _roll_ant_open_loop_return(
                        env=env,
                        anchor_state=state,
                        policy_mode="constant_gaussian",
                        seed=open_loop_seed,
                        horizon=int(open_loop_horizon),
                        gamma=float(discount_gamma),
                        action_dim=action_dim,
                        action_low=action_low,
                        action_high=action_high,
                    )
                )
                zero_returns.append(
                    _roll_ant_open_loop_return(
                        env=env,
                        anchor_state=state,
                        policy_mode="zero",
                        seed=open_loop_seed,
                        horizon=int(open_loop_horizon),
                        gamma=float(discount_gamma),
                        action_dim=action_dim,
                        action_low=action_low,
                        action_high=action_high,
                    )
                )
        mean_drift = float(np.mean(obs_drift)) if obs_drift else float("nan")
        iid_stats = _stats(iid_returns)
        constant_stats = _stats(constant_returns)
        zero_stats = _stats(zero_returns)
        iid_mean = _finite_float(iid_stats.get("mean"))
        iid_q90 = _finite_float(iid_stats.get("q90"))
        constant_mean = _finite_float(constant_stats.get("mean"))
        constant_q90 = _finite_float(constant_stats.get("q90"))
        zero_mean = _finite_float(zero_stats.get("mean"))
        return {
            "env_id": str(env_id),
            "reference_state_count": int(reference_state_count),
            "anchor_step": int(anchor_step),
            "policy_type": "unit_gaussian",
            "action_delta": float(action_delta),
            "obs_dim": int(obs_dim),
            "action_dim": int(action_dim),
            "anchor_obs_value_std": float(np.std(np.asarray(anchor_obs_values, dtype=np.float64))),
            "anchor_reward": _stats(anchor_reward),
            "mean_l2_obs_t1_minus_obs_t": float(mean_drift),
            "mean_l2_obs_t1_action_sensitivity": float(np.mean(obs_sens_gap)) if obs_sens_gap else None,
            "mean_l2_obs_t1_action_sensitivity_per_action_unit": (
                float(np.mean(obs_sens_per_unit)) if obs_sens_per_unit else None
            ),
            "obs_action_sensitivity_to_step1_drift_ratio": (
                float(np.mean(obs_sens_gap) / max(mean_drift, 1e-12)) if obs_sens_gap else None
            ),
            "mean_abs_reward_t1_action_sensitivity": float(np.mean(reward_sens_gap)) if reward_sens_gap else None,
            "mean_abs_reward_t1_action_sensitivity_per_action_unit": (
                float(np.mean(reward_sens_per_unit)) if reward_sens_per_unit else None
            ),
            "open_loop_horizon": int(open_loop_horizon),
            "open_loop_repeats": int(open_loop_repeats),
            "open_loop_discount_gamma": float(discount_gamma),
            "unit_gaussian_iid_policy_discounted_effective_return": iid_stats,
            "constant_gaussian_policy_discounted_effective_return": constant_stats,
            "zero_policy_discounted_effective_return": zero_stats,
            "open_loop_constant_minus_iid_return_mean": (
                float(constant_mean - iid_mean) if constant_mean is not None and iid_mean is not None else None
            ),
            "open_loop_constant_minus_iid_return_q90": (
                float(constant_q90 - iid_q90) if constant_q90 is not None and iid_q90 is not None else None
            ),
            "open_loop_zero_minus_iid_return_mean": (
                float(zero_mean - iid_mean) if zero_mean is not None and iid_mean is not None else None
            ),
        }
    finally:
        env.close()


def _roll_ant_open_loop_return(
    *,
    env,
    anchor_state: dict[str, np.ndarray],
    policy_mode: str,
    seed: int,
    horizon: int,
    gamma: float,
    action_dim: int,
    action_low: np.ndarray,
    action_high: np.ndarray,
) -> float:
    obs = _restore_mujoco_state(env, anchor_state)
    del obs
    rng = np.random.default_rng(int(seed))
    constant_action = None
    if str(policy_mode) == "constant_gaussian":
        constant_action = np.clip(rng.standard_normal((int(action_dim),)), action_low, action_high).astype(np.float32)
    total = 0.0
    discount = 1.0
    for _ in range(int(horizon)):
        if str(policy_mode) == "unit_gaussian_iid":
            action = np.clip(rng.standard_normal((int(action_dim),)), action_low, action_high).astype(np.float32)
        elif str(policy_mode) == "zero":
            action = np.zeros((int(action_dim),), dtype=np.float32)
        elif str(policy_mode) == "constant_gaussian":
            action = constant_action
        else:
            raise ValueError(f"unknown ant open-loop policy_mode={policy_mode!r}")
        _, reward, terminated, truncated, _ = env.step(action)
        total += float(discount) * float(reward)
        discount *= float(gamma)
        if bool(terminated) or bool(truncated):
            break
    return float(total)


def summarize_prior_control(measured_rows: list[dict[str, Any]]) -> dict[str, Any]:
    keys = [
        "mean_l2_s_t1_minus_s_t",
        "mean_l2_s_t1_action_sensitivity",
        "mean_l2_s_t1_action_sensitivity_per_action_unit",
        "state_action_sensitivity_to_step1_drift_ratio",
        "mean_abs_reward_effective_t1_action_sensitivity",
        "mean_abs_reward_effective_t1_action_sensitivity_per_action_unit",
        "unit_gaussian_iid_policy_discounted_effective_return_mean",
        "unit_gaussian_iid_policy_discounted_effective_return_std",
        "constant_gaussian_policy_discounted_effective_return_mean",
        "constant_gaussian_policy_discounted_effective_return_std",
        "zero_policy_discounted_effective_return_mean",
        "open_loop_constant_minus_iid_return_mean",
        "open_loop_constant_minus_iid_return_q90",
        "open_loop_zero_minus_iid_return_mean",
        "topology_reward_state_input_gain_fraction",
        "topology_reward_action_input_gain_fraction",
        "topology_reward_noise_input_gain_fraction",
        "anchor_obs_value_std",
        "anchor_reward_std",
        "all_return_std",
        "horizon_2_bias_corrected_identity_score",
        "horizon_5_bias_corrected_identity_score",
    ]
    return {
        "row_count": int(len(measured_rows)),
        "stats": {key: _stats([row.get(key) for row in measured_rows]) for key in keys},
        "correlations": {
            f"{key}__vs_reward_action_sensitivity": _pearson(
                measured_rows,
                key,
                "mean_abs_reward_effective_t1_action_sensitivity_per_action_unit",
            )
            for key in keys
        },
        "rows": measured_rows,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Audit prior environment features against Gym Ant references, with optional finite-difference "
            "action/reward controllability measurements."
        )
    )
    parser.add_argument("--prior-rows-jsonl", type=str, default=DEFAULT_PRIOR_ROWS)
    parser.add_argument("--ant-clean-json", type=str, default=DEFAULT_ANT_CLEAN_JSON)
    parser.add_argument("--ant-full-json", type=str, default=DEFAULT_ANT_FULL_JSON)
    parser.add_argument("--output-json", type=str, required=True)
    parser.add_argument("--measure-prior-control-count", type=int, default=0)
    parser.add_argument("--prior-control-min-gain", type=float, default=0.7)
    parser.add_argument("--prior-control-sample-seed", type=int, default=20260428)
    parser.add_argument("--prior-control-device", type=str, default="cpu")
    parser.add_argument("--prior-control-rollout-count", type=int, default=128)
    parser.add_argument("--prior-control-n-steps", type=int, default=64)
    parser.add_argument("--measure-prior-open-loop", type=str, default="false")
    parser.add_argument("--action-delta", type=float, default=0.1)
    parser.add_argument("--measure-ant-control", type=str, default="true")
    parser.add_argument("--ant-control-env-id", type=str, default="Ant-v5_clean_obs")
    parser.add_argument(
        "--gym-control-env-ids",
        type=str,
        default="",
        help=(
            "Comma-separated Gym/MuJoCo env ids for control/open-loop reference probes. "
            "Empty keeps backward-compatible --ant-control-env-id behavior."
        ),
    )
    parser.add_argument("--ant-control-reference-state-count", type=int, default=128)
    parser.add_argument("--ant-control-anchor-step", type=int, default=150)
    parser.add_argument("--ant-control-policy-seed", type=int, default=20260422)
    parser.add_argument("--ant-control-reference-seed-start", type=int, default=5000)
    parser.add_argument("--ant-open-loop-horizon", type=int, default=32)
    parser.add_argument("--ant-open-loop-repeats", type=int, default=8)
    parser.add_argument("--open-loop-discount-gamma", type=float, default=0.99)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    prior_rows = _read_jsonl(args.prior_rows_jsonl)
    ant_clean = _read_json(args.ant_clean_json)
    ant_full = _read_json(args.ant_full_json)
    prior_control_rows: list[dict[str, Any]] = []
    if int(args.measure_prior_control_count) > 0:
        prior_control_rows = run_prior_control_measurements(
            prior_rows,
            count=int(args.measure_prior_control_count),
            seed=int(args.prior_control_sample_seed),
            min_gain=float(args.prior_control_min_gain) if args.prior_control_min_gain is not None else None,
            device=str(args.prior_control_device),
            rollout_count=int(args.prior_control_rollout_count),
            n_steps=int(args.prior_control_n_steps),
            action_delta=float(args.action_delta),
            measure_open_loop=str(args.measure_prior_open_loop).lower() in {"1", "true", "yes", "y"},
        )
    gym_control_env_ids = _parse_env_id_list(str(args.gym_control_env_ids), str(args.ant_control_env_id))
    ant_control = None
    gym_control_probes = {}
    if str(args.measure_ant_control).lower() in {"1", "true", "yes", "y"}:
        for env_idx, env_id in enumerate(gym_control_env_ids, start=1):
            print(
                f"[prior-control-audit] gym control {env_idx}/{len(gym_control_env_ids)} env={env_id}",
                flush=True,
            )
            probe = run_ant_control_measurements(
                env_id=str(env_id),
                reference_state_count=int(args.ant_control_reference_state_count),
                anchor_step=int(args.ant_control_anchor_step),
                policy_seed=int(args.ant_control_policy_seed),
                reference_seed_start=int(args.ant_control_reference_seed_start) + 100000 * (env_idx - 1),
                action_delta=float(args.action_delta),
                open_loop_horizon=int(args.ant_open_loop_horizon),
                open_loop_repeats=int(args.ant_open_loop_repeats),
                discount_gamma=float(args.open_loop_discount_gamma),
            )
            gym_control_probes[str(env_id)] = probe
            if ant_control is None:
                ant_control = probe
    report = {
        "audit_entry": "phase2_prior_gym_control_feature_audit",
        "core_feature_contract": CORE_FEATURE_CONTRACT,
        "inputs": {
            "prior_rows_jsonl": str(Path(args.prior_rows_jsonl).expanduser().resolve()),
            "ant_clean_json": str(Path(args.ant_clean_json).expanduser().resolve()),
            "ant_full_json": str(Path(args.ant_full_json).expanduser().resolve()),
            "action_delta": float(args.action_delta),
            "gym_control_env_ids": gym_control_env_ids,
        },
        "existing_prior_summary": summarize_existing_prior(prior_rows),
        "ant_reference": {
            "clean_obs": _extract_ant_scale_summary(ant_clean),
            "full_obs": _extract_ant_scale_summary(ant_full),
        },
        "prior_control_probe": summarize_prior_control(prior_control_rows) if prior_control_rows else None,
        "ant_control_probe": ant_control,
        "gym_control_probes": gym_control_probes,
        "next_decision_rule": (
            "Only propose rejection/parameter changes after the prior control probe is compared to the Ant control "
            "probe. If gain>=0.7 rows have low action/reward sensitivity or high noise dominance relative to Ant, "
            "the next main contradiction is prior controllability/open-loop risk rather than PPO optimizer noise."
        ),
    }
    output_path = Path(args.output_json).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(_json_safe(report), sort_keys=True, indent=2)
    output_path.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
