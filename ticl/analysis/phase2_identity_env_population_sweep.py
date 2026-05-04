import argparse
import contextlib
import copy
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ticl.analysis.phase2_suffix_state_identity_probe import (
    _build_audit_env_cfg,
    _build_fixed_unit_gaussian_policy_step_fn,
    _capture_suffix_bundles_unit_gaussian_batch,
    _json_safe_value,
    _load_full_frozen_h,
    _resolve_probe_config,
    run_phase2_suffix_state_identity_probe,
)
from ticl.analysis.critic_free_single_env_audit import _seed_all
from ticl.priors.environment_prior import EnvironmentPrior


def _json_dumps_stable(payload: Any) -> str:
    return json.dumps(_json_safe_value(payload), sort_keys=True, separators=(",", ":"))


def _sha256_json(payload: Any) -> str:
    return hashlib.sha256(_json_dumps_stable(payload).encode("utf-8")).hexdigest()


def _as_float(value: Any, default: float = float("nan")) -> float:
    if value is None:
        return float(default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _as_bool01(value: Any) -> int:
    return int(bool(value))


def _activation_name(value: Any) -> str:
    text = str(value)
    for name in ("Tanh", "ReLU", "Identity", "Sigmoid", "GELU"):
        if name in text:
            return name.lower()
    return text.lower()


def _safe_quantile(values: np.ndarray, q: float) -> float:
    if values.size == 0:
        return float("nan")
    return float(np.quantile(values.astype(np.float64, copy=False), float(q)))


def _vector_stats(prefix: str, values: np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return {
            f"{prefix}_count": 0.0,
            f"{prefix}_mean": float("nan"),
            f"{prefix}_std": float("nan"),
            f"{prefix}_min": float("nan"),
            f"{prefix}_max": float("nan"),
            f"{prefix}_q10": float("nan"),
            f"{prefix}_q25": float("nan"),
            f"{prefix}_q50": float("nan"),
            f"{prefix}_q75": float("nan"),
            f"{prefix}_q90": float("nan"),
        }
    return {
        f"{prefix}_count": float(arr.size),
        f"{prefix}_mean": float(np.mean(arr)),
        f"{prefix}_std": float(np.std(arr)),
        f"{prefix}_min": float(np.min(arr)),
        f"{prefix}_max": float(np.max(arr)),
        f"{prefix}_q10": _safe_quantile(arr, 0.10),
        f"{prefix}_q25": _safe_quantile(arr, 0.25),
        f"{prefix}_q50": _safe_quantile(arr, 0.50),
        f"{prefix}_q75": _safe_quantile(arr, 0.75),
        f"{prefix}_q90": _safe_quantile(arr, 0.90),
    }


def _norm_stats(prefix: str, tensor: torch.Tensor) -> dict[str, float]:
    arr = tensor.detach().float().cpu().numpy()
    if arr.ndim == 1:
        norms = np.asarray([np.linalg.norm(arr)], dtype=np.float64)
    else:
        norms = np.linalg.norm(arr.reshape(arr.shape[0], -1), axis=1)
    out = _vector_stats(f"{prefix}_l2", norms)
    flat = arr.reshape(-1).astype(np.float64, copy=False)
    out.update(_vector_stats(f"{prefix}_value", flat))
    out[f"{prefix}_absmax"] = float(np.max(np.abs(flat))) if flat.size else float("nan")
    return out


def _covariance_shape_features(prefix: str, tensor: torch.Tensor) -> dict[str, float]:
    x = tensor.detach().float().cpu().numpy().astype(np.float64, copy=False)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    n, d = int(x.shape[0]), int(x.shape[1])
    out = {f"{prefix}_sample_count": float(n), f"{prefix}_dim": float(d)}
    if n <= 1 or d <= 0:
        out.update(
            {
                f"{prefix}_cov_top_eigen_share": float("nan"),
                f"{prefix}_cov_effective_rank": float("nan"),
                f"{prefix}_cov_top_to_median_positive_eigenvalue": float("nan"),
                f"{prefix}_pairwise_l2_mean": float("nan"),
                f"{prefix}_pairwise_l2_q50": float("nan"),
                f"{prefix}_pairwise_l2_q90": float("nan"),
            }
        )
        return out
    xc = x - np.mean(x, axis=0, keepdims=True)
    singular_values = np.linalg.svd(xc, compute_uv=False)
    eigvals = np.square(singular_values) / max(1, n - 1)
    positive = eigvals[eigvals > 1e-12]
    eig_sum = float(np.sum(positive))
    out[f"{prefix}_cov_top_eigen_share"] = (
        float(np.max(positive) / eig_sum) if eig_sum > 0.0 and positive.size > 0 else float("nan")
    )
    out[f"{prefix}_cov_effective_rank"] = (
        float((eig_sum * eig_sum) / np.sum(np.square(positive)))
        if eig_sum > 0.0 and positive.size > 0
        else float("nan")
    )
    out[f"{prefix}_cov_top_to_median_positive_eigenvalue"] = (
        float(np.max(positive) / max(float(np.median(positive)), 1e-12))
        if positive.size > 0
        else float("nan")
    )
    norms_sq = np.sum(np.square(x), axis=1)
    d2 = norms_sq[:, None] + norms_sq[None, :] - 2.0 * (x @ x.T)
    d2 = np.maximum(d2, 0.0)
    upper = np.sqrt(d2[np.triu_indices(n, k=1)])
    out[f"{prefix}_pairwise_l2_mean"] = float(np.mean(upper)) if upper.size else float("nan")
    out[f"{prefix}_pairwise_l2_q50"] = _safe_quantile(upper, 0.50)
    out[f"{prefix}_pairwise_l2_q90"] = _safe_quantile(upper, 0.90)
    return out


def _return_features(report: dict[str, Any]) -> dict[str, float]:
    aggregate = dict(report.get("aggregate", {}))
    anchors = list(report.get("anchors", []))
    state_means = np.asarray([float(a.get("continuation_return_mean", float("nan"))) for a in anchors])
    state_stds = np.asarray([float(a.get("continuation_return_std", float("nan"))) for a in anchors])
    flat_returns = np.asarray(
        [float(v) for a in anchors for v in a.get("continuation_returns", [])],
        dtype=np.float64,
    )
    out = {
        "identity_score": _as_float(aggregate.get("identity_score")),
        "signal": _as_float(aggregate.get("signal")),
        "noise": _as_float(aggregate.get("noise")),
        "total_variance": _as_float(aggregate.get("total_variance")),
        "mean_of_state_means": _as_float(aggregate.get("mean_of_state_means")),
        "std_of_state_means": _as_float(aggregate.get("std_of_state_means")),
        "mean_within_state_std": _as_float(aggregate.get("mean_within_state_std")),
    }
    noise = out["noise"]
    signal = out["signal"]
    out["signal_to_noise"] = float(signal / max(noise, 1e-12)) if math.isfinite(noise) else float("nan")
    out["noise_to_signal"] = float(noise / max(signal, 1e-12)) if math.isfinite(signal) else float("nan")
    out.update(_vector_stats("state_return_mean", state_means[np.isfinite(state_means)]))
    out.update(_vector_stats("state_return_std", state_stds[np.isfinite(state_stds)]))
    out.update(_vector_stats("all_return", flat_returns[np.isfinite(flat_returns)]))
    out["state_return_mean_abs_mean"] = (
        float(np.mean(np.abs(state_means[np.isfinite(state_means)]))) if np.isfinite(state_means).any() else float("nan")
    )
    out["state_return_mean_positive_fraction"] = (
        float(np.mean(state_means[np.isfinite(state_means)] > 0.0)) if np.isfinite(state_means).any() else float("nan")
    )
    return out


def _horizon_features(report: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for item in list(report.get("horizon_diagnostics") or []):
        horizon_step = int(item.get("horizon_step"))
        prefix = f"horizon_{horizon_step}"
        for key in (
            "identity_score",
            "bias_corrected_identity_score",
            "signal",
            "bias_corrected_signal",
            "noise",
            "total_variance",
            "finite_repeat_signal_bias_estimate",
            "signal_to_noise",
            "bias_corrected_signal_to_noise",
        ):
            out[f"{prefix}_{key}"] = _as_float(item.get(key))
    return out


def _snapshot_features(snapshot: dict[str, Any], sampled_snapshot: dict[str, Any], *, n_steps: int) -> dict[str, Any]:
    state_dim = _as_float(snapshot.get("state_dim"))
    obs_dim = _as_float(snapshot.get("obs_dim"))
    action_dim = _as_float(snapshot.get("action_dim"))
    noise_dim = _as_float(snapshot.get("noise_dim"))
    pad_dim = _as_float(snapshot.get("zero_pad_dim"))
    budget_sum = state_dim + action_dim + noise_dim + pad_dim
    out: dict[str, Any] = {
        "family": str(snapshot.get("family")),
        "activation": _activation_name(snapshot.get("prior_mlp_activations")),
        "state_dim": state_dim,
        "obs_dim": obs_dim,
        "action_dim": action_dim,
        "noise_dim": noise_dim,
        "zero_pad_dim": pad_dim,
        "dimension_sum_state_action_noise_pad": budget_sum,
        "state_dim_fraction": float(state_dim / budget_sum) if budget_sum > 0 else float("nan"),
        "action_dim_fraction": float(action_dim / budget_sum) if budget_sum > 0 else float("nan"),
        "noise_dim_fraction": float(noise_dim / budget_sum) if budget_sum > 0 else float("nan"),
        "pad_dim_fraction": float(pad_dim / budget_sum) if budget_sum > 0 else float("nan"),
        "obs_to_state_ratio": float(obs_dim / max(state_dim, 1e-12)),
        "alpha": _as_float(snapshot.get("alpha")),
        "reference_state_inertia_enabled": _as_bool01(snapshot.get("reference_state_inertia_enabled")),
        "terminal_reset_enabled": _as_bool01(snapshot.get("terminal_reset_enabled")),
        "terminal_reset_count_target": _as_float(snapshot.get("terminal_reset_count_target")),
        "terminal_reset_expected_prob_per_step": float(
            _as_float(snapshot.get("terminal_reset_count_target"), 0.0) / max(1, int(n_steps))
        ),
        "num_layers": _as_float(snapshot.get("num_layers")),
        "prior_mlp_hidden_dim": _as_float(snapshot.get("prior_mlp_hidden_dim")),
        "scm_standard_linear_init_enabled": _as_bool01(snapshot.get("scm_standard_linear_init_enabled")),
        "init_state_std": _as_float(snapshot.get("init_state_std")),
        "init_std": _as_float(snapshot.get("init_std")),
        "noise_std": _as_float(snapshot.get("noise_std")),
        "log10_init_std": float(np.log10(max(_as_float(snapshot.get("init_std"), 0.0), 1e-12))),
        "log10_noise_std": float(np.log10(max(_as_float(snapshot.get("noise_std"), 0.0), 1e-12))),
        "sampled_state_dim": _as_float(sampled_snapshot.get("state_dim")),
        "sampled_obs_dim": _as_float(sampled_snapshot.get("obs_dim")),
        "sampled_action_dim": _as_float(sampled_snapshot.get("action_dim")),
        "sampled_noise_dim": _as_float(sampled_snapshot.get("noise_dim")),
        "sampled_zero_pad_dim": _as_float(sampled_snapshot.get("zero_pad_dim")),
        "state_noise_std": _as_float(sampled_snapshot.get("state_noise_std")),
        "reward_scale": _as_float(sampled_snapshot.get("reward_scale")),
        "ctrl_reward_enabled": _as_bool01(sampled_snapshot.get("ctrl_reward_enabled")),
        "survival_reward_enabled": _as_bool01(sampled_snapshot.get("survival_reward_enabled")),
        "state_highway_enabled": _as_bool01(sampled_snapshot.get("state_highway_enabled")),
        "state_highway_lambda": _as_float(sampled_snapshot.get("state_highway_lambda")),
    }
    return out


def _stable_report_fingerprint(report: dict[str, Any]) -> str:
    stable_payload = {
        "aggregate": report.get("aggregate"),
        "anchors": report.get("anchors"),
        "effective_frozen_h_snapshot": report.get("effective_frozen_h_snapshot"),
        "sampled_env_snapshot": report.get("sampled_env_snapshot"),
        "fixed_env_contract": report.get("fixed_env_contract"),
        "frozen_h_seed": report.get("frozen_h_seed"),
        "train_env_seed": report.get("train_env_seed"),
        "train_rollout_seed": report.get("train_rollout_seed"),
        "single_eval_pos": report.get("single_eval_pos"),
        "anchor_step": report.get("anchor_step"),
        "n_steps": report.get("n_steps"),
        "reference_state_count": report.get("reference_state_count"),
        "continuation_repeats": report.get("continuation_repeats"),
        "reference_rollout_seed_start": report.get("reference_rollout_seed_start"),
        "continuation_rollout_seed_start": report.get("continuation_rollout_seed_start"),
        "discount_gamma": report.get("discount_gamma"),
        "behavior_policy": report.get("behavior_policy"),
        "sampled_policy": report.get("sampled_policy"),
        "horizon_diagnostics_steps": report.get("horizon_diagnostics_steps"),
        "horizon_diagnostics": report.get("horizon_diagnostics"),
    }
    return _sha256_json(stable_payload)


def _anchor_state_features(
    *,
    full_frozen_h_json: str,
    device: str | None,
    build_seed: int,
    train_env_seed: int,
    single_eval_pos: int,
    anchor_step: int,
    n_steps: int,
    reference_state_count: int,
    reference_rollout_seed_start: int,
    sampled_policy: bool,
) -> dict[str, float]:
    _seed_all(int(build_seed))
    device_obj = torch.device(str(device or ("cuda:0" if torch.cuda.is_available() else "cpu")))
    config = _resolve_probe_config(
        checkpoint_path=None,
        from_scratch=True,
        from_scratch_model_type="rlpfn",
        device_obj=device_obj,
        build_seed=int(build_seed),
        allow_config_only=True,
    )
    env_cfg = _build_audit_env_cfg(config["prior"]["environment"])
    frozen_h, _ = _load_full_frozen_h(str(full_frozen_h_json))
    prior = EnvironmentPrior(copy.deepcopy(env_cfg))
    env = prior._sample_environment(copy.deepcopy(frozen_h), device=device_obj, rng_seed=int(train_env_seed))
    policy_step_fn = _build_fixed_unit_gaussian_policy_step_fn(sampled=bool(sampled_policy))
    reference_seeds = [int(reference_rollout_seed_start) + int(i) for i in range(int(reference_state_count))]
    bundle = _capture_suffix_bundles_unit_gaussian_batch(
        prior=prior,
        env=env,
        policy_step_fn=policy_step_fn,
        sampled_policy=bool(sampled_policy),
        anchor_step=int(anchor_step),
        n_steps=int(n_steps),
        single_eval_pos=int(single_eval_pos),
        device=device_obj,
        rng_seeds=reference_seeds,
    )
    state_t = bundle["state_t"].detach()
    obs_t = state_t[:, : int(env["obs_dim"])].detach()
    action_t = bundle["action_t"].detach()
    reward_t = bundle["reward_t"].detach()
    terminal_t = bundle["terminal_t"].detach()
    initial_state = env.get("initial_state")
    out: dict[str, float] = {}
    if torch.is_tensor(initial_state):
        out.update(_norm_stats("initial_state", initial_state.detach()))
    out.update(_norm_stats("anchor_state", state_t))
    out.update(_norm_stats("anchor_obs", obs_t))
    out.update(_norm_stats("anchor_action", action_t))
    out.update(_vector_stats("anchor_reward", reward_t.detach().float().cpu().numpy()))
    out["anchor_terminal_rate"] = float(terminal_t.detach().float().mean().cpu().item())
    out.update(_covariance_shape_features("anchor_state", state_t))
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


def _build_probe_kwargs(args: argparse.Namespace, *, frozen_h_seed: int, full_frozen_h_path: Path | None) -> dict[str, Any]:
    return {
        "checkpoint_path": None,
        "device": args.device,
        "from_scratch": True,
        "from_scratch_model_type": "rlpfn",
        "frozen_h_seed": int(frozen_h_seed),
        "train_env_seed": int(args.train_env_seed),
        "train_rollout_seed": int(args.train_rollout_seed),
        "single_eval_pos": int(args.single_eval_pos),
        "anchor_step": int(args.anchor_step),
        "n_steps": int(args.n_steps),
        "batch_size": int(args.batch_size),
        "n_epochs": 1,
        "learning_rate": 2e-4,
        "target_kl": 0.03,
        "outer_epochs": 0,
        "build_seed": int(args.build_seed),
        "ppo_reset_env_state_at_sep": True,
        "strict_native_rollout": False,
        "value_head_impl": "scalar_mlp",
        "value_path_adapter_impl": "none",
        "value_head_mlp_hidden_dim": 256,
        "space_contract": "normalized",
        "actor_baseline_mode": "learned",
        "vf_coef_override": None,
        "policy_loss_coef_override": None,
        "behavior_policy": "unit_gaussian",
        "reference_state_count": int(args.reference_state_count),
        "continuation_repeats": int(args.continuation_repeats),
        "reference_rollout_seed_start": int(args.reference_rollout_seed_start),
        "continuation_rollout_seed_start": int(args.continuation_rollout_seed_start),
        "sampled_policy": bool(args.sampled_policy),
        "discount_gamma": float(args.discount_gamma),
        "constrained_dim_sampling_total_budget_override": args.constrained_dim_sampling_total_budget_override,
        "state_dim_override": args.state_dim_override,
        "action_dim_override": args.action_dim_override,
        "noise_dim_override": args.noise_dim_override,
        "zero_pad_dim_override": args.zero_pad_dim_override,
        "alpha_override": args.alpha_override,
        "reference_state_inertia_enabled_override": args.reference_state_inertia_enabled_override,
        "terminal_reset_enabled_override": args.terminal_reset_enabled_override,
        "terminal_reset_count_target_override": args.terminal_reset_count_target_override,
        "frozen_h_overrides": (
            None
            if args.init_state_std_override is None
            else {"init_state_std": float(args.init_state_std_override)}
        ),
        "fixed_frozen_h_json": None,
        "save_full_frozen_h_json": None if full_frozen_h_path is None else str(full_frozen_h_path),
        "progress_jsonl": None,
        "print_progress_every_repeat": False,
        "horizon_diagnostics_steps": args.horizon_diagnostics_steps,
    }


def _run_identity_once(
    *,
    args: argparse.Namespace,
    frozen_h_seed: int,
    output_root: Path,
    run_tag: str,
    save_report: bool,
) -> tuple[dict[str, Any], str | None, str]:
    _seed_all(int(args.build_seed))
    env_dir = output_root / "envs" / f"seed_{int(frozen_h_seed)}"
    env_dir.mkdir(parents=True, exist_ok=True)
    full_h_path = env_dir / f"{run_tag}.full_frozen_h.json"
    stdout_path = env_dir / f"{run_tag}.stdout.log"
    report_path = env_dir / f"{run_tag}.report.json"
    kwargs = _build_probe_kwargs(args, frozen_h_seed=int(frozen_h_seed), full_frozen_h_path=full_h_path)
    if bool(args.redirect_probe_stdout):
        with stdout_path.open("w", encoding="utf-8") as handle, contextlib.redirect_stdout(handle):
            report = run_phase2_suffix_state_identity_probe(**kwargs)
    else:
        report = run_phase2_suffix_state_identity_probe(**kwargs)
    report_fingerprint = _stable_report_fingerprint(report)
    if bool(save_report):
        report_path.write_text(json.dumps(_json_safe_value(report), sort_keys=True) + "\n", encoding="utf-8")
        report_path_str: str | None = str(report_path)
    else:
        report_path_str = None
    return report, report_path_str, report_fingerprint


def build_row_from_report(
    *,
    args: argparse.Namespace,
    frozen_h_seed: int,
    report: dict[str, Any],
    report_json: str | None,
    report_fingerprint: str,
) -> dict[str, Any]:
    snapshot = dict(report.get("effective_frozen_h_snapshot", {}))
    sampled_snapshot = dict(report.get("sampled_env_snapshot", {}))
    full_h_path = str(report.get("save_full_frozen_h_json"))
    row: dict[str, Any] = {
        "frozen_h_seed": int(frozen_h_seed),
        "report_fingerprint": str(report_fingerprint),
        "report_json": report_json,
        "full_frozen_h_json": full_h_path,
        "reference_state_count": int(args.reference_state_count),
        "continuation_repeats": int(args.continuation_repeats),
        "batch_size": int(args.batch_size),
        "anchor_step": int(args.anchor_step),
        "n_steps": int(args.n_steps),
        "discount_gamma": float(args.discount_gamma),
        "sampled_policy": bool(args.sampled_policy),
    }
    row.update(_snapshot_features(snapshot, sampled_snapshot, n_steps=int(args.n_steps)))
    row.update(_return_features(report))
    row.update(_horizon_features(report))
    if bool(args.extract_anchor_state_features):
        row.update(
            _anchor_state_features(
                full_frozen_h_json=full_h_path,
                device=args.device,
                build_seed=int(args.build_seed),
                train_env_seed=int(args.train_env_seed),
                single_eval_pos=int(args.single_eval_pos),
                anchor_step=int(args.anchor_step),
                n_steps=int(args.n_steps),
                reference_state_count=int(args.reference_state_count),
                reference_rollout_seed_start=int(args.reference_rollout_seed_start),
                sampled_policy=bool(args.sampled_policy),
            )
        )
    return _json_safe_value(row)


def _read_existing_rows(rows_path: Path) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    if not rows_path.exists():
        return rows
    for line in rows_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        rows[int(row["frozen_h_seed"])] = row
    return rows


def _write_rows_jsonl(rows_path: Path, rows: list[dict[str, Any]]) -> None:
    rows_path.parent.mkdir(parents=True, exist_ok=True)
    with rows_path.open("w", encoding="utf-8") as handle:
        for row in sorted(rows, key=lambda item: int(item["frozen_h_seed"])):
            handle.write(json.dumps(_json_safe_value(row), sort_keys=True) + "\n")


def _rank_and_summarize(args: argparse.Namespace, rows: list[dict[str, Any]], sentinel_checks: list[dict[str, Any]]) -> dict[str, Any]:
    ranked = sorted(rows, key=lambda item: float(item["identity_score"]), reverse=True)
    for idx, row in enumerate(ranked, start=1):
        row["rank_by_identity"] = int(idx)
    ranked_horizon2 = sorted(
        [row for row in rows if "horizon_2_bias_corrected_identity_score" in row],
        key=lambda item: float(item["horizon_2_bias_corrected_identity_score"]),
        reverse=True,
    )
    for idx, row in enumerate(ranked_horizon2, start=1):
        row["rank_by_horizon_2_bias_corrected_identity"] = int(idx)
    by_activation: dict[str, list[float]] = {}
    for row in rows:
        by_activation.setdefault(str(row.get("activation")), []).append(float(row["identity_score"]))
    activation_summary = {
        key: {
            "count": len(vals),
            "mean_identity": float(np.mean(vals)),
            "median_identity": float(np.median(vals)),
            "max_identity": float(np.max(vals)),
        }
        for key, vals in sorted(by_activation.items())
    }
    identities = np.asarray([float(row["identity_score"]) for row in rows], dtype=np.float64)
    horizon2_corrected = np.asarray(
        [float(row["horizon_2_bias_corrected_identity_score"]) for row in rows if "horizon_2_bias_corrected_identity_score" in row],
        dtype=np.float64,
    )
    horizon2_raw = np.asarray(
        [float(row["horizon_2_identity_score"]) for row in rows if "horizon_2_identity_score" in row],
        dtype=np.float64,
    )
    summary = {
        "audit_entry": "phase2_identity_env_population_sweep",
        "assumption": {
            "frozen_h_seed_start": int(args.frozen_h_seed_start),
            "env_count_requested": int(args.env_count),
            "env_count_completed": int(len(rows)),
            "reference_state_count": int(args.reference_state_count),
            "continuation_repeats": int(args.continuation_repeats),
            "batch_size": int(args.batch_size),
            "single_eval_pos": int(args.single_eval_pos),
            "anchor_step": int(args.anchor_step),
            "n_steps": int(args.n_steps),
            "discount_gamma": float(args.discount_gamma),
            "horizon_diagnostics_steps": args.horizon_diagnostics_steps,
            "behavior_policy": "unit_gaussian",
            "sampled_policy": bool(args.sampled_policy),
            "train_env_seed": int(args.train_env_seed),
            "train_rollout_seed": int(args.train_rollout_seed),
            "reference_rollout_seed_start": int(args.reference_rollout_seed_start),
            "continuation_rollout_seed_start": int(args.continuation_rollout_seed_start),
            "overrides": {
                "constrained_dim_sampling_total_budget_override": args.constrained_dim_sampling_total_budget_override,
                "state_dim_override": args.state_dim_override,
                "action_dim_override": args.action_dim_override,
                "noise_dim_override": args.noise_dim_override,
                "zero_pad_dim_override": args.zero_pad_dim_override,
                "alpha_override": args.alpha_override,
                "init_state_std_override": args.init_state_std_override,
                "reference_state_inertia_enabled_override": args.reference_state_inertia_enabled_override,
                "terminal_reset_enabled_override": args.terminal_reset_enabled_override,
                "terminal_reset_count_target_override": args.terminal_reset_count_target_override,
            },
        },
        "identity_distribution": _vector_stats("identity", identities),
        "horizon_2_bias_corrected_identity_distribution": _vector_stats(
            "horizon_2_bias_corrected_identity",
            horizon2_corrected,
        ),
        "horizon_2_identity_distribution": _vector_stats("horizon_2_identity", horizon2_raw),
        "activation_summary": activation_summary,
        "top_envs": ranked[: int(args.top_k)],
        "bottom_envs": ranked[-int(args.top_k):] if ranked else [],
        "top_horizon_2_envs": ranked_horizon2[: int(args.top_k)],
        "bottom_horizon_2_envs": ranked_horizon2[-int(args.top_k):] if ranked_horizon2 else [],
        "sentinel_checks": sentinel_checks,
        "sentinel_all_exact": bool(sentinel_checks) and all(bool(item.get("exact_match")) for item in sentinel_checks),
    }
    return _json_safe_value(summary)


def run_population_sweep(args: argparse.Namespace) -> dict[str, Any]:
    output_root = Path(args.output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    rows_path = output_root / "rows.jsonl"
    summary_path = output_root / "summary.json"
    existing = _read_existing_rows(rows_path) if bool(args.resume) else {}
    rows_by_seed = dict(existing)
    seeds = [int(args.frozen_h_seed_start) + int(i) for i in range(int(args.env_count))]
    t0 = time.time()
    for idx, seed in enumerate(seeds, start=1):
        if bool(args.resume) and int(seed) in rows_by_seed:
            print(f"[population-sweep] skip existing seed={seed}", flush=True)
            continue
        print(f"[population-sweep] run seed={seed} ({idx}/{len(seeds)})", flush=True)
        report, report_json, fingerprint = _run_identity_once(
            args=args,
            frozen_h_seed=int(seed),
            output_root=output_root,
            run_tag="main",
            save_report=bool(args.save_full_reports),
        )
        row = build_row_from_report(
            args=args,
            frozen_h_seed=int(seed),
            report=report,
            report_json=report_json,
            report_fingerprint=fingerprint,
        )
        rows_by_seed[int(seed)] = row
        _write_rows_jsonl(rows_path, list(rows_by_seed.values()))

    sentinel_checks: list[dict[str, Any]] = []
    sentinel_seeds = [int(args.frozen_h_seed_start) + int(i) for i in range(min(int(args.sentinel_count), len(seeds)))]
    for seed in sentinel_seeds:
        if seed not in rows_by_seed:
            continue
        print(f"[population-sweep] sentinel rerun seed={seed}", flush=True)
        report, report_json, fingerprint = _run_identity_once(
            args=args,
            frozen_h_seed=int(seed),
            output_root=output_root,
            run_tag="sentinel",
            save_report=bool(args.save_sentinel_reports),
        )
        main_row = rows_by_seed[int(seed)]
        identity = float(report["aggregate"]["identity_score"])
        main_identity = float(main_row["identity_score"])
        sentinel_checks.append(
            {
                "frozen_h_seed": int(seed),
                "main_identity": main_identity,
                "sentinel_identity": identity,
                "identity_abs_diff": float(abs(main_identity - identity)),
                "main_report_fingerprint": str(main_row["report_fingerprint"]),
                "sentinel_report_fingerprint": str(fingerprint),
                "exact_match": bool(str(main_row["report_fingerprint"]) == str(fingerprint)),
                "sentinel_report_json": report_json,
            }
        )

    rows = list(rows_by_seed.values())
    summary = _rank_and_summarize(args, rows, sentinel_checks)
    summary["elapsed_sec"] = float(time.time() - t0)
    summary_path.write_text(json.dumps(_json_safe_value(summary), sort_keys=True) + "\n", encoding="utf-8")
    print(f"SUMMARY_WRITTEN {summary_path}", flush=True)
    return summary


def _str_to_bool_or_none(text: str | None) -> bool | None:
    if text is None:
        return None
    value = str(text).strip().lower()
    if value in {"none", "null", ""}:
        return None
    if value in {"1", "true", "yes", "y"}:
        return True
    if value in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected bool or none, got {text!r}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Sample many exact-SCM prior environments, measure state-return identity, extract "
            "configuration and trajectory/return features, and rerun sentinel seeds for determinism."
        )
    )
    parser.add_argument("--output-dir", type=str, default="/home/chen/RLPFN/artifacts/phase2_identity_env_population_sweep")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--build-seed", type=int, default=4040)
    parser.add_argument("--frozen-h-seed-start", type=int, default=20000)
    parser.add_argument("--env-count", type=int, default=8)
    parser.add_argument("--train-env-seed", type=int, default=2020)
    parser.add_argument("--train-rollout-seed", type=int, default=4040)
    parser.add_argument("--single-eval-pos", type=int, default=64)
    parser.add_argument("--anchor-step", type=int, default=128)
    parser.add_argument("--n-steps", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--reference-state-count", type=int, default=64)
    parser.add_argument("--continuation-repeats", type=int, default=64)
    parser.add_argument("--reference-rollout-seed-start", type=int, default=5000)
    parser.add_argument("--continuation-rollout-seed-start", type=int, default=10000)
    parser.add_argument("--discount-gamma", type=float, default=0.99)
    parser.add_argument(
        "--horizon-diagnostics-steps",
        type=str,
        default=None,
        help="Comma-separated suffix horizons for G_k diagnostics, e.g. '2' or '1,2,4'.",
    )
    parser.add_argument("--sampled-policy", type=_str_to_bool_or_none, default=True)
    parser.add_argument("--constrained-dim-sampling-total-budget-override", type=int, default=None)
    parser.add_argument("--state-dim-override", type=int, default=None)
    parser.add_argument("--action-dim-override", type=int, default=None)
    parser.add_argument("--noise-dim-override", type=int, default=None)
    parser.add_argument("--zero-pad-dim-override", type=int, default=None)
    parser.add_argument("--alpha-override", type=float, default=None)
    parser.add_argument("--init-state-std-override", type=float, default=None)
    parser.add_argument("--reference-state-inertia-enabled-override", type=_str_to_bool_or_none, default=None)
    parser.add_argument("--terminal-reset-enabled-override", type=_str_to_bool_or_none, default=None)
    parser.add_argument("--terminal-reset-count-target-override", type=float, default=None)
    parser.add_argument("--sentinel-count", type=int, default=2)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--resume", type=_str_to_bool_or_none, default=True)
    parser.add_argument("--save-full-reports", type=_str_to_bool_or_none, default=True)
    parser.add_argument("--save-sentinel-reports", type=_str_to_bool_or_none, default=True)
    parser.add_argument("--extract-anchor-state-features", type=_str_to_bool_or_none, default=True)
    parser.add_argument("--redirect-probe-stdout", type=_str_to_bool_or_none, default=True)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    run_population_sweep(args)


if __name__ == "__main__":
    main()
