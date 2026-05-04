import argparse
import copy
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ticl.analysis.critic_free_single_env_audit import _build_audit_env_cfg
from ticl.analysis.phase2_alpha_temporal_control_probe import (
    _compute_preterminal_state_next_batch,
    _make_generators_from_seeds,
)
from ticl.analysis.phase2_prior_gym_alignment_deep_feature_audit import (
    measure_prior_deep_features,
)
from ticl.analysis.phase2_suffix_state_identity_probe import (
    _broadcast_like_batch,
    _load_full_frozen_h,
    _resolve_probe_config,
    _to_float,
    run_phase2_suffix_state_identity_probe,
)
from ticl.priors.environment_prior import EnvironmentPrior


DEFAULT_SOURCE_ROWS = (
    "/home/chen/RLPFN/artifacts/"
    "phase2_prior_controlled_rms_noise_deep_gate_0429/rows.jsonl"
)
DEFAULT_HARD_GATE_REPORT = (
    "/home/chen/RLPFN/artifacts/"
    "phase2_prior_gym_hard_gate_report_0429/hard_gate_report.json"
)


def _finite(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _get(data: dict[str, Any], path: str, default: Any = None) -> Any:
    cur: Any = data
    for part in path.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return default
    return cur


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, torch.Tensor):
        return _json_safe(value.detach().cpu().numpy())
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return str(value)


def _stats(values: list[Any]) -> dict[str, Any]:
    arr = np.asarray([v for v in (_finite(v) for v in values) if v is not None], dtype=np.float64)
    if arr.size == 0:
        return {"count": 0, "mean": None, "std": None, "q10": None, "q50": None, "q90": None, "min": None, "max": None}
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "q10": float(np.quantile(arr, 0.10)),
        "q50": float(np.quantile(arr, 0.50)),
        "q90": float(np.quantile(arr, 0.90)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def _load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).expanduser().open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _load_thresholds(path: str | Path) -> dict[str, float]:
    report = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    return dict((report.get("gym_reference") or {}).get("thresholds") or {})


def _safe_ratio(value: Any, denom: Any) -> float | None:
    val = _finite(value)
    den = _finite(denom)
    if val is None or den is None or abs(den) <= 1e-12:
        return None
    return float(val / den)


def _read_h(row: dict[str, Any]) -> dict[str, Any]:
    path = row.get("full_frozen_h_json")
    if not path:
        return {}
    return json.loads(Path(path).expanduser().read_text(encoding="utf-8"))


def _param(row: dict[str, Any], h: dict[str, Any], key: str) -> float | None:
    val = row.get(key)
    if val is None:
        val = h.get(key)
    return _finite(val)


def _quantile(values: list[float], frac: float) -> float:
    arr = np.asarray(values, dtype=np.float64)
    return float(np.quantile(arr, float(frac)))


def _enrich_baseline_rows(rows: list[dict[str, Any]], thresholds: dict[str, float]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        if str(row.get("case")) != "rms1":
            continue
        h = _read_h(row)
        rec = dict(row)
        for key in (
            "init_std",
            "init_action_std",
            "_constrained_obs_u",
            "_constrained_noise_u",
            "terminal_reset_count_target",
            "alpha",
            "noise_std",
            "lengthscale",
            "outputscale",
            "prior_mlp_hidden_dim",
            "num_layers",
            "obs_dim",
            "state_dim",
            "action_dim",
            "noise_dim",
            "zero_pad_dim",
        ):
            val = _param(row, h, key)
            if val is not None:
                rec[f"param_{key}"] = val
        metrics = {
            "obs_std": _get(row, "obs_value_std_all"),
            "reward_std": _get(row, "reward_effective.std"),
            "return_std": _get(row, "discounted_effective_return.std"),
            "obs_rank": _get(row, "obs_cov.effective_rank_fraction"),
            "obs_top": _get(row, "obs_cov.top_eigen_share"),
            "state_sens": _get(row, "per_action_state_sensitivity.q50"),
            "reward_sens": _get(row, "per_action_reward_sensitivity.q50"),
            "G5_identity": _get(row, "horizon_5_bias_corrected_identity_score"),
        }
        rec.update({f"metric_{k}": _finite(v) for k, v in metrics.items()})
        rec["obs_ratio"] = _safe_ratio(metrics["obs_std"], thresholds.get("obs_std_min"))
        rec["reward_ratio"] = _safe_ratio(metrics["reward_std"], thresholds.get("reward_std_min"))
        rec["return_ratio"] = _safe_ratio(metrics["return_std"], thresholds.get("return_std_min"))
        ratios = [rec.get("obs_ratio"), rec.get("reward_ratio"), rec.get("return_ratio")]
        ratios = [max(1e-9, min(float(v), 10.0)) for v in ratios if v is not None]
        rec["amp_geom_ratio"] = float(np.prod(ratios) ** (1.0 / len(ratios))) if ratios else None
        out.append(rec)
    return out


def _select_rows(rows: list[dict[str, Any]], count: int, seed: int) -> list[dict[str, Any]]:
    if int(count) <= 0 or int(count) >= len(rows):
        return list(rows)
    rng = np.random.default_rng(int(seed))
    selected: list[dict[str, Any]] = []

    def add(candidates: list[dict[str, Any]], take: int) -> None:
        nonlocal selected
        seen = {str(row.get("frozen_h_seed")) for row in selected}
        candidates = [row for row in candidates if str(row.get("frozen_h_seed")) not in seen]
        if not candidates or take <= 0:
            return
        take = min(int(take), len(candidates))
        idx = sorted(rng.choice(len(candidates), size=take, replace=False).tolist())
        selected.extend([candidates[i] for i in idx])

    by_amp = sorted([r for r in rows if _finite(r.get("amp_geom_ratio")) is not None], key=lambda r: float(r["amp_geom_ratio"]))
    by_obs_u = sorted([r for r in rows if _finite(r.get("param__constrained_obs_u")) is not None], key=lambda r: float(r["param__constrained_obs_u"]))
    by_init = sorted([r for r in rows if _finite(r.get("param_init_std")) is not None], key=lambda r: float(r["param_init_std"]))
    bucket = max(1, int(round(int(count) / 6)))
    add(by_amp[: max(bucket * 2, 1)], bucket)
    add(by_amp[-max(bucket * 2, 1) :], bucket)
    add(by_obs_u[: max(bucket * 2, 1)], bucket)
    add(by_obs_u[-max(bucket * 2, 1) :], bucket)
    add(by_init[-max(bucket * 2, 1) :], bucket)
    add(list(rows), int(count) - len(selected))
    return selected[: int(count)]


def _tertile_label(value: Any, lo: float, hi: float) -> str:
    val = _finite(value)
    if val is None:
        return "missing"
    if val <= lo:
        return "low"
    if val >= hi:
        return "high"
    return "mid"


def _add_selection_labels(rows: list[dict[str, Any]], all_rows: list[dict[str, Any]]) -> None:
    fields = {
        "amp": "amp_geom_ratio",
        "obs_u": "param__constrained_obs_u",
        "init_std": "param_init_std",
        "init_action_std": "param_init_action_std",
        "reward_noise_gain": "topology_reward_noise_input_gain_fraction",
        "reward_state_gain": "topology_reward_state_input_gain_fraction",
    }
    cuts: dict[str, tuple[float, float]] = {}
    for name, key in fields.items():
        vals = [_finite(row.get(key)) for row in all_rows]
        vals = [v for v in vals if v is not None]
        if vals:
            cuts[name] = (_quantile(vals, 1.0 / 3.0), _quantile(vals, 2.0 / 3.0))
    for row in rows:
        labels = {}
        for name, key in fields.items():
            if name in cuts:
                labels[name] = _tertile_label(row.get(key), cuts[name][0], cuts[name][1])
        row["selection_labels"] = labels


def _case_overrides(h: dict[str, Any], case_name: str) -> dict[str, Any]:
    overrides: dict[str, Any] = {
        "state_full_rms_enabled": True,
        "state_full_rms_target": 1.0,
    }
    if case_name == "baseline":
        return overrides
    if "noiseband_" in case_name:
        # Clamp into a target band rather than setting a single global value:
        # this raises under-controllable envs while avoiding extra noise in
        # already-high-noise envs. Format: noiseband_0p01_0p03.
        band_part = case_name.split("noiseband_", 1)[1]
        pieces = band_part.split("_")
        if len(pieces) < 2:
            raise ValueError(f"Malformed noiseband case: {case_name!r}")
        lo = float(pieces[0].replace("p", "."))
        hi = float(pieces[1].replace("p", "."))
        if not (math.isfinite(lo) and math.isfinite(hi) and 0.0 <= lo <= hi):
            raise ValueError(f"Invalid noiseband bounds in case: {case_name!r}")
        current = float(h.get("noise_std", 0.0))
        overrides["noise_std"] = float(min(max(current, lo), hi))
    if case_name in {"initstd_x1p5", "initstd_x1p5_initaction_x0p5"}:
        overrides["init_std"] = float(h["init_std"]) * 1.5
    if case_name == "initstd_x2":
        overrides["init_std"] = float(h["init_std"]) * 2.0
    if case_name in {"initaction_x0p5", "initstd_x1p5_initaction_x0p5"}:
        overrides["init_action_std"] = float(h["init_action_std"]) * 0.5
    if case_name == "obsu_high_rejection_like":
        # This is intentionally a no-op transform. The associated labels/subsets
        # are used to audit rejection-style selection without pretending that
        # _constrained_obs_u can be changed after a frozen_h has been materialized.
        return overrides
    if case_name.startswith("stateout_x"):
        # state_output_scale is resolved at environment construction and applied
        # before the state_full_rms safety cap. Unlike frozen dimensions/weights,
        # it is a genuine transition-scale knob rather than a post-hoc rewrite of
        # materialized generator tensors.
        suffix = case_name[len("stateout_x") :].split("_", 1)[0]
        multiplier = float(suffix.replace("p", "."))
        overrides["state_output_scale"] = float(h.get("state_output_scale", 1.0)) * multiplier
    return overrides


def _identity_curve(report: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for item in report.get("horizon_diagnostics") or []:
        h = int(item["horizon_step"])
        out[f"G{h}_bias_corrected_identity"] = float(item["bias_corrected_identity_score"])
        out[f"G{h}_signal_to_noise"] = float(item["signal_to_noise"])
    return out


def _run_identity(
    *,
    source_row: dict[str, Any],
    overrides: dict[str, Any],
    device: str,
    reference_state_count: int,
    continuation_repeats: int,
    anchor_step: int,
    n_steps: int,
) -> dict[str, Any]:
    return run_phase2_suffix_state_identity_probe(
        checkpoint_path=None,
        device=str(device),
        from_scratch=True,
        from_scratch_model_type="rlpfn",
        fixed_frozen_h_json=str(source_row["full_frozen_h_json"]),
        frozen_h_seed=int(float(source_row["frozen_h_seed"])),
        train_env_seed=2020,
        train_rollout_seed=4040,
        single_eval_pos=64,
        anchor_step=int(anchor_step),
        n_steps=int(n_steps),
        batch_size=max(128, int(reference_state_count) * int(continuation_repeats)),
        behavior_policy="unit_gaussian",
        reference_state_count=int(reference_state_count),
        continuation_repeats=int(continuation_repeats),
        reference_rollout_seed_start=5000,
        continuation_rollout_seed_start=10000,
        sampled_policy=True,
        discount_gamma=0.99,
        horizon_diagnostics_steps="2,3,4,5",
        frozen_h_overrides=copy.deepcopy(overrides),
        print_progress_every_repeat=False,
    )


def _tensor_stats(values: torch.Tensor) -> dict[str, Any]:
    flat = values.detach().reshape(-1).to(device="cpu", dtype=torch.float64)
    finite = flat[torch.isfinite(flat)]
    if int(finite.numel()) == 0:
        return {"count": int(flat.numel()), "finite_fraction": 0.0, "mean": None, "q50": None, "q90": None, "max": None}
    return {
        "count": int(flat.numel()),
        "finite_fraction": float(finite.numel() / max(flat.numel(), 1)),
        "mean": float(finite.mean().item()),
        "q50": float(torch.quantile(finite, 0.50).item()),
        "q90": float(torch.quantile(finite, 0.90).item()),
        "max": float(finite.max().item()),
    }


def run_no_terminal_stability(
    *,
    source_row: dict[str, Any],
    overrides: dict[str, Any],
    device: str,
    n_steps: int,
    rollout_count: int,
    rollout_seed_start: int,
) -> dict[str, Any]:
    device_obj = torch.device(str(device))
    config = _resolve_probe_config(
        checkpoint_path=None,
        from_scratch=True,
        from_scratch_model_type="rlpfn",
        device_obj=device_obj,
        build_seed=4040,
        allow_config_only=True,
    )
    env_cfg = _build_audit_env_cfg(config["prior"]["environment"])
    frozen_h, _ = _load_full_frozen_h(str(source_row["full_frozen_h_json"]))
    frozen_h.update(copy.deepcopy(overrides))
    frozen_h["terminal_reset_enabled"] = False
    frozen_h["terminal_reset_count_target"] = 0.0
    prior = EnvironmentPrior(copy.deepcopy(env_cfg))
    env = prior._sample_environment(copy.deepcopy(frozen_h), device=device_obj, rng_seed=2020)
    batch_size = int(rollout_count)
    dtype = torch.float32
    state_dim = int(env["state_dim"])
    obs_dim = int(env["obs_dim"])
    action_dim = int(env["action_dim"])
    noise_dim = int(env["noise_dim"])
    rollout_seeds = [int(rollout_seed_start) + i for i in range(batch_size)]
    generators = _make_generators_from_seeds(prior, rollout_seeds, device_obj)
    initial = prior._resolve_env_initial_state_batch(env, batch_size=batch_size, state_dim=state_dim, device=device_obj, dtype=dtype)
    carry_state_t, state_t = prior._init_exact_scm_carry_and_visible_state(initial)
    state_noise_std = _to_float(env.get("state_noise_std", 0.0))
    checkpoints = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, int(n_steps)}
    rows = []
    finite_all = True
    max_obs_rms = 0.0
    max_state_rms = 0.0
    max_state_abs = 0.0
    reward_sum = torch.zeros((batch_size,), device=device_obj, dtype=dtype)
    for step_idx in range(1, int(n_steps) + 1):
        action = prior._stack_randn_with_generators(generators, (batch_size, action_dim), device_obj, dtype)
        if noise_dim > 0:
            noise_t = prior._stack_randn_with_generators(generators, (batch_size, noise_dim), device_obj, dtype)
        else:
            noise_t = torch.zeros((batch_size, 0), device=device_obj, dtype=dtype)
        state_noise_t = None
        if state_noise_std > 0.0:
            state_noise_t = prior._stack_randn_with_generators(generators, (batch_size, state_dim), device_obj, dtype)
        state_next, reward_raw, _, _ = _compute_preterminal_state_next_batch(
            prior=prior,
            env=env,
            carry_state_t=carry_state_t,
            state_t=state_t,
            action_next=action,
            noise_t=noise_t,
            state_noise_t=state_noise_t,
            rollout_generators=generators,
            device=device_obj,
        )
        reward_sum = reward_sum + reward_raw.reshape(batch_size)
        state_rms = torch.sqrt(state_next.square().mean(dim=1))
        obs_rms = torch.sqrt(state_next[:, :obs_dim].square().mean(dim=1))
        state_abs = state_next.abs().amax(dim=1)
        finite_step = bool(torch.isfinite(state_next).all().item()) and bool(torch.isfinite(reward_raw).all().item())
        finite_all = bool(finite_all and finite_step)
        max_obs_rms = max(max_obs_rms, float(obs_rms.max().item()))
        max_state_rms = max(max_state_rms, float(state_rms.max().item()))
        max_state_abs = max(max_state_abs, float(state_abs.max().item()))
        if step_idx in checkpoints:
            rows.append(
                {
                    "step": int(step_idx),
                    "finite": bool(finite_step),
                    "obs_rms": _tensor_stats(obs_rms),
                    "state_rms": _tensor_stats(state_rms),
                    "state_abs_max": _tensor_stats(state_abs),
                    "return_raw_sum": _tensor_stats(reward_sum),
                }
            )
        carry_state_t = prior._mix_exact_scm_carry_state(carry_state_t, state_next, env["alpha"])
        state_t = state_next
    return {
        "finite_all": bool(finite_all),
        "n_steps": int(n_steps),
        "rollout_count": int(rollout_count),
        "max_obs_rms_seen": float(max_obs_rms),
        "max_state_rms_seen": float(max_state_rms),
        "max_state_abs_seen": float(max_state_abs),
        "last_checkpoint": rows[-1] if rows else {},
        "sampled_env": {
            "state_full_rms_enabled": bool(env.get("state_full_rms_enabled", False)),
            "state_full_rms_target": float(env.get("state_full_rms_target", 1.0)),
            "state_output_scale": float(env.get("state_output_scale", 1.0)),
            "terminal_reset_enabled": bool(env.get("terminal_reset_enabled", False)),
            "effective_reward_scale": float(env.get("reward_scale", 1.0)),
            "init_std": float(frozen_h.get("init_std", 0.0)),
            "init_action_std": float(frozen_h.get("init_action_std", 0.0)),
        },
        "checkpoints": rows,
    }


def _summarize(rows: list[dict[str, Any]], thresholds: dict[str, float]) -> dict[str, Any]:
    metric_paths = {
        "obs_std": "deep.obs_value_std_all",
        "obs_rank": "deep.obs_cov.effective_rank_fraction",
        "obs_top": "deep.obs_cov.top_eigen_share",
        "reward_std": "deep.reward_effective.std",
        "return_std": "deep.discounted_effective_return.std",
        "state_sens": "deep.per_action_state_sensitivity.q50",
        "reward_sens": "deep.per_action_reward_sensitivity.q50",
        "G2_identity": "identity.G2_bias_corrected_identity",
        "G3_identity": "identity.G3_bias_corrected_identity",
        "G4_identity": "identity.G4_bias_corrected_identity",
        "G5_identity": "identity.G5_bias_corrected_identity",
    }
    by_case: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_case.setdefault(str(row["case"]), []).append(row)
    out: dict[str, Any] = {}
    for case, case_rows in sorted(by_case.items()):
        gates = {
            "obs_scale": sum(
                1
                for row in case_rows
                if (v := _finite(_get(row, "deep.obs_value_std_all"))) is not None
                and v >= thresholds.get("obs_std_min", math.inf)
            ),
            "reward_scale": sum(
                1
                for row in case_rows
                if (v := _finite(_get(row, "deep.reward_effective.std"))) is not None
                and v >= thresholds.get("reward_std_min", math.inf)
            ),
            "return_scale": sum(
                1
                for row in case_rows
                if (v := _finite(_get(row, "deep.discounted_effective_return.std"))) is not None
                and v >= thresholds.get("return_std_min", math.inf)
            ),
            "state_sensitivity": sum(
                1
                for row in case_rows
                if (v := _finite(_get(row, "deep.per_action_state_sensitivity.q50"))) is not None
                and v >= thresholds.get("state_action_sensitivity_min", math.inf)
            ),
            "reward_sensitivity": sum(
                1
                for row in case_rows
                if (v := _finite(_get(row, "deep.per_action_reward_sensitivity.q50"))) is not None
                and v >= thresholds.get("reward_action_sensitivity_min", math.inf)
            ),
            "obs_covariance": sum(
                1
                for row in case_rows
                if (rank := _finite(_get(row, "deep.obs_cov.effective_rank_fraction"))) is not None
                and (top := _finite(_get(row, "deep.obs_cov.top_eigen_share"))) is not None
                and rank >= thresholds.get("obs_rank_min", math.inf)
                and top <= thresholds.get("obs_top_eig_max", -math.inf)
            ),
            "finite_2048": sum(1 for row in case_rows if bool(_get(row, "stability.finite_all", False))),
        }
        out[case] = {
            "row_count": len(case_rows),
            "stats": {name: _stats([_get(row, path) for row in case_rows]) for name, path in metric_paths.items()},
            "gates": {key: {"count": int(val), "total": len(case_rows), "fraction": float(val / max(len(case_rows), 1))} for key, val in gates.items()},
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Paired structure intervention validation with full phase2 gates.")
    parser.add_argument("--source-rows-jsonl", default=DEFAULT_SOURCE_ROWS)
    parser.add_argument("--hard-gate-report", default=DEFAULT_HARD_GATE_REPORT)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--env-count", type=int, default=24)
    parser.add_argument("--sample-seed", type=int, default=20260429)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cases", default="baseline,initstd_x1p5,initstd_x2,initaction_x0p5,initstd_x1p5_initaction_x0p5")
    parser.add_argument("--deep-rollout-count", type=int, default=32)
    parser.add_argument("--deep-n-steps", type=int, default=128)
    parser.add_argument("--deep-per-action-stride", type=int, default=32)
    parser.add_argument("--identity-reference-state-count", type=int, default=12)
    parser.add_argument("--identity-continuation-repeats", type=int, default=12)
    parser.add_argument("--identity-anchor-step", type=int, default=128)
    parser.add_argument("--identity-n-steps", type=int, default=133)
    parser.add_argument("--stability-env-count", type=int, default=8)
    parser.add_argument("--stability-n-steps", type=int, default=2048)
    parser.add_argument("--stability-rollout-count", type=int, default=128)
    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    thresholds = _load_thresholds(args.hard_gate_report)
    all_rows = _enrich_baseline_rows(_load_jsonl(args.source_rows_jsonl), thresholds)
    selected = _select_rows(all_rows, count=int(args.env_count), seed=int(args.sample_seed))
    _add_selection_labels(selected, all_rows)
    cases = [piece.strip() for piece in str(args.cases).split(",") if piece.strip()]
    rows_path = output_dir / "rows.jsonl"
    stability_limit = max(0, int(args.stability_env_count))
    out_rows: list[dict[str, Any]] = []
    with rows_path.open("w", encoding="utf-8") as handle:
        for env_idx, source_row in enumerate(selected, start=1):
            frozen_h = _read_h(source_row)
            for case_idx, case in enumerate(cases, start=1):
                overrides = _case_overrides(frozen_h, case)
                print(
                    f"[structure-intervention] env {env_idx}/{len(selected)} "
                    f"case {case_idx}/{len(cases)} seed={source_row.get('frozen_h_seed')} case={case}",
                    flush=True,
                )
                deep = measure_prior_deep_features(
                    row=source_row,
                    case_name=case,
                    frozen_h_overrides=copy.deepcopy(overrides),
                    device=str(args.device),
                    train_env_seed=2020,
                    rollout_count=int(args.deep_rollout_count),
                    n_steps=int(args.deep_n_steps),
                    rollout_seed_start=9100000 + env_idx * 100000,
                    action_delta=0.1,
                    per_action_stride=int(args.deep_per_action_stride),
                )
                identity_raw = _run_identity(
                    source_row=source_row,
                    overrides=overrides,
                    device=str(args.device),
                    reference_state_count=int(args.identity_reference_state_count),
                    continuation_repeats=int(args.identity_continuation_repeats),
                    anchor_step=int(args.identity_anchor_step),
                    n_steps=int(args.identity_n_steps),
                )
                identity = {
                    "identity_score": _get(identity_raw, "aggregate.identity_score"),
                    **_identity_curve(identity_raw),
                }
                stability = None
                if env_idx <= stability_limit:
                    stability = run_no_terminal_stability(
                        source_row=source_row,
                        overrides=overrides,
                        device=str(args.device),
                        n_steps=int(args.stability_n_steps),
                        rollout_count=int(args.stability_rollout_count),
                        rollout_seed_start=88000000 + env_idx * 100000,
                    )
                out = {
                    "case": case,
                    "env_index": env_idx,
                    "case_index": case_idx,
                    "frozen_h_seed": source_row.get("frozen_h_seed"),
                    "full_frozen_h_json": source_row.get("full_frozen_h_json"),
                    "selection_labels": source_row.get("selection_labels"),
                    "source_baseline": {
                        "amp_geom_ratio": source_row.get("amp_geom_ratio"),
                        "obs_ratio": source_row.get("obs_ratio"),
                        "reward_ratio": source_row.get("reward_ratio"),
                        "return_ratio": source_row.get("return_ratio"),
                        "param_init_std": source_row.get("param_init_std"),
                        "param_init_action_std": source_row.get("param_init_action_std"),
                        "param__constrained_obs_u": source_row.get("param__constrained_obs_u"),
                    },
                    "overrides": overrides,
                    "deep": deep,
                    "identity": identity,
                    "stability": stability,
                }
                out = _json_safe(out)
                out_rows.append(out)
                handle.write(json.dumps(out, sort_keys=True) + "\n")
                handle.flush()
    selected_path = output_dir / "selected_envs.csv"
    with selected_path.open("w", newline="", encoding="utf-8") as handle:
        fields = sorted({key for row in selected for key in row.keys() if key != "selection_labels"})
        fields.append("selection_labels_json")
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in selected:
            payload = {key: row.get(key) for key in fields if key != "selection_labels_json"}
            payload["selection_labels_json"] = json.dumps(row.get("selection_labels"), sort_keys=True)
            writer.writerow(payload)
    summary = {
        "audit_entry": "phase2_prior_structure_intervention_validation",
        "inputs": {
            "source_rows_jsonl": str(Path(args.source_rows_jsonl).expanduser().resolve()),
            "hard_gate_report": str(Path(args.hard_gate_report).expanduser().resolve()),
            "selected_env_count": int(len(selected)),
            "cases": cases,
            "device": str(args.device),
            "deep_rollout_count": int(args.deep_rollout_count),
            "deep_n_steps": int(args.deep_n_steps),
            "identity_reference_state_count": int(args.identity_reference_state_count),
            "identity_continuation_repeats": int(args.identity_continuation_repeats),
            "stability_env_count": int(args.stability_env_count),
            "stability_n_steps": int(args.stability_n_steps),
        },
        "selected_envs_csv": str(selected_path),
        "rows_jsonl": str(rows_path),
        "case_summary": _summarize(out_rows, thresholds),
        "interpretation_guardrail": (
            "This validation tests candidate structure transforms only. Passing local gates is not sufficient; "
            "candidate defaults require fixed-group trainability and held-out Gym/Prior checks."
        ),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(_json_safe(summary), sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(_json_safe(summary), sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
