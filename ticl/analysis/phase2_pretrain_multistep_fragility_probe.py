import argparse
import copy
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ticl.analysis.phase2_gym_prior_pack_training_runner import (  # noqa: E402
    _build_algo,
    _json_safe,
    _load_fixed_frozen_h_list_from_csv,
    _reward_component_arrays_from_pack,
)
from ticl.analysis.phase2_pretrain_reward_basin_action_distribution_probe import (  # noqa: E402
    _env_meta_from_vec_env,
    _parse_action_seeds,
    _stats,
    _write_csv,
    _write_jsonl,
)
from ticl.config_utils import str2bool  # noqa: E402
from ticl.model_configs import get_model_default_config  # noqa: E402
from ticl.rlpfn_maintained_path import (  # noqa: E402
    resolve_rlpfn_token_layout,
    validate_rlpfn_maintained_path_config,
)


DEFAULT_SELECTED_CSV = (
    "/home/chen/RLPFN/artifacts/"
    "phase2_reserve_probe_selected_ratio10_q90p1_posp15_n64_seed9090_0430/selected_envs.csv"
)
DEFAULT_DELTA_CSV = (
    "/home/chen/RLPFN/artifacts/"
    "phase2_delta_worst_env_analysis_0430/topology_ratio10_reserve_n64_delta_rank_with_seed.csv"
)
DEFAULT_OUTPUT_DIR = (
    "/home/chen/RLPFN/artifacts/"
    "phase2_pretrain_multistep_fragility_probe_0501"
)


def _finite(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _parse_ints(value: str) -> list[int]:
    return [int(part.strip()) for part in str(value).split(",") if part.strip()]


def _parse_horizons(value: str) -> list[int]:
    horizons = sorted({int(part.strip()) for part in str(value).split(",") if part.strip()})
    if not horizons or min(horizons) <= 0:
        raise ValueError(f"Expected positive comma-separated horizons, got {value!r}")
    return horizons


def _parse_modes(value: str) -> list[str]:
    modes = [part.strip() for part in str(value).split(",") if part.strip()]
    if "canonical" not in modes:
        modes = ["canonical", *modes]
    return modes


def _decode_float_suffix(text: str) -> float:
    suffix = str(text).strip()
    sign = 1.0
    if suffix.startswith("m"):
        sign = -1.0
        suffix = suffix[1:]
    elif suffix.startswith("p"):
        suffix = suffix[1:]
    return float(sign * float(suffix.replace("p", ".")))


def _load_delta_rows(delta_csv: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(delta_csv).expanduser().open("r", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            row = dict(raw)
            for key, value in list(row.items()):
                val = _finite(value)
                row[key] = val if val is not None else value
            row["env_idx"] = int(float(row["env_idx"]))
            rows.append(row)
    return rows


def _default_control_envs(delta_csv: str | Path, hard_envs: list[int], count: int) -> list[int]:
    hard_set = {int(v) for v in hard_envs}
    rows = [row for row in _load_delta_rows(delta_csv) if int(row["env_idx"]) not in hard_set]
    rows.sort(key=lambda row: float(row.get("delta_env_sum") or 0.0), reverse=True)
    return [int(row["env_idx"]) for row in rows[: int(count)]]


def _transform_base_action(mode: str, base: np.ndarray) -> np.ndarray:
    mode = str(mode)
    if mode == "canonical":
        return base.astype(np.float32, copy=True)
    if mode.startswith("scale_"):
        scale = _decode_float_suffix(mode.split("_", 1)[1])
        return (base * float(scale)).astype(np.float32, copy=False)
    if mode.startswith("shift_"):
        shift = _decode_float_suffix(mode.split("_", 1)[1])
        return (base + float(shift)).astype(np.float32, copy=False)
    if mode == "zero":
        return np.zeros_like(base, dtype=np.float32)
    raise ValueError(f"Unsupported action perturbation mode={mode!r}")


def _component_stats_for_prefix(values: np.ndarray, horizon: int) -> dict[str, Any]:
    prefix = np.asarray(values[: int(horizon)], dtype=np.float64)
    return {
        "sum": float(np.sum(prefix)),
        "mean": float(np.mean(prefix)),
        "q10": float(np.quantile(prefix, 0.10)),
        "q50": float(np.quantile(prefix, 0.50)),
        "q90": float(np.quantile(prefix, 0.90)),
        "positive_fraction": float(np.mean(prefix > 0.0)),
    }


def run_probe(
    *,
    output_dir: str,
    device: str,
    selected_csv: str,
    selected_rule: str,
    delta_csv: str,
    n_envs: int,
    hard_env_indices: str,
    control_env_indices: str,
    control_count: int,
    horizons: str,
    action_modes: str,
    action_seeds: str,
    reset_seeds: str,
    base_seed: int,
    sb3_reward_normalization_enabled: bool,
) -> dict[str, Any]:
    out_dir = Path(output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    hard_envs = _parse_ints(hard_env_indices)
    if str(control_env_indices).strip():
        control_envs = _parse_ints(control_env_indices)
    else:
        control_envs = _default_control_envs(delta_csv, hard_envs, int(control_count))
    target_envs = [*hard_envs, *[idx for idx in control_envs if idx not in set(hard_envs)]]
    target_set = {int(v) for v in target_envs}
    horizon_values = _parse_horizons(horizons)
    max_horizon = int(max(horizon_values))
    mode_values = _parse_modes(action_modes)
    action_seed_values = _parse_action_seeds(action_seeds)
    reset_seed_values = _parse_action_seeds(reset_seeds)

    cfg = copy.deepcopy(get_model_default_config("rlpfn"))
    validate_rlpfn_maintained_path_config(cfg)
    env_cfg = copy.deepcopy(cfg["prior"]["environment"])
    num_features = int(cfg["prior"]["num_features"])
    layout = resolve_rlpfn_token_layout(env_cfg, num_features=num_features)
    action_slot_dim = int(layout["action_slot_dim"])

    all_h_list, all_h_paths, all_env_seeds = _load_fixed_frozen_h_list_from_csv(
        selected_csv,
        rule=str(selected_rule),
        limit=int(n_envs),
    )
    if max(target_set) >= len(all_h_list):
        raise ValueError(f"target env index out of range for loaded n_envs={len(all_h_list)}: {sorted(target_set)}")

    source_indices = [int(idx) for idx in target_envs]
    h_list = [copy.deepcopy(all_h_list[idx]) for idx in source_indices]
    h_paths = [str(all_h_paths[idx]) for idx in source_indices]
    source_env_seeds = [int(all_env_seeds[idx]) for idx in source_indices]
    labels = ["hard" if idx in set(hard_envs) else "control" for idx in source_indices]
    n_probe_envs = int(len(h_list))

    algo, vec_env = _build_algo(
        cfg=cfg,
        env_cfg=env_cfg,
        frozen_h=None,
        frozen_h_list=h_list,
        frozen_h_list_env_seeds=source_env_seeds,
        prior_mode="fixed_frozen_list",
        device_obj=torch.device(device),
        num_features=int(num_features),
        n_envs=int(n_probe_envs),
        n_steps=int(max_horizon),
        build_seed=int(base_seed),
        sb3_reward_normalization_enabled=bool(sb3_reward_normalization_enabled),
        sb3_observation_normalization_enabled=False,
        sb3_observation_normalization_clip=10.0,
        sb3_observation_normalization_epsilon=1e-8,
        gain_min=0.0,
        topology_state_gain_min=0.0,
        topology_action_gain_min=0.0,
        topology_state_to_action_ratio_max=0.0,
        topology_max_attempts=4096,
        fixed_env_group_across_updates=True,
    )

    rows: list[dict[str, Any]] = []
    try:
        meta = _env_meta_from_vec_env(vec_env, h_list, n_probe_envs)
        for reset_seed in reset_seed_values:
            trial_env_seeds = [
                int(reset_seed) + 1009 * int(source_idx) + 17 * local_idx
                for local_idx, source_idx in enumerate(source_indices)
            ]
            for action_seed in action_seed_values:
                rng = np.random.default_rng(int(action_seed))
                base_actions = rng.normal(
                    0.0,
                    1.0,
                    size=(int(max_horizon), int(n_probe_envs), int(action_slot_dim)),
                ).astype(np.float32)
                for mode in mode_values:
                    vec_env._full_reset_batch(seeds=trial_env_seeds)
                    rewards_by_env = [[] for _ in range(n_probe_envs)]
                    total_by_env = [[] for _ in range(n_probe_envs)]
                    actions_by_env = [[] for _ in range(n_probe_envs)]
                    masks_by_env = [[] for _ in range(n_probe_envs)]
                    dones_by_env = [[] for _ in range(n_probe_envs)]
                    obs_norm_by_env = [[] for _ in range(n_probe_envs)]
                    for step_idx in range(int(max_horizon)):
                        masks = vec_env.action_masks().astype(np.float32, copy=False)
                        action = _transform_base_action(mode, base_actions[int(step_idx)])
                        action = (action * masks).astype(np.float32, copy=False)
                        vec_env.step_async(action)
                        obs, reward, done, _infos = vec_env.step_wait()
                        obs_arr = np.asarray(obs, dtype=np.float32)
                        reward_arr = np.asarray(reward, dtype=np.float64)
                        done_arr = np.asarray(done, dtype=bool)
                        for local_idx in range(n_probe_envs):
                            action_dims = meta.get("action_dims", [action_slot_dim] * n_probe_envs)
                            action_dim = int(action_dims[local_idx])
                            obs_dims = meta.get("obs_dims", [obs_arr.shape[1]] * n_probe_envs)
                            obs_dim = int(min(int(obs_dims[local_idx]), int(obs_arr.shape[1])))
                            total_by_env[local_idx].append(float(reward_arr[local_idx]))
                            actions_by_env[local_idx].append(
                                action[local_idx, :action_dim].astype(np.float64, copy=True)
                            )
                            masks_by_env[local_idx].append(
                                masks[local_idx, :action_dim].astype(np.float64, copy=True)
                            )
                            dones_by_env[local_idx].append(bool(done_arr[local_idx]))
                            obs_norm_by_env[local_idx].append(
                                float(np.linalg.norm(obs_arr[local_idx, :obs_dim])) if obs_dim > 0 else 0.0
                            )
                    for local_idx, source_idx in enumerate(source_indices):
                        total_np = np.asarray(total_by_env[local_idx], dtype=np.float64)
                        action_np = np.asarray(actions_by_env[local_idx], dtype=np.float64)
                        mask_np = np.asarray(masks_by_env[local_idx], dtype=np.float64)
                        components, component_meta = _reward_component_arrays_from_pack(
                            arm="prior",
                            reward=total_np,
                            active_actions=action_np,
                            active_action_masks=mask_np,
                            meta=meta,
                            env_idx=int(local_idx),
                        )
                        reward_env = np.asarray(
                            components.get("reward_env", total_np),
                            dtype=np.float64,
                        )
                        for horizon in horizon_values:
                            stats = _component_stats_for_prefix(reward_env, int(horizon))
                            total_stats = _component_stats_for_prefix(total_np, int(horizon))
                            action_prefix = action_np[: int(horizon)]
                            row = {
                                "source_env_idx": int(source_idx),
                                "local_env_idx": int(local_idx),
                                "group_label": labels[local_idx],
                                "source_env_seed": int(source_env_seeds[local_idx]),
                                "full_frozen_h_json": h_paths[local_idx],
                                "reset_seed": int(reset_seed),
                                "trial_env_seed": int(trial_env_seeds[local_idx]),
                                "action_seed": int(action_seed),
                                "action_mode": str(mode),
                                "horizon": int(horizon),
                                "component_contract_exact": bool(
                                    component_meta.get("contract_exact_for_current_none_transform_runs", False)
                                ),
                                "done_count": int(np.sum(dones_by_env[local_idx][: int(horizon)])),
                                "reward_env_sum": stats["sum"],
                                "reward_env_mean": stats["mean"],
                                "reward_env_q10": stats["q10"],
                                "reward_env_q50": stats["q50"],
                                "reward_env_q90": stats["q90"],
                                "reward_env_positive_fraction": stats["positive_fraction"],
                                "reward_total_sum": total_stats["sum"],
                                "action_norm_mean": float(
                                    np.mean(np.linalg.norm(action_prefix, axis=1))
                                )
                                if action_prefix.size
                                else 0.0,
                                "action_linf_mean": float(np.mean(np.max(np.abs(action_prefix), axis=1)))
                                if action_prefix.size
                                else 0.0,
                                "obs_norm_mean": float(np.mean(obs_norm_by_env[local_idx][: int(horizon)])),
                                "obs_norm_max": float(np.max(obs_norm_by_env[local_idx][: int(horizon)])),
                            }
                            rows.append(row)
    finally:
        vec_env.close()
        del algo
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    keyed: dict[tuple[int, int, int, int], dict[str, dict[str, Any]]] = {}
    for row in rows:
        key = (
            int(row["source_env_idx"]),
            int(row["reset_seed"]),
            int(row["action_seed"]),
            int(row["horizon"]),
        )
        keyed.setdefault(key, {})[str(row["action_mode"])] = row
    pair_rows: list[dict[str, Any]] = []
    for (source_idx, reset_seed, action_seed, horizon), values in sorted(keyed.items()):
        canonical = values.get("canonical")
        if canonical is None:
            continue
        for mode, row in sorted(values.items()):
            if mode == "canonical":
                continue
            pair_rows.append(
                {
                    "source_env_idx": int(source_idx),
                    "group_label": str(row["group_label"]),
                    "reset_seed": int(reset_seed),
                    "action_seed": int(action_seed),
                    "action_mode": str(mode),
                    "horizon": int(horizon),
                    "perturb_minus_canonical_reward_env_sum": float(
                        row["reward_env_sum"] - canonical["reward_env_sum"]
                    ),
                    "perturb_minus_canonical_reward_env_q90": float(
                        row["reward_env_q90"] - canonical["reward_env_q90"]
                    ),
                    "perturb_minus_canonical_reward_env_positive_fraction": float(
                        row["reward_env_positive_fraction"]
                        - canonical["reward_env_positive_fraction"]
                    ),
                    "perturb_minus_canonical_done_count": int(row["done_count"] - canonical["done_count"]),
                    "canonical_reward_env_sum": float(canonical["reward_env_sum"]),
                    "perturb_reward_env_sum": float(row["reward_env_sum"]),
                    "canonical_reward_env_q90": float(canonical["reward_env_q90"]),
                    "perturb_reward_env_q90": float(row["reward_env_q90"]),
                    "canonical_positive_fraction": float(canonical["reward_env_positive_fraction"]),
                    "perturb_positive_fraction": float(row["reward_env_positive_fraction"]),
                }
            )

    env_summary_rows: list[dict[str, Any]] = []
    for source_idx in source_indices:
        env_pairs = [row for row in pair_rows if int(row["source_env_idx"]) == int(source_idx)]
        h20_pairs = [row for row in env_pairs if int(row["horizon"]) == 20]
        env_rows = [row for row in rows if int(row["source_env_idx"]) == int(source_idx)]
        canonical_h20 = [
            row for row in env_rows if row["action_mode"] == "canonical" and int(row["horizon"]) == 20
        ]
        env_summary_rows.append(
            {
                "source_env_idx": int(source_idx),
                "group_label": "hard" if int(source_idx) in set(hard_envs) else "control",
                "source_env_seed": int(source_env_seeds[source_indices.index(source_idx)]),
                "canonical_h20_reward_env_sum": _stats([row["reward_env_sum"] for row in canonical_h20]),
                "canonical_h20_reward_env_q90": _stats([row["reward_env_q90"] for row in canonical_h20]),
                "canonical_h20_positive_fraction": _stats(
                    [row["reward_env_positive_fraction"] for row in canonical_h20]
                ),
                "h20_min_perturb_minus_canonical_reward_env_sum": (
                    None
                    if not h20_pairs
                    else float(min(row["perturb_minus_canonical_reward_env_sum"] for row in h20_pairs))
                ),
                "h20_mean_perturb_minus_canonical_reward_env_sum": _stats(
                    [row["perturb_minus_canonical_reward_env_sum"] for row in h20_pairs]
                ),
                "h20_min_perturb_minus_canonical_reward_env_q90": (
                    None
                    if not h20_pairs
                    else float(min(row["perturb_minus_canonical_reward_env_q90"] for row in h20_pairs))
                ),
                "h20_min_perturb_minus_canonical_positive_fraction": (
                    None
                    if not h20_pairs
                    else float(min(row["perturb_minus_canonical_reward_env_positive_fraction"] for row in h20_pairs))
                ),
                "h20_negative_sum_fraction": float(
                    np.mean([row["perturb_minus_canonical_reward_env_sum"] < 0.0 for row in h20_pairs])
                )
                if h20_pairs
                else None,
                "all_horizon_min_perturb_minus_canonical_reward_env_sum": (
                    None
                    if not env_pairs
                    else float(min(row["perturb_minus_canonical_reward_env_sum"] for row in env_pairs))
                ),
            }
        )

    group_summary: dict[str, dict[str, Any]] = {}
    for label in ("hard", "control"):
        subset = [row for row in env_summary_rows if row["group_label"] == label]
        group_summary[label] = {
            "count": int(len(subset)),
            "h20_min_delta_sum": _stats(
                [row["h20_min_perturb_minus_canonical_reward_env_sum"] for row in subset]
            ),
            "h20_mean_delta_sum_mean": _stats(
                [
                    (row["h20_mean_perturb_minus_canonical_reward_env_sum"] or {}).get("mean")
                    for row in subset
                ]
            ),
            "h20_negative_sum_fraction": _stats([row["h20_negative_sum_fraction"] for row in subset]),
            "canonical_h20_reward_env_sum_mean": _stats(
                [
                    (row["canonical_h20_reward_env_sum"] or {}).get("mean")
                    for row in subset
                ]
            ),
        }

    hard_min_mean = group_summary["hard"]["h20_min_delta_sum"]["mean"]
    control_min_mean = group_summary["control"]["h20_min_delta_sum"]["mean"]
    distinguishes = (
        hard_min_mean is not None
        and control_min_mean is not None
        and float(hard_min_mean) < float(control_min_mean)
    )
    summary = {
        "audit_entry": "phase2_pretrain_multistep_fragility_probe",
        "contract": (
            "No training/update. Fixed frozen_h envs are reset with identical seeds across action modes. "
            "Perturbation modes share the exact same base Gaussian action tensor as canonical, then apply "
            "small scale/shift transforms before masking. This tests train-before multi-step closed-loop "
            "fragility, not PPO-induced policy drift."
        ),
        "output_dir": str(out_dir),
        "selected_csv": str(selected_csv),
        "selected_rule": str(selected_rule),
        "delta_csv": str(delta_csv),
        "hard_env_indices": [int(v) for v in hard_envs],
        "control_env_indices": [int(v) for v in control_envs],
        "source_indices": [int(v) for v in source_indices],
        "horizons": [int(v) for v in horizon_values],
        "action_modes": [str(v) for v in mode_values],
        "action_seeds": [int(v) for v in action_seed_values],
        "reset_seeds": [int(v) for v in reset_seed_values],
        "rows_csv": str(out_dir / "rows.csv"),
        "pairs_csv": str(out_dir / "pairs.csv"),
        "env_summary_csv": str(out_dir / "env_summary.csv"),
        "group_summary": group_summary,
        "distinguishes_hard_from_control_by_h20_min_delta_sum_mean": bool(distinguishes),
        "interpretation_guardrail": (
            "Only consider this a candidate gate if hard envs are clearly more fragile than controls "
            "and an independent fixed-group smoke improves without diversity collapse."
        ),
    }
    _write_jsonl(out_dir / "rows.jsonl", rows)
    _write_csv(out_dir / "rows.csv", rows)
    _write_csv(out_dir / "pairs.csv", pair_rows)
    _write_csv(out_dir / "env_summary.csv", env_summary_rows)
    (out_dir / "summary.json").write_text(
        json.dumps(_json_safe(summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Training-before multi-step fragility probe.")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--selected-csv", type=str, default=DEFAULT_SELECTED_CSV)
    parser.add_argument("--selected-rule", type=str, default="ratio10_q90p1_posp15")
    parser.add_argument("--delta-csv", type=str, default=DEFAULT_DELTA_CSV)
    parser.add_argument("--n-envs", type=int, default=64)
    parser.add_argument("--hard-env-indices", type=str, default="34,9,36")
    parser.add_argument("--control-env-indices", type=str, default="")
    parser.add_argument("--control-count", type=int, default=6)
    parser.add_argument("--horizons", type=str, default="5,10,20")
    parser.add_argument(
        "--action-modes",
        type=str,
        default="canonical,scale_0p9,scale_1p1,shift_p0p1,shift_m0p1",
    )
    parser.add_argument("--action-seeds", type=str, default="4040,4041,4042,4043")
    parser.add_argument("--reset-seeds", type=str, default="5050,5051,5052,5053")
    parser.add_argument("--base-seed", type=int, default=4040)
    parser.add_argument("--sb3-reward-normalization-enabled", type=str2bool, default=True)
    args = parser.parse_args()
    summary = run_probe(
        output_dir=args.output_dir,
        device=args.device,
        selected_csv=args.selected_csv,
        selected_rule=args.selected_rule,
        delta_csv=args.delta_csv,
        n_envs=int(args.n_envs),
        hard_env_indices=str(args.hard_env_indices),
        control_env_indices=str(args.control_env_indices),
        control_count=int(args.control_count),
        horizons=str(args.horizons),
        action_modes=str(args.action_modes),
        action_seeds=str(args.action_seeds),
        reset_seeds=str(args.reset_seeds),
        base_seed=int(args.base_seed),
        sb3_reward_normalization_enabled=bool(args.sb3_reward_normalization_enabled),
    )
    print(
        json.dumps(
            _json_safe(
                {
                    "summary_json": str(Path(args.output_dir).expanduser().resolve() / "summary.json"),
                    "rows_csv": summary["rows_csv"],
                    "pairs_csv": summary["pairs_csv"],
                    "env_summary_csv": summary["env_summary_csv"],
                    "group_summary": summary["group_summary"],
                    "distinguishes": summary["distinguishes_hard_from_control_by_h20_min_delta_sum_mean"],
                }
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
