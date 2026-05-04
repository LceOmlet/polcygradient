import argparse
import copy
import csv
import hashlib
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
    _add_prior_reward_component_meta_from_h_list,
    _build_algo,
    _json_safe,
    _load_fixed_frozen_h_list_from_csv,
    _reward_component_arrays_from_pack,
    _reward_component_stats,
    _stable_digest,
    _tensor_int_vector,
    _tensor_vector,
)
from ticl.config_utils import str2bool  # noqa: E402
from ticl.model_configs import get_model_default_config  # noqa: E402
from ticl.priors.environment_prior import EnvironmentPrior  # noqa: E402
from ticl.rlpfn_maintained_path import resolve_rlpfn_token_layout, validate_rlpfn_maintained_path_config  # noqa: E402


DEFAULT_OUTPUT_DIR = (
    "/home/chen/RLPFN/artifacts/"
    "phase2_pretrain_reward_basin_action_distribution_probe_0429"
)
DEFAULT_SELECTED_CSV = (
    "/home/chen/RLPFN/artifacts/"
    "phase2_prior_balanced_topology_dim_noisebudget_validation_0429/selected_envs.csv"
)
DEFAULT_POPULATION_CSV = (
    "/home/chen/RLPFN/artifacts/"
    "phase2_prior_generation_rule_audit_0429/enriched_population.csv"
)


def _finite(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _quantile(values: list[float], frac: float) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        raise ValueError("Cannot compute quantile on an empty list.")
    return float(np.quantile(arr, float(frac)))


def _label(value: Any, lo: float, hi: float) -> str:
    val = _finite(value)
    if val is None:
        return "missing"
    if val <= lo:
        return "low"
    if val >= hi:
        return "high"
    return "mid"


def _stats(values: list[Any]) -> dict[str, Any]:
    arr = np.asarray([float(v) for v in values if v is not None and math.isfinite(float(v))], dtype=np.float64)
    if arr.size == 0:
        return {"count": 0, "mean": None, "q10": None, "q50": None, "q90": None, "min": None, "max": None}
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "q10": float(np.quantile(arr, 0.10)),
        "q50": float(np.quantile(arr, 0.50)),
        "q90": float(np.quantile(arr, 0.90)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def _parse_action_modes(value: str) -> list[str]:
    return [part.strip() for part in str(value).split(",") if part.strip()]


def _parse_action_seeds(value: str) -> list[int]:
    return [int(part.strip()) for part in str(value).split(",") if part.strip()]


def _load_population_rows(csv_path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(csv_path).expanduser().open("r", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            row: dict[str, Any] = {}
            for key, value in raw.items():
                val = _finite(value)
                row[key] = val if val is not None else value
            seed = _finite(row.get("seed"))
            if seed is not None:
                row["seed"] = int(seed)
            rows.append(row)
    return rows


def _add_population_tertile_labels(rows: list[dict[str, Any]]) -> None:
    fields = [
        "reward_state_gain",
        "reward_action_gain",
        "reward_noise_gain",
        "reward_action_to_noise_gain_ratio",
        "reward_state_to_noise_gain_ratio",
        "_constrained_obs_u",
        "noise_budget_frac",
        "zero_budget_frac",
        "obs_frac",
        "state_dim",
        "action_dim",
        "noise_dim",
        "zero_pad_dim",
        "noise_std",
        "G5",
        "obs_rank",
        "reward_std",
        "return_std",
    ]
    for row in rows:
        noise_gain = max(float(row.get("reward_noise_gain") or 0.0), 1e-12)
        row["reward_action_to_noise_gain_ratio"] = float((row.get("reward_action_gain") or 0.0) / noise_gain)
        row["reward_state_to_noise_gain_ratio"] = float((row.get("reward_state_gain") or 0.0) / noise_gain)
    cuts: dict[str, tuple[float, float]] = {}
    for field in fields:
        values = [_finite(row.get(field)) for row in rows]
        values = [v for v in values if v is not None]
        if values:
            cuts[field] = (_quantile(values, 1.0 / 3.0), _quantile(values, 2.0 / 3.0))
    for row in rows:
        row["rule_labels"] = {
            field: _label(row.get(field), lo, hi)
            for field, (lo, hi) in cuts.items()
        }


def _lab(row: dict[str, Any], field: str) -> str:
    return str((row.get("rule_labels") or {}).get(field, "missing"))


def _population_rule_predicate(rule: str):
    rule = str(rule)
    if rule == "population_random_control":
        return lambda row: True
    if rule == "balanced_topology_mid_noise":
        return lambda row: (
            _lab(row, "reward_noise_gain") == "mid"
            and _lab(row, "reward_state_gain") != "low"
            and _lab(row, "reward_action_gain") != "high"
        )
    if rule == "balanced_topology_obsu_nonhigh":
        return lambda row: (
            _lab(row, "reward_noise_gain") == "mid"
            and _lab(row, "reward_state_gain") != "low"
            and _lab(row, "reward_action_gain") != "high"
            and _lab(row, "_constrained_obs_u") != "high"
        )
    if rule == "balanced_topology_noisebudget_mid":
        return lambda row: (
            _lab(row, "reward_noise_gain") == "mid"
            and _lab(row, "reward_state_gain") != "low"
            and _lab(row, "reward_action_gain") != "high"
            and _lab(row, "noise_budget_frac") == "mid"
        )
    if rule == "strict_balanced_topology_nonextreme_dim_mid_noisebudget":
        return lambda row: (
            _lab(row, "reward_noise_gain") == "mid"
            and _lab(row, "reward_state_gain") != "low"
            and _lab(row, "reward_action_gain") != "high"
            and _lab(row, "_constrained_obs_u") != "high"
            and _lab(row, "noise_budget_frac") == "mid"
        )
    if rule == "relaxed_midnoise_state_notlow_obsu_nonhigh_nb_nonhigh":
        return lambda row: (
            _lab(row, "reward_noise_gain") == "mid"
            and _lab(row, "reward_state_gain") != "low"
            and _lab(row, "_constrained_obs_u") != "high"
            and _lab(row, "noise_budget_frac") != "high"
        )
    raise ValueError(f"Unsupported population rule={rule!r}")


def _load_population_frozen_h_list_from_csv(
    csv_path: str | Path,
    *,
    rule: str,
    limit: int,
) -> tuple[list[dict[str, Any]], list[str], list[int], dict[int, dict[str, Any]]]:
    rows = _load_population_rows(csv_path)
    _add_population_tertile_labels(rows)
    predicate = _population_rule_predicate(str(rule))
    selected = [row for row in rows if predicate(row)]
    selected.sort(key=lambda row: int(row.get("seed", 0)))
    if int(limit) > 0:
        selected = selected[: int(limit)]
    h_list: list[dict[str, Any]] = []
    h_paths: list[str] = []
    env_seeds: list[int] = []
    full_gate_by_seed: dict[int, dict[str, Any]] = {}
    for row in selected:
        path = Path(str(row["full_frozen_h_json"])).expanduser()
        with path.open("r", encoding="utf-8") as handle:
            h = json.load(handle)
        seed = int(row["seed"])
        h_list.append(h)
        h_paths.append(str(path))
        env_seeds.append(seed)
        full_gate_by_seed[seed] = row
    return h_list, h_paths, env_seeds, full_gate_by_seed


def _mode_action(
    *,
    mode: str,
    rng: np.random.Generator,
    step_idx: int,
    n_envs: int,
    action_slot_dim: int,
    masks: np.ndarray,
    open_loop_cache: dict[str, np.ndarray],
) -> np.ndarray:
    mode = str(mode)
    masks_f = np.asarray(masks, dtype=np.float32)
    if mode == "zero":
        action = np.zeros((int(n_envs), int(action_slot_dim)), dtype=np.float32)
    elif mode.startswith("iid_gaussian_"):
        scale = float(mode.rsplit("_", 1)[1].replace("p", "."))
        action = rng.normal(0.0, scale, size=(int(n_envs), int(action_slot_dim))).astype(np.float32)
    elif mode.startswith("openloop_gaussian_"):
        scale = float(mode.rsplit("_", 1)[1].replace("p", "."))
        if mode not in open_loop_cache:
            open_loop_cache[mode] = rng.normal(0.0, scale, size=(int(n_envs), int(action_slot_dim))).astype(np.float32)
        action = open_loop_cache[mode].copy()
    elif mode.startswith("constant_"):
        value = float(mode.rsplit("_", 1)[1].replace("m", "-").replace("p", "."))
        action = np.full((int(n_envs), int(action_slot_dim)), value, dtype=np.float32)
    else:
        raise ValueError(f"Unsupported action mode={mode!r}")
    return (action * masks_f).astype(np.float32, copy=False)


def _env_meta_from_vec_env(vec_env, h_list: list[dict[str, Any]], n_envs: int) -> dict[str, Any]:
    meta: dict[str, Any] = {}
    _add_prior_reward_component_meta_from_h_list(meta, h_list, int(n_envs))
    if isinstance(getattr(vec_env, "_env", None), dict):
        env_batch = vec_env._env
        for key in (
            "obs_dim_per_sample",
            "obs_slot_dim_per_sample",
            "action_dim_per_sample",
            "action_slot_dim_per_sample",
            "state_dim_per_sample",
        ):
            values = _tensor_int_vector(env_batch, key)
            if values is not None:
                public_key = {
                    "obs_dim_per_sample": "obs_dims",
                    "obs_slot_dim_per_sample": "obs_slot_dims",
                    "action_dim_per_sample": "action_dims",
                    "action_slot_dim_per_sample": "action_slot_dims",
                    "state_dim_per_sample": "state_dims",
                }[key]
                meta[public_key] = values
        for key in (
            "reward_state_input_gain_fraction",
            "reward_action_input_gain_fraction",
            "reward_noise_input_gain_fraction",
            "reward_state_to_noise_gain_ratio",
            "reward_state_to_action_gain_ratio",
        ):
            values = _tensor_vector(env_batch, key)
            if values is not None:
                meta[key] = values
    return meta


def _load_full_gate_rows(csv_path: str | Path, rule: str) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    with Path(csv_path).expanduser().open("r", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            if str(raw.get("rule", "")).strip() != str(rule):
                continue
            seed = int(float(raw["seed"]))
            out[seed] = raw
    return out


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(_json_safe(row), sort_keys=True) + "\n")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(_json_safe(row))


def run_probe(
    *,
    output_dir: str,
    device: str,
    selected_csv: str,
    population_csv: str,
    input_kind: str,
    rule: str,
    limit: int,
    n_steps: int,
    seed: int,
    action_modes: str,
    action_seeds: str,
    sb3_observation_normalization_enabled: bool,
) -> dict[str, Any]:
    out_dir = Path(output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    action_mode_list = _parse_action_modes(action_modes)
    action_seed_list = _parse_action_seeds(action_seeds)

    cfg = copy.deepcopy(get_model_default_config("rlpfn"))
    validate_rlpfn_maintained_path_config(cfg)
    env_cfg = copy.deepcopy(cfg["prior"]["environment"])
    num_features = int(cfg["prior"]["num_features"])
    layout = resolve_rlpfn_token_layout(env_cfg, num_features=num_features)
    action_slot_dim = int(layout["action_slot_dim"])
    obs_slot_dim = int(layout["obs_slot_dim"])
    input_kind = str(input_kind)
    if input_kind == "selected":
        h_list, h_paths, env_seeds = _load_fixed_frozen_h_list_from_csv(
            selected_csv,
            rule=str(rule),
            limit=int(limit),
        )
        full_gate_by_seed = _load_full_gate_rows(selected_csv, rule)
        source_csv = str(selected_csv)
    elif input_kind == "population":
        h_list, h_paths, env_seeds, full_gate_by_seed = _load_population_frozen_h_list_from_csv(
            population_csv,
            rule=str(rule),
            limit=int(limit),
        )
        source_csv = str(population_csv)
    else:
        raise ValueError(f"Unsupported input_kind={input_kind!r}; expected 'selected' or 'population'.")
    n_envs = int(len(h_list))
    if n_envs <= 0:
        raise ValueError("No frozen_h rows selected.")
    dim_probe_prior = EnvironmentPrior(copy.deepcopy(env_cfg))
    next_state_target_dim = int(
        max(
            *[int((h or {}).get("state_dim", 1)) for h in h_list],
            int(dim_probe_prior._resolve_dim_upper_bound(env_cfg.get("state_dim", None), default=0)),
            1,
        )
    )
    algo, vec_env = _build_algo(
        cfg=cfg,
        env_cfg=env_cfg,
        frozen_h=None,
        frozen_h_list=h_list,
        frozen_h_list_env_seeds=env_seeds,
        prior_mode="fixed_frozen_list",
        device_obj=torch.device(device),
        num_features=int(num_features),
        n_envs=int(n_envs),
        n_steps=int(n_steps),
        build_seed=int(seed),
        sb3_reward_normalization_enabled=True,
        sb3_observation_normalization_enabled=bool(sb3_observation_normalization_enabled),
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
        for action_seed in action_seed_list:
            for mode in action_mode_list:
                env_group_seeds = [int(v) for v in env_seeds]
                obs = vec_env._full_reset_batch(seeds=env_group_seeds)
                del obs
                meta = _env_meta_from_vec_env(vec_env, h_list, n_envs)
                meta["env_group_seeds"] = env_group_seeds
                meta["env_group_digest"] = _stable_digest({"h_list": h_list, "env_group_seeds": env_group_seeds})
                rewards_by_env = [[] for _ in range(n_envs)]
                actions_by_env = [[] for _ in range(n_envs)]
                masks_by_env = [[] for _ in range(n_envs)]
                dones_by_env = [[] for _ in range(n_envs)]
                rng = np.random.default_rng(int(action_seed))
                open_loop_cache: dict[str, np.ndarray] = {}
                for step_idx in range(int(n_steps)):
                    masks = vec_env.action_masks().astype(np.float32, copy=False)
                    action = _mode_action(
                        mode=mode,
                        rng=rng,
                        step_idx=int(step_idx),
                        n_envs=int(n_envs),
                        action_slot_dim=int(action_slot_dim),
                        masks=masks,
                        open_loop_cache=open_loop_cache,
                    )
                    vec_env.step_async(action)
                    _obs, reward, done, _infos = vec_env.step_wait()
                    reward_arr = np.asarray(reward, dtype=np.float64)
                    done_arr = np.asarray(done, dtype=bool)
                    for env_idx in range(n_envs):
                        action_dim = int(meta.get("action_dims", [action_slot_dim] * n_envs)[env_idx])
                        rewards_by_env[env_idx].append(float(reward_arr[env_idx]))
                        actions_by_env[env_idx].append(action[env_idx, :action_dim].astype(np.float64, copy=True))
                        masks_by_env[env_idx].append(masks[env_idx, :action_dim].astype(np.float64, copy=True))
                        dones_by_env[env_idx].append(bool(done_arr[env_idx]))
                for env_idx in range(n_envs):
                    reward_np = np.asarray(rewards_by_env[env_idx], dtype=np.float64)
                    action_np = np.asarray(actions_by_env[env_idx], dtype=np.float64)
                    mask_np = np.asarray(masks_by_env[env_idx], dtype=np.float64)
                    components, component_meta = _reward_component_arrays_from_pack(
                        arm="prior",
                        reward=reward_np,
                        active_actions=action_np,
                        active_action_masks=mask_np,
                        meta=meta,
                        env_idx=int(env_idx),
                    )
                    env_seed = int(env_seeds[env_idx])
                    full_gate = full_gate_by_seed.get(env_seed, {})
                    row = {
                        "rule": str(rule),
                        "env_idx": int(env_idx),
                        "env_seed": int(env_seed),
                        "full_frozen_h_json": h_paths[env_idx],
                        "action_mode": str(mode),
                        "action_seed": int(action_seed),
                        "n_steps": int(n_steps),
                        "done_count": int(np.sum(dones_by_env[env_idx])),
                        "component_contract_exact": bool(
                            component_meta.get("contract_exact_for_current_none_transform_runs", False)
                        ),
                        "ctrl_enabled": bool(component_meta.get("ctrl_enabled", False)),
                        "ctrl_weight": component_meta.get("ctrl_weight"),
                    }
                    for key in (
                        "reward_state_gain",
                        "reward_action_gain",
                        "reward_noise_gain",
                        "_constrained_obs_u",
                        "noise_budget_frac",
                        "obs_std",
                        "obs_rank",
                        "reward_std",
                        "return_std",
                        "reward_sens",
                        "state_sens",
                        "G5",
                    ):
                        if key in full_gate:
                            try:
                                row[f"full_gate_{key}"] = float(full_gate[key])
                            except (TypeError, ValueError):
                                row[f"full_gate_{key}"] = full_gate[key]
                    row.update({f"total_reward_{k}": v for k, v in _reward_component_stats(reward_np).items()})
                    for name, values in components.items():
                        row.update({f"{name}_{k}": v for k, v in _reward_component_stats(values).items()})
                    rows.append(row)
    finally:
        vec_env.close()
        del algo
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    _write_jsonl(out_dir / "rows.jsonl", rows)
    _write_csv(out_dir / "rows.csv", rows)

    by_env: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        by_env.setdefault(int(row["env_idx"]), []).append(row)
    env_rows = []
    for env_idx, env_rows_raw in sorted(by_env.items()):
        env_seed = int(env_rows_raw[0]["env_seed"])
        env_q90_values = [row.get("reward_env_q90") for row in env_rows_raw]
        env_positive = [row.get("reward_env_positive_fraction") for row in env_rows_raw]
        env_sum = [row.get("reward_env_sum") for row in env_rows_raw]
        bad_rows = [
            row for row in env_rows_raw
            if row.get("reward_env_q90") is not None and float(row["reward_env_q90"]) < 0.0
        ]
        low_tail_rows = [
            row for row in env_rows_raw
            if row.get("reward_env_positive_fraction") is not None
            and float(row["reward_env_positive_fraction"]) < 0.20
        ]
        env_rows.append(
            {
                "env_idx": int(env_idx),
                "env_seed": int(env_seed),
                "probe_count": int(len(env_rows_raw)),
                "bad_q90_fraction": float(len(bad_rows) / max(1, len(env_rows_raw))),
                "low_positive_tail_fraction": float(len(low_tail_rows) / max(1, len(env_rows_raw))),
                "reward_env_q90": _stats(env_q90_values),
                "reward_env_positive_fraction": _stats(env_positive),
                "reward_env_sum": _stats(env_sum),
                "full_gate_reward_sens": env_rows_raw[0].get("full_gate_reward_sens"),
                "full_gate_state_sens": env_rows_raw[0].get("full_gate_state_sens"),
                "full_gate_G5": env_rows_raw[0].get("full_gate_G5"),
                "full_gate_reward_state_gain": env_rows_raw[0].get("full_gate_reward_state_gain"),
                "full_gate_reward_action_gain": env_rows_raw[0].get("full_gate_reward_action_gain"),
                "full_gate_reward_noise_gain": env_rows_raw[0].get("full_gate_reward_noise_gain"),
                "full_gate_noise_budget_frac": env_rows_raw[0].get("full_gate_noise_budget_frac"),
                "full_gate__constrained_obs_u": env_rows_raw[0].get("full_gate__constrained_obs_u"),
            }
        )
    _write_csv(out_dir / "env_summary.csv", env_rows)
    report = {
        "audit_entry": "phase2_pretrain_reward_basin_action_distribution_probe",
        "contract": (
            "No training/update. Fixed frozen_h list is rolled out under deterministic action "
            "distribution families and action seeds. Component reward fields are exact for "
            "none-transform exact-SCM configs."
        ),
        "output_dir": str(out_dir),
        "source_csv": source_csv,
        "input_kind": input_kind,
        "rows_jsonl": str(out_dir / "rows.jsonl"),
        "rows_csv": str(out_dir / "rows.csv"),
        "env_summary_csv": str(out_dir / "env_summary.csv"),
        "action_modes": action_mode_list,
        "action_seeds": action_seed_list,
        "env_count": int(n_envs),
        "row_count": int(len(rows)),
        "env_summary": env_rows,
        "digest": hashlib.sha256(
            json.dumps(_json_safe(env_rows), sort_keys=True).encode("utf-8")
        ).hexdigest(),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(_json_safe(report), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe fixed prior env reward basins under action distributions.")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--selected-csv", type=str, default=DEFAULT_SELECTED_CSV)
    parser.add_argument("--population-csv", type=str, default=DEFAULT_POPULATION_CSV)
    parser.add_argument("--input-kind", type=str, default="selected", choices=("selected", "population"))
    parser.add_argument("--rule", type=str, default="balanced_topology_mid_noise")
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--n-steps", type=int, default=256)
    parser.add_argument("--seed", type=int, default=4040)
    parser.add_argument(
        "--action-modes",
        type=str,
        default="zero,iid_gaussian_0p25,iid_gaussian_1p0,openloop_gaussian_0p25,openloop_gaussian_1p0",
    )
    parser.add_argument("--action-seeds", type=str, default="4040,4041,4042,4043")
    parser.add_argument("--sb3-observation-normalization-enabled", type=str2bool, default=False)
    args = parser.parse_args()
    report = run_probe(
        output_dir=args.output_dir,
        device=args.device,
        selected_csv=args.selected_csv,
        population_csv=args.population_csv,
        input_kind=args.input_kind,
        rule=args.rule,
        limit=int(args.limit),
        n_steps=int(args.n_steps),
        seed=int(args.seed),
        action_modes=args.action_modes,
        action_seeds=args.action_seeds,
        sb3_observation_normalization_enabled=bool(args.sb3_observation_normalization_enabled),
    )
    print(
        json.dumps(
            _json_safe(
                {
                    "summary_json": str(Path(args.output_dir).expanduser().resolve() / "summary.json"),
                    "env_summary_csv": report["env_summary_csv"],
                    "row_count": report["row_count"],
                    "env_count": report["env_count"],
                }
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
