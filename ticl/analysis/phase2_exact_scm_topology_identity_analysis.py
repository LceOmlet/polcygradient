import argparse
import copy
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ticl.analysis.critic_free_single_env_audit import _seed_all
from ticl.analysis.phase2_identity_env_population_sweep import _json_safe_value, _safe_quantile, _vector_stats
from ticl.analysis.phase2_suffix_state_identity_probe import (
    _build_audit_env_cfg,
    _load_full_frozen_h,
    _resolve_probe_config,
)
from ticl.priors.environment_prior import EnvironmentPrior


def _unwrap_reference_transition_fn(fn):
    current = fn
    visited = set()
    while callable(current) and id(current) not in visited:
        visited.add(id(current))
        if hasattr(current, "_reference_state_indices") and current.__closure__ is not None:
            freevars = dict(zip(current.__code__.co_freevars, current.__closure__))
            if "first_weight" in freevars and "hidden_weights" in freevars:
                return current
        if current.__closure__ is None:
            break
        next_fn = None
        for name, cell in zip(current.__code__.co_freevars, current.__closure__):
            if name in {"transition_fn_full", "joint_fn", "transition_core"} and callable(cell.cell_contents):
                next_fn = cell.cell_contents
                break
        if next_fn is None:
            break
        current = next_fn
    raise RuntimeError("could not unwrap reference exact-SCM transition function")


def _closure_dict(fn) -> dict[str, Any]:
    if fn.__closure__ is None:
        return {}
    return {str(name): cell.cell_contents for name, cell in zip(fn.__code__.co_freevars, fn.__closure__)}


def _tensor_cpu(value: Any) -> torch.Tensor:
    if not torch.is_tensor(value):
        raise TypeError(f"expected tensor, got {type(value).__name__}")
    return value.detach().float().cpu()


def _cosine(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= eps:
        return float("nan")
    return float(np.dot(a, b) / denom)


def _group_slices(env: dict[str, Any]) -> dict[str, slice]:
    state_dim = int(env["state_dim"])
    obs_dim = int(env["obs_dim"])
    action_dim = int(env["action_dim"])
    noise_dim = int(env["noise_dim"])
    zero_pad_dim = int(env["zero_pad_dim"])
    ref = bool(env.get("reference_semantics_enabled", False))
    cursor = 0
    groups: dict[str, slice] = {}
    groups["state_input"] = slice(cursor, cursor + state_dim)
    cursor += state_dim
    if not ref:
        groups["obs_input"] = slice(cursor, cursor + obs_dim)
        cursor += obs_dim
    groups["action_input"] = slice(cursor, cursor + action_dim)
    cursor += action_dim
    groups["noise_input"] = slice(cursor, cursor + noise_dim)
    cursor += noise_dim
    groups["zero_pad_input"] = slice(cursor, cursor + zero_pad_dim)
    cursor += zero_pad_dim
    groups["non_pad_input"] = slice(0, cursor - zero_pad_dim)
    groups["all_input"] = slice(0, cursor)
    return groups


def _path_gain_to_layers(first_weight: np.ndarray, hidden_weights: list[np.ndarray]) -> list[np.ndarray]:
    abs_path = np.abs(first_weight).astype(np.float64, copy=False)
    paths = []
    for weight in hidden_weights:
        abs_path = abs_path @ np.abs(weight).astype(np.float64, copy=False)
        paths.append(abs_path.copy())
    return paths


def _direct_gain_between_outputs(
    *,
    state_layers: np.ndarray,
    state_units: np.ndarray,
    reward_layer: int,
    reward_unit: int,
    hidden_weights: list[np.ndarray],
) -> np.ndarray:
    gains = np.zeros((int(state_layers.size),), dtype=np.float64)
    for i, (layer, unit) in enumerate(zip(state_layers, state_units)):
        layer_i = int(layer)
        unit_i = int(unit)
        if layer_i >= int(reward_layer):
            continue
        path = np.eye(hidden_weights[0].shape[0], dtype=np.float64)
        for next_layer in range(layer_i + 1, int(reward_layer) + 1):
            path = path @ np.abs(hidden_weights[int(next_layer)]).astype(np.float64, copy=False)
        gains[int(i)] = float(path[unit_i, int(reward_unit)])
    return gains


def extract_topology_features_for_frozen_h(
    *,
    full_frozen_h_json: str,
    build_seed: int,
    train_env_seed: int,
    device: str | None,
) -> dict[str, Any]:
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
    fn = _unwrap_reference_transition_fn(env["transition_generator"])
    closure = _closure_dict(fn)

    first_weight = _tensor_cpu(closure["first_weight"]).numpy().astype(np.float64, copy=False)
    hidden_weights = [
        _tensor_cpu(weight).numpy().astype(np.float64, copy=False) for weight in list(closure["hidden_weights"])
    ]
    hidden_dim = int(first_weight.shape[1])
    hidden_blocks = int(len(hidden_weights))
    state_dim = int(env["state_dim"])
    state_indices_all = _tensor_cpu(getattr(fn, "_reference_state_indices")).long().numpy()
    state_indices = state_indices_all[:state_dim]
    reward_index = int(_tensor_cpu(getattr(fn, "_reference_reward_index")).long().reshape(-1)[0].item())
    reward_layer = int(reward_index // hidden_dim)
    reward_unit = int(reward_index % hidden_dim)
    state_layers = (state_indices // hidden_dim).astype(np.int64)
    state_units = (state_indices % hidden_dim).astype(np.int64)
    paths = _path_gain_to_layers(first_weight, hidden_weights)
    reward_path = paths[int(reward_layer)][:, int(reward_unit)]
    state_paths = np.stack(
        [paths[int(layer)][:, int(unit)] for layer, unit in zip(state_layers, state_units)],
        axis=0,
    )
    state_path_mean = np.mean(state_paths, axis=0)
    groups = _group_slices(env)

    def group_sum(vec: np.ndarray, key: str) -> float:
        sl = groups[key]
        return float(np.sum(vec[sl]))

    reward_nonpad_gain = max(group_sum(reward_path, "non_pad_input"), 1e-12)
    reward_state_gain = group_sum(reward_path, "state_input")
    reward_action_gain = group_sum(reward_path, "action_input")
    reward_noise_gain = group_sum(reward_path, "noise_input")
    reward_pad_gain = group_sum(reward_path, "zero_pad_input")
    state_mean_nonpad_gain = max(group_sum(state_path_mean, "non_pad_input"), 1e-12)
    direct_gains = _direct_gain_between_outputs(
        state_layers=state_layers,
        state_units=state_units,
        reward_layer=reward_layer,
        reward_unit=reward_unit,
        hidden_weights=hidden_weights,
    )
    state_reward_path_cosines = np.asarray([_cosine(path, reward_path) for path in state_paths], dtype=np.float64)
    state_reward_path_cosines = state_reward_path_cosines[np.isfinite(state_reward_path_cosines)]
    state_before = state_layers < int(reward_layer)
    state_same = state_layers == int(reward_layer)
    state_after = state_layers > int(reward_layer)

    out: dict[str, Any] = {
        "topology_hidden_dim": float(hidden_dim),
        "topology_hidden_blocks": float(hidden_blocks),
        "topology_input_dim": float(first_weight.shape[0]),
        "topology_reward_layer": float(reward_layer),
        "topology_reward_layer_fraction": float(reward_layer / max(1, hidden_blocks - 1)),
        "topology_reward_unit_fraction": float(reward_unit / max(1, hidden_dim - 1)),
        "topology_state_output_layer_mean": float(np.mean(state_layers)),
        "topology_state_output_layer_std": float(np.std(state_layers)),
        "topology_state_before_reward_fraction": float(np.mean(state_before)),
        "topology_state_same_reward_layer_fraction": float(np.mean(state_same)),
        "topology_state_after_reward_fraction": float(np.mean(state_after)),
        "topology_state_reward_layer_distance_mean": float(np.mean(np.abs(state_layers - int(reward_layer)))),
        "topology_state_reward_layer_distance_q90": _safe_quantile(np.abs(state_layers - int(reward_layer)), 0.90),
        "topology_state_to_reward_direct_gain_mean": float(np.mean(direct_gains)),
        "topology_state_to_reward_direct_gain_max": float(np.max(direct_gains)) if direct_gains.size else float("nan"),
        "topology_state_to_reward_direct_gain_q90": _safe_quantile(direct_gains, 0.90),
        "topology_state_to_reward_direct_gain_positive_fraction": float(np.mean(direct_gains > 0.0)),
        "topology_reward_input_gain_total": float(np.sum(reward_path)),
        "topology_reward_nonpad_input_gain_total": reward_nonpad_gain,
        "topology_reward_state_input_gain": reward_state_gain,
        "topology_reward_action_input_gain": reward_action_gain,
        "topology_reward_noise_input_gain": reward_noise_gain,
        "topology_reward_pad_input_gain": reward_pad_gain,
        "topology_reward_state_input_gain_fraction": float(reward_state_gain / reward_nonpad_gain),
        "topology_reward_action_input_gain_fraction": float(reward_action_gain / reward_nonpad_gain),
        "topology_reward_noise_input_gain_fraction": float(reward_noise_gain / reward_nonpad_gain),
        "topology_reward_state_to_noise_gain_ratio": float(reward_state_gain / max(reward_noise_gain, 1e-12)),
        "topology_reward_state_to_action_gain_ratio": float(reward_state_gain / max(reward_action_gain, 1e-12)),
        "topology_mean_state_output_nonpad_input_gain_total": state_mean_nonpad_gain,
        "topology_reward_to_state_mean_input_gain_ratio": float(reward_nonpad_gain / state_mean_nonpad_gain),
        "topology_reward_path_to_mean_state_path_cosine": _cosine(reward_path, state_path_mean),
        "topology_reward_path_to_state_path_cosine_mean": float(np.mean(state_reward_path_cosines))
        if state_reward_path_cosines.size
        else float("nan"),
        "topology_reward_path_to_state_path_cosine_q10": _safe_quantile(state_reward_path_cosines, 0.10),
        "topology_reward_path_to_state_path_cosine_q90": _safe_quantile(state_reward_path_cosines, 0.90),
    }
    out.update(_vector_stats("topology_state_layers", state_layers.astype(np.float64)))
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return _json_safe_value(out)


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 3:
        return float("nan")
    x0 = x[mask] - float(np.mean(x[mask]))
    y0 = y[mask] - float(np.mean(y[mask]))
    denom = float(np.linalg.norm(x0) * np.linalg.norm(y0))
    if denom <= 1e-12:
        return float("nan")
    return float(np.dot(x0, y0) / denom)


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    sorted_values = values[order]
    n = int(values.size)
    i = 0
    while i < n:
        j = i + 1
        while j < n and sorted_values[j] == sorted_values[i]:
            j += 1
        rank = 0.5 * (i + j - 1)
        ranks[order[i:j]] = rank
        i = j
    return ranks


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 3:
        return float("nan")
    return _pearson(_rankdata(x[mask]), _rankdata(y[mask]))


def _feature_correlations(rows: list[dict[str, Any]], target: str, prefixes: tuple[str, ...]) -> list[dict[str, Any]]:
    y = np.asarray([float(row.get(target, float("nan"))) for row in rows], dtype=np.float64)
    keys = sorted(
        {
            key
            for row in rows
            for key, value in row.items()
            if any(str(key).startswith(prefix) for prefix in prefixes)
            and isinstance(value, (int, float))
        }
    )
    out = []
    for key in keys:
        x = np.asarray([float(row.get(key, float("nan"))) for row in rows], dtype=np.float64)
        pearson = _pearson(x, y)
        spearman = _spearman(x, y)
        if not (math.isfinite(pearson) or math.isfinite(spearman)):
            continue
        out.append(
            {
                "feature": key,
                "target": target,
                "pearson": pearson,
                "spearman": spearman,
                "abs_spearman": abs(spearman) if math.isfinite(spearman) else float("nan"),
            }
        )
    out.sort(key=lambda item: float(item["abs_spearman"]), reverse=True)
    return out


def run_analysis(args: argparse.Namespace) -> dict[str, Any]:
    input_dir = Path(args.input_dir).expanduser().resolve()
    rows_path = input_dir / "rows.jsonl"
    rows = [json.loads(line) for line in rows_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    enriched = []
    for idx, row in enumerate(rows, start=1):
        print(f"[topology] seed={row['frozen_h_seed']} ({idx}/{len(rows)})", flush=True)
        topo = extract_topology_features_for_frozen_h(
            full_frozen_h_json=str(row["full_frozen_h_json"]),
            build_seed=int(args.build_seed),
            train_env_seed=int(args.train_env_seed),
            device=args.device,
        )
        new_row = dict(row)
        new_row.update(topo)
        enriched.append(_json_safe_value(new_row))
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    enriched_path = output_dir / "rows_with_topology.jsonl"
    with enriched_path.open("w", encoding="utf-8") as handle:
        for row in sorted(enriched, key=lambda item: int(item["frozen_h_seed"])):
            handle.write(json.dumps(_json_safe_value(row), sort_keys=True) + "\n")

    targets = [
        key
        for key in ("horizon_2_bias_corrected_identity_score", "horizon_3_bias_corrected_identity_score",
                    "horizon_4_bias_corrected_identity_score", "horizon_5_bias_corrected_identity_score")
        if key in enriched[0]
    ]
    correlations = {
        target: _feature_correlations(
            enriched,
            target,
            prefixes=("topology_", "state_dim", "action_dim", "noise_dim", "zero_pad_dim", "num_layers",
                      "prior_mlp_hidden_dim", "init_std", "noise_std", "alpha", "reference_state_inertia_enabled",
                      "terminal_reset", "scm_standard", "state_noise_std", "reward_scale"),
        )[: int(args.top_k)]
        for target in targets
    }
    summary = {
        "audit_entry": "phase2_exact_scm_topology_identity_analysis",
        "input_dir": str(input_dir),
        "row_count": int(len(enriched)),
        "targets": targets,
        "top_correlations_by_target": correlations,
        "rows_with_topology_jsonl": str(enriched_path),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(_json_safe_value(summary), sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(f"SUMMARY_WRITTEN {summary_path}", flush=True)
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract exact-SCM topology features and relate them to horizon identity.")
    parser.add_argument("--input-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--build-seed", type=int, default=4040)
    parser.add_argument("--train-env-seed", type=int, default=2020)
    parser.add_argument("--top-k", type=int, default=25)
    return parser


def main() -> None:
    run_analysis(build_arg_parser().parse_args())


if __name__ == "__main__":
    main()
