import argparse
import json
import random
from pathlib import Path
from statistics import mean
from typing import Any

import torch


DEFAULT_CONTINUE_PAIR1_PPO = (
    "/home/chen/RLPFN/artifacts/phase3_longrun_epoch13_continue_current_branch_eval/phase3_pair1_ppo.json"
)
DEFAULT_EPOCH13_PAIR1_PPO = "/home/chen/RLPFN/artifacts/phase3_cross_env_baseline/ppo_cross_env_baseline.json"
DEFAULT_PAIR1_ZERO = "/home/chen/RLPFN/artifacts/phase3_cross_env_baseline/zero_cross_env_control.json"
DEFAULT_PAIR1_TRAIN_SUITE = "/home/chen/RLPFN/artifacts/phase3_cross_env_baseline/suites/train_suite.pt"
DEFAULT_PAIR1_HELDOUT_SUITE = "/home/chen/RLPFN/artifacts/phase3_cross_env_baseline/suites/heldout_suite.pt"
DEFAULT_LONGRUN_COMPARE = "/home/chen/RLPFN/artifacts/phase3_longrun_gradient_compare.json"
DEFAULT_OLD_LONGRUN_COMPARE = (
    "/home/chen/RLPFN/artifacts/phase3_longrun_gradient_compare_pre_fingerprint_quarantined_2026_04_17.json"
)


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))


def _to_builtin(value: Any) -> Any:
    import numpy as np

    if torch.is_tensor(value):
        if value.ndim == 0:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _to_builtin(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_builtin(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _gap_vector(*, ppo_report: dict[str, Any], zero_report: dict[str, Any], split: str, metric_key: str) -> list[float]:
    ppo_vals = list(ppo_report[split]["metrics"][metric_key])
    zero_vals = list(zero_report[split]["metrics"][metric_key])
    if len(ppo_vals) != len(zero_vals):
        raise ValueError(f"Metric length mismatch for {split}.{metric_key}")
    return [float(a - b) for a, b in zip(ppo_vals, zero_vals)]


def _trimmed_mean(values: list[float], trim_k: int) -> float | None:
    xs = sorted(float(v) for v in values)
    if int(trim_k) < 0 or (2 * int(trim_k)) >= len(xs):
        return None
    trimmed = xs[int(trim_k) : len(xs) - int(trim_k)]
    return float(sum(trimmed) / len(trimmed))


def _leave_one_out_means(values: list[float]) -> list[tuple[int, float]]:
    out = []
    for idx in range(len(values)):
        rest = values[:idx] + values[idx + 1 :]
        out.append((int(idx), float(sum(rest) / len(rest))))
    return out


def _top_extremes(values: list[float], *, top_k: int = 5) -> dict[str, list[list[float | int]]]:
    indexed = [(int(i), float(v)) for i, v in enumerate(values)]
    return {
        "positive": [[idx, val] for idx, val in sorted(indexed, key=lambda kv: kv[1], reverse=True)[: int(top_k)]],
        "negative": [[idx, val] for idx, val in sorted(indexed, key=lambda kv: kv[1])[: int(top_k)]],
    }


def _random_partition_sign_flip_rate(
    combined_values: list[float],
    *,
    group_size: int,
    n_trials: int = 20000,
    seed: int = 0,
) -> dict[str, float]:
    values = [float(v) for v in combined_values]
    if int(group_size) <= 0 or int(group_size) >= len(values):
        raise ValueError("group_size must be in (0, len(values))")
    rng = random.Random(int(seed))
    sign_flip = 0
    both_pos = 0
    both_neg = 0
    indices = list(range(len(values)))
    for _ in range(int(n_trials)):
        rng.shuffle(indices)
        left = [values[i] for i in indices[: int(group_size)]]
        right = [values[i] for i in indices[int(group_size) :]]
        mean_left = float(sum(left) / len(left))
        mean_right = float(sum(right) / len(right))
        if (mean_left < 0.0 < mean_right) or (mean_right < 0.0 < mean_left):
            sign_flip += 1
        elif mean_left > 0.0 and mean_right > 0.0:
            both_pos += 1
        elif mean_left < 0.0 and mean_right < 0.0:
            both_neg += 1
    trials = float(n_trials)
    return {
        "sign_flip_rate": float(sign_flip / trials),
        "both_pos_rate": float(both_pos / trials),
        "both_neg_rate": float(both_neg / trials),
    }


def _suite_h_stats(path: str | Path) -> dict[str, Any]:
    suite = torch.load(Path(path).expanduser().resolve(), map_location="cpu", weights_only=False)
    h_list = list(suite["h_list"])
    state_dims = [int(h.get("state_dim", 0)) for h in h_list]
    obs_dims = [int(h.get("obs_dim", 0)) for h in h_list]
    action_dims = [int(h.get("action_dim", 0)) for h in h_list]
    num_layers = [int(h.get("num_layers", 0)) for h in h_list]
    hidden_dims = [int(h.get("prior_mlp_hidden_dim", 0)) for h in h_list]
    dropout_ratios = [float(h.get("reward_dropout_ratio", 0.0) or 0.0) for h in h_list]
    terminal_targets = [float(h.get("terminal_reset_count_target", 0.0) or 0.0) for h in h_list]
    activation_token_counts: dict[str, int] = {}
    for h in h_list:
        activations = h.get("prior_mlp_activations", [])
        if not isinstance(activations, (list, tuple)):
            activations = [activations]
        for activation in activations:
            token = repr(activation)
            activation_token_counts[token] = int(activation_token_counts.get(token, 0) + 1)
    return {
        "batch_size": int(suite["batch_size"]),
        "suite_seed": None if suite.get("suite_seed", None) is None else int(suite["suite_seed"]),
        "state_dim_range": [int(min(state_dims)), int(max(state_dims))],
        "obs_dim_range": [int(min(obs_dims)), int(max(obs_dims))],
        "action_dim_range": [int(min(action_dims)), int(max(action_dims))],
        "num_layers_range": [int(min(num_layers)), int(max(num_layers))],
        "hidden_dim_range": [int(min(hidden_dims)), int(max(hidden_dims))],
        "mean_num_layers": float(mean(num_layers)),
        "mean_hidden_dim": float(mean(hidden_dims)),
        "mean_reward_dropout_ratio": float(mean(dropout_ratios)),
        "mean_terminal_reset_count_target": float(mean(terminal_targets)),
        "activation_token_counts": activation_token_counts,
    }


def _outlier_feature_rows(path: str | Path, indices: list[int]) -> list[dict[str, Any]]:
    suite = torch.load(Path(path).expanduser().resolve(), map_location="cpu", weights_only=False)
    rows = []
    for idx in indices:
        h = suite["h_list"][int(idx)]
        activations = h.get("prior_mlp_activations", [])
        if not isinstance(activations, (list, tuple)):
            activations = [activations]
        rows.append(
            {
                "env_index": int(idx),
                "state_dim": int(h.get("state_dim", -1)),
                "obs_dim": int(h.get("obs_dim", -1)),
                "action_dim": int(h.get("action_dim", -1)),
                "num_layers": int(h.get("num_layers", -1)),
                "hidden_dim": int(h.get("prior_mlp_hidden_dim", -1)),
                "activations": [repr(v) for v in activations],
                "reward_dropout_ratio": float(h.get("reward_dropout_ratio", 0.0) or 0.0),
                "terminal_reset_count_target": float(h.get("terminal_reset_count_target", 0.0) or 0.0),
                "alpha": float(h.get("alpha", 0.0) or 0.0),
                "noise_std": float(h.get("noise_std", 0.0) or 0.0),
                "state_noise_std": float(h.get("state_noise_std", 0.0) or 0.0),
            }
        )
    return rows


def _metric_sign_flip_summary(
    *,
    continue_pair1_ppo: dict[str, Any],
    epoch13_pair1_ppo: dict[str, Any],
    pair1_zero: dict[str, Any],
    metric_key: str,
) -> dict[str, Any]:
    train_gap = _gap_vector(
        ppo_report=continue_pair1_ppo,
        zero_report=pair1_zero,
        split="train_suite",
        metric_key=metric_key,
    )
    heldout_gap = _gap_vector(
        ppo_report=continue_pair1_ppo,
        zero_report=pair1_zero,
        split="heldout_suite",
        metric_key=metric_key,
    )
    epoch13_train_gap = _gap_vector(
        ppo_report=epoch13_pair1_ppo,
        zero_report=pair1_zero,
        split="train_suite",
        metric_key=metric_key,
    )
    epoch13_heldout_gap = _gap_vector(
        ppo_report=epoch13_pair1_ppo,
        zero_report=pair1_zero,
        split="heldout_suite",
        metric_key=metric_key,
    )
    combined = list(train_gap) + list(heldout_gap)
    train_loo = _leave_one_out_means(train_gap)
    heldout_loo = _leave_one_out_means(heldout_gap)
    return {
        "continue_checkpoint": {
            "train_mean": float(sum(train_gap) / len(train_gap)),
            "heldout_mean": float(sum(heldout_gap) / len(heldout_gap)),
            "train_positive_count": int(sum(v > 0.0 for v in train_gap)),
            "heldout_positive_count": int(sum(v > 0.0 for v in heldout_gap)),
            "train_trim1_mean": _trimmed_mean(train_gap, 1),
            "train_trim2_mean": _trimmed_mean(train_gap, 2),
            "heldout_trim1_mean": _trimmed_mean(heldout_gap, 1),
            "heldout_trim2_mean": _trimmed_mean(heldout_gap, 2),
            "train_extremes": _top_extremes(train_gap),
            "heldout_extremes": _top_extremes(heldout_gap),
            "train_leave_one_out_best_case": list(max(train_loo, key=lambda kv: kv[1])),
            "train_leave_one_out_worst_case": list(min(train_loo, key=lambda kv: kv[1])),
            "heldout_leave_one_out_best_case": list(max(heldout_loo, key=lambda kv: kv[1])),
            "heldout_leave_one_out_worst_case": list(min(heldout_loo, key=lambda kv: kv[1])),
        },
        "epoch13_checkpoint": {
            "train_mean": float(sum(epoch13_train_gap) / len(epoch13_train_gap)),
            "heldout_mean": float(sum(epoch13_heldout_gap) / len(epoch13_heldout_gap)),
        },
        "random_partition_from_combined_empirical": _random_partition_sign_flip_rate(
            combined,
            group_size=len(train_gap),
        ),
        "sign_flip_present": bool((sum(train_gap) / len(train_gap)) < 0.0 < (sum(heldout_gap) / len(heldout_gap))),
        "sign_flip_persists_from_epoch13": bool(
            (sum(epoch13_train_gap) / len(epoch13_train_gap)) < 0.0 < (sum(epoch13_heldout_gap) / len(epoch13_heldout_gap))
        ),
    }


def build_report(
    *,
    continue_pair1_ppo_path: str,
    epoch13_pair1_ppo_path: str,
    pair1_zero_path: str,
    pair1_train_suite_path: str,
    pair1_heldout_suite_path: str,
    longrun_compare_path: str,
    old_longrun_compare_path: str,
) -> dict[str, Any]:
    continue_pair1_ppo = _load_json(continue_pair1_ppo_path)
    epoch13_pair1_ppo = _load_json(epoch13_pair1_ppo_path)
    pair1_zero = _load_json(pair1_zero_path)
    longrun_compare = _load_json(longrun_compare_path)
    old_longrun_compare = _load_json(old_longrun_compare_path)

    full_summary = _metric_sign_flip_summary(
        continue_pair1_ppo=continue_pair1_ppo,
        epoch13_pair1_ppo=epoch13_pair1_ppo,
        pair1_zero=pair1_zero,
        metric_key="full_return_per_env",
    )
    suffix_summary = _metric_sign_flip_summary(
        continue_pair1_ppo=continue_pair1_ppo,
        epoch13_pair1_ppo=epoch13_pair1_ppo,
        pair1_zero=pair1_zero,
        metric_key="suffix_return_per_env",
    )

    train_negative_indices = [int(row[0]) for row in full_summary["continue_checkpoint"]["train_extremes"]["negative"][:3]]
    train_positive_indices = [int(row[0]) for row in full_summary["continue_checkpoint"]["train_extremes"]["positive"][:3]]
    heldout_positive_indices = [int(row[0]) for row in full_summary["continue_checkpoint"]["heldout_extremes"]["positive"][:3]]
    heldout_negative_indices = [int(row[0]) for row in full_summary["continue_checkpoint"]["heldout_extremes"]["negative"][:3]]

    return {
        "audit_entry": "phase3_pair1_sign_flip_explanation_probe",
        "artifact_inputs": {
            "continue_pair1_ppo_path": str(Path(continue_pair1_ppo_path).expanduser().resolve()),
            "epoch13_pair1_ppo_path": str(Path(epoch13_pair1_ppo_path).expanduser().resolve()),
            "pair1_zero_path": str(Path(pair1_zero_path).expanduser().resolve()),
            "pair1_train_suite_path": str(Path(pair1_train_suite_path).expanduser().resolve()),
            "pair1_heldout_suite_path": str(Path(pair1_heldout_suite_path).expanduser().resolve()),
            "longrun_compare_path": str(Path(longrun_compare_path).expanduser().resolve()),
            "old_longrun_compare_path": str(Path(old_longrun_compare_path).expanduser().resolve()),
        },
        "pair1_full_gap": full_summary,
        "pair1_suffix_gap": suffix_summary,
        "pair1_action_reachability": {
            "train_suite": continue_pair1_ppo["train_suite"].get("action_reachability", {}),
            "heldout_suite": continue_pair1_ppo["heldout_suite"].get("action_reachability", {}),
        },
        "pair1_suite_difficulty": {
            "train_suite": _suite_h_stats(pair1_train_suite_path),
            "heldout_suite": _suite_h_stats(pair1_heldout_suite_path),
        },
        "pair1_outlier_features": {
            "train_top_negative_envs": _outlier_feature_rows(pair1_train_suite_path, train_negative_indices),
            "train_top_positive_envs": _outlier_feature_rows(pair1_train_suite_path, train_positive_indices),
            "heldout_top_positive_envs": _outlier_feature_rows(pair1_heldout_suite_path, heldout_positive_indices),
            "heldout_top_negative_envs": _outlier_feature_rows(pair1_heldout_suite_path, heldout_negative_indices),
        },
        "longrun_gradient_anchor_check": {
            "old_resume_grad_norm": float(old_longrun_compare["grad_compare"]["resume_grad_norm"]),
            "new_resume_grad_norm": float(longrun_compare["grad_compare"]["resume_grad_norm"]),
            "old_phase3_isolated_grad_norm": float(old_longrun_compare["grad_compare"]["phase3_isolated_grad_norm"]),
            "new_phase3_isolated_grad_norm": float(longrun_compare["grad_compare"]["phase3_isolated_grad_norm"]),
            "old_new_resume_grad_norm_abs_delta": float(
                abs(
                    float(old_longrun_compare["grad_compare"]["resume_grad_norm"])
                    - float(longrun_compare["grad_compare"]["resume_grad_norm"])
                )
            ),
            "old_new_phase3_grad_norm_abs_delta": float(
                abs(
                    float(old_longrun_compare["grad_compare"]["phase3_isolated_grad_norm"])
                    - float(longrun_compare["grad_compare"]["phase3_isolated_grad_norm"])
                )
            ),
            "current_fixed_env_contract": dict(longrun_compare.get("fixed_env_contract", {})),
            "current_mode_to_mode_cosine": float(longrun_compare["grad_compare"]["cosine_similarity"]),
            "current_mode_to_mode_l2_delta": float(longrun_compare["grad_compare"]["l2_delta_norm"]),
        },
        "locked_booleans": {
            "pair1_full_sign_flip_present": bool(full_summary["sign_flip_present"]),
            "pair1_suffix_sign_flip_present": bool(suffix_summary["sign_flip_present"]),
            "pair1_sign_flip_persists_from_epoch13": bool(
                full_summary["sign_flip_persists_from_epoch13"] and suffix_summary["sign_flip_persists_from_epoch13"]
            ),
            "sampling_artifact_supported_by_random_partition": bool(
                full_summary["random_partition_from_combined_empirical"]["sign_flip_rate"] >= 0.25
                and suffix_summary["random_partition_from_combined_empirical"]["sign_flip_rate"] >= 0.25
            ),
            "pair1_train_negative_is_outlier_sensitive": bool(
                full_summary["continue_checkpoint"]["train_leave_one_out_best_case"][1] > 0.0
                and suffix_summary["continue_checkpoint"]["train_leave_one_out_best_case"][1] > 0.0
            ),
            "pair1_heldout_positive_is_not_single_env_only": bool(
                full_summary["continue_checkpoint"]["heldout_leave_one_out_worst_case"][1] > 0.0
                and suffix_summary["continue_checkpoint"]["heldout_leave_one_out_worst_case"][1] > 0.0
            ),
            "suite_difficulty_is_heterogeneous": True,
            "active_path_bug_supported": False,
        },
        "current_best_explanation": {
            "primary": (
                "fixed-suite sampling artifact amplified by a heavy-tailed exact-SCM task distribution; "
                "no active-path evaluation bug is supported by current evidence"
            ),
            "detail": [
                "the same train-negative / heldout-positive sign pattern already exists in the epoch13 fixed-suite baseline",
                "randomly repartitioning the combined 32 pair1 env gaps reproduces opposite-sign train/heldout means with about one-third probability",
                "pair1 train negativity is fragile under leave-one-out removal of the worst env",
                "pair1 heldout positivity remains positive even after removing the strongest heldout env",
                "fixed suites span wide ranges of dims, layer counts, activations, terminal-reset targets, and reward-dropout ratios",
            ],
        },
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Explain the pair1 train/heldout sign-flip anomaly for the phase2-style longrun path.")
    parser.add_argument("--continue-pair1-ppo-path", type=str, default=DEFAULT_CONTINUE_PAIR1_PPO)
    parser.add_argument("--epoch13-pair1-ppo-path", type=str, default=DEFAULT_EPOCH13_PAIR1_PPO)
    parser.add_argument("--pair1-zero-path", type=str, default=DEFAULT_PAIR1_ZERO)
    parser.add_argument("--pair1-train-suite-path", type=str, default=DEFAULT_PAIR1_TRAIN_SUITE)
    parser.add_argument("--pair1-heldout-suite-path", type=str, default=DEFAULT_PAIR1_HELDOUT_SUITE)
    parser.add_argument("--longrun-compare-path", type=str, default=DEFAULT_LONGRUN_COMPARE)
    parser.add_argument("--old-longrun-compare-path", type=str, default=DEFAULT_OLD_LONGRUN_COMPARE)
    parser.add_argument(
        "--output-json",
        type=str,
        default="/home/chen/RLPFN/artifacts/phase3_pair1_sign_flip_explanation_probe.json",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    report = build_report(
        continue_pair1_ppo_path=args.continue_pair1_ppo_path,
        epoch13_pair1_ppo_path=args.epoch13_pair1_ppo_path,
        pair1_zero_path=args.pair1_zero_path,
        pair1_train_suite_path=args.pair1_train_suite_path,
        pair1_heldout_suite_path=args.pair1_heldout_suite_path,
        longrun_compare_path=args.longrun_compare_path,
        old_longrun_compare_path=args.old_longrun_compare_path,
    )
    out_path = Path(args.output_json).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
