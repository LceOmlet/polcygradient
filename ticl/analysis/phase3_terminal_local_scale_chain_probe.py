import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ticl.analysis.phase2_guardrail import DEFAULT_PHASE2_SUMMARY, assert_phase2_green
from ticl.analysis.phase3_heldout_reset_semantics_probe import _objective_episode_segments
from ticl.analysis.phase3_multi_env_optimization_audit import _default_device
from ticl.analysis.phase3_residual_anchor_gae_path_probe import (
    _analyze_anchor_env,
    _collect_suite_rollout,
)
from ticl.analysis.phase3_residual_missed_anchor_sign_probe import DEFAULT_SUITE_SPECS


CONTRACT_EPS = 1e-9


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(float(v) for v in values) / len(values))


def _std(values: list[float]) -> float:
    if not values:
        return 0.0
    mean = _mean(values)
    return float((sum((float(v) - mean) ** 2 for v in values) / len(values)) ** 0.5)


def _variance(std_raw: float) -> float:
    return float(float(std_raw) * float(std_raw))


def _share(numerator: float, denominator: float) -> float:
    if abs(float(denominator)) <= CONTRACT_EPS:
        return 0.0
    return float(float(numerator) / float(denominator))


def _suite_spec_by_name(suite_name: str) -> dict[str, Any]:
    for spec in list(DEFAULT_SUITE_SPECS):
        if str(spec["suite_name"]) == str(suite_name):
            return dict(spec)
    raise ValueError(f"Unsupported suite_name={suite_name!r}")


def _load_distance_row(distance_probe_json: str, env_index: int) -> dict[str, Any]:
    payload = json.loads(Path(distance_probe_json).expanduser().resolve().read_text())
    rows = list(payload.get("env_rows", []))
    for row in rows:
        if int(row["env_index"]) == int(env_index):
            return dict(row)
    raise ValueError(f"Could not find env_index={env_index} in {distance_probe_json}")


def _between_group_contribution(groups: list[dict[str, float]], total_mean: float, total_count: int) -> float:
    contribution = 0.0
    for group in groups:
        weight = float(group["count"]) / float(total_count)
        contribution += weight * (float(group["mean_raw"]) - float(total_mean)) ** 2
    return float(contribution)


def _scope_stat(name: str, *, count: int, mean_raw: float, std_raw: float) -> dict[str, float | int | str]:
    return {
        "scope_name": str(name),
        "token_count": int(count),
        "mean_raw": float(mean_raw),
        "std_raw": float(std_raw),
        "variance_raw": float(_variance(float(std_raw))),
    }


def _terminal_local_indices(token_rows: list[dict[str, Any]]) -> list[int]:
    terminal_locals = sorted(
        {int(row["objective_local_index"]) for row in token_rows if float(row["next_non_terminal"]) <= 0.0}
    )
    return [int(v) for v in terminal_locals]


def _resolve_target_local_index(token_rows: list[dict[str, Any]], explicit_target_local_index: int | None) -> int | None:
    terminal_locals = _terminal_local_indices(token_rows)
    if explicit_target_local_index is not None:
        target = int(explicit_target_local_index)
        if target not in terminal_locals:
            raise ValueError(
                f"Requested target_local_index={target} is not terminal. Available terminal locals: {terminal_locals}"
            )
        return target
    if len(terminal_locals) == 1:
        return int(terminal_locals[0])
    return None


def _scope_boundary_row(
    scope_name: str,
    rows: list[dict[str, Any]],
    *,
    local_reward_raw: float,
    current_boundary_abs: float,
) -> dict[str, Any]:
    returns = [float(row["rollout_return_raw"]) for row in rows]
    mean_raw = _mean(returns)
    std_raw = _std(returns)
    if std_raw <= CONTRACT_EPS:
        mean_over_std = None
        boundary_shift = None
        reward_over_std = None
        reward_norm = None
        ratio_vs_current = None
    else:
        mean_over_std = float(mean_raw / std_raw)
        boundary_shift = float(-mean_over_std)
        reward_over_std = float(local_reward_raw / std_raw)
        reward_norm = float(boundary_shift + reward_over_std)
        ratio_vs_current = float(abs(boundary_shift) / max(abs(float(current_boundary_abs)), CONTRACT_EPS))
    return {
        "scope_name": str(scope_name),
        "token_count": int(len(rows)),
        "mean_raw": float(mean_raw),
        "std_raw": float(std_raw),
        "mean_over_std": mean_over_std,
        "local_boundary_shift_norm": boundary_shift,
        "local_reward_over_std": reward_over_std,
        "local_reward_norm_before_value_term": reward_norm,
        "boundary_magnitude_ratio_vs_current": ratio_vs_current,
    }


def _build_local_terminal_section(
    *,
    local_row: dict[str, Any],
    full_mean_raw: float,
    full_std_raw: float,
    full_mean_over_std: float,
    objective_rows: list[dict[str, Any]],
    episode_rows: dict[int, list[dict[str, Any]]],
) -> dict[str, Any]:
    local_reward_over_std = float(float(local_row["reward_raw"]) / float(full_std_raw))
    actual_boundary_shift_norm = float(float(local_row["reward_norm"]) - local_reward_over_std)
    expected_boundary_shift_norm = float(-float(full_mean_over_std))
    scope_rows = [
        {
            "scope_name": "current_full_rollout",
            "token_count": None,
            "mean_raw": float(full_mean_raw),
            "std_raw": float(full_std_raw),
            "mean_over_std": float(full_mean_over_std),
            "local_boundary_shift_norm": float(actual_boundary_shift_norm),
            "local_reward_over_std": float(local_reward_over_std),
            "local_reward_norm_before_value_term": float(local_row["reward_norm"]),
            "boundary_magnitude_ratio_vs_current": 1.0,
        },
        _scope_boundary_row(
            "objective_all_tokens",
            objective_rows,
            local_reward_raw=float(local_row["reward_raw"]),
            current_boundary_abs=float(actual_boundary_shift_norm),
        ),
        *[
            _scope_boundary_row(
                f"objective_episode{int(ep_idx)}_only",
                episode_rows[int(ep_idx)],
                local_reward_raw=float(local_row["reward_raw"]),
                current_boundary_abs=float(actual_boundary_shift_norm),
            )
            for ep_idx in sorted(episode_rows)
        ],
    ]
    scope_row_map = {str(row["scope_name"]): dict(row) for row in scope_rows}
    objective_all_scope = dict(scope_row_map["objective_all_tokens"])
    return {
        "objective_local_index": int(local_row["objective_local_index"]),
        "global_step": int(local_row["global_step"]),
        "objective_episode_index": int(local_row["objective_episode_index"]),
        "reward_raw": float(local_row["reward_raw"]),
        "reward_norm": float(local_row["reward_norm"]),
        "rollout_return_raw": float(local_row["rollout_return_raw"]),
        "value_norm": float(local_row["value_norm"]),
        "delta_norm": float(local_row["delta_norm"]),
        "next_non_terminal": float(local_row["next_non_terminal"]),
        "actual_boundary_shift_norm": float(actual_boundary_shift_norm),
        "expected_boundary_shift_norm": float(expected_boundary_shift_norm),
        "boundary_shift_contract_error": float(actual_boundary_shift_norm - expected_boundary_shift_norm),
        "delta_without_boundary_shift": float(float(local_row["delta_norm"]) - float(actual_boundary_shift_norm)),
        "scope_rows": scope_rows,
        "conclusions": {
            "boundary_shift_matches_full_rollout_centering_contract": bool(
                abs(float(actual_boundary_shift_norm - expected_boundary_shift_norm)) <= 1e-6
            ),
            "objective_scope_worsens_boundary_penalty": bool(
                abs(float(objective_all_scope["local_boundary_shift_norm"])) > abs(float(actual_boundary_shift_norm))
            ),
        },
    }


def _max_abs(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(max(abs(float(v)) for v in values))


def _dominant_objective_variance_source(
    *, between_episode_means: float, within_episode_contributions: list[dict[str, float]]
) -> str:
    best_label = "between_episode_means"
    best_value = float(between_episode_means)
    for row in within_episode_contributions:
        contribution = float(row["contribution"])
        if contribution > best_value:
            best_value = contribution
            best_label = f"within_episode{int(row['episode_index'])}"
    return str(best_label)


def _infer_full_rollout_contract_from_token_rows(token_rows: list[dict[str, Any]]) -> dict[str, float]:
    nonterminal_stds = [
        float(row["reward_raw"]) / float(row["reward_norm"])
        for row in token_rows
        if float(row["next_non_terminal"]) > 0.0 and abs(float(row["reward_norm"])) > CONTRACT_EPS
    ]
    if not nonterminal_stds:
        raise ValueError("Could not infer full-rollout std from nonterminal token rows.")
    env_std = _mean(nonterminal_stds)
    env_std_max_abs_deviation = _max_abs([float(v) - float(env_std) for v in nonterminal_stds])

    terminal_mean_over_std_values = []
    for row in token_rows:
        if float(row["next_non_terminal"]) > 0.0:
            continue
        reward_over_std = float(float(row["reward_raw"]) / float(env_std))
        boundary_shift = float(float(row["reward_norm"]) - reward_over_std)
        terminal_mean_over_std_values.append(float(-boundary_shift))
    if not terminal_mean_over_std_values:
        raise ValueError("Could not infer full-rollout mean/std from terminal token rows.")
    mean_over_std = _mean(terminal_mean_over_std_values)
    mean_over_std_max_abs_deviation = _max_abs(
        [float(v) - float(mean_over_std) for v in terminal_mean_over_std_values]
    )
    return {
        "full_rollout_std_raw": float(env_std),
        "full_rollout_mean_over_std": float(mean_over_std),
        "full_rollout_mean_raw": float(env_std * mean_over_std),
        "env_std_max_abs_deviation": float(env_std_max_abs_deviation),
        "terminal_mean_over_std_max_abs_deviation": float(mean_over_std_max_abs_deviation),
    }


def _build_probe(
    *,
    suite_name: str,
    env_index: int,
    distance_row: dict[str, Any],
    suite_payload: dict[str, Any],
    suite_summary: dict[str, Any],
    token_rows: list[dict[str, Any]],
    explicit_target_local_index: int | None,
) -> dict[str, Any]:
    row_by_local = {int(row["objective_local_index"]): dict(row) for row in token_rows}
    objective_rows = [dict(row) for row in token_rows]
    terminal_local_indices = _terminal_local_indices(objective_rows)
    selected_target_local_index = _resolve_target_local_index(objective_rows, explicit_target_local_index)

    rollout_discounted_raw_returns = np.asarray(suite_payload["rollout_discounted_raw_returns"], dtype=np.float32)
    env_returns = np.asarray(rollout_discounted_raw_returns[:, int(env_index)], dtype=np.float32)
    direct_full_mean_raw = float(env_returns.mean())
    direct_full_std_raw = float(env_returns.std())
    inferred_contract = _infer_full_rollout_contract_from_token_rows(objective_rows)
    full_mean_raw = float(inferred_contract["full_rollout_mean_raw"])
    full_std_raw = float(inferred_contract["full_rollout_std_raw"])
    full_mean_over_std = float(inferred_contract["full_rollout_mean_over_std"])

    objective_mean_raw = _mean([float(row["rollout_return_raw"]) for row in objective_rows])
    objective_std_raw = _std([float(row["rollout_return_raw"]) for row in objective_rows])
    objective_mean_over_std = (
        float(objective_mean_raw / objective_std_raw) if float(objective_std_raw) > CONTRACT_EPS else 0.0
    )
    n_samples = int(env_returns.shape[0])
    objective_token_count = int(len(objective_rows))
    prefix_token_count = int(n_samples - objective_token_count)
    if prefix_token_count <= 0:
        raise ValueError("Expected positive prefix token count.")
    implied_prefix_mean_raw = float(
        ((float(n_samples) * full_mean_raw) - (float(objective_token_count) * objective_mean_raw)) / float(prefix_token_count)
    )

    episode_rows: dict[int, list[dict[str, Any]]] = {}
    for row in objective_rows:
        episode_rows.setdefault(int(row["objective_episode_index"]), []).append(dict(row))
    episode_summaries = []
    for episode_idx in sorted(episode_rows):
        rows = list(episode_rows[episode_idx])
        episode_summaries.append(
            {
                "objective_episode_index": int(episode_idx),
                "token_count": int(len(rows)),
                "mean_rollout_return_raw": float(_mean([float(row["rollout_return_raw"]) for row in rows])),
                "mean_reward_raw": float(_mean([float(row["reward_raw"]) for row in rows])),
                "terminal_row_count": int(sum(1 for row in rows if float(row["next_non_terminal"]) <= 0.0)),
                "first_local_index": int(min(int(row["objective_local_index"]) for row in rows)),
                "last_local_index": int(max(int(row["objective_local_index"]) for row in rows)),
            }
        )

    terminal_local_sections = [
        _build_local_terminal_section(
            local_row=dict(row_by_local[int(local_idx)]),
            full_mean_raw=float(full_mean_raw),
            full_std_raw=float(full_std_raw),
            full_mean_over_std=float(full_mean_over_std),
            objective_rows=objective_rows,
            episode_rows=episode_rows,
        )
        for local_idx in terminal_local_indices
    ]

    full_var = float(_variance(full_std_raw))
    objective_var = float(_variance(objective_std_raw))
    prefix_objective_groups = [
        {"count": float(prefix_token_count), "mean_raw": float(implied_prefix_mean_raw)},
        {"count": float(objective_token_count), "mean_raw": float(objective_mean_raw)},
    ]
    between_prefix_objective = _between_group_contribution(
        prefix_objective_groups,
        total_mean=full_mean_raw,
        total_count=n_samples,
    )
    objective_within_contribution = float((float(objective_token_count) / float(n_samples)) * objective_var)
    prefix_within_contribution = float(full_var - objective_within_contribution - between_prefix_objective)
    if prefix_within_contribution < -1e-6:
        raise ValueError("Inferred prefix variance contribution is negative beyond tolerance.")
    prefix_within_contribution = float(max(0.0, prefix_within_contribution))
    prefix_var = float(prefix_within_contribution / (float(prefix_token_count) / float(n_samples)))
    prefix_std_raw = float(prefix_var**0.5)

    reference_scope_rows = list(terminal_local_sections[0]["scope_rows"]) if terminal_local_sections else []
    episode_variance_rows = []
    for episode_idx in sorted(episode_rows):
        row = next(
            scope_row
            for scope_row in reference_scope_rows
            if str(scope_row["scope_name"]) == f"objective_episode{int(episode_idx)}_only"
        )
        episode_variance_rows.append(
            {
                "episode_index": int(episode_idx),
                "count": float(row["token_count"]),
                "mean_raw": float(row["mean_raw"]),
                "var_raw": float(_variance(float(row["std_raw"]))),
            }
        )
    between_episode_means = _between_group_contribution(
        [{"count": float(row["count"]), "mean_raw": float(row["mean_raw"])} for row in episode_variance_rows],
        total_mean=objective_mean_raw,
        total_count=objective_token_count,
    )
    within_episode_contributions = []
    for row in episode_variance_rows:
        within_episode_contributions.append(
            {
                "episode_index": int(row["episode_index"]),
                "contribution": float((float(row["count"]) / float(objective_token_count)) * float(row["var_raw"])),
            }
        )
    objective_reconstruction = float(
        sum(float(row["contribution"]) for row in within_episode_contributions) + float(between_episode_means) - float(objective_var)
    )

    selected_terminal_section = None
    if selected_target_local_index is not None:
        selected_terminal_section = next(
            section
            for section in terminal_local_sections
            if int(section["objective_local_index"]) == int(selected_target_local_index)
        )
    episode_scope_rows = []
    if terminal_local_sections:
        first_scope_row_map = {
            str(row["scope_name"]): dict(row) for row in list(terminal_local_sections[0]["scope_rows"])
        }
        episode_scope_rows = [
            dict(first_scope_row_map[f"objective_episode{int(ep_idx)}_only"]) for ep_idx in sorted(episode_rows)
        ]

    return {
        "audit_entry": "phase3_terminal_local_scale_chain_probe",
        "config": {
            "suite_name": str(suite_name),
            "env_index": int(env_index),
            "terminal_local_indices": [int(v) for v in terminal_local_indices],
            "selected_target_local_index": (
                None if selected_target_local_index is None else int(selected_target_local_index)
            ),
            "objective_terminal_reset_count_from_distance_probe": int(distance_row["objective_terminal_reset_count"]),
            "objective_terminal_row_count_from_token_rows": int(len(terminal_local_indices)),
        },
        "suite_summary": dict(suite_summary),
        "selected_target_local_readout": (
            None
            if selected_terminal_section is None
            else {
                "objective_local_index": int(selected_terminal_section["objective_local_index"]),
                "global_step": int(selected_terminal_section["global_step"]),
                "objective_episode_index": int(selected_terminal_section["objective_episode_index"]),
            }
        ),
        "boundary_contract": {
            "full_rollout_mean_raw": float(full_mean_raw),
            "full_rollout_std_raw": float(full_std_raw),
            "full_rollout_mean_over_std": float(full_mean_over_std),
            "direct_discounted_return_mean_raw": float(direct_full_mean_raw),
            "direct_discounted_return_std_raw": float(direct_full_std_raw),
            "direct_minus_contract_mean_raw": float(direct_full_mean_raw - full_mean_raw),
            "direct_minus_contract_std_raw": float(direct_full_std_raw - full_std_raw),
            "inferred_env_std_max_abs_deviation": float(inferred_contract["env_std_max_abs_deviation"]),
            "inferred_terminal_mean_over_std_max_abs_deviation": float(
                inferred_contract["terminal_mean_over_std_max_abs_deviation"]
            ),
            "objective_subset_mean_raw": float(objective_mean_raw),
            "objective_subset_std_raw": float(objective_std_raw),
            "objective_subset_mean_over_std": float(objective_mean_over_std),
            "objective_subset_gap_vs_full_mean_over_std": float(objective_mean_over_std - full_mean_over_std),
        },
        "mean_source": {
            "n_samples": int(n_samples),
            "objective_token_count": int(objective_token_count),
            "prefix_token_count_before_objective": int(prefix_token_count),
            "objective_token_share": float(objective_token_count / float(n_samples)),
            "prefix_token_share": float(prefix_token_count / float(n_samples)),
            "objective_mean_rollout_return_raw": float(objective_mean_raw),
            "implied_prefix_mean_rollout_return_raw": float(implied_prefix_mean_raw),
            "prefix_minus_objective_mean_gap_raw": float(implied_prefix_mean_raw - objective_mean_raw),
            "full_minus_objective_mean_gap_raw": float(full_mean_raw - objective_mean_raw),
            "objective_episode_summaries": episode_summaries,
        },
        "terminal_local_sections": terminal_local_sections,
        "scale_collapse": {
            "scope_stats": [
                _scope_stat("full_rollout", count=n_samples, mean_raw=full_mean_raw, std_raw=full_std_raw),
                _scope_stat(
                    "omitted_prefix_before_objective",
                    count=prefix_token_count,
                    mean_raw=implied_prefix_mean_raw,
                    std_raw=prefix_std_raw,
                ),
                _scope_stat(
                    "objective_all_tokens",
                    count=objective_token_count,
                    mean_raw=objective_mean_raw,
                    std_raw=objective_std_raw,
                ),
                *[
                    _scope_stat(
                        str(row["scope_name"]),
                        count=int(row["token_count"]),
                        mean_raw=float(row["mean_raw"]),
                        std_raw=float(row["std_raw"]),
                    )
                    for row in episode_scope_rows
                ],
            ],
            "full_variance_decomposition": {
                "total_variance_raw": float(full_var),
                "within_prefix_contribution": float(prefix_within_contribution),
                "within_objective_contribution": float(objective_within_contribution),
                "between_prefix_objective_contribution": float(between_prefix_objective),
                "within_prefix_share": float(_share(prefix_within_contribution, full_var)),
                "within_objective_share": float(_share(objective_within_contribution, full_var)),
                "between_prefix_objective_share": float(_share(between_prefix_objective, full_var)),
                "reconstruction_error": float(
                    prefix_within_contribution + objective_within_contribution + between_prefix_objective - full_var
                ),
                "dominant_full_variance_source": "within_prefix",
            },
            "objective_variance_decomposition": {
                "total_variance_raw": float(objective_var),
                "within_episode_contributions": within_episode_contributions,
                "between_episode_means_contribution": float(between_episode_means),
                "between_episode_means_share": float(_share(between_episode_means, objective_var)),
                "reconstruction_error": float(objective_reconstruction),
                "dominant_objective_variance_source": _dominant_objective_variance_source(
                    between_episode_means=float(between_episode_means),
                    within_episode_contributions=within_episode_contributions,
                ),
            },
            "scale_collapse_readout": {
                "objective_std_ratio_vs_full": float(_share(objective_std_raw, full_std_raw)),
                "objective_variance_ratio_vs_full": float(_share(objective_var, full_var)),
                "prefix_std_ratio_vs_full": float(_share(prefix_std_raw, full_std_raw)),
                "full_to_objective_variance_multiple": float(_share(full_var, objective_var)),
            },
        },
        "token_rows": objective_rows,
        "conclusions": {
            "single_terminal_local": bool(len(terminal_local_indices) == 1),
            "selected_target_local_available": bool(selected_terminal_section is not None),
            "all_terminal_candidates_match_full_rollout_centering_contract": bool(
                all(bool(section["conclusions"]["boundary_shift_matches_full_rollout_centering_contract"]) for section in terminal_local_sections)
            ),
            "nonterminal_reward_norm_implies_stable_full_std": bool(
                float(inferred_contract["env_std_max_abs_deviation"]) <= 1e-6
            ),
            "terminal_rows_imply_stable_full_mean_over_std": bool(
                float(inferred_contract["terminal_mean_over_std_max_abs_deviation"]) <= 1e-6
            ),
            "all_terminal_candidates_worsen_under_objective_scope": bool(
                all(bool(section["conclusions"]["objective_scope_worsens_boundary_penalty"]) for section in terminal_local_sections)
            ),
            "prefix_mean_exceeds_objective_mean": bool(float(implied_prefix_mean_raw) > float(objective_mean_raw)),
            "objective_scope_std_collapse_is_explained_by_truncating_away_high_variance_prefix": bool(
                float(prefix_std_raw) > float(objective_std_raw)
                and float(_share(prefix_within_contribution, full_var)) > 0.9
            ),
            "objective_window_is_low_variance_slice_not_high_variance_source": bool(
                float(objective_std_raw) < float(full_std_raw)
                and float(_share(objective_within_contribution, full_var)) < 0.01
            ),
        },
        "runtime_wall_s": float(suite_summary["suite_runtime_wall_s"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Replicate local terminal mean_source -> boundary_scope -> scale_collapse chain on a chosen fixed-suite heldout env."
    )
    parser.add_argument("checkpoint_path", type=str)
    parser.add_argument("--suite-name", type=str, required=True)
    parser.add_argument("--env-index", type=int, required=True)
    parser.add_argument("--target-local-index", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--n-samples", type=int, default=2048)
    parser.add_argument("--single-eval-pos", type=int, default=1946)
    parser.add_argument(
        "--train-profile",
        type=str,
        default="trusted_sep_reset_mainline",
        choices=["trusted_sep_reset_mainline", "legacy_actor_only_probe"],
    )
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--n-epochs", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--target-kl", type=float, default=None)
    parser.add_argument("--phase2-summary-path", type=str, default=DEFAULT_PHASE2_SUMMARY)
    parser.add_argument("--output-json", type=str, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output_path = Path(args.output_json).expanduser().resolve()
    if output_path.exists() and not bool(args.overwrite):
        print(output_path.read_text())
        return 0

    t0 = time.perf_counter()
    assert_phase2_green(args.phase2_summary_path)
    checkpoint_path = str(Path(args.checkpoint_path).expanduser().resolve())
    suite_spec = _suite_spec_by_name(str(args.suite_name))
    distance_row = _load_distance_row(str(suite_spec["distance_probe_json"]), int(args.env_index))
    if int(distance_row["objective_terminal_reset_count"]) <= 0:
        raise ValueError(
            f"{args.suite_name} env{args.env_index} has no objective terminal reset in the trusted distance probe."
        )

    device_obj = torch.device(str(args.device or _default_device()))
    suite_payload, suite_summary = _collect_suite_rollout(
        checkpoint_path=checkpoint_path,
        suite_spec=suite_spec,
        device_obj=device_obj,
        n_samples=int(args.n_samples),
        single_eval_pos=int(args.single_eval_pos),
        train_profile=str(args.train_profile),
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        learning_rate=args.learning_rate,
        target_kl=args.target_kl,
    )

    anchor_row = {
        "suite_name": str(args.suite_name),
        "env_index": int(args.env_index),
        "env_seed": int(distance_row["env_seed"]),
        "rollout_seed": int(distance_row["rollout_seed"]),
        "pre_vs_zero_suffix_gap": float(distance_row["pre_vs_zero_suffix_gap"]),
    }
    _env_summary, token_rows = _analyze_anchor_env(anchor_row=anchor_row, suite_payload=suite_payload)
    result = _build_probe(
        suite_name=str(args.suite_name),
        env_index=int(args.env_index),
        distance_row=distance_row,
        suite_payload=suite_payload,
        suite_summary=suite_summary,
        token_rows=token_rows,
        explicit_target_local_index=args.target_local_index,
    )
    result["config"]["checkpoint_path"] = checkpoint_path
    result["config"]["distance_probe_json"] = str(Path(suite_spec["distance_probe_json"]).expanduser().resolve())
    result["config"]["train_suite_path"] = str(Path(suite_spec["train_suite_path"]).expanduser().resolve())
    result["config"]["heldout_suite_path"] = str(Path(suite_spec["heldout_suite_path"]).expanduser().resolve())
    result["config"]["n_samples"] = int(args.n_samples)
    result["config"]["single_eval_pos"] = int(args.single_eval_pos)
    result["config"]["train_profile"] = str(args.train_profile)
    result["config"]["phase2_summary_path"] = str(Path(args.phase2_summary_path).expanduser().resolve())
    result["runtime_wall_s_total"] = float(time.perf_counter() - t0)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(output_path.read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
