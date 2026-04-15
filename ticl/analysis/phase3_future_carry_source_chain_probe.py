import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


RECON_EPS = 1e-6
FACTOR_SPREAD_EPS = 1e-4


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(float(v) for v in values) / len(values))


def _max_abs(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(max(abs(float(v)) for v in values))


def _dominant_label(counter: Counter[str]) -> str:
    if not counter:
        return "none"
    label, value = max(counter.items(), key=lambda item: (float(item[1]), str(item[0])))
    if float(value) <= 0.0:
        return "none"
    return str(label)


def _classify_negative_source(row: dict[str, Any]) -> str:
    if float(row["raw_rollout_residual_raw"]) <= 0.0:
        return "true_negative_future_residual"
    if float(row["next_non_terminal"]) <= 0.0:
        return "terminal_reset_tail_semantic_mismatch"
    return "baseline_bootstrap_semantic_mismatch"


def _load_token_rows(gae_payload: dict[str, Any], suite_name: str, env_index: int) -> list[dict[str, Any]]:
    for bundle in list(gae_payload.get("token_row_bundles", [])):
        if str(bundle.get("suite_name", "")) == str(suite_name) and int(bundle.get("env_index", -1)) == int(env_index):
            rows = [dict(row) for row in list(bundle.get("token_rows", []))]
            if not rows:
                raise ValueError(f"No token rows found for {suite_name} env{env_index}.")
            return rows
    raise ValueError(f"Could not find token rows for {suite_name} env{env_index}.")


def _load_start_rows(
    split_payload: dict[str, Any],
    suite_name: str,
    env_index: int,
    flip_mode: str,
) -> list[dict[str, Any]]:
    rows = [
        dict(row)
        for row in list(split_payload.get("flip_token_rows", []))
        if str(row.get("suite_name", "")) == str(suite_name)
        and int(row.get("env_index", -1)) == int(env_index)
        and str(row.get("flip_mode", "")) == str(flip_mode)
    ]
    if not rows:
        raise ValueError(f"No flip_token_rows found for {suite_name} env{env_index} flip_mode={flip_mode}.")
    return sorted(rows, key=lambda row: (int(row["objective_episode_index"]), int(row["objective_local_index"])))


def _group_episode_rows(token_rows: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in token_rows:
        grouped[int(row["objective_episode_index"])].append(dict(row))
    for episode_index, rows in grouped.items():
        rows.sort(key=lambda row: int(row["objective_local_index"]))
        for idx in range(len(rows) - 1):
            curr = rows[idx]
            nxt = rows[idx + 1]
            if int(nxt["objective_local_index"]) != int(curr["objective_local_index"]) + 1:
                raise RuntimeError(
                    f"Non-consecutive objective_local_index inside episode {episode_index}: "
                    f"{curr['objective_local_index']} -> {nxt['objective_local_index']}"
                )
            if int(nxt["global_step"]) != int(curr["global_step"]) + 1:
                raise RuntimeError(
                    f"Non-consecutive global_step inside episode {episode_index}: "
                    f"{curr['global_step']} -> {nxt['global_step']}"
                )
            if not bool(curr["next_step_is_objective"]) or int(curr["steps_until_next_objective"]) != 1:
                raise RuntimeError(
                    f"Episode {episode_index} broke contiguous objective chain at local "
                    f"{curr['objective_local_index']}: next_step_is_objective={curr['next_step_is_objective']} "
                    f"steps_until_next_objective={curr['steps_until_next_objective']}"
                )
    return dict(sorted(grouped.items(), key=lambda item: int(item[0])))


def _infer_effective_recursion_factor(episode_rows: list[dict[str, Any]]) -> dict[str, float]:
    factors: list[float] = []
    for idx in range(len(episode_rows) - 1):
        curr = episode_rows[idx]
        nxt = episode_rows[idx + 1]
        next_actor_adv = float(nxt["actor_adv_norm"])
        if abs(next_actor_adv) <= 1e-12:
            continue
        factor = float((float(curr["actor_adv_norm"]) - float(curr["delta_norm"])) / next_actor_adv)
        factors.append(factor)
    if not factors:
        raise RuntimeError("Could not infer effective recursion factor from episode rows.")
    mean_factor = _mean(factors)
    return {
        "pair_count": int(len(factors)),
        "mean": float(mean_factor),
        "min": float(min(factors)),
        "max": float(max(factors)),
        "max_abs_deviation_from_mean": _max_abs([float(v) - float(mean_factor) for v in factors]),
    }


def _trace_start_source_chain(
    start_row: dict[str, Any],
    episode_rows: list[dict[str, Any]],
    effective_factor: float,
) -> dict[str, Any]:
    start_local = int(start_row["objective_local_index"])
    row_by_local = {int(row["objective_local_index"]): row for row in episode_rows}
    max_local = max(row_by_local)
    negative_source_contributions: Counter[str] = Counter()
    negative_contribution_total = 0.0
    positive_offset_contribution_total = 0.0
    weighted_sum = 0.0
    contributor_rows: list[dict[str, Any]] = []

    for future_local in range(start_local + 1, max_local + 1):
        row = row_by_local[future_local]
        distance = int(future_local - start_local)
        weight = float(effective_factor**distance)
        weighted_delta = float(weight * float(row["delta_norm"]))
        weighted_sum += weighted_delta
        source_bucket = "positive_offset"
        if weighted_delta < 0.0:
            source_bucket = _classify_negative_source(row)
            negative_source_contributions[source_bucket] += float(-weighted_delta)
            negative_contribution_total += float(-weighted_delta)
        else:
            positive_offset_contribution_total += float(weighted_delta)
        contributor_rows.append(
            {
                "objective_episode_index": int(row["objective_episode_index"]),
                "objective_local_index": int(row["objective_local_index"]),
                "global_step": int(row["global_step"]),
                "distance_from_start": int(distance),
                "weight": float(weight),
                "weighted_delta_contribution": float(weighted_delta),
                "source_bucket": str(source_bucket),
                "delta_norm": float(row["delta_norm"]),
                "raw_rollout_residual_norm": float(row["raw_rollout_residual_norm"]),
                "raw_rollout_residual_raw": float(row["raw_rollout_residual_raw"]),
                "next_non_terminal": float(row["next_non_terminal"]),
                "tokens_to_objective_episode_end": int(row["tokens_to_objective_episode_end"]),
                "flip_cause": str(row["flip_cause"]),
            }
        )

    top_negative_contributors = sorted(
        [row for row in contributor_rows if float(row["weighted_delta_contribution"]) < 0.0],
        key=lambda row: abs(float(row["weighted_delta_contribution"])),
        reverse=True,
    )[:5]
    reconstruction_error = float(weighted_sum - float(start_row["gae_future_carry_norm"]))
    negative_source_shares = {
        str(key): float(float(value) / max(negative_contribution_total, 1e-12))
        for key, value in sorted(negative_source_contributions.items())
    }

    return {
        "suite_name": str(start_row["suite_name"]),
        "env_index": int(start_row["env_index"]),
        "start_objective_episode_index": int(start_row["objective_episode_index"]),
        "start_objective_local_index": int(start_row["objective_local_index"]),
        "start_global_step": int(start_row["global_step"]),
        "start_token_in_objective_episode": int(start_row["token_in_objective_episode"]),
        "start_tokens_to_objective_episode_end": int(start_row["tokens_to_objective_episode_end"]),
        "start_raw_rollout_residual_norm": float(start_row["raw_rollout_residual_norm"]),
        "start_delta_norm": float(start_row["delta_norm"]),
        "start_gae_future_carry_norm": float(start_row["gae_future_carry_norm"]),
        "future_chain_step_count": int(len(contributor_rows)),
        "negative_contribution_total": float(negative_contribution_total),
        "positive_offset_contribution_total": float(positive_offset_contribution_total),
        "negative_source_contributions": {
            str(key): float(value) for key, value in sorted(negative_source_contributions.items())
        },
        "negative_source_shares": negative_source_shares,
        "dominant_negative_source": _dominant_label(negative_source_contributions),
        "weighted_delta_sum": float(weighted_sum),
        "reconstruction_error": float(reconstruction_error),
        "top_negative_contributors": top_negative_contributors,
    }


def _build_probe(
    gae_payload: dict[str, Any],
    split_payload: dict[str, Any],
    suite_name: str,
    env_index: int,
    flip_mode: str,
) -> dict[str, Any]:
    token_rows = _load_token_rows(gae_payload, suite_name=suite_name, env_index=env_index)
    start_rows = _load_start_rows(split_payload, suite_name=suite_name, env_index=env_index, flip_mode=flip_mode)
    grouped_episode_rows = _group_episode_rows(token_rows)

    episode_guard_rows = []
    factor_rows = []
    all_factor_means = []
    for episode_index, rows in grouped_episode_rows.items():
        factor_stats = _infer_effective_recursion_factor(rows)
        factor_rows.append(
            {
                "objective_episode_index": int(episode_index),
                **factor_stats,
            }
        )
        all_factor_means.append(float(factor_stats["mean"]))
        episode_guard_rows.append(
            {
                "objective_episode_index": int(episode_index),
                "token_count": int(len(rows)),
                "global_step_start": int(rows[0]["global_step"]),
                "global_step_end": int(rows[-1]["global_step"]),
                "local_index_start": int(rows[0]["objective_local_index"]),
                "local_index_end": int(rows[-1]["objective_local_index"]),
                "episode_terminal_token_next_non_terminal": float(rows[-1]["next_non_terminal"]),
            }
        )

    effective_factor = float(_mean(all_factor_means))
    factor_spread = _max_abs([float(row["mean"]) - effective_factor for row in factor_rows])
    if factor_spread > float(FACTOR_SPREAD_EPS):
        raise RuntimeError(
            f"Effective recursion factor spread too large for {suite_name} env{env_index}: {factor_spread}"
        )

    rows_by_episode_local = {
        int(episode_index): {int(row["objective_local_index"]): row for row in rows}
        for episode_index, rows in grouped_episode_rows.items()
    }
    start_summaries = []
    negative_source_contributions: Counter[str] = Counter()
    dominant_negative_source_counts: Counter[str] = Counter()
    top_negative_token_contributions: defaultdict[tuple[int, int, str], float] = defaultdict(float)
    positive_offset_contribution_total = 0.0
    negative_contribution_total = 0.0
    reconstruction_errors = []
    start_next_step_nonobjective_count = 0

    for start_row in start_rows:
        if not bool(start_row["next_step_is_objective"]):
            start_next_step_nonobjective_count += 1
        episode_index = int(start_row["objective_episode_index"])
        summary = _trace_start_source_chain(
            start_row=start_row,
            episode_rows=grouped_episode_rows[episode_index],
            effective_factor=effective_factor,
        )
        start_summaries.append(summary)
        positive_offset_contribution_total += float(summary["positive_offset_contribution_total"])
        negative_contribution_total += float(summary["negative_contribution_total"])
        reconstruction_errors.append(float(summary["reconstruction_error"]))
        dominant_negative_source_counts[str(summary["dominant_negative_source"])] += 1
        for key, value in dict(summary["negative_source_contributions"]).items():
            negative_source_contributions[str(key)] += float(value)
        for contributor in list(summary["top_negative_contributors"]):
            key = (
                int(contributor["objective_episode_index"]),
                int(contributor["objective_local_index"]),
                str(contributor["source_bucket"]),
            )
            top_negative_token_contributions[key] += float(-float(contributor["weighted_delta_contribution"]))

    max_reconstruction_error = _max_abs(reconstruction_errors)
    if max_reconstruction_error > float(RECON_EPS):
        raise RuntimeError(
            f"Future-carry source-chain reconstruction error too large for {suite_name} env{env_index}: "
            f"{max_reconstruction_error}"
        )

    negative_source_shares = {
        str(key): float(float(value) / max(negative_contribution_total, 1e-12))
        for key, value in sorted(negative_source_contributions.items())
    }
    top_negative_source_tokens = []
    for (episode_index, local_index, source_bucket), contribution in sorted(
        top_negative_token_contributions.items(), key=lambda item: float(item[1]), reverse=True
    )[:10]:
        row = rows_by_episode_local[int(episode_index)][int(local_index)]
        top_negative_source_tokens.append(
            {
                "objective_episode_index": int(episode_index),
                "objective_local_index": int(local_index),
                "global_step": int(row["global_step"]),
                "source_bucket": str(source_bucket),
                "aggregated_negative_contribution": float(contribution),
                "delta_norm": float(row["delta_norm"]),
                "raw_rollout_residual_norm": float(row["raw_rollout_residual_norm"]),
                "raw_rollout_residual_raw": float(row["raw_rollout_residual_raw"]),
                "next_non_terminal": float(row["next_non_terminal"]),
                "tokens_to_objective_episode_end": int(row["tokens_to_objective_episode_end"]),
                "flip_cause": str(row["flip_cause"]),
            }
        )

    split_config = dict(split_payload.get("config", {}))
    gae_config = dict(gae_payload.get("config", {}))
    conclusions = {
        "objective_episode_chains_contiguous": True,
        "future_carry_chain_reconstruction_exact": bool(max_reconstruction_error <= float(RECON_EPS)),
        "effective_recursion_factor_stable": bool(float(factor_spread) <= float(FACTOR_SPREAD_EPS)),
        "target_starts_all_next_step_objective": bool(int(start_next_step_nonobjective_count) == 0),
        "no_true_negative_future_residual_source": bool(
            float(negative_source_contributions.get("true_negative_future_residual", 0.0)) == 0.0
        ),
        "negative_future_carry_explained_by_semantic_mismatch_only": bool(
            float(negative_source_contributions.get("true_negative_future_residual", 0.0)) == 0.0
        ),
        "terminal_reset_tail_is_dominant_negative_source_aggregate": bool(
            float(negative_source_contributions.get("terminal_reset_tail_semantic_mismatch", 0.0))
            > float(negative_source_contributions.get("baseline_bootstrap_semantic_mismatch", 0.0))
        ),
        "terminal_reset_tail_dominates_most_start_tokens": bool(
            int(dominant_negative_source_counts.get("terminal_reset_tail_semantic_mismatch", 0))
            > int(dominant_negative_source_counts.get("baseline_bootstrap_semantic_mismatch", 0))
        ),
    }

    return {
        "audit_entry": "phase3_future_carry_source_chain_probe",
        "config": {
            "source_gae_path_json": str(
                Path(
                    gae_config.get(
                        "source_gae_path_json",
                        "/home/chen/RLPFN/artifacts/phase3_residual_anchor_gae_path_probe.json",
                    )
                ).expanduser().resolve()
            ),
            "source_flip_mode_split_json": str(
                Path(
                    split_config.get(
                        "source_flip_mode_split_json",
                        "/home/chen/RLPFN/artifacts/phase3_pair2_flip_mode_split_probe.json",
                    )
                ).expanduser().resolve()
            ),
            "suite_name": str(suite_name),
            "env_index": int(env_index),
            "flip_mode": str(flip_mode),
            "reconstruction_eps": float(RECON_EPS),
            "factor_spread_eps": float(FACTOR_SPREAD_EPS),
        },
        "guardrails": {
            "episode_guard_rows": episode_guard_rows,
            "factor_rows": factor_rows,
            "effective_recursion_factor_mean": float(effective_factor),
            "effective_recursion_factor_spread": float(factor_spread),
            "max_source_chain_reconstruction_error": float(max_reconstruction_error),
            "mean_source_chain_reconstruction_error": float(_mean(reconstruction_errors)),
            "start_next_step_nonobjective_count": int(start_next_step_nonobjective_count),
        },
        "aggregate": {
            "start_token_count": int(len(start_summaries)),
            "objective_episode_counts": {
                f"episode{int(k)}": int(v)
                for k, v in sorted(Counter(int(row["start_objective_episode_index"]) for row in start_summaries).items())
            },
            "negative_contribution_total": float(negative_contribution_total),
            "positive_offset_contribution_total": float(positive_offset_contribution_total),
            "negative_source_contributions": {
                str(key): float(value) for key, value in sorted(negative_source_contributions.items())
            },
            "negative_source_shares": negative_source_shares,
            "dominant_negative_source_counts": {
                str(key): int(value) for key, value in sorted(dominant_negative_source_counts.items())
            },
            "dominant_negative_source": _dominant_label(dominant_negative_source_counts),
        },
        "top_negative_source_tokens": top_negative_source_tokens,
        "start_rows": start_summaries,
        "conclusions": conclusions,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Trace pair-level future-carry source chains and separate true negative future residual from "
            "baseline/tail semantic mismatch."
        )
    )
    parser.add_argument(
        "--gae-path-json",
        type=str,
        default="/home/chen/RLPFN/artifacts/phase3_residual_anchor_gae_path_probe.json",
    )
    parser.add_argument(
        "--flip-mode-split-json",
        type=str,
        default="/home/chen/RLPFN/artifacts/phase3_pair2_flip_mode_split_probe.json",
    )
    parser.add_argument("--suite-name", type=str, default="pair2")
    parser.add_argument("--env-index", type=int, default=12)
    parser.add_argument("--flip-mode", type=str, default="future_carry_dominant")
    parser.add_argument("--output-json", type=str, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output_path = Path(args.output_json).expanduser().resolve()
    if output_path.exists() and not bool(args.overwrite):
        print(output_path.read_text())
        return 0

    gae_input_path = Path(args.gae_path_json).expanduser().resolve()
    split_input_path = Path(args.flip_mode_split_json).expanduser().resolve()
    gae_payload = json.loads(gae_input_path.read_text())
    split_payload = json.loads(split_input_path.read_text())

    result = _build_probe(
        gae_payload=gae_payload,
        split_payload=split_payload,
        suite_name=str(args.suite_name),
        env_index=int(args.env_index),
        flip_mode=str(args.flip_mode),
    )
    result["config"]["source_gae_path_json"] = str(gae_input_path)
    result["config"]["source_flip_mode_split_json"] = str(split_input_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(output_path.read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
