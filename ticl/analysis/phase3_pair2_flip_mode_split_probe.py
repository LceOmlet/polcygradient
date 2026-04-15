import argparse
import json
from pathlib import Path
from typing import Any


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(float(v) for v in values) / len(values))


def _classify_flip_mode(row: dict[str, Any]) -> str:
    cause = str(row.get("flip_cause", ""))
    if cause == "negative_gae_future_carry":
        return "future_carry_dominant"
    if cause in {"bootstrap_term_negative", "delta_negative_mixed"}:
        return "bootstrap_dominant"
    raise ValueError(f"Unsupported flip_cause for split probe: {cause}")


def _iter_target_rows(payload: dict[str, Any], suite_name: str) -> list[dict[str, Any]]:
    target_rows: list[dict[str, Any]] = []
    for bundle in list(payload.get("token_row_bundles", [])):
        if str(bundle.get("suite_name", "")) != str(suite_name):
            continue
        env_index = int(bundle["env_index"])
        pre_gap = float(bundle["pre_vs_zero_suffix_gap"])
        for row in list(bundle.get("token_rows", [])):
            if not bool(row.get("positive_residual_wrong_sign_actor", False)):
                continue
            enriched = dict(row)
            enriched["suite_name"] = str(suite_name)
            enriched["env_index"] = int(env_index)
            enriched["pre_vs_zero_suffix_gap"] = float(pre_gap)
            enriched["flip_mode"] = _classify_flip_mode(enriched)
            target_rows.append(enriched)
    return target_rows


def _summarize_mode_rows(rows: list[dict[str, Any]], all_target_rows: list[dict[str, Any]]) -> dict[str, Any]:
    env_counts: dict[str, int] = {}
    cause_counts: dict[str, int] = {}
    episode_counts: dict[str, int] = {}
    for row in rows:
        env_key = f"env{int(row['env_index'])}"
        env_counts[env_key] = int(env_counts.get(env_key, 0) + 1)
        cause = str(row["flip_cause"])
        cause_counts[cause] = int(cause_counts.get(cause, 0) + 1)
        episode_key = f"episode{int(row['objective_episode_index'])}"
        episode_counts[episode_key] = int(episode_counts.get(episode_key, 0) + 1)

    return {
        "flip_mode": str(rows[0]["flip_mode"]) if rows else "none",
        "token_count": int(len(rows)),
        "share_within_target_tokens": float(len(rows) / max(1, len(all_target_rows))),
        "env_counts": dict(sorted(env_counts.items(), key=lambda item: item[0])),
        "flip_cause_counts": dict(sorted(cause_counts.items(), key=lambda item: item[0])),
        "objective_episode_counts": dict(sorted(episode_counts.items(), key=lambda item: item[0])),
        "mean_pre_vs_zero_suffix_gap": _mean([float(row["pre_vs_zero_suffix_gap"]) for row in rows]),
        "mean_raw_rollout_residual_norm": _mean([float(row["raw_rollout_residual_norm"]) for row in rows]),
        "mean_reward_norm": _mean([float(row["reward_norm"]) for row in rows]),
        "mean_bootstrap_term_norm": _mean([float(row["bootstrap_term_norm"]) for row in rows]),
        "mean_delta_norm": _mean([float(row["delta_norm"]) for row in rows]),
        "mean_mc_future_carry_equiv_norm": _mean([float(row["mc_future_carry_equiv_norm"]) for row in rows]),
        "mean_gae_future_carry_norm": _mean([float(row["gae_future_carry_norm"]) for row in rows]),
        "mean_actor_adv_norm": _mean([float(row["actor_adv_norm"]) for row in rows]),
        "mean_objective_episode_index": _mean([float(row["objective_episode_index"]) for row in rows]),
        "mean_token_in_objective_episode": _mean([float(row["token_in_objective_episode"]) for row in rows]),
        "mean_tokens_to_objective_episode_end": _mean(
            [float(row["tokens_to_objective_episode_end"]) for row in rows]
        ),
        "mean_steps_until_next_objective": _mean(
            [float(row["steps_until_next_objective"]) for row in rows if int(row["steps_until_next_objective"]) >= 0]
        ),
        "next_step_nonobjective_share": float(
            sum(1 for row in rows if not bool(row["next_step_is_objective"])) / max(1, len(rows))
        ),
    }


def _build_probe_from_payload(payload: dict[str, Any], suite_name: str) -> dict[str, Any]:
    gae_config = dict(payload.get("config", {}))
    target_rows = _iter_target_rows(payload, suite_name)
    if not target_rows:
        raise ValueError(f"No positive_residual_wrong_sign_actor rows found for suite {suite_name}.")

    mode_names = ["bootstrap_dominant", "future_carry_dominant"]
    mode_rows = []
    for mode_name in mode_names:
        mode_target_rows = [row for row in target_rows if str(row["flip_mode"]) == mode_name]
        if not mode_target_rows:
            raise ValueError(f"Expected at least one {mode_name} token in suite {suite_name}.")
        mode_rows.append(_summarize_mode_rows(mode_target_rows, target_rows))

    mode_by_name = {str(row["flip_mode"]): row for row in mode_rows}
    bootstrap_row = mode_by_name["bootstrap_dominant"]
    future_row = mode_by_name["future_carry_dominant"]

    contrast = {
        "future_minus_bootstrap_token_share": float(
            float(future_row["share_within_target_tokens"]) - float(bootstrap_row["share_within_target_tokens"])
        ),
        "future_minus_bootstrap_mean_raw_rollout_residual_norm": float(
            float(future_row["mean_raw_rollout_residual_norm"]) - float(bootstrap_row["mean_raw_rollout_residual_norm"])
        ),
        "future_minus_bootstrap_mean_reward_norm": float(
            float(future_row["mean_reward_norm"]) - float(bootstrap_row["mean_reward_norm"])
        ),
        "future_minus_bootstrap_mean_bootstrap_term_norm": float(
            float(future_row["mean_bootstrap_term_norm"]) - float(bootstrap_row["mean_bootstrap_term_norm"])
        ),
        "future_minus_bootstrap_mean_delta_norm": float(
            float(future_row["mean_delta_norm"]) - float(bootstrap_row["mean_delta_norm"])
        ),
        "future_minus_bootstrap_mean_mc_future_carry_equiv_norm": float(
            float(future_row["mean_mc_future_carry_equiv_norm"])
            - float(bootstrap_row["mean_mc_future_carry_equiv_norm"])
        ),
        "future_minus_bootstrap_mean_gae_future_carry_norm": float(
            float(future_row["mean_gae_future_carry_norm"]) - float(bootstrap_row["mean_gae_future_carry_norm"])
        ),
        "future_minus_bootstrap_mean_actor_adv_norm": float(
            float(future_row["mean_actor_adv_norm"]) - float(bootstrap_row["mean_actor_adv_norm"])
        ),
        "future_minus_bootstrap_mean_objective_episode_index": float(
            float(future_row["mean_objective_episode_index"]) - float(bootstrap_row["mean_objective_episode_index"])
        ),
        "future_minus_bootstrap_mean_token_in_objective_episode": float(
            float(future_row["mean_token_in_objective_episode"])
            - float(bootstrap_row["mean_token_in_objective_episode"])
        ),
        "future_minus_bootstrap_mean_tokens_to_objective_episode_end": float(
            float(future_row["mean_tokens_to_objective_episode_end"])
            - float(bootstrap_row["mean_tokens_to_objective_episode_end"])
        ),
        "future_minus_bootstrap_next_step_nonobjective_share": float(
            float(future_row["next_step_nonobjective_share"]) - float(bootstrap_row["next_step_nonobjective_share"])
        ),
    }

    conclusions = {
        "bootstrap_mode_present": bool(int(bootstrap_row["token_count"]) > 0),
        "future_carry_mode_present": bool(int(future_row["token_count"]) > 0),
        "future_carry_tokens_have_more_negative_gae_future_carry": bool(
            float(future_row["mean_gae_future_carry_norm"]) < float(bootstrap_row["mean_gae_future_carry_norm"])
        ),
        "bootstrap_tokens_have_more_negative_bootstrap_term": bool(
            float(bootstrap_row["mean_bootstrap_term_norm"]) < float(future_row["mean_bootstrap_term_norm"])
        ),
        "future_carry_tokens_are_more_front_loaded": bool(
            float(future_row["mean_token_in_objective_episode"])
            < float(bootstrap_row["mean_token_in_objective_episode"])
        ),
        "bootstrap_tokens_are_more_tail_skewed": bool(
            float(bootstrap_row["mean_tokens_to_objective_episode_end"])
            < float(future_row["mean_tokens_to_objective_episode_end"])
        ),
    }

    flip_token_rows = sorted(
        target_rows,
        key=lambda row: (
            int(row["env_index"]),
            int(row["objective_episode_index"]),
            int(row["objective_local_index"]),
        ),
    )

    return {
        "audit_entry": "phase3_pair2_flip_mode_split_probe",
        "config": {
            "source_gae_path_json": str(
                Path(
                    gae_config.get(
                        "source_gae_path_json",
                        "/home/chen/RLPFN/artifacts/phase3_residual_anchor_gae_path_probe.json",
                    )
                ).expanduser().resolve()
            ),
            "source_train_profile": str(gae_config.get("train_profile", "")),
            "source_n_samples": int(gae_config.get("n_samples", 0)),
            "source_single_eval_pos": int(gae_config.get("single_eval_pos", 0)),
            "suite_name": str(suite_name),
        },
        "aggregate": {
            "target_token_count": int(len(target_rows)),
            "env_count": int(len({int(row["env_index"]) for row in target_rows})),
            "flip_mode_counts": {
                str(row["flip_mode"]): int(row["token_count"]) for row in mode_rows
            },
        },
        "mode_rows": mode_rows,
        "contrast": contrast,
        "conclusions": conclusions,
        "flip_token_rows": flip_token_rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Split pair2 residual-flip tokens into bootstrap-dominant vs future-carry-dominant buckets."
    )
    parser.add_argument(
        "--gae-path-json",
        type=str,
        default="/home/chen/RLPFN/artifacts/phase3_residual_anchor_gae_path_probe.json",
    )
    parser.add_argument("--suite-name", type=str, default="pair2")
    parser.add_argument("--output-json", type=str, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output_path = Path(args.output_json).expanduser().resolve()
    if output_path.exists() and not bool(args.overwrite):
        print(output_path.read_text())
        return 0

    input_path = Path(args.gae_path_json).expanduser().resolve()
    payload = json.loads(input_path.read_text())
    result = _build_probe_from_payload(payload, str(args.suite_name))
    result["config"]["source_gae_path_json"] = str(input_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(output_path.read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
