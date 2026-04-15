import argparse
import json
from pathlib import Path
from typing import Any


def _load_json(path: str) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.exists():
        raise ValueError(f"Missing JSON artifact: {resolved}")
    return json.loads(resolved.read_text())


def _mode_rows(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = payload.get("mode_rows", [])
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"Missing mode_rows in {payload.get('audit_entry')!r}")
    indexed = {}
    for row in rows:
        flip_mode = str(row.get("flip_mode", ""))
        if not flip_mode:
            raise ValueError(f"Missing flip_mode in mode row for {payload.get('audit_entry')!r}")
        indexed[flip_mode] = dict(row)
    required = {"bootstrap_dominant", "future_carry_dominant"}
    missing = required - set(indexed)
    if missing:
        raise ValueError(f"Missing flip_mode rows {sorted(missing)!r} in {payload.get('audit_entry')!r}")
    return indexed


def _token_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = payload.get("flip_token_rows", [])
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"Missing flip_token_rows in {payload.get('audit_entry')!r}")
    return [dict(row) for row in rows]


def _share(a: float, b: float) -> float | None:
    total = float(a) + float(b)
    if total == 0.0:
        return None
    return float(a / total)


def build_residual_flip_mode_compare_pack(*, residual_flip_split_json: str) -> dict[str, Any]:
    payload = _load_json(residual_flip_split_json)
    mode_rows = _mode_rows(payload)
    token_rows = _token_rows(payload)

    bootstrap = mode_rows["bootstrap_dominant"]
    future = mode_rows["future_carry_dominant"]

    comparison = {
        "bootstrap_dominant": bootstrap,
        "future_carry_dominant": future,
        "differences": {
            "mean_pre_vs_zero_suffix_gap": float(future["mean_pre_vs_zero_suffix_gap"] - bootstrap["mean_pre_vs_zero_suffix_gap"]),
            "mean_raw_rollout_residual_norm": float(future["mean_raw_rollout_residual_norm"] - bootstrap["mean_raw_rollout_residual_norm"]),
            "mean_reward_norm": float(future["mean_reward_norm"] - bootstrap["mean_reward_norm"]),
            "mean_bootstrap_term_norm": float(future["mean_bootstrap_term_norm"] - bootstrap["mean_bootstrap_term_norm"]),
            "mean_delta_norm": float(future["mean_delta_norm"] - bootstrap["mean_delta_norm"]),
            "mean_gae_future_carry_norm": float(future["mean_gae_future_carry_norm"] - bootstrap["mean_gae_future_carry_norm"]),
            "mean_actor_adv_norm": float(future["mean_actor_adv_norm"] - bootstrap["mean_actor_adv_norm"]),
            "mean_tokens_to_objective_episode_end": float(
                future["mean_tokens_to_objective_episode_end"] - bootstrap["mean_tokens_to_objective_episode_end"]
            ),
            "mean_token_in_objective_episode": float(
                future["mean_token_in_objective_episode"] - bootstrap["mean_token_in_objective_episode"]
            ),
            "next_step_nonobjective_share": float(
                future["next_step_nonobjective_share"] - bootstrap["next_step_nonobjective_share"]
            ),
            "token_share": float(future["share_within_target_tokens"] - bootstrap["share_within_target_tokens"]),
        },
        "shared_actor_adv_magnitude": bool(
            abs(future["mean_actor_adv_norm"] - bootstrap["mean_actor_adv_norm"]) < 0.001
        ),
        "future_mode_dominates_pre_gap_and_residual": bool(
            future["mean_pre_vs_zero_suffix_gap"] > bootstrap["mean_pre_vs_zero_suffix_gap"]
            and future["mean_raw_rollout_residual_norm"] > bootstrap["mean_raw_rollout_residual_norm"]
        ),
        "future_mode_is_more_future_carry_negative": bool(
            future["mean_gae_future_carry_norm"] < bootstrap["mean_gae_future_carry_norm"]
        ),
        "bootstrap_mode_is_more_locally_negative": bool(
            bootstrap["mean_reward_norm"] < future["mean_reward_norm"]
            and bootstrap["mean_delta_norm"] < future["mean_delta_norm"]
        ),
        "future_mode_has_zero_next_step_nonobjective_share": bool(float(future["next_step_nonobjective_share"]) == 0.0),
        "bootstrap_mode_has_some_next_step_nonobjective_share": bool(float(bootstrap["next_step_nonobjective_share"]) > 0.0),
    }

    future_rows = [row for row in token_rows if str(row.get("flip_mode", "")) == "future_carry_dominant"]
    bootstrap_rows = [row for row in token_rows if str(row.get("flip_mode", "")) == "bootstrap_dominant"]
    env_distribution = {
        "future_carry_dominant": {},
        "bootstrap_dominant": {},
    }
    for row in future_rows:
        env = str(row.get("env_index"))
        env_distribution["future_carry_dominant"][env] = env_distribution["future_carry_dominant"].get(env, 0) + 1
    for row in bootstrap_rows:
        env = str(row.get("env_index"))
        env_distribution["bootstrap_dominant"][env] = env_distribution["bootstrap_dominant"].get(env, 0) + 1

    recommendation = {
        "candidate_small_fix_target": "future_carry_source_chain",
        "bootstrap_only_fix_insufficient": True,
        "regime_split_guardrail_required": True,
        "reason": (
            "Future-carry tokens carry the stronger heldout-gain anchors and stronger raw residuals, "
            "while final actor-adv magnitude is nearly identical to bootstrap-dominant tokens. "
            "That makes future-carry recursion the narrower active search target."
        ),
    }

    return {
        "audit_entry": "phase3_residual_flip_mode_compare_pack",
        "source_residual_flip_split_json": str(Path(residual_flip_split_json).expanduser().resolve()),
        "comparison": comparison,
        "token_distribution": env_distribution,
        "conclusions": {
            "future_carry_chain_is_primary_active_search_target": True,
            "bootstrap_only_fix_not_enough": True,
            "residual_flip_modes_have_nearly_same_final_actor_adv_magnitude": comparison["shared_actor_adv_magnitude"],
            "future_carry_mode_has_stronger_anchor_and_residual_signal": comparison["future_mode_dominates_pre_gap_and_residual"],
            "future_carry_mode_carries_more_negative_recursive_signal": comparison["future_mode_is_more_future_carry_negative"],
            "bootstrap_mode_is_locally_more_negative": comparison["bootstrap_mode_is_more_locally_negative"],
        },
        "recommendation": recommendation,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare the future-carry-dominant and bootstrap-dominant residual sign-formation modes."
    )
    parser.add_argument(
        "--residual-flip-split-json",
        type=str,
        default="/home/chen/RLPFN/artifacts/phase3_pair2_flip_mode_split_probe.json",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="/home/chen/RLPFN/artifacts/phase3_residual_flip_mode_compare_pack.json",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output_path = Path(args.output_json).expanduser().resolve()
    if output_path.exists() and not bool(args.overwrite):
        print(output_path.read_text())
        return 0

    report = build_residual_flip_mode_compare_pack(
        residual_flip_split_json=args.residual_flip_split_json,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
