import argparse
import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ticl.analysis.critic_free_single_env_audit import _make_deterministic_batch_plan
from ticl.analysis.phase2_guardrail import DEFAULT_PHASE2_SUMMARY, assert_phase2_green
from ticl.analysis.phase3_multi_env_optimization_audit import (
    _bind_vec_env_to_fixed_suite,
    _clone_suite_h_list,
    _default_device,
    _load_required_suite,
    _resolve_audit_env_config,
    _resolve_train_profile,
)
from ticl.analysis.phase3_pair2_train_segment_identity_probe import (
    _build_env_objective_identity_rows,
)
from ticl.analysis.phase3_pair2_train_update_ab_compare_pack import (
    PAIR2_CHECKPOINT_PATH,
    PAIR2_HELDOUT_SUITE_PATH,
    PAIR2_TRAIN_SUITE_PATH,
)
from ticl.model_builder import load_model
from ticl.priors.environment_prior import EnvironmentPrior
from ticl.sb3_recurrent_ppo import build_recurrent_ppo


DEFAULT_SOURCE_JSON = "/home/chen/RLPFN/artifacts/phase3_residual_anchor_gae_path_probe.json"
DEFAULT_SOURCE_SUITE_NAME = "pair2"


def _source_expected_compare_suite_role(source_json: str) -> str:
    payload = json.loads(Path(source_json).expanduser().resolve().read_text())
    audit_entry = str(payload.get("audit_entry", "")).strip().lower()
    if audit_entry == "phase3_residual_anchor_gae_path_probe":
        return "heldout"
    return "unknown"


def _suite_role_from_path(suite_path: str) -> str:
    name = Path(suite_path).expanduser().resolve().name
    if name == "heldout_suite.pt":
        return "heldout"
    if name == "train_suite.pt":
        return "train"
    return "unknown"


def _resolve_compare_suite_path(
    *,
    compare_suite_path: str | None,
    source_json: str,
    allow_source_suite_mismatch: bool,
) -> tuple[str, str, str]:
    expected_role = _source_expected_compare_suite_role(source_json)
    resolved_path = str(compare_suite_path or "")
    if not resolved_path:
        if expected_role == "heldout":
            resolved_path = PAIR2_HELDOUT_SUITE_PATH
        else:
            resolved_path = PAIR2_TRAIN_SUITE_PATH
    resolved_path = str(Path(resolved_path).expanduser().resolve())
    actual_role = _suite_role_from_path(resolved_path)
    if (
        not bool(allow_source_suite_mismatch)
        and expected_role != "unknown"
        and actual_role != "unknown"
        and actual_role != expected_role
    ):
        raise ValueError(
            "Source/current suite-role mismatch: "
            f"source_json={Path(source_json).expanduser().resolve()} expects compare suite role "
            f"{expected_role!r}, but compare_suite_path={resolved_path!r} resolves to role {actual_role!r}. "
            "Pass allow_source_suite_mismatch=True only for intentional counterfactual mismatch probes."
        )
    return resolved_path, expected_role, actual_role


def _extract_source_window_rows(
    *,
    source_json: str,
    source_suite_name: str,
    env_index: int,
    global_step_start: int,
    global_step_end: int,
) -> list[dict[str, Any]]:
    payload = json.loads(Path(source_json).expanduser().resolve().read_text())
    bundles = list(payload.get("token_row_bundles", []))
    env_bundle = next(
        (
            bundle
            for bundle in bundles
            if str(bundle.get("suite_name", "")) == str(source_suite_name)
            and int(bundle.get("env_index", -1)) == int(env_index)
        ),
        None,
    )
    if env_bundle is None:
        raise ValueError(
            f"Could not find suite_name={source_suite_name!r}, env_index={int(env_index)} in {source_json}"
        )
    rows = []
    for row in list(env_bundle.get("token_rows", [])):
        global_step = int(row.get("global_step", -1))
        if int(global_step_start) <= global_step <= int(global_step_end):
            rows.append(
                {
                    "global_step": global_step,
                    "objective_episode_index": int(row["objective_episode_index"]),
                    "objective_local_index": int(row["objective_local_index"]),
                    "tokens_to_objective_episode_end": int(row["tokens_to_objective_episode_end"]),
                }
            )
    rows.sort(key=lambda row: int(row["global_step"]))
    return rows


def _extract_current_window_rows(
    *,
    checkpoint_path: str,
    compare_suite_path: str,
    device: str | None,
    n_samples: int,
    single_eval_pos: int,
    batch_size: int,
    n_epochs: int,
    learning_rate: float,
    target_kl: float,
    train_profile: str,
    env_index: int,
    global_step_start: int,
    global_step_end: int,
) -> list[dict[str, Any]]:
    checkpoint_path = str(Path(checkpoint_path).expanduser().resolve())
    compare_suite_path = str(Path(compare_suite_path).expanduser().resolve())
    suite = _load_required_suite(compare_suite_path)
    device_obj = torch.device(str(device or _default_device()))

    load_model.cache_clear()
    model, config = load_model(checkpoint_path, device=device_obj, verbose=False)
    env_cfg = _resolve_audit_env_config(config, core_a=False, reference_semantics_enabled=False)
    prior = EnvironmentPrior(copy.deepcopy(env_cfg))
    prior._sample_batch_hypers = lambda batch_n: _clone_suite_h_list(suite)
    prior._sample_single_eval_pos = lambda n_samples_arg, rng_seed=None: int(single_eval_pos)

    optimizer_cfg = dict(config.get("optimizer", {}))
    profile_cfg = _resolve_train_profile(
        optimizer_cfg=optimizer_cfg,
        train_profile=str(train_profile),
        batch_size=int(batch_size),
        n_epochs=int(n_epochs),
        learning_rate=float(learning_rate),
        target_kl=float(target_kl),
    )

    algo, callback, vec_env = build_recurrent_ppo(
        model=model,
        env_prior=prior,
        device=str(device_obj),
        num_features=int(config["prior"]["num_features"]),
        n_envs=int(suite["batch_size"]),
        n_steps=int(n_samples),
        learning_rate=float(profile_cfg["learning_rate"]),
        batch_size=int(profile_cfg["batch_size"]),
        n_epochs=int(profile_cfg["n_epochs"]),
        gamma=float(optimizer_cfg.get("ppo_gamma", 1.0)),
        gae_lambda=float(optimizer_cfg.get("ppo_gae_lambda", 0.95)),
        clip_range=optimizer_cfg.get("ppo_clip_range", 0.2),
        clip_range_vf=None,
        normalize_advantage=bool(profile_cfg["normalize_advantage"]),
        actor_gae_space=str(profile_cfg["actor_gae_space"]),
        actor_baseline_mode=str(profile_cfg["actor_baseline_mode"]),
        actor_objective_mode=str(profile_cfg["actor_objective_mode"]),
        strict_fixed_env_mode=True,
        env_rng_seeds=[int(v) for v in suite["env_seeds"]],
        rollout_rng_seeds=[int(v) for v in suite["rollout_seeds"]],
        deterministic_actor_sampling=True,
        deterministic_batch_plan=True,
        actor_objective_runtime_current_suite_name="pair2",
        reset_env_state_at_sep=bool(profile_cfg["reset_env_state_at_sep"]),
        separate_value_backbone=bool(profile_cfg["separate_value_backbone"]),
        restore_validation_policy_state=True,
        runtime_normalized_q_value_weight_override=profile_cfg["runtime_normalized_q_value_weight_override"],
        runtime_next_state_flow_matching_weight_override=profile_cfg["runtime_next_state_flow_matching_weight_override"],
        ent_coef=float(optimizer_cfg.get("ppo_ent_coef", 0.0)),
        vf_coef=float(profile_cfg["vf_coef"]),
        max_grad_norm=float(optimizer_cfg.get("ppo_max_grad_norm", 0.5)),
        target_kl=profile_cfg["target_kl"],
        verbose=0,
    )
    _bind_vec_env_to_fixed_suite(vec_env, suite=suite, single_eval_pos=int(single_eval_pos))
    _make_deterministic_batch_plan(algo.rollout_buffer)
    algo.ep_info_buffer = []
    algo.ep_success_buffer = []
    algo._last_obs = vec_env.reset()
    algo._last_episode_starts = np.ones((vec_env.num_envs,), dtype=bool)
    algo._last_lstm_states = algo.policy._dummy_states(vec_env.num_envs)
    callback.init_callback(algo)
    assert algo.collect_rollouts(vec_env, callback, algo.rollout_buffer, n_rollout_steps=algo.n_steps)
    vec_env.close()

    env_rows = _build_env_objective_identity_rows(
        episode_starts=np.asarray(algo.rollout_buffer.episode_starts, dtype=np.float32),
        objective_masks=np.asarray(algo.rollout_buffer.objective_masks, dtype=np.float32),
        env_index=int(env_index),
    )
    out = [
        dict(row)
        for row in env_rows
        if int(global_step_start) <= int(row["global_step"]) <= int(global_step_end)
    ]
    out.sort(key=lambda row: int(row["global_step"]))
    return out


def _summarize_window_segments(
    rows: list[dict[str, Any]],
    *,
    episode_key: str,
    local_key: str,
) -> list[dict[str, int]]:
    if not rows:
        return []
    rows_sorted = sorted(rows, key=lambda row: int(row["global_step"]))
    segments: list[dict[str, int]] = []
    current = {
        "episode_index": int(rows_sorted[0][episode_key]),
        "global_step_start": int(rows_sorted[0]["global_step"]),
        "global_step_end": int(rows_sorted[0]["global_step"]),
        "local_index_start": int(rows_sorted[0][local_key]),
        "local_index_end": int(rows_sorted[0][local_key]),
        "token_count": 1,
    }
    for row in rows_sorted[1:]:
        same_episode = int(row[episode_key]) == int(current["episode_index"])
        contiguous_step = int(row["global_step"]) == int(current["global_step_end"]) + 1
        contiguous_local = int(row[local_key]) == int(current["local_index_end"]) + 1
        if same_episode and contiguous_step and contiguous_local:
            current["global_step_end"] = int(row["global_step"])
            current["local_index_end"] = int(row[local_key])
            current["token_count"] = int(current["token_count"]) + 1
            continue
        segments.append(dict(current))
        current = {
            "episode_index": int(row[episode_key]),
            "global_step_start": int(row["global_step"]),
            "global_step_end": int(row["global_step"]),
            "local_index_start": int(row[local_key]),
            "local_index_end": int(row[local_key]),
            "token_count": 1,
        }
    segments.append(dict(current))
    return segments


def _segment_boundary_steps(segments: list[dict[str, int]]) -> list[int]:
    return [int(seg["global_step_start"]) for seg in list(segments)[1:]]


def build_pair2_env12_window_reset_segment_compare(
    *,
    checkpoint_path: str = PAIR2_CHECKPOINT_PATH,
    compare_suite_path: str | None = None,
    source_json: str = DEFAULT_SOURCE_JSON,
    source_suite_name: str = DEFAULT_SOURCE_SUITE_NAME,
    device: str | None = None,
    n_samples: int = 2048,
    single_eval_pos: int = 1946,
    batch_size: int = 256,
    n_epochs: int = 1,
    learning_rate: float = 0.0002,
    target_kl: float = 0.03,
    train_profile: str = "trusted_sep_reset_mainline",
    phase2_summary_path: str = DEFAULT_PHASE2_SUMMARY,
    env_index: int = 12,
    global_step_start: int = 1946,
    global_step_end: int = 2002,
    allow_source_suite_mismatch: bool = False,
) -> dict[str, Any]:
    assert_phase2_green(phase2_summary_path)
    compare_suite_path, expected_suite_role, compare_suite_role = _resolve_compare_suite_path(
        compare_suite_path=compare_suite_path,
        source_json=source_json,
        allow_source_suite_mismatch=bool(allow_source_suite_mismatch),
    )
    source_rows = _extract_source_window_rows(
        source_json=source_json,
        source_suite_name=source_suite_name,
        env_index=int(env_index),
        global_step_start=int(global_step_start),
        global_step_end=int(global_step_end),
    )
    current_rows = _extract_current_window_rows(
        checkpoint_path=checkpoint_path,
        compare_suite_path=compare_suite_path,
        device=device,
        n_samples=int(n_samples),
        single_eval_pos=int(single_eval_pos),
        batch_size=int(batch_size),
        n_epochs=int(n_epochs),
        learning_rate=float(learning_rate),
        target_kl=float(target_kl),
        train_profile=str(train_profile),
        env_index=int(env_index),
        global_step_start=int(global_step_start),
        global_step_end=int(global_step_end),
    )
    source_segments = _summarize_window_segments(
        source_rows,
        episode_key="objective_episode_index",
        local_key="objective_local_index",
    )
    current_raw_segments = _summarize_window_segments(
        current_rows,
        episode_key="raw_objective_episode_index",
        local_key="raw_objective_episode_position",
    )
    current_compressed_segments = _summarize_window_segments(
        current_rows,
        episode_key="compressed_objective_episode_index",
        local_key="compressed_objective_episode_position",
    )

    target_steps = [1964, 1965, 1966, 1967]
    source_target_rows = [
        dict(row) for row in source_rows if int(row["global_step"]) in set(target_steps)
    ]
    current_target_rows = [
        dict(row) for row in current_rows if int(row["global_step"]) in set(target_steps)
    ]

    source_local = [int(row["objective_local_index"]) for row in source_target_rows]
    current_raw_local = [int(row["raw_objective_episode_position"]) for row in current_target_rows]
    current_raw_global = [int(row["raw_objective_global_position"]) for row in current_target_rows]

    return {
        "audit_entry": "phase3_pair2_env12_window_reset_segment_compare",
        "config": {
            "checkpoint_path": str(Path(checkpoint_path).expanduser().resolve()),
            "compare_suite_path": str(Path(compare_suite_path).expanduser().resolve()),
            "source_json": str(Path(source_json).expanduser().resolve()),
            "device": str(device or _default_device()),
            "n_samples": int(n_samples),
            "single_eval_pos": int(single_eval_pos),
            "batch_size": int(batch_size),
            "n_epochs": int(n_epochs),
            "learning_rate": float(learning_rate),
            "target_kl": float(target_kl),
            "train_profile": str(train_profile),
            "source_suite_name": str(source_suite_name),
            "source_expected_compare_suite_role": str(expected_suite_role),
            "compare_suite_role": str(compare_suite_role),
            "allow_source_suite_mismatch": bool(allow_source_suite_mismatch),
            "strict_fixed_env_mode": True,
            "deterministic_actor_sampling": True,
            "deterministic_batch_plan": True,
            "env_index": int(env_index),
            "global_step_start": int(global_step_start),
            "global_step_end": int(global_step_end),
        },
        "source_window": {
            "row_count": int(len(source_rows)),
            "rows_excerpt_head": source_rows[:8],
            "rows_excerpt_tail": source_rows[-8:],
            "segments": source_segments,
            "segment_boundary_steps_inside_window": _segment_boundary_steps(source_segments),
        },
        "current_strict_window": {
            "row_count": int(len(current_rows)),
            "rows_excerpt_head": current_rows[:8],
            "rows_excerpt_tail": current_rows[-8:],
            "raw_segments": current_raw_segments,
            "compressed_segments": current_compressed_segments,
            "raw_segment_boundary_steps_inside_window": _segment_boundary_steps(current_raw_segments),
            "compressed_segment_boundary_steps_inside_window": _segment_boundary_steps(current_compressed_segments),
        },
        "target_block_alignment": {
            "global_steps": target_steps,
            "source_rows": source_target_rows,
            "current_rows": current_target_rows,
            "source_local_indices": source_local,
            "current_raw_episode_positions": current_raw_local,
            "current_raw_global_positions": current_raw_global,
            "pre_split_token_count": None if not current_compressed_segments else int(current_compressed_segments[0]["token_count"]),
            "local_index_shift_vs_current_raw_episode_positions": [
                int(src - cur) for src, cur in zip(source_local, current_raw_local)
            ],
        },
        "conclusions": {
            "source_window_is_one_long_episode0_segment": bool(
                len(source_segments) == 1 and int(source_segments[0]["episode_index"]) == 0
            ),
            "current_strict_window_is_split_into_multiple_segments": bool(
                len(current_compressed_segments) > 1
            ),
            "current_strict_split_starts_inside_window": _segment_boundary_steps(current_compressed_segments),
            "source_defers_first_boundary_until_after_window": bool(
                len(source_segments) == 1
            ),
            "source_local_indices_align_with_current_raw_global_positions": bool(
                source_local == current_raw_global
            ),
            "source_local_indices_do_not_align_with_current_episode_local_positions": bool(
                source_local != current_raw_local
            ),
            "window_mismatch_is_reset_segment_contract_mismatch": bool(
                len(source_segments) != len(current_compressed_segments)
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-path", type=str, default=PAIR2_CHECKPOINT_PATH)
    parser.add_argument("--compare-suite-path", type=str, default=None)
    parser.add_argument("--train-suite-path", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--source-json", type=str, default=DEFAULT_SOURCE_JSON)
    parser.add_argument("--source-suite-name", type=str, default=DEFAULT_SOURCE_SUITE_NAME)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--n-samples", type=int, default=2048)
    parser.add_argument("--single-eval-pos", type=int, default=1946)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=0.0002)
    parser.add_argument("--target-kl", type=float, default=0.03)
    parser.add_argument("--train-profile", type=str, default="trusted_sep_reset_mainline")
    parser.add_argument("--phase2-summary-path", type=str, default=DEFAULT_PHASE2_SUMMARY)
    parser.add_argument("--env-index", type=int, default=12)
    parser.add_argument("--global-step-start", type=int, default=1946)
    parser.add_argument("--global-step-end", type=int, default=2002)
    parser.add_argument("--allow-source-suite-mismatch", action="store_true")
    parser.add_argument("--output-json", type=str, required=True)
    args = parser.parse_args()
    if args.compare_suite_path is not None and args.train_suite_path is not None:
        raise ValueError("Use only one of --compare-suite-path or deprecated --train-suite-path.")

    report = build_pair2_env12_window_reset_segment_compare(
        checkpoint_path=args.checkpoint_path,
        compare_suite_path=args.compare_suite_path or args.train_suite_path,
        source_json=args.source_json,
        source_suite_name=args.source_suite_name,
        device=args.device,
        n_samples=args.n_samples,
        single_eval_pos=args.single_eval_pos,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        learning_rate=args.learning_rate,
        target_kl=args.target_kl,
        train_profile=args.train_profile,
        phase2_summary_path=args.phase2_summary_path,
        env_index=args.env_index,
        global_step_start=args.global_step_start,
        global_step_end=args.global_step_end,
        allow_source_suite_mismatch=bool(args.allow_source_suite_mismatch),
    )
    output_path = Path(args.output_json).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
