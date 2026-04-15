import argparse
import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ticl.analysis.critic_free_single_env_audit import _make_deterministic_batch_plan
from ticl.analysis.phase2_guardrail import DEFAULT_PHASE2_SUMMARY, assert_phase2_green
from ticl.analysis.phase3_heldout_reset_semantics_probe import _objective_episode_segments
from ticl.analysis.phase3_multi_env_optimization_audit import (
    _bind_vec_env_to_fixed_suite,
    _clone_suite_h_list,
    _default_device,
    _load_required_suite,
    _resolve_audit_env_config,
    _resolve_train_profile,
)
from ticl.analysis.phase3_pair2_train_update_ab_compare_pack import (
    PAIR2_CHECKPOINT_PATH,
    PAIR2_TRAIN_SUITE_PATH,
)
from ticl.model_builder import load_model
from ticl.priors.environment_prior import EnvironmentPrior
from ticl.sb3_recurrent_ppo import _build_objective_position_metadata, build_recurrent_ppo


DEFAULT_SOURCE_COMPARE_PACK_JSON = (
    "/home/chen/RLPFN/artifacts/phase3_env12_mid_episode_extension_compare_pack.json"
)


def _build_env_objective_identity_rows(
    *,
    episode_starts: np.ndarray,
    objective_masks: np.ndarray,
    env_index: int,
) -> list[dict[str, int]]:
    metadata = _build_objective_position_metadata(
        episode_starts=np.asarray(episode_starts, dtype=np.float32),
        objective_masks=np.asarray(objective_masks, dtype=np.float32),
    )
    env_idx = int(env_index)
    objective_mask_env = np.asarray(objective_masks[:, env_idx], dtype=bool).reshape(-1)
    episode_starts_env = np.asarray(episode_starts[:, env_idx], dtype=bool).reshape(-1)
    objective_steps = np.flatnonzero(objective_mask_env).astype(np.int64, copy=False)
    segments = _objective_episode_segments(objective_mask_env, episode_starts_env)
    compressed_episode_index_by_step = {}
    compressed_episode_position_by_step = {}
    for compressed_episode_index, (start, end) in enumerate(segments):
        segment_steps = objective_steps[int(start) : int(end)]
        for compressed_episode_position, global_step in enumerate(segment_steps.tolist()):
            compressed_episode_index_by_step[int(global_step)] = int(compressed_episode_index)
            compressed_episode_position_by_step[int(global_step)] = int(compressed_episode_position)

    rows: list[dict[str, int]] = []
    for global_step in objective_steps.tolist():
        rows.append(
            {
                "global_step": int(global_step),
                "episode_start": int(bool(episode_starts_env[int(global_step)])),
                "raw_objective_episode_index": int(
                    metadata["objective_episode_indices"][int(global_step), env_idx]
                ),
                "raw_objective_episode_position": int(
                    metadata["objective_episode_positions"][int(global_step), env_idx]
                ),
                "raw_objective_global_position": int(
                    metadata["objective_global_positions"][int(global_step), env_idx]
                ),
                "compressed_objective_episode_index": int(
                    compressed_episode_index_by_step[int(global_step)]
                ),
                "compressed_objective_episode_position": int(
                    compressed_episode_position_by_step[int(global_step)]
                ),
            }
        )
    return rows


def _summarize_rows_by_key(
    rows: list[dict[str, int]],
    *,
    episode_key: str,
    position_key: str,
) -> list[dict[str, int]]:
    grouped: dict[int, list[dict[str, int]]] = {}
    for row in rows:
        grouped.setdefault(int(row[episode_key]), []).append(dict(row))
    out: list[dict[str, int]] = []
    for episode_index in sorted(grouped):
        group = grouped[int(episode_index)]
        steps = [int(row["global_step"]) for row in group]
        positions = [int(row[position_key]) for row in group]
        out.append(
            {
                "episode_index": int(episode_index),
                "token_count": int(len(group)),
                "global_step_min": int(min(steps)),
                "global_step_max": int(max(steps)),
                "position_min": int(min(positions)),
                "position_max": int(max(positions)),
            }
        )
    return out


def _rows_for_global_steps(rows: list[dict[str, int]], target_global_steps: list[int]) -> list[dict[str, int]]:
    target_set = {int(v) for v in target_global_steps}
    out = [dict(row) for row in rows if int(row["global_step"]) in target_set]
    out.sort(key=lambda row: int(row["global_step"]))
    return out


def _strip_to_raw_identity(rows: list[dict[str, int]]) -> list[dict[str, int]]:
    return [
        {
            "global_step": int(row["global_step"]),
            "raw_objective_episode_index": int(row["raw_objective_episode_index"]),
            "raw_objective_episode_position": int(row["raw_objective_episode_position"]),
            "raw_objective_global_position": int(row["raw_objective_global_position"]),
        }
        for row in rows
    ]


def _load_source_block(compare_pack_json: str) -> dict[str, Any]:
    payload = json.loads(Path(compare_pack_json).expanduser().resolve().read_text())
    extension_block = dict(payload["comparison"]["extension_block"])
    source_env12_json = payload.get("source_env12_json", None)
    source_rows = []
    if source_env12_json is not None:
        source_payload = json.loads(Path(source_env12_json).expanduser().resolve().read_text())
        target_steps = {int(v) for v in extension_block["global_steps"]}
        target_locals = {int(v) for v in extension_block["local_indices"]}
        for row in list(source_payload.get("start_rows", [])):
            if int(row.get("start_global_step", -1)) in target_steps and int(
                row.get("start_objective_local_index", -999)
            ) in target_locals:
                source_rows.append(
                    {
                        "global_step": int(row["start_global_step"]),
                        "source_objective_episode_index": int(row["start_objective_episode_index"]),
                        "source_objective_local_index": int(row["start_objective_local_index"]),
                        "tokens_to_objective_episode_end": int(row["start_tokens_to_objective_episode_end"]),
                    }
                )
        source_rows.sort(key=lambda row: int(row["global_step"]))
    return {
        "source_compare_pack_json": str(Path(compare_pack_json).expanduser().resolve()),
        "source_env12_json": None
        if source_env12_json is None
        else str(Path(source_env12_json).expanduser().resolve()),
        "global_steps": [int(v) for v in extension_block["global_steps"]],
        "local_indices": [int(v) for v in extension_block["local_indices"]],
        "tokens_to_objective_episode_end": [int(v) for v in extension_block["tokens_to_objective_episode_end"]],
        "source_rows": source_rows,
    }


def _init_algo_runtime(algo, callback, vec_env) -> None:
    algo.ep_info_buffer = []
    algo.ep_success_buffer = []
    algo._last_obs = vec_env.reset()
    algo._last_episode_starts = np.ones((vec_env.num_envs,), dtype=bool)
    algo._last_lstm_states = algo.policy._dummy_states(vec_env.num_envs)
    callback.init_callback(algo)


def build_pair2_train_segment_identity_probe(
    *,
    checkpoint_path: str = PAIR2_CHECKPOINT_PATH,
    train_suite_path: str = PAIR2_TRAIN_SUITE_PATH,
    device: str | None = None,
    n_samples: int = 2048,
    single_eval_pos: int = 1946,
    batch_size: int = 256,
    n_epochs: int = 1,
    learning_rate: float = 0.0002,
    target_kl: float = 0.03,
    train_profile: str = "trusted_sep_reset_mainline",
    phase2_summary_path: str = DEFAULT_PHASE2_SUMMARY,
    source_compare_pack_json: str = DEFAULT_SOURCE_COMPARE_PACK_JSON,
    env_index: int = 12,
    outer_batch_idx: int = 104,
) -> dict[str, Any]:
    assert_phase2_green(phase2_summary_path)
    checkpoint_path = str(Path(checkpoint_path).expanduser().resolve())
    train_suite_path = str(Path(train_suite_path).expanduser().resolve())
    suite = _load_required_suite(train_suite_path)
    device_obj = torch.device(str(device or _default_device()))

    load_model.cache_clear()
    model, config = load_model(checkpoint_path, device=device_obj, verbose=False)
    env_cfg = _resolve_audit_env_config(
        config,
        core_a=False,
        reference_semantics_enabled=False,
    )
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
    _init_algo_runtime(algo, callback, vec_env)
    assert algo.collect_rollouts(vec_env, callback, algo.rollout_buffer, n_rollout_steps=algo.n_steps)
    _make_deterministic_batch_plan(algo.rollout_buffer)

    source_block = _load_source_block(source_compare_pack_json)
    env_rows = _build_env_objective_identity_rows(
        episode_starts=np.asarray(algo.rollout_buffer.episode_starts, dtype=np.float32),
        objective_masks=np.asarray(algo.rollout_buffer.objective_masks, dtype=np.float32),
        env_index=int(env_index),
    )
    target_global_steps = list(source_block["global_steps"])
    rollout_stage_rows = _rows_for_global_steps(env_rows, target_global_steps)

    total_steps = int(algo.rollout_buffer.buffer_size * algo.rollout_buffer.n_envs)
    batch_plan_entries = list(algo.rollout_buffer._iter_batch_plan(int(batch_size)))
    target_batch_inds, target_env_change = batch_plan_entries[int(outer_batch_idx) - 1]
    algo.rollout_buffer._ensure_generator_ready()
    flat_all = algo.rollout_buffer.get_flat_position_metadata(np.arange(total_steps, dtype=np.int64))
    flat_rows = [
        {
            "flat_batch_index": int(flat_idx),
            "env_index": int(env_idx),
            "global_step": int(step_idx),
            "raw_objective_episode_index": int(ep_idx),
            "raw_objective_episode_position": int(pos_idx),
            "raw_objective_global_position": int(global_pos_idx),
        }
        for flat_idx, env_idx, step_idx, ep_idx, pos_idx, global_pos_idx in zip(
            flat_all["flat_batch_indices"].tolist(),
            flat_all["flat_env_indices"].tolist(),
            flat_all["flat_step_indices"].tolist(),
            flat_all["flat_objective_episode_indices"].tolist(),
            flat_all["flat_objective_episode_positions"].tolist(),
            flat_all["flat_objective_global_positions"].tolist(),
        )
        if int(env_idx) == int(env_index) and int(pos_idx) >= 0
    ]
    flat_stage_rows = [
        dict(row)
        for row in flat_rows
        if int(row["global_step"]) in {int(v) for v in target_global_steps}
    ]
    flat_stage_rows.sort(key=lambda row: int(row["global_step"]))

    target_rollout_data = algo.rollout_buffer._get_flat_batch_gpu(target_batch_inds, target_env_change)
    batch_rows = [
        {
            "flat_batch_index": int(flat_idx),
            "env_index": int(env_idx),
            "global_step": int(step_idx),
            "raw_objective_episode_index": int(ep_idx),
            "raw_objective_episode_position": int(pos_idx),
            "raw_objective_global_position": int(global_pos_idx),
        }
        for flat_idx, env_idx, step_idx, ep_idx, pos_idx, global_pos_idx in zip(
            np.asarray(target_rollout_data.flat_batch_indices, dtype=np.int64).tolist(),
            np.asarray(target_rollout_data.flat_env_indices, dtype=np.int64).tolist(),
            np.asarray(target_rollout_data.flat_step_indices, dtype=np.int64).tolist(),
            np.asarray(target_rollout_data.flat_objective_episode_indices, dtype=np.int64).tolist(),
            np.asarray(target_rollout_data.flat_objective_episode_positions, dtype=np.int64).tolist(),
            np.asarray(target_rollout_data.flat_objective_global_positions, dtype=np.int64).tolist(),
        )
        if int(env_idx) == int(env_index) and int(pos_idx) >= 0
    ]
    outer_batch_target_rows = [
        dict(row)
        for row in batch_rows
        if int(row["global_step"]) in {int(v) for v in target_global_steps}
    ]
    outer_batch_target_rows.sort(key=lambda row: int(row["global_step"]))

    raw_episode_spans = _summarize_rows_by_key(
        env_rows,
        episode_key="raw_objective_episode_index",
        position_key="raw_objective_episode_position",
    )
    compressed_episode_spans = _summarize_rows_by_key(
        env_rows,
        episode_key="compressed_objective_episode_index",
        position_key="compressed_objective_episode_position",
    )

    objective_rows_present = [row for row in batch_rows]
    current_target_raw_episode_indices = sorted(
        {int(row["raw_objective_episode_index"]) for row in rollout_stage_rows}
    )
    current_target_compressed_episode_indices = sorted(
        {int(row["compressed_objective_episode_index"]) for row in rollout_stage_rows}
    )
    current_target_raw_episode_positions = [int(row["raw_objective_episode_position"]) for row in rollout_stage_rows]
    current_target_raw_global_positions = [int(row["raw_objective_global_position"]) for row in rollout_stage_rows]
    current_target_compressed_episode_positions = [
        int(row["compressed_objective_episode_position"]) for row in rollout_stage_rows
    ]
    source_episode_indices = sorted(
        {int(row["source_objective_episode_index"]) for row in source_block.get("source_rows", [])}
    )
    source_local_indices = [int(v) for v in source_block["local_indices"]]

    vec_env.close()

    return {
        "audit_entry": "phase3_pair2_train_segment_identity_probe",
        "config": {
            "checkpoint_path": checkpoint_path,
            "train_suite_path": train_suite_path,
            "device": str(device_obj),
            "n_samples": int(n_samples),
            "single_eval_pos": int(single_eval_pos),
            "batch_size": int(batch_size),
            "n_epochs": int(n_epochs),
            "learning_rate": float(learning_rate),
            "target_kl": float(target_kl),
            "train_profile": str(train_profile),
            "strict_fixed_env_mode": True,
            "deterministic_actor_sampling": True,
            "deterministic_batch_plan": True,
            "env_index": int(env_index),
            "outer_batch_idx": int(outer_batch_idx),
            "source_compare_pack_json": str(Path(source_compare_pack_json).expanduser().resolve()),
        },
        "source_block": source_block,
        "rollout_stage": {
            "target_global_step_rows": rollout_stage_rows,
            "raw_objective_episode_spans": raw_episode_spans,
            "compressed_objective_episode_spans": compressed_episode_spans,
        },
        "flatten_stage": {
            "target_global_step_rows": flat_stage_rows,
        },
        "outer_batch_stage": {
            "outer_batch_idx": int(outer_batch_idx),
            "target_global_step_rows": outer_batch_target_rows,
            "env_objective_row_count": int(len(objective_rows_present)),
            "raw_objective_episode_indices_present": sorted(
                {int(row["raw_objective_episode_index"]) for row in objective_rows_present}
            ),
            "raw_objective_episode_spans_present": _summarize_rows_by_key(
                objective_rows_present,
                episode_key="raw_objective_episode_index",
                position_key="raw_objective_episode_position",
            ),
        },
        "conclusions": {
            "source_block_is_episode0_local_under_source_artifact": bool(source_episode_indices == [0]),
            "train_side_rollout_uses_raw_rollout_episode_identity": bool(
                current_target_raw_episode_indices != current_target_compressed_episode_indices
            ),
            "flatten_stage_preserves_rollout_raw_identity": bool(
                _strip_to_raw_identity(rollout_stage_rows) == _strip_to_raw_identity(flat_stage_rows)
            ),
            "outer_batch_104_preserves_flat_raw_identity": bool(
                flat_stage_rows == outer_batch_target_rows
            ),
            "current_strict_train_side_target_is_not_episode0": bool(
                current_target_raw_episode_indices != [0]
            ),
            "source_local_indices_match_current_raw_global_positions": bool(
                source_local_indices == current_target_raw_global_positions
            ),
            "source_local_indices_do_not_match_current_raw_episode_positions": bool(
                source_local_indices != current_target_raw_episode_positions
            ),
            "source_local_indices_do_not_match_current_compressed_episode_positions": bool(
                source_local_indices != current_target_compressed_episode_positions
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-path", type=str, default=PAIR2_CHECKPOINT_PATH)
    parser.add_argument("--train-suite-path", type=str, default=PAIR2_TRAIN_SUITE_PATH)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--n-samples", type=int, default=2048)
    parser.add_argument("--single-eval-pos", type=int, default=1946)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=0.0002)
    parser.add_argument("--target-kl", type=float, default=0.03)
    parser.add_argument("--train-profile", type=str, default="trusted_sep_reset_mainline")
    parser.add_argument("--phase2-summary-path", type=str, default=DEFAULT_PHASE2_SUMMARY)
    parser.add_argument("--source-compare-pack-json", type=str, default=DEFAULT_SOURCE_COMPARE_PACK_JSON)
    parser.add_argument("--env-index", type=int, default=12)
    parser.add_argument("--outer-batch-idx", type=int, default=104)
    parser.add_argument("--output-json", type=str, required=True)
    args = parser.parse_args()

    report = build_pair2_train_segment_identity_probe(
        checkpoint_path=args.checkpoint_path,
        train_suite_path=args.train_suite_path,
        device=args.device,
        n_samples=args.n_samples,
        single_eval_pos=args.single_eval_pos,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        learning_rate=args.learning_rate,
        target_kl=args.target_kl,
        train_profile=args.train_profile,
        phase2_summary_path=args.phase2_summary_path,
        source_compare_pack_json=args.source_compare_pack_json,
        env_index=args.env_index,
        outer_batch_idx=args.outer_batch_idx,
    )
    output_path = Path(args.output_json).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
