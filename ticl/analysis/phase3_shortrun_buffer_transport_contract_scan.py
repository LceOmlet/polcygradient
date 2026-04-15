import argparse
import copy
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ticl.analysis.phase2_guardrail import DEFAULT_PHASE2_SUMMARY, assert_phase2_green
from ticl.analysis.critic_free_single_env_audit import _make_deterministic_batch_plan
from ticl.analysis.phase3_multi_env_optimization_audit import _load_required_suite
from ticl.analysis.phase3_residual_missed_anchor_sign_probe import DEFAULT_SUITE_SPECS
from ticl.analysis.phase3_shortrun_sequence_layout_source_probe import (
    _build_many_env_algo,
    _hash_ndarray,
    _to_numpy,
)
from ticl.sb3_recurrent_ppo import _slice_masked_rollout_sequence_batch


def _default_device() -> str:
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def _load_suite_specs(path: str | None) -> list[dict[str, Any]]:
    if path is None:
        return [dict(spec) for spec in list(DEFAULT_SUITE_SPECS)]
    payload = json.loads(Path(path).expanduser().resolve().read_text())
    return [dict(spec) for spec in list(payload)]


def _select_suite(spec: dict[str, Any], *, suite_split: str) -> tuple[dict[str, Any], str]:
    split = str(suite_split).strip().lower()
    if split == "train":
        return _load_required_suite(str(spec["train_suite_path"])), str(Path(spec["train_suite_path"]).expanduser().resolve())
    if split == "heldout":
        return _load_required_suite(str(spec["heldout_suite_path"])), str(
            Path(spec["heldout_suite_path"]).expanduser().resolve()
        )
    raise ValueError(f"Unsupported suite_split={suite_split!r}")


def _max_abs_diff(lhs: np.ndarray, rhs: np.ndarray) -> float:
    lhs = np.asarray(lhs, dtype=np.float32).reshape(-1)
    rhs = np.asarray(rhs, dtype=np.float32).reshape(-1)
    if lhs.shape != rhs.shape:
        raise ValueError(f"shape mismatch: {tuple(lhs.shape)} vs {tuple(rhs.shape)}")
    if lhs.size == 0:
        return 0.0
    return float(np.max(np.abs(lhs - rhs)))


def _compare_flat_series(source: np.ndarray, observed: np.ndarray) -> dict[str, Any]:
    source_np = np.asarray(source, dtype=np.float32).reshape(-1)
    observed_np = np.asarray(observed, dtype=np.float32).reshape(-1)
    if source_np.shape != observed_np.shape:
        raise ValueError(f"shape mismatch: {tuple(source_np.shape)} vs {tuple(observed_np.shape)}")
    diffs = np.abs(source_np - observed_np)
    return {
        "exact_match": bool(np.array_equal(source_np, observed_np)),
        "mismatch_count": int(np.count_nonzero(diffs > 1e-8)),
        "max_abs_diff": float(diffs.max(initial=0.0)),
        "mean_abs_diff": float(diffs.mean() if diffs.size > 0 else 0.0),
    }


def _compare_transport_field(
    *,
    source_flat: np.ndarray,
    flat_batch_field: np.ndarray,
    padded_batch_field: np.ndarray,
    padded_valid_mask: np.ndarray,
) -> dict[str, Any]:
    source_np = np.asarray(source_flat, dtype=np.float32).reshape(-1)
    flat_np = np.asarray(flat_batch_field, dtype=np.float32).reshape(-1)
    padded_np = np.asarray(padded_batch_field, dtype=np.float32).reshape(-1)
    valid_mask = np.asarray(padded_valid_mask, dtype=bool).reshape(-1)
    padded_valid_np = padded_np[valid_mask]
    source_to_flat = _compare_flat_series(source_np, flat_np)
    source_to_padded = _compare_flat_series(source_np, padded_valid_np)
    flat_to_padded = _compare_flat_series(flat_np, padded_valid_np)
    return {
        "source_to_flat": source_to_flat,
        "source_to_padded_valid": source_to_padded,
        "flat_to_padded_valid": flat_to_padded,
    }


def _compare_seq_layout(flat_seq_start_indices: np.ndarray, flat_seq_lengths: np.ndarray, padded_mask: np.ndarray) -> dict[str, Any]:
    flat_seq_start_indices = np.asarray(flat_seq_start_indices, dtype=np.int64).reshape(-1)
    flat_seq_lengths = np.asarray(flat_seq_lengths, dtype=np.int64).reshape(-1)
    padded_mask = np.asarray(padded_mask, dtype=np.float32).reshape(-1)
    if int(flat_seq_start_indices.size) != int(flat_seq_lengths.size):
        raise ValueError(
            f"seq layout mismatch: {int(flat_seq_start_indices.size)} starts vs {int(flat_seq_lengths.size)} lengths"
        )
    expected_starts = np.zeros_like(flat_seq_start_indices)
    if flat_seq_lengths.size > 0:
        expected_starts[1:] = np.cumsum(flat_seq_lengths[:-1], dtype=np.int64)
    valid_count = int(np.count_nonzero(padded_mask > 1e-8))
    if flat_seq_lengths.size == 0:
        padded_seq_lengths = np.asarray([], dtype=np.int64)
    else:
        padded_seq_lengths = np.asarray(padded_mask.reshape((int(flat_seq_lengths.size), -1)) > 1e-8, dtype=np.int64).sum(axis=1)
    return {
        "seq_start_indices_exact_match": bool(np.array_equal(flat_seq_start_indices, expected_starts)),
        "seq_lengths_exact_match": bool(np.array_equal(flat_seq_lengths, padded_seq_lengths)),
        "seq_lengths_max_abs_diff": _max_abs_diff(flat_seq_lengths.astype(np.float32), padded_seq_lengths.astype(np.float32)),
        "padded_valid_count_match": bool(int(flat_seq_lengths.sum()) == int(valid_count)),
        "expected_seq_start_indices_hash": _hash_ndarray(expected_starts.astype(np.int64, copy=False)),
        "flat_seq_start_indices_hash": _hash_ndarray(flat_seq_start_indices.astype(np.int64, copy=False)),
        "flat_seq_lengths_hash": _hash_ndarray(flat_seq_lengths.astype(np.int64, copy=False)),
        "padded_seq_lengths_hash": _hash_ndarray(padded_seq_lengths.astype(np.int64, copy=False)),
    }


def _build_transport_report(
    *,
    algo,
    batch_plan: list[tuple[np.ndarray, np.ndarray]],
    flat_batches,
    padded_batches,
) -> dict[str, Any]:
    transport_fields = (
        "objective_masks",
        "episode_starts",
        "rollout_return_means",
        "rollout_return_stds",
    )
    field_stats: dict[str, dict[str, Any]] = {
        field: {
            "source_to_flat_mismatch_count": 0,
            "source_to_flat_max_abs_diff": 0.0,
            "source_to_padded_valid_mismatch_count": 0,
            "source_to_padded_valid_max_abs_diff": 0.0,
            "flat_to_padded_valid_mismatch_count": 0,
            "flat_to_padded_valid_max_abs_diff": 0.0,
            "subbatch_flat_to_padded_mismatch_count": 0,
            "subbatch_flat_to_padded_max_abs_diff": 0.0,
            "first_mismatch_examples": [],
        }
        for field in transport_fields
    }
    overall_examples: list[dict[str, Any]] = []
    batch_count = 0
    source_to_flat_mismatch_count = 0
    source_to_padded_mismatch_count = 0
    flat_to_padded_mismatch_count = 0
    subbatch_mismatch_count = 0
    seq_layout_mismatch_count = 0
    subbatch_count_total = 0
    max_abs_diff_by_field = {field: 0.0 for field in transport_fields}

    for batch_idx, ((batch_inds, env_change), flat_batch, padded_batch) in enumerate(
        zip(batch_plan, flat_batches, padded_batches)
    ):
        batch_count += 1
        del env_change
        valid_mask = _to_numpy(padded_batch.mask).reshape(-1) > 1e-8
        n_seq = int(flat_batch.n_seq)
        seq_lengths = np.asarray(flat_batch.seq_lengths, dtype=np.int64).reshape(-1)
        seq_start_indices = np.asarray(flat_batch.seq_start_indices, dtype=np.int64).reshape(-1)
        if seq_lengths.size != seq_start_indices.size:
            seq_layout_mismatch_count += 1
            overall_examples.append(
                {
                    "batch_idx": int(batch_idx),
                    "kind": "flat_seq_layout_shape_mismatch",
                    "seq_start_count": int(seq_start_indices.size),
                    "seq_length_count": int(seq_lengths.size),
                }
            )
            continue

        seq_layout = _compare_seq_layout(
            seq_start_indices,
            seq_lengths,
            _to_numpy(padded_batch.mask),
        )
        if not bool(seq_layout["seq_start_indices_exact_match"]) or not bool(seq_layout["seq_lengths_exact_match"]):
            seq_layout_mismatch_count += 1
            overall_examples.append(
                {
                    "batch_idx": int(batch_idx),
                    "kind": "seq_layout_mismatch",
                    "seq_layout": seq_layout,
                }
            )

        source_fields = {
            field: np.asarray(getattr(algo.rollout_buffer, field)[batch_inds], dtype=np.float32).reshape(-1)
            for field in transport_fields
        }
        flat_fields = {field: _to_numpy(getattr(flat_batch, field)).reshape(-1) for field in transport_fields}
        padded_fields = {field: _to_numpy(getattr(padded_batch, field)).reshape(-1) for field in transport_fields}

        for field in transport_fields:
            comparison = _compare_transport_field(
                source_flat=source_fields[field],
                flat_batch_field=flat_fields[field],
                padded_batch_field=padded_fields[field],
                padded_valid_mask=valid_mask,
            )
            source_to_flat = comparison["source_to_flat"]
            source_to_padded = comparison["source_to_padded_valid"]
            flat_to_padded = comparison["flat_to_padded_valid"]
            field_stats[field]["source_to_flat_mismatch_count"] += int(source_to_flat["mismatch_count"])
            field_stats[field]["source_to_flat_max_abs_diff"] = max(
                float(field_stats[field]["source_to_flat_max_abs_diff"]),
                float(source_to_flat["max_abs_diff"]),
            )
            field_stats[field]["source_to_padded_valid_mismatch_count"] += int(source_to_padded["mismatch_count"])
            field_stats[field]["source_to_padded_valid_max_abs_diff"] = max(
                float(field_stats[field]["source_to_padded_valid_max_abs_diff"]),
                float(source_to_padded["max_abs_diff"]),
            )
            field_stats[field]["flat_to_padded_valid_mismatch_count"] += int(flat_to_padded["mismatch_count"])
            field_stats[field]["flat_to_padded_valid_max_abs_diff"] = max(
                float(field_stats[field]["flat_to_padded_valid_max_abs_diff"]),
                float(flat_to_padded["max_abs_diff"]),
            )
            max_abs_diff_by_field[field] = max(
                float(max_abs_diff_by_field[field]),
                float(source_to_flat["max_abs_diff"]),
                float(source_to_padded["max_abs_diff"]),
                float(flat_to_padded["max_abs_diff"]),
            )
            if (
                not bool(source_to_flat["exact_match"])
                or not bool(source_to_padded["exact_match"])
                or not bool(flat_to_padded["exact_match"])
            ):
                source_to_flat_mismatch_count += int(source_to_flat["mismatch_count"])
                source_to_padded_mismatch_count += int(source_to_padded["mismatch_count"])
                flat_to_padded_mismatch_count += int(flat_to_padded["mismatch_count"])
                if len(field_stats[field]["first_mismatch_examples"]) < 2:
                    field_stats[field]["first_mismatch_examples"].append(
                        {
                            "batch_idx": int(batch_idx),
                            "field": str(field),
                            "source_to_flat": source_to_flat,
                            "source_to_padded_valid": source_to_padded,
                            "flat_to_padded_valid": flat_to_padded,
                        }
                    )
                if len(overall_examples) < 5:
                    overall_examples.append(
                        {
                            "batch_idx": int(batch_idx),
                            "field": str(field),
                            "source_to_flat": source_to_flat,
                            "source_to_padded_valid": source_to_padded,
                            "flat_to_padded_valid": flat_to_padded,
                        }
                    )

        seq_subbatch_size = int(algo._resolve_sequence_subbatch_size(flat_batch))
        for start_seq in range(0, n_seq, seq_subbatch_size):
            end_seq = min(n_seq, start_seq + seq_subbatch_size)
            flat_start = int(seq_start_indices[start_seq])
            flat_end = int(seq_start_indices[end_seq]) if int(end_seq) < int(n_seq) else int(len(flat_batch.observations))
            padded_subbatch = _slice_masked_rollout_sequence_batch(
                padded_batch,
                start_seq=int(start_seq),
                end_seq=int(end_seq),
            )
            subbatch_count_total += 1
            padded_sub_valid = _to_numpy(padded_subbatch.mask).reshape(-1) > 1e-8
            for field in transport_fields:
                flat_slice = flat_fields[field][flat_start:flat_end]
                padded_sub_slice = _to_numpy(getattr(padded_subbatch, field)).reshape(-1)[padded_sub_valid]
                sub_cmp = _compare_flat_series(flat_slice, padded_sub_slice)
                field_stats[field]["subbatch_flat_to_padded_mismatch_count"] += int(sub_cmp["mismatch_count"])
                field_stats[field]["subbatch_flat_to_padded_max_abs_diff"] = max(
                    float(field_stats[field]["subbatch_flat_to_padded_max_abs_diff"]),
                    float(sub_cmp["max_abs_diff"]),
                )
                if not bool(sub_cmp["exact_match"]):
                    subbatch_mismatch_count += int(sub_cmp["mismatch_count"])
                    if len(field_stats[field]["first_mismatch_examples"]) < 2:
                        field_stats[field]["first_mismatch_examples"].append(
                            {
                                "batch_idx": int(batch_idx),
                                "start_seq": int(start_seq),
                                "end_seq": int(end_seq),
                                "field": str(field),
                                "subbatch_comparison": sub_cmp,
                            }
                        )
                    if len(overall_examples) < 5:
                        overall_examples.append(
                            {
                                "batch_idx": int(batch_idx),
                                "start_seq": int(start_seq),
                                "end_seq": int(end_seq),
                                "field": str(field),
                                "subbatch_comparison": sub_cmp,
                            }
                        )

        batch_seq_lengths = np.asarray(flat_batch.seq_lengths, dtype=np.int64).reshape(-1)
        padded_seq_lengths = _to_numpy(padded_batch.mask).reshape(n_seq, -1) > 1e-8
        padded_seq_lengths = padded_seq_lengths.sum(axis=1).astype(np.int64, copy=False)
        if not np.array_equal(batch_seq_lengths, padded_seq_lengths):
            seq_layout_mismatch_count += 1
            overall_examples.append(
                {
                    "batch_idx": int(batch_idx),
                    "kind": "seq_length_mismatch",
                    "flat_seq_lengths": batch_seq_lengths.tolist(),
                    "padded_seq_lengths": padded_seq_lengths.tolist(),
                }
            )
    if not (
        int(batch_count) == int(len(batch_plan))
        and int(batch_count) == int(len(flat_batches))
        and int(batch_count) == int(len(padded_batches))
    ):
        raise RuntimeError(
            "transport scan batch iterator length mismatch: "
            f"plan={int(len(batch_plan))}, flat={int(len(flat_batches))}, padded={int(len(padded_batches))}, "
            f"scanned={int(batch_count)}"
        )

    transport_ok = bool(
        batch_count > 0
        and seq_layout_mismatch_count == 0
        and source_to_flat_mismatch_count == 0
        and source_to_padded_mismatch_count == 0
        and flat_to_padded_mismatch_count == 0
        and subbatch_mismatch_count == 0
    )
    return {
        "batch_count": int(batch_count),
        "subbatch_count": int(subbatch_count_total),
        "transport_contract_ok": bool(transport_ok),
        "field_stats": field_stats,
        "seq_layout_mismatch_count": int(seq_layout_mismatch_count),
        "source_to_flat_mismatch_count": int(source_to_flat_mismatch_count),
        "source_to_padded_valid_mismatch_count": int(source_to_padded_mismatch_count),
        "flat_to_padded_valid_mismatch_count": int(flat_to_padded_mismatch_count),
        "subbatch_flat_to_padded_mismatch_count": int(subbatch_mismatch_count),
        "max_abs_diff_by_field": {field: float(value) for field, value in max_abs_diff_by_field.items()},
        "first_mismatch_examples": overall_examples[:5],
    }


def _run_one_suite_transport_contract(
    *,
    checkpoint_path: str,
    suite_spec: dict[str, Any],
    suite_split: str,
    device_obj: torch.device,
    n_samples: int,
    single_eval_pos: int,
    train_profile: str,
) -> dict[str, Any]:
    suite, suite_path = _select_suite(suite_spec, suite_split=str(suite_split))
    train_suite = {
        "h_list": [copy.deepcopy(h) for h in list(suite["h_list"])],
        "env_seeds": [int(v) for v in suite["env_seeds"]],
        "rollout_seeds": [int(v) for v in suite["rollout_seeds"]],
        "batch_size": int(suite["batch_size"]),
        "suite_seed": suite.get("suite_seed", None),
    }
    algo, callback, vec_env = _build_many_env_algo(
        checkpoint_path=str(checkpoint_path),
        train_suite=train_suite,
        device_obj=device_obj,
        n_samples=int(n_samples),
        single_eval_pos=int(single_eval_pos),
        train_profile=str(train_profile),
    )
    try:
        _make_deterministic_batch_plan(algo.rollout_buffer)
        algo.ep_info_buffer = []
        algo.ep_success_buffer = []
        algo.policy.reset_rollout_cache(vec_env.num_envs)
        algo._last_obs = vec_env.reset()
        algo._last_episode_starts = np.ones((vec_env.num_envs,), dtype=bool)
        algo._last_lstm_states = algo.policy._dummy_states(vec_env.num_envs)
        callback.init_callback(algo)
        algo._update_current_progress_remaining(0, int(n_samples))
        import time

        t0 = time.perf_counter()
        ok = algo.collect_rollouts(vec_env, callback, algo.rollout_buffer, n_rollout_steps=algo.n_steps)
        runtime_wall_s = float(time.perf_counter() - t0)
        if not ok:
            raise RuntimeError("collect_rollouts returned False")

        batch_plan = list(algo.rollout_buffer._iter_batch_plan(algo.batch_size))
        flat_batches = list(algo.rollout_buffer.get_gpu_flat(algo.batch_size))
        padded_batches = list(algo.rollout_buffer.get_gpu(algo.batch_size))
        transport_report = _build_transport_report(
            algo=algo,
            batch_plan=batch_plan,
            flat_batches=flat_batches,
            padded_batches=padded_batches,
        )
        return {
            "suite_name": str(suite_spec["suite_name"]),
            "suite_split": str(suite_split),
            "suite_path": str(suite_path),
            "batch_size": int(train_suite["batch_size"]),
            "env_seeds": [int(v) for v in train_suite["env_seeds"]],
            "rollout_seeds": [int(v) for v in train_suite["rollout_seeds"]],
            "runtime_wall_s": float(runtime_wall_s),
            "collect_contract_debug": {
                "strict_fixed_env_mode": bool(getattr(algo, "_rwkv_strict_fixed_env_mode", False)),
                "env_rng_seed_spec": (
                    None
                    if getattr(algo, "_rwkv_env_rng_seeds", None) is None
                    else [int(v) for v in getattr(algo, "_rwkv_env_rng_seeds", None)]
                ),
                "rollout_rng_seed_spec": (
                    None
                    if getattr(algo, "_rwkv_rollout_rng_seeds", None) is None
                    else [int(v) for v in getattr(algo, "_rwkv_rollout_rng_seeds", None)]
                ),
                "deterministic_actor_sampling": bool(getattr(algo, "_rwkv_deterministic_actor_sampling", False)),
                "deterministic_batch_plan": bool(getattr(algo, "_rwkv_deterministic_batch_plan", False)),
                "last_collect_rollout_env_rng_seeds": (
                    None
                    if getattr(algo, "_rwkv_last_collect_rollout_env_rng_seeds", None) is None
                    else [int(v) for v in getattr(algo, "_rwkv_last_collect_rollout_env_rng_seeds", None)]
                ),
                "last_collect_rollout_rollout_rng_seeds": (
                    None
                    if getattr(algo, "_rwkv_last_collect_rollout_rollout_rng_seeds", None) is None
                    else [int(v) for v in getattr(algo, "_rwkv_last_collect_rollout_rollout_rng_seeds", None)]
                ),
            },
            "transport_contract": transport_report,
        }
    finally:
        vec_env.close()
        del algo, callback, vec_env
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def run_phase3_shortrun_buffer_transport_contract_scan(
    *,
    checkpoint_path: str,
    suite_specs_path: str | None = None,
    suite_split: str = "train",
    device: str | None = None,
    n_samples: int = 2048,
    single_eval_pos: int = 1946,
    train_profile: str = "trusted_sep_reset_mainline",
    phase2_summary_path: str = DEFAULT_PHASE2_SUMMARY,
) -> dict[str, Any]:
    phase2_summary = assert_phase2_green(phase2_summary_path)
    suite_specs = _load_suite_specs(suite_specs_path)
    checkpoint_path = str(Path(checkpoint_path).expanduser().resolve())
    device_obj = torch.device(str(device or _default_device()))
    suite_reports = [
        _run_one_suite_transport_contract(
            checkpoint_path=checkpoint_path,
            suite_spec=dict(spec),
            suite_split=str(suite_split),
            device_obj=device_obj,
            n_samples=int(n_samples),
            single_eval_pos=int(single_eval_pos),
            train_profile=str(train_profile),
        )
        for spec in suite_specs
    ]
    aggregate = {
        "suite_count": int(len(suite_reports)),
        "transport_contract_ok": bool(all(bool(report["transport_contract"]["transport_contract_ok"]) for report in suite_reports)),
        "batch_count": int(sum(int(report["transport_contract"]["batch_count"]) for report in suite_reports)),
        "seq_layout_mismatch_count": int(sum(int(report["transport_contract"]["seq_layout_mismatch_count"]) for report in suite_reports)),
        "source_to_flat_mismatch_count": int(sum(int(report["transport_contract"]["source_to_flat_mismatch_count"]) for report in suite_reports)),
        "source_to_padded_valid_mismatch_count": int(sum(int(report["transport_contract"]["source_to_padded_valid_mismatch_count"]) for report in suite_reports)),
        "flat_to_padded_valid_mismatch_count": int(sum(int(report["transport_contract"]["flat_to_padded_valid_mismatch_count"]) for report in suite_reports)),
        "subbatch_flat_to_padded_mismatch_count": int(sum(int(report["transport_contract"]["subbatch_flat_to_padded_mismatch_count"]) for report in suite_reports)),
        "max_abs_diff_by_field": {},
        "suite_names": [str(report["suite_name"]) for report in suite_reports],
        "phase2_all_checks_pass": bool(phase2_summary["validation"]["all_checks_pass"]),
    }
    transport_fields = ("objective_masks", "episode_starts", "rollout_return_means", "rollout_return_stds")
    for field in transport_fields:
        aggregate["max_abs_diff_by_field"][field] = float(
            max(float(report["transport_contract"]["max_abs_diff_by_field"][field]) for report in suite_reports)
        )
    conclusion = {
        "transport_contract_ok": bool(aggregate["transport_contract_ok"]),
        "drift_source": "none" if bool(aggregate["transport_contract_ok"]) else "transport_alignment_mismatch",
        "small_fix_candidate_supported": bool(aggregate["transport_contract_ok"]),
    }
    return {
        "audit_entry": "phase3_shortrun_buffer_transport_contract_scan",
        "checkpoint_path": checkpoint_path,
        "contract": {
            "phase2_summary_path": str(Path(phase2_summary_path).expanduser().resolve()),
            "phase2_all_checks_pass": bool(phase2_summary["validation"]["all_checks_pass"]),
            "suite_specs_path": None if suite_specs_path is None else str(Path(suite_specs_path).expanduser().resolve()),
            "suite_split": str(suite_split),
            "device": str(device_obj),
            "n_samples": int(n_samples),
            "single_eval_pos": int(single_eval_pos),
            "train_profile": str(train_profile),
            "strict_fixed_env_mode": True,
            "deterministic_actor_sampling": True,
            "deterministic_batch_plan": True,
            "restore_validation_policy_state": True,
        },
        "suites": suite_reports,
        "aggregate": aggregate,
        "conclusion": conclusion,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Phase 3 many-env read-only buffer packing / minibatch transport contract scan."
    )
    parser.add_argument("checkpoint_path", type=str)
    parser.add_argument("--suite-specs-path", type=str, default=None)
    parser.add_argument("--suite-split", type=str, default="train", choices=["train", "heldout"])
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--n-samples", type=int, default=2048)
    parser.add_argument("--single-eval-pos", type=int, default=1946)
    parser.add_argument(
        "--train-profile",
        type=str,
        default="trusted_sep_reset_mainline",
        choices=["trusted_sep_reset_mainline", "legacy_actor_only_probe"],
    )
    parser.add_argument("--phase2-summary-path", type=str, default=DEFAULT_PHASE2_SUMMARY)
    parser.add_argument(
        "--output-json",
        type=str,
        default="/home/chen/RLPFN/artifacts/phase3_shortrun_buffer_transport_contract_scan.json",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output_path = Path(args.output_json).expanduser().resolve()
    if output_path.exists() and not bool(args.overwrite):
        print(output_path.read_text())
        return 0

    random.seed(2020)
    np.random.seed(2020)
    torch.manual_seed(2020)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(2020)

    report = run_phase3_shortrun_buffer_transport_contract_scan(
        checkpoint_path=args.checkpoint_path,
        suite_specs_path=args.suite_specs_path,
        suite_split=str(args.suite_split),
        device=args.device,
        n_samples=int(args.n_samples),
        single_eval_pos=int(args.single_eval_pos),
        train_profile=str(args.train_profile),
        phase2_summary_path=args.phase2_summary_path,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
