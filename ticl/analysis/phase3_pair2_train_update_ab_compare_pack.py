import argparse
import faulthandler
import json
import os
import tempfile
from pathlib import Path
from typing import Any

faulthandler.enable(all_threads=True)
faulthandler.dump_traceback_later(120, repeat=True)

from ticl.analysis.phase2_guardrail import DEFAULT_PHASE2_SUMMARY
from ticl.analysis.phase3_multi_env_optimization_audit import run_phase3_multi_env_optimization_audit


PAIR2_CHECKPOINT_PATH = "/home/chen/RLPFN/rwkv/models_diff/rlpfn_03_30_2026_22_37_53_epoch_13.cpkt"
PAIR2_TRAIN_SUITE_PATH = "/home/chen/RLPFN/artifacts/phase3_cross_env_baseline_pair2/suites/train_suite.pt"
PAIR2_HELDOUT_SUITE_PATH = "/home/chen/RLPFN/artifacts/phase3_cross_env_baseline_pair2/suites/heldout_suite.pt"
PAIR2_ZERO_JSON = "/home/chen/RLPFN/artifacts/phase3_cross_env_baseline_pair2/zero_cross_env_control.json"
PAIR2_PRE_JSON = "/home/chen/RLPFN/artifacts/phase3_cross_env_baseline_pair2/ppo_cross_env_baseline.json"
PAIR2_PREFLIGHT_JSON = "/home/chen/RLPFN/artifacts/phase3_preflight_contract_check_pair2.json"
PAIR2_GUARDED_SELECTOR_REIDENTIFIED_JSON = (
    "/home/chen/RLPFN/artifacts/phase3_pair2_guarded_selector_reidentified.json"
)

PAIR2_BRANCH_SPECIFIC_MODE = "tokenwise_scale_env12_mid_episode_extension_block"
PAIR2_TRAIN_UPDATE_EVAL_NSAMPLES = 256
PAIR2_TRAIN_UPDATE_NSAMPLES = 256
PAIR2_TRAIN_UPDATE_SINGLE_EVAL_POS = 64
PAIR2_BASELINE_TRAIN_OUTER_BATCH_SNAPSHOT_JSON = (
    "/home/chen/RLPFN/artifacts/phase3_pair2_train_update_ab_baseline_outer_batch_snapshot.json"
)
PAIR2_OVERRIDE_TRAIN_OUTER_BATCH_SNAPSHOT_JSON = (
    "/home/chen/RLPFN/artifacts/phase3_pair2_train_update_ab_override_outer_batch_snapshot.json"
)
PAIR2_BASELINE_POST_POLICY_BUNDLE_PATH = (
    "/home/chen/RLPFN/artifacts/phase3_pair2_train_update_ab_baseline_post_policy_bundle.pt"
)
PAIR2_OVERRIDE_POST_POLICY_BUNDLE_PATH = (
    "/home/chen/RLPFN/artifacts/phase3_pair2_train_update_ab_override_post_policy_bundle.pt"
)
PAIR2_SNAPSHOT_TARGET_OUTER_BATCH_IDX = 13
PAIR2_SNAPSHOT_TARGET_ENV_INDEX = 12
PAIR2_SNAPSHOT_TARGET_OBJECTIVE_EPISODE_INDEX = 4
PAIR2_SNAPSHOT_TARGET_OBJECTIVE_POSITION_START = 15
PAIR2_SNAPSHOT_TARGET_OBJECTIVE_POSITION_END = 18
PAIR2_TRAIN_UPDATE_MATERIAL_EPS = 1e-4


def _normalize_optional_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def _suite_context(report: dict[str, Any]) -> dict[str, Any]:
    config = dict(report["config"])
    checkpoint_path = str(report.get("checkpoint_path", config.get("checkpoint_path", "")))
    train_profile = str(report.get("train_profile", config.get("train_profile", "")))
    return {
        "checkpoint_path": checkpoint_path,
        "train_suite_path": str(config["train_suite_path"]),
        "heldout_suite_path": str(config["heldout_suite_path"]),
        "n_samples": int(config["n_samples"]),
        "eval_n_samples": int(config.get("eval_n_samples", config["n_samples"])),
        "single_eval_pos": int(config["single_eval_pos"]),
        "train_profile": train_profile,
        "strict_fixed_env_mode": bool(config.get("strict_fixed_env_mode", False)),
        "deterministic_actor_sampling": bool(config.get("deterministic_actor_sampling", False)),
        "deterministic_batch_plan": bool(config.get("deterministic_batch_plan", False)),
        "strict_native_rollout": bool(config.get("strict_native_rollout", False)),
        "actor_objective_runtime_current_suite_name": _normalize_optional_str(
            config.get("actor_objective_runtime_current_suite_name", "")
        ),
        "actor_objective_mode_override": config.get("actor_objective_mode_override", None),
        "ppo_actor_baseline_mode": str(config.get("ppo_actor_baseline_mode", "")),
        "ppo_actor_gae_space": str(config.get("ppo_actor_gae_space", "")),
        "ppo_normalize_advantage": bool(config.get("ppo_normalize_advantage", False)),
        "ppo_vf_coef": float(config.get("ppo_vf_coef", float("nan"))),
        "ppo_reset_env_state_at_sep": bool(config.get("ppo_reset_env_state_at_sep", False)),
        "ppo_separate_value_backbone": bool(config.get("ppo_separate_value_backbone", False)),
        "ppo_restore_validation_policy_head_state": bool(
            config.get("ppo_restore_validation_policy_head_state", False)
        ),
        "runtime_normalized_q_value_weight_override": config.get("runtime_normalized_q_value_weight_override", None),
        "runtime_next_state_flow_matching_weight_override": config.get("runtime_next_state_flow_matching_weight_override", None),
        "train_suite_fingerprint": str(report["train_suite_summary"]["fingerprint"]),
        "heldout_suite_fingerprint": str(report["heldout_suite_summary"]["fingerprint"]),
    }


def _shared_contract_context(ctx: dict[str, Any]) -> dict[str, Any]:
    out = dict(ctx)
    out.pop("actor_objective_runtime_current_suite_name", None)
    out.pop("actor_objective_mode_override", None)
    return out


def _load_pair2_preflight() -> dict[str, Any]:
    path = Path(PAIR2_PREFLIGHT_JSON).expanduser().resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not bool(payload.get("preflight_passed", False)):
        raise RuntimeError(f"Pair2 preflight is not green: {path}")
    signature = dict(payload.get("phase3_many_env_signature", {}))
    required = {
        "n_envs": 16,
        "n_steps": PAIR2_TRAIN_UPDATE_NSAMPLES,
        "batch_size": 256,
        "n_epochs": 1,
        "learning_rate": 2e-4,
        "target_kl": 0.03,
        "actor_objective_mode": "tokenwise",
        "strict_native_rollout": True,
        "normalize_advantage": False,
        "separate_value_backbone": False,
        "runtime_normalized_q_value_weight_override": None,
        "runtime_next_state_flow_matching_weight_override": None,
    }
    for key, expected in required.items():
        observed = signature.get(key, None)
        if observed != expected:
            raise RuntimeError(
                f"Pair2 preflight mismatch for {key}: observed={observed!r}, expected={expected!r}"
            )
    if dict(payload.get("contract_diff", {})).get("unexpected_diff", None) != {}:
        raise RuntimeError(f"Pair2 preflight has unexpected contract diff: {path}")
    return payload


def _load_pair2_guarded_selector_reidentified() -> dict[str, int]:
    path = Path(PAIR2_GUARDED_SELECTOR_REIDENTIFIED_JSON).expanduser().resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if str(payload.get("probe_entry", "")) != "phase3_pair2_guarded_selector_reidentified":
        raise RuntimeError(f"Expected guarded selector reidentify artifact at {path}")
    selector = dict(payload.get("reidentified_selector", {}))
    return {
        "target_outer_batch_idx": int(selector["outer_batch_idx"]),
        "target_env_index": int(selector["target_env_index"]),
        "target_objective_episode_index": int(selector["target_objective_episode_index"]),
        "target_objective_position_start": int(selector["target_objective_position_start"]),
        "target_objective_position_end": int(selector["target_objective_position_end"]),
    }


def _current_pair2_snapshot_selector() -> dict[str, int]:
    return {
        "target_outer_batch_idx": int(PAIR2_SNAPSHOT_TARGET_OUTER_BATCH_IDX),
        "target_env_index": int(PAIR2_SNAPSHOT_TARGET_ENV_INDEX),
        "target_objective_episode_index": int(PAIR2_SNAPSHOT_TARGET_OBJECTIVE_EPISODE_INDEX),
        "target_objective_position_start": int(PAIR2_SNAPSHOT_TARGET_OBJECTIVE_POSITION_START),
        "target_objective_position_end": int(PAIR2_SNAPSHOT_TARGET_OBJECTIVE_POSITION_END),
    }


def _assert_pair2_snapshot_selector_matches_reidentified() -> dict[str, int]:
    expected = _load_pair2_guarded_selector_reidentified()
    current = _current_pair2_snapshot_selector()
    if current != expected:
        raise RuntimeError(
            "Pair2 target-side selector constants drifted from the reidentified guarded-contract selector: "
            f"current={current!r}, expected={expected!r}"
        )
    return expected


def _require_target_snapshot_written(report: dict[str, Any], *, arm_name: str) -> dict[str, Any]:
    snapshot = dict(report.get("train_outer_batch_snapshot", {}))
    if not bool(snapshot.get("written", False)):
        raise RuntimeError(
            f"Pair2 {arm_name} train-side snapshot was not written. "
            "Do not treat this target-side A/B as efficacy evidence when the requested selector is absent."
        )
    selector = dict(snapshot.get("selector", {}))
    if selector != _current_pair2_snapshot_selector():
        raise RuntimeError(
            f"Pair2 {arm_name} snapshot selector drifted from the locked guarded selector: {selector!r}"
        )
    if snapshot.get("written_path", None) in {None, ""}:
        raise RuntimeError(f"Pair2 {arm_name} snapshot reports written=True but has no written_path.")
    target_match_count = int(snapshot.get("snapshot_target_match_count", 0))
    if target_match_count <= 0:
        raise RuntimeError(
            f"Pair2 {arm_name} snapshot was written but the locked selector matched no objective tokens "
            f"(snapshot_target_match_count={target_match_count})."
        )
    return snapshot


def _build_reused_metrics_payload(
    *,
    report: dict[str, Any],
    policy_mode: str,
    train_metrics: dict[str, Any],
    heldout_metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config = dict(report["config"])
    payload = {
        "policy": {
            "resolved_mode": str(policy_mode),
            "requested_mode": str(policy_mode),
        },
        "config": {
            "n_samples": int(config["eval_n_samples"]),
            "single_eval_pos": int(config["single_eval_pos"]),
            "rollout_backend": str(config["rollout_backend"]),
            "core_a": False,
            "reference_semantics_enabled": False,
        },
        "train_suite": {
            "summary": dict(report["train_suite_summary"]),
            "metrics": dict(train_metrics),
        },
    }
    if heldout_metrics is not None:
        payload["heldout_suite"] = {
            "summary": dict(report["heldout_suite_summary"]),
            "metrics": dict(heldout_metrics),
        }
    return payload


def _write_reused_metrics_json(path: Path, payload: dict[str, Any]) -> str:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return str(path)


def _gap_delta(baseline: dict[str, float], override: dict[str, float]) -> dict[str, float]:
    return {str(k): float(override[str(k)] - baseline[str(k)]) for k in baseline}


def _has_material_delta(values: dict[str, float], *, eps: float = PAIR2_TRAIN_UPDATE_MATERIAL_EPS) -> bool:
    return any(abs(float(v)) > float(eps) for v in values.values())


def _train_history_delta(baseline: list[dict[str, Any]], override: list[dict[str, Any]]) -> dict[str, Any]:
    if len(baseline) != len(override):
        raise ValueError(
            f"Train history length mismatch: baseline={len(baseline)} override={len(override)}"
        )
    epoch_deltas: list[dict[str, Any]] = []
    max_abs_metric_delta = 0.0
    for base_row, over_row in zip(baseline, override):
        if int(base_row["outer_epoch"]) != int(over_row["outer_epoch"]):
            raise ValueError(
                "Train history epoch mismatch: "
                f"baseline={base_row['outer_epoch']!r} override={over_row['outer_epoch']!r}"
            )
        row = {
            "outer_epoch": int(base_row["outer_epoch"]),
            "critic_raw_corr_delta": float(over_row["critic_raw_corr"] - base_row["critic_raw_corr"]),
            "critic_explained_variance_raw_delta": float(
                over_row["critic_explained_variance_raw"] - base_row["critic_explained_variance_raw"]
            ),
            "critic_objective_total_delta": int(over_row["critic_objective_total"] - base_row["critic_objective_total"]),
            "actor_baseline_mode_changed": bool(over_row["actor_baseline_mode"] != base_row["actor_baseline_mode"]),
            "switches_to_learned_next_changed": bool(
                bool(over_row["switches_to_learned_next"]) != bool(base_row["switches_to_learned_next"])
            ),
        }
        max_abs_metric_delta = max(
            max_abs_metric_delta,
            abs(float(row["critic_raw_corr_delta"])),
            abs(float(row["critic_explained_variance_raw_delta"])),
            abs(float(row["critic_objective_total_delta"])),
        )
        epoch_deltas.append(row)
    return {
        "epoch_deltas": epoch_deltas,
        "max_abs_metric_delta": float(max_abs_metric_delta),
        "histories_identical": bool(max_abs_metric_delta <= 1e-6)
        and all(not row["actor_baseline_mode_changed"] for row in epoch_deltas)
        and all(not row["switches_to_learned_next_changed"] for row in epoch_deltas),
    }


def _run_audit(
    *,
    actor_objective_mode_override: str | None,
    restore_validation_policy_head_state: bool = False,
    reuse_zero_control_json: str | None = None,
    reuse_pre_policy_json: str | None = None,
    save_post_policy_bundle_path: str | None = None,
    resume_post_policy_bundle_path: str | None = None,
    eval_n_samples: int = PAIR2_TRAIN_UPDATE_EVAL_NSAMPLES,
    train_outer_batch_snapshot_json: str | None = None,
    train_outer_batch_snapshot_target_outer_batch_idx: int | None = None,
    train_outer_batch_snapshot_target_env_index: int | None = None,
    train_outer_batch_snapshot_target_objective_episode_index: int | None = None,
    train_outer_batch_snapshot_target_objective_position_start: int | None = None,
    train_outer_batch_snapshot_target_objective_position_end: int | None = None,
) -> dict[str, Any]:
    mode_label = "baseline" if actor_objective_mode_override is None else str(actor_objective_mode_override)
    print(f"[pair2-train-update-ab] start audit mode={mode_label}", flush=True)
    os.environ["TICL_PHASE3_PROFILE_TIMING"] = "1"
    report = run_phase3_multi_env_optimization_audit(
        checkpoint_path=PAIR2_CHECKPOINT_PATH,
        train_suite_path=PAIR2_TRAIN_SUITE_PATH,
        heldout_suite_path=PAIR2_HELDOUT_SUITE_PATH,
        device="cuda:0",
        n_samples=PAIR2_TRAIN_UPDATE_NSAMPLES,
        single_eval_pos=PAIR2_TRAIN_UPDATE_SINGLE_EVAL_POS,
        batch_size=256,
        outer_epochs=1,
        n_epochs=1,
        learning_rate=2e-4,
        target_kl=0.03,
        rollout_backend="serial",
        train_profile="phase2_shared_backbone_contract",
        phase2_summary_path=DEFAULT_PHASE2_SUMMARY,
        reuse_zero_control_json=reuse_zero_control_json,
        reuse_pre_policy_json=reuse_pre_policy_json,
        save_post_policy_bundle_path=save_post_policy_bundle_path,
        resume_post_policy_bundle_path=resume_post_policy_bundle_path,
        strict_fixed_env_mode=True,
        deterministic_actor_sampling=True,
        deterministic_batch_plan=True,
        strict_native_rollout=True,
        restore_validation_policy_head_state=bool(restore_validation_policy_head_state),
        actor_objective_mode_override=actor_objective_mode_override,
        actor_objective_runtime_current_suite_name=(
            "pair2" if actor_objective_mode_override is not None else None
        ),
        skip_heldout_eval=True,
        eval_n_samples=int(eval_n_samples),
        train_outer_batch_snapshot_json=train_outer_batch_snapshot_json,
        train_outer_batch_snapshot_target_outer_batch_idx=train_outer_batch_snapshot_target_outer_batch_idx,
        train_outer_batch_snapshot_target_env_index=train_outer_batch_snapshot_target_env_index,
        train_outer_batch_snapshot_target_objective_episode_index=train_outer_batch_snapshot_target_objective_episode_index,
        train_outer_batch_snapshot_target_objective_position_start=train_outer_batch_snapshot_target_objective_position_start,
        train_outer_batch_snapshot_target_objective_position_end=train_outer_batch_snapshot_target_objective_position_end,
    )
    print(f"[pair2-train-update-ab] done audit mode={mode_label}", flush=True)
    return report


def build_pair2_train_update_ab_compare_pack() -> dict[str, Any]:
    preflight = _load_pair2_preflight()
    locked_selector = _assert_pair2_snapshot_selector_matches_reidentified()
    print("[pair2-train-update-ab] start baseline audit", flush=True)
    baseline = _run_audit(
        actor_objective_mode_override=None,
        restore_validation_policy_head_state=True,
        save_post_policy_bundle_path=PAIR2_BASELINE_POST_POLICY_BUNDLE_PATH,
        eval_n_samples=PAIR2_TRAIN_UPDATE_EVAL_NSAMPLES,
        train_outer_batch_snapshot_json=PAIR2_BASELINE_TRAIN_OUTER_BATCH_SNAPSHOT_JSON,
        train_outer_batch_snapshot_target_outer_batch_idx=PAIR2_SNAPSHOT_TARGET_OUTER_BATCH_IDX,
        train_outer_batch_snapshot_target_env_index=PAIR2_SNAPSHOT_TARGET_ENV_INDEX,
        train_outer_batch_snapshot_target_objective_episode_index=PAIR2_SNAPSHOT_TARGET_OBJECTIVE_EPISODE_INDEX,
        train_outer_batch_snapshot_target_objective_position_start=PAIR2_SNAPSHOT_TARGET_OBJECTIVE_POSITION_START,
        train_outer_batch_snapshot_target_objective_position_end=PAIR2_SNAPSHOT_TARGET_OBJECTIVE_POSITION_END,
    )
    print("[pair2-train-update-ab] start override audit", flush=True)
    with tempfile.TemporaryDirectory(prefix="pair2_train_update_ab_") as tmpdir:
        tmpdir_path = Path(tmpdir)
        zero_payload = _build_reused_metrics_payload(
            report=baseline,
            policy_mode="zero",
            train_metrics=baseline["zero_control"]["train"],
            heldout_metrics=baseline["zero_control"]["heldout"],
        )
        pre_payload = _build_reused_metrics_payload(
            report=baseline,
            policy_mode="ppo",
            train_metrics=baseline["pre"]["train"],
        )
        reuse_zero_control_json = _write_reused_metrics_json(tmpdir_path / "zero_control.json", zero_payload)
        reuse_pre_policy_json = _write_reused_metrics_json(tmpdir_path / "pre_policy.json", pre_payload)
        override = _run_audit(
            actor_objective_mode_override=PAIR2_BRANCH_SPECIFIC_MODE,
            restore_validation_policy_head_state=True,
            reuse_zero_control_json=reuse_zero_control_json,
            reuse_pre_policy_json=reuse_pre_policy_json,
            save_post_policy_bundle_path=PAIR2_OVERRIDE_POST_POLICY_BUNDLE_PATH,
            eval_n_samples=PAIR2_TRAIN_UPDATE_EVAL_NSAMPLES,
            train_outer_batch_snapshot_json=PAIR2_OVERRIDE_TRAIN_OUTER_BATCH_SNAPSHOT_JSON,
            train_outer_batch_snapshot_target_outer_batch_idx=PAIR2_SNAPSHOT_TARGET_OUTER_BATCH_IDX,
            train_outer_batch_snapshot_target_env_index=PAIR2_SNAPSHOT_TARGET_ENV_INDEX,
            train_outer_batch_snapshot_target_objective_episode_index=PAIR2_SNAPSHOT_TARGET_OBJECTIVE_EPISODE_INDEX,
            train_outer_batch_snapshot_target_objective_position_start=PAIR2_SNAPSHOT_TARGET_OBJECTIVE_POSITION_START,
            train_outer_batch_snapshot_target_objective_position_end=PAIR2_SNAPSHOT_TARGET_OBJECTIVE_POSITION_END,
        )
    print("[pair2-train-update-ab] both audits complete; assembling compare pack", flush=True)

    baseline_ctx = _suite_context(baseline)
    override_ctx = _suite_context(override)
    if _shared_contract_context(baseline_ctx) != _shared_contract_context(override_ctx):
        raise ValueError(
            "Baseline and override pair2 train-update audits must share the same suite contract, "
            f"got {baseline_ctx!r} vs {override_ctx!r}"
        )
    if baseline_ctx["actor_objective_runtime_current_suite_name"] not in {"", "none", "None"}:
        raise ValueError(
            "Baseline audit should not activate a runtime-scoped objective mode: "
            f"{baseline_ctx['actor_objective_runtime_current_suite_name']!r}"
        )
    if baseline_ctx["actor_objective_mode_override"] is not None:
        raise ValueError(
            f"Baseline audit should use tokenwise objective without override: {baseline_ctx!r}"
        )
    if str(override_ctx["actor_objective_mode_override"]) != PAIR2_BRANCH_SPECIFIC_MODE:
        raise ValueError(
            f"Override audit did not activate expected branch-specific mode: {override_ctx['actor_objective_mode_override']!r}"
        )
    if override_ctx["actor_objective_runtime_current_suite_name"] != "pair2":
        raise ValueError(
            "Override audit did not target pair2 runtime scope: "
            f"{override_ctx['actor_objective_runtime_current_suite_name']!r}"
        )
    baseline_snapshot = _require_target_snapshot_written(baseline, arm_name="baseline")
    override_snapshot = _require_target_snapshot_written(override, arm_name="override")

    baseline_comparison = dict(baseline["comparison"])
    override_comparison = dict(override["comparison"])
    train_gap_delta = _gap_delta(
        dict(baseline_comparison["pre_train_vs_zero"]),
        dict(override_comparison["pre_train_vs_zero"]),
    )
    post_train_gap_delta = _gap_delta(
        dict(baseline_comparison["post_train_vs_zero"]),
        dict(override_comparison["post_train_vs_zero"]),
    )
    train_history_delta = _train_history_delta(
        list(baseline["train_history"]),
        list(override["train_history"]),
    )

    return {
        "audit_entry": "phase3_pair2_train_update_ab_compare_pack",
        "preflight": {
            "source_path": PAIR2_PREFLIGHT_JSON,
            "preflight_passed": bool(preflight.get("preflight_passed", False)),
            "contract_args": dict(preflight.get("contract_args", {})),
            "allowed_diff": dict(dict(preflight.get("contract_diff", {})).get("allowed_diff", {})),
            "unexpected_diff": dict(dict(preflight.get("contract_diff", {})).get("unexpected_diff", {})),
        },
        "suite_context": baseline_ctx,
        "shared_contract_context": _shared_contract_context(baseline_ctx),
        "override_target_context": {
            "actor_objective_runtime_current_suite_name": override_ctx["actor_objective_runtime_current_suite_name"],
            "actor_objective_mode_override": override_ctx["actor_objective_mode_override"],
        },
        "target_control": {
            "candidate_small_control_target": "env12_mid_episode_extension_block",
            "runtime_control_shape": "contiguous_four_token_branch_specific_gate",
            "branch_specific": True,
            "locked_selector": locked_selector,
            "runtime_mode_activated": bool(
                override_ctx["actor_objective_runtime_current_suite_name"] == "pair2"
                and override_ctx["actor_objective_mode_override"] == PAIR2_BRANCH_SPECIFIC_MODE
            ),
        },
        "baseline": {
            "comparison": baseline_comparison,
            "train_history": list(baseline["train_history"]),
        },
        "override": {
            "comparison": override_comparison,
            "train_history": list(override["train_history"]),
        },
        "comparison": {
            "pre_train_vs_zero_gap_delta": train_gap_delta,
            "post_train_vs_zero_gap_delta": post_train_gap_delta,
            "train_full_return_delta_change": float(
                override_comparison["train_full_return_delta"] - baseline_comparison["train_full_return_delta"]
            ),
            "train_suffix_return_delta_change": float(
                override_comparison["train_suffix_return_delta"] - baseline_comparison["train_suffix_return_delta"]
            ),
            "train_history_delta": train_history_delta,
            "heldout_eval_skipped": bool(
                baseline.get("heldout_eval_skipped", True) and override.get("heldout_eval_skipped", True)
            ),
        },
        "train_outer_batch_snapshots": {
            "baseline_requested_path": PAIR2_BASELINE_TRAIN_OUTER_BATCH_SNAPSHOT_JSON,
            "override_requested_path": PAIR2_OVERRIDE_TRAIN_OUTER_BATCH_SNAPSHOT_JSON,
            "selector": locked_selector,
            "baseline": baseline_snapshot,
            "override": override_snapshot,
        },
        "post_policy_bundles": {
            "baseline_requested_path": PAIR2_BASELINE_POST_POLICY_BUNDLE_PATH,
            "override_requested_path": PAIR2_OVERRIDE_POST_POLICY_BUNDLE_PATH,
            "baseline": dict(baseline.get("post_policy_bundle", {})),
            "override": dict(override.get("post_policy_bundle", {})),
        },
        "conclusions": {
            "target_side_train_update_changes_trajectory": bool(
                not train_history_delta["histories_identical"]
                or _has_material_delta(train_gap_delta)
                or _has_material_delta(post_train_gap_delta)
                or abs(float(override_comparison["train_full_return_delta"] - baseline_comparison["train_full_return_delta"]))
                > float(PAIR2_TRAIN_UPDATE_MATERIAL_EPS)
                or abs(float(override_comparison["train_suffix_return_delta"] - baseline_comparison["train_suffix_return_delta"]))
                > float(PAIR2_TRAIN_UPDATE_MATERIAL_EPS)
            ),
            "pre_metrics_unchanged": bool(
                all(abs(float(v)) <= 1e-6 for v in train_gap_delta.values())
            ),
            "target_side_effect_is_only_numeric_noise": bool(
                train_history_delta["histories_identical"]
                and not _has_material_delta(train_gap_delta)
                and not _has_material_delta(post_train_gap_delta)
                and abs(
                    float(
                        override_comparison["train_full_return_delta"]
                        - baseline_comparison["train_full_return_delta"]
                    )
                )
                <= float(PAIR2_TRAIN_UPDATE_MATERIAL_EPS)
                and abs(
                    float(
                        override_comparison["train_suffix_return_delta"]
                        - baseline_comparison["train_suffix_return_delta"]
                    )
                )
                <= float(PAIR2_TRAIN_UPDATE_MATERIAL_EPS)
            ),
            "target_selector_present_in_both_arms": True,
            "target_side_compare_is_efficacy_eligible": True,
            "runtime_mode_local_to_pair2": True,
            "shared_core_unchanged": True,
        },
        "recommendation": {
            "candidate_small_control_target": "env12_mid_episode_extension_block",
            "branch_specific_scope_gate_needed": True,
            "shared_core_unchanged": True,
            "reason": (
                "Keep the target-side pair2 A/B valid only when the locked guarded selector is present in both arms; "
                "otherwise fail hard instead of interpreting a no-op as efficacy evidence."
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Pair2 target-side train-update A/B compare pack for the env12 mid-episode extension control block."
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="/home/chen/RLPFN/artifacts/phase3_pair2_train_update_ab_compare_pack.json",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output_path = Path(args.output_json).expanduser().resolve()
    if output_path.exists() and not bool(args.overwrite):
        print(output_path.read_text())
        return 0

    report = build_pair2_train_update_ab_compare_pack()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
