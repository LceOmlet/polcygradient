from typing import Any

from ticl.analysis.phase3_env_delta_concentration_probe import _load_reused_per_env_metrics
from ticl.analysis.phase3_heldout_token_bucket_probe import _load_heldout_subset_deltas


def _resolve_metrics_for_expected_count(
    *,
    metrics: dict[str, Any] | None,
    reuse_json: str | None,
    expected_policy_mode: str,
    train_suite_summary: dict[str, Any],
    heldout_suite_summary: dict[str, Any],
    n_samples: int,
    single_eval_pos: int,
    rollout_backend: str,
) -> dict[str, Any]:
    if metrics is not None:
        return metrics
    if reuse_json is None:
        raise ValueError(
            f"Need {expected_policy_mode} metrics or reuse_{expected_policy_mode}_control_json to validate heldout subset coverage."
        )
    return _load_reused_per_env_metrics(
        report_json_path=str(reuse_json),
        expected_policy_mode=str(expected_policy_mode),
        train_suite_summary=train_suite_summary,
        heldout_suite_summary=heldout_suite_summary,
        n_samples=int(n_samples),
        single_eval_pos=int(single_eval_pos),
        rollout_backend=str(rollout_backend),
    )


def resolve_heldout_subset_deltas(
    *,
    reuse_heldout_subset_delta_json: str | None,
    zero_metrics: dict[str, Any] | None,
    pre_metrics: dict[str, Any] | None,
    reuse_zero_control_json: str | None,
    reuse_pre_policy_json: str | None,
    train_suite_summary: dict[str, Any],
    heldout_suite_summary: dict[str, Any],
    n_samples: int,
    single_eval_pos: int,
    rollout_backend: str,
) -> dict[int, dict[str, float]]:
    if reuse_heldout_subset_delta_json is not None:
        resolved_zero = _resolve_metrics_for_expected_count(
            metrics=zero_metrics,
            reuse_json=reuse_zero_control_json,
            expected_policy_mode="zero",
            train_suite_summary=train_suite_summary,
            heldout_suite_summary=heldout_suite_summary,
            n_samples=int(n_samples),
            single_eval_pos=int(single_eval_pos),
            rollout_backend=str(rollout_backend),
        )
        resolved_pre = _resolve_metrics_for_expected_count(
            metrics=pre_metrics,
            reuse_json=reuse_pre_policy_json,
            expected_policy_mode="ppo",
            train_suite_summary=train_suite_summary,
            heldout_suite_summary=heldout_suite_summary,
            n_samples=int(n_samples),
            single_eval_pos=int(single_eval_pos),
            rollout_backend=str(rollout_backend),
        )
        zero_suffix = [float(v) for v in resolved_zero["heldout"]["suffix_return_per_env"]]
        pre_suffix = [float(v) for v in resolved_pre["heldout"]["suffix_return_per_env"]]
        expected_count = min(len(zero_suffix), len(pre_suffix))
        subset = _load_heldout_subset_deltas(str(reuse_heldout_subset_delta_json))
        missing_indices = [idx for idx in range(expected_count) if idx not in subset]
        if missing_indices:
            preview = missing_indices[:8]
            raise ValueError(
                "Heldout subset delta JSON does not cover the full heldout suite. "
                f"expected_count={expected_count}, missing_indices={preview}, "
                "use suite-matched zero/PPO fallback instead of a partial subset artifact."
            )
        return subset

    resolved_zero = zero_metrics
    resolved_pre = pre_metrics
    if resolved_zero is None:
        if reuse_zero_control_json is None:
            raise ValueError("Need zero metrics or reuse_zero_control_json to derive heldout subset deltas.")
        resolved_zero = _load_reused_per_env_metrics(
            report_json_path=str(reuse_zero_control_json),
            expected_policy_mode="zero",
            train_suite_summary=train_suite_summary,
            heldout_suite_summary=heldout_suite_summary,
            n_samples=int(n_samples),
            single_eval_pos=int(single_eval_pos),
            rollout_backend=str(rollout_backend),
        )
    if resolved_pre is None:
        if reuse_pre_policy_json is None:
            raise ValueError("Need pre metrics or reuse_pre_policy_json to derive heldout subset deltas.")
        resolved_pre = _load_reused_per_env_metrics(
            report_json_path=str(reuse_pre_policy_json),
            expected_policy_mode="ppo",
            train_suite_summary=train_suite_summary,
            heldout_suite_summary=heldout_suite_summary,
            n_samples=int(n_samples),
            single_eval_pos=int(single_eval_pos),
            rollout_backend=str(rollout_backend),
        )

    zero_suffix = [float(v) for v in resolved_zero["heldout"]["suffix_return_per_env"]]
    zero_full = [float(v) for v in resolved_zero["heldout"]["full_return_per_env"]]
    pre_suffix = [float(v) for v in resolved_pre["heldout"]["suffix_return_per_env"]]
    pre_full = [float(v) for v in resolved_pre["heldout"]["full_return_per_env"]]
    count = min(len(zero_suffix), len(pre_suffix), len(zero_full), len(pre_full))
    return {
        idx: {
            "pre_vs_zero_suffix_gap": float(pre_suffix[idx] - zero_suffix[idx]),
            "pre_vs_zero_full_gap": float(pre_full[idx] - zero_full[idx]),
            "suffix_gap_delta": 0.0,
            "full_gap_delta": 0.0,
        }
        for idx in range(count)
    }
