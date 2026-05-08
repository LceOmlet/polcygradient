import json
from pathlib import Path


DEFAULT_PHASE2_SUMMARY = (
    "/home/chen/RLPFN/artifacts/phase2_regression_pack/phase2_regression_suite_summary.json"
)
TRUSTED_PHASE2_LAUNCH_SUMMARY = (
    "/home/chen/RLPFN/artifacts/phase2_launch_anchor_green_summary.json"
)
CANONICAL_PHASE2_LAUNCH_ANCHOR = (
    "/home/chen/RLPFN/artifacts/"
    "phase2_launch_anchor_shared_seed12345_current_contract_quick.json"
)
CANONICAL_PHASE3_UNIQUE_CHAIN_MANIFEST = (
    "/home/chen/RLPFN/reinforce-terminal-explore/ticl/analysis/phase3_regression_manifest.json"
)
CANONICAL_PHASE2_FIXED_ENV_FINGERPRINT = (
    "7ef09e0736b8fd205af47d313b0b79723a6749bae126dff723cf8c08793ebee2"
)


def load_phase2_summary(path: str = DEFAULT_PHASE2_SUMMARY) -> dict:
    return json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))


def phase2_green_failures(summary: dict) -> list[str]:
    failures: list[str] = []

    if bool(summary.get("phase2_frozen", False)) is not True:
        failures.append("phase2_frozen != true")

    validation = summary.get("validation", {})
    if bool(validation.get("all_checks_pass", False)) is not True:
        failures.append("validation.all_checks_pass != true")

    pass_flags = summary.get("pass_flags", {})
    for key in (
        "official_vs_monkey_numeric_match",
        "shared_learned_beats_zero",
        "shared_learned_delta_positive",
        "shared_heldout_critic_quality_recorded",
    ):
        if bool(pass_flags.get(key, False)) is not True:
            failures.append(f"pass_flags.{key} != true")

    contract = summary.get("trusted_contract", {})
    expected_true_flags = (
        "ppo_reset_env_state_at_sep",
        "strict_fixed_env_mode",
        "deterministic_actor_sampling",
        "deterministic_batch_plan",
        "record_suite_critic_quality",
        "shared_actor_critic_backbone",
    )
    for key in expected_true_flags:
        if bool(contract.get(key, False)) is not True:
            failures.append(f"trusted_contract.{key} != true")
    if bool(contract.get("strict_native_rollout", True)) is not False:
        failures.append("trusted_contract.strict_native_rollout != false")
    if bool(contract.get("restore_validation_policy_state", True)) is not False:
        failures.append("trusted_contract.restore_validation_policy_state != false")
    if str(contract.get("frozen_h_seed_mode", "")).strip().lower() != "current":
        failures.append("trusted_contract.frozen_h_seed_mode != current")
    if bool(contract.get("zero_eval_prior_reuses_sampling_state", True)) is not False:
        failures.append("trusted_contract.zero_eval_prior_reuses_sampling_state != false")

    fixed_env_contract = summary.get("fixed_env_contract", {})
    if (
        str(fixed_env_contract.get("frozen_h_fingerprint", "")).strip()
        != CANONICAL_PHASE2_FIXED_ENV_FINGERPRINT
    ):
        failures.append("fixed_env_contract.frozen_h_fingerprint drift")
    if str(fixed_env_contract.get("frozen_h_seed_mode", "")).strip().lower() != "current":
        failures.append("fixed_env_contract.frozen_h_seed_mode != current")
    if bool(fixed_env_contract.get("zero_eval_prior_reuses_sampling_state", True)) is not False:
        failures.append("fixed_env_contract.zero_eval_prior_reuses_sampling_state != false")

    return failures


def assert_phase2_green(path: str = DEFAULT_PHASE2_SUMMARY) -> dict:
    summary = load_phase2_summary(path)
    failures = phase2_green_failures(summary)
    if failures:
        raise RuntimeError(
            "Phase 2 regression pack is not green; refusing Phase 3 work. "
            f"summary={Path(path).expanduser().resolve()} failures={failures}"
        )
    return summary


def phase2_launch_chain_failures(summary: dict) -> list[str]:
    failures = phase2_green_failures(summary)

    source_of_truth = dict(summary.get("source_of_truth", {}))
    expected_anchor = str(Path(CANONICAL_PHASE2_LAUNCH_ANCHOR).expanduser().resolve())
    observed_anchor = str(source_of_truth.get("phase2_launch_anchor_artifact", "")).strip()
    if observed_anchor != expected_anchor:
        failures.append("source_of_truth.phase2_launch_anchor_artifact drift")

    expected_manifest = str(Path(CANONICAL_PHASE3_UNIQUE_CHAIN_MANIFEST).expanduser().resolve())
    observed_manifest = str(source_of_truth.get("phase3_unique_chain_manifest", "")).strip()
    if observed_manifest != expected_manifest:
        failures.append("source_of_truth.phase3_unique_chain_manifest drift")

    return failures


def assert_phase2_launch_chain_green(path: str = TRUSTED_PHASE2_LAUNCH_SUMMARY) -> dict:
    summary = load_phase2_summary(path)
    failures = phase2_launch_chain_failures(summary)
    if failures:
        raise RuntimeError(
            "Trusted Phase 2 launch-chain summary is not green; refusing trusted Phase 3 pre/post work. "
            f"summary={Path(path).expanduser().resolve()} failures={failures}"
        )
    return summary
