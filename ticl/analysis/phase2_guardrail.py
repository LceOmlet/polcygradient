import json
from pathlib import Path


DEFAULT_PHASE2_SUMMARY = (
    "/home/chen/RLPFN/artifacts/phase2_regression_pack/phase2_regression_suite_summary.json"
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
        "shared_learned_post_return_positive",
    ):
        if bool(pass_flags.get(key, False)) is not True:
            failures.append(f"pass_flags.{key} != true")

    contract = summary.get("trusted_contract", {})
    expected_true_flags = (
        "ppo_reset_env_state_at_sep",
        "strict_fixed_env_mode",
        "deterministic_actor_sampling",
        "deterministic_batch_plan",
        "shared_actor_critic_backbone",
    )
    for key in expected_true_flags:
        if bool(contract.get(key, False)) is not True:
            failures.append(f"trusted_contract.{key} != true")

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
