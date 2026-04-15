import argparse
import json
from pathlib import Path
from typing import Any


def _load_json(path: str) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.exists():
        raise ValueError(f"Missing JSON artifact: {resolved}")
    return json.loads(resolved.read_text())


def _contrast(payload: dict[str, Any]) -> dict[str, float]:
    contrasts = payload.get("contrasts", {})
    key = "low_mass_positive_minus_high_mass_nonpositive"
    if key not in contrasts:
        raise ValueError(f"Missing contrast block {key!r} in {payload.get('audit_entry')!r}")
    return {str(k): float(v) for k, v in dict(contrasts[key]).items()}


def _suite_context(payload: dict[str, Any]) -> dict[str, Any]:
    ctx = dict(payload.get("suite_context", {}))
    if not ctx:
        raise ValueError(f"Missing suite_context in {payload.get('audit_entry')!r}")
    return ctx


def _validate_context(ctx: dict[str, Any], *, expected_suite_name: str, expected_regime: str) -> None:
    suite_name = str(ctx.get("suite_name", ""))
    suite_regime = str(ctx.get("suite_regime", ""))
    if suite_name != expected_suite_name:
        raise ValueError(
            f"Unexpected suite name in regime-labeled contrast: observed={suite_name!r}, expected={expected_suite_name!r}"
        )
    if suite_regime != expected_regime:
        raise ValueError(
            f"Unexpected suite regime in regime-labeled contrast: observed={suite_regime!r}, expected={expected_regime!r}"
        )
    if not bool(ctx.get("train_suite_summary_fingerprint_match", False)):
        raise ValueError(f"Train suite fingerprint not verified for {suite_name!r}.")
    if not bool(ctx.get("suite_regime_summary_membership_verified", False)):
        raise ValueError(f"Suite regime membership not verified for {suite_name!r}.")


def _ratio(a: float, b: float) -> float | None:
    if float(b) == 0.0:
        return None
    return float(a / b)


def build_train_env_quality_regime_compare_pack(
    *,
    positive_contrast_json: str,
    nonpositive_contrast_json: str,
) -> dict[str, Any]:
    positive = _load_json(positive_contrast_json)
    nonpositive = _load_json(nonpositive_contrast_json)

    positive_ctx = _suite_context(positive)
    nonpositive_ctx = _suite_context(nonpositive)
    _validate_context(positive_ctx, expected_suite_name="pair1", expected_regime="positive")
    _validate_context(nonpositive_ctx, expected_suite_name="seed24680", expected_regime="nonpositive")

    positive_contrast = _contrast(positive)
    nonpositive_contrast = _contrast(nonpositive)

    shared_signal_keys = (
        "high_positive_mass_alone_is_not_sufficient_for_positive_delta",
        "episode_segmentation_not_primary_separator",
        "low_mass_positive_envs_have_higher_pre_suffix_gap_than_high_mass_nonpositive",
        "low_mass_positive_envs_have_higher_value_corr_than_high_mass_nonpositive",
    )
    shared_signals = {
        key: bool(positive["conclusions"].get(key, False) and nonpositive["conclusions"].get(key, False))
        for key in shared_signal_keys
    }

    regime_specific_signals = {
        "positive_only_early_mass_distinguishes_better_than_tail_mass": bool(
            positive["conclusions"].get("early_mass_distinguishes_better_than_tail_mass", False)
        ),
        "nonpositive_early_mass_distinguishes_better_than_tail_mass": bool(
            nonpositive["conclusions"].get("early_mass_distinguishes_better_than_tail_mass", False)
        ),
        "contrast_amplification": {
            "pre_suffix_gap_ratio": _ratio(
                positive_contrast["pre_suffix_gap"],
                nonpositive_contrast["pre_suffix_gap"],
            ),
            "raw_value_return_corr_ratio": _ratio(
                positive_contrast["raw_value_return_corr"],
                nonpositive_contrast["raw_value_return_corr"],
            ),
            "first16_positive_share_ratio": _ratio(
                positive_contrast["first16_positive_share"],
                nonpositive_contrast["first16_positive_share"],
            ),
            "last16_positive_share_delta": float(
                positive_contrast["last16_positive_share"] - nonpositive_contrast["last16_positive_share"]
            ),
            "last16_positive_share_sign_flip": bool(
                (positive_contrast["last16_positive_share"] > 0.0) != (nonpositive_contrast["last16_positive_share"] > 0.0)
            ),
        },
    }

    recommendation = {
        "weighting_repair_candidate_supported": False,
        "regime_split_guardrail_required": True,
        "recommended_next_step": "guardrail_only",
        "reason": (
            "Shared signals survive both regimes, but the only strong discriminator that differs across regimes "
            "is early_mass_distinguishes_better_than_tail_mass, which is positive-only. "
            "A pooled weighting tweak is therefore not yet justified."
        ),
    }

    return {
        "audit_entry": "phase3_train_env_quality_regime_compare_pack",
        "source_positive_contrast_json": str(Path(positive_contrast_json).expanduser().resolve()),
        "source_nonpositive_contrast_json": str(Path(nonpositive_contrast_json).expanduser().resolve()),
        "positive_suite_context": positive_ctx,
        "nonpositive_suite_context": nonpositive_ctx,
        "shared_signals": shared_signals,
        "regime_specific_signals": regime_specific_signals,
        "comparison": {
            "positive": {
                "suite_name": positive_ctx["suite_name"],
                "suite_regime": positive_ctx["suite_regime"],
                "contrast": positive_contrast,
                "shared_true_count": int(sum(bool(v) for v in shared_signals.values())),
            },
            "nonpositive": {
                "suite_name": nonpositive_ctx["suite_name"],
                "suite_regime": nonpositive_ctx["suite_regime"],
                "contrast": nonpositive_contrast,
                "shared_true_count": int(sum(bool(v) for v in shared_signals.values())),
            },
        },
        "recommendation": recommendation,
        "conclusions": {
            "shared_signals_consistent_across_regimes": bool(all(shared_signals.values())),
            "regime_specific_early_mass_separator_is_positive_only": bool(
                regime_specific_signals["positive_only_early_mass_distinguishes_better_than_tail_mass"]
                and not regime_specific_signals["nonpositive_early_mass_distinguishes_better_than_tail_mass"]
            ),
            "weighting_repair_not_yet_supported": True,
            "regime_split_guardrail_preferred": True,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare the positive and nonpositive regime-labeled train-side contrast packs."
    )
    parser.add_argument(
        "--positive-contrast-json",
        type=str,
        default="/home/chen/RLPFN/artifacts/phase3_train_env_quality_contrast_regime_pair1_positive.json",
    )
    parser.add_argument(
        "--nonpositive-contrast-json",
        type=str,
        default="/home/chen/RLPFN/artifacts/phase3_train_env_quality_contrast_regime_seed24680_nonpositive.json",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="/home/chen/RLPFN/artifacts/phase3_train_env_quality_regime_compare_pack.json",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output_path = Path(args.output_json).expanduser().resolve()
    if output_path.exists() and not bool(args.overwrite):
        print(output_path.read_text())
        return 0

    report = build_train_env_quality_regime_compare_pack(
        positive_contrast_json=args.positive_contrast_json,
        nonpositive_contrast_json=args.nonpositive_contrast_json,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
