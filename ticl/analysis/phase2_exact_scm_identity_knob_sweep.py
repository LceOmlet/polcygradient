import argparse
import copy
import json
from pathlib import Path
from typing import Any

from ticl.analysis.critic_free_single_env_audit import _seed_all
from ticl.analysis.phase2_suffix_state_identity_probe import run_phase2_suffix_state_identity_probe


def _case(name: str, *, overrides: dict[str, Any], note: str) -> dict[str, Any]:
    return {
        "name": str(name),
        "overrides": copy.deepcopy(overrides),
        "note": str(note),
    }


def build_default_cases() -> list[dict[str, Any]]:
    return [
        _case(
            "baseline",
            overrides={},
            note="Current exact-SCM sampled initialization under the low-noise-dim mainline regime.",
        ),
        _case(
            "low_init_std",
            overrides={"init_std": 0.05},
            note="Shrink the overall linear weight scale while keeping the rest of the sampler fixed.",
        ),
        _case(
            "low_hidden_noise_std",
            overrides={"noise_std": 0.01},
            note="Reduce hidden-layer stochastic weight noise without touching the input/output structure.",
        ),
        _case(
            "tanh_activation",
            overrides={"prior_mlp_activations": "tanh"},
            note="Use bounded smooth nonlinearities that are closer to common Gym-style control approximators.",
        ),
        _case(
            "relu_activation",
            overrides={"prior_mlp_activations": "relu"},
            note="Use piecewise-linear nonlinearities while keeping the rest of the exact SCM unchanged.",
        ),
        _case(
            "tanh_low_init_low_noise",
            overrides={
                "prior_mlp_activations": "tanh",
                "init_std": 0.05,
                "noise_std": 0.01,
            },
            note="Combine bounded nonlinearities with a milder initialization/noise scale.",
        ),
        _case(
            "relu_low_init_low_noise",
            overrides={
                "prior_mlp_activations": "relu",
                "init_std": 0.05,
                "noise_std": 0.01,
            },
            note="Combine ReLU with a milder initialization/noise scale.",
        ),
        _case(
            "tanh_low_init_low_noise_no_standard_init",
            overrides={
                "prior_mlp_activations": "tanh",
                "init_std": 0.05,
                "noise_std": 0.01,
                "scm_standard_linear_init_enabled": False,
            },
            note="Keep the same mild scale, but disable fan-aware standard SCM linear initialization.",
        ),
    ]


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def run_sweep(
    *,
    output_dir: str,
    device: str | None = None,
    build_seed: int = 4040,
    frozen_h_seed: int = 12345,
    train_env_seed: int = 2020,
    train_rollout_seed: int = 4040,
    single_eval_pos: int = 64,
    anchor_step: int = 128,
    n_steps: int = 256,
    batch_size: int = 1024,
    reference_state_count: int = 32,
    continuation_repeats: int = 32,
    discount_gamma: float = 0.99,
) -> dict[str, Any]:
    output_root = Path(output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    cases = build_default_cases()

    base_kwargs = {
        "checkpoint_path": None,
        "device": device,
        "from_scratch": True,
        "from_scratch_model_type": "rlpfn",
        "frozen_h_seed": int(frozen_h_seed),
        "train_env_seed": int(train_env_seed),
        "train_rollout_seed": int(train_rollout_seed),
        "single_eval_pos": int(single_eval_pos),
        "anchor_step": int(anchor_step),
        "n_steps": int(n_steps),
        "batch_size": int(batch_size),
        "n_epochs": 1,
        "learning_rate": 2e-4,
        "target_kl": 0.03,
        "outer_epochs": 0,
        "build_seed": int(build_seed),
        "ppo_reset_env_state_at_sep": True,
        "strict_native_rollout": False,
        "value_head_impl": "scalar_mlp",
        "value_path_adapter_impl": "none",
        "value_head_mlp_hidden_dim": 256,
        "space_contract": "normalized",
        "actor_baseline_mode": "learned",
        "vf_coef_override": None,
        "policy_loss_coef_override": None,
        "behavior_policy": "unit_gaussian",
        "reference_state_count": int(reference_state_count),
        "continuation_repeats": int(continuation_repeats),
        "reference_rollout_seed_start": 5000,
        "continuation_rollout_seed_start": 10000,
        "sampled_policy": True,
        "discount_gamma": float(discount_gamma),
        "constrained_dim_sampling_total_budget_override": 305,
        "state_dim_override": 300,
        "noise_dim_override": 5,
        "zero_pad_dim_override": 0,
        "action_dim_override": None,
        "alpha_override": 1.0,
        "reference_state_inertia_enabled_override": True,
        "terminal_reset_enabled_override": None,
        "terminal_reset_count_target_override": None,
        "print_progress_every_repeat": False,
    }

    results: list[dict[str, Any]] = []
    baseline_identity = None
    baseline_noise = None
    baseline_signal = None

    for case in cases:
        case_name = str(case["name"])
        case_dir = output_root / case_name
        case_dir.mkdir(parents=True, exist_ok=True)
        progress_path = case_dir / f"{case_name}.progress.jsonl"
        output_path = case_dir / f"{case_name}.json"
        report = run_phase2_suffix_state_identity_probe(
            **base_kwargs,
            frozen_h_overrides=copy.deepcopy(case["overrides"]),
            progress_jsonl=str(progress_path),
        )
        output_path.write_text(json.dumps(report, sort_keys=True) + "\n", encoding="utf-8")
        aggregate = report["aggregate"]
        row = {
            "case": case_name,
            "note": str(case["note"]),
            "overrides": copy.deepcopy(case["overrides"]),
            "output_json": str(output_path),
            "progress_jsonl": str(progress_path),
            "identity_score": float(aggregate["identity_score"]),
            "noise": float(aggregate["noise"]),
            "signal": float(aggregate["signal"]),
            "total_variance": float(aggregate["total_variance"]),
            "mean_of_state_means": float(aggregate["mean_of_state_means"]),
            "std_of_state_means": float(aggregate["std_of_state_means"]),
            "mean_within_state_std": float(aggregate["mean_within_state_std"]),
            "effective_frozen_h_snapshot": copy.deepcopy(report.get("effective_frozen_h_snapshot")),
            "sampled_env_snapshot": copy.deepcopy(report.get("sampled_env_snapshot")),
        }
        if baseline_identity is None:
            baseline_identity = float(row["identity_score"])
            baseline_noise = float(row["noise"])
            baseline_signal = float(row["signal"])
        row["identity_delta_vs_baseline"] = float(row["identity_score"] - float(baseline_identity))
        row["identity_ratio_vs_baseline"] = (
            None
            if float(baseline_identity) == 0.0
            else float(row["identity_score"]) / float(baseline_identity)
        )
        row["noise_ratio_vs_baseline"] = (
            None if float(baseline_noise) == 0.0 else float(row["noise"]) / float(baseline_noise)
        )
        row["signal_ratio_vs_baseline"] = (
            None if float(baseline_signal) == 0.0 else float(row["signal"]) / float(baseline_signal)
        )
        results.append(row)
        print(
            (
                f"[exact-scm-sweep] {case_name} "
                f"identity={float(row['identity_score']):.4f} "
                f"noise={float(row['noise']):.4f} "
                f"signal={float(row['signal']):.4f}"
            ),
            flush=True,
        )

    ranked = sorted(results, key=lambda item: float(item["identity_score"]), reverse=True)
    for rank_idx, item in enumerate(ranked, start=1):
        item["rank_by_identity"] = int(rank_idx)

    summary = {
        "audit_entry": "phase2_exact_scm_identity_knob_sweep",
        "mainline_settings": {
            "behavior_policy": "unit_gaussian",
            "sampled_policy": True,
            "anchor_step": int(anchor_step),
            "single_eval_pos": int(single_eval_pos),
            "n_steps": int(n_steps),
            "discount_gamma": float(discount_gamma),
            "reference_state_count": int(reference_state_count),
            "continuation_repeats": int(continuation_repeats),
            "batch_size": int(batch_size),
            "constrained_dim_sampling_total_budget_override": 305,
            "state_dim_override": 300,
            "noise_dim_override": 5,
            "zero_pad_dim_override": 0,
            "alpha_override": 1.0,
            "terminal_reset_enabled_override": None,
            "terminal_reset_count_target_override": None,
        },
        "cases": ranked,
        "best_case": None if len(ranked) == 0 else copy.deepcopy(ranked[0]),
    }
    (output_root / "summary.json").write_text(json.dumps(summary, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep exact-SCM initialization and scalar knobs that keep the SCM structure fixed, "
            "and measure midpoint discounted state-return identity."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/home/chen/RLPFN/artifacts/phase2_exact_scm_identity_knob_sweep",
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--build-seed", type=int, default=4040)
    parser.add_argument("--frozen-h-seed", type=int, default=12345)
    parser.add_argument("--train-env-seed", type=int, default=2020)
    parser.add_argument("--train-rollout-seed", type=int, default=4040)
    parser.add_argument("--single-eval-pos", type=int, default=64)
    parser.add_argument("--anchor-step", type=int, default=128)
    parser.add_argument("--n-steps", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--reference-state-count", type=int, default=32)
    parser.add_argument("--continuation-repeats", type=int, default=32)
    parser.add_argument("--discount-gamma", type=float, default=0.99)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    _seed_all(int(args.build_seed))
    summary = run_sweep(
        output_dir=str(args.output_dir),
        device=args.device,
        build_seed=int(args.build_seed),
        frozen_h_seed=int(args.frozen_h_seed),
        train_env_seed=int(args.train_env_seed),
        train_rollout_seed=int(args.train_rollout_seed),
        single_eval_pos=int(args.single_eval_pos),
        anchor_step=int(args.anchor_step),
        n_steps=int(args.n_steps),
        batch_size=int(args.batch_size),
        reference_state_count=int(args.reference_state_count),
        continuation_repeats=int(args.continuation_repeats),
        discount_gamma=float(args.discount_gamma),
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
