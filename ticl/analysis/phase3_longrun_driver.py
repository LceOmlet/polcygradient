import argparse
import json
from pathlib import Path
from typing import Any

import torch

from ticl.analysis.prior_generalization_audit import run_prior_generalization_audit
from ticl.fit_model import main as fit_model_main


DEFAULT_PAIR1_TRAIN_SUITE = (
    "/home/chen/RLPFN/artifacts/phase3_cross_env_baseline/suites/train_suite.pt"
)
DEFAULT_PAIR1_HELDOUT_SUITE = (
    "/home/chen/RLPFN/artifacts/phase3_cross_env_baseline/suites/heldout_suite.pt"
)
DEFAULT_PAIR1_ZERO = (
    "/home/chen/RLPFN/artifacts/phase3_cross_env_baseline/zero_cross_env_control.json"
)

DEFAULT_PAIR2_TRAIN_SUITE = (
    "/home/chen/RLPFN/artifacts/phase3_cross_env_baseline_pair2/suites/train_suite.pt"
)
DEFAULT_PAIR2_HELDOUT_SUITE = (
    "/home/chen/RLPFN/artifacts/phase3_cross_env_baseline_pair2/suites/heldout_suite.pt"
)
DEFAULT_PAIR2_ZERO = (
    "/home/chen/RLPFN/artifacts/phase3_cross_env_baseline_pair2/zero_cross_env_control.json"
)


def _default_device(gpu_id: int) -> str:
    if torch.cuda.is_available():
        return f"cuda:{int(gpu_id)}"
    return "cpu"


def _latest_checkpoint(base_path: Path) -> Path:
    checkpoints = sorted(
        (path for path in base_path.rglob("*.cpkt") if path.is_file()),
        key=lambda path: (path.stat().st_mtime_ns, str(path)),
    )
    if not checkpoints:
        raise FileNotFoundError(f"No .cpkt checkpoints found under {base_path}")
    return checkpoints[-1]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _phase3_isolated_extra_config() -> dict[str, Any]:
    return {
        "optimizer": {
            "rl_objective": "ppo",
            "ppo_n_epochs": 4,
            "ppo_actor_gae_space": "normalized",
            "ppo_actor_baseline_mode": "learned",
            "ppo_separate_value_backbone": False,
            "ppo_reset_env_state_at_sep": True,
            "ppo_vf_coef": 0.5,
        },
        "prior": {
            "environment": {
                "normalized_q_value_weight": 0.0,
            }
        },
    }


def _phase3_shortrun_aligned_extra_config() -> dict[str, Any]:
    return {
        "optimizer": {
            "rl_objective": "ppo",
            "ppo_n_epochs": 4,
            "ppo_actor_gae_space": "normalized",
            "ppo_actor_baseline_mode": "zero",
            "ppo_separate_value_backbone": False,
            "ppo_reset_env_state_at_sep": False,
            "ppo_runtime_normalized_q_value_weight_override": 0.0,
            "ppo_runtime_next_state_flow_matching_weight_override": 0.0,
            "ppo_normalize_advantage": False,
            "ppo_vf_coef": 0.0,
        },
    }


def _suite_delta_summary(*, ppo_report: dict[str, Any], zero_report: dict[str, Any]) -> dict[str, Any]:
    ppo_train = ppo_report["train_suite"]["metrics"]
    ppo_heldout = ppo_report["heldout_suite"]["metrics"]
    zero_train = zero_report["train_suite"]["metrics"]
    zero_heldout = zero_report["heldout_suite"]["metrics"]
    return {
        "train": {
            "ppo_full_return_mean": float(ppo_train["full_return_mean"]),
            "ppo_suffix_return_mean": float(ppo_train["suffix_return_mean"]),
            "zero_full_return_mean": float(zero_train["full_return_mean"]),
            "zero_suffix_return_mean": float(zero_train["suffix_return_mean"]),
            "delta_full_return_mean": float(ppo_train["full_return_mean"] - zero_train["full_return_mean"]),
            "delta_suffix_return_mean": float(ppo_train["suffix_return_mean"] - zero_train["suffix_return_mean"]),
        },
        "heldout": {
            "ppo_full_return_mean": float(ppo_heldout["full_return_mean"]),
            "ppo_suffix_return_mean": float(ppo_heldout["suffix_return_mean"]),
            "zero_full_return_mean": float(zero_heldout["full_return_mean"]),
            "zero_suffix_return_mean": float(zero_heldout["suffix_return_mean"]),
            "delta_full_return_mean": float(ppo_heldout["full_return_mean"] - zero_heldout["full_return_mean"]),
            "delta_suffix_return_mean": float(ppo_heldout["suffix_return_mean"] - zero_heldout["suffix_return_mean"]),
        },
    }


def _run_pair(
    *,
    checkpoint_path: Path,
    device: str,
    n_samples: int,
    single_eval_pos: int,
    rollout_backend: str,
    train_suite_path: Path,
    heldout_suite_path: Path,
    zero_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ppo_report = run_prior_generalization_audit(
        checkpoint_path=str(checkpoint_path),
        policy_mode="ppo",
        device=device,
        n_samples=int(n_samples),
        single_eval_pos=int(single_eval_pos),
        suite_regime="cross_env",
        train_suite_path=str(train_suite_path),
        heldout_suite_path=str(heldout_suite_path),
        rollout_backend=str(rollout_backend),
    )
    output_path.write_text(json.dumps(ppo_report, indent=2, sort_keys=True))
    zero_report = _load_json(zero_path)
    return {
        "ppo_report_path": str(output_path.resolve()),
        "zero_report_path": str(zero_path.resolve()),
        "comparison_vs_zero": _suite_delta_summary(ppo_report=ppo_report, zero_report=zero_report),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Phase 3 long-run driver: continue training, then evaluate pair1/pair2 fixed suites.")
    parser.add_argument("--warm-start-from", required=True, help="Checkpoint to continue training from.")
    parser.add_argument("--base-path", required=True, help="Run directory for the continued-training job and outputs.")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--stop-after-epochs", type=int, default=8)
    parser.add_argument("--n-samples", type=int, default=2048)
    parser.add_argument("--single-eval-pos", type=int, default=1946)
    parser.add_argument("--rollout-backend", type=str, default="serial", choices=["family_vectorized", "serial"])
    parser.add_argument(
        "--phase3-train-profile",
        type=str,
        default=None,
        choices=["isolated", "shortrun_aligned"],
        help="Explicit training-profile override. 'shortrun_aligned' matches the actor-only/no-aux short-run surrogate.",
    )
    parser.add_argument("--phase3-isolated", dest="phase3_isolated", action="store_true")
    parser.add_argument("--no-phase3-isolated", dest="phase3_isolated", action="store_false")
    parser.add_argument("--pair1-train-suite-path", type=str, default=DEFAULT_PAIR1_TRAIN_SUITE)
    parser.add_argument("--pair1-heldout-suite-path", type=str, default=DEFAULT_PAIR1_HELDOUT_SUITE)
    parser.add_argument("--pair1-zero-path", type=str, default=DEFAULT_PAIR1_ZERO)
    parser.add_argument("--pair2-train-suite-path", type=str, default=DEFAULT_PAIR2_TRAIN_SUITE)
    parser.add_argument("--pair2-heldout-suite-path", type=str, default=DEFAULT_PAIR2_HELDOUT_SUITE)
    parser.add_argument("--pair2-zero-path", type=str, default=DEFAULT_PAIR2_ZERO)
    parser.add_argument("--output", type=str, default=None, help="Summary JSON path. Defaults to <base-path>/phase3_longrun_summary.json.")
    parser.set_defaults(phase3_isolated=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    base_path = Path(args.base_path).expanduser().resolve()
    base_path.mkdir(parents=True, exist_ok=True)
    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output is not None
        else (base_path / "phase3_longrun_summary.json").resolve()
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    device = str(args.device or _default_device(args.gpu_id))
    if args.phase3_train_profile == "shortrun_aligned":
        extra_config = _phase3_shortrun_aligned_extra_config()
        train_profile = "shortrun_aligned"
    elif args.phase3_train_profile == "isolated":
        extra_config = _phase3_isolated_extra_config()
        train_profile = "isolated"
    else:
        extra_config = _phase3_isolated_extra_config() if bool(args.phase3_isolated) else None
        train_profile = "isolated" if extra_config is not None else "checkpoint_resume"
    fit_args = [
        "rlpfn",
        "-g", str(args.gpu_id),
        "-f", str(Path(args.warm_start_from).expanduser().resolve()),
        "-c",
        "-B", str(base_path),
        "--save-every", str(args.save_every),
        "--stop-after-epochs", str(args.stop_after_epochs),
        "--validate", "false",
        "--rl-validate-enabled", "false",
        "--progress-bar", "false",
    ]

    print("[phase3-longrun] continue-training start")
    if extra_config is not None:
        print(f"[phase3-longrun] training-profile={train_profile}")
    fit_model_main(fit_args, extra_config=extra_config)

    latest_checkpoint = _latest_checkpoint(base_path)
    print(f"[phase3-longrun] latest_checkpoint={latest_checkpoint}")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    pair1 = _run_pair(
        checkpoint_path=latest_checkpoint,
        device=device,
        n_samples=int(args.n_samples),
        single_eval_pos=int(args.single_eval_pos),
        rollout_backend=str(args.rollout_backend),
        train_suite_path=Path(args.pair1_train_suite_path).expanduser().resolve(),
        heldout_suite_path=Path(args.pair1_heldout_suite_path).expanduser().resolve(),
        zero_path=Path(args.pair1_zero_path).expanduser().resolve(),
        output_path=(base_path / "phase3_pair1_ppo.json").resolve(),
    )
    pair2 = _run_pair(
        checkpoint_path=latest_checkpoint,
        device=device,
        n_samples=int(args.n_samples),
        single_eval_pos=int(args.single_eval_pos),
        rollout_backend=str(args.rollout_backend),
        train_suite_path=Path(args.pair2_train_suite_path).expanduser().resolve(),
        heldout_suite_path=Path(args.pair2_heldout_suite_path).expanduser().resolve(),
        zero_path=Path(args.pair2_zero_path).expanduser().resolve(),
        output_path=(base_path / "phase3_pair2_ppo.json").resolve(),
    )

    summary = {
        "driver": "phase3_longrun_driver",
        "warm_start_from": str(Path(args.warm_start_from).expanduser().resolve()),
        "latest_checkpoint": str(latest_checkpoint),
        "config": {
            "gpu_id": int(args.gpu_id),
            "device": device,
            "save_every": int(args.save_every),
            "stop_after_epochs": int(args.stop_after_epochs),
            "n_samples": int(args.n_samples),
            "single_eval_pos": int(args.single_eval_pos),
            "rollout_backend": str(args.rollout_backend),
            "phase3_isolated": bool(args.phase3_isolated),
            "phase3_train_profile": str(train_profile),
        },
        "training_overrides": extra_config,
        "pair1": pair1,
        "pair2": pair2,
    }
    output_path.write_text(json.dumps(summary, indent=2, sort_keys=True))

    for pair_name in ("pair1", "pair2"):
        heldout = summary[pair_name]["comparison_vs_zero"]["heldout"]
        print(
            f"[phase3-longrun] {pair_name} "
            f"heldout_delta_full={heldout['delta_full_return_mean']:.6f} "
            f"heldout_delta_suffix={heldout['delta_suffix_return_mean']:.6f}"
        )
    print(f"[phase3-longrun] saved_summary={output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
