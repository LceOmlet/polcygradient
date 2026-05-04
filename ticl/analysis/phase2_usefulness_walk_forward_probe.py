import argparse
import json
from pathlib import Path

import torch

from ticl.analysis.critic_free_single_env_audit import (
    _build_audit_ppo_policy_step_fn,
    _build_zero_policy_step_fn,
    _run_eval,
)
from ticl.analysis.phase2_walk_forward_critic_generalization_probe import (
    _build_algo,
    _collect_rollout,
    _suite_from_seed_range,
)
from ticl.analysis.critic_free_single_env_audit import (
    _build_audit_env_cfg,
    _measure_current_policy_rollout_critic_quality,
    _measure_rollout_critic_quality,
    _prepare_audit_fixed_env_contract,
    _seed_all,
)
from ticl.model_builder import load_model


def _default_device() -> str:
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def _evaluate_policy_suffix_returns(
    *,
    algo,
    prior,
    suite: dict,
    n_steps: int,
    num_features: int,
    single_eval_pos: int,
    device_obj: torch.device,
) -> dict:
    smp_metrics = _run_eval(
        prior=prior,
        suite=suite,
        policy_step_fn=_build_audit_ppo_policy_step_fn(
            algo.policy,
            sampled=True,
            single_eval_pos=int(single_eval_pos),
            boundary_contract_mode="normal",
            deterministic_actor_sampling=True,
        ),
        n_samples=int(n_steps),
        num_features=int(num_features),
        single_eval_pos=int(single_eval_pos),
        device=device_obj,
    )
    det_metrics = _run_eval(
        prior=prior,
        suite=suite,
        policy_step_fn=_build_audit_ppo_policy_step_fn(
            algo.policy,
            sampled=False,
            single_eval_pos=int(single_eval_pos),
            boundary_contract_mode="normal",
            deterministic_actor_sampling=True,
        ),
        n_samples=int(n_steps),
        num_features=int(num_features),
        single_eval_pos=int(single_eval_pos),
        device=device_obj,
    )
    return {
        "smp_suffix_return_mean": float(smp_metrics["suffix_return_mean"]),
        "det_suffix_return_mean": float(det_metrics["suffix_return_mean"]),
    }


def _run_mode(
    *,
    checkpoint_path: str,
    device_obj: torch.device,
    frozen_h,
    train_env_seed: int,
    train_rollout_seed: int,
    heldout_suite: dict,
    zero_policy_suffix_return_mean: float,
    single_eval_pos: int,
    n_steps: int,
    batch_size: int,
    n_epochs: int,
    learning_rate: float,
    target_kl: float,
    outer_epochs: int,
    build_seed: int,
    ppo_reset_env_state_at_sep: bool,
    actor_baseline_mode: str,
) -> dict:
    algo, callback, vec_env, env_cfg, num_features = _build_algo(
        checkpoint_path=checkpoint_path,
        device_obj=device_obj,
        frozen_h=frozen_h,
        train_env_seed=int(train_env_seed),
        train_rollout_seed=int(train_rollout_seed),
        single_eval_pos=int(single_eval_pos),
        n_steps=int(n_steps),
        batch_size=int(batch_size),
        n_epochs=int(n_epochs),
        learning_rate=float(learning_rate),
        target_kl=float(target_kl),
        ppo_reset_env_state_at_sep=bool(ppo_reset_env_state_at_sep),
        actor_baseline_mode=str(actor_baseline_mode),
        build_seed=int(build_seed),
    )
    total_timesteps = int(n_steps) * int(outer_epochs)
    history: list[dict] = []
    try:
        pre_eval = _evaluate_policy_suffix_returns(
            algo=algo,
            prior=vec_env.env_prior,
            suite=heldout_suite,
            n_steps=int(n_steps),
            num_features=int(num_features),
            single_eval_pos=int(single_eval_pos),
            device_obj=device_obj,
        )
        initial_smp = float(pre_eval["smp_suffix_return_mean"])
        initial_det = float(pre_eval["det_suffix_return_mean"])

        _collect_rollout(algo, callback, vec_env, progress_timestep=0, total_timesteps=total_timesteps)
        for outer_idx in range(int(outer_epochs)):
            pre_quality = _measure_rollout_critic_quality(algo)
            algo.train()
            post_same_quality = _measure_current_policy_rollout_critic_quality(algo)
            heldout_eval = _evaluate_policy_suffix_returns(
                algo=algo,
                prior=vec_env.env_prior,
                suite=heldout_suite,
                n_steps=int(n_steps),
                num_features=int(num_features),
                single_eval_pos=int(single_eval_pos),
                device_obj=device_obj,
            )
            progress_next = min(int((outer_idx + 1) * n_steps), total_timesteps)
            _collect_rollout(
                algo,
                callback,
                vec_env,
                progress_timestep=progress_next,
                total_timesteps=total_timesteps,
            )
            walk_forward_quality = _measure_rollout_critic_quality(algo)
            smp_suffix = float(heldout_eval["smp_suffix_return_mean"])
            det_suffix = float(heldout_eval["det_suffix_return_mean"])
            history.append(
                {
                    "outer_epoch": int(outer_idx + 1),
                    "pre_rollout_ev_norm": float(pre_quality["explained_variance_normalized"]),
                    "post_same_rollout_ev_norm": float(post_same_quality["explained_variance_normalized"]),
                    "walk_forward_next_rollout_ev_norm": float(walk_forward_quality["explained_variance_normalized"]),
                    "walk_forward_next_rollout_raw_corr": float(walk_forward_quality["raw_corr"]),
                    "heldout_smp_suffix_return_mean": smp_suffix,
                    "heldout_det_suffix_return_mean": det_suffix,
                    "heldout_smp_gap_vs_zero_policy": float(smp_suffix - float(zero_policy_suffix_return_mean)),
                    "heldout_det_gap_vs_zero_policy": float(det_suffix - float(zero_policy_suffix_return_mean)),
                    "heldout_smp_delta_from_init": float(smp_suffix - initial_smp),
                    "heldout_det_delta_from_init": float(det_suffix - initial_det),
                }
            )
        return {
            "initial_eval": {
                "smp_suffix_return_mean": float(initial_smp),
                "det_suffix_return_mean": float(initial_det),
            },
            "history": history,
        }
    finally:
        vec_env.close()
        del algo, callback, vec_env
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def run_phase2_usefulness_walk_forward_probe(
    *,
    checkpoint_path: str,
    device: str | None = None,
    frozen_h_seed: int = 12345,
    train_env_seed: int = 2020,
    train_rollout_seed: int = 4040,
    heldout_env_seed_start: int = 3000,
    heldout_rollout_seed_start: int = 4000,
    heldout_env_count: int = 8,
    single_eval_pos: int = 64,
    n_steps: int = 256,
    batch_size: int = 256,
    n_epochs: int = 1,
    learning_rate: float = 2e-4,
    target_kl: float = 0.03,
    outer_epochs: int = 4,
    build_seed: int = 4040,
    ppo_reset_env_state_at_sep: bool = True,
) -> dict:
    device_obj = torch.device(str(device or _default_device()))
    _seed_all(int(build_seed))
    load_model.cache_clear()
    _, config = load_model(checkpoint_path, device=device_obj, verbose=False)
    env_cfg = _build_audit_env_cfg(config["prior"]["environment"])
    num_features = int(config["prior"]["num_features"])
    fixed_bundle = _prepare_audit_fixed_env_contract(
        env_cfg=env_cfg,
        boundary_contract_mode="normal",
        ppo_reset_env_state_at_sep=bool(ppo_reset_env_state_at_sep),
        frozen_h_seed=int(frozen_h_seed),
        frozen_h_seed_mode="current",
    )
    frozen_h = fixed_bundle["frozen_h"]
    heldout_suite = _suite_from_seed_range(
        frozen_h=frozen_h,
        env_seed_start=int(heldout_env_seed_start),
        rollout_seed_start=int(heldout_rollout_seed_start),
        count=int(heldout_env_count),
    )

    # Zero-policy baseline on the same heldout suite.
    from ticl.priors.environment_prior import EnvironmentPrior
    import copy

    prior = EnvironmentPrior(copy.deepcopy(env_cfg))
    prior._sample_batch_hypers = lambda batch_n, _h=frozen_h: [copy.deepcopy(_h) for _ in range(int(batch_n))]
    prior._sample_single_eval_pos = lambda n_samples_arg, rng_seed=None: int(single_eval_pos)
    zero_metrics = _run_eval(
        prior=prior,
        suite=heldout_suite,
        policy_step_fn=_build_zero_policy_step_fn(),
        n_samples=int(n_steps),
        num_features=int(num_features),
        single_eval_pos=int(single_eval_pos),
        device=device_obj,
    )
    zero_policy_suffix_return_mean = float(zero_metrics["suffix_return_mean"])

    learned = _run_mode(
        checkpoint_path=checkpoint_path,
        device_obj=device_obj,
        frozen_h=frozen_h,
        train_env_seed=int(train_env_seed),
        train_rollout_seed=int(train_rollout_seed),
        heldout_suite=heldout_suite,
        zero_policy_suffix_return_mean=float(zero_policy_suffix_return_mean),
        single_eval_pos=int(single_eval_pos),
        n_steps=int(n_steps),
        batch_size=int(batch_size),
        n_epochs=int(n_epochs),
        learning_rate=float(learning_rate),
        target_kl=float(target_kl),
        outer_epochs=int(outer_epochs),
        build_seed=int(build_seed),
        ppo_reset_env_state_at_sep=bool(ppo_reset_env_state_at_sep),
        actor_baseline_mode="learned",
    )
    zero = _run_mode(
        checkpoint_path=checkpoint_path,
        device_obj=device_obj,
        frozen_h=frozen_h,
        train_env_seed=int(train_env_seed),
        train_rollout_seed=int(train_rollout_seed),
        heldout_suite=heldout_suite,
        zero_policy_suffix_return_mean=float(zero_policy_suffix_return_mean),
        single_eval_pos=int(single_eval_pos),
        n_steps=int(n_steps),
        batch_size=int(batch_size),
        n_epochs=int(n_epochs),
        learning_rate=float(learning_rate),
        target_kl=float(target_kl),
        outer_epochs=int(outer_epochs),
        build_seed=int(build_seed),
        ppo_reset_env_state_at_sep=bool(ppo_reset_env_state_at_sep),
        actor_baseline_mode="zero",
    )

    side_by_side: list[dict] = []
    for learned_row, zero_row in zip(learned["history"], zero["history"]):
        side_by_side.append(
            {
                "outer_epoch": int(learned_row["outer_epoch"]),
                "learned_walk_forward_ev_norm": float(learned_row["walk_forward_next_rollout_ev_norm"]),
                "zero_walk_forward_ev_norm": float(zero_row["walk_forward_next_rollout_ev_norm"]),
                "learned_walk_forward_raw_corr": float(learned_row["walk_forward_next_rollout_raw_corr"]),
                "zero_walk_forward_raw_corr": float(zero_row["walk_forward_next_rollout_raw_corr"]),
                "learned_heldout_smp_delta_from_init": float(learned_row["heldout_smp_delta_from_init"]),
                "zero_heldout_smp_delta_from_init": float(zero_row["heldout_smp_delta_from_init"]),
                "learned_minus_zero_heldout_smp_delta": float(
                    learned_row["heldout_smp_delta_from_init"] - zero_row["heldout_smp_delta_from_init"]
                ),
                "learned_heldout_smp_gap_vs_zero_policy": float(learned_row["heldout_smp_gap_vs_zero_policy"]),
                "zero_heldout_smp_gap_vs_zero_policy": float(zero_row["heldout_smp_gap_vs_zero_policy"]),
                "learned_minus_zero_heldout_smp_gap": float(
                    learned_row["heldout_smp_gap_vs_zero_policy"] - zero_row["heldout_smp_gap_vs_zero_policy"]
                ),
            }
        )

    return {
        "audit_entry": "phase2_usefulness_walk_forward_probe",
        "checkpoint_path": str(Path(checkpoint_path).expanduser().resolve()),
        "device": str(device_obj),
        "frozen_h_seed": int(frozen_h_seed),
        "train_env_seed": int(train_env_seed),
        "train_rollout_seed": int(train_rollout_seed),
        "heldout_env_seed_start": int(heldout_env_seed_start),
        "heldout_rollout_seed_start": int(heldout_rollout_seed_start),
        "heldout_env_count": int(heldout_env_count),
        "single_eval_pos": int(single_eval_pos),
        "n_steps": int(n_steps),
        "batch_size": int(batch_size),
        "n_epochs": int(n_epochs),
        "learning_rate": float(learning_rate),
        "target_kl": float(target_kl),
        "outer_epochs": int(outer_epochs),
        "build_seed": int(build_seed),
        "ppo_reset_env_state_at_sep": bool(ppo_reset_env_state_at_sep),
        "fixed_env_contract": dict(fixed_bundle["fixed_env_contract"]),
        "zero_policy_suffix_return_mean": float(zero_policy_suffix_return_mean),
        "learned": learned,
        "zero": zero,
        "side_by_side": side_by_side,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_path", type=str)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--frozen-h-seed", type=int, default=12345)
    parser.add_argument("--train-env-seed", type=int, default=2020)
    parser.add_argument("--train-rollout-seed", type=int, default=4040)
    parser.add_argument("--heldout-env-seed-start", type=int, default=3000)
    parser.add_argument("--heldout-rollout-seed-start", type=int, default=4000)
    parser.add_argument("--heldout-env-count", type=int, default=8)
    parser.add_argument("--single-eval-pos", type=int, default=64)
    parser.add_argument("--n-steps", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--target-kl", type=float, default=0.03)
    parser.add_argument("--outer-epochs", type=int, default=4)
    parser.add_argument("--build-seed", type=int, default=4040)
    parser.add_argument("--ppo-reset-env-state-at-sep", action="store_true")
    parser.add_argument(
        "--output-json",
        type=str,
        default="/home/chen/RLPFN/artifacts/phase2_usefulness_walk_forward_probe.json",
    )
    args = parser.parse_args()

    report = run_phase2_usefulness_walk_forward_probe(
        checkpoint_path=args.checkpoint_path,
        device=args.device,
        frozen_h_seed=int(args.frozen_h_seed),
        train_env_seed=int(args.train_env_seed),
        train_rollout_seed=int(args.train_rollout_seed),
        heldout_env_seed_start=int(args.heldout_env_seed_start),
        heldout_rollout_seed_start=int(args.heldout_rollout_seed_start),
        heldout_env_count=int(args.heldout_env_count),
        single_eval_pos=int(args.single_eval_pos),
        n_steps=int(args.n_steps),
        batch_size=int(args.batch_size),
        n_epochs=int(args.n_epochs),
        learning_rate=float(args.learning_rate),
        target_kl=float(args.target_kl),
        outer_epochs=int(args.outer_epochs),
        build_seed=int(args.build_seed),
        ppo_reset_env_state_at_sep=bool(args.ppo_reset_env_state_at_sep),
    )
    out_path = Path(args.output_json).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(report, sort_keys=True)
    out_path.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
