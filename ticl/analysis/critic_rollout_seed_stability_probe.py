import argparse
import json
from pathlib import Path

import torch

from ticl.analysis.critic_value_fit_probe import (
    _build_algo,
    _build_fixed_env_h,
    _collect_fixed_rollout,
)
from ticl.analysis.critic_free_single_env_audit import _build_audit_env_cfg
from ticl.model_builder import load_model
from ticl.sb3_recurrent_ppo import _recover_raw_from_value_space


def _default_device() -> str:
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def _masked_corrcoef(x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> float:
    mask = mask.bool().reshape(-1)
    if int(mask.sum().item()) <= 1:
        return 0.0
    x = x.reshape(-1)[mask].float()
    y = y.reshape(-1)[mask].float()
    x = x - x.mean()
    y = y - y.mean()
    denom = torch.linalg.vector_norm(x) * torch.linalg.vector_norm(y)
    if float(denom.item()) <= 0.0:
        return 0.0
    return float((x @ y / denom).item())


def _rollout_summary(rollout_data) -> dict:
    objective_mask = (rollout_data.objective_masks > 1e-8).detach().reshape(-1)
    raw_returns = _recover_raw_from_value_space(
        rollout_data.returns.detach(),
        value_means=rollout_data.rollout_return_means.detach(),
        value_stds=rollout_data.rollout_return_stds.detach(),
    ).reshape(-1)
    return {
        "objective_mask": objective_mask,
        "raw_returns": raw_returns,
        "raw_return_mean": float(raw_returns[objective_mask].float().mean().item()) if int(objective_mask.sum().item()) > 0 else 0.0,
        "raw_return_std": float(raw_returns[objective_mask].float().std().item()) if int(objective_mask.sum().item()) > 1 else 0.0,
    }


def _pairwise_rollout_compare(anchor: dict, other: dict) -> dict:
    mask = anchor["objective_mask"] & other["objective_mask"]
    return {
        "objective_total": int(mask.sum().item()),
        "raw_target_corr": _masked_corrcoef(anchor["raw_returns"], other["raw_returns"], mask),
        "raw_return_mean_delta": float(other["raw_return_mean"] - anchor["raw_return_mean"]),
        "raw_return_std_delta": float(other["raw_return_std"] - anchor["raw_return_std"]),
    }


def _set_deterministic_actor_sampling(algo) -> None:
    original_make_step_fn = algo.policy.make_vectorized_rollout_step_fn

    def _wrapped_make_step_fn():
        fn = original_make_step_fn()
        fn._policy_actor_sample_fn = (  # type: ignore[attr-defined]
            lambda actor_outputs, noise: actor_outputs["action_mean"]
        )
        return fn

    algo.policy.make_vectorized_rollout_step_fn = _wrapped_make_step_fn  # type: ignore[method-assign]


def _force_rollout_seed_injection(algo, *, env_seed: int, rollout_seed: int):
    env_prior = algo._rwkv_env_prior
    original = env_prior._rollout_family_group_vectorized_with_policy

    def _wrapped_rollout(*args, **kwargs):
        batch_size = int(len(args[0])) if len(args) > 0 else int(len(kwargs["h_list"]))
        kwargs = dict(kwargs)
        kwargs["env_rng_seeds"] = [int(env_seed)] * batch_size
        kwargs["rollout_rng_seeds"] = [int(rollout_seed)] * batch_size
        return original(*args, **kwargs)

    env_prior._rollout_family_group_vectorized_with_policy = _wrapped_rollout
    return original


def run_critic_rollout_seed_stability_probe(
    *,
    checkpoint_path: str,
    device: str | None = None,
    frozen_h_seed: int = 12345,
    train_env_seed: int = 2020,
    rollout_seed: int = 4040,
    single_eval_pos: int = 64,
    n_steps: int = 256,
    batch_size: int = 256,
    n_epochs: int = 1,
    learning_rate: float = 2e-4,
    target_kl: float = 0.03,
    ppo_reset_env_state_at_sep: bool = True,
    ppo_separate_value_backbone: bool = True,
    deterministic_actor_sampling: bool = True,
    num_rollouts: int = 3,
):
    device_obj = torch.device(str(device or _default_device()))
    load_model.cache_clear()
    _, config = load_model(checkpoint_path, device=device_obj, verbose=False)
    env_cfg = _build_audit_env_cfg(config["prior"]["environment"])
    frozen_h = _build_fixed_env_h(prior_cfg=env_cfg, frozen_h_seed=int(frozen_h_seed))

    def _build():
        algo, callback, vec_env, prior, _ = _build_algo(
            checkpoint_path=checkpoint_path,
            device_obj=device_obj,
            frozen_h=frozen_h,
            train_env_seed=int(train_env_seed),
            single_eval_pos=int(single_eval_pos),
            n_steps=int(n_steps),
            batch_size=int(batch_size),
            n_epochs=int(n_epochs),
            learning_rate=float(learning_rate),
            target_kl=float(target_kl),
            ppo_reset_env_state_at_sep=bool(ppo_reset_env_state_at_sep),
            ppo_separate_value_backbone=bool(ppo_separate_value_backbone),
        )
        if bool(deterministic_actor_sampling):
            _set_deterministic_actor_sampling(algo)
        return algo, callback, vec_env, prior

    def _collect_series(*, force_seed: bool):
        algo, callback, vec_env, prior = _build()
        original = None
        if force_seed:
            original = _force_rollout_seed_injection(
                algo,
                env_seed=int(train_env_seed),
                rollout_seed=int(rollout_seed),
            )
        try:
            collected = []
            for _ in range(int(num_rollouts)):
                rollout_data = _collect_fixed_rollout(algo, callback, vec_env)
                collected.append(_rollout_summary(rollout_data))
            anchor = collected[0]
            return {
                "anchor_raw_return_mean": float(anchor["raw_return_mean"]),
                "anchor_raw_return_std": float(anchor["raw_return_std"]),
                "pairwise_vs_anchor": [
                    {
                        "rollout_idx": int(idx + 1),
                        **_pairwise_rollout_compare(anchor, item),
                    }
                    for idx, item in enumerate(collected[1:])
                ],
            }
        finally:
            if original is not None:
                prior._rollout_family_group_vectorized_with_policy = original
            vec_env.close()
            del algo, callback, vec_env, prior
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return {
        "audit_entry": "critic_rollout_seed_stability_probe",
        "checkpoint_path": str(Path(checkpoint_path).expanduser().resolve()),
        "device": str(device_obj),
        "frozen_h_seed": int(frozen_h_seed),
        "train_env_seed": int(train_env_seed),
        "rollout_seed": int(rollout_seed),
        "single_eval_pos": int(single_eval_pos),
        "n_steps": int(n_steps),
        "batch_size": int(batch_size),
        "n_epochs": int(n_epochs),
        "learning_rate": float(learning_rate),
        "target_kl": float(target_kl),
        "ppo_reset_env_state_at_sep": bool(ppo_reset_env_state_at_sep),
        "ppo_separate_value_backbone": bool(ppo_separate_value_backbone),
        "deterministic_actor_sampling": bool(deterministic_actor_sampling),
        "num_rollouts": int(num_rollouts),
        "current_collect_rollouts": _collect_series(force_seed=False),
        "forced_seed_collect_rollouts": _collect_series(force_seed=True),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_path", type=str)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--frozen-h-seed", type=int, default=12345)
    parser.add_argument("--train-env-seed", type=int, default=2020)
    parser.add_argument("--rollout-seed", type=int, default=4040)
    parser.add_argument("--single-eval-pos", type=int, default=64)
    parser.add_argument("--n-steps", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--target-kl", type=float, default=0.03)
    parser.add_argument("--ppo-reset-env-state-at-sep", action="store_true")
    parser.add_argument("--ppo-separate-value-backbone", action="store_true")
    parser.add_argument("--deterministic-actor-sampling", action="store_true")
    parser.add_argument("--num-rollouts", type=int, default=3)
    parser.add_argument(
        "--output-json",
        type=str,
        default="/home/chen/RLPFN/artifacts/critic_rollout_seed_stability_probe.json",
    )
    args = parser.parse_args()

    report = run_critic_rollout_seed_stability_probe(
        checkpoint_path=args.checkpoint_path,
        device=args.device,
        frozen_h_seed=args.frozen_h_seed,
        train_env_seed=args.train_env_seed,
        rollout_seed=args.rollout_seed,
        single_eval_pos=args.single_eval_pos,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        learning_rate=args.learning_rate,
        target_kl=args.target_kl,
        ppo_reset_env_state_at_sep=bool(args.ppo_reset_env_state_at_sep),
        ppo_separate_value_backbone=bool(args.ppo_separate_value_backbone),
        deterministic_actor_sampling=bool(args.deterministic_actor_sampling),
        num_rollouts=int(args.num_rollouts),
    )
    payload = json.dumps(report, sort_keys=True)
    out_path = Path(args.output_json).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
