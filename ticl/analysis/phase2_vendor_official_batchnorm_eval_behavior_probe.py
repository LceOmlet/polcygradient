import argparse
import json
from contextlib import contextmanager
from pathlib import Path

import torch
import torch.nn.functional as F

from ticl.analysis.critic_free_single_env_audit import (
    _measure_current_policy_rollout_critic_quality,
    _measure_rollout_critic_quality,
)
from ticl.analysis.phase2_legacy_bar_longrun_probe import (
    _build_algo,
    _build_audit_env_cfg,
    _collect_rollout,
    _default_device,
    _prepare_audit_fixed_env_contract,
    _resolve_model_and_config,
)
from ticl.sb3_recurrent_ppo import _ValuePathBatchNormAdapter


@contextmanager
def _force_batch_stats_eval(algo):
    adapter = getattr(algo.policy, "value_path_adapter", None)
    if not isinstance(adapter, _ValuePathBatchNormAdapter):
        yield False
        return

    original_forward = adapter.forward

    def _forward(latent_vf: torch.Tensor) -> torch.Tensor:
        if latent_vf.ndim != 2:
            raise ValueError(
                "_ValuePathBatchNormAdapter expects a rank-2 latent tensor, "
                f"got shape {tuple(latent_vf.shape)}"
            )
        if int(latent_vf.shape[0]) <= 1:
            return original_forward(latent_vf)
        running_mean = adapter.norm.running_mean.detach().clone()
        running_var = adapter.norm.running_var.detach().clone()
        return F.batch_norm(
            latent_vf,
            running_mean,
            running_var,
            adapter.norm.weight,
            adapter.norm.bias,
            True,
            adapter.norm.momentum,
            adapter.norm.eps,
        )

    adapter.forward = _forward
    try:
        yield True
    finally:
        adapter.forward = original_forward


@torch.no_grad()
def _measure_current_rollout_quality_with_mode(algo, *, force_batch_stats_eval: bool) -> dict:
    algo.policy.set_training_mode(False)
    if force_batch_stats_eval:
        with _force_batch_stats_eval(algo):
            return _measure_current_policy_rollout_critic_quality(algo)
    return _measure_current_policy_rollout_critic_quality(algo)


def run_probe(
    *,
    checkpoint_path: str | None = None,
    from_scratch: bool = True,
    from_scratch_model_type: str = "rlpfn",
    device: str | None = None,
    build_seed: int = 4040,
    frozen_h_seed: int = 12345,
    train_env_seed: int = 2020,
    train_rollout_seed: int = 4040,
    single_eval_pos: int = 64,
    n_steps: int = 256,
    batch_size: int = 256,
    n_epochs: int = 1,
    learning_rate: float = 2e-4,
    target_kl: float = 0.03,
    outer_epochs: int = 8,
    strict_native_rollout: bool = False,
) -> dict:
    device_obj = torch.device(str(device or _default_device()))
    _, config = _resolve_model_and_config(
        checkpoint_path=checkpoint_path,
        from_scratch=bool(from_scratch),
        from_scratch_model_type=str(from_scratch_model_type),
        device_obj=device_obj,
        build_seed=int(build_seed),
    )
    env_cfg = _build_audit_env_cfg(config["prior"]["environment"])
    fixed_bundle = _prepare_audit_fixed_env_contract(
        env_cfg=env_cfg,
        boundary_contract_mode="normal",
        ppo_reset_env_state_at_sep=True,
        frozen_h_seed=int(frozen_h_seed),
        frozen_h_seed_mode="current",
    )
    algo, callback, vec_env = _build_algo(
        checkpoint_path=checkpoint_path,
        device_obj=device_obj,
        frozen_h=fixed_bundle["frozen_h"],
        env_cfg=env_cfg,
        num_features=int(config["prior"]["num_features"]),
        train_env_seed=int(train_env_seed),
        train_rollout_seed=int(train_rollout_seed),
        single_eval_pos=int(single_eval_pos),
        n_steps=int(n_steps),
        batch_size=int(batch_size),
        n_epochs=int(n_epochs),
        learning_rate=float(learning_rate),
        target_kl=float(target_kl),
        ppo_reset_env_state_at_sep=True,
        actor_baseline_mode="learned",
        build_seed=int(build_seed),
        strict_native_rollout=bool(strict_native_rollout),
        value_head_impl="vendor_official",
        value_path_adapter_impl="batchnorm",
        from_scratch=bool(from_scratch),
        from_scratch_model_type=str(from_scratch_model_type),
        vf_coef_override=1.0,
        policy_loss_coef_override=0.0,
    )
    rows = []
    try:
        total_timesteps = int(n_steps) * int(outer_epochs)
        _collect_rollout(algo, callback, vec_env, progress_timestep=0, total_timesteps=total_timesteps)
        for outer_idx in range(int(outer_epochs)):
            pre_rollout_buffer = _measure_rollout_critic_quality(algo)
            pre_rollout_default_eval = _measure_current_rollout_quality_with_mode(
                algo,
                force_batch_stats_eval=False,
            )
            pre_rollout_force_batch_stats_eval = _measure_current_rollout_quality_with_mode(
                algo,
                force_batch_stats_eval=True,
            )
            rows.append(
                {
                    "outer_epoch": int(outer_idx + 1),
                    "pre_rollout_buffer": pre_rollout_buffer,
                    "pre_rollout_default_eval": pre_rollout_default_eval,
                    "pre_rollout_force_batch_stats_eval": pre_rollout_force_batch_stats_eval,
                }
            )
            algo.train()
            if outer_idx + 1 < int(outer_epochs):
                progress_next = min(int((outer_idx + 1) * n_steps), total_timesteps)
                _collect_rollout(
                    algo,
                    callback,
                    vec_env,
                    progress_timestep=progress_next,
                    total_timesteps=total_timesteps,
                )
        return {
            "audit_entry": "phase2_vendor_official_batchnorm_eval_behavior_probe",
            "init_mode": "from_scratch" if bool(from_scratch) else "checkpoint",
            "checkpoint_path": None if checkpoint_path is None else str(Path(checkpoint_path).expanduser().resolve()),
            "from_scratch_model_type": str(from_scratch_model_type),
            "device": str(device_obj),
            "build_seed": int(build_seed),
            "frozen_h_seed": int(frozen_h_seed),
            "train_env_seed": int(train_env_seed),
            "train_rollout_seed": int(train_rollout_seed),
            "single_eval_pos": int(single_eval_pos),
            "n_steps": int(n_steps),
            "batch_size": int(batch_size),
            "n_epochs": int(n_epochs),
            "learning_rate": float(learning_rate),
            "target_kl": float(target_kl),
            "outer_epochs": int(outer_epochs),
            "strict_native_rollout": bool(strict_native_rollout),
            "value_head_impl": "vendor_official",
            "value_path_adapter_impl": "batchnorm",
            "vf_coef_override": 1.0,
            "policy_loss_coef_override": 0.0,
            "history": rows,
        }
    finally:
        vec_env.close()
        del algo, callback, vec_env
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_path", nargs="?", default=None)
    parser.add_argument("--from-scratch", action="store_true", default=False)
    parser.add_argument("--from-scratch-model-type", type=str, default="rlpfn")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--build-seed", type=int, default=4040)
    parser.add_argument("--frozen-h-seed", type=int, default=12345)
    parser.add_argument("--train-env-seed", type=int, default=2020)
    parser.add_argument("--train-rollout-seed", type=int, default=4040)
    parser.add_argument("--single-eval-pos", type=int, default=64)
    parser.add_argument("--n-steps", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--target-kl", type=float, default=0.03)
    parser.add_argument("--outer-epochs", type=int, default=8)
    parser.add_argument("--strict-native-rollout", action="store_true", default=False)
    parser.add_argument(
        "--output-json",
        type=str,
        default="/home/chen/RLPFN/artifacts/phase2_vendor_official_batchnorm_eval_behavior_probe.json",
    )
    args = parser.parse_args()

    output_path = Path(args.output_json).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report = run_probe(
        checkpoint_path=args.checkpoint_path,
        from_scratch=bool(args.from_scratch),
        from_scratch_model_type=str(args.from_scratch_model_type),
        device=args.device,
        build_seed=int(args.build_seed),
        frozen_h_seed=int(args.frozen_h_seed),
        train_env_seed=int(args.train_env_seed),
        train_rollout_seed=int(args.train_rollout_seed),
        single_eval_pos=int(args.single_eval_pos),
        n_steps=int(args.n_steps),
        batch_size=int(args.batch_size),
        n_epochs=int(args.n_epochs),
        learning_rate=float(args.learning_rate),
        target_kl=float(args.target_kl),
        outer_epochs=int(args.outer_epochs),
        strict_native_rollout=bool(args.strict_native_rollout),
    )
    payload = json.dumps(report, sort_keys=True)
    output_path.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
