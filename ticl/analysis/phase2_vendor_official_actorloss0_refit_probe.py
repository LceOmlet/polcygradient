import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ticl.analysis.critic_free_single_env_audit import (
    _build_audit_env_cfg,
    _make_deterministic_batch_plan,
    _measure_current_policy_rollout_critic_quality,
    _measure_rollout_critic_quality,
    _prepare_audit_fixed_env_contract,
)
from ticl.analysis.phase2_legacy_bar_longrun_probe import (
    _build_algo,
    _collect_rollout,
    _default_device,
    _resolve_model_and_config,
)
from ticl.analysis.critic_free_single_env_audit import _masked_corrcoef_np
from ticl.sb3_recurrent_ppo import _explained_variance_with_mask


def _refit_adapter_from_current_rollout(algo) -> None:
    obs_steps = torch.as_tensor(algo.rollout_buffer.observations, device=algo.device, dtype=torch.float32)
    episode_starts_steps = torch.as_tensor(
        algo.rollout_buffer.episode_starts,
        device=algo.device,
        dtype=torch.float32,
    )
    objective_masks_steps = torch.as_tensor(
        algo.rollout_buffer.objective_masks,
        device=algo.device,
        dtype=torch.float32,
    )
    if obs_steps.ndim == 2:
        obs_steps = obs_steps.unsqueeze(1)
    if episode_starts_steps.ndim == 1:
        episode_starts_steps = episode_starts_steps.unsqueeze(1)
    if objective_masks_steps.ndim == 1:
        objective_masks_steps = objective_masks_steps.unsqueeze(1)
    algo.policy.fit_value_path_adapter_from_rollout_steps(
        obs_steps,
        episode_starts_steps,
        objective_masks_steps,
    )


@torch.no_grad()
def _measure_current_policy_rollout_calibration(algo) -> dict:
    rollout_buffer = algo.rollout_buffer
    buffer_getter_flat = getattr(rollout_buffer, "get_gpu_flat", None)
    if not callable(buffer_getter_flat):
        raise RuntimeError("current-policy rollout calibration requires rollout_buffer.get_gpu_flat().")

    _make_deterministic_batch_plan(rollout_buffer)
    pred_chunks = []
    return_chunks = []
    mask_chunks = []
    for rollout_data in buffer_getter_flat(algo.batch_size):
        eval_outputs = algo.policy.evaluate_actions_with_hidden_flat(
            rollout_data.observations,
            rollout_data.actions,
            seq_lengths=rollout_data.seq_lengths,
            action_masks=rollout_data.action_masks,
        )
        pred_chunks.append(eval_outputs["values"].flatten().detach().cpu())
        return_chunks.append(rollout_data.returns.flatten().detach().cpu())
        mask_chunks.append((rollout_data.objective_masks.flatten() > 1e-8).detach().cpu())
    _make_deterministic_batch_plan(rollout_buffer)

    preds = torch.cat(pred_chunks, dim=0).numpy()
    targets = torch.cat(return_chunks, dim=0).numpy()
    mask = torch.cat(mask_chunks, dim=0).numpy().astype(bool, copy=False)
    if int(mask.sum()) <= 1:
        return {
            "objective_total": int(mask.sum()),
            "pred_mean": 0.0,
            "target_mean": 0.0,
            "pred_std": 0.0,
            "target_std": 0.0,
            "raw_corr": 0.0,
            "affine_slope": 0.0,
            "affine_intercept": 0.0,
            "affine_ev": float("nan"),
        }

    pred_m = np.asarray(preds[mask], dtype=np.float32)
    target_m = np.asarray(targets[mask], dtype=np.float32)
    pred_mean = float(pred_m.mean())
    target_mean = float(target_m.mean())
    pred_std = float(pred_m.std())
    target_std = float(target_m.std())
    centered_pred = pred_m - pred_mean
    centered_target = target_m - target_mean
    denom = float(np.dot(centered_pred, centered_pred))
    if denom <= 0.0:
        slope = 0.0
    else:
        slope = float(np.dot(centered_pred, centered_target) / denom)
    intercept = float(target_mean - slope * pred_mean)
    affine_pred = slope * pred_m + intercept
    full_affine_pred = np.asarray(preds, dtype=np.float32).copy()
    full_affine_pred[mask] = affine_pred
    return {
        "objective_total": int(mask.sum()),
        "pred_mean": pred_mean,
        "target_mean": target_mean,
        "pred_std": pred_std,
        "target_std": target_std,
        "raw_corr": _masked_corrcoef_np(preds, targets, mask),
        "affine_slope": float(slope),
        "affine_intercept": float(intercept),
        "affine_ev": float(
            _explained_variance_with_mask(
                full_affine_pred,
                np.asarray(targets, dtype=np.float32),
                mask=mask,
            )
        ),
    }


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
    outer_epochs: int = 2,
    strict_native_rollout: bool = False,
    value_path_adapter_impl: str = "frozen_zscore",
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
        value_path_adapter_impl=str(value_path_adapter_impl),
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
            pre_rollout = _measure_rollout_critic_quality(algo)
            algo.train()
            post_same_stale = _measure_current_policy_rollout_critic_quality(algo)
            post_same_stale_calibration = _measure_current_policy_rollout_calibration(algo)
            if str(value_path_adapter_impl).strip().lower() == "frozen_zscore":
                _refit_adapter_from_current_rollout(algo)
            post_same_refit = _measure_current_policy_rollout_critic_quality(algo)
            post_same_refit_calibration = _measure_current_policy_rollout_calibration(algo)
            rows.append(
                {
                    "outer_epoch": int(outer_idx + 1),
                    "pre_rollout": pre_rollout,
                    "post_same_rollout_stale": post_same_stale,
                    "post_same_rollout_stale_calibration": post_same_stale_calibration,
                    "post_same_rollout_refit": post_same_refit,
                    "post_same_rollout_refit_calibration": post_same_refit_calibration,
                }
            )
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
            "audit_entry": "phase2_vendor_official_actorloss0_refit_probe",
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
            "value_path_adapter_impl": str(value_path_adapter_impl),
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
    parser.add_argument("--outer-epochs", type=int, default=2)
    parser.add_argument("--strict-native-rollout", action="store_true", default=False)
    parser.add_argument("--value-path-adapter-impl", type=str, default="frozen_zscore")
    parser.add_argument(
        "--output-json",
        type=str,
        default="/home/chen/RLPFN/artifacts/phase2_vendor_official_actorloss0_refit_probe.json",
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
        value_path_adapter_impl=str(args.value_path_adapter_impl),
    )
    payload = json.dumps(report, sort_keys=True)
    output_path.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
