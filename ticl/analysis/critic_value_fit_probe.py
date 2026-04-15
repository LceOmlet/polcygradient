import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch

from ticl.analysis.critic_free_single_env_audit import _build_audit_env_cfg, _make_deterministic_batch_plan
from ticl.model_builder import load_model
from ticl.priors.environment_prior import EnvironmentPrior
from ticl.sb3_recurrent_ppo import (
    _explained_variance_with_mask,
    _recover_raw_from_value_space,
    build_recurrent_ppo,
)
from ticl.train import _freeze_env_h_list_for_replay


def _default_device() -> str:
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def _clone_h_repeated(h, count: int):
    return [copy.deepcopy(h) for _ in range(int(count))]


def _build_fixed_env_h(*, prior_cfg: dict, frozen_h_seed: int):
    prior_for_h = EnvironmentPrior(copy.deepcopy(prior_cfg))
    rng_state = np.random.get_state()
    np.random.seed(int(frozen_h_seed))
    frozen_h = _freeze_env_h_list_for_replay(
        prior_for_h,
        list(prior_for_h._sample_batch_hypers(1)),
    )[0]
    np.random.set_state(rng_state)
    return frozen_h


def _build_algo(
    *,
    checkpoint_path: str,
    device_obj: torch.device,
    frozen_h,
    train_env_seed: int,
    single_eval_pos: int,
    n_steps: int,
    batch_size: int,
    n_epochs: int,
    learning_rate: float,
    target_kl: float,
    ppo_reset_env_state_at_sep: bool,
    ppo_separate_value_backbone: bool,
):
    load_model.cache_clear()
    model, config = load_model(checkpoint_path, device=device_obj, verbose=False)
    num_features = int(config["prior"]["num_features"])
    env_cfg = _build_audit_env_cfg(config["prior"]["environment"])
    prior = EnvironmentPrior(copy.deepcopy(env_cfg))
    prior._sample_batch_hypers = lambda batch_n, _h=frozen_h: _clone_h_repeated(_h, int(batch_n))
    prior._sample_single_eval_pos = lambda n_samples_arg, rng_seed=None: int(single_eval_pos)

    algo, callback, vec_env = build_recurrent_ppo(
        model=model,
        env_prior=prior,
        device=str(device_obj),
        num_features=num_features,
        n_envs=1,
        n_steps=int(n_steps),
        learning_rate=float(learning_rate),
        batch_size=int(batch_size),
        n_epochs=int(n_epochs),
        gamma=float(config["optimizer"].get("ppo_gamma", 1.0)),
        gae_lambda=float(config["optimizer"].get("ppo_gae_lambda", 0.95)),
        clip_range=config["optimizer"].get("ppo_clip_range", 0.2),
        clip_range_vf=None,
        normalize_advantage=False,
        actor_gae_space="normalized",
        actor_baseline_mode="learned",
        actor_objective_mode="tokenwise",
        reset_env_state_at_sep=bool(ppo_reset_env_state_at_sep),
        separate_value_backbone=bool(ppo_separate_value_backbone),
        ent_coef=float(config["optimizer"].get("ppo_ent_coef", 0.0)),
        vf_coef=float(config["optimizer"].get("ppo_vf_coef", 0.5)),
        max_grad_norm=float(config["optimizer"].get("ppo_max_grad_norm", 0.5)),
        target_kl=float(target_kl),
        verbose=0,
    )
    vec_env._sample_seed_list = lambda: [int(train_env_seed)]
    _make_deterministic_batch_plan(algo.rollout_buffer)
    algo.ep_info_buffer = []
    algo.ep_success_buffer = []
    algo._last_obs = vec_env.reset()
    algo._last_episode_starts = np.ones((vec_env.num_envs,), dtype=bool)
    algo._last_lstm_states = algo.policy._dummy_states(vec_env.num_envs)
    callback.init_callback(algo)
    algo._update_current_progress_remaining(0, int(n_steps))
    return algo, callback, vec_env, prior, num_features


def _collect_fixed_rollout(algo, callback, vec_env):
    assert algo.collect_rollouts(vec_env, callback, algo.rollout_buffer, n_rollout_steps=algo.n_steps)
    return next(algo.rollout_buffer.get_gpu_flat(algo.batch_size))


def _masked_corrcoef(x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> float:
    mask = mask.bool()
    if int(mask.sum().item()) <= 1:
        return 0.0
    x = x[mask].float()
    y = y[mask].float()
    x = x - x.mean()
    y = y - y.mean()
    denom = torch.linalg.vector_norm(x) * torch.linalg.vector_norm(y)
    if float(denom.item()) <= 0.0:
        return 0.0
    return float((x @ y / denom).item())


@torch.no_grad()
def _measure_critic(algo, rollout_data) -> dict:
    eval_outputs = algo.policy.evaluate_actions_with_hidden_flat(
        rollout_data.observations,
        rollout_data.actions,
        seq_lengths=rollout_data.seq_lengths,
        action_masks=rollout_data.action_masks,
    )
    values = eval_outputs["values"].flatten()
    objective_mask = rollout_data.objective_masks > 1e-8
    raw_pred = _recover_raw_from_value_space(
        values.detach(),
        value_means=rollout_data.rollout_return_means.detach(),
        value_stds=rollout_data.rollout_return_stds.detach(),
    )
    raw_returns = _recover_raw_from_value_space(
        rollout_data.returns.detach(),
        value_means=rollout_data.rollout_return_means.detach(),
        value_stds=rollout_data.rollout_return_stds.detach(),
    )
    value_std = float(raw_pred[objective_mask].float().std().item()) if int(objective_mask.sum().item()) > 1 else 0.0
    return_std = float(raw_returns[objective_mask].float().std().item()) if int(objective_mask.sum().item()) > 1 else 0.0
    return {
        "objective_total": int(objective_mask.sum().item()),
        "raw_value_std": value_std,
        "raw_return_std": return_std,
        "raw_corr": _masked_corrcoef(raw_pred, raw_returns, objective_mask),
        "explained_variance_raw": float(
            _explained_variance_with_mask(
                raw_pred.detach().cpu().numpy(),
                raw_returns.detach().cpu().numpy(),
                mask=objective_mask.detach().cpu().numpy(),
            )
        ),
    }


def _critic_only_step(algo, rollout_data) -> float:
    objective_mask = rollout_data.objective_masks > 1e-8
    valid_total = objective_mask.to(device=rollout_data.returns.device, dtype=rollout_data.returns.dtype).sum().clamp_min(1.0)
    eval_outputs = algo.policy.evaluate_actions_with_hidden_flat(
        rollout_data.observations,
        rollout_data.actions,
        seq_lengths=rollout_data.seq_lengths,
        action_masks=rollout_data.action_masks,
    )
    value_logits = eval_outputs["value_logits"]
    value_bardist = algo.policy.get_value_bardist()
    value_errors = value_bardist(
        value_logits.to(dtype=torch.float32),
        rollout_data.returns.to(device=value_logits.device, dtype=torch.float32),
    )
    objective_f = objective_mask.to(device=value_errors.device, dtype=value_errors.dtype)
    value_num = (value_errors * objective_f).sum()
    value_loss = value_num / valid_total.to(device=value_num.device, dtype=value_num.dtype)
    algo.policy.optimizer.zero_grad(set_to_none=True)
    value_loss.backward()
    algo.policy.optimizer.step()
    return float(value_loss.detach().cpu().item())


def run_critic_value_fit_probe(
    *,
    checkpoint_path: str,
    device: str | None = None,
    frozen_h_seed: int = 12345,
    train_env_seed: int = 2020,
    single_eval_pos: int = 64,
    n_steps: int = 256,
    batch_size: int = 256,
    n_epochs: int = 1,
    learning_rate: float = 2e-4,
    target_kl: float = 0.03,
    ppo_reset_env_state_at_sep: bool = True,
    ppo_separate_value_backbone: bool = False,
    fit_steps: list[int] | None = None,
):
    fit_steps = sorted(set(int(v) for v in (fit_steps or [0, 10, 50, 200, 1000])))
    device_obj = torch.device(str(device or _default_device()))
    load_model.cache_clear()
    _, config = load_model(checkpoint_path, device=device_obj, verbose=False)
    env_cfg = _build_audit_env_cfg(config["prior"]["environment"])
    frozen_h = _build_fixed_env_h(prior_cfg=env_cfg, frozen_h_seed=int(frozen_h_seed))

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
    try:
        rollout_data = _collect_fixed_rollout(algo, callback, vec_env)
        report = {
            "audit_entry": "critic_value_fit_probe",
            "checkpoint_path": str(Path(checkpoint_path).expanduser().resolve()),
            "device": str(device_obj),
            "frozen_h_seed": int(frozen_h_seed),
            "train_env_seed": int(train_env_seed),
            "single_eval_pos": int(single_eval_pos),
            "n_steps": int(n_steps),
            "batch_size": int(batch_size),
            "n_epochs": int(n_epochs),
            "learning_rate": float(learning_rate),
            "target_kl": float(target_kl),
            "ppo_reset_env_state_at_sep": bool(ppo_reset_env_state_at_sep),
            "ppo_separate_value_backbone": bool(ppo_separate_value_backbone),
            "fit_steps": fit_steps,
            "measurements": {},
        }
        total_steps = 0
        for target_step in fit_steps:
            while total_steps < int(target_step):
                _critic_only_step(algo, rollout_data)
                total_steps += 1
            report["measurements"][str(target_step)] = _measure_critic(algo, rollout_data)
        return report
    finally:
        vec_env.close()
        del algo, callback, vec_env, prior
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_path", type=str)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--frozen-h-seed", type=int, default=12345)
    parser.add_argument("--train-env-seed", type=int, default=2020)
    parser.add_argument("--single-eval-pos", type=int, default=64)
    parser.add_argument("--n-steps", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--target-kl", type=float, default=0.03)
    parser.add_argument("--ppo-reset-env-state-at-sep", action="store_true")
    parser.add_argument("--ppo-separate-value-backbone", action="store_true")
    parser.add_argument("--fit-steps", type=int, nargs="*", default=[0, 10, 50, 200, 1000])
    parser.add_argument("--output-json", type=str, default="/home/chen/RLPFN/artifacts/critic_value_fit_probe.json")
    args = parser.parse_args()

    report = run_critic_value_fit_probe(
        checkpoint_path=args.checkpoint_path,
        device=args.device,
        frozen_h_seed=args.frozen_h_seed,
        train_env_seed=args.train_env_seed,
        single_eval_pos=args.single_eval_pos,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        learning_rate=args.learning_rate,
        target_kl=args.target_kl,
        ppo_reset_env_state_at_sep=bool(args.ppo_reset_env_state_at_sep),
        ppo_separate_value_backbone=bool(args.ppo_separate_value_backbone),
        fit_steps=args.fit_steps,
    )
    output_path = Path(args.output_json).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"[wrote] {output_path}")


if __name__ == "__main__":
    main()
