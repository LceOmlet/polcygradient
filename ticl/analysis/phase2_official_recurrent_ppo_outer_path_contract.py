#!/usr/bin/env python3
"""Contract checks for the non-core paths around the PPO update.

The core PPO math is covered by ``phase2_official_recurrent_ppo_core_parity``.
This probe covers the surrounding code that can still invalidate a clean PPO
comparison:

* RWKV policy forward/evaluate_actions paths and action masks.
* Pack -> rollout-buffer bridging.
* SB3-style observation/reward VecNorm plumbing.
* Custom objective masks that differ from the official valid-token mask.
* Value distribution/bar head loss and masking.

It is intentionally synthetic: no prior generator, no Gym env, and no training
tmux.  The goal is to make path-level bugs visible before interpreting any
long-run learning curve.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from gymnasium import spaces

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ticl.analysis.phase2_gym_prior_pack_isomorphism_sentinel import (  # noqa: E402
    _fill_existing_rollout_buffer_from_pack,
)
from ticl.analysis.phase2_gym_prior_pack_training_runner import (  # noqa: E402
    _apply_observation_normalization_if_enabled,
    _apply_observation_normalization_snapshot,
    _apply_reward_normalization_if_enabled,
    _normalize_reward_step_online,
)
from ticl.analysis.phase2_official_recurrent_ppo_core_parity import (  # noqa: E402
    _CaptureLogger,
    _CustomFakePolicy,
    _build_train_samples,
    _make_custom_fake_algo,
)
from ticl.models.tabpfn_bar_distribution import make_standardized_full_support_bar_distribution  # noqa: E402
from ticl.sb3_recurrent_ppo import (  # noqa: E402
    MaskedRecurrentPPO,
    MaskedRecurrentRolloutBuffer,
    OfficialRWKVRecurrentPPOPolicy,
    RNNStates,
)
from ticl.tests.test_sb3_recurrent_ppo import _FakeRWKVModel  # noqa: E402


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, np.ndarray):
        return [_json_safe(v) for v in value.tolist()]
    if torch.is_tensor(value):
        return _json_safe(value.detach().cpu().numpy())
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return str(value)


def _max_abs_tensor(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.detach().to(dtype=torch.float64) - b.detach().to(dtype=torch.float64)).abs().max().cpu().item())


def _all_finite_tensors(payload: dict[str, Any], keys: list[str]) -> bool:
    for key in keys:
        value = payload.get(key)
        if torch.is_tensor(value) and not bool(torch.isfinite(value).all().item()):
            return False
    return True


def _make_policy(*, device: torch.device, num_features: int, obs_slot_dim: int, action_dim: int, normalized_q_head: bool):
    fake_model = _FakeRWKVModel(
        num_features=int(num_features),
        x_obs_dim=int(obs_slot_dim) + int(action_dim),
        action_dim=int(action_dim),
        emsize=12,
        normalized_q_head=bool(normalized_q_head),
    ).to(device)
    policy = OfficialRWKVRecurrentPPOPolicy(
        observation_space=spaces.Box(low=-np.inf, high=np.inf, shape=(int(num_features),), dtype=np.float32),
        action_space=spaces.Box(low=-1.0, high=1.0, shape=(int(action_dim),), dtype=np.float32),
        lr_schedule=lambda _: 1e-3,
        rlpfn_model=fake_model,
        num_features=int(num_features),
        obs_slot_dim=int(obs_slot_dim),
        net_arch=[],
    ).to(device)
    return policy, fake_model


def _check_rwkv_policy_forward(seed: int, *, device: torch.device, tol: float) -> dict[str, Any]:
    torch.manual_seed(int(seed))
    num_features = 8
    obs_slot_dim = 4
    action_dim = 4
    policy, fake_model = _make_policy(
        device=device,
        num_features=num_features,
        obs_slot_dim=obs_slot_dim,
        action_dim=action_dim,
        normalized_q_head=True,
    )
    policy.eval()
    obs = torch.randn((6, num_features), device=device, dtype=torch.float32)
    actions = torch.randn((6, action_dim), device=device, dtype=torch.float32) * 0.3
    episode_starts = torch.tensor([1.0, 0.0, 0.0, 1.0, 0.0, 0.0], device=device, dtype=torch.float32)
    action_masks = torch.tensor(
        [
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 0.0],
            [1.0, 1.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0],
        ],
        device=device,
        dtype=torch.float32,
    )

    with torch.no_grad():
        seq_outputs = policy.evaluate_actions_with_hidden(
            obs,
            actions,
            policy._dummy_states(2),
            episode_starts,
            action_masks=action_masks,
        )
        flat_outputs = policy.evaluate_actions_with_hidden_flat(
            obs,
            actions,
            seq_lengths=np.asarray([3, 3], dtype=np.int64),
            action_masks=action_masks,
        )
        masked_actions = actions + torch.randn_like(actions) * (1.0 - action_masks) * 10.0
        mask_a = policy.evaluate_actions_with_hidden_flat(
            obs,
            actions,
            seq_lengths=np.asarray([3, 3], dtype=np.int64),
            action_masks=action_masks,
        )
        mask_b = policy.evaluate_actions_with_hidden_flat(
            obs,
            masked_actions,
            seq_lengths=np.asarray([3, 3], dtype=np.int64),
            action_masks=action_masks,
        )
        rollout_actions, rollout_values, rollout_log_prob, rollout_states = policy.forward(
            obs[:2],
            policy._dummy_states(2),
            episode_starts[:2],
        )

    seq_flat_diffs = {
        "values": _max_abs_tensor(seq_outputs["values"], flat_outputs["values"]),
        "log_prob": _max_abs_tensor(seq_outputs["log_prob"], flat_outputs["log_prob"]),
        "entropy": _max_abs_tensor(seq_outputs["entropy"], flat_outputs["entropy"]),
        "value_logits": _max_abs_tensor(seq_outputs["value_logits"], flat_outputs["value_logits"]),
    }
    inactive_action_diffs = {
        "log_prob": _max_abs_tensor(mask_a["log_prob"], mask_b["log_prob"]),
        "entropy": _max_abs_tensor(mask_a["entropy"], mask_b["entropy"]),
    }
    finite = bool(
        _all_finite_tensors(seq_outputs, ["values", "log_prob", "entropy", "value_logits"])
        and torch.isfinite(rollout_actions).all().item()
        and torch.isfinite(rollout_values).all().item()
        and torch.isfinite(rollout_log_prob).all().item()
    )
    return {
        "passed": bool(
            finite
            and max(seq_flat_diffs.values()) <= float(tol)
            and max(inactive_action_diffs.values()) <= float(tol)
            and tuple(rollout_actions.shape) == (2, action_dim)
            and tuple(rollout_values.shape) == (2, 1)
            and tuple(rollout_log_prob.shape) == (2,)
        ),
        "device": str(device),
        "sequence_calls": int(fake_model.rwkv_core.sequence_calls),
        "step_calls": int(fake_model.rwkv_core.step_calls),
        "seq_vs_flat_max_abs_diff": seq_flat_diffs,
        "inactive_action_dims_max_abs_diff": inactive_action_diffs,
        "rollout_shapes": {
            "actions": list(rollout_actions.shape),
            "values": list(rollout_values.shape),
            "log_prob": list(rollout_log_prob.shape),
            "state_pi_h": list(rollout_states.pi[0].shape),
        },
    }


def _check_value_bar_head(seed: int, *, device: torch.device, tol: float) -> dict[str, Any]:
    torch.manual_seed(int(seed))
    policy, _fake_model = _make_policy(
        device=device,
        num_features=8,
        obs_slot_dim=4,
        action_dim=3,
        normalized_q_head=True,
    )
    policy.eval()
    obs = torch.randn((6, 8), device=device, dtype=torch.float32)
    actions = torch.randn((6, 3), device=device, dtype=torch.float32) * 0.2
    episode_starts = torch.tensor([1.0, 0.0, 0.0, 1.0, 0.0, 0.0], device=device, dtype=torch.float32)
    with torch.no_grad():
        outputs = policy.evaluate_actions_with_hidden(
            obs,
            actions,
            policy._dummy_states(2),
            episode_starts,
            action_masks=torch.ones((6, 3), device=device, dtype=torch.float32),
        )
    bardist = policy.get_value_bardist()
    expected_values = bardist.mean(outputs["value_logits"].to(dtype=torch.float32)).unsqueeze(-1)
    value_mean_diff = _max_abs_tensor(outputs["values"], expected_values)
    targets = torch.linspace(-1.5, 1.5, 6, device=device, dtype=torch.float32)
    mask = torch.tensor([1, 0, 1, 0, 1, 0], device=device, dtype=torch.bool)
    loss_a = policy.compute_value_loss_from_logits(
        value_logits=outputs["value_logits"],
        targets=targets,
        objective_mask=mask,
    )
    targets_perturbed = targets.clone()
    targets_perturbed[~mask] += 10000.0
    loss_b = policy.compute_value_loss_from_logits(
        value_logits=outputs["value_logits"],
        targets=targets_perturbed,
        objective_mask=mask,
    )
    masked_loss_diff = float(abs(float(loss_a.detach().cpu().item()) - float(loss_b.detach().cpu().item())))
    return {
        "passed": bool(
            int(bardist.num_bars) == 100
            and tuple(outputs["value_logits"].shape) == (6, int(bardist.num_bars))
            and tuple(outputs["values"].shape) == (6, 1)
            and value_mean_diff <= float(tol)
            and masked_loss_diff <= float(tol)
            and bool(torch.isfinite(loss_a).all().item())
        ),
        "num_bars": int(bardist.num_bars),
        "value_mean_max_abs_diff": value_mean_diff,
        "masked_loss_perturb_diff": masked_loss_diff,
        "loss": float(loss_a.detach().cpu().item()),
    }


def _check_pack_rollout_bridge(seed: int, *, tol: float) -> dict[str, Any]:
    rng = np.random.default_rng(int(seed))
    n_steps = 6
    n_envs = 3
    num_features = 7
    action_dim = 4
    next_state_dim = 5
    single_eval_pos = 2
    pack = {
        "tokens": rng.normal(size=(n_steps, n_envs, num_features)).astype(np.float32),
        "actions": rng.normal(scale=0.3, size=(n_steps, n_envs, action_dim)).astype(np.float32),
        "action_masks": rng.integers(0, 2, size=(n_steps, n_envs, action_dim)).astype(np.float32),
        "rewards": rng.normal(loc=0.1, scale=1.2, size=(n_steps, n_envs)).astype(np.float32),
        "raw_rewards": rng.normal(loc=0.1, scale=1.2, size=(n_steps, n_envs)).astype(np.float32),
        "episode_starts": np.zeros((n_steps, n_envs), dtype=np.float32),
        "dones": np.zeros((n_steps, n_envs), dtype=bool),
        "terminated": np.zeros((n_steps, n_envs), dtype=bool),
        "truncated": np.zeros((n_steps, n_envs), dtype=bool),
        "next_state_targets": rng.normal(size=(n_steps, n_envs, next_state_dim)).astype(np.float32),
        "next_state_masks": rng.integers(0, 2, size=(n_steps, n_envs, next_state_dim)).astype(np.float32),
        "last_values": rng.normal(size=(n_envs,)).astype(np.float32),
    }
    pack["episode_starts"][0, :] = 1.0
    pack["episode_starts"][3, 1] = 1.0
    pack["dones"][-1, 1] = True
    values = rng.normal(size=(n_steps, n_envs)).astype(np.float32)
    log_probs = rng.normal(size=(n_steps, n_envs)).astype(np.float32)
    buffer = MaskedRecurrentRolloutBuffer(
        n_steps,
        spaces.Box(low=-np.inf, high=np.inf, shape=(num_features,), dtype=np.float32),
        spaces.Box(low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32),
        hidden_state_shape=(n_steps, 1, n_envs, 1),
        device="cpu",
        gae_lambda=0.95,
        gamma=0.99,
        n_envs=n_envs,
        next_state_dim=next_state_dim,
    )
    summary = _fill_existing_rollout_buffer_from_pack(
        buffer,
        pack,
        num_features=num_features,
        action_slot_dim=action_dim,
        next_state_target_dim=next_state_dim,
        single_eval_pos=single_eval_pos,
        values_by_step_env=values,
        log_probs_by_step_env=log_probs,
        last_values_by_env=pack["last_values"],
        compute_returns=True,
    )
    expected_objective = np.zeros((n_steps, n_envs), dtype=np.float32)
    expected_objective[single_eval_pos:] = 1.0
    checks = {
        "tokens": float(np.max(np.abs(buffer.observations - pack["tokens"]))),
        "actions": float(np.max(np.abs(buffer.actions - pack["actions"]))),
        "rewards": float(np.max(np.abs(buffer.rewards - pack["rewards"]))),
        "episode_starts": float(np.max(np.abs(buffer.episode_starts - pack["episode_starts"]))),
        "action_masks": float(np.max(np.abs(buffer.action_masks - pack["action_masks"]))),
        "next_states": float(np.max(np.abs(buffer.next_states - pack["next_state_targets"]))),
        "next_state_masks": float(np.max(np.abs(buffer.next_state_masks - pack["next_state_masks"]))),
        "objective_masks": float(np.max(np.abs(buffer.objective_masks - expected_objective))),
    }
    finite_returns = bool(np.isfinite(buffer.returns).all() and np.isfinite(buffer.advantages).all())
    return {
        "passed": bool(max(checks.values()) <= float(tol) and finite_returns),
        "max_abs_diffs": checks,
        "objective_total": int(buffer.objective_masks.sum()),
        "returns_finite": finite_returns,
        "summary": summary,
    }


def _check_vecnorm_online(seed: int, *, tol: float) -> dict[str, Any]:
    del seed
    raw_rewards = np.asarray(
        [
            [1.0, -2.0, 0.5],
            [0.25, 3.0, -1.0],
            [2.0, -0.5, 0.0],
            [-1.5, 1.0, 4.0],
        ],
        dtype=np.float32,
    )
    dones = np.asarray(
        [
            [False, False, False],
            [False, True, False],
            [False, False, True],
            [True, False, False],
        ],
        dtype=bool,
    )

    def _algo() -> SimpleNamespace:
        return SimpleNamespace(
            gamma=0.97,
            _rwkv_sb3_reward_normalization_enabled=True,
            _rwkv_sb3_reward_normalization_epsilon=1e-8,
            _rwkv_sb3_reward_normalization_clip=10.0,
            _rwkv_sb3_observation_normalization_enabled=True,
            _rwkv_sb3_observation_normalization_epsilon=1e-8,
            _rwkv_sb3_observation_normalization_clip=10.0,
        )

    online_algo = _algo()
    online_scaled = np.zeros_like(raw_rewards)
    for step_idx in range(raw_rewards.shape[0]):
        online_scaled[step_idx] = _normalize_reward_step_online(online_algo, raw_rewards[step_idx], dones[step_idx])

    pack_algo = _algo()
    pack = {"rewards": raw_rewards.copy(), "dones": dones.copy(), "raw_rewards": raw_rewards.copy()}
    normalized_pack, stats = _apply_reward_normalization_if_enabled(pack_algo, pack)
    offline_scaled = np.asarray(normalized_pack["rewards"], dtype=np.float32)
    reward_diff = float(np.max(np.abs(online_scaled - offline_scaled)))

    already_pack = {
        "rewards": online_scaled.copy(),
        "raw_rewards": raw_rewards.copy(),
        "dones": dones.copy(),
        "rewards_are_sb3_normalized": True,
    }
    already_out, already_stats = _apply_reward_normalization_if_enabled(online_algo, already_pack)
    already_diff = float(np.max(np.abs(np.asarray(already_out["rewards"], dtype=np.float32) - online_scaled)))

    obs_algo = _algo()
    obs = np.asarray([[1.0, 2.0, 99.0, 99.0], [3.0, 4.0, 5.0, 99.0]], dtype=np.float32)
    obs_norm, obs_stats = _apply_observation_normalization_if_enabled(
        obs_algo,
        obs,
        obs_dims=[2, 3],
        obs_slot_dim=4,
    )
    count_before = np.asarray(obs_algo._rwkv_sb3_observation_normalization_count, dtype=np.float64).copy()
    snapshot = _apply_observation_normalization_snapshot(
        obs_algo,
        obs + 10.0,
        obs_dims=[2, 3],
        obs_slot_dim=4,
    )
    count_after = np.asarray(obs_algo._rwkv_sb3_observation_normalization_count, dtype=np.float64).copy()
    snapshot_count_diff = float(np.max(np.abs(count_after - count_before)))
    inactive_zero = bool(np.allclose(obs_norm[:, 3], 0.0) and np.allclose(snapshot[:, 3], 0.0))
    return {
        "passed": bool(
            reward_diff <= float(tol)
            and already_diff <= float(tol)
            and snapshot_count_diff <= float(tol)
            and inactive_zero
            and bool(np.isfinite(online_scaled).all())
        ),
        "reward_online_vs_pack_max_abs_diff": reward_diff,
        "already_normalized_no_double_apply_max_abs_diff": already_diff,
        "observation_snapshot_count_diff": snapshot_count_diff,
        "inactive_obs_dim_zero": inactive_zero,
        "reward_stats": stats,
        "already_normalized_stats": already_stats,
        "observation_stats": obs_stats,
    }


def _run_custom_mask_train(sample, outputs, *, normalize_advantage: bool) -> dict[str, float]:
    cfg = {
        "n_epochs": 1,
        "clip_range": 0.2,
        "normalize_advantage": bool(normalize_advantage),
        "ent_coef": 0.03,
        "vf_coef": 0.5,
        "target_kl": None,
        "max_grad_norm": 0.5,
    }
    logger = _CaptureLogger()
    policy = _CustomFakePolicy(outputs["new_values"], outputs["new_log_prob"], outputs["entropy"])
    algo = _make_custom_fake_algo(sample, policy, cfg=cfg, logger=logger)
    MaskedRecurrentPPO.train(algo)  # type: ignore[arg-type]
    keys = [
        "train/policy_gradient_loss",
        "train/value_loss",
        "train/entropy_loss",
        "train/approx_kl",
        "train/clip_fraction",
        "train/loss",
    ]
    return {key: float(logger.rows[key]) for key in keys}


def _check_custom_objective_mask(seed: int, *, tol: float) -> dict[str, Any]:
    _official_sample, sample, outputs = _build_train_samples(int(seed), n=43, obs_dim=6, act_dim=3)
    base = _run_custom_mask_train(sample, outputs, normalize_advantage=True)
    mask = sample.objective_masks.to(dtype=torch.bool)
    perturbed_returns = sample.returns.clone()
    perturbed_advantages = sample.advantages.clone()
    perturbed_actor_advantages = sample.actor_advantages.clone()
    perturbed_old_log_prob = sample.old_log_prob.clone()
    perturbed_old_values = sample.old_values.clone()
    perturbed_returns[~mask] += 10000.0
    perturbed_advantages[~mask] -= 10000.0
    perturbed_actor_advantages[~mask] += 5000.0
    perturbed_old_log_prob[~mask] += 100.0
    perturbed_old_values[~mask] -= 10000.0
    perturbed_sample = sample._replace(
        returns=perturbed_returns,
        advantages=perturbed_advantages,
        actor_advantages=perturbed_actor_advantages,
        old_log_prob=perturbed_old_log_prob,
        old_values=perturbed_old_values,
    )
    perturbed = _run_custom_mask_train(perturbed_sample, outputs, normalize_advantage=True)
    diffs = {key: abs(base[key] - perturbed[key]) for key in base}
    return {
        "passed": bool(max(diffs.values()) <= float(tol)),
        "masked_out_row_perturb_diffs": diffs,
        "objective_total": int(mask.sum().item()),
        "masked_out_total": int((~mask).sum().item()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/phase2_official_recurrent_ppo_outer_path_contract_0501"))
    parser.add_argument("--seed", type=int, default=20260501)
    parser.add_argument("--tol", type=float, default=2e-5)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {
        "seed": int(args.seed),
        "tolerance": float(args.tol),
        "device": str(device),
        "scope": {
            "covered": [
                "RWKV policy forward/evaluate_actions and flat sequence replay",
                "inactive action dimension masking in log-prob/entropy",
                "pack to MaskedRecurrentRolloutBuffer field transport",
                "SB3-style online reward VecNorm and no-double-normalization guard",
                "SB3-style online observation VecNorm active-dim masking and snapshot no-update path",
                "custom objective mask exclusion in PPO losses",
                "value bar head mean/value-loss/objective-mask semantics",
            ],
            "not_covered": [
                "real Gym or prior environment dynamics",
                "full RWKV checkpoint numerical behavior",
                "multi-update optimizer drift",
                "GPU0-vs-GPU1 cross-device determinism",
            ],
        },
    }
    checks = {
        "rwkv_policy_forward": _check_rwkv_policy_forward(args.seed + 1, device=device, tol=float(args.tol)),
        "value_bar_head": _check_value_bar_head(args.seed + 2, device=device, tol=float(args.tol)),
        "pack_rollout_bridge": _check_pack_rollout_bridge(args.seed + 3, tol=float(args.tol)),
        "vecnorm_online": _check_vecnorm_online(args.seed + 4, tol=float(args.tol)),
        "custom_objective_mask": _check_custom_objective_mask(args.seed + 5, tol=float(args.tol)),
    }
    result["checks"] = checks
    result["passed"] = bool(all(bool(row.get("passed", False)) for row in checks.values()))
    out_path = args.output_dir / "outer_path_contract_report.json"
    out_path.write_text(json.dumps(_json_safe(result), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(_json_safe(result), indent=2, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
