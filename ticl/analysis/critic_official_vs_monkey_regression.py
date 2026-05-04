import argparse
import copy
import json
import time
from pathlib import Path
from statistics import mean

import numpy as np
import torch

from ticl.analysis.critic_free_single_env_audit import _build_audit_env_cfg
from ticl.analysis.fixed_env_h import build_fixed_env_h
from ticl.model_builder import load_model
from ticl.priors.environment_prior import EnvironmentPrior
from ticl.sb3_recurrent_ppo import _resolve_actor_advantages, build_recurrent_ppo


def _default_device() -> str:
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def _to_cpu_tensor(x):
    if torch.is_tensor(x):
        return x.detach().cpu()
    return torch.as_tensor(x)


def _build_fixed_env_h(*, prior_cfg: dict, frozen_h_seed: int):
    return build_fixed_env_h(prior_cfg=prior_cfg, frozen_h_seed=int(frozen_h_seed))


def _clone_h_repeated(h, count: int):
    return [copy.deepcopy(h) for _ in range(int(count))]


def _sync_last_obs(algo, callback, vec_env, *, train_env_seed: int) -> None:
    vec_env._sample_seed_list = lambda: [int(train_env_seed)]
    algo._last_obs = vec_env.reset()
    algo._last_episode_starts = np.ones((vec_env.num_envs,), dtype=bool)
    algo._last_lstm_states = algo.policy._dummy_states(vec_env.num_envs)
    callback.init_callback(algo)
    algo._update_current_progress_remaining(0, int(algo.n_steps))


def _install_monkey_patch(algo, *, env_seed: int, rollout_seed: int) -> None:
    env_prior = algo._rwkv_env_prior
    original_rollout = env_prior._rollout_family_group_vectorized_with_policy

    def _wrapped_rollout(*args, **kwargs):
        batch_size = int(len(args[0])) if len(args) > 0 else int(len(kwargs["h_list"]))
        kwargs = dict(kwargs)
        kwargs["env_rng_seeds"] = [int(env_seed)] * batch_size
        kwargs["rollout_rng_seeds"] = [int(rollout_seed)] * batch_size
        return original_rollout(*args, **kwargs)

    env_prior._rollout_family_group_vectorized_with_policy = _wrapped_rollout

    original_make_step_fn = algo.policy.make_vectorized_rollout_step_fn

    def _wrapped_make_step_fn():
        fn = original_make_step_fn()
        fn._policy_actor_sample_fn = (  # type: ignore[attr-defined]
            lambda actor_outputs, noise: actor_outputs["action_mean"]
        )
        return fn

    algo.policy.make_vectorized_rollout_step_fn = _wrapped_make_step_fn


def _build_algo(
    *,
    checkpoint_path: str,
    device_obj: torch.device,
    frozen_h_seed: int,
    train_env_seed: int,
    train_rollout_seed: int,
    single_eval_pos: int,
    n_steps: int,
    batch_size: int,
    n_epochs: int,
    learning_rate: float,
    target_kl: float,
    strict_fixed_env_mode: bool,
    separate_value_backbone: bool,
):
    load_model.cache_clear()
    model, config = load_model(checkpoint_path, device=device_obj, verbose=False)
    prior_cfg = _build_audit_env_cfg(config["prior"]["environment"])
    frozen_h = _build_fixed_env_h(prior_cfg=prior_cfg, frozen_h_seed=int(frozen_h_seed))
    prior = EnvironmentPrior(copy.deepcopy(prior_cfg))
    prior._sample_batch_hypers = lambda batch_n, _h=frozen_h: _clone_h_repeated(_h, int(batch_n))
    prior._sample_single_eval_pos = lambda n_samples_arg, rng_seed=None: int(single_eval_pos)

    algo, callback, vec_env = build_recurrent_ppo(
        model=model,
        env_prior=prior,
        device=str(device_obj),
        num_features=int(config["prior"]["num_features"]),
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
        reset_env_state_at_sep=True,
        ent_coef=float(config["optimizer"].get("ppo_ent_coef", 0.0)),
        vf_coef=float(config["optimizer"].get("ppo_vf_coef", 0.5)),
        max_grad_norm=float(config["optimizer"].get("ppo_max_grad_norm", 0.5)),
        target_kl=float(target_kl),
        verbose=0,
        strict_fixed_env_mode=bool(strict_fixed_env_mode),
        env_rng_seeds=[int(train_env_seed)] if bool(strict_fixed_env_mode) else None,
        rollout_rng_seeds=[int(train_rollout_seed)] if bool(strict_fixed_env_mode) else None,
        deterministic_actor_sampling=bool(strict_fixed_env_mode),
        deterministic_batch_plan=True,
        separate_value_backbone=bool(separate_value_backbone),
    )
    algo.ep_info_buffer = []
    algo.ep_success_buffer = []
    return algo, callback, vec_env


def _param_diff_summary(policy_a, policy_b) -> dict:
    max_diff = 0.0
    differing = []
    for (name_a, param_a), (name_b, param_b) in zip(
        policy_a.named_parameters(),
        policy_b.named_parameters(),
    ):
        if name_a != name_b:
            raise RuntimeError(f"Policy param order mismatch: {name_a} vs {name_b}")
        diff = float((param_a.detach().cpu() - param_b.detach().cpu()).abs().max().item())
        max_diff = max(max_diff, diff)
        if diff > 0.0:
            differing.append(
                {
                    "name": str(name_a),
                    "max_abs_diff": diff,
                }
            )
    return {
        "policy_param_max_abs_diff": float(max_diff),
        "num_param_tensors_with_diff": int(len(differing)),
        "differing_tensors": differing,
    }


def _capture_first_step(algo, callback, vec_env, *, train_env_seed: int) -> dict:
    record = {}
    original_make = algo.policy.make_vectorized_rollout_step_fn

    def _wrapped_make():
        base = original_make()

        def _rec(obs_t, action_t, reward_t, reward_mask_t, cache, step_idx, env_info):
            out = base(obs_t, action_t, reward_t, reward_mask_t, cache, step_idx, env_info)
            if not record:
                actor_out, _ = out
                record["obs_t"] = _to_cpu_tensor(obs_t)
                record["action_t"] = _to_cpu_tensor(action_t)
                record["reward_t"] = _to_cpu_tensor(reward_t)
                record["reward_mask_t"] = _to_cpu_tensor(reward_mask_t)
                record["step_idx"] = int(step_idx)
                record["action_mean"] = _to_cpu_tensor(actor_out["action_mean"])
                record["values"] = _to_cpu_tensor(actor_out["values"])
            return out

        for name in (
            "_policy_actor_log_prob_fn",
            "_policy_actor_score_fn",
            "_policy_actor_decomp_stats_fn",
            "_fit_action_dim_fn",
            "_clear_buffers",
            "_model_ref",
            "_policy_actor_sample_fn",
        ):
            if hasattr(base, name):
                setattr(_rec, name, getattr(base, name))
        return _rec

    algo.policy.make_vectorized_rollout_step_fn = _wrapped_make
    _sync_last_obs(algo, callback, vec_env, train_env_seed=train_env_seed)
    assert algo.collect_rollouts(vec_env, callback, algo.rollout_buffer, algo.n_steps)
    return record


def _collect_and_measure(algo, callback, vec_env, *, train_env_seed: int) -> dict:
    _sync_last_obs(algo, callback, vec_env, train_env_seed=train_env_seed)
    t0 = time.perf_counter()
    assert algo.collect_rollouts(vec_env, callback, algo.rollout_buffer, algo.n_steps)
    if vec_env.num_envs > 0 and algo.device.type == "cuda":
        torch.cuda.synchronize(device=algo.device)
    collect_s = float(time.perf_counter() - t0)

    rollout_data = next(algo.rollout_buffer.get_gpu_flat(algo.batch_size))
    objective_mask = rollout_data.objective_masks > 1e-8
    valid_total = objective_mask.to(
        device=rollout_data.returns.device,
        dtype=rollout_data.returns.dtype,
    ).sum().clamp_min(1.0)
    advantages = _resolve_actor_advantages(rollout_data)
    eval_outputs = algo.policy.evaluate_actions_with_hidden_flat(
        rollout_data.observations,
        rollout_data.actions,
        seq_lengths=rollout_data.seq_lengths,
        action_masks=rollout_data.action_masks,
    )
    clip_range = algo.clip_range(algo._current_progress_remaining)
    ratio = torch.exp(eval_outputs["log_prob"] - rollout_data.old_log_prob)
    policy_loss_1 = advantages * ratio
    policy_loss_2 = advantages * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
    clipped_objective = torch.min(policy_loss_1, policy_loss_2)
    policy_num = (
        clipped_objective
        * objective_mask.to(device=clipped_objective.device, dtype=clipped_objective.dtype)
    ).sum()
    policy_loss = -(policy_num / valid_total.to(device=policy_num.device, dtype=policy_num.dtype))

    algo.policy.optimizer.zero_grad(set_to_none=True)
    policy_loss.backward()
    grad_parts = []
    for param in algo.policy.parameters():
        if param.requires_grad:
            grad = param.grad
            if grad is None:
                grad_parts.append(torch.zeros(param.numel(), device=param.device, dtype=param.dtype))
            else:
                grad_parts.append(grad.reshape(-1))
    grad_vector = torch.cat(grad_parts).to(dtype=torch.float32).detach().cpu()
    rb = algo.rollout_buffer
    return {
        "collect_s": float(collect_s),
        "policy_loss": float(policy_loss.detach().cpu().item()),
        "actions": _to_cpu_tensor(rb.actions),
        "returns": _to_cpu_tensor(rb.returns),
        "values": _to_cpu_tensor(rb.values),
        "advantages": _to_cpu_tensor(advantages),
        "log_probs": _to_cpu_tensor(rb.log_probs),
        "grad_vector": grad_vector,
    }


def _max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().max().item())


def run_official_vs_monkey_regression(
    *,
    checkpoint_path: str,
    device: str | None = None,
    frozen_h_seed: int = 12345,
    train_env_seed: int = 2020,
    train_rollout_seed: int = 4040,
    single_eval_pos: int = 64,
    n_steps: int = 256,
    batch_size: int = 256,
    n_epochs: int = 1,
    learning_rate: float = 2e-4,
    target_kl: float = 0.03,
    timing_repeats: int = 3,
    separate_value_backbone: bool = False,
) -> dict:
    device_obj = torch.device(str(device or _default_device()))

    def _build_pair():
        official_algo, official_callback, official_env = _build_algo(
            checkpoint_path=checkpoint_path,
            device_obj=device_obj,
            frozen_h_seed=int(frozen_h_seed),
            train_env_seed=int(train_env_seed),
            train_rollout_seed=int(train_rollout_seed),
            single_eval_pos=int(single_eval_pos),
            n_steps=int(n_steps),
            batch_size=int(batch_size),
            n_epochs=int(n_epochs),
            learning_rate=float(learning_rate),
            target_kl=float(target_kl),
            strict_fixed_env_mode=True,
            separate_value_backbone=bool(separate_value_backbone),
        )
        monkey_algo, monkey_callback, monkey_env = _build_algo(
            checkpoint_path=checkpoint_path,
            device_obj=device_obj,
            frozen_h_seed=int(frozen_h_seed),
            train_env_seed=int(train_env_seed),
            train_rollout_seed=int(train_rollout_seed),
            single_eval_pos=int(single_eval_pos),
            n_steps=int(n_steps),
            batch_size=int(batch_size),
            n_epochs=int(n_epochs),
            learning_rate=float(learning_rate),
            target_kl=float(target_kl),
            strict_fixed_env_mode=False,
            separate_value_backbone=bool(separate_value_backbone),
        )
        _install_monkey_patch(monkey_algo, env_seed=int(train_env_seed), rollout_seed=int(train_rollout_seed))
        return official_algo, official_callback, official_env, monkey_algo, monkey_callback, monkey_env

    official_algo, official_callback, official_env, monkey_algo, monkey_callback, monkey_env = _build_pair()
    unsynced_param_diff = _param_diff_summary(official_algo.policy, monkey_algo.policy)
    monkey_algo.policy.load_state_dict(copy.deepcopy(official_algo.policy.state_dict()))
    synced_param_diff = _param_diff_summary(official_algo.policy, monkey_algo.policy)

    def _build_toggle_algo():
        algo, callback, env = _build_algo(
            checkpoint_path=checkpoint_path,
            device_obj=device_obj,
            frozen_h_seed=int(frozen_h_seed),
            train_env_seed=int(train_env_seed),
            train_rollout_seed=int(train_rollout_seed),
            single_eval_pos=int(single_eval_pos),
            n_steps=int(n_steps),
            batch_size=int(batch_size),
            n_epochs=int(n_epochs),
            learning_rate=float(learning_rate),
            target_kl=float(target_kl),
            strict_fixed_env_mode=True,
            separate_value_backbone=bool(separate_value_backbone),
        )
        env_prior = algo._rwkv_env_prior
        original_rollout = env_prior._rollout_family_group_vectorized_with_policy
        original_make = algo.policy.make_vectorized_rollout_step_fn

        def _set_official_mode():
            algo._rwkv_strict_fixed_env_mode = True
            algo._rwkv_env_rng_seeds = [int(train_env_seed)]
            algo._rwkv_rollout_rng_seeds = [int(train_rollout_seed)]
            algo._rwkv_deterministic_actor_sampling = True
            env_prior._rollout_family_group_vectorized_with_policy = original_rollout
            algo.policy.make_vectorized_rollout_step_fn = original_make

        def _set_monkey_mode():
            algo._rwkv_strict_fixed_env_mode = False
            algo._rwkv_env_rng_seeds = None
            algo._rwkv_rollout_rng_seeds = None
            algo._rwkv_deterministic_actor_sampling = False

            def _wrapped_rollout(*args, **kwargs):
                batch_size_local = int(len(args[0])) if len(args) > 0 else int(len(kwargs["h_list"]))
                kwargs = dict(kwargs)
                kwargs["env_rng_seeds"] = [int(train_env_seed)] * batch_size_local
                kwargs["rollout_rng_seeds"] = [int(train_rollout_seed)] * batch_size_local
                return original_rollout(*args, **kwargs)

            def _wrapped_make_step_fn():
                fn = original_make()
                fn._policy_actor_sample_fn = (  # type: ignore[attr-defined]
                    lambda actor_outputs, noise: actor_outputs["action_mean"]
                )
                return fn

            env_prior._rollout_family_group_vectorized_with_policy = _wrapped_rollout
            algo.policy.make_vectorized_rollout_step_fn = _wrapped_make_step_fn

        return algo, callback, env, _set_official_mode, _set_monkey_mode

    first_algo, first_callback, first_env, first_set_official, first_set_monkey = _build_toggle_algo()
    first_set_official()
    first_step_official = _capture_first_step(
        first_algo,
        first_callback,
        first_env,
        train_env_seed=int(train_env_seed),
    )
    first_set_monkey()
    first_step_monkey = _capture_first_step(
        first_algo,
        first_callback,
        first_env,
        train_env_seed=int(train_env_seed),
    )
    first_step_compare = {
        "obs_t_equal": bool(torch.equal(first_step_official["obs_t"], first_step_monkey["obs_t"])),
        "action_t_equal": bool(torch.equal(first_step_official["action_t"], first_step_monkey["action_t"])),
        "reward_t_equal": bool(torch.equal(first_step_official["reward_t"], first_step_monkey["reward_t"])),
        "reward_mask_t_equal": bool(
            torch.equal(first_step_official["reward_mask_t"], first_step_monkey["reward_mask_t"])
        ),
        "action_mean_max_abs_diff": _max_abs_diff(
            first_step_official["action_mean"],
            first_step_monkey["action_mean"],
        ),
        "values_max_abs_diff": _max_abs_diff(
            first_step_official["values"],
            first_step_monkey["values"],
        ),
    }

    (
        regression_algo,
        regression_callback,
        regression_env,
        regression_set_official,
        regression_set_monkey,
    ) = _build_toggle_algo()
    regression_set_official()
    regression_official = _collect_and_measure(
        regression_algo,
        regression_callback,
        regression_env,
        train_env_seed=int(train_env_seed),
    )
    regression_set_monkey()
    regression_monkey = _collect_and_measure(
        regression_algo,
        regression_callback,
        regression_env,
        train_env_seed=int(train_env_seed),
    )
    grad_cosine = float(
        torch.nn.functional.cosine_similarity(
            regression_official["grad_vector"].unsqueeze(0),
            regression_monkey["grad_vector"].unsqueeze(0),
            dim=1,
            eps=1e-8,
        ).item()
    )
    regression_compare = {
        "policy_loss_abs_diff": abs(
            float(regression_official["policy_loss"]) - float(regression_monkey["policy_loss"])
        ),
        "grad_cosine": float(grad_cosine),
        "grad_l2_delta": float(
            torch.linalg.vector_norm(
                regression_official["grad_vector"] - regression_monkey["grad_vector"]
            ).item()
        ),
        "actions_max_abs_diff": _max_abs_diff(
            regression_official["actions"],
            regression_monkey["actions"],
        ),
        "returns_max_abs_diff": _max_abs_diff(
            regression_official["returns"],
            regression_monkey["returns"],
        ),
        "values_max_abs_diff": _max_abs_diff(
            regression_official["values"],
            regression_monkey["values"],
        ),
        "advantages_max_abs_diff": _max_abs_diff(
            regression_official["advantages"],
            regression_monkey["advantages"],
        ),
        "log_probs_max_abs_diff": _max_abs_diff(
            regression_official["log_probs"],
            regression_monkey["log_probs"],
        ),
        "official_collect_s_single": float(regression_official["collect_s"]),
        "monkey_collect_s_single": float(regression_monkey["collect_s"]),
    }

    (
        timing_algo,
        timing_callback,
        timing_env,
        timing_set_official,
        timing_set_monkey,
    ) = _build_toggle_algo()
    timing_set_official()
    _ = _collect_and_measure(
        timing_algo,
        timing_callback,
        timing_env,
        train_env_seed=int(train_env_seed),
    )
    timing_set_monkey()
    _ = _collect_and_measure(
        timing_algo,
        timing_callback,
        timing_env,
        train_env_seed=int(train_env_seed),
    )
    official_timing = []
    monkey_timing = []
    for _ in range(int(timing_repeats)):
        timing_set_official()
        official_timing.append(
            _collect_and_measure(
                timing_algo,
                timing_callback,
                timing_env,
                train_env_seed=int(train_env_seed),
            )["collect_s"]
        )
        timing_set_monkey()
        monkey_timing.append(
            _collect_and_measure(
                timing_algo,
                timing_callback,
                timing_env,
                train_env_seed=int(train_env_seed),
            )["collect_s"]
        )
    official_mean = float(mean(official_timing))
    monkey_mean = float(mean(monkey_timing))
    timing_compare = {
        "official_times_s": [float(v) for v in official_timing],
        "monkey_times_s": [float(v) for v in monkey_timing],
        "official_mean_s": float(official_mean),
        "monkey_mean_s": float(monkey_mean),
        "official_over_monkey_ratio": float(official_mean / monkey_mean) if monkey_mean > 0 else None,
    }

    return {
        "audit_entry": "critic_official_vs_monkey_regression",
        "checkpoint_path": str(Path(checkpoint_path).expanduser().resolve()),
        "device": str(device_obj),
        "frozen_h_seed": int(frozen_h_seed),
        "train_env_seed": int(train_env_seed),
        "train_rollout_seed": int(train_rollout_seed),
        "single_eval_pos": int(single_eval_pos),
        "n_steps": int(n_steps),
        "batch_size": int(batch_size),
        "n_epochs": int(n_epochs),
        "learning_rate": float(learning_rate),
        "target_kl": float(target_kl),
        "strict_fixed_env_mode": True,
        "deterministic_actor_sampling": True,
        "deterministic_batch_plan": True,
        "ppo_separate_value_backbone": bool(separate_value_backbone),
        "unsynced_param_diff": unsynced_param_diff,
        "synced_param_diff": synced_param_diff,
        "first_step_compare_after_sync": first_step_compare,
        "regression_compare_after_sync": regression_compare,
        "timing_compare_after_sync": timing_compare,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_path", type=str)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--frozen-h-seed", type=int, default=12345)
    parser.add_argument("--train-env-seed", type=int, default=2020)
    parser.add_argument("--train-rollout-seed", type=int, default=4040)
    parser.add_argument("--single-eval-pos", type=int, default=64)
    parser.add_argument("--n-steps", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--target-kl", type=float, default=0.03)
    parser.add_argument("--timing-repeats", type=int, default=3)
    parser.add_argument("--ppo-separate-value-backbone", action="store_true")
    parser.add_argument("--output-json", type=str, default=None)
    args = parser.parse_args()

    report = run_official_vs_monkey_regression(
        checkpoint_path=args.checkpoint_path,
        device=args.device,
        frozen_h_seed=int(args.frozen_h_seed),
        train_env_seed=int(args.train_env_seed),
        train_rollout_seed=int(args.train_rollout_seed),
        single_eval_pos=int(args.single_eval_pos),
        n_steps=int(args.n_steps),
        batch_size=int(args.batch_size),
        n_epochs=int(args.n_epochs),
        learning_rate=float(args.learning_rate),
        target_kl=float(args.target_kl),
        timing_repeats=int(args.timing_repeats),
        separate_value_backbone=bool(args.ppo_separate_value_backbone),
    )
    output_json = args.output_json
    if output_json is not None:
        output_path = Path(output_json).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
