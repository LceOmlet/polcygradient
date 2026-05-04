import argparse
import copy
import json
import math
from pathlib import Path
from typing import Any

import torch

from ticl.analysis.critic_free_single_env_audit import (
    _build_audit_env_cfg,
    _prepare_audit_fixed_env_contract,
    _seed_all,
)
from ticl.analysis.phase2_alpha_temporal_control_probe import (
    _compute_preterminal_state_next_batch,
    _make_generators_from_seeds,
    _resolve_probe_config,
)
from ticl.analysis.phase2_suffix_state_identity_probe import (
    _broadcast_like_batch,
    _to_float,
)
from ticl.config_utils import str2bool
from ticl.priors.environment_prior import EnvironmentPrior


def _parse_case(raw: str) -> tuple[str, dict[str, Any]]:
    name, sep, rest = str(raw).partition(":")
    overrides: dict[str, Any] = {}
    if sep and rest.strip():
        for item in rest.split(","):
            token = item.strip()
            if not token:
                continue
            key, eq, value = token.partition("=")
            if not eq:
                raise ValueError(f"Bad case override {token!r}; expected key=value")
            value_s = value.strip()
            low = value_s.lower()
            if low in {"true", "false"}:
                parsed: Any = low == "true"
            else:
                try:
                    parsed = int(value_s)
                except ValueError:
                    try:
                        parsed = float(value_s)
                    except ValueError:
                        parsed = value_s
            overrides[key.strip()] = parsed
    return name.strip(), overrides


def _stats(values: torch.Tensor) -> dict[str, float | int | None]:
    flat = values.detach().reshape(-1).to(device="cpu", dtype=torch.float64)
    finite = flat[torch.isfinite(flat)]
    if int(finite.numel()) == 0:
        return {
            "count": 0,
            "finite_fraction": 0.0,
            "mean": None,
            "q50": None,
            "q90": None,
            "q99": None,
            "max": None,
        }
    return {
        "count": int(flat.numel()),
        "finite_fraction": float(finite.numel() / max(flat.numel(), 1)),
        "mean": float(finite.mean().item()),
        "q50": float(torch.quantile(finite, 0.50).item()),
        "q90": float(torch.quantile(finite, 0.90).item()),
        "q99": float(torch.quantile(finite, 0.99).item()),
        "max": float(finite.max().item()),
    }


def _checkpoint_set(n_steps: int) -> set[int]:
    checkpoints = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, int(n_steps)}
    return {v for v in checkpoints if 1 <= int(v) <= int(n_steps)}


def run_case(
    *,
    case_name: str,
    overrides: dict[str, Any],
    device: torch.device,
    n_steps: int,
    rollout_count: int,
    build_seed: int,
    frozen_h_seed: int,
    train_env_seed: int,
    rollout_seed_start: int,
    policy: str,
    fast_random: bool,
) -> dict[str, Any]:
    config = _resolve_probe_config(
        checkpoint_path=None,
        from_scratch=True,
        from_scratch_model_type="rlpfn",
        device_obj=device,
        build_seed=int(build_seed),
        allow_config_only=True,
    )
    env_cfg = _build_audit_env_cfg(config["prior"]["environment"])
    fixed_bundle = _prepare_audit_fixed_env_contract(
        env_cfg=env_cfg,
        boundary_contract_mode="normal",
        ppo_reset_env_state_at_sep=True,
        frozen_h_seed=int(frozen_h_seed),
        frozen_h_seed_mode="current",
    )
    frozen_h = copy.deepcopy(fixed_bundle["frozen_h"])
    frozen_h.update(copy.deepcopy(overrides))
    frozen_h["terminal_reset_enabled"] = False
    frozen_h["terminal_reset_count_target"] = 0.0

    prior = EnvironmentPrior(copy.deepcopy(env_cfg))
    env = prior._sample_environment(copy.deepcopy(frozen_h), device=device, rng_seed=int(train_env_seed))
    batch_size = int(rollout_count)
    dtype = torch.float32
    state_dim = int(env["state_dim"])
    obs_dim = int(env["obs_dim"])
    action_dim = int(env["action_dim"])
    noise_dim = int(env["noise_dim"])
    rollout_seeds = [int(rollout_seed_start) + int(i) for i in range(batch_size)]
    rollout_generators = _make_generators_from_seeds(prior, rollout_seeds, device)
    fast_generator = torch.Generator(device=device)
    fast_generator.manual_seed(int(rollout_seed_start) + 9_973)
    fixed_initial_state_batch = prior._resolve_env_initial_state_batch(
        env,
        batch_size=batch_size,
        state_dim=state_dim,
        device=device,
        dtype=dtype,
    )
    carry_state_t, state_t = prior._init_exact_scm_carry_and_visible_state(fixed_initial_state_batch)
    init_action_std = _broadcast_like_batch(env["init_action_std"], batch_size=batch_size, device=device, dtype=dtype)
    if bool(fast_random):
        action_t = torch.randn(
            (batch_size, action_dim),
            device=device,
            dtype=dtype,
            generator=fast_generator,
        ) * init_action_std[:, None]
    else:
        action_t = prior._stack_randn_with_generators(
            rollout_generators, (batch_size, action_dim), device, dtype
        ) * init_action_std[:, None]

    checkpoints = _checkpoint_set(int(n_steps))
    rows: list[dict[str, Any]] = []
    finite_all = True
    max_state_abs_seen = 0.0
    max_state_rms_seen = 0.0
    max_obs_rms_seen = 0.0
    reward_sum = torch.zeros((batch_size,), device=device, dtype=dtype)
    state_noise_std = _to_float(env.get("state_noise_std", 0.0))

    for step_idx in range(1, int(n_steps) + 1):
        if policy == "unit_gaussian_iid":
            if bool(fast_random):
                action_next = torch.randn(
                    (batch_size, action_dim),
                    device=device,
                    dtype=dtype,
                    generator=fast_generator,
                )
            else:
                action_next = prior._stack_randn_with_generators(
                    rollout_generators, (batch_size, action_dim), device, dtype
                )
        elif policy == "zero":
            action_next = torch.zeros((batch_size, action_dim), device=device, dtype=dtype)
        elif policy == "prior_generator":
            zero_pad_dim = int(env["zero_pad_dim"])
            zero_pad_t = torch.zeros((batch_size, zero_pad_dim), device=device, dtype=dtype)
            obs_t = state_t[:, :obs_dim]
            env_in = prior._pack_env_input(
                carry_state_t,
                obs_t,
                action_t,
                torch.zeros((batch_size, noise_dim), device=device, dtype=dtype),
                zero_pad_t,
                reference_semantics_enabled=bool(prior._env_uses_reference_semantics(env)),
                state_input_scale=env.get("state_input_scale", 1.0),
            )
            action_next = env["policy_generator"](env_in, generators_for_noise=rollout_generators)
            action_next = prior._transform_reinforce_action(
                action_next,
                mode=env.get("reinforce_action_transform", "clip"),
                rms_eps=env.get("reinforce_action_rms_eps", 1e-6),
                clip_bound=env.get("reinforce_action_clip_bound", 5.0),
            )
        else:
            raise ValueError(f"Unknown policy {policy!r}")

        if noise_dim > 0:
            if bool(fast_random):
                noise_t = torch.randn(
                    (batch_size, noise_dim),
                    device=device,
                    dtype=dtype,
                    generator=fast_generator,
                )
            else:
                noise_t = prior._stack_randn_with_generators(
                    rollout_generators, (batch_size, noise_dim), device, dtype
                )
        else:
            noise_t = torch.zeros((batch_size, 0), device=device, dtype=dtype)
        state_noise_t = None
        if state_noise_std > 0.0:
            if bool(fast_random):
                state_noise_t = torch.randn(
                    (batch_size, state_dim),
                    device=device,
                    dtype=dtype,
                    generator=fast_generator,
                )
            else:
                state_noise_t = prior._stack_randn_with_generators(
                    rollout_generators, (batch_size, state_dim), device, dtype
                )

        state_next, reward_next_raw, _, _ = _compute_preterminal_state_next_batch(
            prior=prior,
            env=env,
            carry_state_t=carry_state_t,
            state_t=state_t,
            action_next=action_next,
            noise_t=noise_t,
            state_noise_t=state_noise_t,
            rollout_generators=rollout_generators,
            device=device,
        )
        reward_sum = reward_sum + reward_next_raw.reshape(batch_size)
        state_rms = torch.sqrt(state_next.square().mean(dim=1))
        obs_rms = torch.sqrt(state_next[:, :obs_dim].square().mean(dim=1))
        state_abs = state_next.abs().amax(dim=1)
        finite_step = bool(torch.isfinite(state_next).all().item()) and bool(torch.isfinite(reward_next_raw).all().item())
        finite_all = bool(finite_all and finite_step)
        max_state_abs_seen = max(max_state_abs_seen, float(state_abs.max().item()))
        max_state_rms_seen = max(max_state_rms_seen, float(state_rms.max().item()))
        max_obs_rms_seen = max(max_obs_rms_seen, float(obs_rms.max().item()))

        if step_idx in checkpoints:
            rows.append(
                {
                    "step": int(step_idx),
                    "finite": bool(finite_step),
                    "state_rms": _stats(state_rms),
                    "obs_rms": _stats(obs_rms),
                    "state_abs_max": _stats(state_abs),
                    "reward_raw": _stats(reward_next_raw.reshape(batch_size)),
                    "return_raw_sum": _stats(reward_sum),
                }
            )

        carry_state_t = prior._mix_exact_scm_carry_state(carry_state_t, state_next, env["alpha"])
        state_t = state_next
        action_t = action_next

    return {
        "case": case_name,
        "overrides": overrides,
        "forced_terminal_reset_enabled": False,
        "n_steps": int(n_steps),
        "rollout_count": int(rollout_count),
        "policy": policy,
        "fast_random": bool(fast_random),
        "finite_all": bool(finite_all),
        "max_state_abs_seen": float(max_state_abs_seen),
        "max_state_rms_seen": float(max_state_rms_seen),
        "max_obs_rms_seen": float(max_obs_rms_seen),
        "sampled_env": {
            "state_dim": int(env["state_dim"]),
            "obs_dim": int(env["obs_dim"]),
            "action_dim": int(env["action_dim"]),
            "noise_dim": int(env["noise_dim"]),
            "alpha": float(env["alpha"]),
            "init_state_std": float(env["init_state_std"]),
            "noise_std": (float(env["noise_std"]) if "noise_std" in env else None),
            "frozen_noise_std": (float(frozen_h["noise_std"]) if "noise_std" in frozen_h else None),
            "state_output_scale": float(env.get("state_output_scale", 1.0)),
            "state_full_rms_enabled": bool(env.get("state_full_rms_enabled", False)),
            "state_full_rms_target": float(env.get("state_full_rms_target", 1.0)),
            "effective_reward_scale": float(env.get("reward_scale", 1.0)),
            "requested_reward_scale": float(frozen_h.get("reward_scale", 1.0)),
            "reward_clip": float(env.get("reward_clip", float("inf"))),
            "state_clip": float(env.get("state_clip", float("inf"))),
            "state_highway_enabled": bool(env.get("state_highway_enabled", False)),
            "state_highway_lambda": float(env.get("state_highway_lambda", 0.0)),
            "terminal_reset_enabled": bool(env.get("terminal_reset_enabled", False)),
            "terminal_reset_count_target": float(env.get("terminal_reset_count_target", 0.0)),
        },
        "checkpoints": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="No-terminal long-rollout stability probe for exact SCM prior envs.")
    parser.add_argument("--cases", type=str, required=True, help="Semicolon-separated name:key=value cases.")
    parser.add_argument("--output-jsonl", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--n-steps", type=int, default=2048)
    parser.add_argument("--rollout-count", type=int, default=2048)
    parser.add_argument("--policy", type=str, default="unit_gaussian_iid", choices=["unit_gaussian_iid", "zero", "prior_generator"])
    parser.add_argument("--fast-random", type=str2bool, default=True)
    parser.add_argument("--paired-case-rollout-seeds", type=str2bool, default=True)
    parser.add_argument("--build-seed", type=int, default=4040)
    parser.add_argument("--frozen-h-seed", type=int, default=12345)
    parser.add_argument("--train-env-seed", type=int, default=2020)
    parser.add_argument("--rollout-seed-start", type=int, default=88000000)
    args = parser.parse_args()

    _seed_all(int(args.build_seed))
    output_path = Path(args.output_jsonl).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    cases = [_parse_case(piece) for piece in str(args.cases).split(";") if piece.strip()]
    with output_path.open("w", encoding="utf-8") as f:
        for case_idx, (case_name, overrides) in enumerate(cases):
            case_rollout_seed_start = int(args.rollout_seed_start)
            if not bool(args.paired_case_rollout_seeds):
                case_rollout_seed_start += case_idx * 10_000_000
            report = run_case(
                case_name=case_name,
                overrides=overrides,
                device=device,
                n_steps=int(args.n_steps),
                rollout_count=int(args.rollout_count),
                build_seed=int(args.build_seed),
                frozen_h_seed=int(args.frozen_h_seed),
                train_env_seed=int(args.train_env_seed),
                rollout_seed_start=int(case_rollout_seed_start),
                policy=str(args.policy),
                fast_random=bool(args.fast_random),
            )
            f.write(json.dumps(report, sort_keys=True) + "\n")
            f.flush()
            print(
                (
                    f"[long-rollout-stability] case={case_name} "
                    f"finite={report['finite_all']} "
                    f"max_abs={report['max_state_abs_seen']:.6g} "
                    f"max_rms={report['max_state_rms_seen']:.6g} "
                    f"max_obs_rms={report['max_obs_rms_seen']:.6g}"
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
