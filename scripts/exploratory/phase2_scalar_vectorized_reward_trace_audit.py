#!/usr/bin/env python
"""Scalar vs family-vectorized reward/action tensor trace audit.

This is a read-only computation-chain audit.  It compares the same frozen h and
env seed through the scalar environment construction path and the maintained
family-vectorized construction path, then traces:

    action_raw -> action_env -> transition reward_unit -> reward_raw
    -> reward_env/aux -> dropout-imputed reward -> visible state_next

It does not change the prior distribution and does not select or repair h.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from ticl.model_configs import get_model_default_config  # noqa: E402
from ticl.priors.environment_prior import EnvironmentPrior  # noqa: E402


ART = Path("/home/chen/RLPFN/artifacts")
DEFAULT_SELECTED_CSV = (
    ART
    / "phase2_profile_v2p_dimcoupled_reward_balance_probe_0516"
    / "selected_prior_envs.csv"
)
DEFAULT_OUTPUT_DIR = ART / "phase2_profile_v2p_scalar_vectorized_reward_trace_audit_0516"


def _jsonable(value: Any) -> Any:
    if torch.is_tensor(value):
        if value.numel() == 1:
            return _jsonable(value.detach().cpu().item())
        return [_jsonable(v) for v in value.detach().cpu().reshape(-1).tolist()]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    return value


def _load_h_rows(selected_csv: Path, rule: str, n_envs: int) -> list[tuple[dict[str, Any], int, str]]:
    rows: list[tuple[dict[str, Any], int, str]] = []
    with selected_csv.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if str(row.get("rule")) != str(rule):
                continue
            h_path = Path(str(row["full_frozen_h_json"])).expanduser()
            with h_path.open("r", encoding="utf-8") as h_file:
                h = json.load(h_file)
            if not isinstance(h, dict):
                raise ValueError(f"Expected dict h in {h_path}")
            seed = int(float(row.get("env_seed", row.get("seed", 0)) or 0))
            rows.append((h, seed, str(h_path)))
            if len(rows) >= int(n_envs):
                break
    if len(rows) < int(n_envs):
        raise ValueError(f"Need {n_envs} rows for rule={rule!r}, got {len(rows)}")
    return rows


def _as_float_tensor(value: Any, *, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    return torch.as_tensor(value, device=device, dtype=dtype)


def _scalar_trace(
    prior: EnvironmentPrior,
    env: dict[str, Any],
    *,
    device: torch.device,
    transition_noise_seed: int,
) -> dict[str, torch.Tensor]:
    state_dim = int(env["state_dim"])
    obs_dim = int(env["obs_dim"])
    action_dim = int(env["action_dim"])
    noise_dim = int(env["noise_dim"])
    state = torch.linspace(-0.55, 0.55, steps=max(1, state_dim), device=device, dtype=torch.float32)[:state_dim]
    obs = state[:obs_dim]
    noise = torch.linspace(-0.35, 0.35, steps=max(1, noise_dim), device=device, dtype=torch.float32)[:noise_dim]
    action_raw = torch.linspace(-1.5, 1.5, steps=max(1, action_dim), device=device, dtype=torch.float32)[:action_dim]
    action_env = prior._transform_reinforce_action(
        action_raw,
        mode=env.get("reinforce_action_transform", "none"),
        rms_eps=env.get("reinforce_action_rms_eps", 1e-6),
        clip_bound=env.get("reinforce_action_clip_bound", 5.0),
    )
    env_in = torch.zeros((1, int(env["env_input_dim"])), device=device, dtype=torch.float32)
    env_in[:, :state_dim] = prior._scale_state_env_input(
        state.unsqueeze(0),
        env.get("state_input_scale", 1.0),
    )
    if env.get("env_obs_start", None) is not None and obs_dim > 0:
        env_in[:, int(env["env_obs_start"]) : int(env["env_obs_start"]) + obs_dim] = obs.unsqueeze(0)
    env_in[:, int(env["env_action_start"]) : int(env["env_action_start"]) + action_dim] = action_env.unsqueeze(0)
    env_in[:, int(env["env_noise_start"]) : int(env["env_noise_start"]) + noise_dim] = noise.unsqueeze(0)
    transition_noise_generator = torch.Generator(device=device)
    transition_noise_generator.manual_seed(int(transition_noise_seed))
    transition_out = prior._require_transition_generator(env)(env_in, generator=transition_noise_generator)
    x_next, reward_unit, terminal_signal, terminal_bonus_base = prior._unpack_transition_output(transition_out)
    x_next = x_next.reshape(1, -1)[:, :state_dim]
    reward_unit = reward_unit.reshape(1)
    reward_raw = _as_float_tensor(env["reward_scale"], device=device).reshape(1) * reward_unit
    aux = prior._exact_scm_aux_reward_terms(
        action_env,
        ctrl_weight=env.get("ctrl_reward_weight", 0.0),
        ctrl_enabled=env.get("ctrl_reward_enabled", False),
        survival_weight=env.get("survival_reward_weight", 0.0),
        survival_enabled=env.get("survival_reward_enabled", False),
    ).reshape(1)
    reward_env = prior._transform_exact_scm_base_reward(
        reward_raw,
        reward_clip=env["reward_clip"],
        mode=env.get("reinforce_reward_transform", "none"),
        rms_eps=env.get("reinforce_reward_rms_eps", 1e-6),
        tanh_c=env.get("reinforce_reward_tanh_c", 1.0),
        tanh_bound=env.get("reinforce_reward_tanh_bound", 2.0),
    ).reshape(1)
    reward_total = prior._compose_exact_scm_reward(
        reward_raw,
        aux_reward=aux,
        reward_clip=env["reward_clip"],
        mode=env.get("reinforce_reward_transform", "none"),
        rms_eps=env.get("reinforce_reward_rms_eps", 1e-6),
        tanh_c=env.get("reinforce_reward_tanh_c", 1.0),
        tanh_bound=env.get("reinforce_reward_tanh_bound", 2.0),
    ).reshape(1)
    drop_draw = torch.full((1,), 0.05, device=device, dtype=torch.float32)
    drop_mask = (
        bool(env.get("reward_dropout_enabled", True))
        and float(env.get("reward_dropout_ratio", 0.0)) > 0.0
        and bool(drop_draw.item() < float(env.get("reward_dropout_ratio", 0.0)))
    )
    reward_mask = torch.ones((1,), device=device, dtype=torch.float32)
    reward_after_dropout = reward_total.clone()
    reward_env_after_dropout = reward_env.clone()
    if drop_mask:
        reward_mask.zero_()
        if bool(env.get("reward_dropout_impute_zero", True)):
            reward_after_dropout.zero_()
            reward_env_after_dropout.zero_()
    state_next = prior._finalize_exact_scm_visible_state_next(
        generated_state_next=x_next.squeeze(0),
        visible_state_prev=state,
        state_noise_t=None,
        state_noise_std=env["state_noise_std"],
        reference_semantics_enabled=env.get("reference_semantics_enabled", False),
        state_clip=env["state_clip"],
        state_output_scale=env.get("state_output_scale", 1.0),
        state_highway_enabled=env.get("state_highway_enabled", False),
        state_highway_lambda=env.get("state_highway_lambda", 0.0),
        state_full_rms_enabled=env.get("state_full_rms_enabled", False),
        state_full_rms_target=env.get("state_full_rms_target", 1.0),
    ).reshape(1, -1)[:, :state_dim]
    return {
        "state": state.reshape(1, -1),
        "action_raw": action_raw.reshape(1, -1),
        "action_env": action_env.reshape(1, -1),
        "noise": noise.reshape(1, -1),
        "transition_input": env_in,
        "x_next_raw": x_next,
        "reward_unit": reward_unit,
        "reward_raw": reward_raw,
        "reward_env_pre_dropout": reward_env,
        "reward_aux": aux,
        "reward_total_pre_dropout": reward_total,
        "reward_mask_after_dropout": reward_mask,
        "reward_env_after_dropout": reward_env_after_dropout,
        "reward_total_after_dropout": reward_after_dropout,
        "state_next_visible": state_next,
        "terminal_signal": torch.zeros((1,), device=device) if terminal_signal is None else terminal_signal.reshape(1),
        "terminal_bonus_base": (
            torch.zeros((1,), device=device) if terminal_bonus_base is None else terminal_bonus_base.reshape(1)
        ),
    }


def _vector_trace(
    prior: EnvironmentPrior,
    env: dict[str, Any],
    *,
    device: torch.device,
    transition_noise_seed: int,
) -> dict[str, torch.Tensor]:
    state_dim = int(env["state_dim"])
    obs_dim = int(env["obs_dim"])
    action_dim = int(env["action_dim"])
    noise_dim = int(env["noise_dim"])
    state = torch.linspace(-0.55, 0.55, steps=max(1, state_dim), device=device, dtype=torch.float32)[:state_dim].reshape(1, -1)
    obs = state[:, :obs_dim]
    noise = torch.linspace(-0.35, 0.35, steps=max(1, noise_dim), device=device, dtype=torch.float32)[:noise_dim].reshape(1, -1)
    action_raw = torch.linspace(-1.5, 1.5, steps=max(1, action_dim), device=device, dtype=torch.float32)[:action_dim].reshape(1, -1)
    action_mask = (
        torch.arange(action_dim, device=device).unsqueeze(0)
        < env["action_dim_per_sample"].to(device=device, dtype=torch.long).unsqueeze(1)
    ).to(dtype=torch.float32)
    action_env = prior._transform_reinforce_action(
        action_raw,
        mode=env.get("reinforce_action_transform", "none"),
        rms_eps=env.get("reinforce_action_rms_eps", 1e-6),
        clip_bound=env.get("reinforce_action_clip_bound", 5.0),
        mask=action_mask,
    )
    env_total_dim = state_dim + int(env.get("env_obs_input_dim", obs_dim)) + action_dim + noise_dim + int(env["zero_pad_dim"])
    env_in = torch.zeros((1, env_total_dim), device=device, dtype=torch.float32)
    env_in[:, :state_dim] = prior._scale_state_env_input(state, env.get("state_input_scale", 1.0))
    obs_start = state_dim if int(env.get("env_obs_input_dim", obs_dim)) > 0 else None
    action_start = state_dim + int(env.get("env_obs_input_dim", obs_dim))
    noise_start = action_start + action_dim
    if obs_start is not None and obs_dim > 0:
        env_in[:, obs_start : obs_start + obs_dim] = obs
    env_in[:, action_start : action_start + action_dim] = action_env
    env_in[:, noise_start : noise_start + noise_dim] = noise
    transition_noise_generator = torch.Generator(device=device)
    transition_noise_generator.manual_seed(int(transition_noise_seed))
    transition_out = prior._require_transition_generator(env)(
        env_in,
        generators_for_noise=[transition_noise_generator],
    )
    x_next, reward_unit, terminal_signal, terminal_bonus_base = prior._unpack_transition_output(transition_out)
    x_next = x_next.reshape(1, -1)[:, :state_dim]
    reward_unit = reward_unit.reshape(1)
    reward_raw = env["reward_scale"].to(device=device, dtype=torch.float32).reshape(1) * reward_unit
    ctrl, survival = prior._exact_scm_aux_reward_components(
        action_env,
        action_mask=action_mask,
        ctrl_weight=env.get("ctrl_reward_weight", 0.0),
        ctrl_enabled=env.get("ctrl_reward_enabled", False),
        survival_weight=env.get("survival_reward_weight", 0.0),
        survival_enabled=env.get("survival_reward_enabled", False),
    )
    aux = (ctrl + survival).reshape(1)
    reward_env = prior._transform_exact_scm_base_reward(
        reward_raw,
        reward_clip=env["reward_clip"],
        mode=env.get("reinforce_reward_transform", "none"),
        rms_eps=env.get("reinforce_reward_rms_eps", 1e-6),
        tanh_c=env.get("reinforce_reward_tanh_c", 1.0),
        tanh_bound=env.get("reinforce_reward_tanh_bound", 2.0),
    ).reshape(1)
    reward_total = prior._compose_exact_scm_reward(
        reward_raw,
        aux_reward=aux,
        reward_clip=env["reward_clip"],
        mode=env.get("reinforce_reward_transform", "none"),
        rms_eps=env.get("reinforce_reward_rms_eps", 1e-6),
        tanh_c=env.get("reinforce_reward_tanh_c", 1.0),
        tanh_bound=env.get("reinforce_reward_tanh_bound", 2.0),
    ).reshape(1)
    drop_draw = torch.full((1,), 0.05, device=device, dtype=torch.float32)
    drop_mask = env["reward_dropout_enabled"].to(device=device, dtype=torch.bool) & (
        drop_draw < env["reward_dropout_ratio"].to(device=device, dtype=torch.float32)
    )
    reward_mask = torch.where(drop_mask, torch.zeros_like(reward_total), torch.ones_like(reward_total))
    impute = drop_mask & env["reward_dropout_impute_zero"].to(device=device, dtype=torch.bool)
    reward_after_dropout = torch.where(impute, torch.zeros_like(reward_total), reward_total)
    reward_env_after_dropout = torch.where(impute, torch.zeros_like(reward_env), reward_env)
    state_next = prior._finalize_exact_scm_visible_state_next(
        generated_state_next=x_next,
        visible_state_prev=state,
        state_noise_t=None,
        state_noise_std=env["state_noise_std"],
        reference_semantics_enabled=env.get("reference_semantics_enabled", False),
        state_clip=env["state_clip"],
        state_output_scale=env.get("state_output_scale", 1.0),
        state_highway_enabled=env.get("state_highway_enabled", False),
        state_highway_lambda=env.get("state_highway_lambda", 0.0),
        state_full_rms_enabled=env.get("state_full_rms_enabled", False),
        state_full_rms_target=env.get("state_full_rms_target", 1.0),
        state_mask=(torch.arange(state_dim, device=device).unsqueeze(0) < env["state_dim_per_sample"].unsqueeze(1)).to(dtype=torch.float32),
    ).reshape(1, -1)[:, :state_dim]
    return {
        "state": state,
        "action_raw": action_raw,
        "action_env": action_env,
        "noise": noise,
        "transition_input": env_in,
        "x_next_raw": x_next,
        "reward_unit": reward_unit,
        "reward_raw": reward_raw,
        "reward_env_pre_dropout": reward_env,
        "reward_aux": aux,
        "reward_total_pre_dropout": reward_total,
        "reward_mask_after_dropout": reward_mask,
        "reward_env_after_dropout": reward_env_after_dropout,
        "reward_total_after_dropout": reward_after_dropout,
        "state_next_visible": state_next,
        "terminal_signal": torch.zeros((1,), device=device) if terminal_signal is None else terminal_signal.reshape(1),
        "terminal_bonus_base": (
            torch.zeros((1,), device=device) if terminal_bonus_base is None else terminal_bonus_base.reshape(1)
        ),
    }


def _max_abs_delta(a: torch.Tensor, b: torch.Tensor) -> float:
    if tuple(a.shape) != tuple(b.shape):
        return float("inf")
    if a.numel() == 0:
        return 0.0
    return float((a.detach() - b.detach()).abs().max().cpu().item())


def _audit_one(
    *,
    prior: EnvironmentPrior,
    h: dict[str, Any],
    seed: int,
    h_path: str,
    env_idx: int,
    device: torch.device,
) -> dict[str, Any]:
    h_eff = copy.deepcopy(h)
    scalar_env = prior._sample_environment(h_eff, device=device, rng_seed=int(seed))
    vector_env = prior._sample_environment_family_coarse_batch(
        [copy.deepcopy(h_eff)],
        device=device,
        rng_seeds=[int(seed)],
        build_policy_generator=False,
        preserve_skipped_generator_rng=True,
    )
    transition_noise_seed = int(seed) ^ 0x5EED123
    scalar = _scalar_trace(
        prior,
        scalar_env,
        device=device,
        transition_noise_seed=transition_noise_seed,
    )
    vector = _vector_trace(
        prior,
        vector_env,
        device=device,
        transition_noise_seed=transition_noise_seed,
    )
    keys = sorted(set(scalar) & set(vector))
    deltas = {key: _max_abs_delta(scalar[key], vector[key]) for key in keys}
    hard_pass = all(float(v) <= 1e-5 for v in deltas.values())
    return {
        "env_idx": int(env_idx),
        "h_path": str(h_path),
        "env_seed": int(seed),
        "family": str(h_eff.get("family", "scm")),
        "reference_semantics_enabled": bool(prior._resolve_reference_semantics_enabled(h_eff)),
        "reward_transform": prior._resolve_reinforce_reward_transform(h_eff),
        "action_transform": prior._resolve_reinforce_action_transform(h_eff),
        "state_dim": int(scalar_env["state_dim"]),
        "obs_dim": int(scalar_env["obs_dim"]),
        "action_dim": int(scalar_env["action_dim"]),
        "noise_dim": int(scalar_env["noise_dim"]),
        "zero_pad_dim": int(scalar_env["zero_pad_dim"]),
        "transition_noise_seed": int(transition_noise_seed),
        "deltas": deltas,
        "hard_pass": bool(hard_pass),
        "trace_scalar": {
            key: scalar[key]
            for key in (
                "reward_unit",
                "reward_raw",
                "reward_env_pre_dropout",
                "reward_aux",
                "reward_total_pre_dropout",
                "reward_mask_after_dropout",
                "reward_total_after_dropout",
                "terminal_signal",
                "terminal_bonus_base",
            )
        },
        "trace_vectorized": {
            key: vector[key]
            for key in (
                "reward_unit",
                "reward_raw",
                "reward_env_pre_dropout",
                "reward_aux",
                "reward_total_pre_dropout",
                "reward_mask_after_dropout",
                "reward_total_after_dropout",
                "terminal_signal",
                "terminal_bonus_base",
            )
        },
    }


def _write_md(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Scalar/vectorized reward trace audit",
        "",
        f"- selected_csv: `{report['selected_csv']}`",
        f"- rule: `{report['rule']}`",
        f"- hard_pass: `{report['hard_pass']}`",
        f"- max_delta: `{report['max_delta']}`",
        "",
        "| env | pass | max delta | reward unit | reward raw | reward total | state next |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report["rows"]:
        deltas = row["deltas"]
        lines.append(
            "| {env_idx} | {hard_pass} | {max_delta:.6g} | {reward_unit:.6g} | {reward_raw:.6g} | {reward_total:.6g} | {state_next:.6g} |".format(
                env_idx=int(row["env_idx"]),
                hard_pass=row["hard_pass"],
                max_delta=max(float(v) for v in deltas.values()),
                reward_unit=float(deltas.get("reward_unit", 0.0)),
                reward_raw=float(deltas.get("reward_raw", 0.0)),
                reward_total=float(deltas.get("reward_total_after_dropout", 0.0)),
                state_next=float(deltas.get("state_next_visible", 0.0)),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selected-csv", type=Path, default=DEFAULT_SELECTED_CSV)
    parser.add_argument("--rule", type=str, default="gym_q90_obs_action_range")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--n-envs", type=int, default=8)
    args = parser.parse_args()

    device = torch.device(str(args.device))
    cfg = copy.deepcopy(get_model_default_config("rlpfn"))
    prior = EnvironmentPrior(copy.deepcopy(cfg["prior"]["environment"]))
    h_rows = _load_h_rows(args.selected_csv, args.rule, int(args.n_envs))
    rows = [
        _audit_one(
            prior=prior,
            h=h,
            seed=seed,
            h_path=h_path,
            env_idx=idx,
            device=device,
        )
        for idx, (h, seed, h_path) in enumerate(h_rows)
    ]
    max_delta = max((max(float(v) for v in row["deltas"].values()) for row in rows), default=0.0)
    report = {
        "analysis_entry": "phase2_scalar_vectorized_reward_trace_audit",
        "selected_csv": str(args.selected_csv),
        "rule": str(args.rule),
        "device": str(device),
        "n_envs": int(args.n_envs),
        "hard_pass": bool(all(row["hard_pass"] for row in rows)),
        "max_delta": float(max_delta),
        "rows": rows,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "scalar_vectorized_reward_trace_audit.json"
    md_path = args.output_dir / "scalar_vectorized_reward_trace_audit.md"
    json_path.write_text(json.dumps(_jsonable(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_md(_jsonable(report), md_path)
    print(json.dumps({"hard_pass": report["hard_pass"], "max_delta": report["max_delta"], "json": str(json_path)}, indent=2))


if __name__ == "__main__":
    main()
