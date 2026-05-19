#!/usr/bin/env python3
"""Read-only evidence audit for M4 state/obs controllability axes.

This script intentionally does not modify h, sampler, PPO, or fit_model state.
It instantiates the fixed M4 h-list through EnvironmentPrior and probes the
same transition/finalize/carry path used by true rollout.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch

from ticl.model_configs import get_model_default_config
from ticl.priors.environment_prior import EnvironmentPrior


def _to_device_tensor(value: Any, *, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    if torch.is_tensor(value):
        return value.to(device=device, dtype=dtype)
    return torch.as_tensor(value, device=device, dtype=dtype)


def _rank_and_spectrum(mat: torch.Tensor, *, rel_tol: float, abs_tol: float) -> Tuple[int, List[float], float]:
    if mat.numel() == 0:
        return 0, [], 0.0
    m = mat.detach().to(dtype=torch.float32)
    if m.ndim != 2:
        m = m.reshape(int(m.shape[0]), -1)
    if min(m.shape) == 0:
        return 0, [], 0.0
    s = torch.linalg.svdvals(m)
    smax = float(s.max().item()) if s.numel() else 0.0
    tol = max(float(abs_tol), float(rel_tol) * smax)
    rank = int((s > tol).sum().item())
    return rank, [float(v) for v in s[:16].detach().cpu().tolist()], smax


def _transition_call(fn, env_in: torch.Tensor):
    try:
        return fn(env_in, generators_for_noise=None)
    except TypeError:
        return fn(env_in)


def _expand_batch(x: torch.Tensor, batch: int) -> torch.Tensor:
    if x.ndim == 1:
        x = x.unsqueeze(0)
    if int(x.shape[0]) == batch:
        return x
    if int(x.shape[0]) == 1:
        return x.expand(batch, -1).clone()
    raise ValueError(f"cannot expand tensor with shape={tuple(x.shape)} to batch={batch}")


def _make_env_input(
    prior: EnvironmentPrior,
    env: Dict[str, Any],
    *,
    carry_state: torch.Tensor,
    visible_state: torch.Tensor,
    action: torch.Tensor,
    noise: torch.Tensor,
) -> torch.Tensor:
    batch = int(action.shape[0])
    device = action.device
    dtype = action.dtype
    state_dim = int(env["state_dim"])
    obs_dim = int(env["obs_dim"])
    action_dim = int(env["action_dim"])
    noise_dim = int(env["noise_dim"])
    env_input_dim = int(env["env_input_dim"])
    env_action_start = int(env["env_action_start"])
    env_noise_start = int(env["env_noise_start"])
    env_obs_start = env.get("env_obs_start", None)
    state_input_scale = env.get("state_input_scale", 1.0)

    env_in = torch.zeros((batch, env_input_dim), device=device, dtype=dtype)
    env_in[:, :state_dim] = prior._scale_state_env_input(carry_state, state_input_scale)
    if env_obs_start is not None:
        env_in[:, int(env_obs_start) : int(env_obs_start) + obs_dim] = visible_state[:, :obs_dim]
    env_in[:, env_action_start : env_action_start + action_dim] = action
    env_in[:, env_noise_start : env_noise_start + noise_dim] = noise
    return env_in


def _step_true_runner_path(
    prior: EnvironmentPrior,
    env: Dict[str, Any],
    *,
    carry_state: torch.Tensor,
    visible_state: torch.Tensor,
    action: torch.Tensor,
    noise: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    env_in = _make_env_input(
        prior,
        env,
        carry_state=carry_state,
        visible_state=visible_state,
        action=action,
        noise=noise,
    )
    transition_generator = prior._require_transition_generator(env)
    transition_out = _transition_call(transition_generator, env_in)
    x_next, _reward_unit, _terminal_signal, _terminal_bonus_base = prior._unpack_transition_output(
        transition_out
    )
    reference_semantics_enabled = prior._env_uses_reference_semantics(env)
    visible_next = prior._finalize_exact_scm_visible_state_next(
        generated_state_next=x_next,
        visible_state_prev=visible_state,
        state_noise_t=None,
        state_noise_std=env.get("state_noise_std", 0.0),
        reference_semantics_enabled=reference_semantics_enabled,
        state_clip=env.get("state_clip", float("inf")),
        state_output_scale=env.get("state_output_scale", 1.0),
        state_highway_enabled=env.get("state_highway_enabled", False),
        state_highway_lambda=env.get("state_highway_lambda", 0.0),
        state_full_rms_enabled=env.get("state_full_rms_enabled", False),
        state_full_rms_target=env.get("state_full_rms_target", 1.0),
        state_full_rms_floor_enabled=env.get("state_full_rms_floor_enabled", False),
        state_full_rms_max_upscale=env.get("state_full_rms_max_upscale", 4.0),
    )
    carry_next = prior._mix_exact_scm_carry_state(
        carry_state,
        visible_next,
        env.get("alpha", 1.0),
    )
    return carry_next, visible_next


def _audit_one(
    prior: EnvironmentPrior,
    h: Dict[str, Any],
    *,
    env_index: int,
    device: torch.device,
    eps: float,
    max_delay: int,
    rel_tol: float,
    abs_tol: float,
    rng_seed: int,
) -> Dict[str, Any]:
    env = prior._sample_environment(h=h, device=device, rng_seed=int(rng_seed))
    state_dim = int(env["state_dim"])
    obs_dim = int(env["obs_dim"])
    action_dim = int(env["action_dim"])
    noise_dim = int(env["noise_dim"])
    initial = _to_device_tensor(env["initial_state"], device=device)
    if initial.ndim == 1:
        initial = initial.unsqueeze(0)
    batch = action_dim + 1
    carry0, visible0 = prior._init_exact_scm_carry_and_visible_state(initial)
    carry = _expand_batch(carry0, batch)
    visible = _expand_batch(visible0, batch)

    action = torch.zeros((batch, action_dim), device=device, dtype=torch.float32)
    if action_dim > 0:
        action[1:, :] = torch.eye(action_dim, device=device, dtype=torch.float32) * float(eps)
    noise = torch.zeros((batch, noise_dim), device=device, dtype=torch.float32)

    rank_by_delay: List[int] = []
    smax_by_delay: List[float] = []
    first_delay = None
    for delay_idx in range(1, int(max_delay) + 1):
        carry, visible = _step_true_runner_path(
            prior,
            env,
            carry_state=carry,
            visible_state=visible,
            action=action if delay_idx == 1 else torch.zeros_like(action),
            noise=noise,
        )
        obs_delta = (visible[1:, :obs_dim] - visible[:1, :obs_dim]) / float(eps)
        rank, _spectrum, smax = _rank_and_spectrum(obs_delta, rel_tol=rel_tol, abs_tol=abs_tol)
        rank_by_delay.append(rank)
        smax_by_delay.append(smax)
        if first_delay is None and rank > 0:
            first_delay = delay_idx

    one_step_rank = int(rank_by_delay[0]) if rank_by_delay else 0
    max_rank = int(max(rank_by_delay)) if rank_by_delay else 0
    controllability_cap = int(min(action_dim, obs_dim))
    obs_projection_rank = int(min(obs_dim, state_dim))
    obs_projection_cap = int(min(obs_dim, state_dim))
    return {
        "env_index": int(env_index),
        "state_dim": state_dim,
        "obs_dim": obs_dim,
        "action_dim": action_dim,
        "noise_dim": noise_dim,
        "zero_pad_dim": int(env.get("zero_pad_dim", 0)),
        "obs_projection_rank": obs_projection_rank,
        "obs_projection_cap": obs_projection_cap,
        "obs_projection_rank_ratio": float(obs_projection_rank / max(1, obs_projection_cap)),
        "controllability_rank_step1": one_step_rank,
        "controllability_rank_max_delay": max_rank,
        "controllability_rank_cap": controllability_cap,
        "controllability_rank_step1_ratio": float(one_step_rank / max(1, controllability_cap)),
        "controllability_rank_max_ratio": float(max_rank / max(1, controllability_cap)),
        "controllability_delay": int(first_delay) if first_delay is not None else None,
        "rank_by_delay": rank_by_delay,
        "smax_by_delay": smax_by_delay,
        "reference_semantics_enabled": bool(prior._env_uses_reference_semantics(env)),
        "state_full_rms_enabled": bool(env.get("state_full_rms_enabled", False)),
        "state_highway_enabled": bool(env.get("state_highway_enabled", False)),
        "alpha": float(_to_device_tensor(env.get("alpha", 0.0), device=device).detach().flatten()[0].item()),
        "status": "ok",
    }


def _bin_rank_ratio(ratio: float) -> str:
    if ratio <= 0.0:
        return "zero"
    if ratio < 0.25:
        return "low"
    if ratio < 0.75:
        return "mid"
    if ratio < 0.999:
        return "high"
    return "full"


def _summarize(rows: List[Dict[str, Any]], *, elapsed_s: float, h_list_path: str) -> Dict[str, Any]:
    ok = [r for r in rows if r.get("status") == "ok"]
    failed = [r for r in rows if r.get("status") != "ok"]
    delay_bins: Dict[str, int] = {}
    step1_bins: Dict[str, int] = {}
    max_bins: Dict[str, int] = {}
    obs_bins: Dict[str, int] = {}
    for r in ok:
        delay = r.get("controllability_delay")
        delay_key = "none" if delay is None else f"d{int(delay)}"
        delay_bins[delay_key] = delay_bins.get(delay_key, 0) + 1
        step1_key = _bin_rank_ratio(float(r.get("controllability_rank_step1_ratio", 0.0)))
        max_key = _bin_rank_ratio(float(r.get("controllability_rank_max_ratio", 0.0)))
        obs_key = _bin_rank_ratio(float(r.get("obs_projection_rank_ratio", 0.0)))
        step1_bins[step1_key] = step1_bins.get(step1_key, 0) + 1
        max_bins[max_key] = max_bins.get(max_key, 0) + 1
        obs_bins[obs_key] = obs_bins.get(obs_key, 0) + 1

    def _mean(key: str) -> float:
        vals = [float(r.get(key, 0.0)) for r in ok]
        return float(sum(vals) / max(1, len(vals)))

    return {
        "audit": "phase2_m4_state_obs_evidence_audit",
        "h_list_path": h_list_path,
        "elapsed_s": float(elapsed_s),
        "count": len(rows),
        "ok_count": len(ok),
        "failed_count": len(failed),
        "failure_examples": failed[:8],
        "definition": {
            "obs_projection_rank": "rank of true rollout observation projection obs_t = visible_state_t[:obs_dim]",
            "controllability_rank_step1": "rank of finite-difference Jacobian d obs_{t+1} / d action_t through transition_generator, finalize, and carry update path",
            "controllability_delay": "first delay k<=max_delay where d obs_{t+k} / d action_t has positive finite-difference rank under zero follow-up action/noise",
        },
        "coverage": {
            "obs_projection_rank_ratio_bins": obs_bins,
            "controllability_rank_step1_ratio_bins": step1_bins,
            "controllability_rank_max_delay_ratio_bins": max_bins,
            "controllability_delay_bins": delay_bins,
        },
        "means": {
            "obs_projection_rank_ratio": _mean("obs_projection_rank_ratio"),
            "controllability_rank_step1_ratio": _mean("controllability_rank_step1_ratio"),
            "controllability_rank_max_ratio": _mean("controllability_rank_max_ratio"),
        },
        "hard_evidence_flags": {
            "obs_projection_full_for_all_ok": all(
                float(r.get("obs_projection_rank_ratio", 0.0)) >= 0.999 for r in ok
            ) if ok else False,
            "some_action_to_obs_controllability_for_all_ok": all(
                int(r.get("controllability_rank_max_delay", 0)) > 0 for r in ok
            ) if ok else False,
            "immediate_action_to_obs_controllability_for_all_ok": all(
                int(r.get("controllability_rank_step1", 0)) > 0 for r in ok
            ) if ok else False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--h-list", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--limit", type=int, default=4096)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--eps", type=float, default=1e-3)
    parser.add_argument("--max-delay", type=int, default=4)
    parser.add_argument("--rel-tol", type=float, default=1e-4)
    parser.add_argument("--abs-tol", type=float, default=1e-6)
    parser.add_argument("--rng-seed-base", type=int, default=451900)
    parser.add_argument("--progress-every", type=int, default=64)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    h_list_path = Path(args.h_list)
    h_list = json.loads(h_list_path.read_text())
    start = max(0, int(args.offset))
    stop = min(len(h_list), start + max(0, int(args.limit)))
    h_slice = h_list[start:stop]

    device = torch.device(args.device if torch.cuda.is_available() or not str(args.device).startswith("cuda") else "cpu")
    cfg = get_model_default_config("rlpfn")
    env_cfg = dict(cfg["prior"]["environment"])
    prior = EnvironmentPrior(env_cfg)

    rows: List[Dict[str, Any]] = []
    t0 = time.time()
    for local_i, h in enumerate(h_slice):
        env_index = start + local_i
        try:
            with torch.no_grad():
                row = _audit_one(
                    prior,
                    h,
                    env_index=env_index,
                    device=device,
                    eps=float(args.eps),
                    max_delay=int(args.max_delay),
                    rel_tol=float(args.rel_tol),
                    abs_tol=float(args.abs_tol),
                    rng_seed=int(args.rng_seed_base) + int(env_index),
                )
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower() and str(device).startswith("cuda"):
                row = {
                    "env_index": env_index,
                    "status": "oom",
                    "error": str(exc).splitlines()[0],
                }
                rows.append(row)
                break
            row = {
                "env_index": env_index,
                "status": "error",
                "error": str(exc).splitlines()[0],
                "traceback": traceback.format_exc(limit=8),
            }
        except Exception as exc:  # noqa: BLE001 - audit should record failures.
            row = {
                "env_index": env_index,
                "status": "error",
                "error": str(exc).splitlines()[0],
                "traceback": traceback.format_exc(limit=8),
            }
        rows.append(row)
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()
        if (local_i + 1) % max(1, int(args.progress_every)) == 0:
            elapsed = time.time() - t0
            print(
                f"[state-obs-audit] done={local_i + 1}/{len(h_slice)} "
                f"elapsed_s={elapsed:.1f}",
                flush=True,
            )

    elapsed_s = time.time() - t0
    csv_path = out_dir / "state_obs_evidence_audit_rows.csv"
    fieldnames = sorted({k for r in rows for k in r.keys()})
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            encoded = {}
            for k, v in r.items():
                if isinstance(v, (list, dict)):
                    encoded[k] = json.dumps(v, sort_keys=True)
                else:
                    encoded[k] = v
            writer.writerow(encoded)

    report = _summarize(rows, elapsed_s=elapsed_s, h_list_path=str(h_list_path))
    (out_dir / "state_obs_evidence_audit_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True)
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
