import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ticl.analysis.phase2_attention_gate_input_alignment_probe import (
    _colspace_visibility,
    _direction_from_head,
    _safe_cosine,
)
from ticl.analysis.phase2_ln_x_direction_transform_probe import _extract_ln_x_dataset
from ticl.analysis.phase2_simple_scalar_value_head_probe import (
    _build_algo,
    _collect_rollout,
    _default_device,
    _fit_ridge_scalar_head,
)
from ticl.analysis.critic_free_single_env_audit import (
    _build_audit_env_cfg,
    _prepare_audit_fixed_env_contract,
    _seed_all,
)
from ticl.model_builder import load_model


def _safe_corrcoef(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    finite = np.isfinite(x) & np.isfinite(y)
    if int(finite.sum()) <= 1:
        return 0.0
    x = x[finite] - float(np.asarray(x[finite], dtype=np.float32).mean())
    y = y[finite] - float(np.asarray(y[finite], dtype=np.float32).mean())
    denom = float(np.linalg.norm(x) * np.linalg.norm(y))
    if denom <= 0.0:
        return 0.0
    return float(np.dot(x, y) / denom)


def _project_array(x: torch.Tensor, direction: torch.Tensor | None, mask: torch.Tensor) -> np.ndarray:
    mask = mask.reshape(-1).to(dtype=torch.bool)
    if direction is None:
        return np.zeros((int(mask.sum().item()),), dtype=np.float32)
    x_obj = x[mask].to(dtype=torch.float32)
    centered = x_obj - x_obj.mean(dim=0, keepdim=True)
    return (centered @ direction.to(dtype=torch.float32)).detach().cpu().numpy()


def _proj_stats(x: torch.Tensor, direction: torch.Tensor | None, returns: torch.Tensor, mask: torch.Tensor) -> dict:
    proj = _project_array(x, direction, mask)
    y = returns[mask.reshape(-1).to(dtype=torch.bool)].detach().cpu().numpy()
    if proj.size <= 1:
        return {
            "projection_std": 0.0,
            "projection_mean_abs": 0.0,
            "corr_with_target": 0.0,
        }
    return {
        "projection_std": float(np.asarray(proj, dtype=np.float32).std()),
        "projection_mean_abs": float(np.asarray(np.abs(proj), dtype=np.float32).mean()),
        "corr_with_target": _safe_corrcoef(proj, y),
    }


def _gate_gain_for_direction(g: torch.Tensor, direction: torch.Tensor | None, returns: torch.Tensor, mask: torch.Tensor) -> dict:
    mask = mask.reshape(-1).to(dtype=torch.bool)
    if direction is None:
        return {
            "weighted_gain_mean": 0.0,
            "weighted_gain_std": 0.0,
            "corr_with_target": 0.0,
        }
    g_obj = g[mask].to(dtype=torch.float32)
    d = direction.to(dtype=torch.float32).reshape(-1)
    weighted_gain = (g_obj * (d ** 2)).sum(dim=-1).detach().cpu().numpy()
    y = returns[mask].detach().cpu().numpy()
    return {
        "weighted_gain_mean": float(np.asarray(weighted_gain, dtype=np.float32).mean()),
        "weighted_gain_std": float(np.asarray(weighted_gain, dtype=np.float32).std()),
        "corr_with_target": _safe_corrcoef(weighted_gain, y),
    }


def _ratio_stats(numer: np.ndarray, denom: np.ndarray) -> dict:
    numer = np.asarray(numer, dtype=np.float32).reshape(-1)
    denom = np.asarray(denom, dtype=np.float32).reshape(-1)
    valid = np.abs(denom) > 1e-8
    if int(valid.sum()) <= 1:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    ratio = numer[valid] / denom[valid]
    return {
        "mean": float(np.asarray(ratio, dtype=np.float32).mean()),
        "std": float(np.asarray(ratio, dtype=np.float32).std()),
        "min": float(np.asarray(ratio, dtype=np.float32).min()),
        "max": float(np.asarray(ratio, dtype=np.float32).max()),
    }


def _direction_visibility_bundle(
    *,
    name: str,
    direction: torch.Tensor | None,
    xg: torch.Tensor,
    g: torch.Tensor,
    pre_output_sum: torch.Tensor,
    pre_gated: torch.Tensor,
    att_out: torch.Tensor,
    returns: torch.Tensor,
    mask: torch.Tensor,
    g1: torch.Tensor,
    output_weight: torch.Tensor,
) -> dict:
    proj_xg = _project_array(xg, direction, mask)
    proj_pre = _project_array(pre_output_sum, direction, mask)
    proj_gated = _project_array(pre_gated, direction, mask)
    proj_att = _project_array(att_out, direction, mask)
    return {
        "name": str(name),
        "g1_colspace_visibility": _colspace_visibility(g1, direction),
        "output_colspace_visibility": _colspace_visibility(output_weight, direction),
        "xg": _proj_stats(xg, direction, returns, mask),
        "pre_output_sum": _proj_stats(pre_output_sum, direction, returns, mask),
        "pre_gated": _proj_stats(pre_gated, direction, returns, mask),
        "att_out": _proj_stats(att_out, direction, returns, mask),
        "gate_weighted_gain": _gate_gain_for_direction(g, direction, returns, mask),
        "suppression_ratios": {
            "pre_gated_vs_pre_output_sum": _ratio_stats(proj_gated, proj_pre),
            "att_out_vs_pre_gated": _ratio_stats(proj_att, proj_gated),
            "att_out_vs_pre_output_sum": _ratio_stats(proj_att, proj_pre),
        },
    }


def run_phase2_gate_output_visibility_probe(
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
    outer_epochs: int = 1,
    build_seed: int = 4040,
    ppo_reset_env_state_at_sep: bool = True,
    actor_baseline_mode: str = "learned",
    max_layer: int = 1,
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
    algo, callback, vec_env = _build_algo(
        checkpoint_path=checkpoint_path,
        device_obj=device_obj,
        frozen_h=frozen_h,
        env_cfg=env_cfg,
        num_features=int(num_features),
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
    from ticl.analysis.phase2_attention_gate_path_probe import _extract_gate_path_dataset

    try:
        _collect_rollout(algo, callback, vec_env, progress_timestep=0, total_timesteps=total_timesteps)
        algo.train()
        dataset = _extract_ln_x_dataset(algo, max_layer=int(max_layer))
        gate_dataset = _extract_gate_path_dataset(algo, max_layer=int(max_layer))
    finally:
        vec_env.close()

    layer_results = {}
    for layer_idx in range(int(max_layer) + 1):
        prefix = f"layer{layer_idx}"
        att_core = dataset["stages"][f"{prefix}.att_core_raw"]
        ln_x = dataset["stages"][f"{prefix}.ln_x_out"]
        pre_output_sum = dataset["stages"][f"{prefix}.pre_output_sum"]
        att_out = dataset["stages"][f"{prefix}.att_out"]

        xg = gate_dataset["stages"][f"{prefix}.xg"]
        g = gate_dataset["stages"][f"{prefix}.g"]
        g1 = gate_dataset["layer_stats"][f"{prefix}.gate_weights"]["g1"]

        output_weight = algo.policy.rlpfn_model.rwkv_core.blocks[layer_idx].att.output.weight.detach().cpu().to(dtype=torch.float32)

        att_head = _fit_ridge_scalar_head(att_core, dataset["returns"], dataset["objective_mask"], ridge_lambda=1e-3)
        ln_head = _fit_ridge_scalar_head(ln_x, dataset["returns"], dataset["objective_mask"], ridge_lambda=1e-3)
        d_att = _direction_from_head(att_head)
        d_ln = _direction_from_head(ln_head)

        layer_results[prefix] = {
            "direction_cosines": {
                "att_vs_ln": _safe_cosine(d_att, d_ln),
            },
            "att_direction_bundle": _direction_visibility_bundle(
                name="att_core_direction",
                direction=d_att,
                xg=xg,
                g=g,
                pre_output_sum=pre_output_sum,
                pre_gated=gate_dataset["stages"][f"{prefix}.pre_gated"],
                att_out=att_out,
                returns=dataset["returns"],
                mask=dataset["objective_mask"],
                g1=g1,
                output_weight=output_weight,
            ),
            "ln_direction_bundle": _direction_visibility_bundle(
                name="ln_x_direction",
                direction=d_ln,
                xg=xg,
                g=g,
                pre_output_sum=pre_output_sum,
                pre_gated=gate_dataset["stages"][f"{prefix}.pre_gated"],
                att_out=att_out,
                returns=dataset["returns"],
                mask=dataset["objective_mask"],
                g1=g1,
                output_weight=output_weight,
            ),
        }

    return {
        "checkpoint_path": str(Path(checkpoint_path).expanduser().resolve()),
        "contract": {
            "frozen_h_seed": int(frozen_h_seed),
            "train_env_seed": int(train_env_seed),
            "train_rollout_seed": int(train_rollout_seed),
            "single_eval_pos": int(single_eval_pos),
            "n_steps": int(n_steps),
            "batch_size": int(batch_size),
            "n_epochs": int(n_epochs),
            "learning_rate": float(learning_rate),
            "target_kl": float(target_kl),
            "build_seed": int(build_seed),
            "max_layer": int(max_layer),
        },
        "layers": layer_results,
    }


def main():
    parser = argparse.ArgumentParser(description="Gate/output visibility probe for att_core vs ln_x directions.")
    parser.add_argument("checkpoint_path", type=str)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output-json", type=str, required=True)
    args = parser.parse_args()

    result = run_phase2_gate_output_visibility_probe(
        checkpoint_path=args.checkpoint_path,
        device=args.device,
    )
    output_path = Path(args.output_json).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(str(output_path))


if __name__ == "__main__":
    main()
