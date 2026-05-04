import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from ticl.analysis.phase2_attention_gate_input_alignment_probe import _direction_from_head
from ticl.analysis.phase2_attention_internal_geometry_probe import _evaluate_scalar_head
from ticl.analysis.phase2_delayed_gate_visibility_probe import _extract_delayed_gate_dataset
from ticl.analysis.phase2_gate_output_visibility_probe import _direction_visibility_bundle
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


def _fit_gate_reader_with_scalar_head(
    *,
    source: torch.Tensor,
    pre_output_sum: torch.Tensor,
    returns: torch.Tensor,
    mask: torch.Tensor,
    output_weight: torch.Tensor,
    init_g1: torch.Tensor,
    init_g2: torch.Tensor,
    train_steps: int = 400,
    lr: float = 1e-2,
    weight_decay: float = 1e-5,
    gate_reg: float = 1e-5,
) -> dict:
    source = source.to(dtype=torch.float32)
    pre_output_sum = pre_output_sum.to(dtype=torch.float32)
    returns = returns.to(dtype=torch.float32).reshape(-1)
    mask = mask.reshape(-1).to(dtype=torch.bool)
    output_weight = output_weight.to(dtype=torch.float32)
    g1 = torch.nn.Parameter(init_g1.detach().clone().to(dtype=torch.float32))
    g2 = torch.nn.Parameter(init_g2.detach().clone().to(dtype=torch.float32))
    scalar_w = torch.nn.Parameter(torch.zeros((int(output_weight.shape[0]),), dtype=torch.float32))
    scalar_b = torch.nn.Parameter(torch.zeros((), dtype=torch.float32))
    optimizer = torch.optim.Adam(
        [g1, g2, scalar_w, scalar_b],
        lr=float(lr),
        weight_decay=float(weight_decay),
    )
    losses = []
    for _ in range(int(train_steps)):
        gate_pre = source @ g1
        gate_sigmoid = torch.sigmoid(gate_pre)
        g = gate_sigmoid @ g2
        pre_gated = pre_output_sum * g
        att_out = F.linear(pre_gated, output_weight)
        pred = att_out @ scalar_w + scalar_b
        pred_obj = pred[mask]
        target_obj = returns[mask]
        mse = F.mse_loss(pred_obj, target_obj)
        reg = float(gate_reg) * (g1.square().mean() + g2.square().mean())
        loss = mse + reg
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([g1, g2, scalar_w], 1.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu().item()))
    return {
        "g1": g1.detach().cpu(),
        "g2": g2.detach().cpu(),
        "train_loss_final": float(losses[-1]) if losses else 0.0,
        "train_loss_min": float(min(losses)) if losses else 0.0,
    }


def _apply_gate_reader(
    *,
    source: torch.Tensor,
    pre_output_sum: torch.Tensor,
    output_weight: torch.Tensor,
    g1: torch.Tensor,
    g2: torch.Tensor,
) -> dict:
    source = source.to(dtype=torch.float32)
    pre_output_sum = pre_output_sum.to(dtype=torch.float32)
    output_weight = output_weight.to(dtype=torch.float32)
    g1 = g1.to(dtype=torch.float32)
    g2 = g2.to(dtype=torch.float32)
    gate_pre = source @ g1
    gate_sigmoid = torch.sigmoid(gate_pre)
    g = gate_sigmoid @ g2
    pre_gated = pre_output_sum * g
    att_out = F.linear(pre_gated, output_weight)
    return {
        "gate_pre": gate_pre.detach().cpu(),
        "gate_sigmoid": gate_sigmoid.detach().cpu(),
        "g": g.detach().cpu(),
        "pre_gated": pre_gated.detach().cpu(),
        "att_out": att_out.detach().cpu(),
    }


def run_phase2_delayed_gate_refit_probe(
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
    gate_train_steps: int = 400,
    gate_lr: float = 1e-2,
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
    try:
        _collect_rollout(algo, callback, vec_env, progress_timestep=0, total_timesteps=total_timesteps)
        algo.train()
        same_dataset = _extract_delayed_gate_dataset(algo, max_layer=int(max_layer))
        progress_next = min(int(n_steps), total_timesteps)
        _collect_rollout(algo, callback, vec_env, progress_timestep=progress_next, total_timesteps=total_timesteps)
        next_dataset = _extract_delayed_gate_dataset(algo, max_layer=int(max_layer))
    finally:
        vec_env.close()

    source_names = ["xg", "ln_x_out", "pre_output_sum"]
    layer_results = {}
    for layer_idx in range(int(max_layer) + 1):
        prefix = f"layer{layer_idx}"
        same_att = same_dataset["stages"][f"{prefix}.att_core_raw"]
        same_ln = same_dataset["stages"][f"{prefix}.ln_x_out"]
        same_pre = same_dataset["stages"][f"{prefix}.pre_output_sum"]
        same_mask = same_dataset["objective_mask"]
        same_returns = same_dataset["returns"]
        weights = same_dataset["layer_stats"][f"{prefix}.weights"]
        d_att = _direction_from_head(_fit_ridge_scalar_head(same_att, same_returns, same_mask, ridge_lambda=1e-3))
        d_ln = _direction_from_head(_fit_ridge_scalar_head(same_ln, same_returns, same_mask, ridge_lambda=1e-3))

        layer_sources = {}
        for source_name in source_names:
            same_source = same_dataset["stages"][f"{prefix}.source.{source_name}"]
            next_source = next_dataset["stages"][f"{prefix}.source.{source_name}"]
            frozen_same = {
                "g": same_dataset["stages"][f"{prefix}.gate_from_{source_name}.g"],
                "pre_gated": same_dataset["stages"][f"{prefix}.gate_from_{source_name}.pre_gated"],
                "att_out": same_dataset["stages"][f"{prefix}.gate_from_{source_name}.att_out"],
            }
            frozen_next = {
                "g": next_dataset["stages"][f"{prefix}.gate_from_{source_name}.g"],
                "pre_gated": next_dataset["stages"][f"{prefix}.gate_from_{source_name}.pre_gated"],
                "att_out": next_dataset["stages"][f"{prefix}.gate_from_{source_name}.att_out"],
            }

            refit = _fit_gate_reader_with_scalar_head(
                source=same_source,
                pre_output_sum=same_pre,
                returns=same_returns,
                mask=same_mask,
                output_weight=weights["output"],
                init_g1=weights["g1"],
                init_g2=weights["g2"],
                train_steps=int(gate_train_steps),
                lr=float(gate_lr),
            )
            refit_same = _apply_gate_reader(
                source=same_source,
                pre_output_sum=same_pre,
                output_weight=weights["output"],
                g1=refit["g1"],
                g2=refit["g2"],
            )
            refit_next = _apply_gate_reader(
                source=next_source,
                pre_output_sum=next_dataset["stages"][f"{prefix}.pre_output_sum"],
                output_weight=weights["output"],
                g1=refit["g1"],
                g2=refit["g2"],
            )

            layer_sources[source_name] = {
                "training_summary": {
                    "train_loss_final": refit["train_loss_final"],
                    "train_loss_min": refit["train_loss_min"],
                },
                "frozen": {
                    "same_rollout_scalar_eval": {
                        "pre_gated": _evaluate_scalar_head(frozen_same["pre_gated"], same_returns, same_mask),
                        "att_out": _evaluate_scalar_head(frozen_same["att_out"], same_returns, same_mask),
                    },
                    "next_rollout_scalar_eval": {
                        "pre_gated": _evaluate_scalar_head(frozen_next["pre_gated"], next_dataset["returns"], next_dataset["objective_mask"]),
                        "att_out": _evaluate_scalar_head(frozen_next["att_out"], next_dataset["returns"], next_dataset["objective_mask"]),
                    },
                    "same_rollout_direction_bundles": {
                        "att_core_direction": _direction_visibility_bundle(
                            name="att_core_direction",
                            direction=d_att,
                            xg=same_source,
                            g=frozen_same["g"],
                            pre_output_sum=same_pre,
                            pre_gated=frozen_same["pre_gated"],
                            att_out=frozen_same["att_out"],
                            returns=same_returns,
                            mask=same_mask,
                            g1=weights["g1"],
                            output_weight=weights["output"],
                        ),
                        "ln_x_direction": _direction_visibility_bundle(
                            name="ln_x_direction",
                            direction=d_ln,
                            xg=same_source,
                            g=frozen_same["g"],
                            pre_output_sum=same_pre,
                            pre_gated=frozen_same["pre_gated"],
                            att_out=frozen_same["att_out"],
                            returns=same_returns,
                            mask=same_mask,
                            g1=weights["g1"],
                            output_weight=weights["output"],
                        ),
                    },
                },
                "refit": {
                    "same_rollout_scalar_eval": {
                        "pre_gated": _evaluate_scalar_head(refit_same["pre_gated"], same_returns, same_mask),
                        "att_out": _evaluate_scalar_head(refit_same["att_out"], same_returns, same_mask),
                    },
                    "next_rollout_scalar_eval": {
                        "pre_gated": _evaluate_scalar_head(refit_next["pre_gated"], next_dataset["returns"], next_dataset["objective_mask"]),
                        "att_out": _evaluate_scalar_head(refit_next["att_out"], next_dataset["returns"], next_dataset["objective_mask"]),
                    },
                    "same_rollout_direction_bundles": {
                        "att_core_direction": _direction_visibility_bundle(
                            name="att_core_direction",
                            direction=d_att,
                            xg=same_source,
                            g=refit_same["g"],
                            pre_output_sum=same_pre,
                            pre_gated=refit_same["pre_gated"],
                            att_out=refit_same["att_out"],
                            returns=same_returns,
                            mask=same_mask,
                            g1=refit["g1"],
                            output_weight=weights["output"],
                        ),
                        "ln_x_direction": _direction_visibility_bundle(
                            name="ln_x_direction",
                            direction=d_ln,
                            xg=same_source,
                            g=refit_same["g"],
                            pre_output_sum=same_pre,
                            pre_gated=refit_same["pre_gated"],
                            att_out=refit_same["att_out"],
                            returns=same_returns,
                            mask=same_mask,
                            g1=refit["g1"],
                            output_weight=weights["output"],
                        ),
                    },
                },
            }

        layer_results[prefix] = {
            "sources": layer_sources,
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
            "gate_train_steps": int(gate_train_steps),
            "gate_lr": float(gate_lr),
        },
        "layers": layer_results,
    }


def main():
    parser = argparse.ArgumentParser(description="Compare frozen delayed gate vs refit delayed gate reader.")
    parser.add_argument("checkpoint_path", type=str)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output-json", type=str, required=True)
    args = parser.parse_args()

    result = run_phase2_delayed_gate_refit_probe(
        checkpoint_path=args.checkpoint_path,
        device=args.device,
    )
    output_path = Path(args.output_json).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(str(output_path))


if __name__ == "__main__":
    main()
