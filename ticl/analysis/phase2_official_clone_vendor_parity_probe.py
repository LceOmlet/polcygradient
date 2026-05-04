import argparse
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any, Literal

import torch
from torch import nn

from ticl.analysis.phase2_distribution_head_root_cause_probe import (
    _bucket_entropy,
    _extract_latent_dataset_with_bucket_idx,
)
from ticl.analysis.phase2_simple_scalar_value_head_probe import (
    _build_algo,
    _collect_rollout,
    _default_device,
    _evaluate_predictions,
)
from ticl.analysis.critic_free_single_env_audit import (
    _build_audit_env_cfg,
    _prepare_audit_fixed_env_contract,
)
from ticl.model_builder import load_model
from ticl.models.tabpfn_regressor_vendor import TabPFNOfficialRegressorHarness
from ticl.models.vendor.tabpfn_v7_1_1.runtime_regression_loss import (
    _compute_regression_loss as _vendor_compute_regression_loss,
)


CLONE_ROOT = Path("/home/chen/RLPFN/external/TabPFN_v7_1_1")
VENDORED_UPSTREAM_ROOT = Path(
    "/home/chen/RLPFN/reinforce-terminal-explore/ticl/models/vendor/tabpfn_v7_1_1/upstream"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _file_parity_record(clone_rel: str, vendor_rel: str) -> dict:
    clone_path = CLONE_ROOT / clone_rel
    vendor_path = VENDORED_UPSTREAM_ROOT / vendor_rel
    clone_hash = _sha256(clone_path)
    vendor_hash = _sha256(vendor_path)
    return {
        "clone_path": str(clone_path),
        "vendor_path": str(vendor_path),
        "clone_sha256": clone_hash,
        "vendor_sha256": vendor_hash,
        "byte_identical": bool(clone_hash == vendor_hash),
    }


def _load_module_from_file(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_official_loss_functions(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    selected = []
    want = {"_compute_regression_loss", "_ranked_probability_score_loss_from_bar_logits"}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in want:
            selected.append(node)
    mod = ast.Module(body=selected, type_ignores=[])
    code = compile(mod, str(path), "exec")
    ns = {
        "torch": torch,
        "Any": Any,
        "Literal": Literal,
    }
    exec(code, ns)
    return ns["_compute_regression_loss"], ns["_ranked_probability_score_loss_from_bar_logits"]


def _make_official_components(*, latent_dim: int):
    bar_mod = _load_module_from_file(
        "official_clone_bar_distribution",
        CLONE_ROOT / "src/tabpfn/architectures/base/bar_distribution.py",
    )
    compute_loss, _ = _load_official_loss_functions(
        CLONE_ROOT / "src/tabpfn/finetuning/finetuned_regressor.py"
    )
    head = nn.Sequential(
        nn.Linear(int(latent_dim), 2 * int(latent_dim)),
        nn.GELU(),
        nn.Linear(2 * int(latent_dim), 31),
    )
    borders = torch.linspace(-5.0, 5.0, 32, dtype=torch.float32)
    bardist = bar_mod.FullSupportBarDistribution(borders, ignore_nan_targets=True)
    return head, bardist, compute_loss


def _objective_dataset(dataset: dict, *, device: torch.device) -> dict:
    mask = dataset["objective_mask"]
    return {
        "x": dataset["latent_vf"][mask].to(device=device, dtype=torch.float32),
        "y": dataset["returns"][mask].to(device=device, dtype=torch.float32),
    }


@torch.no_grad()
def _metrics_for_distribution_model(*, model: nn.Module, bardist, dataset_obj: dict) -> dict:
    logits = model(dataset_obj["x"]).to(dtype=torch.float32)
    probs = torch.softmax(logits, dim=-1)
    mean_pred = bardist.mean(logits)
    center_idx = int(bardist.num_bars) // 2
    band_lo = max(center_idx - 1, 0)
    band_hi = min(center_idx + 2, int(bardist.num_bars))
    eval_stats = _evaluate_predictions(
        mean_pred.detach().cpu(),
        dataset_obj["y"].detach().cpu(),
        torch.ones_like(dataset_obj["y"], dtype=torch.bool).cpu(),
    )
    return {
        "bucket_entropy": float(_bucket_entropy(logits)),
        "logit_l2_mean": float(logits.norm(dim=-1).mean().detach().cpu().item()),
        "mean_pred_std": float(mean_pred.std(unbiased=False).detach().cpu().item()),
        "central_bucket_prob_mean": float(probs[:, center_idx].mean().detach().cpu().item()),
        "central_band_mass_mean": float(probs[:, band_lo:band_hi].sum(dim=-1).mean().detach().cpu().item()),
        "central_bucket_argmax_fraction": float((logits.argmax(dim=-1) == center_idx).to(dtype=torch.float32).mean().detach().cpu().item()),
        **eval_stats,
    }


def _state_diff_stats(model_a: nn.Module, model_b: nn.Module) -> dict:
    sq = 0.0
    max_abs = 0.0
    mean_abs_num = 0.0
    mean_abs_den = 0
    for (_, ta), (_, tb) in zip(model_a.state_dict().items(), model_b.state_dict().items()):
        diff = (ta.detach().cpu().to(dtype=torch.float32) - tb.detach().cpu().to(dtype=torch.float32)).reshape(-1)
        if diff.numel() > 0:
            max_abs = max(max_abs, float(diff.abs().max().item()))
            sq += float(diff.pow(2).sum().item())
            mean_abs_num += float(diff.abs().sum().item())
            mean_abs_den += int(diff.numel())
    return {
        "param_diff_l2": float(sq**0.5),
        "param_diff_max_abs": float(max_abs),
        "param_diff_mean_abs": float(mean_abs_num / max(mean_abs_den, 1)),
    }


@torch.no_grad()
def _logit_diff_stats(model_a: nn.Module, model_b: nn.Module, x: torch.Tensor) -> dict:
    logits_a = model_a(x).to(dtype=torch.float32)
    logits_b = model_b(x).to(dtype=torch.float32)
    diff = (logits_a - logits_b).reshape(-1)
    return {
        "logit_diff_max_abs": float(diff.abs().max().detach().cpu().item()),
        "logit_diff_mean_abs": float(diff.abs().mean().detach().cpu().item()),
    }


def _vendor_loss(harness: TabPFNOfficialRegressorHarness, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    model = harness.output_projection
    param = next(model.parameters())
    x = x.to(device=param.device, dtype=param.dtype)
    logits = model(x).to(dtype=torch.float32).unsqueeze(0)
    targets = y.to(device=logits.device, dtype=torch.float32).unsqueeze(0)
    bardist = harness.znorm_space_bardist_.to(device=logits.device)
    return _vendor_compute_regression_loss(
        logits_BQL=logits,
        targets_BQ=targets,
        bardist_loss_fn=bardist,
        ce_loss_weight=0.0,
        crps_loss_weight=1.0,
        crls_loss_weight=0.0,
        mse_loss_weight=1.0,
        mse_loss_clip=None,
        mae_loss_weight=0.0,
        mae_loss_clip=None,
    )


def _official_loss(compute_loss_fn, bardist, model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    logits = model(x).to(dtype=torch.float32).unsqueeze(0)
    return compute_loss_fn(
        logits_BQL=logits,
        targets_BQ=y.unsqueeze(0),
        bardist_loss_fn=bardist,
        ce_loss_weight=0.0,
        crps_loss_weight=1.0,
        crls_loss_weight=0.0,
        mse_loss_weight=1.0,
        mse_loss_clip=None,
        mae_loss_weight=0.0,
        mae_loss_clip=None,
    )


def run_phase2_official_clone_vendor_parity_probe(
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
    build_seed: int = 4040,
    ppo_reset_env_state_at_sep: bool = True,
    actor_baseline_mode: str = "learned",
    fit_steps: int = 20,
    checkpoint_steps: str = "0,1,2,5,10,20",
    head_fit_learning_rate: float = 1e-3,
    head_fit_max_grad_norm: float = 0.5,
) -> dict:
    device_obj = torch.device(str(device or _default_device()))
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
    try:
        _collect_rollout(algo, callback, vec_env, progress_timestep=0, total_timesteps=int(n_steps))
        algo.train()
        same_dataset = _extract_latent_dataset_with_bucket_idx(algo)
        _collect_rollout(algo, callback, vec_env, progress_timestep=int(n_steps), total_timesteps=2 * int(n_steps))
        next_dataset = _extract_latent_dataset_with_bucket_idx(algo)

        same_obj = _objective_dataset(same_dataset, device=device_obj)
        next_obj = _objective_dataset(next_dataset, device=device_obj)
        latent_dim = int(same_obj["x"].shape[-1])

        official_model, official_bardist, official_compute_loss = _make_official_components(latent_dim=latent_dim)
        official_model = official_model.to(device_obj)
        official_bardist = official_bardist.to(device_obj)

        vendor_harness = TabPFNOfficialRegressorHarness(
            emsize=latent_dim,
            num_buckets=31,
            value_range=5.0,
            ignore_nan_targets=True,
        ).to(device_obj)
        vendor_harness.reset_target_statistics()
        vendor_harness.output_projection.load_state_dict(copy.deepcopy(official_model.state_dict()))

        opt_official = torch.optim.Adam(official_model.parameters(), lr=float(head_fit_learning_rate))
        opt_vendor = torch.optim.Adam(vendor_harness.parameters(), lr=float(head_fit_learning_rate))

        checkpoints = sorted({int(v.strip()) for v in checkpoint_steps.split(",") if v.strip()})
        if 0 not in checkpoints:
            checkpoints = [0] + checkpoints
        if int(fit_steps) not in checkpoints:
            checkpoints.append(int(fit_steps))

        history = []
        for step_idx in range(int(fit_steps) + 1):
            if step_idx in checkpoints:
                vendor_model = vendor_harness.output_projection
                history.append(
                    {
                        "fit_step": int(step_idx),
                        "same_rollout": {
                            "official_clone": _metrics_for_distribution_model(
                                model=official_model,
                                bardist=official_bardist,
                                dataset_obj=same_obj,
                            ),
                            "vendor_harness": _metrics_for_distribution_model(
                                model=vendor_model,
                                bardist=vendor_harness.znorm_space_bardist_,
                                dataset_obj=same_obj,
                            ),
                        },
                        "next_rollout": {
                            "official_clone": _metrics_for_distribution_model(
                                model=official_model,
                                bardist=official_bardist,
                                dataset_obj=next_obj,
                            ),
                            "vendor_harness": _metrics_for_distribution_model(
                                model=vendor_model,
                                bardist=vendor_harness.znorm_space_bardist_,
                                dataset_obj=next_obj,
                            ),
                        },
                        "parity": {
                            **_state_diff_stats(official_model, vendor_model),
                            **_logit_diff_stats(official_model, vendor_model, same_obj["x"]),
                            "official_loss_same": float(_official_loss(official_compute_loss, official_bardist, official_model, same_obj["x"], same_obj["y"]).detach().cpu().item()),
                            "vendor_loss_same": float(_vendor_loss(vendor_harness, same_obj["x"], same_obj["y"]).detach().cpu().item()),
                        },
                    }
                )
            if step_idx == int(fit_steps):
                break

            official_loss = _official_loss(
                official_compute_loss,
                official_bardist,
                official_model,
                same_obj["x"],
                same_obj["y"],
            )
            opt_official.zero_grad(set_to_none=True)
            official_loss.backward()
            torch.nn.utils.clip_grad_norm_(official_model.parameters(), float(head_fit_max_grad_norm))
            opt_official.step()

            vendor_loss = _vendor_loss(vendor_harness, same_obj["x"], same_obj["y"])
            opt_vendor.zero_grad(set_to_none=True)
            vendor_loss.backward()
            torch.nn.utils.clip_grad_norm_(vendor_harness.parameters(), float(head_fit_max_grad_norm))
            opt_vendor.step()

        return {
            "audit_entry": "phase2_official_clone_vendor_parity_probe",
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
            "build_seed": int(build_seed),
            "ppo_reset_env_state_at_sep": bool(ppo_reset_env_state_at_sep),
            "actor_baseline_mode": str(actor_baseline_mode),
            "fit_steps": int(fit_steps),
            "checkpoint_steps": checkpoints,
            "head_fit_learning_rate": float(head_fit_learning_rate),
            "head_fit_max_grad_norm": float(head_fit_max_grad_norm),
            "fixed_env_contract": dict(fixed_bundle["fixed_env_contract"]),
            "source_parity": {
                "bar_distribution": _file_parity_record(
                    "src/tabpfn/architectures/base/bar_distribution.py",
                    "src/tabpfn/architectures/base/bar_distribution.py",
                ),
                "finetuned_regressor": _file_parity_record(
                    "src/tabpfn/finetuning/finetuned_regressor.py",
                    "src/tabpfn/finetuning/finetuned_regressor.py",
                ),
                "tabpfn_v2_6": _file_parity_record(
                    "src/tabpfn/architectures/tabpfn_v2_6.py",
                    "src/tabpfn/architectures/tabpfn_v2_6.py",
                ),
                "clone_git_head": _sha256(CLONE_ROOT / ".git/HEAD") if (CLONE_ROOT / ".git/HEAD").exists() else None,
            },
            "history": history,
        }
    finally:
        vec_env.close()
        del algo, callback, vec_env
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> None:
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
    parser.add_argument("--build-seed", type=int, default=4040)
    parser.add_argument("--ppo-reset-env-state-at-sep", action="store_true")
    parser.add_argument("--actor-baseline-mode", type=str, default="learned")
    parser.add_argument("--fit-steps", type=int, default=20)
    parser.add_argument("--checkpoint-steps", type=str, default="0,1,2,5,10,20")
    parser.add_argument("--head-fit-learning-rate", type=float, default=1e-3)
    parser.add_argument("--head-fit-max-grad-norm", type=float, default=0.5)
    parser.add_argument(
        "--output-json",
        type=str,
        default="/home/chen/RLPFN/artifacts/phase2_official_clone_vendor_parity_probe.json",
    )
    args = parser.parse_args()
    report = run_phase2_official_clone_vendor_parity_probe(
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
        build_seed=int(args.build_seed),
        ppo_reset_env_state_at_sep=bool(args.ppo_reset_env_state_at_sep),
        actor_baseline_mode=str(args.actor_baseline_mode),
        fit_steps=int(args.fit_steps),
        checkpoint_steps=str(args.checkpoint_steps),
        head_fit_learning_rate=float(args.head_fit_learning_rate),
        head_fit_max_grad_norm=float(args.head_fit_max_grad_norm),
    )
    output_path = Path(args.output_json).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
