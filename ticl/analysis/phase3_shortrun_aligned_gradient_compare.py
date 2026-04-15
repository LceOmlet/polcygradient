import argparse
import copy
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ticl.analysis.phase3_longrun_driver import _phase3_shortrun_aligned_extra_config
from ticl.analysis.phase3_longrun_gradient_compare import (
    _build_algo_for_config,
    _collect_and_measure_total_grad,
    _default_device,
    _relevant_training_signature,
    _resolve_continue_run_config,
    _signature_diff,
)
from ticl.analysis.sep_state_reset_gradient_compare import _build_fixed_env_h
from ticl.analysis.critic_free_single_env_audit import _build_audit_env_cfg


def _short_run_aligned_training_config(base_config: dict[str, Any]) -> dict[str, Any]:
    cfg = copy.deepcopy(base_config)
    update = _phase3_shortrun_aligned_extra_config()
    from ticl.config_utils import update_config
    update_config(cfg, update)
    return cfg


def run_phase3_shortrun_aligned_gradient_compare(
    *,
    checkpoint_path: str,
    device: str | None = None,
    frozen_h_seed: int = 12345,
    train_env_seed: int = 2020,
    rollout_seed: int = 2020,
    single_eval_pos: int = 1946,
    n_steps: int = 2048,
    batch_size: int = 2048,
    learning_rate: float = 2e-4,
) -> dict[str, Any]:
    checkpoint_path = str(Path(checkpoint_path).expanduser().resolve())
    device_obj = torch.device(str(device or _default_device()))
    resume_config = _resolve_continue_run_config(checkpoint_path=checkpoint_path)
    short_run_aligned = _short_run_aligned_training_config(resume_config)

    resume_sig = _relevant_training_signature(resume_config)
    shortrun_sig = _relevant_training_signature(short_run_aligned)
    sig_diff = _signature_diff(resume_sig, shortrun_sig)

    env_cfg = _build_audit_env_cfg(resume_config["prior"]["environment"])
    frozen_h = _build_fixed_env_h(prior_cfg=env_cfg, frozen_h_seed=int(frozen_h_seed))

    reports = {}
    grad_vectors = {}
    for mode_name, resolved_config in (
        ("resume_checkpoint_semantics", resume_config),
        ("short_run_aligned_training_semantics", short_run_aligned),
    ):
        random.seed(int(rollout_seed))
        np.random.seed(int(rollout_seed))
        torch.manual_seed(int(rollout_seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(rollout_seed))
        algo, callback, vec_env = _build_algo_for_config(
            checkpoint_path=checkpoint_path,
            resolved_config=resolved_config,
            device_obj=device_obj,
            frozen_h=frozen_h,
            train_env_seed=int(train_env_seed),
            rollout_seed=int(rollout_seed),
            single_eval_pos=int(single_eval_pos),
            n_steps=int(n_steps),
            batch_size=int(batch_size),
            learning_rate=float(learning_rate),
        )
        try:
            report = _collect_and_measure_total_grad(algo, callback, vec_env)
            grad_vectors[mode_name] = report.pop("grad_vector")
            reports[mode_name] = report
        finally:
            vec_env.close()
            del algo, callback, vec_env
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    grad_a = grad_vectors["resume_checkpoint_semantics"]
    grad_b = grad_vectors["short_run_aligned_training_semantics"]
    cosine = torch.nn.functional.cosine_similarity(
        grad_a.unsqueeze(0),
        grad_b.unsqueeze(0),
        dim=1,
        eps=1e-8,
    ).item()
    grad_delta = grad_b - grad_a
    return {
        "audit_entry": "phase3_shortrun_aligned_gradient_compare",
        "checkpoint_path": checkpoint_path,
        "device": str(device_obj),
        "frozen_h_seed": int(frozen_h_seed),
        "train_env_seed": int(train_env_seed),
        "rollout_seed": int(rollout_seed),
        "single_eval_pos": int(single_eval_pos),
        "n_steps": int(n_steps),
        "batch_size": int(batch_size),
        "learning_rate": float(learning_rate),
        "training_signature": {
            "resume_checkpoint_semantics": resume_sig,
            "short_run_aligned_training_semantics": shortrun_sig,
            "diff": sig_diff,
            "exact_match": len(sig_diff) == 0,
        },
        "modes": reports,
        "grad_compare": {
            "cosine_similarity": float(cosine),
            "l2_delta_norm": float(torch.linalg.vector_norm(grad_delta).item()),
            "resume_grad_norm": float(torch.linalg.vector_norm(grad_a).item()),
            "short_run_aligned_grad_norm": float(torch.linalg.vector_norm(grad_b).item()),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Gradient compare between checkpoint continue-run semantics and short-run-aligned actor-only/no-aux training semantics.")
    parser.add_argument("checkpoint_path", type=str)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--frozen-h-seed", type=int, default=12345)
    parser.add_argument("--train-env-seed", type=int, default=2020)
    parser.add_argument("--rollout-seed", type=int, default=2020)
    parser.add_argument("--single-eval-pos", type=int, default=1946)
    parser.add_argument("--n-steps", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--output-json", type=str, default=None)
    args = parser.parse_args()

    report = run_phase3_shortrun_aligned_gradient_compare(
        checkpoint_path=args.checkpoint_path,
        device=args.device,
        frozen_h_seed=args.frozen_h_seed,
        train_env_seed=args.train_env_seed,
        rollout_seed=args.rollout_seed,
        single_eval_pos=args.single_eval_pos,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )
    output_path = args.output_json or "/home/chen/RLPFN/artifacts/phase3_shortrun_aligned_gradient_compare.json"
    out = Path(output_path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
