import argparse
import copy
import json
from pathlib import Path
from typing import Any

import torch

from ticl.analysis.prior_generalization_audit import save_prior_suite, sample_fixed_prior_suite, _summarize_suite
from ticl.analysis.phase3_multi_env_optimization_audit import _default_device
from ticl.analysis.prior_generalization_audit import _resolve_audit_env_config
from ticl.model_builder import load_model
from ticl.priors.environment_prior import EnvironmentPrior


def main() -> int:
    parser = argparse.ArgumentParser(description="Build fixed train/heldout suites for cross-suite probes.")
    parser.add_argument("checkpoint_path", type=str)
    parser.add_argument("--train-suite-seed", type=int, required=True)
    parser.add_argument("--heldout-suite-seed", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device_obj = torch.device(str(args.device or _default_device()))
    checkpoint_path = str(Path(args.checkpoint_path).expanduser().resolve())
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    load_model.cache_clear()
    model, config = load_model(checkpoint_path, device=device_obj, verbose=False)
    env_cfg = _resolve_audit_env_config(config, core_a=False, reference_semantics_enabled=False)
    prior = EnvironmentPrior(copy.deepcopy(env_cfg))

    train_suite = sample_fixed_prior_suite(
        prior,
        batch_size=int(args.batch_size),
        suite_seed=int(args.train_suite_seed),
    )
    heldout_suite = sample_fixed_prior_suite(
        prior,
        batch_size=int(args.batch_size),
        suite_seed=int(args.heldout_suite_seed),
    )

    train_path = save_prior_suite(str(output_dir / "train_suite.pt"), train_suite)
    heldout_path = save_prior_suite(str(output_dir / "heldout_suite.pt"), heldout_suite)

    summary = {
        "checkpoint_path": checkpoint_path,
        "train_suite_seed": int(args.train_suite_seed),
        "heldout_suite_seed": int(args.heldout_suite_seed),
        "batch_size": int(args.batch_size),
        "train_suite_path": train_path,
        "heldout_suite_path": heldout_path,
        "train_suite_summary": _summarize_suite(prior, train_suite),
        "heldout_suite_summary": _summarize_suite(prior, heldout_suite),
    }
    summary_path = output_dir / "suite_config.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(summary_path.read_text())
    _ = model  # keep lint quiet if model unused
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
