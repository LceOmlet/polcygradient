import argparse
import faulthandler
import json
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ticl.analysis import phase3_multi_env_optimization_audit as audit_mod


def _phase_name(kind: str, idx: int) -> str:
    if kind == "evaluate_prior_suite":
        mapping = {
            0: "zero_train",
            1: "zero_heldout",
        }
        return mapping.get(idx, f"{kind}_{idx}")
    if kind == "_run_policy_eval":
        mapping = {
            0: "pre_train",
            1: "pre_heldout",
            2: "post_train",
            3: "post_heldout",
        }
        return mapping.get(idx, f"{kind}_{idx}")
    if kind == "_audit_train_loop":
        return "train_loop"
    return f"{kind}_{idx}"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run Phase 3 many-env optimization audit with phase timing and periodic stack dumps."
    )
    parser.add_argument("checkpoint_path", type=str)
    parser.add_argument("--train-suite-path", type=str, required=True)
    parser.add_argument("--heldout-suite-path", type=str, required=True)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--n-samples", type=int, default=2048)
    parser.add_argument("--single-eval-pos", type=int, default=1946)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--outer-epochs", type=int, default=1)
    parser.add_argument("--n-epochs", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--target-kl", type=float, default=None)
    parser.add_argument("--rollout-backend", type=str, default="serial", choices=["serial", "family_vectorized"])
    parser.add_argument(
        "--train-profile",
        type=str,
        default="trusted_sep_reset_mainline",
        choices=["trusted_sep_reset_mainline", "legacy_actor_only_probe"],
    )
    parser.add_argument("--phase2-summary-path", type=str, required=True)
    parser.add_argument("--reuse-zero-control-json", type=str, default=None)
    parser.add_argument("--reuse-pre-policy-json", type=str, default=None)
    parser.add_argument("--save-post-policy-bundle-path", type=str, default=None)
    parser.add_argument("--resume-post-policy-bundle-path", type=str, default=None)
    parser.add_argument("--output-json", type=str, required=True)
    parser.add_argument("--phase-log-jsonl", type=str, required=True)
    parser.add_argument("--stack-dump-log", type=str, required=True)
    parser.add_argument("--stack-dump-interval-s", type=float, default=60.0)
    parser.add_argument("--monitor-summary-json", type=str, required=True)
    args = parser.parse_args()

    output_path = Path(args.output_json).expanduser().resolve()
    phase_log_path = Path(args.phase_log_jsonl).expanduser().resolve()
    stack_log_path = Path(args.stack_dump_log).expanduser().resolve()
    summary_path = Path(args.monitor_summary_json).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    phase_log_path.parent.mkdir(parents=True, exist_ok=True)
    stack_log_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    phase_events = []
    eval_suite_start_idx = 2 if args.reuse_zero_control_json is not None else 0
    policy_eval_start_idx = 2 if args.reuse_pre_policy_json is not None else 0
    counters = {
        "evaluate_prior_suite": int(eval_suite_start_idx),
        "_run_policy_eval": int(policy_eval_start_idx),
        "_audit_train_loop": 0,
    }

    phase_log_fp = phase_log_path.open("w")
    stack_log_fp = stack_log_path.open("w")
    faulthandler.enable(file=stack_log_fp)
    faulthandler.dump_traceback_later(float(args.stack_dump_interval_s), repeat=True, file=stack_log_fp)

    def _log_event(event: dict) -> None:
        phase_events.append(event)
        phase_log_fp.write(json.dumps(event, sort_keys=True) + "\n")
        phase_log_fp.flush()

    def _wrap(name: str, fn):
        def _inner(*inner_args, **inner_kwargs):
            idx = int(counters[name])
            counters[name] += 1
            phase = _phase_name(name, idx)
            start = time.perf_counter()
            _log_event({"event": "phase_start", "phase": phase, "t": start})
            try:
                result = fn(*inner_args, **inner_kwargs)
            except BaseException as exc:
                end = time.perf_counter()
                _log_event(
                    {
                        "event": "phase_error",
                        "phase": phase,
                        "t": end,
                        "wall_s": float(end - start),
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
                raise
            end = time.perf_counter()
            _log_event({"event": "phase_end", "phase": phase, "t": end, "wall_s": float(end - start)})
            return result

        return _inner

    original_eval = audit_mod.evaluate_prior_suite
    original_run_policy_eval = audit_mod._run_policy_eval
    original_train_loop = audit_mod._audit_train_loop
    audit_mod.evaluate_prior_suite = _wrap("evaluate_prior_suite", audit_mod.evaluate_prior_suite)
    audit_mod._run_policy_eval = _wrap("_run_policy_eval", audit_mod._run_policy_eval)
    audit_mod._audit_train_loop = _wrap("_audit_train_loop", audit_mod._audit_train_loop)

    t0 = time.perf_counter()
    summary = {
        "status": "started",
        "config": {
            "checkpoint_path": str(Path(args.checkpoint_path).expanduser().resolve()),
            "train_suite_path": str(Path(args.train_suite_path).expanduser().resolve()),
            "heldout_suite_path": str(Path(args.heldout_suite_path).expanduser().resolve()),
            "device": args.device,
            "n_samples": int(args.n_samples),
            "single_eval_pos": int(args.single_eval_pos),
            "batch_size": None if args.batch_size is None else int(args.batch_size),
            "outer_epochs": int(args.outer_epochs),
            "n_epochs": None if args.n_epochs is None else int(args.n_epochs),
            "learning_rate": None if args.learning_rate is None else float(args.learning_rate),
            "target_kl": None if args.target_kl is None else float(args.target_kl),
            "rollout_backend": str(args.rollout_backend),
            "train_profile": str(args.train_profile),
            "phase2_summary_path": str(Path(args.phase2_summary_path).expanduser().resolve()),
            "reuse_zero_control_json": None
            if args.reuse_zero_control_json is None
            else str(Path(args.reuse_zero_control_json).expanduser().resolve()),
            "reuse_pre_policy_json": None
            if args.reuse_pre_policy_json is None
            else str(Path(args.reuse_pre_policy_json).expanduser().resolve()),
            "save_post_policy_bundle_path": None
            if args.save_post_policy_bundle_path is None
            else str(Path(args.save_post_policy_bundle_path).expanduser().resolve()),
            "resume_post_policy_bundle_path": None
            if args.resume_post_policy_bundle_path is None
            else str(Path(args.resume_post_policy_bundle_path).expanduser().resolve()),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))

    try:
        report = audit_mod.run_phase3_multi_env_optimization_audit(
            checkpoint_path=args.checkpoint_path,
            train_suite_path=args.train_suite_path,
            heldout_suite_path=args.heldout_suite_path,
            device=args.device,
            n_samples=args.n_samples,
            single_eval_pos=args.single_eval_pos,
            batch_size=args.batch_size,
            outer_epochs=args.outer_epochs,
            n_epochs=args.n_epochs,
            learning_rate=args.learning_rate,
            target_kl=args.target_kl,
            rollout_backend=args.rollout_backend,
            train_profile=args.train_profile,
            phase2_summary_path=args.phase2_summary_path,
            reuse_zero_control_json=args.reuse_zero_control_json,
            reuse_pre_policy_json=args.reuse_pre_policy_json,
            save_post_policy_bundle_path=args.save_post_policy_bundle_path,
            resume_post_policy_bundle_path=args.resume_post_policy_bundle_path,
        )
        output_path.write_text(json.dumps(report, indent=2, sort_keys=True))
        summary = {
            "status": "completed",
            "wall_s": float(time.perf_counter() - t0),
            "output_json": str(output_path),
            "phase_log_jsonl": str(phase_log_path),
            "stack_dump_log": str(stack_log_path),
            "phase_events": phase_events,
        }
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    except KeyboardInterrupt:
        summary = {
            "status": "interrupted",
            "wall_s": float(time.perf_counter() - t0),
            "output_json": str(output_path),
            "phase_log_jsonl": str(phase_log_path),
            "stack_dump_log": str(stack_log_path),
            "phase_events": phase_events,
        }
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
        raise
    except BaseException as exc:
        summary = {
            "status": "error",
            "wall_s": float(time.perf_counter() - t0),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "output_json": str(output_path),
            "phase_log_jsonl": str(phase_log_path),
            "stack_dump_log": str(stack_log_path),
            "phase_events": phase_events,
        }
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
        raise
    finally:
        audit_mod.evaluate_prior_suite = original_eval
        audit_mod._run_policy_eval = original_run_policy_eval
        audit_mod._audit_train_loop = original_train_loop
        faulthandler.cancel_dump_traceback_later()
        stack_log_fp.flush()
        phase_log_fp.flush()
        stack_log_fp.close()
        phase_log_fp.close()


if __name__ == "__main__":
    raise SystemExit(main())
