import argparse
import time
import random
import json
from collections import Counter

import numpy as np
import torch

from ticl.model_configs import get_model_default_config
from ticl.priors.environment_prior import EnvironmentPrior
from ticl.models.encoders import Linear
from ticl.models.tabpfn import TabPFN
from ticl.train import _build_policy_step_fn


def _percentile(values, q):
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def _build_tabpfn_policy_step(model_cfg, n_features, n_samples, device, pg_torch_compile=False):
    tcfg = model_cfg["transformer"]
    model = TabPFN(
        n_out=int(tcfg.get("x_action_dim", 30)),
        n_features=int(n_features),
        emsize=int(tcfg["emsize"]),
        nhead=int(tcfg["nhead"]),
        nhid_factor=int(tcfg["nhid_factor"]),
        nlayers=int(tcfg["nlayers"]),
        dropout=float(tcfg["dropout"]),
        y_encoder_layer=Linear(1, emsize=int(tcfg["emsize"])),
        classification_task=False,
        y_encoder="linear",
        single_eval_causal=bool(tcfg.get("single_eval_causal", True)),
    )
    model = model.to(device)
    model.eval()
    return _build_policy_step_fn(
        model,
        num_features=int(n_features),
        max_cache_len=int(n_samples),
        kv_cache_mode="paged",
        kv_cache_page_size=128,
        allow_grad_mutable_cache=True,
        pg_torch_compile=bool(pg_torch_compile),
        pg_torch_compile_backend="inductor",
        pg_torch_compile_mode="reduce-overhead",
        pg_torch_compile_fullgraph=False,
        pg_torch_compile_dynamic=False,
    )


def run_diagnosis(
    *,
    batch_size,
    n_samples,
    num_features,
    single_eval_pos,
    device,
    policy_mode,
    warmup,
    repeats,
    grouping,
    pg_torch_compile,
    seed,
    summary_json,
):
    if seed is not None:
        seed_i = int(seed)
        random.seed(seed_i)
        np.random.seed(seed_i)
        torch.manual_seed(seed_i)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed_i)

    cfg = get_model_default_config("rlpfn")
    env_cfg = dict(cfg["prior"]["environment"])
    env_cfg["batch_parallel_backend"] = "torch_vectorized"
    env_cfg["batch_shared_environment"] = False
    env_cfg["batch_vectorized_grouping"] = str(grouping)
    env_cfg["batch_vectorized_strict_rng_match"] = False
    prior = EnvironmentPrior(env_cfg)

    sampled_h = []
    orig_sample_h = prior._sample_batch_hypers

    def _sample_h_capture(bs):
        h = orig_sample_h(bs)
        sampled_h.clear()
        sampled_h.extend(h)
        return h

    prior._sample_batch_hypers = _sample_h_capture

    counters = {
        "env_group_sizes": [],
        "x_calls": 0,
        "y_calls": 0,
        "x_wall_time": 0.0,
        "y_wall_time": 0.0,
        "policy_calls": 0,
        "policy_wall_time": 0.0,
        "policy_bs_counter": Counter(),
        "rollout_wall_times": [],
    }

    def _wrap_env_generators(env, group_size):
        counters["env_group_sizes"].append(int(group_size))
        x_gen = env["x_generator"]
        y_gen = env["y_generator"]

        def _x_wrapped(x, generators_for_noise=None):
            t0 = time.perf_counter()
            out = x_gen(x, generators_for_noise=generators_for_noise)
            counters["x_wall_time"] += (time.perf_counter() - t0)
            counters["x_calls"] += 1
            return out

        def _y_wrapped(x, generators_for_noise=None):
            t0 = time.perf_counter()
            out = y_gen(x, generators_for_noise=generators_for_noise)
            counters["y_wall_time"] += (time.perf_counter() - t0)
            counters["y_calls"] += 1
            return out

        env["x_generator"] = _x_wrapped
        env["y_generator"] = _y_wrapped
        return env

    orig_sample_env_batch = prior._sample_environment_batch

    def _sample_env_batch_wrapped(h_list, device, rng_seeds=None):
        env = orig_sample_env_batch(h_list=h_list, device=device, rng_seeds=rng_seeds)
        return _wrap_env_generators(env, len(h_list))

    prior._sample_environment_batch = _sample_env_batch_wrapped

    if hasattr(prior, "_sample_environment_family_coarse_batch"):
        orig_sample_env_coarse = prior._sample_environment_family_coarse_batch

        def _sample_env_coarse_wrapped(h_list, device, rng_seeds=None):
            env = orig_sample_env_coarse(h_list=h_list, device=device, rng_seeds=rng_seeds)
            return _wrap_env_generators(env, len(h_list))

        prior._sample_environment_family_coarse_batch = _sample_env_coarse_wrapped

    if policy_mode == "zero":
        def _policy_step(obs_t, action_t, reward_t, reward_mask_t, cache, step_idx, env_info):
            del obs_t, reward_t, reward_mask_t, cache, step_idx, env_info
            return torch.zeros_like(action_t)
    elif policy_mode == "tabpfn":
        _policy_step = _build_tabpfn_policy_step(
            model_cfg=cfg,
            n_features=num_features,
            n_samples=n_samples,
            device=device,
            pg_torch_compile=pg_torch_compile,
        )
    else:
        raise ValueError(f"Unknown policy_mode={policy_mode}")

    def _policy_wrapped(obs_t, action_t, reward_t, reward_mask_t, cache, step_idx, env_info):
        t0 = time.perf_counter()
        out = _policy_step(obs_t, action_t, reward_t, reward_mask_t, cache, step_idx, env_info)
        counters["policy_wall_time"] += (time.perf_counter() - t0)
        counters["policy_calls"] += 1
        counters["policy_bs_counter"][int(obs_t.shape[0])] += 1
        return out

    if "cuda" in str(device):
        torch.cuda.synchronize(device=device)
    with torch.no_grad():
        for _ in range(int(max(0, warmup))):
            prior.rollout_with_policy(
                policy_step_fn=_policy_wrapped,
                batch_size=int(batch_size),
                n_samples=int(n_samples),
                num_features=int(num_features),
                device=device,
                single_eval_pos=int(single_eval_pos),
                collect_x=False,
            )
        if "cuda" in str(device):
            torch.cuda.synchronize(device=device)

        for _ in range(int(max(1, repeats))):
            if "cuda" in str(device):
                torch.cuda.synchronize(device=device)
            t0 = time.perf_counter()
            rollout = prior.rollout_with_policy(
                policy_step_fn=_policy_wrapped,
                batch_size=int(batch_size),
                n_samples=int(n_samples),
                num_features=int(num_features),
                device=device,
                single_eval_pos=int(single_eval_pos),
                collect_x=False,
            )
            if "cuda" in str(device):
                torch.cuda.synchronize(device=device)
            counters["rollout_wall_times"].append(time.perf_counter() - t0)
            assert rollout["rewards"].shape == (int(n_samples), int(batch_size))

    struct_unique = 0
    struct_max_group = 0
    family_counter = Counter()
    if sampled_h:
        struct_counter = Counter(prior._environment_structure_signature(h) for h in sampled_h)
        struct_unique = int(len(struct_counter))
        struct_max_group = int(max(struct_counter.values()))
        family_counter = Counter(prior._normalize_family(h.get("family", "scm")) for h in sampled_h)

    env_group_sizes = counters["env_group_sizes"]
    rollout_wall_times = counters["rollout_wall_times"]
    x_calls = int(counters["x_calls"])
    y_calls = int(counters["y_calls"])
    policy_calls = int(counters["policy_calls"])
    x_wall = float(counters["x_wall_time"])
    y_wall = float(counters["y_wall_time"])
    policy_wall = float(counters["policy_wall_time"])
    total_rollout_wall = float(sum(rollout_wall_times))
    approx_env_wall = x_wall + y_wall
    approx_other_wall = max(0.0, total_rollout_wall - approx_env_wall - policy_wall)

    summary = {
        "device": str(device),
        "policy_mode": str(policy_mode),
        "grouping": str(grouping),
        "seed": None if seed is None else int(seed),
        "batch_size": int(batch_size),
        "n_samples": int(n_samples),
        "single_eval_pos": int(single_eval_pos),
        "repeats": int(repeats),
        "rollout_wall_sec": {
            "mean": float(np.mean(rollout_wall_times)),
            "p50": _percentile(rollout_wall_times, 50),
            "p90": _percentile(rollout_wall_times, 90),
        },
        "env_structure": {
            "unique": int(struct_unique),
            "max_group": int(struct_max_group),
            "family_counts": dict(family_counter),
        },
        "transition_groups": {
            "total": int(len(env_group_sizes)),
            "mean_group_size": float(np.mean(env_group_sizes) if env_group_sizes else 0.0),
            "p50": _percentile(env_group_sizes, 50),
            "p90": _percentile(env_group_sizes, 90),
        },
        "policy_calls": {
            "total": int(policy_calls),
            "batch_size_hist": dict(sorted(counters["policy_bs_counter"].items())),
            "wall_sec": float(policy_wall),
        },
        "x_generator": {"calls": int(x_calls), "wall_sec": float(x_wall)},
        "y_generator": {"calls": int(y_calls), "wall_sec": float(y_wall)},
        "approx_wall_breakdown": {
            "env_sec": float(approx_env_wall),
            "policy_sec": float(policy_wall),
            "other_sec": float(approx_other_wall),
            "env_pct": float((approx_env_wall / total_rollout_wall * 100.0) if total_rollout_wall > 0 else 0.0),
            "policy_pct": float((policy_wall / total_rollout_wall * 100.0) if total_rollout_wall > 0 else 0.0),
            "other_pct": float((approx_other_wall / total_rollout_wall * 100.0) if total_rollout_wall > 0 else 0.0),
        },
    }

    print("=== Rollout Diagnosis ===")
    print(f"device={device} policy_mode={policy_mode} grouping={grouping} repeats={int(repeats)}")
    print(f"batch_size={int(batch_size)} n_samples={int(n_samples)} single_eval_pos={int(single_eval_pos)}")
    print(
        "rollout_wall_sec:",
        f"mean={summary['rollout_wall_sec']['mean']:.4f}",
        f"p50={summary['rollout_wall_sec']['p50']:.4f}",
        f"p90={summary['rollout_wall_sec']['p90']:.4f}",
    )
    print(
        "env_structure:",
        f"unique={summary['env_structure']['unique']}",
        f"max_group={summary['env_structure']['max_group']}",
        f"family_counts={summary['env_structure']['family_counts']}",
    )
    print(
        "transition_groups(sample_environment_batch calls):",
        f"total={summary['transition_groups']['total']}",
        f"mean_group_size={summary['transition_groups']['mean_group_size']:.3f}",
        f"p50={summary['transition_groups']['p50']:.1f}",
        f"p90={summary['transition_groups']['p90']:.1f}",
    )
    print(
        "policy_calls:",
        f"total={summary['policy_calls']['total']}",
        f"batch_size_hist={summary['policy_calls']['batch_size_hist']}",
        f"wall_sec={summary['policy_calls']['wall_sec']:.4f}",
    )
    print("x_generator:", f"calls={summary['x_generator']['calls']}", f"wall_sec={summary['x_generator']['wall_sec']:.4f}")
    print("y_generator:", f"calls={summary['y_generator']['calls']}", f"wall_sec={summary['y_generator']['wall_sec']:.4f}")
    print("approx_wall_breakdown:")
    print(f"  env(x+y)={summary['approx_wall_breakdown']['env_sec']:.4f}s ({summary['approx_wall_breakdown']['env_pct']:.1f}%)")
    print(f"  policy={summary['approx_wall_breakdown']['policy_sec']:.4f}s ({summary['approx_wall_breakdown']['policy_pct']:.1f}%)")
    print(f"  other={summary['approx_wall_breakdown']['other_sec']:.4f}s ({summary['approx_wall_breakdown']['other_pct']:.1f}%)")

    if summary_json:
        with open(str(summary_json), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, sort_keys=True)


def main():
    parser = argparse.ArgumentParser(description="Diagnose rollout hotspots for rlpfn environment exploration.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-samples", type=int, default=256)
    parser.add_argument("--num-features", type=int, default=432)
    parser.add_argument("--single-eval-pos", type=int, default=128)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--policy-mode", type=str, choices=["zero", "tabpfn"], default="zero")
    parser.add_argument("--grouping", type=str, choices=["family", "structure"], default="family")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--pg-torch-compile", action="store_true")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--summary-json", type=str, default=None)
    args = parser.parse_args()

    run_diagnosis(
        batch_size=int(args.batch_size),
        n_samples=int(args.n_samples),
        num_features=int(args.num_features),
        single_eval_pos=int(args.single_eval_pos),
        device=str(args.device),
        policy_mode=str(args.policy_mode),
        warmup=int(args.warmup),
        repeats=int(args.repeats),
        grouping=str(args.grouping),
        pg_torch_compile=bool(args.pg_torch_compile),
        seed=int(args.seed),
        summary_json=args.summary_json,
    )


if __name__ == "__main__":
    main()
