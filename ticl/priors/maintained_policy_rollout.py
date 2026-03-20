import math

from ticl.distributions import sample_distributions
from ticl.priors.maintained_exact_scm import coerce_bool, resolve_scalar


def normalize_policy_objective_kind(policy_objective_kind):
    objective_kind = str(policy_objective_kind).strip().lower()
    if objective_kind not in {"policy_gradient", "first_policy_gradient", "reinforce", "alpha_grad"}:
        raise ValueError(f"Unknown policy objective kind: {policy_objective_kind}")
    return objective_kind


def policy_rollout_objective_flags(policy_objective_kind):
    objective_kind = normalize_policy_objective_kind(policy_objective_kind)
    return {
        "objective_kind": objective_kind,
        "sample_action": objective_kind in {"first_policy_gradient", "reinforce", "alpha_grad"},
        "collect_log_probs": objective_kind in {"reinforce", "alpha_grad"},
        "detach_action_in_env": objective_kind == "reinforce",
        "first_policy_gradient": objective_kind == "first_policy_gradient",
        "reinforce": objective_kind == "reinforce",
        "alpha_grad": objective_kind == "alpha_grad",
    }


def resolve_alpha_grad_variance_eps(h):
    v = resolve_scalar(h.get("alpha_grad_variance_eps", 1e-6))
    return max(float(v), 0.0)


def resolve_alpha_grad_local_coordinate_enabled(h):
    return coerce_bool(h.get("alpha_grad_local_coordinate_enabled", True))


def resolve_alpha_grad_unit_grad_enabled(h):
    return coerce_bool(h.get("alpha_grad_unit_grad_enabled", True))


def resolve_alpha_grad_unit_grad_delta(h):
    v = resolve_scalar(h.get("alpha_grad_unit_grad_delta", 1e-6))
    if not math.isfinite(v) or v < 0.0:
        return 1e-6
    return float(v)


def resolve_pg_one_hop_replay_enabled(h):
    if "pg_one_hop_replay_enabled" in h:
        return coerce_bool(h.get("pg_one_hop_replay_enabled", True))
    return coerce_bool(h.get("alpha_grad_one_hop_replay_enabled", True))


def resolve_pg_replay_window_depth(h):
    del h
    return 1


def resolve_pg_markov_adjacent_replay_enabled(h):
    return coerce_bool(h.get("pg_markov_adjacent_replay_enabled", False))


def resolve_pg_markov_adjacent_replay_sample_prob(h):
    v = resolve_scalar(h.get("pg_markov_adjacent_replay_sample_prob", 0.125))
    if not math.isfinite(v):
        return 0.125
    return float(min(1.0, max(0.0, v)))


def resolve_alpha_grad_one_hop_replay_enabled(h):
    return resolve_pg_one_hop_replay_enabled(h)


def resolve_batch_parallel_workers(config, batch_size):
    workers_cfg = config.get("batch_parallel_workers", 1)
    if isinstance(workers_cfg, dict) and "distribution" in workers_cfg:
        workers = int(sample_distributions({"v": workers_cfg})["v"])
    else:
        workers = int(workers_cfg)
    workers = max(1, workers)
    return min(int(batch_size), workers)


def resolve_batch_parallel_backend(config):
    backend_cfg = config.get("batch_parallel_backend", "python_thread")
    if isinstance(backend_cfg, dict) and "distribution" in backend_cfg:
        backend = sample_distributions({"v": backend_cfg})["v"]
    else:
        backend = backend_cfg
    backend = str(backend).strip().lower()
    if backend not in {"python_thread", "torch_vectorized"}:
        backend = "python_thread"
    return backend


def resolve_batch_shared_environment(config):
    shared_cfg = config.get("batch_shared_environment", False)
    if isinstance(shared_cfg, dict) and "distribution" in shared_cfg:
        shared = sample_distributions({"v": shared_cfg})["v"]
    else:
        shared = shared_cfg
    if isinstance(shared, str):
        token = shared.strip().lower()
        if token in {"1", "true", "yes", "on"}:
            return True
        if token in {"0", "false", "no", "off"}:
            return False
    return bool(shared)


def resolve_batch_vectorized_strict_rng_match(config):
    strict_cfg = config.get("batch_vectorized_strict_rng_match", False)
    if isinstance(strict_cfg, dict) and "distribution" in strict_cfg:
        strict = sample_distributions({"v": strict_cfg})["v"]
    else:
        strict = strict_cfg
    if isinstance(strict, str):
        token = strict.strip().lower()
        if token in {"1", "true", "yes", "on"}:
            return True
        if token in {"0", "false", "no", "off"}:
            return False
    return bool(strict)


def resolve_batch_vectorized_grouping(config):
    grouping_cfg = config.get("batch_vectorized_grouping", "structure")
    if isinstance(grouping_cfg, dict) and "distribution" in grouping_cfg:
        grouping = sample_distributions({"v": grouping_cfg})["v"]
    else:
        grouping = grouping_cfg
    grouping = str(grouping).strip().lower()
    if grouping not in {"structure", "family"}:
        grouping = "structure"
    return grouping
