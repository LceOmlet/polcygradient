import copy
import hashlib
import json
from typing import Any

import numpy as np
import torch

from ticl.analysis.prior_generalization_audit import _temporary_seed
from ticl.priors.environment_prior import EnvironmentPrior
from ticl.train import _freeze_env_h_list_for_replay


def _to_builtin(value: Any):
    if torch.is_tensor(value):
        if value.ndim == 0:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [_to_builtin(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _to_builtin(v) for k, v in value.items()}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _leaf_count(value: Any) -> int:
    if isinstance(value, dict):
        if not value:
            return 1
        return int(sum(_leaf_count(v) for v in value.values()))
    if isinstance(value, list):
        if not value:
            return 1
        return int(sum(_leaf_count(v) for v in value))
    return 1


def summarize_fixed_env_h(frozen_h: Any) -> dict[str, Any]:
    frozen_h_builtin = _to_builtin(frozen_h)
    payload = json.dumps(frozen_h_builtin, sort_keys=True, ensure_ascii=True)
    top_level_keys = (
        sorted(str(k) for k in frozen_h_builtin.keys()) if isinstance(frozen_h_builtin, dict) else []
    )
    return {
        "frozen_h_fingerprint": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
        "frozen_h_leaf_count": int(_leaf_count(frozen_h_builtin)),
        "top_level_key_count": int(len(top_level_keys)),
        "top_level_keys_sample": top_level_keys[:16],
        "latent_uniform_keys_sample": [k for k in top_level_keys if k.endswith("_u")][:16],
    }


def sample_fixed_env_h_from_prior(*, prior, frozen_h_seed: int, seed_mode: str = "current"):
    seed_mode_resolved = str(seed_mode).strip().lower()
    if seed_mode_resolved == "current":
        with _temporary_seed(int(frozen_h_seed)):
            frozen_h = _freeze_env_h_list_for_replay(
                prior,
                list(prior._sample_batch_hypers(1)),
            )[0]
        return frozen_h
    if seed_mode_resolved == "legacy_numpy_only":
        numpy_state = np.random.get_state()
        np.random.seed(int(frozen_h_seed))
        try:
            frozen_h = _freeze_env_h_list_for_replay(
                prior,
                list(prior._sample_batch_hypers(1)),
            )[0]
        finally:
            np.random.set_state(numpy_state)
        return frozen_h
    raise ValueError(f"Unsupported fixed env seed mode: {seed_mode}")


def build_fixed_env_h(*, prior_cfg: dict[str, Any], frozen_h_seed: int, seed_mode: str = "current"):
    prior = EnvironmentPrior(copy.deepcopy(prior_cfg))
    return sample_fixed_env_h_from_prior(
        prior=prior,
        frozen_h_seed=int(frozen_h_seed),
        seed_mode=str(seed_mode),
    )
