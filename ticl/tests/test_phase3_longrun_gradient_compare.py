import random

import numpy as np
import torch

from ticl.analysis.phase3_longrun_gradient_compare import _fixed_env_contract_summary
from ticl.analysis import fixed_env_h as fixed_env_h_mod


def test_fixed_env_contract_summary_is_stable_for_equivalent_content():
    left = {
        "alpha": 1.0,
        "nested": {
            "vec": [1, 2, 3],
            "latent_u": 0.25,
        },
    }
    right = {
        "nested": {
            "latent_u": 0.25,
            "vec": [1, 2, 3],
        },
        "alpha": 1.0,
    }

    left_summary = _fixed_env_contract_summary(left)
    right_summary = _fixed_env_contract_summary(right)

    assert left_summary["frozen_h_fingerprint"] == right_summary["frozen_h_fingerprint"]
    assert left_summary["frozen_h_leaf_count"] == right_summary["frozen_h_leaf_count"] == 5


def test_fixed_env_contract_summary_detects_value_changes():
    base = {"alpha": 1.0, "nested": {"latent_u": 0.25}}
    changed = {"alpha": 1.0, "nested": {"latent_u": 0.5}}

    base_summary = _fixed_env_contract_summary(base)
    changed_summary = _fixed_env_contract_summary(changed)

    assert base_summary["frozen_h_fingerprint"] != changed_summary["frozen_h_fingerprint"]
    assert "latent_u" in base_summary["top_level_keys_sample"] or "nested" in base_summary["top_level_keys_sample"]


def test_build_fixed_env_h_is_invariant_to_ambient_torch_rng(monkeypatch):
    class _DummyPrior:
        def __init__(self, cfg):
            self.cfg = cfg

        def _sample_batch_hypers(self, batch_size):
            return [
                {
                    "np_draw": float(np.random.random()),
                    "torch_draw": float(torch.rand(()).item()),
                }
                for _ in range(int(batch_size))
            ]

    monkeypatch.setattr(fixed_env_h_mod, "EnvironmentPrior", _DummyPrior)
    monkeypatch.setattr(
        fixed_env_h_mod,
        "_freeze_env_h_list_for_replay",
        lambda prior, h_list: h_list,
    )

    random.seed(111)
    np.random.seed(222)
    torch.manual_seed(333)
    left = fixed_env_h_mod.build_fixed_env_h(prior_cfg={"dummy": True}, frozen_h_seed=12345)

    random.seed(444)
    np.random.seed(555)
    torch.manual_seed(666)
    right = fixed_env_h_mod.build_fixed_env_h(prior_cfg={"dummy": True}, frozen_h_seed=12345)

    assert left == right


def test_build_fixed_env_h_restores_ambient_rng_state(monkeypatch):
    class _DummyPrior:
        def __init__(self, cfg):
            self.cfg = cfg

        def _sample_batch_hypers(self, batch_size):
            return [
                {
                    "np_draw": float(np.random.random()),
                    "torch_draw": float(torch.rand(()).item()),
                }
                for _ in range(int(batch_size))
            ]

    monkeypatch.setattr(fixed_env_h_mod, "EnvironmentPrior", _DummyPrior)
    monkeypatch.setattr(
        fixed_env_h_mod,
        "_freeze_env_h_list_for_replay",
        lambda prior, h_list: h_list,
    )

    random.seed(777)
    np.random.seed(888)
    torch.manual_seed(999)
    py_state = random.getstate()
    np_state = np.random.get_state()
    torch_state = torch.get_rng_state()

    fixed_env_h_mod.build_fixed_env_h(prior_cfg={"dummy": True}, frozen_h_seed=12345)

    assert random.getstate() == py_state
    np_after = np.random.get_state()
    assert np_after[0] == np_state[0]
    assert np_after[2:] == np_state[2:]
    assert np.array_equal(np_after[1], np_state[1])
    assert torch.equal(torch.get_rng_state(), torch_state)
