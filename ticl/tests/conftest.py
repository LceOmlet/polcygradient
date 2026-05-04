import inspect

import pytest


_LEGACY_POLICY_GRADIENT_DISABLED_REASON = (
    "legacy EnvironmentPrior policy-gradient rollout tests are disabled; "
    "maintained RLPFN trust is now anchored on sampled-topology pack PPO parity "
    "and the canonical pack runner smoke path"
)

_LEGACY_POLICY_GRADIENT_ALLOWLIST = {
    "test_legacy_environment_policy_gradient_loss_is_disabled",
}


def pytest_collection_modifyitems(config, items):
    del config
    skip_legacy_pg = pytest.mark.skip(reason=_LEGACY_POLICY_GRADIENT_DISABLED_REASON)
    for item in items:
        if item.name in _LEGACY_POLICY_GRADIENT_ALLOWLIST:
            continue
        obj = getattr(item, "obj", None)
        if obj is None:
            continue
        try:
            source = inspect.getsource(obj)
        except (OSError, TypeError):
            continue
        if "rollout_policy_gradient_loss(" in source or "train_epoch_policy_gradient(" in source:
            item.add_marker(skip_legacy_pg)
