import torch

from ticl.analysis.critic_free_single_env_audit import (
    _build_audit_ppo_policy_step_fn,
    _masked_corrcoef_np,
)


class _DummyPolicy:
    def __init__(self):
        self.calls = []

    def make_vectorized_rollout_step_fn(self):
        def _step_fn(obs_t, action_t, reward_t, reward_mask_t, cache, step_idx, env_info):
            self.calls.append(
                {
                    "obs_t": obs_t.clone(),
                    "action_t": action_t.clone(),
                    "reward_t": reward_t.clone(),
                    "reward_mask_t": reward_mask_t.clone(),
                    "cache": cache,
                    "step_idx": int(step_idx),
                    "terminal_t": env_info.get("terminal_t"),
                }
            )
            return {"ok": True}

        return _step_fn


def test_build_audit_ppo_policy_step_fn_clear_first_eval_reward_terminal_history_only():
    policy = _DummyPolicy()
    step_fn = _build_audit_ppo_policy_step_fn(
        policy,
        sampled=True,
        single_eval_pos=4,
        boundary_contract_mode="clear_first_eval_reward_terminal_history",
    )

    obs_t = torch.ones(2, 3)
    action_t = torch.full((2, 2), 7.0)
    reward_t = torch.full((2,), 5.0)
    reward_mask_t = torch.ones(2)
    terminal_t = torch.ones(2)

    step_fn(
        obs_t,
        action_t,
        reward_t,
        reward_mask_t,
        cache={"k": 1},
        step_idx=4,
        env_info={"action_dim": 2, "terminal_t": terminal_t},
    )

    call = policy.calls[-1]
    assert torch.allclose(call["action_t"], action_t)
    assert torch.allclose(call["reward_t"], torch.zeros_like(reward_t))
    assert torch.allclose(call["reward_mask_t"], torch.zeros_like(reward_mask_t))
    assert torch.allclose(call["terminal_t"], torch.zeros_like(terminal_t))
    assert call["cache"] == {"k": 1}


def test_build_audit_ppo_policy_step_fn_state_reset_plus_hidden_keeps_step_history():
    policy = _DummyPolicy()
    step_fn = _build_audit_ppo_policy_step_fn(
        policy,
        sampled=True,
        single_eval_pos=4,
        boundary_contract_mode="reset_env_state_at_sep_and_reset_hidden_keep_actor_history",
    )

    obs_t = torch.ones(2, 3)
    action_t = torch.full((2, 2), 7.0)
    reward_t = torch.full((2,), 5.0)
    reward_mask_t = torch.ones(2)
    terminal_t = torch.ones(2)

    step_fn(
        obs_t,
        action_t,
        reward_t,
        reward_mask_t,
        cache={"k": 1},
        step_idx=4,
        env_info={"action_dim": 2, "terminal_t": terminal_t},
    )

    call = policy.calls[-1]
    assert torch.allclose(call["action_t"], action_t)
    assert torch.allclose(call["reward_t"], reward_t)
    assert torch.allclose(call["reward_mask_t"], reward_mask_t)
    assert torch.allclose(call["terminal_t"], terminal_t)
    assert call["cache"] is None


def test_masked_corrcoef_np_uses_only_masked_entries():
    x = [1.0, 2.0, 100.0]
    y = [2.0, 4.0, -999.0]
    mask = [True, True, False]
    assert _masked_corrcoef_np(x, y, mask) > 0.999
