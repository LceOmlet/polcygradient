import io
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from gymnasium import spaces

import ticl.sb3_recurrent_ppo as sb3_recurrent_ppo_module
from ticl.model_configs import get_model_default_config
from ticl.models.tabpfn_bar_distribution import make_standardized_full_support_bar_distribution
from ticl.priors.environment_prior import EnvironmentPrior
from ticl.rlpfn_maintained_path import resolve_rlpfn_token_layout
from ticl.sb3_recurrent_ppo import (
    EnvironmentPriorPPOGymEnv,
    CROSS_ROLLOUT_RUNNING_EMA_RMS_ALPHA,
    CROSS_ROLLOUT_RUNNING_EMA_RMS_SPACE,
    MaskedRecurrentPPO,
    MaskedRecurrentFlatBatchSamples,
    MaskedRecurrentRolloutBuffer,
    MaskedRecurrentRolloutBufferSamples,
    OfficialRWKVRecurrentPPOPolicy,
    RNNStates,
    ENV12_MID_EPISODE_EXTENSION_BLOCK_SCALE,
    _apply_actor_objective_postprocess,
    _discounted_returns_from_rewards,
    _explained_variance_with_mask,
    _recover_raw_from_value_space,
    _sampled_array_summary,
    _normalize_advantages_with_mask,
    _resolve_actor_advantages,
    _resolve_runtime_scoped_actor_objective_mode,
    _rollout_return_norm_stats_from_raw_returns,
    _slice_flat_sequence_batch_to_padded,
    _slice_masked_rollout_sequence_batch,
    _slice_padded_sequence_tensor,
    build_rlpfn_obs_token_batch_from_components,
    extract_validation_recurrent_ppo_policy_state,
    build_validation_recurrent_ppo_policy,
    build_recurrent_ppo,
    resolve_official_ppo_update_batch_size,
)
from ticl.train import _resolve_official_ppo_rollout_shape, train_epoch_official_recurrent_ppo
from ticl.utils import make_training_callback


def _cuda_or_skip():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for strict official RWKV PPO tests.")
    return torch.device("cuda")


def test_sampled_array_summary_does_not_alias_single_feature_column():
    values = np.zeros((64, 64, 100), dtype=np.float32)
    values[:, :, 1] = 1.0

    summary = _sampled_array_summary(values, prefix="obs", sample_size=4096)

    assert summary["obs_sample_absmax"] == pytest.approx(1.0)
    assert summary["obs_sample_std"] > 0.0


def test_rlpfn_obs_token_builder_uses_shared_rollout_schema():
    obs_t = torch.tensor(
        [
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
        ],
        dtype=torch.float32,
    )
    action_t = torch.tensor(
        [
            [0.1, 0.2, 0.3],
            [-0.1, -0.2, -0.3],
        ],
        dtype=torch.float32,
    )
    reward_t = torch.tensor([9.0, -9.0], dtype=torch.float32)
    reward_mask_t = torch.tensor([1.0, 0.0], dtype=torch.float32)
    env_info = {
        "obs_slot_dim": torch.tensor([4, 3], dtype=torch.long),
        "action_slot_dim": torch.tensor([3, 2], dtype=torch.long),
        "action_dim_per_sample": torch.tensor([2, 3], dtype=torch.long),
        "terminal_reset_enabled": torch.tensor([False, True], dtype=torch.bool),
        "obs_dim": torch.tensor([4, 2], dtype=torch.long),
        "phase_t": torch.tensor([0.0, 1.0], dtype=torch.float32),
        "terminal_t": torch.tensor([0.0, 1.0], dtype=torch.float32),
    }

    tokens = build_rlpfn_obs_token_batch_from_components(
        obs_t=obs_t,
        action_t=action_t,
        reward_t=reward_t,
        reward_mask_t=reward_mask_t,
        env_info=env_info,
        num_features=12,
    )

    assert tokens.shape == (2, 12)
    assert torch.allclose(tokens[0, :4], torch.tensor([1.0, 2.0, 3.0, 4.0]))
    assert tokens[0, 4].item() == pytest.approx(9.0)
    assert tokens[0, 5].item() == pytest.approx(1.0)
    assert tokens[0, 6].item() == pytest.approx(0.0)
    assert torch.allclose(tokens[0, 7:9], torch.tensor([0.1, 0.2]))
    assert tokens[0, 9].item() == pytest.approx(0.0)
    assert torch.allclose(tokens[1, :3], torch.tensor([5.0, 6.0, 0.0]))
    assert tokens[1, 3].item() == pytest.approx(-9.0)
    assert tokens[1, 4].item() == pytest.approx(0.0)
    assert tokens[1, 5].item() == pytest.approx(1.0)
    assert tokens[1, 6].item() == pytest.approx(1.0)
    assert torch.allclose(tokens[1, 7:9], torch.tensor([-0.1, -0.2]))
    assert tokens[1, 9].item() == pytest.approx(0.0)


class _FakeOfficialCore(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.step_calls = 0
        self.sequence_calls = 0
        self.last_step_batch = None

    def init_state(self, batch_size: int, *, device: torch.device, dtype: torch.dtype):
        return [
            (
                torch.zeros((batch_size, 2), device=device, dtype=dtype),
                torch.zeros((batch_size, 2, 2, 2), device=device, dtype=torch.float32),
                torch.zeros((batch_size, 2), device=device, dtype=dtype),
            )
        ]

    def forward_step(self, token, state=None):
        self.step_calls += 1
        self.last_step_batch = int(token.shape[0])
        hidden = token + 0.5
        next_state = []
        for att_x_prev, att_kv, ffn_x_prev in state:
            next_state.append(
                (
                    att_x_prev + 1.0,
                    att_kv + 1.0,
                    ffn_x_prev + 1.0,
                )
            )
        return hidden, next_state

    def forward_tokens_sequence_only(self, tokens):
        self.sequence_calls += 1
        return tokens + 0.25


class _FakeRWKVModel(torch.nn.Module):
    def __init__(
        self,
        *,
        num_features: int,
        x_obs_dim: int,
        action_dim: int,
        emsize: int = 16,
        replay_batch_chunk_size: int | None = None,
        normalized_q_head: bool = False,
        next_state_flow_dim: int | None = None,
    ):
        super().__init__()
        self.emsize = int(emsize)
        self.x_encoder_type = "split_obs_action"
        self.single_eval_causal = True
        self.policy_action_dim = int(action_dim)
        self.encoder = SimpleNamespace(obs_dim=int(x_obs_dim), action_dim=int(action_dim))
        self.token_proj = torch.nn.Linear(int(num_features), self.emsize, bias=False)
        self.rwkv_core = _FakeOfficialCore()
        self.replay_batch_chunk_size = None if replay_batch_chunk_size is None else int(replay_batch_chunk_size)
        self.replay_chunk_requests = []
        self.encode_sequence_calls = 0
        self.encode_sequence_shapes = []
        self.normalized_q_decode_calls = 0
        self.next_state_flow_decode_calls = 0
        self.last_next_state_flow_hidden_shape = None
        self.normalized_q_value_head = None
        self.normalized_q_value_bardist = None
        self.normalized_q_value_action_proj = None
        if bool(normalized_q_head):
            self.normalized_q_value_bardist = make_standardized_full_support_bar_distribution(
                num_buckets=7,
                value_range=5.0,
            )
            self.normalized_q_value_head = torch.nn.Linear(
                self.emsize,
                self.normalized_q_value_bardist.num_bars,
            )
            self.normalized_q_value_action_proj = torch.nn.Linear(int(action_dim), self.emsize)
        self.next_state_flow_dim = None if next_state_flow_dim in (None, 0, False) else int(next_state_flow_dim)
        self.next_state_flow_head = None
        self.next_state_flow_action_proj = None
        if self.next_state_flow_dim is not None:
            self.next_state_flow_action_proj = torch.nn.Linear(int(action_dim), self.emsize)
            self.next_state_flow_head = torch.nn.Linear(self.emsize + self.next_state_flow_dim + 1, self.next_state_flow_dim)

    def _encode_train_token(self, x_token, y_token):
        del y_token
        return self.token_proj(x_token)

    def encode_train_sequence_tokens(self, x_tokens, y_tokens):
        self.encode_sequence_calls += 1
        self.encode_sequence_shapes.append(tuple(int(x) for x in x_tokens.shape))
        return self._encode_train_token(x_tokens, y_tokens)

    def _cast_token_for_rwkv_core(self, token):
        return token.to(dtype=self.token_proj.weight.dtype)

    def replay_hidden_and_aux_sequence_outputs_from_train_tokens(
        self,
        train_tokens,
        *,
        eval_start: int = 0,
        include_normalized_q_logits: bool = False,
        q_action_query=None,
        flow_action_query=None,
        flow_matching_xt=None,
        flow_matching_t=None,
    ):
        hidden_all = self.rwkv_core.forward_tokens_sequence_only(
            self._cast_token_for_rwkv_core(train_tokens)
        )
        hidden = hidden_all[int(eval_start):]
        outputs = {"hidden": hidden}
        if bool(include_normalized_q_logits):
            outputs["normalized_q_logits"] = self._decode_normalized_q_value_logits(
                hidden,
                action=q_action_query,
            )
        if flow_matching_xt is not None and flow_matching_t is not None:
            outputs["next_state_flow"] = self._decode_next_state_flow(
                hidden,
                flow_matching_xt,
                flow_matching_t,
                action=flow_action_query,
            )
        return outputs

    def resolve_replay_batch_chunk_size(self, *, seq_len: int, total_batch: int):
        self.replay_chunk_requests.append((int(seq_len), int(total_batch)))
        if self.replay_batch_chunk_size is None:
            return int(total_batch)
        return int(max(1, min(int(total_batch), int(self.replay_batch_chunk_size))))

    def has_normalized_q_value_head(self):
        return self.normalized_q_value_head is not None and self.normalized_q_value_bardist is not None

    def has_next_state_flow_head(self):
        return self.next_state_flow_head is not None

    def get_normalized_q_value_bardist(self):
        if not self.has_normalized_q_value_head():
            raise RuntimeError("normalized_q_value_head is not initialized")
        return self.normalized_q_value_bardist

    def _decode_normalized_q_value_logits(self, hidden, *, action=None):
        if not self.has_normalized_q_value_head():
            raise RuntimeError("normalized_q_value_head is not initialized")
        self.normalized_q_decode_calls += 1
        hidden_q = hidden
        if action is not None:
            hidden_q = hidden_q + self.normalized_q_value_action_proj(action.to(dtype=hidden_q.dtype))
        return self.normalized_q_value_head(hidden_q)

    def _decode_next_state_flow(self, hidden, x_t, t, *, action=None):
        if not self.has_next_state_flow_head():
            raise RuntimeError("next_state_flow_head is not initialized")
        self.next_state_flow_decode_calls += 1
        self.last_next_state_flow_hidden_shape = tuple(int(x) for x in hidden.shape)
        hidden_f = hidden
        if action is not None:
            hidden_f = hidden_f + self.next_state_flow_action_proj(action.to(dtype=hidden_f.dtype))
        flow_in = torch.cat([hidden_f, x_t.to(dtype=hidden_f.dtype), t.to(dtype=hidden_f.dtype)], dim=-1)
        flat_in = flow_in.reshape(-1, int(flow_in.shape[-1]))
        flat_out = self.next_state_flow_head(flat_in)
        return flat_out.reshape((*flow_in.shape[:-1], int(flat_out.shape[-1])))


def _build_small_ppo_config():
    cfg = deepcopy(get_model_default_config("rlpfn"))
    cfg["optimizer"]["rl_objective"] = "ppo"
    cfg["prior"]["n_samples"] = 4
    cfg["optimizer"]["ppo_n_envs"] = 2
    cfg["optimizer"]["ppo_n_steps"] = 4
    env_cfg = deepcopy(cfg["prior"]["environment"])
    return cfg, env_cfg


def _strict_env_cfg_for_ppo(env_cfg, *, aux: bool = False):
    env_cfg = deepcopy(env_cfg)
    env_cfg["family"] = {"distribution": "meta_choice", "choice_values": ["scm"]}
    env_cfg["action_dim"] = int(env_cfg["action_slot_dim"])
    env_cfg["state_dim"] = 8
    env_cfg["obs_dim"] = 8
    env_cfg["noise_dim"] = 4
    env_cfg["zero_pad_dim"] = 0
    env_cfg["alpha"] = 0.2
    env_cfg["init_state_std"] = 0.1
    env_cfg["init_action_std"] = 0.1
    env_cfg["state_noise_std"] = 0.01
    env_cfg["action_noise_train_std"] = 0.0
    env_cfg["action_noise_eval_std"] = 0.0
    env_cfg["reward_scale"] = 1.0
    env_cfg["reward_clip"] = 10.0
    env_cfg["reinforce_action_transform"] = "none"
    env_cfg["reinforce_aux_enabled"] = bool(aux)
    env_cfg["normalized_q_value_weight"] = 1.0 if aux else 0.0
    env_cfg["next_state_flow_matching_weight"] = 1.0 if aux else 0.0
    return env_cfg


def _ppo_main_minibatch_loss(algo, rollout_data):
    mask = (rollout_data.mask > 1e-8) & (rollout_data.objective_masks > 1e-8)
    valid_total = mask.sum().clamp_min(1).to(device=rollout_data.returns.device, dtype=rollout_data.returns.dtype)
    advantages = _resolve_actor_advantages(rollout_data)
    if algo.normalize_advantage:
        advantages = _normalize_advantages_with_mask(advantages, mask, eps=1e-8)
    eval_outputs = algo.policy.evaluate_actions_with_hidden(
        rollout_data.observations,
        rollout_data.actions,
        rollout_data.lstm_states,
        rollout_data.episode_starts,
        action_masks=rollout_data.action_masks,
    )
    values = eval_outputs["values"].flatten()
    value_logits = eval_outputs["value_logits"]
    log_prob = eval_outputs["log_prob"]
    entropy = eval_outputs["entropy"]
    ratio = torch.exp(log_prob - rollout_data.old_log_prob)
    clipped_objective = torch.min(
        advantages * ratio,
        advantages * torch.clamp(ratio, 1 - algo.clip_range(1.0), 1 + algo.clip_range(1.0)),
    )
    mask_f = mask.to(device=clipped_objective.device, dtype=clipped_objective.dtype)
    policy_loss = -((clipped_objective * mask_f).sum() / valid_total.to(device=clipped_objective.device, dtype=clipped_objective.dtype))
    value_errors = algo.policy.get_value_bardist()(
        value_logits.to(dtype=torch.float32),
        rollout_data.returns.to(device=value_logits.device, dtype=torch.float32),
    )
    value_loss = (value_errors * mask_f.to(device=value_errors.device, dtype=value_errors.dtype)).sum() / valid_total.to(
        device=value_errors.device,
        dtype=value_errors.dtype,
    )
    entropy_terms = entropy if entropy is not None else -log_prob
    entropy_loss = -(
        (entropy_terms * mask_f.to(device=entropy_terms.device, dtype=entropy_terms.dtype)).sum()
        / valid_total.to(device=entropy_terms.device, dtype=entropy_terms.dtype)
    )
    return policy_loss + algo.ent_coef * entropy_loss + algo.vf_coef * value_loss


def _ppo_main_streamed_flat_minibatch_loss(algo, flat_batch: MaskedRecurrentFlatBatchSamples):
    objective_mask = flat_batch.objective_masks > 1e-8
    valid_total = objective_mask.sum().clamp_min(1).to(device=flat_batch.returns.device, dtype=flat_batch.returns.dtype)
    advantages = _resolve_actor_advantages(flat_batch)
    if algo.normalize_advantage:
        advantages = _normalize_advantages_with_mask(advantages, objective_mask, eps=1e-8)
    total_loss = flat_batch.returns.new_zeros(())
    n_seq_total = int(flat_batch.n_seq)
    seq_subbatch_size = algo._resolve_sequence_subbatch_size(flat_batch)
    for start_seq in range(0, n_seq_total, seq_subbatch_size):
        end_seq = min(n_seq_total, start_seq + seq_subbatch_size)
        flat_start = int(flat_batch.seq_start_indices[int(start_seq)])
        flat_end = (
            int(flat_batch.seq_start_indices[int(end_seq)])
            if int(end_seq) < n_seq_total
            else int(flat_batch.observations.shape[0])
        )
        sub_seq_lengths = np.asarray(flat_batch.seq_lengths[int(start_seq) : int(end_seq)], dtype=np.int64)
        sub_advantages = advantages[flat_start:flat_end]
        sub_actions = flat_batch.actions[flat_start:flat_end]
        sub_old_log_prob = flat_batch.old_log_prob[flat_start:flat_end]
        sub_returns = flat_batch.returns[flat_start:flat_end]
        sub_objective = flat_batch.objective_masks[flat_start:flat_end].to(device=flat_batch.returns.device, dtype=flat_batch.returns.dtype)
        eval_outputs = algo.policy.evaluate_actions_with_hidden_flat(
            flat_batch.observations[flat_start:flat_end],
            sub_actions,
            seq_lengths=sub_seq_lengths,
            action_masks=flat_batch.action_masks[flat_start:flat_end],
        )
        values = eval_outputs["values"].flatten()
        value_logits = eval_outputs["value_logits"]
        log_prob = eval_outputs["log_prob"]
        entropy = eval_outputs["entropy"]
        ratio = torch.exp(log_prob - sub_old_log_prob)
        clipped_objective = torch.min(
            sub_advantages * ratio,
            sub_advantages * torch.clamp(ratio, 1 - algo.clip_range(1.0), 1 + algo.clip_range(1.0)),
        )
        policy_loss = -(
            (clipped_objective * sub_objective.to(device=clipped_objective.device, dtype=clipped_objective.dtype)).sum()
            / valid_total.to(device=clipped_objective.device, dtype=clipped_objective.dtype)
        )
        value_errors = algo.policy.get_value_bardist()(
            value_logits.to(dtype=torch.float32),
            sub_returns.to(device=value_logits.device, dtype=torch.float32),
        )
        value_loss = (
            (value_errors * sub_objective.to(device=value_errors.device, dtype=value_errors.dtype)).sum()
            / valid_total.to(device=value_errors.device, dtype=value_errors.dtype)
        )
        entropy_terms = entropy if entropy is not None else -log_prob
        entropy_loss = -(
            (entropy_terms * sub_objective.to(device=entropy_terms.device, dtype=entropy_terms.dtype)).sum()
            / valid_total.to(device=entropy_terms.device, dtype=entropy_terms.dtype)
        )
        total_loss = total_loss + (policy_loss + algo.ent_coef * entropy_loss + algo.vf_coef * value_loss)
    return total_loss


def _ppo_main_minibatch_loss_legacy_raw_value_targets(algo, rollout_data):
    mask = (rollout_data.mask > 1e-8) & (rollout_data.objective_masks > 1e-8)
    valid_total = mask.sum().clamp_min(1).to(device=rollout_data.returns.device, dtype=rollout_data.returns.dtype)
    advantages = _resolve_actor_advantages(rollout_data)
    if algo.normalize_advantage:
        advantages = _normalize_advantages_with_mask(advantages, mask, eps=1e-8)
    eval_outputs = algo.policy.evaluate_actions_with_hidden(
        rollout_data.observations,
        rollout_data.actions,
        rollout_data.lstm_states,
        rollout_data.episode_starts,
        action_masks=rollout_data.action_masks,
    )
    value_logits = eval_outputs["value_logits"]
    log_prob = eval_outputs["log_prob"]
    entropy = eval_outputs["entropy"]
    ratio = torch.exp(log_prob - rollout_data.old_log_prob)
    clipped_objective = torch.min(
        advantages * ratio,
        advantages * torch.clamp(ratio, 1 - algo.clip_range(1.0), 1 + algo.clip_range(1.0)),
    )
    mask_f = mask.to(device=clipped_objective.device, dtype=clipped_objective.dtype)
    policy_loss = -((clipped_objective * mask_f).sum() / valid_total.to(device=clipped_objective.device, dtype=clipped_objective.dtype))
    value_errors = algo.policy.get_value_bardist()(
        value_logits.to(dtype=torch.float32),
        rollout_data.returns.to(device=value_logits.device, dtype=torch.float32),
    )
    value_loss = (value_errors * mask_f.to(device=value_errors.device, dtype=value_errors.dtype)).sum() / valid_total.to(
        device=value_errors.device,
        dtype=value_errors.dtype,
    )
    entropy_terms = entropy if entropy is not None else -log_prob
    entropy_loss = -(
        (entropy_terms * mask_f.to(device=entropy_terms.device, dtype=entropy_terms.dtype)).sum()
        / valid_total.to(device=entropy_terms.device, dtype=entropy_terms.dtype)
    )
    return policy_loss + algo.ent_coef * entropy_loss + algo.vf_coef * value_loss


def test_explained_variance_with_mask_ignores_prefix_steps():
    predictions = np.asarray([1000.0, -500.0, 1.0, 2.0], dtype=np.float32)
    targets = np.asarray([-1000.0, 500.0, 1.5, 2.5], dtype=np.float32)
    objective_mask = np.asarray([0.0, 0.0, 1.0, 1.0], dtype=np.float32)

    masked_ev = _explained_variance_with_mask(predictions, targets, mask=objective_mask)
    expected_ev = sb3_recurrent_ppo_module.explained_variance(
        predictions[2:],
        targets[2:],
    )
    assert masked_ev == pytest.approx(expected_ev)

    perturbed_prefix_ev = _explained_variance_with_mask(
        np.asarray([1e9, -1e9, 1.0, 2.0], dtype=np.float32),
        np.asarray([-1e9, 1e9, 1.5, 2.5], dtype=np.float32),
        mask=objective_mask,
    )
    assert perturbed_prefix_ev == pytest.approx(masked_ev)


def test_raw_explained_variance_can_stay_high_when_normalized_ev_is_negative():
    normalized_targets = np.asarray(
        [
            [-1.0, -1.0],
            [1.0, 1.0],
            [-1.0, -1.0],
            [1.0, 1.0],
        ],
        dtype=np.float32,
    )
    normalized_predictions = -0.3 * normalized_targets
    rollout_return_means = np.asarray([1000.0, -1000.0], dtype=np.float32)
    rollout_return_stds = np.asarray([1.0, 1.0], dtype=np.float32)

    normalized_ev = _explained_variance_with_mask(
        normalized_predictions.reshape(-1),
        normalized_targets.reshape(-1),
    )
    raw_predictions = _recover_raw_from_value_space(
        torch.as_tensor(normalized_predictions.reshape(-1), dtype=torch.float32),
        value_means=np.repeat(rollout_return_means.reshape((1, -1)), normalized_targets.shape[0], axis=0).reshape(-1),
        value_stds=np.repeat(rollout_return_stds.reshape((1, -1)), normalized_targets.shape[0], axis=0).reshape(-1),
    ).cpu().numpy()
    raw_targets = _recover_raw_from_value_space(
        torch.as_tensor(normalized_targets.reshape(-1), dtype=torch.float32),
        value_means=np.repeat(rollout_return_means.reshape((1, -1)), normalized_targets.shape[0], axis=0).reshape(-1),
        value_stds=np.repeat(rollout_return_stds.reshape((1, -1)), normalized_targets.shape[0], axis=0).reshape(-1),
    ).cpu().numpy()
    raw_ev = _explained_variance_with_mask(raw_predictions, raw_targets)

    assert normalized_ev == pytest.approx(-0.69, abs=1e-6)
    assert raw_ev > 0.999


def test_zero_normalized_value_prediction_is_raw_rollout_mean_predictor():
    normalized_targets = np.asarray(
        [
            [-1.0, -2.0],
            [1.0, 2.0],
            [-1.0, -2.0],
            [1.0, 2.0],
        ],
        dtype=np.float32,
    )
    normalized_predictions = np.zeros_like(normalized_targets)
    rollout_return_means = np.asarray([50.0, -75.0], dtype=np.float32)
    rollout_return_stds = np.asarray([2.0, 4.0], dtype=np.float32)
    means_flat = np.repeat(rollout_return_means.reshape((1, -1)), normalized_targets.shape[0], axis=0).reshape(-1)
    stds_flat = np.repeat(rollout_return_stds.reshape((1, -1)), normalized_targets.shape[0], axis=0).reshape(-1)

    raw_predictions = _recover_raw_from_value_space(
        torch.as_tensor(normalized_predictions.reshape(-1), dtype=torch.float32),
        value_means=means_flat,
        value_stds=stds_flat,
    ).cpu().numpy()
    raw_targets = _recover_raw_from_value_space(
        torch.as_tensor(normalized_targets.reshape(-1), dtype=torch.float32),
        value_means=means_flat,
        value_stds=stds_flat,
    ).cpu().numpy()

    assert np.allclose(raw_predictions, means_flat, atol=1e-6, rtol=1e-6)
    assert _explained_variance_with_mask(normalized_predictions.reshape(-1), normalized_targets.reshape(-1)) == pytest.approx(
        0.0,
        abs=1e-6,
    )
    assert _explained_variance_with_mask(raw_predictions, raw_targets) > 0.99


def test_ppo_rollout_log_accumulator_separates_objective_suffix_from_full_episodes():
    accum = sb3_recurrent_ppo_module._PpoRolloutLogAccumulator(n_envs=2)

    accum.observe_step(
        reward=np.asarray([10.0, 20.0], dtype=np.float32),
        reward_env=np.asarray([7.0, 17.0], dtype=np.float32),
        reward_ctrl=np.asarray([-1.0, -2.0], dtype=np.float32),
        reward_survival=np.asarray([3.0, 4.0], dtype=np.float32),
        reward_terminal_bonus=np.asarray([1.0, 0.0], dtype=np.float32),
        objective_mask=np.asarray([False, False]),
        dones=np.asarray([True, False]),
    )
    accum.observe_step(
        reward=np.asarray([1.0, 2.0], dtype=np.float32),
        reward_env=np.asarray([1.5, 2.5], dtype=np.float32),
        reward_ctrl=np.asarray([-0.1, -0.2], dtype=np.float32),
        reward_survival=np.asarray([0.3, 0.4], dtype=np.float32),
        reward_terminal_bonus=np.asarray([0.0, 0.5], dtype=np.float32),
        objective_mask=np.asarray([True, True]),
        dones=np.asarray([False, True]),
    )
    accum.observe_step(
        reward=np.asarray([3.0, 4.0], dtype=np.float32),
        reward_env=np.asarray([3.5, 4.5], dtype=np.float32),
        reward_ctrl=np.asarray([-0.3, -0.4], dtype=np.float32),
        reward_survival=np.asarray([0.5, 0.6], dtype=np.float32),
        reward_terminal_bonus=np.asarray([0.0, 0.0], dtype=np.float32),
        objective_mask=np.asarray([True, True]),
        dones=np.asarray([False, False]),
    )

    stats = accum.finalize()

    assert stats["reward_mean"] == pytest.approx(2.5)
    assert stats["reward_env_mean"] == pytest.approx(3.0)
    assert stats["reward_ctrl_mean"] == pytest.approx(-0.25)
    assert stats["reward_survival_mean"] == pytest.approx(0.45)
    assert stats["reward_terminal_bonus_mean"] == pytest.approx(0.125)
    assert stats["ep_rew_mean"] == pytest.approx((2.0 + 4.0 + 4.0) / 3.0)
    assert stats["ep_len_mean"] == pytest.approx((1.0 + 2.0 + 1.0) / 3.0)
    assert stats["reward_env_return_mean"] == pytest.approx((2.5 + 5.0 + 4.5) / 3.0)
    assert stats["reward_ctrl_return_mean"] == pytest.approx((-0.2 - 0.4 - 0.4) / 3.0)
    assert stats["reward_survival_return_mean"] == pytest.approx((0.4 + 0.8 + 0.6) / 3.0)
    assert stats["reward_terminal_bonus_return_mean"] == pytest.approx((0.5 + 0.0 + 0.0) / 3.0)
    assert stats["full_ep_rew_mean"] == pytest.approx((10.0 + 22.0) / 2.0)
    assert stats["full_ep_len_mean"] == pytest.approx(1.5)
    assert stats["full_reward_env_return_mean"] == pytest.approx((7.0 + 19.5) / 2.0)
    assert stats["full_reward_ctrl_return_mean"] == pytest.approx((-1.0 - 2.2) / 2.0)


def test_official_ppo_rollout_shape_tracks_env_prior_workload():
    n_envs, n_steps = _resolve_official_ppo_rollout_shape(
        batch_size=128,
        n_samples=512,
        configured_n_envs=None,
        configured_n_steps=None,
    )
    assert n_envs == 128
    assert n_steps == 512

    with pytest.raises(ValueError, match="must match the existing EnvironmentPrior batch_size"):
        _resolve_official_ppo_rollout_shape(
            batch_size=128,
            n_samples=512,
            configured_n_envs=64,
            configured_n_steps=None,
        )

    with pytest.raises(ValueError, match="must match the existing EnvironmentPrior n_samples"):
        _resolve_official_ppo_rollout_shape(
            batch_size=128,
            n_samples=512,
            configured_n_envs=None,
            configured_n_steps=64,
        )


def test_build_recurrent_ppo_accepts_variable_action_dim_within_slot_dim():
    device = _cuda_or_skip()
    cfg, env_cfg = _build_small_ppo_config()
    env_cfg = _strict_env_cfg_for_ppo(env_cfg)
    env_cfg["action_dim"] = {"distribution": "uniform_int", "min": 1, "max": 3}
    prior = EnvironmentPrior(env_cfg)
    fake_model = _FakeRWKVModel(
        num_features=cfg["prior"]["num_features"],
        x_obs_dim=cfg["transformer"]["x_obs_dim"],
        action_dim=cfg["transformer"]["x_action_dim"],
    ).to(device)
    algo, _, vec_env = build_recurrent_ppo(
        model=fake_model,
        env_prior=prior,
        device=str(device),
        num_features=cfg["prior"]["num_features"],
        n_envs=cfg["optimizer"]["ppo_n_envs"],
        n_steps=cfg["optimizer"]["ppo_n_steps"],
        learning_rate=cfg["optimizer"]["learning_rate"],
        batch_size=cfg["optimizer"]["ppo_batch_size"],
        n_epochs=cfg["optimizer"]["ppo_n_epochs"],
        gamma=cfg["optimizer"]["ppo_gamma"],
        gae_lambda=cfg["optimizer"]["ppo_gae_lambda"],
        clip_range=cfg["optimizer"]["ppo_clip_range"],
        clip_range_vf=cfg["optimizer"]["ppo_clip_range_vf"],
        normalize_advantage=cfg["optimizer"]["ppo_normalize_advantage"],
        ent_coef=cfg["optimizer"]["ppo_ent_coef"],
        vf_coef=cfg["optimizer"]["ppo_vf_coef"],
        max_grad_norm=cfg["optimizer"]["ppo_max_grad_norm"],
        target_kl=cfg["optimizer"]["ppo_target_kl"],
    )
    vec_env.reset()
    masks = np.stack(vec_env.env_method("action_masks"))
    assert masks.shape == (cfg["optimizer"]["ppo_n_envs"], cfg["transformer"]["x_action_dim"])
    assert np.all((masks == 0.0) | (masks == 1.0))
    assert np.any(masks.sum(axis=1) < cfg["transformer"]["x_action_dim"])
    assert isinstance(algo.policy, OfficialRWKVRecurrentPPOPolicy)
    vec_env.close()


def test_build_recurrent_ppo_rejects_action_dim_upper_bound_above_slot_dim():
    cfg, env_cfg = _build_small_ppo_config()
    env_cfg = _strict_env_cfg_for_ppo(env_cfg)
    env_cfg["action_dim"] = {"distribution": "uniform_int", "min": 1, "max": int(env_cfg["action_slot_dim"]) + 1}
    prior = EnvironmentPrior(env_cfg)
    fake_model = _FakeRWKVModel(
        num_features=cfg["prior"]["num_features"],
        x_obs_dim=cfg["transformer"]["x_obs_dim"],
        action_dim=cfg["transformer"]["x_action_dim"],
    )
    with pytest.raises(ValueError, match="upper bound"):
        build_recurrent_ppo(
            model=fake_model,
            env_prior=prior,
            device="cuda",
            num_features=cfg["prior"]["num_features"],
            n_envs=cfg["optimizer"]["ppo_n_envs"],
            n_steps=cfg["optimizer"]["ppo_n_steps"],
            learning_rate=cfg["optimizer"]["learning_rate"],
            batch_size=cfg["optimizer"]["ppo_batch_size"],
            n_epochs=cfg["optimizer"]["ppo_n_epochs"],
            gamma=cfg["optimizer"]["ppo_gamma"],
            gae_lambda=cfg["optimizer"]["ppo_gae_lambda"],
            clip_range=cfg["optimizer"]["ppo_clip_range"],
            clip_range_vf=cfg["optimizer"]["ppo_clip_range_vf"],
            normalize_advantage=cfg["optimizer"]["ppo_normalize_advantage"],
            ent_coef=cfg["optimizer"]["ppo_ent_coef"],
            vf_coef=cfg["optimizer"]["ppo_vf_coef"],
            max_grad_norm=cfg["optimizer"]["ppo_max_grad_norm"],
            target_kl=cfg["optimizer"]["ppo_target_kl"],
        )


def test_build_recurrent_ppo_smoke_constructs_official_algo():
    device = _cuda_or_skip()
    cfg, env_cfg = _build_small_ppo_config()
    env_cfg = _strict_env_cfg_for_ppo(env_cfg)
    prior = EnvironmentPrior(env_cfg)
    fake_model = _FakeRWKVModel(
        num_features=cfg["prior"]["num_features"],
        x_obs_dim=cfg["transformer"]["x_obs_dim"],
        action_dim=cfg["transformer"]["x_action_dim"],
    ).to(device)

    algo, callback, vec_env = build_recurrent_ppo(
        model=fake_model,
        env_prior=prior,
        device=str(device),
        num_features=cfg["prior"]["num_features"],
        n_envs=cfg["optimizer"]["ppo_n_envs"],
        n_steps=cfg["optimizer"]["ppo_n_steps"],
        learning_rate=cfg["optimizer"]["learning_rate"],
        batch_size=cfg["optimizer"]["ppo_batch_size"],
        n_epochs=cfg["optimizer"]["ppo_n_epochs"],
        gamma=cfg["optimizer"]["ppo_gamma"],
        gae_lambda=cfg["optimizer"]["ppo_gae_lambda"],
        clip_range=cfg["optimizer"]["ppo_clip_range"],
        clip_range_vf=cfg["optimizer"]["ppo_clip_range_vf"],
        normalize_advantage=cfg["optimizer"]["ppo_normalize_advantage"],
        ent_coef=cfg["optimizer"]["ppo_ent_coef"],
        vf_coef=cfg["optimizer"]["ppo_vf_coef"],
        max_grad_norm=cfg["optimizer"]["ppo_max_grad_norm"],
        target_kl=cfg["optimizer"]["ppo_target_kl"],
    )
    obs = vec_env.reset()
    assert isinstance(algo.policy, OfficialRWKVRecurrentPPOPolicy)
    assert callback is not None
    assert obs.shape == (cfg["optimizer"]["ppo_n_envs"], cfg["prior"]["num_features"])
    assert int(algo.batch_size) == 8
    assert vec_env._rollout_generators is None
    vec_env.close()


def test_build_recurrent_ppo_default_batch_size_uses_rwkv_sequence_capacity():
    device = _cuda_or_skip()
    cfg, env_cfg = _build_small_ppo_config()
    env_cfg = _strict_env_cfg_for_ppo(env_cfg)
    cfg["optimizer"]["ppo_n_envs"] = 6
    cfg["optimizer"]["ppo_n_steps"] = 4
    prior = EnvironmentPrior(env_cfg)
    fake_model = _FakeRWKVModel(
        num_features=cfg["prior"]["num_features"],
        x_obs_dim=cfg["transformer"]["x_obs_dim"],
        action_dim=cfg["transformer"]["x_action_dim"],
        replay_batch_chunk_size=4,
    ).to(device)

    algo, _, vec_env = build_recurrent_ppo(
        model=fake_model,
        env_prior=prior,
        device=str(device),
        num_features=cfg["prior"]["num_features"],
        n_envs=cfg["optimizer"]["ppo_n_envs"],
        n_steps=cfg["optimizer"]["ppo_n_steps"],
        learning_rate=cfg["optimizer"]["learning_rate"],
        batch_size=cfg["optimizer"]["ppo_batch_size"],
        n_epochs=cfg["optimizer"]["ppo_n_epochs"],
        gamma=cfg["optimizer"]["ppo_gamma"],
        gae_lambda=cfg["optimizer"]["ppo_gae_lambda"],
        clip_range=cfg["optimizer"]["ppo_clip_range"],
        clip_range_vf=cfg["optimizer"]["ppo_clip_range_vf"],
        normalize_advantage=cfg["optimizer"]["ppo_normalize_advantage"],
        ent_coef=cfg["optimizer"]["ppo_ent_coef"],
        vf_coef=cfg["optimizer"]["ppo_vf_coef"],
        max_grad_norm=cfg["optimizer"]["ppo_max_grad_norm"],
        target_kl=cfg["optimizer"]["ppo_target_kl"],
    )
    assert int(algo.batch_size) == 16
    vec_env.close()


def test_policy_rollout_forward_uses_single_batched_step_call():
    device = _cuda_or_skip()
    num_features = 8
    obs_slot_dim = 4
    action_dim = 2
    fake_model = _FakeRWKVModel(
        num_features=num_features,
        x_obs_dim=6,
        action_dim=action_dim,
        emsize=12,
        normalized_q_head=True,
    ).to(device)
    policy = OfficialRWKVRecurrentPPOPolicy(
        observation_space=spaces.Box(low=-np.inf, high=np.inf, shape=(num_features,), dtype=np.float32),
        action_space=spaces.Box(low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32),
        lr_schedule=lambda _: 1e-3,
        rlpfn_model=fake_model,
        num_features=num_features,
        obs_slot_dim=obs_slot_dim,
        net_arch=[],
    ).to(device)
    policy.eval()
    obs = torch.randn((3, num_features), device=device, dtype=torch.float32)
    episode_starts = torch.tensor([1.0, 0.0, 1.0], device=device, dtype=torch.float32)
    with torch.no_grad():
        actions, values, log_prob, lstm_states = policy.forward(
            obs,
            policy._dummy_states(3),
            episode_starts,
        )
    assert actions.shape == (3, action_dim)
    assert values.shape == (3, 1)
    assert log_prob.shape == (3,)
    assert int(lstm_states.pi[0].shape[1]) == 3
    assert fake_model.rwkv_core.step_calls == 1
    assert fake_model.rwkv_core.last_step_batch == 3
    assert fake_model.rwkv_core.sequence_calls == 0


def test_policy_evaluate_actions_with_hidden_uses_single_sequence_replay_for_shared_backbone():
    device = _cuda_or_skip()
    num_features = 8
    obs_slot_dim = 4
    action_dim = 2
    fake_model = _FakeRWKVModel(
        num_features=num_features,
        x_obs_dim=6,
        action_dim=action_dim,
        emsize=12,
    ).to(device)
    policy = OfficialRWKVRecurrentPPOPolicy(
        observation_space=spaces.Box(low=-np.inf, high=np.inf, shape=(num_features,), dtype=np.float32),
        action_space=spaces.Box(low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32),
        lr_schedule=lambda _: 1e-3,
        rlpfn_model=fake_model,
        num_features=num_features,
        obs_slot_dim=obs_slot_dim,
        net_arch=[],
    ).to(device)
    policy.eval()
    obs = torch.randn((6, num_features), device=device, dtype=torch.float32)
    actions = torch.randn((6, action_dim), device=device, dtype=torch.float32)
    episode_starts = torch.tensor([1.0, 0.0, 0.0, 1.0, 0.0, 0.0], device=device, dtype=torch.float32)
    with torch.no_grad():
        outputs = policy.evaluate_actions_with_hidden(
            obs,
            actions,
            policy._dummy_states(2),
            episode_starts,
        )
    assert outputs["values"].shape == (6, 1)
    assert outputs["log_prob"].shape == (6,)
    assert outputs["entropy"].shape == (6,)
    assert fake_model.rwkv_core.sequence_calls == 1
    assert fake_model.rwkv_core.step_calls == 0


def test_policy_evaluate_actions_with_hidden_flat_uses_single_sequence_replay_for_shared_backbone():
    device = _cuda_or_skip()
    num_features = 8
    obs_slot_dim = 4
    action_dim = 2
    fake_model = _FakeRWKVModel(
        num_features=num_features,
        x_obs_dim=6,
        action_dim=action_dim,
        emsize=12,
    ).to(device)
    policy = OfficialRWKVRecurrentPPOPolicy(
        observation_space=spaces.Box(low=-np.inf, high=np.inf, shape=(num_features,), dtype=np.float32),
        action_space=spaces.Box(low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32),
        lr_schedule=lambda _: 1e-3,
        rlpfn_model=fake_model,
        num_features=num_features,
        obs_slot_dim=obs_slot_dim,
        net_arch=[],
    ).to(device)
    policy.eval()
    obs = torch.randn((6, num_features), device=device, dtype=torch.float32)
    actions = torch.randn((6, action_dim), device=device, dtype=torch.float32)
    with torch.no_grad():
        outputs = policy.evaluate_actions_with_hidden_flat(
            obs,
            actions,
            seq_lengths=np.array([3, 3], dtype=np.int64),
        )
    assert outputs["values"].shape == (6, 1)
    assert outputs["log_prob"].shape == (6,)
    assert outputs["entropy"].shape == (6,)
    assert fake_model.rwkv_core.sequence_calls == 1
    assert fake_model.rwkv_core.step_calls == 0


def test_policy_vectorized_rollout_step_truncates_action_stats_to_batch_max_action_dim():
    num_features = 8
    obs_slot_dim = 4
    action_dim = 4
    fake_model = _FakeRWKVModel(
        num_features=num_features,
        x_obs_dim=6,
        action_dim=action_dim,
        emsize=12,
    )
    policy = OfficialRWKVRecurrentPPOPolicy(
        observation_space=spaces.Box(low=-np.inf, high=np.inf, shape=(num_features,), dtype=np.float32),
        action_space=spaces.Box(low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32),
        lr_schedule=lambda _: 1e-3,
        rlpfn_model=fake_model,
        num_features=num_features,
        obs_slot_dim=obs_slot_dim,
        net_arch=[],
    )
    policy.eval()
    step_fn = policy.make_vectorized_rollout_step_fn()
    batch_size = 3
    obs_t = torch.randn((batch_size, obs_slot_dim), dtype=torch.float32)
    action_t = torch.zeros((batch_size, action_dim), dtype=torch.float32)
    reward_t = torch.zeros((batch_size, 1), dtype=torch.float32)
    reward_mask_t = torch.ones((batch_size, 1), dtype=torch.float32)
    env_info = {
        "obs_dim": torch.full((batch_size,), obs_slot_dim, dtype=torch.long),
        "obs_slot_dim": torch.full((batch_size,), obs_slot_dim, dtype=torch.long),
        "action_slot_dim": torch.full((batch_size,), action_dim, dtype=torch.long),
        "action_dim_per_sample": torch.tensor([2, 3, 1], dtype=torch.long),
        "terminal_reset_enabled": torch.zeros((batch_size,), dtype=torch.bool),
        "phase_t": torch.zeros((batch_size,), dtype=torch.float32),
    }
    actor_outputs, _ = step_fn(obs_t, action_t, reward_t, reward_mask_t, None, 0, env_info)
    action_mean = actor_outputs["action_mean"]
    action_std = actor_outputs["action_std"]
    assert tuple(action_mean.shape) == (batch_size, 3)
    assert tuple(action_std.shape) == (batch_size, 3)
    assert torch.allclose(action_mean[0, 2:], torch.zeros((1,), dtype=action_mean.dtype))
    assert torch.allclose(action_mean[2, 1:], torch.zeros((2,), dtype=action_mean.dtype))
    assert torch.allclose(action_std[0, 2:], torch.zeros((1,), dtype=action_std.dtype))
    assert torch.allclose(action_std[2, 1:], torch.zeros((2,), dtype=action_std.dtype))


def test_policy_vectorized_rollout_step_resets_cache_rows_on_terminal():
    num_features = 8
    obs_slot_dim = 4
    action_dim = 2
    fake_model = _FakeRWKVModel(
        num_features=num_features,
        x_obs_dim=6,
        action_dim=action_dim,
        emsize=12,
    )

    def _stateful_forward_step(token, state=None):
        carry = state[0][0][:, :1].expand_as(token)
        hidden = token + carry
        next_state = []
        for att_x_prev, att_kv, ffn_x_prev in state:
            next_state.append(
                (
                    att_x_prev + 1.0,
                    att_kv + 1.0,
                    ffn_x_prev + 1.0,
                )
            )
        return hidden, next_state

    fake_model.rwkv_core.forward_step = _stateful_forward_step
    policy = OfficialRWKVRecurrentPPOPolicy(
        observation_space=spaces.Box(low=-np.inf, high=np.inf, shape=(num_features,), dtype=np.float32),
        action_space=spaces.Box(low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32),
        lr_schedule=lambda _: 1e-3,
        rlpfn_model=fake_model,
        num_features=num_features,
        obs_slot_dim=obs_slot_dim,
        net_arch=[],
    )
    policy.eval()
    step_fn = policy.make_vectorized_rollout_step_fn()
    batch_size = 2
    obs_t = torch.randn((batch_size, obs_slot_dim), dtype=torch.float32)
    action_t = torch.zeros((batch_size, action_dim), dtype=torch.float32)
    reward_t = torch.zeros((batch_size, 1), dtype=torch.float32)
    reward_mask_t = torch.ones((batch_size, 1), dtype=torch.float32)
    env_info = {
        "obs_dim": torch.full((batch_size,), obs_slot_dim, dtype=torch.long),
        "obs_slot_dim": torch.full((batch_size,), obs_slot_dim, dtype=torch.long),
        "action_slot_dim": torch.full((batch_size,), action_dim, dtype=torch.long),
        "action_dim_per_sample": torch.full((batch_size,), action_dim, dtype=torch.long),
        "terminal_reset_enabled": torch.ones((batch_size,), dtype=torch.bool),
        "phase_t": torch.zeros((batch_size,), dtype=torch.float32),
        "terminal_t": torch.zeros((batch_size, 1), dtype=torch.float32),
    }

    first_outputs, cache_after_first = step_fn(obs_t, action_t, reward_t, reward_mask_t, None, 0, env_info)
    continued_outputs, _ = step_fn(obs_t, action_t, reward_t, reward_mask_t, cache_after_first, 1, env_info)
    reset_env_info = dict(env_info)
    reset_env_info["terminal_t"] = torch.ones((batch_size, 1), dtype=torch.float32)
    reset_outputs, _ = step_fn(obs_t, action_t, reward_t, reward_mask_t, cache_after_first, 1, reset_env_info)
    fresh_reset_outputs, _ = step_fn(obs_t, action_t, reward_t, reward_mask_t, None, 1, reset_env_info)

    assert not torch.allclose(first_outputs["value_logits"], continued_outputs["value_logits"])
    assert torch.allclose(reset_outputs["value_logits"], fresh_reset_outputs["value_logits"])


def test_official_rollout_hidden_from_cache_handles_batch1_flat_official_eval_state():
    num_features = 8
    obs_slot_dim = 4
    action_dim = 2
    fake_model = _FakeRWKVModel(
        num_features=num_features,
        x_obs_dim=6,
        action_dim=action_dim,
        emsize=12,
    )
    state_inputs = []

    def _flat_official_forward_step(token, state=None):
        state_inputs.append(state)
        hidden = token + 0.5
        if state is None:
            state = fake_model.rwkv_core.init_state(
                int(token.shape[0]),
                device=token.device,
                dtype=token.dtype,
            )
        if isinstance(state, list) and state and isinstance(state[0], tuple):
            flat = []
            for att_x_prev, att_kv, ffn_x_prev in state:
                flat.append(att_x_prev[0] + 1.0)
                flat.append(att_kv[0] + 1.0)
                flat.append(ffn_x_prev[0] + 1.0)
            return hidden, flat
        return hidden, [tensor + 1.0 for tensor in state]

    fake_model.rwkv_core.forward_step = _flat_official_forward_step
    fake_model.rwkv_core._is_official_eval_state = (
        lambda state: isinstance(state, list) and len(state) == 3 and all(torch.is_tensor(t) for t in state)
    )
    policy = OfficialRWKVRecurrentPPOPolicy(
        observation_space=spaces.Box(low=-np.inf, high=np.inf, shape=(num_features,), dtype=np.float32),
        action_space=spaces.Box(low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32),
        lr_schedule=lambda _: 1e-3,
        rlpfn_model=fake_model,
        num_features=num_features,
        obs_slot_dim=obs_slot_dim,
        net_arch=[],
    )
    policy.eval()
    obs = torch.randn((1, num_features), dtype=torch.float32)

    hidden_first, cache_first = policy._official_rollout_hidden_from_cache(
        obs,
        torch.zeros((1,), dtype=torch.float32),
        cache=None,
    )
    hidden_second, cache_second = policy._official_rollout_hidden_from_cache(
        obs,
        torch.zeros((1,), dtype=torch.float32),
        cache=cache_first,
    )
    hidden_reset, _ = policy._official_rollout_hidden_from_cache(
        obs,
        torch.ones((1,), dtype=torch.float32),
        cache=cache_first,
    )

    assert tuple(hidden_first.shape) == (1, fake_model.emsize)
    assert tuple(hidden_second.shape) == (1, fake_model.emsize)
    assert tuple(hidden_reset.shape) == (1, fake_model.emsize)
    assert isinstance(cache_first, list)
    assert len(cache_first) == 3
    assert isinstance(state_inputs[0], list)
    assert isinstance(state_inputs[0][0], tuple)
    assert state_inputs[1] is cache_first
    assert isinstance(state_inputs[2], list)
    assert isinstance(state_inputs[2][0], tuple)


def test_official_rollout_hidden_chunked_cache_matches_full_batch_stateful_path():
    num_features = 8
    obs_slot_dim = 4
    action_dim = 2
    torch.manual_seed(7)
    fake_model = _FakeRWKVModel(
        num_features=num_features,
        x_obs_dim=6,
        action_dim=action_dim,
        emsize=12,
        replay_batch_chunk_size=None,
    )

    def _stateful_forward_step(token, state=None):
        if state is None:
            state = fake_model.rwkv_core.init_state(
                int(token.shape[0]),
                device=token.device,
                dtype=token.dtype,
            )
        carry = state[0][0][:, :1].expand_as(token)
        hidden = token + carry
        next_state = []
        for att_x_prev, att_kv, ffn_x_prev in state:
            next_state.append(
                (
                    att_x_prev + 1.0,
                    att_kv + 1.0,
                    ffn_x_prev + 1.0,
                )
            )
        return hidden, next_state

    fake_model.rwkv_core.forward_step = _stateful_forward_step
    policy = OfficialRWKVRecurrentPPOPolicy(
        observation_space=spaces.Box(low=-np.inf, high=np.inf, shape=(num_features,), dtype=np.float32),
        action_space=spaces.Box(low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32),
        lr_schedule=lambda _: 1e-3,
        rlpfn_model=fake_model,
        num_features=num_features,
        obs_slot_dim=obs_slot_dim,
        net_arch=[],
    )
    policy.eval()
    obs = torch.randn((5, num_features), dtype=torch.float32)
    starts = torch.zeros((5,), dtype=torch.float32)

    full_hidden_0, full_cache_0 = policy._official_rollout_hidden_from_cache(
        obs,
        starts,
        cache=None,
    )
    full_hidden_1, full_cache_1 = policy._official_rollout_hidden_from_cache(
        obs,
        starts,
        cache=full_cache_0,
    )

    fake_model.replay_batch_chunk_size = 2
    chunk_hidden_0, chunk_cache_0 = policy._official_rollout_hidden_from_cache(
        obs,
        starts,
        cache=None,
    )
    chunk_hidden_1, chunk_cache_1 = policy._official_rollout_hidden_from_cache(
        obs,
        starts,
        cache=chunk_cache_0,
    )
    materialized_cache_1 = sb3_recurrent_ppo_module._materialize_rollout_cache_batch(
        chunk_cache_1,
        0,
        int(obs.shape[0]),
    )

    assert sb3_recurrent_ppo_module._is_rollout_chunked_cache(chunk_cache_0)
    assert torch.allclose(chunk_hidden_0, full_hidden_0)
    assert torch.allclose(chunk_hidden_1, full_hidden_1)
    assert torch.allclose(materialized_cache_1[0][0], full_cache_1[0][0])
    assert torch.allclose(materialized_cache_1[0][1], full_cache_1[0][1])
    assert torch.allclose(materialized_cache_1[0][2], full_cache_1[0][2])


def test_policy_evaluate_actions_uses_official_separate_actor_critic_sequence_paths():
    device = _cuda_or_skip()
    num_features = 8
    obs_slot_dim = 4
    action_dim = 2
    fake_model = _FakeRWKVModel(
        num_features=num_features,
        x_obs_dim=6,
        action_dim=action_dim,
        emsize=12,
        normalized_q_head=True,
    ).to(device)
    policy = OfficialRWKVRecurrentPPOPolicy(
        observation_space=spaces.Box(low=-np.inf, high=np.inf, shape=(num_features,), dtype=np.float32),
        action_space=spaces.Box(low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32),
        lr_schedule=lambda _: 1e-3,
        rlpfn_model=fake_model,
        num_features=num_features,
        obs_slot_dim=obs_slot_dim,
        net_arch=[],
    ).to(device)
    policy.train()
    obs = torch.randn((6, num_features), device=device, dtype=torch.float32)
    episode_starts = torch.tensor([1.0, 0.0, 0.0, 1.0, 0.0, 0.0], device=device, dtype=torch.float32)
    actions = torch.zeros((6, action_dim), device=device, dtype=torch.float32)
    values, log_prob, entropy = policy.evaluate_actions(
        obs,
        actions,
        policy._dummy_states(2),
        episode_starts,
    )
    assert values.shape == (6, 1)
    assert log_prob.shape == (6,)
    assert entropy.shape == (6,)
    assert fake_model.rwkv_core.step_calls == 0
    assert fake_model.rwkv_core.sequence_calls == 2
    assert fake_model.replay_chunk_requests == [(3, 2), (3, 2)]
    assert fake_model.encode_sequence_calls == 2
    assert fake_model.encode_sequence_shapes == [(3, 2, num_features), (3, 2, num_features)]


def test_policy_value_head_returns_bar_mean_for_rollout_and_train_paths():
    device = _cuda_or_skip()
    num_features = 8
    obs_slot_dim = 4
    action_dim = 2
    fake_model = _FakeRWKVModel(
        num_features=num_features,
        x_obs_dim=6,
        action_dim=action_dim,
        emsize=12,
        normalized_q_head=True,
    ).to(device)
    policy = OfficialRWKVRecurrentPPOPolicy(
        observation_space=spaces.Box(low=-np.inf, high=np.inf, shape=(num_features,), dtype=np.float32),
        action_space=spaces.Box(low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32),
        lr_schedule=lambda _: 1e-3,
        rlpfn_model=fake_model,
        num_features=num_features,
        obs_slot_dim=obs_slot_dim,
        net_arch=[],
    ).to(device)
    obs = torch.randn((6, num_features), device=device, dtype=torch.float32)
    episode_starts = torch.tensor([1.0, 0.0, 0.0, 1.0, 0.0, 0.0], device=device, dtype=torch.float32)
    actions = torch.zeros((6, action_dim), device=device, dtype=torch.float32)

    eval_outputs = policy.evaluate_actions_with_hidden(
        obs,
        actions,
        policy._dummy_states(2),
        episode_starts,
    )
    assert int(policy.get_value_bardist().num_bars) == 100
    assert torch.allclose(
        policy.get_value_bardist().borders[[0, -1]].detach().cpu(),
        torch.tensor(
            [
                -sb3_recurrent_ppo_module.PPO_VALUE_BARDIST_VALUE_RANGE,
                sb3_recurrent_ppo_module.PPO_VALUE_BARDIST_VALUE_RANGE,
            ],
            dtype=torch.float32,
        ),
        atol=1e-6,
        rtol=0.0,
    )
    assert torch.allclose(
        fake_model.get_normalized_q_value_bardist().borders[[0, -1]].detach().cpu(),
        torch.tensor([-5.0, 5.0], dtype=torch.float32),
        atol=1e-6,
        rtol=0.0,
    )
    expected_values = policy.get_value_bardist().mean(
        eval_outputs["value_logits"].to(dtype=torch.float32)
    ).unsqueeze(-1)
    assert eval_outputs["values"].shape == (6, 1)
    assert eval_outputs["value_logits"].shape == (6, int(policy.get_value_bardist().num_bars))
    assert torch.allclose(eval_outputs["values"], expected_values, atol=1e-6, rtol=1e-6)

    offsets = torch.full((6,), 3.0, device=device, dtype=torch.float32)
    scales = torch.full((6,), 2.5, device=device, dtype=torch.float32)
    shifted_values = policy._value_from_logits(
        eval_outputs["value_logits"],
        value_means=offsets,
        value_stds=scales,
    )
    assert torch.allclose(
        shifted_values,
        offsets.unsqueeze(-1) + scales.unsqueeze(-1) * expected_values,
        atol=1e-6,
        rtol=1e-6,
    )


def test_policy_can_use_scalar_linear_value_head():
    device = _cuda_or_skip()
    num_features = 8
    obs_slot_dim = 4
    action_dim = 2
    fake_model = _FakeRWKVModel(
        num_features=num_features,
        x_obs_dim=6,
        action_dim=action_dim,
        emsize=12,
        normalized_q_head=True,
    ).to(device)
    policy = OfficialRWKVRecurrentPPOPolicy(
        observation_space=spaces.Box(low=-np.inf, high=np.inf, shape=(num_features,), dtype=np.float32),
        action_space=spaces.Box(low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32),
        lr_schedule=lambda _: 1e-3,
        rlpfn_model=fake_model,
        num_features=num_features,
        obs_slot_dim=obs_slot_dim,
        value_head_impl="scalar_linear",
        net_arch=[],
    ).to(device)
    obs = torch.randn((6, num_features), device=device, dtype=torch.float32)
    episode_starts = torch.tensor([1.0, 0.0, 0.0, 1.0, 0.0, 0.0], device=device, dtype=torch.float32)
    actions = torch.zeros((6, action_dim), device=device, dtype=torch.float32)

    eval_outputs = policy.evaluate_actions_with_hidden(
        obs,
        actions,
        policy._dummy_states(2),
        episode_starts,
    )
    assert policy.has_value_bardist() is False
    assert eval_outputs["value_logits"].shape == (6, 1)
    assert eval_outputs["values"].shape == (6, 1)
    assert torch.allclose(
        eval_outputs["values"],
        eval_outputs["value_logits"].to(dtype=torch.float32),
        atol=1e-6,
        rtol=1e-6,
    )

    offsets = torch.full((6,), 3.0, device=device, dtype=torch.float32)
    scales = torch.full((6,), 2.5, device=device, dtype=torch.float32)
    shifted_values = policy._value_from_logits(
        eval_outputs["value_logits"],
        value_means=offsets,
        value_stds=scales,
    )
    assert torch.allclose(
        shifted_values,
        offsets.unsqueeze(-1) + scales.unsqueeze(-1) * eval_outputs["value_logits"].to(dtype=torch.float32),
        atol=1e-6,
        rtol=1e-6,
    )


def test_policy_can_use_batchnorm_value_path_adapter():
    device = _cuda_or_skip()
    num_features = 8
    obs_slot_dim = 4
    action_dim = 2
    fake_model = _FakeRWKVModel(
        num_features=num_features,
        x_obs_dim=6,
        action_dim=action_dim,
        emsize=12,
        normalized_q_head=True,
    ).to(device)
    policy = OfficialRWKVRecurrentPPOPolicy(
        observation_space=spaces.Box(low=-np.inf, high=np.inf, shape=(num_features,), dtype=np.float32),
        action_space=spaces.Box(low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32),
        lr_schedule=lambda _: 1e-3,
        rlpfn_model=fake_model,
        num_features=num_features,
        obs_slot_dim=obs_slot_dim,
        value_head_impl="legacy_bar",
        value_path_adapter_impl="batchnorm",
        net_arch=[],
    ).to(device)
    assert policy.value_path_adapter_impl == "batchnorm"
    assert isinstance(policy.value_path_adapter, sb3_recurrent_ppo_module._ValuePathBatchNormAdapter)

    latent = torch.randn((6, policy.mlp_extractor.latent_dim_vf), device=device, dtype=torch.float32)
    adapted = policy._adapt_value_latent(latent)
    assert adapted.shape == latent.shape
    assert torch.isfinite(adapted).all()
    assert not torch.allclose(adapted, latent)

    single_latent = torch.randn((1, policy.mlp_extractor.latent_dim_vf), device=device, dtype=torch.float32)
    single_adapted = policy._adapt_value_latent(single_latent)
    assert single_adapted.shape == single_latent.shape
    assert torch.isfinite(single_adapted).all()

    obs = torch.randn((6, num_features), device=device, dtype=torch.float32)
    episode_starts = torch.tensor([1.0, 0.0, 0.0, 1.0, 0.0, 0.0], device=device, dtype=torch.float32)
    actions = torch.zeros((6, action_dim), device=device, dtype=torch.float32)
    eval_outputs = policy.evaluate_actions_with_hidden(
        obs,
        actions,
        policy._dummy_states(2),
        episode_starts,
    )
    assert eval_outputs["value_logits"].shape == (6, int(policy.get_value_bardist().num_bars))
    assert eval_outputs["values"].shape == (6, 1)


def test_policy_can_use_batchnorm_batchstats_value_path_adapter():
    device = _cuda_or_skip()
    num_features = 8
    obs_slot_dim = 4
    action_dim = 2
    fake_model = _FakeRWKVModel(
        num_features=num_features,
        x_obs_dim=6,
        action_dim=action_dim,
        emsize=12,
        normalized_q_head=True,
    ).to(device)
    policy = OfficialRWKVRecurrentPPOPolicy(
        observation_space=spaces.Box(low=-np.inf, high=np.inf, shape=(num_features,), dtype=np.float32),
        action_space=spaces.Box(low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32),
        lr_schedule=lambda _: 1e-3,
        rlpfn_model=fake_model,
        num_features=num_features,
        obs_slot_dim=obs_slot_dim,
        value_head_impl="legacy_bar",
        value_path_adapter_impl="batchnorm_batchstats",
        net_arch=[],
    ).to(device)
    assert policy.value_path_adapter_impl == "batchnorm_batchstats"
    assert isinstance(policy.value_path_adapter, sb3_recurrent_ppo_module._ValuePathBatchNormAdapter)
    assert bool(policy.value_path_adapter.use_batch_stats_in_eval) is True

    latent = torch.randn((6, policy.mlp_extractor.latent_dim_vf), device=device, dtype=torch.float32)
    policy.eval()
    adapted = policy._adapt_value_latent(latent)
    expected = F.batch_norm(
        latent,
        policy.value_path_adapter.norm.running_mean.detach().clone(),
        policy.value_path_adapter.norm.running_var.detach().clone(),
        policy.value_path_adapter.norm.weight,
        policy.value_path_adapter.norm.bias,
        True,
        policy.value_path_adapter.norm.momentum,
        policy.value_path_adapter.norm.eps,
    )
    assert adapted.shape == latent.shape
    assert torch.isfinite(adapted).all()
    assert torch.allclose(adapted, expected, atol=1e-6, rtol=1e-6)


def test_policy_can_use_vendor_official_value_head_and_frozen_zscore_adapter():
    device = _cuda_or_skip()
    num_features = 8
    obs_slot_dim = 4
    action_dim = 2
    fake_model = _FakeRWKVModel(
        num_features=num_features,
        x_obs_dim=6,
        action_dim=action_dim,
        emsize=12,
        normalized_q_head=True,
    ).to(device)
    policy = OfficialRWKVRecurrentPPOPolicy(
        observation_space=spaces.Box(low=-np.inf, high=np.inf, shape=(num_features,), dtype=np.float32),
        action_space=spaces.Box(low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32),
        lr_schedule=lambda _: 1e-3,
        rlpfn_model=fake_model,
        num_features=num_features,
        obs_slot_dim=obs_slot_dim,
        value_head_impl="vendor_official",
        value_path_adapter_impl="frozen_zscore",
        net_arch=[],
    ).to(device)
    assert policy.value_head_impl == "vendor_official"
    assert policy.value_path_adapter_impl == "frozen_zscore"
    assert isinstance(policy.value_path_adapter, sb3_recurrent_ppo_module._ValuePathFrozenZScoreAdapter)
    assert policy.has_value_bardist() is True
    assert policy.has_value_vendor_head() is True

    obs = torch.randn((6, num_features), device=device, dtype=torch.float32)
    episode_starts = torch.tensor([1.0, 0.0, 0.0, 1.0, 0.0, 0.0], device=device, dtype=torch.float32)
    actions = torch.zeros((6, action_dim), device=device, dtype=torch.float32)
    eval_outputs = policy.evaluate_actions_with_hidden(
        obs,
        actions,
        policy._dummy_states(2),
        episode_starts,
    )
    assert eval_outputs["value_logits"].shape == (6, int(policy.get_value_bardist().num_bars))
    assert eval_outputs["values"].shape == (6, 1)

    obs_steps = obs.reshape(3, 2, num_features)
    episode_starts_steps = episode_starts.reshape(3, 2)
    objective_masks_steps = torch.tensor(
        [[0.0, 0.0], [1.0, 1.0], [1.0, 1.0]],
        device=device,
        dtype=torch.float32,
    )
    fitted = policy.fit_value_path_adapter_from_rollout_steps(
        obs_steps,
        episode_starts_steps,
        objective_masks_steps,
    )
    assert fitted is True
    assert bool(policy.value_path_adapter._is_fitted.item()) is True
    assert torch.isfinite(eval_outputs["values"]).all()

    policy.eval()
    with torch.no_grad():
        rollout_values = policy.predict_values(obs[:3], policy._dummy_states(3), episode_starts[:3])
    assert rollout_values.shape == (3, 1)


def _expected_official_advantages(values, rewards, episode_starts, *, gamma, gae_lambda, last_value, dones):
    values = np.asarray(values, dtype=np.float32).reshape(-1, 1)
    rewards = np.asarray(rewards, dtype=np.float32).reshape(-1, 1)
    episode_starts = np.asarray(episode_starts, dtype=np.float32).reshape(-1, 1)
    last_value = np.asarray(last_value, dtype=np.float32).reshape(1)
    dones = np.asarray(dones, dtype=bool).reshape(1)
    adv = np.zeros_like(values, dtype=np.float32)
    last_gae = np.zeros((1,), dtype=np.float32)
    for step in reversed(range(int(values.shape[0]))):
        if step == int(values.shape[0]) - 1:
            next_non_terminal = 1.0 - dones.astype(np.float32, copy=False)
            next_value = last_value
        else:
            next_non_terminal = 1.0 - episode_starts[step + 1].astype(np.float32, copy=False)
            next_value = values[step + 1].reshape(1)
        delta = rewards[step].reshape(1) + float(gamma) * next_value * next_non_terminal - values[step].reshape(1)
        last_gae = delta + float(gamma) * float(gae_lambda) * next_non_terminal * last_gae
        adv[step] = last_gae
    return adv.reshape(-1)


def test_masked_rollout_buffer_identity_return_normalization_matches_official_gae():
    gamma = 0.0
    gae_lambda = 0.8
    buf = MaskedRecurrentRolloutBuffer(
        buffer_size=3,
        observation_space=spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32),
        action_space=spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),
        hidden_state_shape=(3, 1, 1, 1),
        device="cpu",
        gae_lambda=gae_lambda,
        gamma=gamma,
        n_envs=1,
        next_state_dim=1,
    )
    buf.reset()
    values = np.asarray([0.25, -0.5, 1.0], dtype=np.float32)
    rewards = np.asarray([-1.2247449, 0.0, 1.2247449], dtype=np.float32)
    episode_starts = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
    buf.values[:, 0] = values
    buf.rewards[:, 0] = rewards
    buf.episode_starts[:, 0] = episode_starts
    last_value = torch.tensor([0.4], dtype=torch.float32)
    dones = np.asarray([True], dtype=bool)

    buf.compute_returns_and_advantage(last_value, dones)

    expected_adv = _expected_official_advantages(
        values,
        rewards,
        episode_starts,
        gamma=gamma,
        gae_lambda=gae_lambda,
        last_value=np.asarray([0.4], dtype=np.float32),
        dones=dones,
    )
    expected_returns = expected_adv + values
    assert np.allclose(buf.rollout_return_means[:, 0], 0.0, atol=1e-6, rtol=1e-6)
    assert np.allclose(buf.rollout_return_stds[:, 0], 1.0, atol=1e-6, rtol=1e-6)
    assert np.allclose(buf.advantages[:, 0], expected_adv, atol=1e-6, rtol=1e-6)
    assert np.allclose(buf.actor_advantages[:, 0], expected_adv, atol=1e-6, rtol=1e-6)
    assert np.allclose(buf.returns[:, 0], expected_returns, atol=1e-6, rtol=1e-6)


def test_masked_rollout_buffer_defaults_to_normalized_value_target_space():
    gamma = 0.5
    gae_lambda = 0.75
    buf = MaskedRecurrentRolloutBuffer(
        buffer_size=2,
        observation_space=spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32),
        action_space=spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),
        hidden_state_shape=(2, 1, 1, 1),
        device="cpu",
        gae_lambda=gae_lambda,
        gamma=gamma,
        n_envs=1,
        next_state_dim=1,
    )
    buf.reset()
    buf.values[:, 0] = np.asarray([1.0, 2.0], dtype=np.float32)
    buf.rewards[:, 0] = np.asarray([3.0, 5.0], dtype=np.float32)
    buf.episode_starts[:, 0] = np.asarray([1.0, 0.0], dtype=np.float32)
    last_value = torch.tensor([3.0], dtype=torch.float32)
    dones = np.asarray([False], dtype=bool)

    buf.compute_returns_and_advantage(last_value, dones)

    expected_adv = np.asarray([4.875, 9.0], dtype=np.float32)
    expected_returns = np.asarray([5.875, 11.0], dtype=np.float32)

    assert buf._value_target_space == "normalized"
    assert np.allclose(buf.rollout_return_means[:, 0], np.asarray([5.25, 5.25], dtype=np.float32), atol=1e-6, rtol=1e-6)
    assert np.allclose(buf.rollout_return_stds[:, 0], 0.25, atol=1e-6, rtol=1e-6)
    assert np.allclose(buf.advantages[:, 0], expected_adv, atol=1e-6, rtol=1e-6)
    assert np.allclose(buf.actor_advantages[:, 0], expected_adv, atol=1e-6, rtol=1e-6)
    assert np.allclose(buf.returns[:, 0], expected_returns, atol=1e-6, rtol=1e-6)


def test_raw_value_target_tensor_uses_gae_value_targets_not_discounted_returns():
    buf = MaskedRecurrentRolloutBuffer(
        buffer_size=2,
        observation_space=spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32),
        action_space=spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),
        hidden_state_shape=(2, 1, 1, 1),
        device="cpu",
        gae_lambda=0.5,
        gamma=1.0,
        n_envs=1,
        next_state_dim=1,
    )
    buf.reset()
    buf._value_target_space = "raw"
    buf.values[:, 0] = np.asarray([0.0, 0.0], dtype=np.float32)
    buf.rewards[:, 0] = np.asarray([1.0, 1.0], dtype=np.float32)
    buf.episode_starts[:, 0] = np.asarray([1.0, 0.0], dtype=np.float32)

    buf.compute_returns_and_advantage(torch.tensor([0.0], dtype=torch.float32), np.asarray([False], dtype=bool))

    raw_value_targets = buf._get_flat_raw_value_targets_tensor(dtype=torch.float32).cpu().numpy()
    raw_discounted_returns = buf._get_flat_raw_returns_tensor(dtype=torch.float32).cpu().numpy()
    assert np.allclose(raw_value_targets, np.asarray([1.5, 1.0], dtype=np.float32), atol=1e-6, rtol=1e-6)
    assert np.allclose(raw_discounted_returns, np.asarray([2.0, 1.0], dtype=np.float32), atol=1e-6, rtol=1e-6)


def test_masked_rollout_buffer_cross_rollout_running_ema_rms_preserves_cross_rollout_scale():
    buf = MaskedRecurrentRolloutBuffer(
        buffer_size=2,
        observation_space=spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32),
        action_space=spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),
        hidden_state_shape=(2, 1, 1, 1),
        device="cpu",
        gae_lambda=1.0,
        gamma=1.0,
        n_envs=1,
        next_state_dim=1,
    )
    buf._value_target_space = CROSS_ROLLOUT_RUNNING_EMA_RMS_SPACE
    buf._actor_gae_space = CROSS_ROLLOUT_RUNNING_EMA_RMS_SPACE
    buf.reset()
    buf.values[:, 0] = 0.0
    buf.rewards[:, 0] = np.asarray([1.0, 1.0], dtype=np.float32)
    buf.episode_starts[:, 0] = np.asarray([1.0, 0.0], dtype=np.float32)
    buf.compute_returns_and_advantage(torch.tensor([0.0], dtype=torch.float32), np.asarray([False], dtype=bool))

    first_scale = np.sqrt((2.0**2 + 1.0**2) / 2.0)
    assert np.allclose(buf.rollout_return_means[:, 0], 0.0, atol=1e-6, rtol=1e-6)
    assert np.allclose(buf.rollout_return_stds[:, 0], first_scale, atol=1e-6, rtol=1e-6)
    assert np.allclose(buf.returns[:, 0], np.asarray([2.0, 1.0]) / first_scale, atol=1e-6, rtol=1e-6)

    buf.reset()
    buf._value_target_space = CROSS_ROLLOUT_RUNNING_EMA_RMS_SPACE
    buf._actor_gae_space = CROSS_ROLLOUT_RUNNING_EMA_RMS_SPACE
    buf.values[:, 0] = 0.0
    buf.rewards[:, 0] = np.asarray([10.0, 10.0], dtype=np.float32)
    buf.episode_starts[:, 0] = np.asarray([1.0, 0.0], dtype=np.float32)
    buf.compute_returns_and_advantage(torch.tensor([0.0], dtype=torch.float32), np.asarray([False], dtype=bool))

    second_rollout_local_scale = np.sqrt((20.0**2 + 10.0**2) / 2.0)
    second_running_scale = np.sqrt(
        (1.0 - CROSS_ROLLOUT_RUNNING_EMA_RMS_ALPHA) * (first_scale**2)
        + CROSS_ROLLOUT_RUNNING_EMA_RMS_ALPHA * (second_rollout_local_scale**2)
    )
    assert np.allclose(buf.rollout_return_stds[:, 0], second_running_scale, atol=1e-6, rtol=1e-6)
    assert second_running_scale < second_rollout_local_scale
    assert np.allclose(
        buf.actor_advantages[:, 0],
        np.asarray([20.0, 10.0], dtype=np.float32) / second_running_scale,
        atol=1e-6,
        rtol=1e-6,
    )


def test_masked_rollout_buffer_reward_normalized_actor_gae_requires_reward_normalized_value_space():
    gamma = 1.0
    gae_lambda = 1.0
    buf = MaskedRecurrentRolloutBuffer(
        buffer_size=2,
        observation_space=spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32),
        action_space=spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),
        hidden_state_shape=(2, 1, 1, 1),
        device="cpu",
        gae_lambda=gae_lambda,
        gamma=gamma,
        n_envs=1,
        next_state_dim=1,
    )
    buf.reset()
    buf._value_target_space = "raw"
    buf._actor_gae_space = "reward_normalized"
    buf.values[:, 0] = np.asarray([0.0, 0.0], dtype=np.float32)
    buf.rewards[:, 0] = np.asarray([1.0, 5.0], dtype=np.float32)
    buf.objective_masks[:, 0] = np.asarray([1.0, 1.0], dtype=np.float32)
    buf.episode_starts[:, 0] = np.asarray([1.0, 0.0], dtype=np.float32)
    last_value = torch.tensor([0.0], dtype=torch.float32)
    dones = np.asarray([False], dtype=bool)

    with pytest.raises(
        ValueError,
        match="actor_gae_space='reward_normalized' requires value_target_space='reward_normalized'",
    ):
        buf.compute_returns_and_advantage(last_value, dones)


def test_masked_rollout_buffer_reward_normalized_value_space_supports_raw_actor_gae():
    buf = MaskedRecurrentRolloutBuffer(
        buffer_size=2,
        observation_space=spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32),
        action_space=spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),
        hidden_state_shape=(2, 1, 1, 1),
        device="cpu",
        gae_lambda=1.0,
        gamma=1.0,
        n_envs=1,
        next_state_dim=1,
    )
    buf.reset()
    buf._value_target_space = "reward_normalized"
    buf._actor_gae_space = "raw"
    buf.values[:, 0] = np.asarray([0.0, 0.0], dtype=np.float32)
    buf.rewards[:, 0] = np.asarray([3.0, 5.0], dtype=np.float32)
    buf.objective_masks[:, 0] = np.asarray([1.0, 1.0], dtype=np.float32)
    buf.episode_starts[:, 0] = np.asarray([1.0, 0.0], dtype=np.float32)

    buf.compute_returns_and_advantage(torch.tensor([0.0], dtype=torch.float32), np.asarray([False], dtype=bool))

    assert np.isfinite(buf.actor_advantages).all()
    assert buf.actor_advantages.shape == (2, 1)


def test_masked_rollout_buffer_cached_bucket_idx_clips_targets_outside_full_support():
    buf = MaskedRecurrentRolloutBuffer(
        buffer_size=3,
        observation_space=spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32),
        action_space=spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),
        hidden_state_shape=(3, 1, 1, 1),
        device="cpu",
        gae_lambda=1.0,
        gamma=1.0,
        n_envs=1,
        next_state_dim=1,
    )
    buf.reset()
    bardist = make_standardized_full_support_bar_distribution(num_buckets=5, value_range=5.0)
    buf._value_target_bardist_borders = bardist.borders.detach().cpu().numpy()
    buf.returns[:, 0] = np.asarray([-99.0, 0.0, 99.0], dtype=np.float32)

    buf._cache_value_target_bucket_idx()

    assert np.array_equal(buf.value_target_bucket_idx[:, 0], np.asarray([0, 2, 4], dtype=np.int64))


def test_masked_rollout_buffer_reward_normalized_value_targets_use_whole_rollout_reward_stats():
    gamma = 0.5
    gae_lambda = 1.0
    buf = MaskedRecurrentRolloutBuffer(
        buffer_size=4,
        observation_space=spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32),
        action_space=spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),
        hidden_state_shape=(4, 1, 1, 1),
        device="cpu",
        gae_lambda=gae_lambda,
        gamma=gamma,
        n_envs=1,
        next_state_dim=1,
    )
    buf.reset()
    buf._value_target_space = "reward_normalized"
    buf.values[:, 0] = np.asarray([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    buf.rewards[:, 0] = np.asarray([100.0, 100.0, 1.0, 5.0], dtype=np.float32)
    buf.objective_masks[:, 0] = np.asarray([0.0, 0.0, 1.0, 1.0], dtype=np.float32)
    buf.episode_starts[:, 0] = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    last_value = torch.tensor([0.0], dtype=torch.float32)
    dones = np.asarray([False], dtype=bool)

    buf.compute_returns_and_advantage(last_value, dones)

    expected_reward_mean = np.asarray([51.5], dtype=np.float32)
    expected_reward_std = np.asarray([48.520615], dtype=np.float32)
    expected_return_means = np.asarray([65.28125, 65.28125, 65.28125, 65.28125], dtype=np.float32)
    expected_return_stds = np.asarray([63.4567, 63.4567, 63.4567, 63.4567], dtype=np.float32)
    expected_raw_returns = np.asarray([150.875, 101.75, 3.5, 5.0], dtype=np.float32)
    expected_adv = np.asarray([1.1193696, 0.23958889, -1.5199726, -0.95835555], dtype=np.float32)

    assert np.allclose(buf.rollout_return_means[:, 0], expected_return_means, atol=1e-6, rtol=1e-6)
    assert np.allclose(buf.rollout_return_stds[:, 0], expected_return_stds, atol=1e-5, rtol=1e-5)
    assert np.allclose(buf.advantages[:, 0], expected_adv, atol=1e-6, rtol=1e-6)
    assert np.allclose(buf.actor_advantages[:, 0], expected_adv, atol=1e-6, rtol=1e-6)
    assert np.allclose(buf.returns[:, 0], expected_adv, atol=1e-6, rtol=1e-6)
    assert np.allclose(expected_reward_mean, np.asarray([51.5], dtype=np.float32), atol=1e-6, rtol=1e-6)
    assert np.allclose(expected_reward_std, np.asarray([48.520615], dtype=np.float32), atol=1e-5, rtol=1e-5)
    assert np.allclose(
        buf._get_flat_raw_returns_tensor(dtype=torch.float32).cpu().numpy(),
        expected_raw_returns,
        atol=1e-6,
        rtol=1e-6,
    )


def test_masked_rollout_buffer_add_pads_action_masks_to_global_mask_dim():
    buf = MaskedRecurrentRolloutBuffer(
        buffer_size=1,
        observation_space=spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32),
        action_space=spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32),
        hidden_state_shape=(1, 1, 1, 1),
        device="cpu",
        gae_lambda=0.95,
        gamma=0.99,
        n_envs=1,
        next_state_dim=1,
    )
    buf.reset()
    lstm_states = RNNStates(
        pi=(torch.zeros((1, 1, 1), dtype=torch.float32), torch.zeros((1, 1, 1), dtype=torch.float32)),
        vf=(torch.zeros((1, 1, 1), dtype=torch.float32), torch.zeros((1, 1, 1), dtype=torch.float32)),
    )
    buf.add(
        np.zeros((1, 2), dtype=np.float32),
        np.zeros((1, 4), dtype=np.float32),
        np.zeros((1,), dtype=np.float32),
        np.zeros((1,), dtype=np.float32),
        torch.zeros((1,), dtype=torch.float32),
        torch.zeros((1,), dtype=torch.float32),
        lstm_states=lstm_states,
        action_masks=np.asarray([[1.0, 0.0, 1.0]], dtype=np.float32),
        next_states=np.zeros((1, 1), dtype=np.float32),
        next_state_masks=np.ones((1, 1), dtype=np.float32),
        objective_masks=np.ones((1,), dtype=np.float32),
    )
    assert np.allclose(buf.action_masks[0, 0], np.asarray([1.0, 0.0, 1.0, 0.0], dtype=np.float32))


def test_masked_rollout_buffer_add_pads_actions_to_global_action_dim():
    buf = MaskedRecurrentRolloutBuffer(
        buffer_size=1,
        observation_space=spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32),
        action_space=spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32),
        hidden_state_shape=(1, 1, 1, 1),
        device="cpu",
        gae_lambda=0.95,
        gamma=0.99,
        n_envs=1,
        next_state_dim=1,
    )
    buf.reset()
    lstm_states = RNNStates(
        pi=(torch.zeros((1, 1, 1), dtype=torch.float32), torch.zeros((1, 1, 1), dtype=torch.float32)),
        vf=(torch.zeros((1, 1, 1), dtype=torch.float32), torch.zeros((1, 1, 1), dtype=torch.float32)),
    )
    buf.add(
        np.zeros((1, 2), dtype=np.float32),
        np.asarray([[0.25, -0.5, 1.5]], dtype=np.float32),
        np.zeros((1,), dtype=np.float32),
        np.zeros((1,), dtype=np.float32),
        torch.zeros((1,), dtype=torch.float32),
        torch.zeros((1,), dtype=torch.float32),
        lstm_states=lstm_states,
        action_masks=np.ones((1, 4), dtype=np.float32),
        next_states=np.zeros((1, 1), dtype=np.float32),
        next_state_masks=np.ones((1, 1), dtype=np.float32),
        objective_masks=np.ones((1,), dtype=np.float32),
    )
    assert np.allclose(buf.actions[0, 0], np.asarray([0.25, -0.5, 1.5, 0.0], dtype=np.float32))


def test_masked_rollout_buffer_rollout_return_stats_drive_normalized_gae():
    gamma = 0.5
    gae_lambda = 0.75
    buf = MaskedRecurrentRolloutBuffer(
        buffer_size=2,
        observation_space=spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32),
        action_space=spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),
        hidden_state_shape=(2, 1, 1, 1),
        device="cpu",
        gae_lambda=gae_lambda,
        gamma=gamma,
        n_envs=1,
        next_state_dim=1,
    )
    buf.reset()
    buf._value_target_space = "normalized"
    buf._actor_gae_space = "normalized"
    buf.values[:, 0] = np.asarray([1.0, 2.0], dtype=np.float32)
    buf.rewards[:, 0] = np.asarray([3.0, 5.0], dtype=np.float32)
    buf.episode_starts[:, 0] = np.asarray([1.0, 0.0], dtype=np.float32)
    last_value = torch.tensor([3.0], dtype=torch.float32)
    dones = np.asarray([False], dtype=bool)

    buf.compute_returns_and_advantage(last_value, dones)

    expected_raw_returns = np.asarray([5.5, 5.0], dtype=np.float32)
    expected_offset = np.asarray([5.25], dtype=np.float32)
    expected_scale = np.asarray([0.25], dtype=np.float32)
    reward_norm_1 = (5.0 / 0.25) + ((gamma * 1.0) - 1.0) * (5.25 / 0.25)
    adv_1 = reward_norm_1 + gamma * 3.0 - 2.0
    reward_norm_0 = (3.0 / 0.25) + ((gamma * 1.0) - 1.0) * (5.25 / 0.25)
    adv_0 = reward_norm_0 + gamma * 2.0 - 1.0 + (gamma * gae_lambda * adv_1)
    expected_adv = np.asarray([adv_0, adv_1], dtype=np.float32)
    expected_returns = expected_adv + np.asarray([1.0, 2.0], dtype=np.float32)
    expected_recovered_returns = np.asarray(
        [
            5.25 + 0.25 * expected_returns[0],
            5.25 + 0.25 * expected_returns[1],
        ],
        dtype=np.float32,
    )
    assert np.allclose(buf.rollout_return_means[:, 0], expected_offset, atol=1e-6, rtol=1e-6)
    assert np.allclose(buf.rollout_return_stds[:, 0], expected_scale, atol=1e-6, rtol=1e-6)
    assert np.allclose(buf.advantages[:, 0], expected_adv, atol=1e-6, rtol=1e-6)
    assert np.allclose(buf.actor_advantages[:, 0], expected_adv, atol=1e-6, rtol=1e-6)
    assert np.allclose(buf.returns[:, 0], expected_returns, atol=1e-6, rtol=1e-6)
    assert np.allclose(
        buf._get_flat_raw_returns_tensor(dtype=torch.float32).cpu().numpy(),
        expected_recovered_returns,
        atol=1e-6,
        rtol=1e-6,
    )
    assert np.allclose(expected_raw_returns, np.asarray([5.5, 5.0], dtype=np.float32), atol=1e-6, rtol=1e-6)


def test_masked_rollout_buffer_raw_actor_gae_space_keeps_critic_normalized_but_actor_advantages_raw():
    gamma = 0.5
    gae_lambda = 0.75
    buf = MaskedRecurrentRolloutBuffer(
        buffer_size=2,
        observation_space=spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32),
        action_space=spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),
        hidden_state_shape=(2, 1, 1, 1),
        device="cpu",
        gae_lambda=gae_lambda,
        gamma=gamma,
        n_envs=1,
        next_state_dim=1,
    )
    buf.reset()
    buf._actor_gae_space = "raw"
    buf._value_target_space = "normalized"
    buf.values[:, 0] = np.asarray([1.0, 2.0], dtype=np.float32)
    buf.rewards[:, 0] = np.asarray([3.0, 5.0], dtype=np.float32)
    buf.episode_starts[:, 0] = np.asarray([1.0, 0.0], dtype=np.float32)
    last_value = torch.tensor([3.0], dtype=torch.float32)
    dones = np.asarray([False], dtype=bool)

    buf.compute_returns_and_advantage(last_value, dones)

    expected_actor_values_raw = np.asarray([5.5, 5.75], dtype=np.float32)
    expected_last_value_raw = np.asarray([6.0], dtype=np.float32)
    expected_actor_adv = _expected_official_advantages(
        expected_actor_values_raw,
        np.asarray([3.0, 5.0], dtype=np.float32),
        np.asarray([1.0, 0.0], dtype=np.float32),
        gamma=gamma,
        gae_lambda=gae_lambda,
        last_value=expected_last_value_raw,
        dones=dones,
    )
    expected_actor_adv = expected_actor_adv.astype(np.float32, copy=False)

    reward_norm_1 = (5.0 / 0.25) + ((gamma * 1.0) - 1.0) * (5.25 / 0.25)
    expected_norm_adv_1 = reward_norm_1 + gamma * 3.0 - 2.0
    reward_norm_0 = (3.0 / 0.25) + ((gamma * 1.0) - 1.0) * (5.25 / 0.25)
    expected_norm_adv_0 = reward_norm_0 + gamma * 2.0 - 1.0 + (gamma * gae_lambda * expected_norm_adv_1)
    expected_norm_adv = np.asarray([expected_norm_adv_0, expected_norm_adv_1], dtype=np.float32)
    expected_norm_returns = expected_norm_adv + np.asarray([1.0, 2.0], dtype=np.float32)

    assert np.allclose(buf.advantages[:, 0], expected_norm_adv, atol=1e-6, rtol=1e-6)
    assert np.allclose(buf.returns[:, 0], expected_norm_returns, atol=1e-6, rtol=1e-6)
    assert np.allclose(buf.actor_advantages[:, 0], expected_actor_adv, atol=1e-6, rtol=1e-6)
    assert not np.allclose(buf.actor_advantages[:, 0], buf.advantages[:, 0], atol=1e-6, rtol=1e-6)


def test_masked_rollout_buffer_normalized_sep_synced_matches_normalized_when_objective_covers_all_steps():
    gamma = 1.0
    gae_lambda = 1.0
    base = MaskedRecurrentRolloutBuffer(
        buffer_size=4,
        observation_space=spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32),
        action_space=spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),
        hidden_state_shape=(4, 1, 1, 1),
        device="cpu",
        gae_lambda=gae_lambda,
        gamma=gamma,
        n_envs=1,
        next_state_dim=1,
    )
    synced = MaskedRecurrentRolloutBuffer(
        buffer_size=4,
        observation_space=spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32),
        action_space=spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),
        hidden_state_shape=(4, 1, 1, 1),
        device="cpu",
        gae_lambda=gae_lambda,
        gamma=gamma,
        n_envs=1,
        next_state_dim=1,
    )
    for buf in (base, synced):
        buf.reset()
        buf._value_target_space = "normalized"
        buf._actor_gae_space = "normalized"
        buf.values[:, 0] = np.asarray([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        buf.rewards[:, 0] = np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        buf.objective_masks[:, 0] = np.asarray([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        buf.episode_starts[:, 0] = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    synced._actor_gae_space = "normalized_sep_synced"
    last_value = torch.tensor([0.0], dtype=torch.float32)
    dones = np.asarray([False], dtype=bool)

    base.compute_returns_and_advantage(last_value, dones)
    synced.compute_returns_and_advantage(last_value, dones)

    assert np.allclose(synced.actor_advantages[:, 0], base.actor_advantages[:, 0], atol=1e-6, rtol=1e-6)
    assert np.allclose(synced.advantages[:, 0], base.advantages[:, 0], atol=1e-6, rtol=1e-6)


def test_masked_rollout_buffer_normalized_sep_synced_recomputes_actor_advantages_on_suffix_local_value_space():
    gamma = 1.0
    gae_lambda = 1.0
    base = MaskedRecurrentRolloutBuffer(
        buffer_size=4,
        observation_space=spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32),
        action_space=spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),
        hidden_state_shape=(4, 1, 1, 1),
        device="cpu",
        gae_lambda=gae_lambda,
        gamma=gamma,
        n_envs=1,
        next_state_dim=1,
    )
    synced = MaskedRecurrentRolloutBuffer(
        buffer_size=4,
        observation_space=spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32),
        action_space=spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),
        hidden_state_shape=(4, 1, 1, 1),
        device="cpu",
        gae_lambda=gae_lambda,
        gamma=gamma,
        n_envs=1,
        next_state_dim=1,
    )
    for buf in (base, synced):
        buf.reset()
        buf._value_target_space = "normalized"
        buf._actor_gae_space = "normalized"
        buf.values[:, 0] = np.asarray([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        buf.rewards[:, 0] = np.asarray([10.0, 10.0, 1.0, 1.0], dtype=np.float32)
        buf.objective_masks[:, 0] = np.asarray([0.0, 0.0, 1.0, 1.0], dtype=np.float32)
        buf.episode_starts[:, 0] = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    synced._actor_gae_space = "normalized_sep_synced"
    last_value = torch.tensor([0.0], dtype=torch.float32)
    dones = np.asarray([False], dtype=bool)

    base.compute_returns_and_advantage(last_value, dones)
    synced.compute_returns_and_advantage(last_value, dones)

    assert np.allclose(synced.advantages[:, 0], base.advantages[:, 0], atol=1e-6, rtol=1e-6)
    assert np.allclose(synced.returns[:, 0], base.returns[:, 0], atol=1e-6, rtol=1e-6)
    adjusted_episode_starts = np.asarray(synced.episode_starts, dtype=np.float32).copy()
    objective_mask_bool = np.asarray(synced.objective_masks, dtype=np.float32) > 1e-8
    for env_idx in range(int(adjusted_episode_starts.shape[1])):
        valid_steps = np.flatnonzero(objective_mask_bool[:, env_idx])
        if int(valid_steps.size) > 0:
            adjusted_episode_starts[int(valid_steps[0]), env_idx] = 1.0
    actor_rollout_raw_returns = _discounted_returns_from_rewards(
        synced.rewards,
        episode_starts=adjusted_episode_starts,
        dones=dones,
        gamma=gamma,
    )
    actor_return_means_np, actor_return_stds_np = _rollout_return_norm_stats_from_raw_returns(
        actor_rollout_raw_returns,
        eps=1e-6,
    )
    values_raw = (
        np.asarray(synced.rollout_return_means, dtype=np.float32)
        + np.asarray(synced.rollout_return_stds, dtype=np.float32) * np.asarray(synced.values, dtype=np.float32)
    )
    last_values_raw = (
        np.asarray(synced.rollout_return_means, dtype=np.float32)[-1]
        + np.asarray(synced.rollout_return_stds, dtype=np.float32)[-1] * np.asarray([0.0], dtype=np.float32)
    )
    actor_values = (
        values_raw - actor_return_means_np.reshape((1, synced.n_envs))
    ) / actor_return_stds_np.reshape((1, synced.n_envs))
    actor_last_values = (last_values_raw - actor_return_means_np) / actor_return_stds_np
    actor_return_mean_over_std = actor_return_means_np / actor_return_stds_np
    expected_actor_adv = np.zeros_like(synced.actor_advantages, dtype=np.float32)
    last_gae_lam = np.zeros((synced.n_envs,), dtype=np.float32)
    for step in reversed(range(synced.buffer_size)):
        if step == synced.buffer_size - 1:
            next_non_terminal = 1.0 - dones.astype(np.float32, copy=False)
            next_values = actor_last_values
        else:
            next_non_terminal = 1.0 - adjusted_episode_starts[step + 1].astype(np.float32, copy=False)
            next_values = actor_values[step + 1]
        curr_values = actor_values[step]
        reward_norm = (
            synced.rewards[step] / actor_return_stds_np
            + ((synced.gamma * next_non_terminal) - 1.0) * actor_return_mean_over_std
        )
        gamma_eff = synced.gamma * next_non_terminal
        delta = reward_norm + gamma_eff * next_values - curr_values
        last_gae_lam = delta + gamma_eff * synced.gae_lambda * last_gae_lam
        expected_actor_adv[step] = last_gae_lam
    assert np.allclose(synced.actor_advantages[:, 0], expected_actor_adv[:, 0], atol=1e-5, rtol=1e-5)
    assert not np.allclose(synced.actor_advantages[2:, 0], base.actor_advantages[2:, 0], atol=1e-6, rtol=1e-6)


def test_masked_rollout_buffer_zero_actor_baseline_uses_raw_returns_for_actor_only():
    gamma = 0.5
    gae_lambda = 0.75
    buf = MaskedRecurrentRolloutBuffer(
        buffer_size=2,
        observation_space=spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32),
        action_space=spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),
        hidden_state_shape=(2, 1, 1, 1),
        device="cpu",
        gae_lambda=gae_lambda,
        gamma=gamma,
        n_envs=1,
        next_state_dim=1,
    )
    buf.reset()
    buf._actor_baseline_mode = "zero"
    buf._value_target_space = "normalized"
    buf.values[:, 0] = np.asarray([1.0, 2.0], dtype=np.float32)
    buf.rewards[:, 0] = np.asarray([3.0, 5.0], dtype=np.float32)
    buf.episode_starts[:, 0] = np.asarray([1.0, 0.0], dtype=np.float32)
    last_value = torch.tensor([3.0], dtype=torch.float32)
    dones = np.asarray([False], dtype=bool)

    buf.compute_returns_and_advantage(last_value, dones)

    expected_raw_returns = np.asarray([5.5, 5.0], dtype=np.float32)
    assert np.allclose(buf.actor_advantages[:, 0], expected_raw_returns, atol=1e-6, rtol=1e-6)
    assert np.allclose(buf.rollout_return_means[:, 0], np.asarray([5.25, 5.25], dtype=np.float32), atol=1e-6, rtol=1e-6)
    assert np.allclose(buf.rollout_return_stds[:, 0], np.asarray([0.25, 0.25], dtype=np.float32), atol=1e-6, rtol=1e-6)
    assert not np.allclose(buf.actor_advantages[:, 0], buf.advantages[:, 0], atol=1e-6, rtol=1e-6)


def test_masked_rollout_buffer_rollout_mean_actor_baseline_centers_raw_returns_for_actor_only():
    gamma = 0.5
    gae_lambda = 0.75
    buf = MaskedRecurrentRolloutBuffer(
        buffer_size=2,
        observation_space=spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32),
        action_space=spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),
        hidden_state_shape=(2, 1, 1, 1),
        device="cpu",
        gae_lambda=gae_lambda,
        gamma=gamma,
        n_envs=1,
        next_state_dim=1,
    )
    buf.reset()
    buf._actor_baseline_mode = "rollout_mean"
    buf._value_target_space = "normalized"
    buf.values[:, 0] = np.asarray([1.0, 2.0], dtype=np.float32)
    buf.rewards[:, 0] = np.asarray([3.0, 5.0], dtype=np.float32)
    buf.episode_starts[:, 0] = np.asarray([1.0, 0.0], dtype=np.float32)
    last_value = torch.tensor([3.0], dtype=torch.float32)
    dones = np.asarray([False], dtype=bool)

    buf.compute_returns_and_advantage(last_value, dones)

    expected_raw_returns = np.asarray([5.5, 5.0], dtype=np.float32)
    expected_centered = expected_raw_returns - np.asarray([5.25, 5.25], dtype=np.float32)
    assert np.allclose(buf.actor_advantages[:, 0], expected_centered, atol=1e-6, rtol=1e-6)
    assert not np.allclose(buf.actor_advantages[:, 0], buf.advantages[:, 0], atol=1e-6, rtol=1e-6)


def test_masked_rollout_buffer_rollout_mean_actor_baseline_centers_raw_returns_under_raw_value_targets():
    gamma = 0.5
    gae_lambda = 0.75
    buf = MaskedRecurrentRolloutBuffer(
        buffer_size=2,
        observation_space=spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32),
        action_space=spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),
        hidden_state_shape=(2, 1, 1, 1),
        device="cpu",
        gae_lambda=gae_lambda,
        gamma=gamma,
        n_envs=1,
        next_state_dim=1,
    )
    buf.reset()
    buf._actor_baseline_mode = "rollout_mean"
    buf._value_target_space = "raw"
    buf.values[:, 0] = np.asarray([1.0, 2.0], dtype=np.float32)
    buf.rewards[:, 0] = np.asarray([3.0, 5.0], dtype=np.float32)
    buf.episode_starts[:, 0] = np.asarray([1.0, 0.0], dtype=np.float32)
    last_value = torch.tensor([3.0], dtype=torch.float32)
    dones = np.asarray([False], dtype=bool)

    buf.compute_returns_and_advantage(last_value, dones)

    expected_raw_returns = np.asarray([5.5, 5.0], dtype=np.float32)
    expected_centered = expected_raw_returns - np.asarray([5.25, 5.25], dtype=np.float32)
    assert np.allclose(buf.actor_advantages[:, 0], expected_centered, atol=1e-6, rtol=1e-6)


def test_masked_rollout_buffer_trajectory_suffix_return_actor_objective_repeats_suffix_score():
    gamma = 0.5
    gae_lambda = 0.75
    buf = MaskedRecurrentRolloutBuffer(
        buffer_size=4,
        observation_space=spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32),
        action_space=spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),
        hidden_state_shape=(4, 1, 1, 1),
        device="cpu",
        gae_lambda=gae_lambda,
        gamma=gamma,
        n_envs=1,
        next_state_dim=1,
    )
    buf.reset()
    buf._actor_objective_mode = "trajectory_suffix_return"
    buf.values[:, 0] = np.asarray([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    buf.rewards[:, 0] = np.asarray([1.0, -2.0, 3.5, 4.0], dtype=np.float32)
    buf.episode_starts[:, 0] = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    buf.objective_masks[:, 0] = np.asarray([0.0, 0.0, 1.0, 1.0], dtype=np.float32)
    last_value = torch.tensor([0.5], dtype=torch.float32)
    dones = np.asarray([False], dtype=bool)

    buf.compute_returns_and_advantage(last_value, dones)

    expected_suffix_score = np.asarray([7.5, 7.5, 7.5, 7.5], dtype=np.float32)
    assert np.allclose(buf.actor_advantages[:, 0], expected_suffix_score, atol=1e-6, rtol=1e-6)
    assert np.allclose(buf.advantages[:, 0], buf.advantages[:, 0], atol=1e-6, rtol=1e-6)
    assert not np.allclose(buf.actor_advantages[:, 0], buf.advantages[:, 0], atol=1e-6, rtol=1e-6)


def test_masked_rollout_buffer_tokenwise_suffix_return_correction_reweights_actor_advantages():
    gamma = 0.5
    gae_lambda = 0.75
    buf = MaskedRecurrentRolloutBuffer(
        buffer_size=2,
        observation_space=spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32),
        action_space=spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),
        hidden_state_shape=(2, 1, 1, 1),
        device="cpu",
        gae_lambda=gae_lambda,
        gamma=gamma,
        n_envs=1,
        next_state_dim=1,
    )
    buf.reset()
    buf._actor_objective_mode = "tokenwise_suffix_return_correction"
    buf.values[:, 0] = np.asarray([1.0, 2.0], dtype=np.float32)
    buf.rewards[:, 0] = np.asarray([3.0, 5.0], dtype=np.float32)
    buf.objective_masks[:, 0] = np.asarray([1.0, 1.0], dtype=np.float32)
    buf.episode_starts[:, 0] = np.asarray([1.0, 0.0], dtype=np.float32)
    last_value = torch.tensor([3.0], dtype=torch.float32)
    dones = np.asarray([False], dtype=bool)

    buf.compute_returns_and_advantage(last_value, dones)

    base_adv = np.asarray([4.875, 9.0], dtype=np.float32)
    correction = 1.0 + np.tanh(np.asarray([1.0], dtype=np.float32))
    expected_actor_adv = base_adv * float(correction[0])
    assert np.allclose(buf.advantages[:, 0], base_adv, atol=1e-6, rtol=1e-6)
    assert np.allclose(buf.actor_advantages[:, 0], expected_actor_adv, atol=1e-6, rtol=1e-6)


def test_masked_rollout_buffer_tokenwise_drop_q1_early_zeros_first_suffix_bucket():
    gamma = 0.5
    gae_lambda = 0.75
    buf = MaskedRecurrentRolloutBuffer(
        buffer_size=4,
        observation_space=spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32),
        action_space=spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),
        hidden_state_shape=(4, 1, 1, 1),
        device="cpu",
        gae_lambda=gae_lambda,
        gamma=gamma,
        n_envs=1,
        next_state_dim=1,
    )
    buf.reset()
    buf._actor_objective_mode = "tokenwise_drop_q1_early"
    buf.values[:, 0] = np.asarray([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    buf.rewards[:, 0] = np.asarray([1.0, -2.0, 3.5, 4.0], dtype=np.float32)
    buf.objective_masks[:, 0] = np.asarray([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    buf.episode_starts[:, 0] = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    last_value = torch.tensor([0.5], dtype=torch.float32)
    dones = np.asarray([False], dtype=bool)

    buf.compute_returns_and_advantage(last_value, dones)

    assert np.isclose(float(buf.actor_advantages[0, 0]), 0.0, atol=1e-6, rtol=1e-6)
    assert np.allclose(buf.actor_advantages[1:, 0], buf.advantages[1:, 0], atol=1e-6, rtol=1e-6)
    assert not np.allclose(buf.actor_advantages[:, 0], buf.advantages[:, 0], atol=1e-6, rtol=1e-6)


def test_masked_rollout_buffer_tokenwise_post_first_terminal_only_keeps_only_post_terminal_suffix():
    gamma = 0.5
    gae_lambda = 0.75
    buf = MaskedRecurrentRolloutBuffer(
        buffer_size=6,
        observation_space=spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32),
        action_space=spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),
        hidden_state_shape=(6, 1, 1, 1),
        device="cpu",
        gae_lambda=gae_lambda,
        gamma=gamma,
        n_envs=1,
        next_state_dim=1,
    )
    buf.reset()
    buf._actor_objective_mode = "tokenwise_post_first_terminal_only"
    buf.values[:, 0] = np.asarray([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], dtype=np.float32)
    buf.rewards[:, 0] = np.asarray([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32)
    buf.objective_masks[:, 0] = np.asarray([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    buf.episode_starts[:, 0] = np.asarray([1.0, 0.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    last_value = torch.tensor([0.0], dtype=torch.float32)
    dones = np.asarray([False], dtype=bool)

    buf.compute_returns_and_advantage(last_value, dones)

    assert np.allclose(buf.actor_advantages[:3, 0], np.zeros((3,), dtype=np.float32), atol=1e-6, rtol=1e-6)
    assert np.allclose(buf.actor_advantages[3:, 0], buf.advantages[3:, 0], atol=1e-6, rtol=1e-6)
    assert not np.allclose(buf.actor_advantages[:, 0], buf.advantages[:, 0], atol=1e-6, rtol=1e-6)


def test_masked_rollout_buffer_tokenwise_shift_q1_contract_updates_objective_mask():
    gamma = 0.5
    gae_lambda = 0.75
    buf = MaskedRecurrentRolloutBuffer(
        buffer_size=4,
        observation_space=spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32),
        action_space=spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),
        hidden_state_shape=(4, 1, 1, 1),
        device="cpu",
        gae_lambda=gae_lambda,
        gamma=gamma,
        n_envs=1,
        next_state_dim=1,
    )
    buf.reset()
    buf._actor_objective_mode = "tokenwise_shift_q1_contract"
    buf.values[:, 0] = np.asarray([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    buf.rewards[:, 0] = np.asarray([1.0, -2.0, 3.5, 4.0], dtype=np.float32)
    buf.objective_masks[:, 0] = np.asarray([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    buf.episode_starts[:, 0] = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    last_value = torch.tensor([0.5], dtype=torch.float32)
    dones = np.asarray([False], dtype=bool)

    buf.compute_returns_and_advantage(last_value, dones)

    assert np.allclose(buf.objective_masks[:, 0], np.asarray([0.0, 1.0, 1.0, 1.0], dtype=np.float32), atol=1e-6, rtol=1e-6)
    assert np.allclose(buf.actor_advantages[:, 0], buf.advantages[:, 0], atol=1e-6, rtol=1e-6)


def test_masked_rollout_buffer_tokenwise_drop_first_k_suffix_steps_zeros_boundary_prefix():
    gamma = 0.5
    gae_lambda = 0.75
    buf = MaskedRecurrentRolloutBuffer(
        buffer_size=5,
        observation_space=spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32),
        action_space=spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),
        hidden_state_shape=(5, 1, 1, 1),
        device="cpu",
        gae_lambda=gae_lambda,
        gamma=gamma,
        n_envs=1,
        next_state_dim=1,
    )
    buf.reset()
    buf._actor_objective_mode = "tokenwise_drop_first_2_suffix_steps"
    buf.values[:, 0] = np.asarray([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32)
    buf.rewards[:, 0] = np.asarray([1.0, -2.0, 3.5, 4.0, -1.0], dtype=np.float32)
    buf.objective_masks[:, 0] = np.asarray([0.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    buf.episode_starts[:, 0] = np.asarray([1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    last_value = torch.tensor([0.5], dtype=torch.float32)
    dones = np.asarray([False], dtype=bool)

    buf.compute_returns_and_advantage(last_value, dones)

    assert np.isclose(float(buf.actor_advantages[1, 0]), 0.0, atol=1e-6, rtol=1e-6)
    assert np.isclose(float(buf.actor_advantages[2, 0]), 0.0, atol=1e-6, rtol=1e-6)
    assert np.allclose(buf.actor_advantages[3:, 0], buf.advantages[3:, 0], atol=1e-6, rtol=1e-6)


def test_masked_rollout_buffer_tokenwise_linear_ramp_first_k_suffix_steps_downweights_boundary_prefix():
    gamma = 0.5
    gae_lambda = 0.75
    buf = MaskedRecurrentRolloutBuffer(
        buffer_size=5,
        observation_space=spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32),
        action_space=spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),
        hidden_state_shape=(5, 1, 1, 1),
        device="cpu",
        gae_lambda=gae_lambda,
        gamma=gamma,
        n_envs=1,
        next_state_dim=1,
    )
    buf.reset()
    buf._actor_objective_mode = "tokenwise_linear_ramp_first_3_suffix_steps"
    buf.values[:, 0] = np.asarray([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32)
    buf.rewards[:, 0] = np.asarray([1.0, -2.0, 3.5, 4.0, -1.0], dtype=np.float32)
    buf.objective_masks[:, 0] = np.asarray([0.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    buf.episode_starts[:, 0] = np.asarray([1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    last_value = torch.tensor([0.5], dtype=torch.float32)
    dones = np.asarray([False], dtype=bool)

    buf.compute_returns_and_advantage(last_value, dones)

    assert np.allclose(
        buf.actor_advantages[1:4, 0],
        buf.advantages[1:4, 0] * np.asarray([1.0 / 3.0, 2.0 / 3.0, 1.0], dtype=np.float32),
        atol=1e-6,
        rtol=1e-6,
    )
    assert np.allclose(buf.actor_advantages[4:, 0], buf.advantages[4:, 0], atol=1e-6, rtol=1e-6)


def test_masked_rollout_buffer_tokenwise_scale_env12_mid_episode_extension_block_scales_only_branch_specific_window():
    actor_advantages = np.zeros((40, 13), dtype=np.float32)
    actor_advantages[27:31, 12] = np.asarray([-0.0901851886883378] * 4, dtype=np.float32)
    actor_advantages[27:31, 11] = np.asarray([-0.0901851886883378] * 4, dtype=np.float32)
    actor_advantages[35, 12] = np.float32(-0.06972909942269324)
    rewards = np.zeros_like(actor_advantages)
    objective_masks = np.ones_like(actor_advantages)
    episode_starts = np.zeros_like(actor_advantages)
    episode_starts[0, :] = 1.0
    for start in (3, 6, 9, 12):
        episode_starts[start, 12] = 1.0
        episode_starts[start, 11] = 1.0

    adjusted = _apply_actor_objective_postprocess(
        actor_advantages,
        rewards=rewards,
        objective_masks=objective_masks,
        episode_starts=episode_starts,
        actor_objective_mode="tokenwise_scale_env12_mid_episode_extension_block",
    )

    expected_scale = float(ENV12_MID_EPISODE_EXTENSION_BLOCK_SCALE)
    assert np.allclose(
        adjusted[27:31, 12],
        actor_advantages[27:31, 12] * expected_scale,
        atol=1e-6,
        rtol=1e-6,
    )
    assert np.allclose(adjusted[27:31, 11], actor_advantages[27:31, 11], atol=1e-6, rtol=1e-6)
    assert np.allclose(adjusted[35, 12], actor_advantages[35, 12], atol=1e-6, rtol=1e-6)
    assert np.allclose(adjusted[:27, 12], actor_advantages[:27, 12], atol=1e-6, rtol=1e-6)
    assert np.allclose(adjusted[31:, 12], actor_advantages[31:, 12], atol=1e-6, rtol=1e-6)


def test_masked_rollout_buffer_tokenwise_scale_env12_selector_matched_objective_episode_scales_only_target_episode():
    actor_advantages = np.zeros((24, 13), dtype=np.float32)
    actor_advantages[12:19, 12] = np.asarray([-0.2] * 7, dtype=np.float32)
    actor_advantages[12:19, 11] = np.asarray([-0.2] * 7, dtype=np.float32)
    actor_advantages[19:, 12] = np.asarray([-0.1] * 5, dtype=np.float32)
    rewards = np.zeros_like(actor_advantages)
    objective_masks = np.ones_like(actor_advantages)
    episode_starts = np.zeros_like(actor_advantages)
    episode_starts[0, :] = 1.0
    for start in (3, 6, 9, 12, 19):
        episode_starts[start, 12] = 1.0
        episode_starts[start, 11] = 1.0

    adjusted = _apply_actor_objective_postprocess(
        actor_advantages,
        rewards=rewards,
        objective_masks=objective_masks,
        episode_starts=episode_starts,
        actor_objective_mode="tokenwise_scale_env12_selector_matched_objective_episode",
    )

    expected_scale = float(ENV12_MID_EPISODE_EXTENSION_BLOCK_SCALE)
    assert np.allclose(
        adjusted[12:19, 12],
        actor_advantages[12:19, 12] * expected_scale,
        atol=1e-6,
        rtol=1e-6,
    )
    assert np.allclose(adjusted[12:19, 11], actor_advantages[12:19, 11], atol=1e-6, rtol=1e-6)
    assert np.allclose(adjusted[:12, 12], actor_advantages[:12, 12], atol=1e-6, rtol=1e-6)
    assert np.allclose(adjusted[19:, 12], actor_advantages[19:, 12], atol=1e-6, rtol=1e-6)


def test_masked_rollout_buffer_tokenwise_scale_env12_high_mass_objective_episode_family_scales_only_target_episodes():
    actor_advantages = np.zeros((28, 13), dtype=np.float32)
    actor_advantages[8:10, 12] = np.asarray([-0.2] * 2, dtype=np.float32)
    actor_advantages[14:16, 12] = np.asarray([-0.3] * 2, dtype=np.float32)
    actor_advantages[22:24, 12] = np.asarray([-0.4] * 2, dtype=np.float32)
    actor_advantages[8:10, 11] = np.asarray([-0.2] * 2, dtype=np.float32)
    actor_advantages[26:, 12] = np.asarray([-0.1] * 2, dtype=np.float32)
    rewards = np.zeros_like(actor_advantages)
    objective_masks = np.ones_like(actor_advantages)
    episode_starts = np.zeros_like(actor_advantages)
    episode_starts[0, :] = 1.0
    for start in (2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26):
        episode_starts[start, 12] = 1.0
        episode_starts[start, 11] = 1.0

    adjusted = _apply_actor_objective_postprocess(
        actor_advantages,
        rewards=rewards,
        objective_masks=objective_masks,
        episode_starts=episode_starts,
        actor_objective_mode="tokenwise_scale_env12_high_mass_objective_episode_family",
    )

    expected_scale = float(ENV12_MID_EPISODE_EXTENSION_BLOCK_SCALE)
    assert np.allclose(adjusted[8:10, 12], actor_advantages[8:10, 12] * expected_scale, atol=1e-6, rtol=1e-6)
    assert np.allclose(adjusted[14:16, 12], actor_advantages[14:16, 12] * expected_scale, atol=1e-6, rtol=1e-6)
    assert np.allclose(adjusted[22:24, 12], actor_advantages[22:24, 12] * expected_scale, atol=1e-6, rtol=1e-6)
    assert np.allclose(adjusted[8:10, 11], actor_advantages[8:10, 11], atol=1e-6, rtol=1e-6)
    assert np.allclose(adjusted[:8, 12], actor_advantages[:8, 12], atol=1e-6, rtol=1e-6)
    assert np.allclose(adjusted[26:, 12], actor_advantages[26:, 12], atol=1e-6, rtol=1e-6)


def test_masked_rollout_buffer_tokenwise_scale_pair2_recovered_anchor_first_objective_non_tail_family_scales_only_target_env_non_tail():
    actor_advantages = np.zeros((60, 16), dtype=np.float32)
    actor_advantages[:55, 0] = np.float32(-0.2)
    actor_advantages[:40, 15] = np.float32(-0.3)
    actor_advantages[:55, 1] = np.float32(-0.4)
    rewards = np.zeros_like(actor_advantages)
    objective_masks = np.ones_like(actor_advantages)
    episode_starts = np.zeros_like(actor_advantages)
    episode_starts[0, :] = 1.0
    episode_starts[55, 0] = 1.0
    episode_starts[40, 15] = 1.0
    episode_starts[55, 1] = 1.0

    adjusted = _apply_actor_objective_postprocess(
        actor_advantages,
        rewards=rewards,
        objective_masks=objective_masks,
        episode_starts=episode_starts,
        actor_objective_mode="tokenwise_scale_pair2_recovered_anchor_first_objective_non_tail_family",
    )

    expected_scale = float(ENV12_MID_EPISODE_EXTENSION_BLOCK_SCALE)
    assert np.allclose(adjusted[:47, 0], actor_advantages[:47, 0] * expected_scale, atol=1e-6, rtol=1e-6)
    assert np.allclose(adjusted[47:55, 0], actor_advantages[47:55, 0], atol=1e-6, rtol=1e-6)
    assert np.allclose(adjusted[:32, 15], actor_advantages[:32, 15] * expected_scale, atol=1e-6, rtol=1e-6)
    assert np.allclose(adjusted[32:40, 15], actor_advantages[32:40, 15], atol=1e-6, rtol=1e-6)
    assert np.allclose(adjusted[:55, 1], actor_advantages[:55, 1], atol=1e-6, rtol=1e-6)


def test_env12_mid_episode_extension_runtime_mode_is_scoped_to_pair2():
    mode = "tokenwise_scale_env12_mid_episode_extension_block"
    assert _resolve_runtime_scoped_actor_objective_mode(mode, current_suite_name="pair2") == mode
    assert _resolve_runtime_scoped_actor_objective_mode(mode, current_suite_name="seed_24680") == "tokenwise"
    assert _resolve_runtime_scoped_actor_objective_mode("tokenwise", current_suite_name="seed_24680") == "tokenwise"


def test_env12_selector_matched_objective_episode_runtime_mode_is_scoped_to_pair2():
    mode = "tokenwise_scale_env12_selector_matched_objective_episode"
    assert _resolve_runtime_scoped_actor_objective_mode(mode, current_suite_name="pair2") == mode
    assert _resolve_runtime_scoped_actor_objective_mode(mode, current_suite_name="seed_24680") == "tokenwise"
    assert _resolve_runtime_scoped_actor_objective_mode("tokenwise", current_suite_name="seed_24680") == "tokenwise"


def test_env12_high_mass_objective_episode_family_runtime_mode_is_scoped_to_pair2():
    mode = "tokenwise_scale_env12_high_mass_objective_episode_family"
    assert _resolve_runtime_scoped_actor_objective_mode(mode, current_suite_name="pair2") == mode
    assert _resolve_runtime_scoped_actor_objective_mode(mode, current_suite_name="seed_24680") == "tokenwise"
    assert _resolve_runtime_scoped_actor_objective_mode("tokenwise", current_suite_name="seed_24680") == "tokenwise"


def test_pair2_recovered_anchor_first_objective_non_tail_family_runtime_mode_is_scoped_to_pair2():
    mode = "tokenwise_scale_pair2_recovered_anchor_first_objective_non_tail_family"
    assert _resolve_runtime_scoped_actor_objective_mode(mode, current_suite_name="pair2") == mode
    assert _resolve_runtime_scoped_actor_objective_mode(mode, current_suite_name="seed_24680") == "tokenwise"
    assert _resolve_runtime_scoped_actor_objective_mode("tokenwise", current_suite_name="seed_24680") == "tokenwise"


def test_resolve_actor_advantages_prefers_actor_specific_tensor():
    advantages = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    actor_advantages = torch.tensor([-1.0, -2.0, -3.0], dtype=torch.float32)
    sample = SimpleNamespace(
        advantages=advantages,
        actor_advantages=actor_advantages,
    )
    assert torch.equal(_resolve_actor_advantages(sample), actor_advantages)
    sample.actor_advantages = None
    assert torch.equal(_resolve_actor_advantages(sample), advantages)


def test_build_recurrent_ppo_rejects_mixed_space_contract_by_default():
    device = _cuda_or_skip()
    cfg, env_cfg = _build_small_ppo_config()
    env_cfg = _strict_env_cfg_for_ppo(env_cfg)
    fake_model = _FakeRWKVModel(
        num_features=cfg["prior"]["num_features"],
        x_obs_dim=cfg["transformer"]["x_obs_dim"],
        action_dim=cfg["transformer"]["x_action_dim"],
    ).to(device)
    env_prior = EnvironmentPrior(env_cfg)
    with pytest.raises(
        ValueError,
        match="Mixed PPO reward/value space contracts are disabled",
    ):
        build_recurrent_ppo(
            model=fake_model,
            env_prior=env_prior,
            device=str(device),
            num_features=cfg["prior"]["num_features"],
            n_envs=cfg["optimizer"]["ppo_n_envs"],
            n_steps=cfg["optimizer"]["ppo_n_steps"],
            learning_rate=3e-4,
            batch_size=4,
            n_epochs=1,
            gamma=1.0,
            gae_lambda=0.95,
            clip_range=0.2,
            clip_range_vf=None,
            normalize_advantage=True,
            ent_coef=0.0,
            vf_coef=0.5,
            max_grad_norm=0.5,
            target_kl=None,
            space_contract="normalized",
            actor_gae_space="raw",
            verbose=0,
        )


def test_build_recurrent_ppo_defaults_value_target_space_to_normalized():
    device = _cuda_or_skip()
    cfg, env_cfg = _build_small_ppo_config()
    env_cfg = _strict_env_cfg_for_ppo(env_cfg)
    fake_model = _FakeRWKVModel(
        num_features=cfg["prior"]["num_features"],
        x_obs_dim=cfg["transformer"]["x_obs_dim"],
        action_dim=cfg["transformer"]["x_action_dim"],
    ).to(device)
    env_prior = EnvironmentPrior(env_cfg)
    algo, _, _ = build_recurrent_ppo(
        model=fake_model,
        env_prior=env_prior,
        device=str(device),
        num_features=cfg["prior"]["num_features"],
        n_envs=cfg["optimizer"]["ppo_n_envs"],
        n_steps=cfg["optimizer"]["ppo_n_steps"],
        learning_rate=3e-4,
        batch_size=4,
        n_epochs=1,
        gamma=1.0,
        gae_lambda=0.95,
        clip_range=0.2,
        clip_range_vf=None,
        normalize_advantage=True,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        target_kl=None,
        verbose=0,
    )
    assert isinstance(algo.rollout_buffer, MaskedRecurrentRolloutBuffer)
    assert algo.rollout_buffer._value_target_space == "normalized"
    assert algo._rwkv_value_target_space == "normalized"
    assert algo.rollout_buffer._actor_gae_space == "normalized"
    assert algo._rwkv_actor_gae_space == "normalized"
    assert algo._rwkv_space_contract["value_target_space_effective"] == "normalized"
    assert algo._rwkv_space_contract["actor_gae_space_effective"] == "normalized"
    assert algo._rwkv_space_contract["alignment"] == "aligned"
    assert algo._rwkv_space_contract["space_contract"] == "normalized"


def test_build_recurrent_ppo_accepts_reward_normalized_space_contract():
    device = _cuda_or_skip()
    cfg, env_cfg = _build_small_ppo_config()
    env_cfg = _strict_env_cfg_for_ppo(env_cfg)
    fake_model = _FakeRWKVModel(
        num_features=cfg["prior"]["num_features"],
        x_obs_dim=cfg["transformer"]["x_obs_dim"],
        action_dim=cfg["transformer"]["x_action_dim"],
    ).to(device)
    env_prior = EnvironmentPrior(env_cfg)
    algo, _, _ = build_recurrent_ppo(
        model=fake_model,
        env_prior=env_prior,
        device=str(device),
        num_features=cfg["prior"]["num_features"],
        n_envs=cfg["optimizer"]["ppo_n_envs"],
        n_steps=cfg["optimizer"]["ppo_n_steps"],
        learning_rate=3e-4,
        batch_size=4,
        n_epochs=1,
        gamma=1.0,
        gae_lambda=0.95,
        clip_range=0.2,
        clip_range_vf=None,
        normalize_advantage=True,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        target_kl=None,
        space_contract="reward_normalized",
        verbose=0,
    )
    assert isinstance(algo.rollout_buffer, MaskedRecurrentRolloutBuffer)
    assert algo.rollout_buffer._value_target_space == "reward_normalized"
    assert algo.rollout_buffer._actor_gae_space == "reward_normalized"


def test_build_recurrent_ppo_accepts_scalar_mlp_value_head():
    device = _cuda_or_skip()
    cfg, env_cfg = _build_small_ppo_config()
    env_cfg = _strict_env_cfg_for_ppo(env_cfg)
    fake_model = _FakeRWKVModel(
        num_features=cfg["prior"]["num_features"],
        x_obs_dim=cfg["transformer"]["x_obs_dim"],
        action_dim=cfg["transformer"]["x_action_dim"],
    ).to(device)
    env_prior = EnvironmentPrior(env_cfg)
    algo, _, _ = build_recurrent_ppo(
        model=fake_model,
        env_prior=env_prior,
        device=str(device),
        num_features=cfg["prior"]["num_features"],
        n_envs=cfg["optimizer"]["ppo_n_envs"],
        n_steps=cfg["optimizer"]["ppo_n_steps"],
        learning_rate=3e-4,
        batch_size=4,
        n_epochs=1,
        gamma=1.0,
        gae_lambda=0.95,
        clip_range=0.2,
        clip_range_vf=None,
        normalize_advantage=True,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        target_kl=None,
        value_head_impl="scalar_mlp",
        value_head_mlp_hidden_dim=32,
        space_contract="reward_normalized",
        verbose=0,
    )
    assert algo.policy.value_head_impl == "scalar_mlp"
    assert algo.policy.value_head_mlp_hidden_dim == 32
    assert isinstance(algo.policy.value_net, torch.nn.Sequential)
    assert algo._rwkv_value_target_space == "reward_normalized"


def test_build_recurrent_ppo_accepts_legacy_mixed_space_only_with_explicit_override():
    device = _cuda_or_skip()
    cfg, env_cfg = _build_small_ppo_config()
    env_cfg = _strict_env_cfg_for_ppo(env_cfg)
    fake_model = _FakeRWKVModel(
        num_features=cfg["prior"]["num_features"],
        x_obs_dim=cfg["transformer"]["x_obs_dim"],
        action_dim=cfg["transformer"]["x_action_dim"],
    ).to(device)
    env_prior = EnvironmentPrior(env_cfg)
    algo, _, _ = build_recurrent_ppo(
        model=fake_model,
        env_prior=env_prior,
        device=str(device),
        num_features=cfg["prior"]["num_features"],
        n_envs=cfg["optimizer"]["ppo_n_envs"],
        n_steps=cfg["optimizer"]["ppo_n_steps"],
        learning_rate=3e-4,
        batch_size=4,
        n_epochs=1,
        gamma=1.0,
        gae_lambda=0.95,
        clip_range=0.2,
        clip_range_vf=None,
        normalize_advantage=True,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        target_kl=None,
        space_contract="normalized",
        actor_gae_space="normalized_sep_synced",
        allow_mixed_space_contract=True,
        verbose=0,
    )
    assert isinstance(algo.rollout_buffer, MaskedRecurrentRolloutBuffer)
    assert algo.rollout_buffer._actor_gae_space == "normalized_sep_synced"


def test_build_recurrent_ppo_legacy_mixed_path_still_resolves_normalized_actor_gae_when_explicitly_enabled():
    device = _cuda_or_skip()
    cfg, env_cfg = _build_small_ppo_config()
    env_cfg = _strict_env_cfg_for_ppo(env_cfg)
    fake_model = _FakeRWKVModel(
        num_features=cfg["prior"]["num_features"],
        x_obs_dim=cfg["transformer"]["x_obs_dim"],
        action_dim=cfg["transformer"]["x_action_dim"],
    ).to(device)
    env_prior = EnvironmentPrior(env_cfg)
    algo, _, _ = build_recurrent_ppo(
        model=fake_model,
        env_prior=env_prior,
        device=str(device),
        num_features=cfg["prior"]["num_features"],
        n_envs=cfg["optimizer"]["ppo_n_envs"],
        n_steps=cfg["optimizer"]["ppo_n_steps"],
        learning_rate=3e-4,
        batch_size=4,
        n_epochs=1,
        gamma=1.0,
        gae_lambda=0.95,
        clip_range=0.2,
        clip_range_vf=None,
        normalize_advantage=False,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        target_kl=None,
        space_contract="normalized",
        actor_gae_space="normalized",
        actor_baseline_mode="learned",
        allow_mixed_space_contract=True,
        verbose=0,
    )
    assert isinstance(algo.rollout_buffer, MaskedRecurrentRolloutBuffer)
    assert algo._rwkv_requested_actor_gae_space == "normalized"
    assert algo._rwkv_actor_gae_space == "raw"
    assert algo.rollout_buffer._actor_gae_space == "raw"


def test_build_recurrent_ppo_propagates_actor_baseline_mode_to_rollout_buffer():
    device = _cuda_or_skip()
    cfg, env_cfg = _build_small_ppo_config()
    env_cfg = _strict_env_cfg_for_ppo(env_cfg)
    fake_model = _FakeRWKVModel(
        num_features=cfg["prior"]["num_features"],
        x_obs_dim=cfg["transformer"]["x_obs_dim"],
        action_dim=cfg["transformer"]["x_action_dim"],
    ).to(device)
    env_prior = EnvironmentPrior(env_cfg)
    algo, _, _ = build_recurrent_ppo(
        model=fake_model,
        env_prior=env_prior,
        device=str(device),
        num_features=cfg["prior"]["num_features"],
        n_envs=cfg["optimizer"]["ppo_n_envs"],
        n_steps=cfg["optimizer"]["ppo_n_steps"],
        learning_rate=3e-4,
        batch_size=4,
        n_epochs=1,
        gamma=1.0,
        gae_lambda=0.95,
        clip_range=0.2,
        clip_range_vf=None,
        normalize_advantage=True,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        target_kl=None,
        actor_gae_space="normalized",
        actor_baseline_mode="rollout_mean",
        verbose=0,
    )
    assert isinstance(algo.rollout_buffer, MaskedRecurrentRolloutBuffer)
    assert algo.rollout_buffer._actor_baseline_mode == "rollout_mean"


def test_build_recurrent_ppo_propagates_actor_objective_mode_to_rollout_buffer():
    device = _cuda_or_skip()
    cfg, env_cfg = _build_small_ppo_config()
    env_cfg = _strict_env_cfg_for_ppo(env_cfg)
    fake_model = _FakeRWKVModel(
        num_features=cfg["prior"]["num_features"],
        x_obs_dim=cfg["transformer"]["x_obs_dim"],
        action_dim=cfg["transformer"]["x_action_dim"],
    ).to(device)
    env_prior = EnvironmentPrior(env_cfg)
    algo, _, _ = build_recurrent_ppo(
        model=fake_model,
        env_prior=env_prior,
        device=str(device),
        num_features=cfg["prior"]["num_features"],
        n_envs=cfg["optimizer"]["ppo_n_envs"],
        n_steps=cfg["optimizer"]["ppo_n_steps"],
        learning_rate=3e-4,
        batch_size=4,
        n_epochs=1,
        gamma=1.0,
        gae_lambda=0.95,
        clip_range=0.2,
        clip_range_vf=None,
        normalize_advantage=True,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        target_kl=None,
        actor_gae_space="normalized",
        actor_baseline_mode="learned",
        actor_objective_mode="trajectory_suffix_return",
        verbose=0,
    )
    assert isinstance(algo.rollout_buffer, MaskedRecurrentRolloutBuffer)
    assert algo.rollout_buffer._actor_objective_mode == "trajectory_suffix_return"


def test_build_recurrent_ppo_propagates_actor_objective_correction_mode_to_rollout_buffer():
    device = _cuda_or_skip()
    cfg, env_cfg = _build_small_ppo_config()
    env_cfg = _strict_env_cfg_for_ppo(env_cfg)
    fake_model = _FakeRWKVModel(
        num_features=cfg["prior"]["num_features"],
        x_obs_dim=cfg["transformer"]["x_obs_dim"],
        action_dim=cfg["transformer"]["x_action_dim"],
    ).to(device)
    env_prior = EnvironmentPrior(env_cfg)
    algo, _, _ = build_recurrent_ppo(
        model=fake_model,
        env_prior=env_prior,
        device=str(device),
        num_features=cfg["prior"]["num_features"],
        n_envs=cfg["optimizer"]["ppo_n_envs"],
        n_steps=cfg["optimizer"]["ppo_n_steps"],
        learning_rate=3e-4,
        batch_size=4,
        n_epochs=1,
        gamma=1.0,
        gae_lambda=0.95,
        clip_range=0.2,
        clip_range_vf=None,
        normalize_advantage=True,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        target_kl=None,
        actor_gae_space="normalized",
        actor_baseline_mode="learned",
        actor_objective_mode="tokenwise_suffix_return_correction",
        verbose=0,
    )
    assert isinstance(algo.rollout_buffer, MaskedRecurrentRolloutBuffer)
    assert algo.rollout_buffer._actor_objective_mode == "tokenwise_suffix_return_correction"


def test_build_recurrent_ppo_propagates_actor_objective_q1_drop_mode_to_rollout_buffer():
    device = _cuda_or_skip()
    cfg, env_cfg = _build_small_ppo_config()
    env_cfg = _strict_env_cfg_for_ppo(env_cfg)
    fake_model = _FakeRWKVModel(
        num_features=cfg["prior"]["num_features"],
        x_obs_dim=cfg["transformer"]["x_obs_dim"],
        action_dim=cfg["transformer"]["x_action_dim"],
    ).to(device)
    env_prior = EnvironmentPrior(env_cfg)
    algo, _, _ = build_recurrent_ppo(
        model=fake_model,
        env_prior=env_prior,
        device=str(device),
        num_features=cfg["prior"]["num_features"],
        n_envs=cfg["optimizer"]["ppo_n_envs"],
        n_steps=cfg["optimizer"]["ppo_n_steps"],
        learning_rate=3e-4,
        batch_size=4,
        n_epochs=1,
        gamma=1.0,
        gae_lambda=0.95,
        clip_range=0.2,
        clip_range_vf=None,
        normalize_advantage=True,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        target_kl=None,
        actor_gae_space="normalized",
        actor_baseline_mode="learned",
        actor_objective_mode="tokenwise_drop_q1_early",
        verbose=0,
    )
    assert isinstance(algo.rollout_buffer, MaskedRecurrentRolloutBuffer)
    assert algo.rollout_buffer._actor_objective_mode == "tokenwise_drop_q1_early"


def test_build_recurrent_ppo_propagates_actor_objective_q1_contract_shift_mode_to_rollout_buffer():
    device = _cuda_or_skip()
    cfg, env_cfg = _build_small_ppo_config()
    env_cfg = _strict_env_cfg_for_ppo(env_cfg)
    fake_model = _FakeRWKVModel(
        num_features=cfg["prior"]["num_features"],
        x_obs_dim=cfg["transformer"]["x_obs_dim"],
        action_dim=cfg["transformer"]["x_action_dim"],
    ).to(device)
    env_prior = EnvironmentPrior(env_cfg)
    algo, _, _ = build_recurrent_ppo(
        model=fake_model,
        env_prior=env_prior,
        device=str(device),
        num_features=cfg["prior"]["num_features"],
        n_envs=cfg["optimizer"]["ppo_n_envs"],
        n_steps=cfg["optimizer"]["ppo_n_steps"],
        learning_rate=3e-4,
        batch_size=4,
        n_epochs=1,
        gamma=1.0,
        gae_lambda=0.95,
        clip_range=0.2,
        clip_range_vf=None,
        normalize_advantage=True,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        target_kl=None,
        actor_gae_space="normalized",
        actor_baseline_mode="learned",
        actor_objective_mode="tokenwise_shift_q1_contract",
        verbose=0,
    )
    assert isinstance(algo.rollout_buffer, MaskedRecurrentRolloutBuffer)
    assert algo.rollout_buffer._actor_objective_mode == "tokenwise_shift_q1_contract"


def test_build_recurrent_ppo_propagates_actor_objective_post_first_terminal_mode_to_rollout_buffer():
    device = _cuda_or_skip()
    cfg, env_cfg = _build_small_ppo_config()
    env_cfg = _strict_env_cfg_for_ppo(env_cfg)
    fake_model = _FakeRWKVModel(
        num_features=cfg["prior"]["num_features"],
        x_obs_dim=cfg["transformer"]["x_obs_dim"],
        action_dim=cfg["transformer"]["x_action_dim"],
    ).to(device)
    env_prior = EnvironmentPrior(env_cfg)
    algo, _, _ = build_recurrent_ppo(
        model=fake_model,
        env_prior=env_prior,
        device=str(device),
        num_features=cfg["prior"]["num_features"],
        n_envs=cfg["optimizer"]["ppo_n_envs"],
        n_steps=cfg["optimizer"]["ppo_n_steps"],
        learning_rate=3e-4,
        batch_size=4,
        n_epochs=1,
        gamma=1.0,
        gae_lambda=0.95,
        clip_range=0.2,
        clip_range_vf=None,
        normalize_advantage=True,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        target_kl=None,
        actor_gae_space="normalized",
        actor_baseline_mode="learned",
        actor_objective_mode="tokenwise_post_first_terminal_only",
        verbose=0,
    )
    assert isinstance(algo.rollout_buffer, MaskedRecurrentRolloutBuffer)
    assert algo.rollout_buffer._actor_objective_mode == "tokenwise_post_first_terminal_only"


def test_build_recurrent_ppo_propagates_env12_mid_episode_extension_runtime_mode_to_rollout_buffer():
    device = _cuda_or_skip()
    cfg, env_cfg = _build_small_ppo_config()
    env_cfg = _strict_env_cfg_for_ppo(env_cfg)
    cfg["optimizer"]["ppo_n_envs"] = 13
    fake_model = _FakeRWKVModel(
        num_features=cfg["prior"]["num_features"],
        x_obs_dim=cfg["transformer"]["x_obs_dim"],
        action_dim=cfg["transformer"]["x_action_dim"],
    ).to(device)
    env_prior = EnvironmentPrior(env_cfg)
    algo, _, _ = build_recurrent_ppo(
        model=fake_model,
        env_prior=env_prior,
        device=str(device),
        num_features=cfg["prior"]["num_features"],
        n_envs=cfg["optimizer"]["ppo_n_envs"],
        n_steps=cfg["optimizer"]["ppo_n_steps"],
        learning_rate=3e-4,
        batch_size=4,
        n_epochs=1,
        gamma=1.0,
        gae_lambda=0.95,
        clip_range=0.2,
        clip_range_vf=None,
        normalize_advantage=True,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        target_kl=None,
        strict_fixed_env_mode=True,
        actor_gae_space="normalized",
        actor_baseline_mode="learned",
        actor_objective_mode="tokenwise_scale_env12_mid_episode_extension_block",
        actor_objective_runtime_current_suite_name="pair2",
        verbose=0,
    )
    assert isinstance(algo.rollout_buffer, MaskedRecurrentRolloutBuffer)
    assert algo.rollout_buffer._actor_objective_mode == "tokenwise_scale_env12_mid_episode_extension_block"
    assert algo.rollout_buffer._actor_objective_runtime_current_suite_name == "pair2"


def test_build_recurrent_ppo_accepts_env12_selector_matched_objective_episode_mode():
    device = _cuda_or_skip()
    cfg, env_cfg = _build_small_ppo_config()
    env_cfg = _strict_env_cfg_for_ppo(env_cfg)
    cfg["optimizer"]["ppo_n_envs"] = 13
    fake_model = _FakeRWKVModel(
        num_features=cfg["prior"]["num_features"],
        x_obs_dim=cfg["transformer"]["x_obs_dim"],
        action_dim=cfg["transformer"]["x_action_dim"],
    ).to(device)
    env_prior = EnvironmentPrior(env_cfg)
    algo, _, _ = build_recurrent_ppo(
        model=fake_model,
        env_prior=env_prior,
        device=str(device),
        num_features=cfg["prior"]["num_features"],
        n_envs=cfg["optimizer"]["ppo_n_envs"],
        n_steps=cfg["optimizer"]["ppo_n_steps"],
        learning_rate=3e-4,
        batch_size=4,
        n_epochs=1,
        gamma=1.0,
        gae_lambda=0.95,
        clip_range=0.2,
        clip_range_vf=None,
        normalize_advantage=True,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        target_kl=None,
        strict_fixed_env_mode=True,
        actor_gae_space="normalized",
        actor_baseline_mode="learned",
        actor_objective_mode="tokenwise_scale_env12_selector_matched_objective_episode",
        actor_objective_runtime_current_suite_name="pair2",
        verbose=0,
    )
    assert isinstance(algo.rollout_buffer, MaskedRecurrentRolloutBuffer)
    assert algo.rollout_buffer._actor_objective_mode == "tokenwise_scale_env12_selector_matched_objective_episode"
    assert algo.rollout_buffer._actor_objective_runtime_current_suite_name == "pair2"


def test_build_recurrent_ppo_accepts_env12_high_mass_objective_episode_family_mode():
    device = _cuda_or_skip()
    cfg, env_cfg = _build_small_ppo_config()
    env_cfg = _strict_env_cfg_for_ppo(env_cfg)
    cfg["optimizer"]["ppo_n_envs"] = 13
    fake_model = _FakeRWKVModel(
        num_features=cfg["prior"]["num_features"],
        x_obs_dim=cfg["transformer"]["x_obs_dim"],
        action_dim=cfg["transformer"]["x_action_dim"],
    ).to(device)
    env_prior = EnvironmentPrior(env_cfg)
    algo, _, _ = build_recurrent_ppo(
        model=fake_model,
        env_prior=env_prior,
        device=str(device),
        num_features=cfg["prior"]["num_features"],
        n_envs=cfg["optimizer"]["ppo_n_envs"],
        n_steps=cfg["optimizer"]["ppo_n_steps"],
        learning_rate=3e-4,
        batch_size=4,
        n_epochs=1,
        gamma=1.0,
        gae_lambda=0.95,
        clip_range=0.2,
        clip_range_vf=None,
        normalize_advantage=True,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        target_kl=None,
        strict_fixed_env_mode=True,
        actor_gae_space="normalized",
        actor_baseline_mode="learned",
        actor_objective_mode="tokenwise_scale_env12_high_mass_objective_episode_family",
        actor_objective_runtime_current_suite_name="pair2",
        verbose=0,
    )
    assert isinstance(algo.rollout_buffer, MaskedRecurrentRolloutBuffer)
    assert algo.rollout_buffer._actor_objective_mode == "tokenwise_scale_env12_high_mass_objective_episode_family"
    assert algo.rollout_buffer._actor_objective_runtime_current_suite_name == "pair2"


def test_build_recurrent_ppo_accepts_pair2_recovered_anchor_first_objective_non_tail_family_mode():
    device = _cuda_or_skip()
    cfg, env_cfg = _build_small_ppo_config()
    env_cfg = _strict_env_cfg_for_ppo(env_cfg)
    cfg["optimizer"]["ppo_n_envs"] = 16
    fake_model = _FakeRWKVModel(
        num_features=cfg["prior"]["num_features"],
        x_obs_dim=cfg["transformer"]["x_obs_dim"],
        action_dim=cfg["transformer"]["x_action_dim"],
    ).to(device)
    env_prior = EnvironmentPrior(env_cfg)
    algo, _, _ = build_recurrent_ppo(
        model=fake_model,
        env_prior=env_prior,
        device=str(device),
        num_features=cfg["prior"]["num_features"],
        n_envs=cfg["optimizer"]["ppo_n_envs"],
        n_steps=cfg["optimizer"]["ppo_n_steps"],
        learning_rate=3e-4,
        batch_size=4,
        n_epochs=1,
        gamma=1.0,
        gae_lambda=0.95,
        clip_range=0.2,
        clip_range_vf=None,
        normalize_advantage=True,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        target_kl=None,
        strict_fixed_env_mode=True,
        actor_gae_space="normalized",
        actor_baseline_mode="learned",
        actor_objective_mode="tokenwise_scale_pair2_recovered_anchor_first_objective_non_tail_family",
        actor_objective_runtime_current_suite_name="pair2",
        verbose=0,
    )
    assert isinstance(algo.rollout_buffer, MaskedRecurrentRolloutBuffer)
    assert (
        algo.rollout_buffer._actor_objective_mode
        == "tokenwise_scale_pair2_recovered_anchor_first_objective_non_tail_family"
    )
    assert algo.rollout_buffer._actor_objective_runtime_current_suite_name == "pair2"


def test_build_recurrent_ppo_rejects_env12_mid_episode_extension_mode_without_strict_fixed_env():
    device = _cuda_or_skip()
    cfg, env_cfg = _build_small_ppo_config()
    env_cfg = _strict_env_cfg_for_ppo(env_cfg)
    cfg["optimizer"]["ppo_n_envs"] = 13
    fake_model = _FakeRWKVModel(
        num_features=cfg["prior"]["num_features"],
        x_obs_dim=cfg["transformer"]["x_obs_dim"],
        action_dim=cfg["transformer"]["x_action_dim"],
    ).to(device)
    env_prior = EnvironmentPrior(env_cfg)
    try:
        build_recurrent_ppo(
            model=fake_model,
            env_prior=env_prior,
            device=str(device),
            num_features=cfg["prior"]["num_features"],
            n_envs=cfg["optimizer"]["ppo_n_envs"],
            n_steps=cfg["optimizer"]["ppo_n_steps"],
            learning_rate=3e-4,
            batch_size=4,
            n_epochs=1,
            gamma=1.0,
            gae_lambda=0.95,
            clip_range=0.2,
            clip_range_vf=None,
            normalize_advantage=True,
            ent_coef=0.0,
            vf_coef=0.5,
            max_grad_norm=0.5,
            target_kl=None,
            actor_gae_space="normalized",
            actor_baseline_mode="learned",
            actor_objective_mode="tokenwise_scale_env12_mid_episode_extension_block",
            actor_objective_runtime_current_suite_name="pair2",
            verbose=0,
        )
    except ValueError as exc:
        assert "strict_fixed_env_mode=True" in str(exc)
    else:
        raise AssertionError("expected strict_fixed_env_mode guard to raise")


def test_build_recurrent_ppo_rejects_env12_mid_episode_extension_mode_without_pair2_scope():
    device = _cuda_or_skip()
    cfg, env_cfg = _build_small_ppo_config()
    env_cfg = _strict_env_cfg_for_ppo(env_cfg)
    cfg["optimizer"]["ppo_n_envs"] = 13
    fake_model = _FakeRWKVModel(
        num_features=cfg["prior"]["num_features"],
        x_obs_dim=cfg["transformer"]["x_obs_dim"],
        action_dim=cfg["transformer"]["x_action_dim"],
    ).to(device)
    env_prior = EnvironmentPrior(env_cfg)
    try:
        build_recurrent_ppo(
            model=fake_model,
            env_prior=env_prior,
            device=str(device),
            num_features=cfg["prior"]["num_features"],
            n_envs=cfg["optimizer"]["ppo_n_envs"],
            n_steps=cfg["optimizer"]["ppo_n_steps"],
            learning_rate=3e-4,
            batch_size=4,
            n_epochs=1,
            gamma=1.0,
            gae_lambda=0.95,
            clip_range=0.2,
            clip_range_vf=None,
            normalize_advantage=True,
            ent_coef=0.0,
            vf_coef=0.5,
            max_grad_norm=0.5,
            target_kl=None,
            strict_fixed_env_mode=True,
            actor_gae_space="normalized",
            actor_baseline_mode="learned",
            actor_objective_mode="tokenwise_scale_env12_mid_episode_extension_block",
            actor_objective_runtime_current_suite_name="seed_24680",
            verbose=0,
        )
    except ValueError as exc:
        assert "reserved for pair2 runtime scope" in str(exc)
    else:
        raise AssertionError("expected pair2 scope guard to raise")


def test_build_recurrent_ppo_accepts_boundary_local_actor_objective_mode():
    device = _cuda_or_skip()
    cfg, env_cfg = _build_small_ppo_config()
    env_cfg = _strict_env_cfg_for_ppo(env_cfg)
    fake_model = _FakeRWKVModel(
        num_features=cfg["prior"]["num_features"],
        x_obs_dim=cfg["transformer"]["x_obs_dim"],
        action_dim=cfg["transformer"]["x_action_dim"],
    ).to(device)
    env_prior = EnvironmentPrior(env_cfg)
    algo, _, _ = build_recurrent_ppo(
        model=fake_model,
        env_prior=env_prior,
        device=str(device),
        num_features=cfg["prior"]["num_features"],
        n_envs=cfg["optimizer"]["ppo_n_envs"],
        n_steps=cfg["optimizer"]["ppo_n_steps"],
        learning_rate=3e-4,
        batch_size=4,
        n_epochs=1,
        gamma=1.0,
        gae_lambda=0.95,
        clip_range=0.2,
        clip_range_vf=None,
        normalize_advantage=True,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        target_kl=None,
        actor_gae_space="normalized",
        actor_baseline_mode="learned",
        actor_objective_mode="tokenwise_linear_ramp_first_8_suffix_steps",
        verbose=0,
    )
    assert isinstance(algo.rollout_buffer, MaskedRecurrentRolloutBuffer)
    assert algo.rollout_buffer._actor_objective_mode == "tokenwise_linear_ramp_first_8_suffix_steps"


def test_build_validation_recurrent_ppo_policy_accepts_legacy_scalar_value_head_state_dict():
    device = _cuda_or_skip()
    cfg, env_cfg = _build_small_ppo_config()
    env_cfg = _strict_env_cfg_for_ppo(env_cfg)
    fake_model = _FakeRWKVModel(
        num_features=cfg["prior"]["num_features"],
        x_obs_dim=cfg["transformer"]["x_obs_dim"],
        action_dim=cfg["transformer"]["x_action_dim"],
        emsize=12,
    ).to(device)
    fresh_policy = OfficialRWKVRecurrentPPOPolicy(
        observation_space=spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(cfg["prior"]["num_features"],),
            dtype=np.float32,
        ),
        action_space=spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(cfg["transformer"]["x_action_dim"],),
            dtype=np.float32,
        ),
        lr_schedule=lambda _: 1e-3,
        rlpfn_model=fake_model,
        num_features=cfg["prior"]["num_features"],
        obs_slot_dim=resolve_rlpfn_token_layout(env_cfg, num_features=cfg["prior"]["num_features"])["obs_slot_dim"],
        net_arch=[],
    ).to(device)
    legacy_state = {
        key: value.detach().cpu()
        for key, value in fresh_policy.state_dict().items()
        if not str(key).startswith("value_bardist.")
    }
    legacy_state["value_net.weight"] = torch.zeros(
        (1, int(fresh_policy.mlp_extractor.latent_dim_vf)),
        dtype=torch.float32,
    )
    legacy_state["value_net.bias"] = torch.zeros((1,), dtype=torch.float32)

    restored = build_validation_recurrent_ppo_policy(
        model=fake_model,
        env_cfg=env_cfg,
        device=device,
        num_features=cfg["prior"]["num_features"],
        policy_state_dict=legacy_state,
    )
    assert restored.value_net.out_features == int(restored.get_value_bardist().num_bars)


def test_build_recurrent_ppo_can_enable_separate_value_backbone():
    device = _cuda_or_skip()
    cfg, env_cfg = _build_small_ppo_config()
    env_cfg = _strict_env_cfg_for_ppo(env_cfg)
    env_prior = EnvironmentPrior(deepcopy(env_cfg))
    fake_model = _FakeRWKVModel(
        num_features=cfg["prior"]["num_features"],
        x_obs_dim=cfg["transformer"]["x_obs_dim"],
        action_dim=cfg["transformer"]["x_action_dim"],
        emsize=12,
    ).to(device)
    algo, callback, vec_env = build_recurrent_ppo(
        model=fake_model,
        env_prior=env_prior,
        device=str(device),
        num_features=cfg["prior"]["num_features"],
        n_envs=cfg["optimizer"]["ppo_n_envs"],
        n_steps=cfg["optimizer"]["ppo_n_steps"],
        learning_rate=3e-4,
        batch_size=4,
        n_epochs=1,
        gamma=1.0,
        gae_lambda=0.95,
        clip_range=0.2,
        clip_range_vf=None,
        normalize_advantage=True,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        target_kl=None,
        separate_value_backbone=True,
        actor_gae_space="normalized",
        actor_baseline_mode="learned",
        actor_objective_mode="tokenwise",
        verbose=0,
    )
    del callback
    try:
        policy = algo.policy
        assert policy.separate_value_backbone is True
        assert policy.value_rlpfn_model is not None
        actor_param = next(policy.rlpfn_model.parameters())
        value_param = next(policy.value_rlpfn_model.parameters())
        assert actor_param.data_ptr() != value_param.data_ptr()
        assert torch.allclose(actor_param, value_param)
    finally:
        vec_env.close()


def test_build_recurrent_ppo_accepts_scalar_linear_value_head():
    device = _cuda_or_skip()
    cfg, env_cfg = _build_small_ppo_config()
    env_cfg = _strict_env_cfg_for_ppo(env_cfg)
    env_prior = EnvironmentPrior(deepcopy(env_cfg))
    fake_model = _FakeRWKVModel(
        num_features=cfg["prior"]["num_features"],
        x_obs_dim=cfg["transformer"]["x_obs_dim"],
        action_dim=cfg["transformer"]["x_action_dim"],
        emsize=12,
    ).to(device)
    algo, callback, vec_env = build_recurrent_ppo(
        model=fake_model,
        env_prior=env_prior,
        device=str(device),
        num_features=cfg["prior"]["num_features"],
        n_envs=cfg["optimizer"]["ppo_n_envs"],
        n_steps=cfg["optimizer"]["ppo_n_steps"],
        learning_rate=3e-4,
        batch_size=4,
        n_epochs=1,
        gamma=1.0,
        gae_lambda=0.95,
        clip_range=0.2,
        clip_range_vf=None,
        normalize_advantage=True,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        target_kl=None,
        value_head_impl="scalar_linear",
        actor_gae_space="normalized",
        actor_baseline_mode="learned",
        actor_objective_mode="tokenwise",
        verbose=0,
    )
    del callback
    try:
        assert algo.policy.value_head_impl == "scalar_linear"
        assert algo.policy.has_value_bardist() is False
        assert int(algo.policy.value_net.out_features) == 1
    finally:
        vec_env.close()


def test_build_recurrent_ppo_accepts_batchnorm_value_path_adapter():
    device = _cuda_or_skip()
    cfg, env_cfg = _build_small_ppo_config()
    env_cfg = _strict_env_cfg_for_ppo(env_cfg)
    env_prior = EnvironmentPrior(deepcopy(env_cfg))
    fake_model = _FakeRWKVModel(
        num_features=cfg["prior"]["num_features"],
        x_obs_dim=cfg["transformer"]["x_obs_dim"],
        action_dim=cfg["transformer"]["x_action_dim"],
        emsize=12,
    ).to(device)
    algo, callback, vec_env = build_recurrent_ppo(
        model=fake_model,
        env_prior=env_prior,
        device=str(device),
        num_features=cfg["prior"]["num_features"],
        n_envs=cfg["optimizer"]["ppo_n_envs"],
        n_steps=cfg["optimizer"]["ppo_n_steps"],
        learning_rate=3e-4,
        batch_size=4,
        n_epochs=1,
        gamma=1.0,
        gae_lambda=0.95,
        clip_range=0.2,
        clip_range_vf=None,
        normalize_advantage=True,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        target_kl=None,
        value_path_adapter_impl="batchnorm",
        actor_gae_space="normalized",
        actor_baseline_mode="learned",
        actor_objective_mode="tokenwise",
        verbose=0,
    )
    del callback
    try:
        assert algo.policy.value_path_adapter_impl == "batchnorm"
        assert isinstance(algo.policy.value_path_adapter, sb3_recurrent_ppo_module._ValuePathBatchNormAdapter)
    finally:
        vec_env.close()


def test_build_recurrent_ppo_accepts_vendor_official_and_frozen_zscore():
    device = _cuda_or_skip()
    cfg, env_cfg = _build_small_ppo_config()
    env_cfg = _strict_env_cfg_for_ppo(env_cfg)
    env_prior = EnvironmentPrior(deepcopy(env_cfg))
    fake_model = _FakeRWKVModel(
        num_features=cfg["prior"]["num_features"],
        x_obs_dim=cfg["transformer"]["x_obs_dim"],
        action_dim=cfg["transformer"]["x_action_dim"],
        emsize=12,
    ).to(device)
    algo, callback, vec_env = build_recurrent_ppo(
        model=fake_model,
        env_prior=env_prior,
        device=str(device),
        num_features=cfg["prior"]["num_features"],
        n_envs=cfg["optimizer"]["ppo_n_envs"],
        n_steps=cfg["optimizer"]["ppo_n_steps"],
        learning_rate=3e-4,
        batch_size=4,
        n_epochs=1,
        gamma=1.0,
        gae_lambda=0.95,
        clip_range=0.2,
        clip_range_vf=None,
        normalize_advantage=True,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        target_kl=None,
        value_head_impl="vendor_official",
        value_path_adapter_impl="frozen_zscore",
        actor_gae_space="normalized",
        actor_baseline_mode="learned",
        actor_objective_mode="tokenwise",
        verbose=0,
    )
    del callback
    try:
        assert algo.policy.value_head_impl == "vendor_official"
        assert algo.policy.value_path_adapter_impl == "frozen_zscore"
        assert algo.policy.has_value_vendor_head() is True
        assert isinstance(algo.policy.value_path_adapter, sb3_recurrent_ppo_module._ValuePathFrozenZScoreAdapter)
    finally:
        vec_env.close()


def test_train_refits_frozen_zscore_adapter_each_epoch(monkeypatch):
    device = _cuda_or_skip()
    cfg, env_cfg = _build_small_ppo_config()
    env_cfg = _strict_env_cfg_for_ppo(env_cfg)
    env_prior = EnvironmentPrior(deepcopy(env_cfg))
    fake_model = _FakeRWKVModel(
        num_features=cfg["prior"]["num_features"],
        x_obs_dim=cfg["transformer"]["x_obs_dim"],
        action_dim=cfg["transformer"]["x_action_dim"],
        emsize=12,
    ).to(device)
    algo, callback, vec_env = build_recurrent_ppo(
        model=fake_model,
        env_prior=env_prior,
        device=str(device),
        num_features=cfg["prior"]["num_features"],
        n_envs=cfg["optimizer"]["ppo_n_envs"],
        n_steps=cfg["optimizer"]["ppo_n_steps"],
        learning_rate=cfg["optimizer"]["learning_rate"],
        batch_size=cfg["optimizer"]["ppo_batch_size"],
        n_epochs=2,
        gamma=cfg["optimizer"]["ppo_gamma"],
        gae_lambda=cfg["optimizer"]["ppo_gae_lambda"],
        clip_range=cfg["optimizer"]["ppo_clip_range"],
        clip_range_vf=cfg["optimizer"]["ppo_clip_range_vf"],
        normalize_advantage=cfg["optimizer"]["ppo_normalize_advantage"],
        ent_coef=cfg["optimizer"]["ppo_ent_coef"],
        vf_coef=cfg["optimizer"]["ppo_vf_coef"],
        max_grad_norm=cfg["optimizer"]["ppo_max_grad_norm"],
        target_kl=cfg["optimizer"]["ppo_target_kl"],
        value_head_impl="vendor_official",
        value_path_adapter_impl="frozen_zscore",
        actor_gae_space="normalized",
        actor_baseline_mode="learned",
        actor_objective_mode="tokenwise",
        verbose=0,
    )
    try:
        algo.ep_info_buffer = []
        algo.ep_success_buffer = []
        algo._last_obs = vec_env.reset()
        algo._last_episode_starts = np.ones((vec_env.num_envs,), dtype=bool)
        algo._last_lstm_states = algo.policy._dummy_states(vec_env.num_envs)
        callback.init_callback(algo)
        assert algo.collect_rollouts(vec_env, callback, algo.rollout_buffer, n_rollout_steps=algo.n_steps)

        calls = 0
        original = algo.policy.fit_value_path_adapter_from_rollout_steps

        def _wrapped(*args, **kwargs):
            nonlocal calls
            calls += 1
            return original(*args, **kwargs)

        monkeypatch.setattr(algo.policy, "fit_value_path_adapter_from_rollout_steps", _wrapped)
        algo._logger = SimpleNamespace(record=lambda *args, **kwargs: None)
        algo.train()
        assert calls == int(algo.n_epochs)
    finally:
        vec_env.close()


def test_vendor_official_value_loss_ignores_non_objective_without_nan_gradients():
    device = _cuda_or_skip()
    num_features = 8
    obs_slot_dim = 4
    action_dim = 2
    fake_model = _FakeRWKVModel(
        num_features=num_features,
        x_obs_dim=6,
        action_dim=action_dim,
        emsize=12,
    ).to(device)
    policy = OfficialRWKVRecurrentPPOPolicy(
        observation_space=spaces.Box(low=-np.inf, high=np.inf, shape=(num_features,), dtype=np.float32),
        action_space=spaces.Box(low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32),
        lr_schedule=lambda _: 1e-3,
        rlpfn_model=fake_model,
        num_features=num_features,
        obs_slot_dim=obs_slot_dim,
        value_head_impl="vendor_official",
        net_arch=[],
    ).to(device)

    value_logits = torch.randn(
        (6, int(policy.get_value_bardist().num_bars)),
        device=device,
        dtype=torch.float32,
        requires_grad=True,
    )
    targets = torch.tensor([0.0, 0.2, -0.4, 0.8, -1.2, 0.3], device=device, dtype=torch.float32)
    objective_mask = torch.tensor([0, 0, 1, 1, 1, 1], device=device, dtype=torch.bool)
    loss = policy.compute_value_loss_from_logits(
        value_logits=value_logits,
        targets=targets,
        objective_mask=objective_mask,
        value_target_bucket_idx=None,
    )
    grad = torch.autograd.grad(loss, value_logits)[0]
    assert torch.isfinite(loss)
    assert torch.isfinite(grad).all()


def test_build_validation_recurrent_ppo_policy_restores_separate_value_backbone_state():
    device = _cuda_or_skip()
    cfg, env_cfg = _build_small_ppo_config()
    env_cfg = _strict_env_cfg_for_ppo(env_cfg)
    fake_model = _FakeRWKVModel(
        num_features=cfg["prior"]["num_features"],
        x_obs_dim=cfg["transformer"]["x_obs_dim"],
        action_dim=cfg["transformer"]["x_action_dim"],
        emsize=12,
    ).to(device)
    fresh_policy = OfficialRWKVRecurrentPPOPolicy(
        observation_space=spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(cfg["prior"]["num_features"],),
            dtype=np.float32,
        ),
        action_space=spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(cfg["transformer"]["x_action_dim"],),
            dtype=np.float32,
        ),
        lr_schedule=lambda _: 1e-3,
        rlpfn_model=fake_model,
        num_features=cfg["prior"]["num_features"],
        obs_slot_dim=resolve_rlpfn_token_layout(env_cfg, num_features=cfg["prior"]["num_features"])["obs_slot_dim"],
        separate_value_backbone=True,
        net_arch=[],
    ).to(device)
    policy_state = extract_validation_recurrent_ppo_policy_state(fresh_policy)
    restored = build_validation_recurrent_ppo_policy(
        model=fake_model,
        env_cfg=env_cfg,
        device=device,
        num_features=cfg["prior"]["num_features"],
        policy_state_dict=policy_state,
    )
    assert restored.separate_value_backbone is True
    assert restored.value_rlpfn_model is not None


def test_build_validation_recurrent_ppo_policy_can_force_native_eval_forward_step():
    device = _cuda_or_skip()
    cfg, env_cfg = _build_small_ppo_config()
    env_cfg = _strict_env_cfg_for_ppo(env_cfg)
    fake_model = _FakeRWKVModel(
        num_features=cfg["prior"]["num_features"],
        x_obs_dim=cfg["transformer"]["x_obs_dim"],
        action_dim=cfg["transformer"]["x_action_dim"],
        emsize=12,
    ).to(device)
    restored = build_validation_recurrent_ppo_policy(
        model=fake_model,
        env_cfg=env_cfg,
        device=device,
        num_features=cfg["prior"]["num_features"],
        strict_native_rollout=True,
    )

    assert getattr(restored.rlpfn_model.rwkv_core, "force_native_eval_forward_step", False) is True


def test_build_recurrent_ppo_can_restore_saved_validation_policy_state():
    device = _cuda_or_skip()
    cfg, env_cfg = _build_small_ppo_config()
    env_cfg = _strict_env_cfg_for_ppo(env_cfg)
    prior = EnvironmentPrior(env_cfg)
    fake_model = _FakeRWKVModel(
        num_features=cfg["prior"]["num_features"],
        x_obs_dim=cfg["transformer"]["x_obs_dim"],
        action_dim=cfg["transformer"]["x_action_dim"],
        emsize=12,
    ).to(device)
    policy_template = OfficialRWKVRecurrentPPOPolicy(
        observation_space=spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(cfg["prior"]["num_features"],),
            dtype=np.float32,
        ),
        action_space=spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(cfg["transformer"]["x_action_dim"],),
            dtype=np.float32,
        ),
        lr_schedule=lambda _: 1e-3,
        rlpfn_model=fake_model,
        num_features=cfg["prior"]["num_features"],
        obs_slot_dim=resolve_rlpfn_token_layout(env_cfg, num_features=cfg["prior"]["num_features"])["obs_slot_dim"],
        net_arch=[],
    ).to(device)
    saved_policy_state = extract_validation_recurrent_ppo_policy_state(policy_template)
    fake_model.__dict__["_validation_ppo_policy_state"] = saved_policy_state

    algo, _, vec_env = build_recurrent_ppo(
        model=fake_model,
        env_prior=prior,
        device=str(device),
        num_features=cfg["prior"]["num_features"],
        n_envs=cfg["optimizer"]["ppo_n_envs"],
        n_steps=cfg["optimizer"]["ppo_n_steps"],
        learning_rate=cfg["optimizer"]["learning_rate"],
        batch_size=cfg["optimizer"]["ppo_batch_size"],
        n_epochs=cfg["optimizer"]["ppo_n_epochs"],
        gamma=cfg["optimizer"]["ppo_gamma"],
        gae_lambda=cfg["optimizer"]["ppo_gae_lambda"],
        clip_range=cfg["optimizer"]["ppo_clip_range"],
        clip_range_vf=cfg["optimizer"]["ppo_clip_range_vf"],
        normalize_advantage=cfg["optimizer"]["ppo_normalize_advantage"],
        ent_coef=cfg["optimizer"]["ppo_ent_coef"],
        vf_coef=cfg["optimizer"]["ppo_vf_coef"],
        max_grad_norm=cfg["optimizer"]["ppo_max_grad_norm"],
        target_kl=cfg["optimizer"]["ppo_target_kl"],
        restore_validation_policy_state=True,
    )
    try:
        restored_state = algo.policy.state_dict()
        assert torch.allclose(restored_state["action_net.weight"], saved_policy_state["action_net.weight"].to(device))
        assert torch.allclose(restored_state["value_net.weight"], saved_policy_state["value_net.weight"].to(device))
        assert torch.allclose(restored_state["log_std"], saved_policy_state["log_std"].to(device))
    finally:
        vec_env.close()


def test_policy_evaluate_actions_respects_rwkv_inner_microbatch_resolver():
    device = _cuda_or_skip()
    num_features = 8
    obs_slot_dim = 4
    action_dim = 2
    fake_model = _FakeRWKVModel(
        num_features=num_features,
        x_obs_dim=6,
        action_dim=action_dim,
        emsize=12,
        replay_batch_chunk_size=1,
    ).to(device)
    policy = OfficialRWKVRecurrentPPOPolicy(
        observation_space=spaces.Box(low=-np.inf, high=np.inf, shape=(num_features,), dtype=np.float32),
        action_space=spaces.Box(low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32),
        lr_schedule=lambda _: 1e-3,
        rlpfn_model=fake_model,
        num_features=num_features,
        obs_slot_dim=obs_slot_dim,
        net_arch=[],
    ).to(device)
    policy.train()
    obs = torch.randn((6, num_features), device=device, dtype=torch.float32)
    episode_starts = torch.tensor([1.0, 0.0, 0.0, 1.0, 0.0, 0.0], device=device, dtype=torch.float32)
    actions = torch.zeros((6, action_dim), device=device, dtype=torch.float32)
    values, log_prob, entropy = policy.evaluate_actions(
        obs,
        actions,
        policy._dummy_states(2),
        episode_starts,
    )
    assert values.shape == (6, 1)
    assert log_prob.shape == (6,)
    assert entropy.shape == (6,)
    assert fake_model.replay_chunk_requests == [(3, 2)]
    assert fake_model.rwkv_core.sequence_calls == 2
    assert fake_model.encode_sequence_calls == 2
    assert fake_model.encode_sequence_shapes == [(3, 1, num_features), (3, 1, num_features)]


def test_resolve_official_ppo_update_batch_size_tracks_sequence_batch_capacity():
    fake_model = _FakeRWKVModel(
        num_features=8,
        x_obs_dim=6,
        action_dim=2,
        replay_batch_chunk_size=64,
    )
    resolved = resolve_official_ppo_update_batch_size(
        model=fake_model,
        n_envs=2048,
        n_steps=2048,
        configured_batch_size=None,
    )
    assert resolved == 131072


def test_build_recurrent_ppo_rejects_value_clip_with_bar_distribution_head():
    device = _cuda_or_skip()
    cfg, env_cfg = _build_small_ppo_config()
    env_cfg = _strict_env_cfg_for_ppo(env_cfg)
    prior = EnvironmentPrior(env_cfg)
    fake_model = _FakeRWKVModel(
        num_features=cfg["prior"]["num_features"],
        x_obs_dim=cfg["transformer"]["x_obs_dim"],
        action_dim=cfg["transformer"]["x_action_dim"],
    ).to(device)
    with pytest.raises(ValueError, match="ppo_clip_range_vf=None"):
        build_recurrent_ppo(
            model=fake_model,
            env_prior=prior,
            device=str(device),
            num_features=cfg["prior"]["num_features"],
            n_envs=cfg["optimizer"]["ppo_n_envs"],
            n_steps=cfg["optimizer"]["ppo_n_steps"],
            learning_rate=cfg["optimizer"]["learning_rate"],
            batch_size=cfg["optimizer"]["ppo_batch_size"],
            n_epochs=cfg["optimizer"]["ppo_n_epochs"],
            gamma=cfg["optimizer"]["ppo_gamma"],
            gae_lambda=cfg["optimizer"]["ppo_gae_lambda"],
            clip_range=cfg["optimizer"]["ppo_clip_range"],
            clip_range_vf=0.1,
            normalize_advantage=cfg["optimizer"]["ppo_normalize_advantage"],
            ent_coef=cfg["optimizer"]["ppo_ent_coef"],
            vf_coef=cfg["optimizer"]["ppo_vf_coef"],
            max_grad_norm=cfg["optimizer"]["ppo_max_grad_norm"],
            target_kl=cfg["optimizer"]["ppo_target_kl"],
        )


def test_policy_masked_log_prob_ignores_inactive_action_dims():
    device = _cuda_or_skip()
    num_features = 8
    obs_slot_dim = 4
    action_dim = 4
    fake_model = _FakeRWKVModel(
        num_features=num_features,
        x_obs_dim=6,
        action_dim=action_dim,
        emsize=12,
    ).to(device)
    policy = OfficialRWKVRecurrentPPOPolicy(
        observation_space=spaces.Box(low=-np.inf, high=np.inf, shape=(num_features,), dtype=np.float32),
        action_space=spaces.Box(low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32),
        lr_schedule=lambda _: 1e-3,
        rlpfn_model=fake_model,
        num_features=num_features,
        obs_slot_dim=obs_slot_dim,
        net_arch=[],
    ).to(device)
    policy.train()
    obs = torch.randn((4, num_features), device=device, dtype=torch.float32)
    episode_starts = torch.tensor([1.0, 0.0, 1.0, 0.0], device=device, dtype=torch.float32)
    action_masks = torch.tensor(
        [
            [1.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 0.0],
            [1.0, 1.0, 0.0, 0.0],
        ],
        device=device,
        dtype=torch.float32,
    )
    actions_a = torch.zeros((4, action_dim), device=device, dtype=torch.float32)
    actions_b = actions_a.clone()
    actions_b = actions_b + torch.randn((4, action_dim), device=device, dtype=torch.float32) * (1.0 - action_masks)
    _, log_prob_a, entropy_a = policy.evaluate_actions(
        obs,
        actions_a,
        policy._dummy_states(2),
        episode_starts,
        action_masks=action_masks,
    )
    _, log_prob_b, entropy_b = policy.evaluate_actions(
        obs,
        actions_b,
        policy._dummy_states(2),
        episode_starts,
        action_masks=action_masks,
    )
    assert torch.allclose(log_prob_a, log_prob_b, atol=1e-6, rtol=1e-6)
    assert torch.allclose(entropy_a, entropy_b, atol=1e-6, rtol=1e-6)


def test_build_recurrent_ppo_accepts_aux_heads_when_enabled():
    device = _cuda_or_skip()
    cfg, env_cfg = _build_small_ppo_config()
    env_cfg = _strict_env_cfg_for_ppo(env_cfg, aux=True)
    prior = EnvironmentPrior(env_cfg)
    fake_model = _FakeRWKVModel(
        num_features=cfg["prior"]["num_features"],
        x_obs_dim=cfg["transformer"]["x_obs_dim"],
        action_dim=cfg["transformer"]["x_action_dim"],
        normalized_q_head=True,
        next_state_flow_dim=8,
    ).to(device)
    algo, _, vec_env = build_recurrent_ppo(
        model=fake_model,
        env_prior=prior,
        device=str(device),
        num_features=cfg["prior"]["num_features"],
        n_envs=cfg["optimizer"]["ppo_n_envs"],
        n_steps=cfg["optimizer"]["ppo_n_steps"],
        learning_rate=cfg["optimizer"]["learning_rate"],
        batch_size=cfg["optimizer"]["ppo_batch_size"],
        n_epochs=1,
        gamma=cfg["optimizer"]["ppo_gamma"],
        gae_lambda=cfg["optimizer"]["ppo_gae_lambda"],
        clip_range=cfg["optimizer"]["ppo_clip_range"],
        clip_range_vf=cfg["optimizer"]["ppo_clip_range_vf"],
        normalize_advantage=cfg["optimizer"]["ppo_normalize_advantage"],
        ent_coef=cfg["optimizer"]["ppo_ent_coef"],
        vf_coef=cfg["optimizer"]["ppo_vf_coef"],
        max_grad_norm=cfg["optimizer"]["ppo_max_grad_norm"],
        target_kl=cfg["optimizer"]["ppo_target_kl"],
    )
    assert float(algo._rwkv_aux_q_weight) == 1.0
    assert float(algo._rwkv_aux_flow_weight) == 1.0
    vec_env.close()


def test_build_recurrent_ppo_disables_q_aux_by_default():
    device = _cuda_or_skip()
    cfg, env_cfg = _build_small_ppo_config()
    env_cfg = deepcopy(env_cfg)
    env_cfg["family"] = {"distribution": "meta_choice", "choice_values": ["scm"]}
    env_cfg["action_dim"] = int(env_cfg["action_slot_dim"])
    env_cfg["state_dim"] = 8
    env_cfg["obs_dim"] = 8
    env_cfg["noise_dim"] = 4
    env_cfg["zero_pad_dim"] = 0
    env_cfg["alpha"] = 0.2
    env_cfg["init_state_std"] = 0.1
    env_cfg["init_action_std"] = 0.1
    env_cfg["state_noise_std"] = 0.01
    env_cfg["action_noise_train_std"] = 0.0
    env_cfg["action_noise_eval_std"] = 0.0
    env_cfg["reward_scale"] = 1.0
    env_cfg["reward_clip"] = 10.0
    env_cfg["reinforce_action_transform"] = "none"
    assert env_cfg["normalized_q_value_weight"] == 0.0
    assert env_cfg["next_state_flow_matching_weight"] == 0.0
    prior = EnvironmentPrior(env_cfg)
    fake_model = _FakeRWKVModel(
        num_features=cfg["prior"]["num_features"],
        x_obs_dim=cfg["transformer"]["x_obs_dim"],
        action_dim=cfg["transformer"]["x_action_dim"],
        normalized_q_head=False,
        next_state_flow_dim=8,
    ).to(device)
    algo, _, vec_env = build_recurrent_ppo(
        model=fake_model,
        env_prior=prior,
        device=str(device),
        num_features=cfg["prior"]["num_features"],
        n_envs=cfg["optimizer"]["ppo_n_envs"],
        n_steps=cfg["optimizer"]["ppo_n_steps"],
        learning_rate=cfg["optimizer"]["learning_rate"],
        batch_size=cfg["optimizer"]["ppo_batch_size"],
        n_epochs=1,
        gamma=cfg["optimizer"]["ppo_gamma"],
        gae_lambda=cfg["optimizer"]["ppo_gae_lambda"],
        clip_range=cfg["optimizer"]["ppo_clip_range"],
        clip_range_vf=cfg["optimizer"]["ppo_clip_range_vf"],
        normalize_advantage=cfg["optimizer"]["ppo_normalize_advantage"],
        ent_coef=cfg["optimizer"]["ppo_ent_coef"],
        vf_coef=cfg["optimizer"]["ppo_vf_coef"],
        max_grad_norm=cfg["optimizer"]["ppo_max_grad_norm"],
        target_kl=cfg["optimizer"]["ppo_target_kl"],
    )
    assert float(algo._rwkv_aux_q_weight) == 0.0
    assert float(algo._rwkv_aux_flow_weight) == 0.0
    vec_env.close()


def test_policy_decode_aux_from_hidden_restores_time_major_flow_shape():
    device = _cuda_or_skip()
    fake_model = _FakeRWKVModel(
        num_features=16,
        x_obs_dim=8,
        action_dim=4,
        normalized_q_head=True,
        next_state_flow_dim=8,
    ).to(device)
    observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(16,), dtype=np.float32)
    action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
    lr_schedule = lambda _: 1e-4
    policy = OfficialRWKVRecurrentPPOPolicy(
        observation_space,
        action_space,
        lr_schedule,
        rlpfn_model=fake_model,
        num_features=16,
        obs_slot_dim=8,
        net_arch=[],
    ).to(device)

    hidden = torch.randn((6, fake_model.emsize), device=device)
    actions = torch.randn((6, 4), device=device)
    flow_xt = torch.randn((6, 8), device=device)
    flow_t = torch.rand((6, 1), device=device)

    outputs = policy.decode_aux_from_hidden(
        hidden,
        actions=actions,
        flow_matching_xt=flow_xt,
        flow_matching_t=flow_t,
        n_seq=2,
    )
    assert tuple(outputs["next_state_flow"].shape) == (6, 8)
    assert fake_model.last_next_state_flow_hidden_shape == (3, 2, fake_model.emsize)


def test_masked_recurrent_ppo_train_smoke_with_aux_enabled():
    device = _cuda_or_skip()
    cfg, env_cfg = _build_small_ppo_config()
    env_cfg = _strict_env_cfg_for_ppo(env_cfg, aux=True)
    prior = EnvironmentPrior(env_cfg)
    fake_model = _FakeRWKVModel(
        num_features=cfg["prior"]["num_features"],
        x_obs_dim=cfg["transformer"]["x_obs_dim"],
        action_dim=cfg["transformer"]["x_action_dim"],
        normalized_q_head=True,
        next_state_flow_dim=8,
    ).to(device)
    algo, _, vec_env = build_recurrent_ppo(
        model=fake_model,
        env_prior=prior,
        device=str(device),
        num_features=cfg["prior"]["num_features"],
        n_envs=cfg["optimizer"]["ppo_n_envs"],
        n_steps=cfg["optimizer"]["ppo_n_steps"],
        learning_rate=cfg["optimizer"]["learning_rate"],
        batch_size=cfg["optimizer"]["ppo_batch_size"],
        n_epochs=1,
        gamma=cfg["optimizer"]["ppo_gamma"],
        gae_lambda=cfg["optimizer"]["ppo_gae_lambda"],
        clip_range=cfg["optimizer"]["ppo_clip_range"],
        clip_range_vf=cfg["optimizer"]["ppo_clip_range_vf"],
        normalize_advantage=cfg["optimizer"]["ppo_normalize_advantage"],
        ent_coef=cfg["optimizer"]["ppo_ent_coef"],
        vf_coef=cfg["optimizer"]["ppo_vf_coef"],
        max_grad_norm=cfg["optimizer"]["ppo_max_grad_norm"],
        target_kl=cfg["optimizer"]["ppo_target_kl"],
    )
    algo.learn(total_timesteps=int(cfg["optimizer"]["ppo_n_envs"] * cfg["optimizer"]["ppo_n_steps"]))
    assert fake_model.normalized_q_decode_calls > 0
    assert fake_model.next_state_flow_decode_calls > 0
    vec_env.close()


def test_masked_recurrent_ppo_train_reuses_main_hidden_for_aux(monkeypatch):
    device = _cuda_or_skip()
    cfg, env_cfg = _build_small_ppo_config()
    env_cfg = _strict_env_cfg_for_ppo(env_cfg, aux=True)
    prior = EnvironmentPrior(env_cfg)
    fake_model = _FakeRWKVModel(
        num_features=cfg["prior"]["num_features"],
        x_obs_dim=cfg["transformer"]["x_obs_dim"],
        action_dim=cfg["transformer"]["x_action_dim"],
        normalized_q_head=True,
        next_state_flow_dim=8,
    ).to(device)
    algo, callback, vec_env = build_recurrent_ppo(
        model=fake_model,
        env_prior=prior,
        device=str(device),
        num_features=cfg["prior"]["num_features"],
        n_envs=cfg["optimizer"]["ppo_n_envs"],
        n_steps=cfg["optimizer"]["ppo_n_steps"],
        learning_rate=cfg["optimizer"]["learning_rate"],
        batch_size=int(cfg["optimizer"]["ppo_n_envs"] * cfg["optimizer"]["ppo_n_steps"]),
        n_epochs=1,
        gamma=cfg["optimizer"]["ppo_gamma"],
        gae_lambda=cfg["optimizer"]["ppo_gae_lambda"],
        clip_range=cfg["optimizer"]["ppo_clip_range"],
        clip_range_vf=cfg["optimizer"]["ppo_clip_range_vf"],
        normalize_advantage=cfg["optimizer"]["ppo_normalize_advantage"],
        ent_coef=cfg["optimizer"]["ppo_ent_coef"],
        vf_coef=cfg["optimizer"]["ppo_vf_coef"],
        max_grad_norm=cfg["optimizer"]["ppo_max_grad_norm"],
        target_kl=cfg["optimizer"]["ppo_target_kl"],
    )
    algo.ep_info_buffer = []
    algo.ep_success_buffer = []
    algo._last_obs = vec_env.reset()
    algo._last_episode_starts = np.ones((vec_env.num_envs,), dtype=bool)
    algo._last_lstm_states = algo.policy._dummy_states(vec_env.num_envs)
    callback.init_callback(algo)
    assert algo.collect_rollouts(vec_env, callback, algo.rollout_buffer, n_rollout_steps=algo.n_steps)

    monkeypatch.setattr(np.random, "randint", lambda *args, **kwargs: 0)
    expected_hidden_calls = 0
    for rollout_data in algo.rollout_buffer.get_gpu_flat(algo.batch_size):
        n_seq_total = int(rollout_data.n_seq)
        seq_subbatch_size = algo._resolve_sequence_subbatch_size(rollout_data)
        expected_hidden_calls += int(np.ceil(n_seq_total / seq_subbatch_size))

    original_hidden = algo.policy._official_sequence_hidden_flat
    original_hidden_and_aux = algo.policy._official_sequence_hidden_and_aux_flat_packed
    hidden_calls = 0
    hidden_and_aux_calls = 0

    def _wrapped_hidden(obs, *, seq_lengths):
        nonlocal hidden_calls
        hidden_calls += 1
        return original_hidden(obs, seq_lengths=seq_lengths)

    def _wrapped_hidden_and_aux(obs, *, seq_lengths, actions=None, include_normalized_q_logits=False, flow_matching_xt=None, flow_matching_t=None):
        nonlocal hidden_and_aux_calls
        hidden_and_aux_calls += 1
        return original_hidden_and_aux(
            obs,
            seq_lengths=seq_lengths,
            actions=actions,
            include_normalized_q_logits=include_normalized_q_logits,
            flow_matching_xt=flow_matching_xt,
            flow_matching_t=flow_matching_t,
        )

    monkeypatch.setattr(algo.policy, "_official_sequence_hidden_flat", _wrapped_hidden)
    monkeypatch.setattr(algo.policy, "_official_sequence_hidden_and_aux_flat_packed", _wrapped_hidden_and_aux)
    algo._logger = SimpleNamespace(record=lambda *args, **kwargs: None)
    algo.train()
    assert hidden_calls == 0
    assert hidden_and_aux_calls == expected_hidden_calls
    vec_env.close()


def test_masked_recurrent_ppo_train_progress_logging_emits_lines(monkeypatch, capsys):
    device = _cuda_or_skip()
    cfg, env_cfg = _build_small_ppo_config()
    env_cfg = _strict_env_cfg_for_ppo(env_cfg, aux=False)
    prior = EnvironmentPrior(env_cfg)
    fake_model = _FakeRWKVModel(
        num_features=cfg["prior"]["num_features"],
        x_obs_dim=cfg["transformer"]["x_obs_dim"],
        action_dim=cfg["transformer"]["x_action_dim"],
    ).to(device)
    algo, callback, vec_env = build_recurrent_ppo(
        model=fake_model,
        env_prior=prior,
        device=str(device),
        num_features=cfg["prior"]["num_features"],
        n_envs=cfg["optimizer"]["ppo_n_envs"],
        n_steps=cfg["optimizer"]["ppo_n_steps"],
        learning_rate=cfg["optimizer"]["learning_rate"],
        batch_size=int(cfg["optimizer"]["ppo_n_envs"] * cfg["optimizer"]["ppo_n_steps"]),
        n_epochs=1,
        gamma=cfg["optimizer"]["ppo_gamma"],
        gae_lambda=cfg["optimizer"]["ppo_gae_lambda"],
        clip_range=cfg["optimizer"]["ppo_clip_range"],
        clip_range_vf=cfg["optimizer"]["ppo_clip_range_vf"],
        normalize_advantage=cfg["optimizer"]["ppo_normalize_advantage"],
        ent_coef=cfg["optimizer"]["ppo_ent_coef"],
        vf_coef=cfg["optimizer"]["ppo_vf_coef"],
        max_grad_norm=cfg["optimizer"]["ppo_max_grad_norm"],
        target_kl=cfg["optimizer"]["ppo_target_kl"],
    )
    algo.ep_info_buffer = []
    algo.ep_success_buffer = []
    algo._last_obs = vec_env.reset()
    algo._last_episode_starts = np.ones((vec_env.num_envs,), dtype=bool)
    algo._last_lstm_states = algo.policy._dummy_states(vec_env.num_envs)
    callback.init_callback(algo)
    assert algo.collect_rollouts(vec_env, callback, algo.rollout_buffer, n_rollout_steps=algo.n_steps)
    capsys.readouterr()
    monkeypatch.setattr(np.random, "randint", lambda *args, **kwargs: 0)
    algo.verbose = 1
    algo._logger = SimpleNamespace(record=lambda *args, **kwargs: None)
    setattr(algo, "_rwkv_progress_log_enabled", True)
    setattr(algo, "_rwkv_progress_log_every", 1)
    setattr(algo, "_rwkv_progress_log_min_interval_sec", 0.0)
    setattr(algo, "_rwkv_progress_log_file", None)

    algo.train()
    logs = capsys.readouterr().out
    assert "[ppo-train-start]" in logs
    assert "[ppo-train-progress]" in logs
    assert "[ppo-train-end]" in logs
    vec_env.close()


def test_slice_masked_rollout_sequence_batch_preserves_sequence_layout():
    device = torch.device("cpu")
    n_seq = 3
    seq_len = 4
    padded_batch = n_seq * seq_len
    hidden_state = torch.arange(n_seq, dtype=torch.float32, device=device).reshape(1, n_seq, 1)
    sample = MaskedRecurrentRolloutBufferSamples(
        observations=torch.arange(padded_batch * 2, dtype=torch.float32, device=device).reshape(padded_batch, 2),
        actions=torch.arange(padded_batch * 3, dtype=torch.float32, device=device).reshape(padded_batch, 3),
        old_values=torch.arange(padded_batch, dtype=torch.float32, device=device),
        old_log_prob=torch.arange(padded_batch, dtype=torch.float32, device=device),
        advantages=torch.arange(padded_batch, dtype=torch.float32, device=device),
        returns=torch.arange(padded_batch, dtype=torch.float32, device=device),
        objective_masks=torch.ones((padded_batch,), dtype=torch.float32, device=device),
        rollout_return_means=torch.zeros((padded_batch,), dtype=torch.float32, device=device),
        rollout_return_stds=torch.ones((padded_batch,), dtype=torch.float32, device=device),
        lstm_states=RNNStates(
            (hidden_state.clone(), hidden_state.clone() + 10),
            (hidden_state.clone() + 20, hidden_state.clone() + 30),
        ),
        episode_starts=torch.zeros((padded_batch,), dtype=torch.float32, device=device),
        mask=torch.ones((padded_batch,), dtype=torch.float32, device=device),
        action_masks=torch.ones((padded_batch, 3), dtype=torch.float32, device=device),
        next_states=torch.arange(padded_batch * 5, dtype=torch.float32, device=device).reshape(padded_batch, 5),
        next_state_masks=torch.ones((padded_batch, 5), dtype=torch.float32, device=device),
    )
    sub = _slice_masked_rollout_sequence_batch(sample, start_seq=1, end_seq=3)
    assert tuple(sub.observations.shape) == (8, 2)
    assert tuple(sub.actions.shape) == (8, 3)
    assert tuple(sub.next_states.shape) == (8, 5)
    assert tuple(sub.lstm_states.pi[0].shape) == (1, 2, 1)
    expected_obs = sample.observations.reshape(n_seq, seq_len, 2)[1:3].reshape(8, 2)
    assert torch.equal(sub.observations, expected_obs)


def test_masked_recurrent_ppo_collect_rollouts_bypasses_vecenv_step_wait(monkeypatch):
    device = _cuda_or_skip()
    cfg, env_cfg = _build_small_ppo_config()
    env_cfg = _strict_env_cfg_for_ppo(env_cfg, aux=False)
    prior = EnvironmentPrior(env_cfg)
    fake_model = _FakeRWKVModel(
        num_features=cfg["prior"]["num_features"],
        x_obs_dim=cfg["transformer"]["x_obs_dim"],
        action_dim=cfg["transformer"]["x_action_dim"],
    ).to(device)
    algo, callback, vec_env = build_recurrent_ppo(
        model=fake_model,
        env_prior=prior,
        device=str(device),
        num_features=cfg["prior"]["num_features"],
        n_envs=cfg["optimizer"]["ppo_n_envs"],
        n_steps=cfg["optimizer"]["ppo_n_steps"],
        learning_rate=cfg["optimizer"]["learning_rate"],
        batch_size=cfg["optimizer"]["ppo_batch_size"],
        n_epochs=1,
        gamma=cfg["optimizer"]["ppo_gamma"],
        gae_lambda=cfg["optimizer"]["ppo_gae_lambda"],
        clip_range=cfg["optimizer"]["ppo_clip_range"],
        clip_range_vf=cfg["optimizer"]["ppo_clip_range_vf"],
        normalize_advantage=cfg["optimizer"]["ppo_normalize_advantage"],
        ent_coef=cfg["optimizer"]["ppo_ent_coef"],
        vf_coef=cfg["optimizer"]["ppo_vf_coef"],
        max_grad_norm=cfg["optimizer"]["ppo_max_grad_norm"],
        target_kl=cfg["optimizer"]["ppo_target_kl"],
    )
    algo.ep_info_buffer = []
    algo.ep_success_buffer = []
    algo._last_obs = vec_env.reset()
    algo._last_episode_starts = np.ones((vec_env.num_envs,), dtype=bool)
    algo._last_lstm_states = algo.policy._dummy_states(vec_env.num_envs)
    callback.init_callback(algo)

    def _fail_step_wait():
        raise AssertionError("collect_rollouts should not call VecEnv.step_wait() anymore")

    monkeypatch.setattr(vec_env, "step_wait", _fail_step_wait)
    assert algo.collect_rollouts(vec_env, callback, algo.rollout_buffer, n_rollout_steps=algo.n_steps)
    vec_env.close()


def test_masked_recurrent_ppo_collect_rollouts_streams_into_buffer_and_uses_only_final_bootstrap_value(monkeypatch):
    device = _cuda_or_skip()
    cfg, env_cfg = _build_small_ppo_config()
    env_cfg = _strict_env_cfg_for_ppo(env_cfg, aux=False)
    prior = EnvironmentPrior(env_cfg)
    fake_model = _FakeRWKVModel(
        num_features=cfg["prior"]["num_features"],
        x_obs_dim=cfg["transformer"]["x_obs_dim"],
        action_dim=cfg["transformer"]["x_action_dim"],
    ).to(device)
    algo, callback, vec_env = build_recurrent_ppo(
        model=fake_model,
        env_prior=prior,
        device=str(device),
        num_features=cfg["prior"]["num_features"],
        n_envs=cfg["optimizer"]["ppo_n_envs"],
        n_steps=cfg["optimizer"]["ppo_n_steps"],
        learning_rate=cfg["optimizer"]["learning_rate"],
        batch_size=cfg["optimizer"]["ppo_batch_size"],
        n_epochs=1,
        gamma=cfg["optimizer"]["ppo_gamma"],
        gae_lambda=cfg["optimizer"]["ppo_gae_lambda"],
        clip_range=cfg["optimizer"]["ppo_clip_range"],
        clip_range_vf=cfg["optimizer"]["ppo_clip_range_vf"],
        normalize_advantage=cfg["optimizer"]["ppo_normalize_advantage"],
        ent_coef=cfg["optimizer"]["ppo_ent_coef"],
        vf_coef=cfg["optimizer"]["ppo_vf_coef"],
        max_grad_norm=cfg["optimizer"]["ppo_max_grad_norm"],
        target_kl=cfg["optimizer"]["ppo_target_kl"],
    )
    algo.ep_info_buffer = []
    algo.ep_success_buffer = []
    algo._last_obs = vec_env.reset()
    algo._last_episode_starts = np.ones((vec_env.num_envs,), dtype=bool)
    algo._last_lstm_states = algo.policy._dummy_states(vec_env.num_envs)
    callback.init_callback(algo)

    original_eval = algo.policy.evaluate_rollout_values
    call_shapes = []

    def _wrapped_eval(obs_steps, episode_starts_steps, **kwargs):
        call_shapes.append(tuple(obs_steps.shape))
        if int(obs_steps.shape[0]) > 1:
            raise AssertionError("collect_rollouts should not recompute full rollout values from obs_steps")
        return original_eval(obs_steps, episode_starts_steps, **kwargs)

    monkeypatch.setattr(algo.policy, "evaluate_rollout_values", _wrapped_eval)
    assert algo.collect_rollouts(vec_env, callback, algo.rollout_buffer, n_rollout_steps=algo.n_steps)
    assert call_shapes == [(1, cfg["optimizer"]["ppo_n_envs"], cfg["prior"]["num_features"])]
    assert prior.last_rollout_ppo_trace["streamed_to_sink"] is True
    assert prior.last_rollout_ppo_trace["obs"] is None
    assert prior.last_rollout_ppo_trace["values"] is None
    assert prior.last_rollout_reinforce is None
    vec_env.close()


def test_collect_rollouts_recomputes_full_rollout_values_for_batchnorm_batchstats(monkeypatch):
    device = _cuda_or_skip()
    cfg, env_cfg = _build_small_ppo_config()
    env_cfg = _strict_env_cfg_for_ppo(env_cfg, aux=False)
    prior = EnvironmentPrior(env_cfg)
    fake_model = _FakeRWKVModel(
        num_features=cfg["prior"]["num_features"],
        x_obs_dim=cfg["transformer"]["x_obs_dim"],
        action_dim=cfg["transformer"]["x_action_dim"],
    ).to(device)
    algo, callback, vec_env = build_recurrent_ppo(
        model=fake_model,
        env_prior=prior,
        device=str(device),
        num_features=cfg["prior"]["num_features"],
        n_envs=cfg["optimizer"]["ppo_n_envs"],
        n_steps=cfg["optimizer"]["ppo_n_steps"],
        learning_rate=cfg["optimizer"]["learning_rate"],
        batch_size=cfg["optimizer"]["ppo_batch_size"],
        n_epochs=1,
        gamma=cfg["optimizer"]["ppo_gamma"],
        gae_lambda=cfg["optimizer"]["ppo_gae_lambda"],
        clip_range=cfg["optimizer"]["ppo_clip_range"],
        clip_range_vf=cfg["optimizer"]["ppo_clip_range_vf"],
        normalize_advantage=cfg["optimizer"]["ppo_normalize_advantage"],
        ent_coef=cfg["optimizer"]["ppo_ent_coef"],
        vf_coef=cfg["optimizer"]["ppo_vf_coef"],
        max_grad_norm=cfg["optimizer"]["ppo_max_grad_norm"],
        target_kl=cfg["optimizer"]["ppo_target_kl"],
        value_head_impl="vendor_official",
        value_path_adapter_impl="batchnorm_batchstats",
        actor_gae_space="normalized",
        actor_baseline_mode="learned",
        actor_objective_mode="tokenwise",
        verbose=0,
    )
    try:
        algo.ep_info_buffer = []
        algo.ep_success_buffer = []
        algo._last_obs = vec_env.reset()
        algo._last_episode_starts = np.ones((vec_env.num_envs,), dtype=bool)
        algo._last_lstm_states = algo.policy._dummy_states(vec_env.num_envs)
        callback.init_callback(algo)

        original_eval = algo.policy.evaluate_rollout_values
        call_shapes = []

        def _wrapped_eval(obs_steps, episode_starts_steps, **kwargs):
            call_shapes.append(tuple(obs_steps.shape))
            return original_eval(obs_steps, episode_starts_steps, **kwargs)

        monkeypatch.setattr(algo.policy, "evaluate_rollout_values", _wrapped_eval)
        assert algo.collect_rollouts(vec_env, callback, algo.rollout_buffer, n_rollout_steps=algo.n_steps)
        assert (algo.n_steps, vec_env.num_envs, cfg["prior"]["num_features"]) in call_shapes
    finally:
        vec_env.close()


def test_masked_recurrent_ppo_collect_rollouts_records_suffix_objective_masks_and_fixed_rollout_value_stats(monkeypatch):
    device = _cuda_or_skip()
    cfg, env_cfg = _build_small_ppo_config()
    env_cfg = _strict_env_cfg_for_ppo(env_cfg, aux=False)
    prior = EnvironmentPrior(env_cfg)
    fake_model = _FakeRWKVModel(
        num_features=cfg["prior"]["num_features"],
        x_obs_dim=cfg["transformer"]["x_obs_dim"],
        action_dim=cfg["transformer"]["x_action_dim"],
    ).to(device)
    algo, callback, vec_env = build_recurrent_ppo(
        model=fake_model,
        env_prior=prior,
        device=str(device),
        num_features=cfg["prior"]["num_features"],
        n_envs=cfg["optimizer"]["ppo_n_envs"],
        n_steps=cfg["optimizer"]["ppo_n_steps"],
        learning_rate=cfg["optimizer"]["learning_rate"],
        batch_size=cfg["optimizer"]["ppo_batch_size"],
        n_epochs=1,
        gamma=cfg["optimizer"]["ppo_gamma"],
        gae_lambda=cfg["optimizer"]["ppo_gae_lambda"],
        clip_range=cfg["optimizer"]["ppo_clip_range"],
        clip_range_vf=cfg["optimizer"]["ppo_clip_range_vf"],
        normalize_advantage=cfg["optimizer"]["ppo_normalize_advantage"],
        ent_coef=cfg["optimizer"]["ppo_ent_coef"],
        vf_coef=cfg["optimizer"]["ppo_vf_coef"],
        max_grad_norm=cfg["optimizer"]["ppo_max_grad_norm"],
        target_kl=cfg["optimizer"]["ppo_target_kl"],
    )
    algo.ep_info_buffer = []
    algo.ep_success_buffer = []
    algo._last_obs = vec_env.reset()
    algo._last_episode_starts = np.ones((vec_env.num_envs,), dtype=bool)
    algo._last_lstm_states = algo.policy._dummy_states(vec_env.num_envs)
    callback.init_callback(algo)
    monkeypatch.setattr(prior, "_sample_single_eval_pos", lambda n_samples, single_eval_pos: 2)
    assert algo.collect_rollouts(vec_env, callback, algo.rollout_buffer, n_rollout_steps=algo.n_steps)
    assert np.allclose(algo.rollout_buffer.objective_masks[:2], 0.0)
    assert np.allclose(algo.rollout_buffer.objective_masks[2:], 1.0)
    assert np.all(algo.rollout_buffer.rollout_return_stds > 0.0)
    assert np.allclose(
        algo.rollout_buffer.rollout_return_means,
        algo.rollout_buffer.rollout_return_means[0:1],
        atol=1e-6,
        rtol=1e-6,
    )
    assert np.allclose(
        algo.rollout_buffer.rollout_return_stds,
        algo.rollout_buffer.rollout_return_stds[0:1],
        atol=1e-6,
        rtol=1e-6,
    )
    vec_env.close()


def test_environment_prior_ppo_env_step_info_includes_step_idx():
    device = _cuda_or_skip()
    cfg, env_cfg = _build_small_ppo_config()
    env_cfg = _strict_env_cfg_for_ppo(env_cfg, aux=False)
    prior = EnvironmentPrior(env_cfg)
    fake_model = _FakeRWKVModel(
        num_features=cfg["prior"]["num_features"],
        x_obs_dim=cfg["transformer"]["x_obs_dim"],
        action_dim=cfg["transformer"]["x_action_dim"],
    ).to(device)
    _, _, vec_env = build_recurrent_ppo(
        model=fake_model,
        env_prior=prior,
        device=str(device),
        num_features=cfg["prior"]["num_features"],
        n_envs=cfg["optimizer"]["ppo_n_envs"],
        n_steps=cfg["optimizer"]["ppo_n_steps"],
        learning_rate=cfg["optimizer"]["learning_rate"],
        batch_size=cfg["optimizer"]["ppo_batch_size"],
        n_epochs=1,
        gamma=cfg["optimizer"]["ppo_gamma"],
        gae_lambda=cfg["optimizer"]["ppo_gae_lambda"],
        clip_range=cfg["optimizer"]["ppo_clip_range"],
        clip_range_vf=cfg["optimizer"]["ppo_clip_range_vf"],
        normalize_advantage=cfg["optimizer"]["ppo_normalize_advantage"],
        ent_coef=cfg["optimizer"]["ppo_ent_coef"],
        vf_coef=cfg["optimizer"]["ppo_vf_coef"],
        max_grad_norm=cfg["optimizer"]["ppo_max_grad_norm"],
        target_kl=cfg["optimizer"]["ppo_target_kl"],
    )
    env = EnvironmentPriorPPOGymEnv(
        env_prior=prior,
        device=str(device),
        num_features=cfg["prior"]["num_features"],
        n_steps=cfg["optimizer"]["ppo_n_steps"],
        action_dim=vec_env.action_dim,
        obs_slot_dim=vec_env.obs_slot_dim,
        action_slot_dim=vec_env.action_slot_dim,
        next_state_target_dim=vec_env.next_state_target_dim,
    )
    env.reset(seed=0)
    _, _, _, _, info = env.step(np.zeros((vec_env.action_dim,), dtype=np.float32))
    assert info["step_idx"] == 1
    vec_env.close()


def test_environment_prior_ppo_batch_vec_env_step_info_includes_step_idx():
    device = _cuda_or_skip()
    cfg, env_cfg = _build_small_ppo_config()
    env_cfg = _strict_env_cfg_for_ppo(env_cfg, aux=False)
    prior = EnvironmentPrior(env_cfg)
    fake_model = _FakeRWKVModel(
        num_features=cfg["prior"]["num_features"],
        x_obs_dim=cfg["transformer"]["x_obs_dim"],
        action_dim=cfg["transformer"]["x_action_dim"],
    ).to(device)
    _, _, vec_env = build_recurrent_ppo(
        model=fake_model,
        env_prior=prior,
        device=str(device),
        num_features=cfg["prior"]["num_features"],
        n_envs=cfg["optimizer"]["ppo_n_envs"],
        n_steps=cfg["optimizer"]["ppo_n_steps"],
        learning_rate=cfg["optimizer"]["learning_rate"],
        batch_size=cfg["optimizer"]["ppo_batch_size"],
        n_epochs=1,
        gamma=cfg["optimizer"]["ppo_gamma"],
        gae_lambda=cfg["optimizer"]["ppo_gae_lambda"],
        clip_range=cfg["optimizer"]["ppo_clip_range"],
        clip_range_vf=cfg["optimizer"]["ppo_clip_range_vf"],
        normalize_advantage=cfg["optimizer"]["ppo_normalize_advantage"],
        ent_coef=cfg["optimizer"]["ppo_ent_coef"],
        vf_coef=cfg["optimizer"]["ppo_vf_coef"],
        max_grad_norm=cfg["optimizer"]["ppo_max_grad_norm"],
        target_kl=cfg["optimizer"]["ppo_target_kl"],
    )
    vec_env.reset()
    vec_env.step_async(np.zeros((vec_env.num_envs, vec_env.action_dim), dtype=np.float32))
    _, _, _, infos = vec_env.step_wait()
    assert infos[0]["step_idx"] == 1
    vec_env.close()


def test_environment_prior_ppo_env_step_returns_terminated_on_terminal(monkeypatch):
    device = _cuda_or_skip()
    cfg, env_cfg = _build_small_ppo_config()
    env_cfg = _strict_env_cfg_for_ppo(env_cfg, aux=False)
    prior = EnvironmentPrior(env_cfg)
    fake_model = _FakeRWKVModel(
        num_features=cfg["prior"]["num_features"],
        x_obs_dim=cfg["transformer"]["x_obs_dim"],
        action_dim=cfg["transformer"]["x_action_dim"],
    ).to(device)
    _, _, vec_env = build_recurrent_ppo(
        model=fake_model,
        env_prior=prior,
        device=str(device),
        num_features=cfg["prior"]["num_features"],
        n_envs=cfg["optimizer"]["ppo_n_envs"],
        n_steps=cfg["optimizer"]["ppo_n_steps"],
        learning_rate=cfg["optimizer"]["learning_rate"],
        batch_size=cfg["optimizer"]["ppo_batch_size"],
        n_epochs=1,
        gamma=cfg["optimizer"]["ppo_gamma"],
        gae_lambda=cfg["optimizer"]["ppo_gae_lambda"],
        clip_range=cfg["optimizer"]["ppo_clip_range"],
        clip_range_vf=cfg["optimizer"]["ppo_clip_range_vf"],
        normalize_advantage=cfg["optimizer"]["ppo_normalize_advantage"],
        ent_coef=cfg["optimizer"]["ppo_ent_coef"],
        vf_coef=cfg["optimizer"]["ppo_vf_coef"],
        max_grad_norm=cfg["optimizer"]["ppo_max_grad_norm"],
        target_kl=cfg["optimizer"]["ppo_target_kl"],
    )
    env = EnvironmentPriorPPOGymEnv(
        env_prior=prior,
        device=str(device),
        num_features=cfg["prior"]["num_features"],
        n_steps=cfg["optimizer"]["ppo_n_steps"],
        action_dim=vec_env.action_dim,
        obs_slot_dim=vec_env.obs_slot_dim,
        action_slot_dim=vec_env.action_slot_dim,
        next_state_target_dim=vec_env.next_state_target_dim,
    )
    env.reset(seed=0)
    env._env["terminal_reset_enabled"] = torch.tensor(True, device=device, dtype=torch.bool)

    def _force_terminal(**kwargs):
        assert kwargs["history_warmup_count"] is env._env.get("terminal_reset_count_target", None)
        assert kwargs["history_relaxed_mask"] is env._terminal_history_relaxed
        return kwargs["state_next"], kwargs["reward_next"], torch.ones_like(kwargs["reward_next"])

    monkeypatch.setattr(prior, "_apply_terminal_reset_step", _force_terminal)
    _, _, terminated, truncated, info = env.step(np.zeros((vec_env.action_dim,), dtype=np.float32))
    assert terminated is True
    assert truncated is False
    assert info["terminal_flag"] == pytest.approx(1.0)
    assert info["TimeLimit.truncated"] is False
    vec_env.close()


def test_environment_prior_ppo_env_step_returns_truncated_at_n_steps():
    device = _cuda_or_skip()
    cfg, env_cfg = _build_small_ppo_config()
    env_cfg = _strict_env_cfg_for_ppo(env_cfg, aux=False)
    prior = EnvironmentPrior(env_cfg)
    fake_model = _FakeRWKVModel(
        num_features=cfg["prior"]["num_features"],
        x_obs_dim=cfg["transformer"]["x_obs_dim"],
        action_dim=cfg["transformer"]["x_action_dim"],
    ).to(device)
    _, _, vec_env = build_recurrent_ppo(
        model=fake_model,
        env_prior=prior,
        device=str(device),
        num_features=cfg["prior"]["num_features"],
        n_envs=cfg["optimizer"]["ppo_n_envs"],
        n_steps=cfg["optimizer"]["ppo_n_steps"],
        learning_rate=cfg["optimizer"]["learning_rate"],
        batch_size=cfg["optimizer"]["ppo_batch_size"],
        n_epochs=1,
        gamma=cfg["optimizer"]["ppo_gamma"],
        gae_lambda=cfg["optimizer"]["ppo_gae_lambda"],
        clip_range=cfg["optimizer"]["ppo_clip_range"],
        clip_range_vf=cfg["optimizer"]["ppo_clip_range_vf"],
        normalize_advantage=cfg["optimizer"]["ppo_normalize_advantage"],
        ent_coef=cfg["optimizer"]["ppo_ent_coef"],
        vf_coef=cfg["optimizer"]["ppo_vf_coef"],
        max_grad_norm=cfg["optimizer"]["ppo_max_grad_norm"],
        target_kl=cfg["optimizer"]["ppo_target_kl"],
    )
    env = EnvironmentPriorPPOGymEnv(
        env_prior=prior,
        device=str(device),
        num_features=cfg["prior"]["num_features"],
        n_steps=cfg["optimizer"]["ppo_n_steps"],
        action_dim=vec_env.action_dim,
        obs_slot_dim=vec_env.obs_slot_dim,
        action_slot_dim=vec_env.action_slot_dim,
        next_state_target_dim=vec_env.next_state_target_dim,
    )
    env.reset(seed=0)
    env._env["terminal_reset_enabled"] = torch.tensor(False, device=device, dtype=torch.bool)
    env._step_idx = env.n_steps - 1
    _, _, terminated, truncated, info = env.step(np.zeros((vec_env.action_dim,), dtype=np.float32))
    assert terminated is False
    assert truncated is True
    assert info["TimeLimit.truncated"] is True
    vec_env.close()


def test_environment_prior_ppo_batch_vec_env_step_wait_returns_done_on_terminal(monkeypatch):
    device = _cuda_or_skip()
    cfg, env_cfg = _build_small_ppo_config()
    env_cfg = _strict_env_cfg_for_ppo(env_cfg, aux=False)
    prior = EnvironmentPrior(env_cfg)
    fake_model = _FakeRWKVModel(
        num_features=cfg["prior"]["num_features"],
        x_obs_dim=cfg["transformer"]["x_obs_dim"],
        action_dim=cfg["transformer"]["x_action_dim"],
    ).to(device)
    _, _, vec_env = build_recurrent_ppo(
        model=fake_model,
        env_prior=prior,
        device=str(device),
        num_features=cfg["prior"]["num_features"],
        n_envs=cfg["optimizer"]["ppo_n_envs"],
        n_steps=cfg["optimizer"]["ppo_n_steps"],
        learning_rate=cfg["optimizer"]["learning_rate"],
        batch_size=cfg["optimizer"]["ppo_batch_size"],
        n_epochs=1,
        gamma=cfg["optimizer"]["ppo_gamma"],
        gae_lambda=cfg["optimizer"]["ppo_gae_lambda"],
        clip_range=cfg["optimizer"]["ppo_clip_range"],
        clip_range_vf=cfg["optimizer"]["ppo_clip_range_vf"],
        normalize_advantage=cfg["optimizer"]["ppo_normalize_advantage"],
        ent_coef=cfg["optimizer"]["ppo_ent_coef"],
        vf_coef=cfg["optimizer"]["ppo_vf_coef"],
        max_grad_norm=cfg["optimizer"]["ppo_max_grad_norm"],
        target_kl=cfg["optimizer"]["ppo_target_kl"],
    )
    vec_env.reset()
    vec_env._env["terminal_reset_enabled"] = torch.ones((vec_env.num_envs,), device=device, dtype=torch.bool)

    def _force_terminal(**kwargs):
        assert kwargs["history_warmup_count"] is vec_env._env.get("terminal_reset_count_target", None)
        assert kwargs["history_relaxed_mask"] is vec_env._terminal_history_relaxed
        return kwargs["state_next"], kwargs["reward_next"], torch.ones_like(kwargs["reward_next"])

    monkeypatch.setattr(prior, "_apply_terminal_reset_step", _force_terminal)
    vec_env.step_async(np.zeros((vec_env.num_envs, vec_env.action_dim), dtype=np.float32))
    _, _, dones, infos = vec_env.step_wait()
    assert bool(np.all(dones))
    assert infos[0]["terminal_flag"] == pytest.approx(1.0)
    assert infos[0]["TimeLimit.truncated"] is False
    vec_env.close()


def test_environment_prior_ppo_batch_vec_env_step_wait_returns_truncated_at_n_steps():
    device = _cuda_or_skip()
    cfg, env_cfg = _build_small_ppo_config()
    env_cfg = _strict_env_cfg_for_ppo(env_cfg, aux=False)
    prior = EnvironmentPrior(env_cfg)
    fake_model = _FakeRWKVModel(
        num_features=cfg["prior"]["num_features"],
        x_obs_dim=cfg["transformer"]["x_obs_dim"],
        action_dim=cfg["transformer"]["x_action_dim"],
    ).to(device)
    _, _, vec_env = build_recurrent_ppo(
        model=fake_model,
        env_prior=prior,
        device=str(device),
        num_features=cfg["prior"]["num_features"],
        n_envs=cfg["optimizer"]["ppo_n_envs"],
        n_steps=cfg["optimizer"]["ppo_n_steps"],
        learning_rate=cfg["optimizer"]["learning_rate"],
        batch_size=cfg["optimizer"]["ppo_batch_size"],
        n_epochs=1,
        gamma=cfg["optimizer"]["ppo_gamma"],
        gae_lambda=cfg["optimizer"]["ppo_gae_lambda"],
        clip_range=cfg["optimizer"]["ppo_clip_range"],
        clip_range_vf=cfg["optimizer"]["ppo_clip_range_vf"],
        normalize_advantage=cfg["optimizer"]["ppo_normalize_advantage"],
        ent_coef=cfg["optimizer"]["ppo_ent_coef"],
        vf_coef=cfg["optimizer"]["ppo_vf_coef"],
        max_grad_norm=cfg["optimizer"]["ppo_max_grad_norm"],
        target_kl=cfg["optimizer"]["ppo_target_kl"],
    )
    vec_env.reset()
    vec_env._env["terminal_reset_enabled"] = torch.zeros((vec_env.num_envs,), device=device, dtype=torch.bool)
    vec_env._step_idx = vec_env.n_steps - 1
    vec_env.step_async(np.zeros((vec_env.num_envs, vec_env.action_dim), dtype=np.float32))
    _, _, dones, infos = vec_env.step_wait()
    assert bool(np.all(dones))
    assert infos[0]["TimeLimit.truncated"] is True
    assert infos[0]["terminal_flag"] == pytest.approx(0.0)
    vec_env.close()


def test_build_recurrent_ppo_enables_sep_state_reset_on_main_training_path():
    device = _cuda_or_skip()
    cfg, env_cfg = _build_small_ppo_config()
    env_cfg = _strict_env_cfg_for_ppo(env_cfg, aux=False)
    prior = EnvironmentPrior(env_cfg)
    fake_model = _FakeRWKVModel(
        num_features=cfg["prior"]["num_features"],
        x_obs_dim=cfg["transformer"]["x_obs_dim"],
        action_dim=cfg["transformer"]["x_action_dim"],
    ).to(device)
    algo, _, vec_env = build_recurrent_ppo(
        model=fake_model,
        env_prior=prior,
        device=str(device),
        num_features=cfg["prior"]["num_features"],
        n_envs=1,
        n_steps=cfg["optimizer"]["ppo_n_steps"],
        learning_rate=cfg["optimizer"]["learning_rate"],
        batch_size=cfg["optimizer"]["ppo_n_steps"],
        n_epochs=1,
        gamma=cfg["optimizer"]["ppo_gamma"],
        gae_lambda=cfg["optimizer"]["ppo_gae_lambda"],
        clip_range=cfg["optimizer"]["ppo_clip_range"],
        clip_range_vf=cfg["optimizer"]["ppo_clip_range_vf"],
        normalize_advantage=cfg["optimizer"]["ppo_normalize_advantage"],
        ent_coef=cfg["optimizer"]["ppo_ent_coef"],
        vf_coef=cfg["optimizer"]["ppo_vf_coef"],
        max_grad_norm=cfg["optimizer"]["ppo_max_grad_norm"],
        target_kl=cfg["optimizer"]["ppo_target_kl"],
        reset_env_state_at_sep=True,
    )
    assert getattr(prior, "_rwkv_reset_env_state_at_sep_keep_actor_history", False) is True
    assert getattr(algo, "_rwkv_reset_env_state_at_sep", False) is True
    vec_env.close()


def test_build_recurrent_ppo_enables_strict_fixed_env_mode_and_seed_specs():
    device = _cuda_or_skip()
    cfg, env_cfg = _build_small_ppo_config()
    env_cfg = _strict_env_cfg_for_ppo(env_cfg, aux=False)
    prior = EnvironmentPrior(env_cfg)
    fake_model = _FakeRWKVModel(
        num_features=cfg["prior"]["num_features"],
        x_obs_dim=cfg["transformer"]["x_obs_dim"],
        action_dim=cfg["transformer"]["x_action_dim"],
    ).to(device)
    algo, _, vec_env = build_recurrent_ppo(
        model=fake_model,
        env_prior=prior,
        device=str(device),
        num_features=cfg["prior"]["num_features"],
        n_envs=1,
        n_steps=cfg["optimizer"]["ppo_n_steps"],
        learning_rate=cfg["optimizer"]["learning_rate"],
        batch_size=cfg["optimizer"]["ppo_n_steps"],
        n_epochs=1,
        gamma=cfg["optimizer"]["ppo_gamma"],
        gae_lambda=cfg["optimizer"]["ppo_gae_lambda"],
        clip_range=cfg["optimizer"]["ppo_clip_range"],
        clip_range_vf=cfg["optimizer"]["ppo_clip_range_vf"],
        normalize_advantage=cfg["optimizer"]["ppo_normalize_advantage"],
        ent_coef=cfg["optimizer"]["ppo_ent_coef"],
        vf_coef=cfg["optimizer"]["ppo_vf_coef"],
        max_grad_norm=cfg["optimizer"]["ppo_max_grad_norm"],
        target_kl=cfg["optimizer"]["ppo_target_kl"],
        strict_fixed_env_mode=True,
        env_rng_seeds=[101],
        rollout_rng_seeds=[202],
    )
    assert getattr(algo, "_rwkv_strict_fixed_env_mode", False) is True
    assert getattr(algo, "_rwkv_env_rng_seeds", None) == [101]
    assert getattr(algo, "_rwkv_rollout_rng_seeds", None) == [202]
    assert getattr(algo, "_rwkv_last_collect_rollout_env_rng_seeds", "sentinel") is None
    assert getattr(algo, "_rwkv_last_collect_rollout_rollout_rng_seeds", "sentinel") is None
    vec_env.close()


def test_build_recurrent_ppo_enables_deterministic_actor_sampling_flag():
    device = _cuda_or_skip()
    cfg, env_cfg = _build_small_ppo_config()
    env_cfg = _strict_env_cfg_for_ppo(env_cfg, aux=False)
    prior = EnvironmentPrior(env_cfg)
    fake_model = _FakeRWKVModel(
        num_features=cfg["prior"]["num_features"],
        x_obs_dim=cfg["transformer"]["x_obs_dim"],
        action_dim=cfg["transformer"]["x_action_dim"],
    ).to(device)
    algo, _, vec_env = build_recurrent_ppo(
        model=fake_model,
        env_prior=prior,
        device=str(device),
        num_features=cfg["prior"]["num_features"],
        n_envs=1,
        n_steps=cfg["optimizer"]["ppo_n_steps"],
        learning_rate=cfg["optimizer"]["learning_rate"],
        batch_size=cfg["optimizer"]["ppo_n_steps"],
        n_epochs=1,
        gamma=cfg["optimizer"]["ppo_gamma"],
        gae_lambda=cfg["optimizer"]["ppo_gae_lambda"],
        clip_range=cfg["optimizer"]["ppo_clip_range"],
        clip_range_vf=cfg["optimizer"]["ppo_clip_range_vf"],
        normalize_advantage=cfg["optimizer"]["ppo_normalize_advantage"],
        ent_coef=cfg["optimizer"]["ppo_ent_coef"],
        vf_coef=cfg["optimizer"]["ppo_vf_coef"],
        max_grad_norm=cfg["optimizer"]["ppo_max_grad_norm"],
        target_kl=cfg["optimizer"]["ppo_target_kl"],
        deterministic_actor_sampling=True,
    )
    assert getattr(algo, "_rwkv_deterministic_actor_sampling", False) is True
    vec_env.close()


def test_build_recurrent_ppo_enables_deterministic_batch_plan_flag():
    device = _cuda_or_skip()
    cfg, env_cfg = _build_small_ppo_config()
    env_cfg = _strict_env_cfg_for_ppo(env_cfg, aux=False)
    prior = EnvironmentPrior(env_cfg)
    fake_model = _FakeRWKVModel(
        num_features=cfg["prior"]["num_features"],
        x_obs_dim=cfg["transformer"]["x_obs_dim"],
        action_dim=cfg["transformer"]["x_action_dim"],
    ).to(device)
    algo, _, vec_env = build_recurrent_ppo(
        model=fake_model,
        env_prior=prior,
        device=str(device),
        num_features=cfg["prior"]["num_features"],
        n_envs=1,
        n_steps=cfg["optimizer"]["ppo_n_steps"],
        learning_rate=cfg["optimizer"]["learning_rate"],
        batch_size=cfg["optimizer"]["ppo_n_steps"],
        n_epochs=1,
        gamma=cfg["optimizer"]["ppo_gamma"],
        gae_lambda=cfg["optimizer"]["ppo_gae_lambda"],
        clip_range=cfg["optimizer"]["ppo_clip_range"],
        clip_range_vf=cfg["optimizer"]["ppo_clip_range_vf"],
        normalize_advantage=cfg["optimizer"]["ppo_normalize_advantage"],
        ent_coef=cfg["optimizer"]["ppo_ent_coef"],
        vf_coef=cfg["optimizer"]["ppo_vf_coef"],
        max_grad_norm=cfg["optimizer"]["ppo_max_grad_norm"],
        target_kl=cfg["optimizer"]["ppo_target_kl"],
        deterministic_batch_plan=True,
    )
    assert getattr(algo, "_rwkv_deterministic_batch_plan", False) is True
    assert getattr(algo.rollout_buffer, "_deterministic_batch_plan", False) is True
    vec_env.close()


def test_masked_recurrent_rollout_buffer_uses_deterministic_batch_plan_when_enabled(monkeypatch):
    device = _cuda_or_skip()
    cfg, env_cfg = _build_small_ppo_config()
    env_cfg = _strict_env_cfg_for_ppo(env_cfg, aux=False)
    prior = EnvironmentPrior(env_cfg)
    fake_model = _FakeRWKVModel(
        num_features=cfg["prior"]["num_features"],
        x_obs_dim=cfg["transformer"]["x_obs_dim"],
        action_dim=cfg["transformer"]["x_action_dim"],
    ).to(device)
    algo, _, vec_env = build_recurrent_ppo(
        model=fake_model,
        env_prior=prior,
        device=str(device),
        num_features=cfg["prior"]["num_features"],
        n_envs=1,
        n_steps=cfg["optimizer"]["ppo_n_steps"],
        learning_rate=cfg["optimizer"]["learning_rate"],
        batch_size=cfg["optimizer"]["ppo_n_steps"],
        n_epochs=1,
        gamma=cfg["optimizer"]["ppo_gamma"],
        gae_lambda=cfg["optimizer"]["ppo_gae_lambda"],
        clip_range=cfg["optimizer"]["ppo_clip_range"],
        clip_range_vf=cfg["optimizer"]["ppo_clip_range_vf"],
        normalize_advantage=cfg["optimizer"]["ppo_normalize_advantage"],
        ent_coef=cfg["optimizer"]["ppo_ent_coef"],
        vf_coef=cfg["optimizer"]["ppo_vf_coef"],
        max_grad_norm=cfg["optimizer"]["ppo_max_grad_norm"],
        target_kl=cfg["optimizer"]["ppo_target_kl"],
        deterministic_batch_plan=False,
    )
    rollout_buffer = algo.rollout_buffer
    rollout_buffer.full = True
    rollout_buffer._ensure_generator_ready()
    monkeypatch.setattr(np.random, "randint", lambda high: 3)

    rollout_buffer.set_deterministic_batch_plan(False)
    batch_inds_random, _ = next(rollout_buffer._iter_batch_plan(batch_size=2))
    assert batch_inds_random.tolist()[:2] == [3, 0]

    rollout_buffer.set_deterministic_batch_plan(True)
    batch_inds_det, _ = next(rollout_buffer._iter_batch_plan(batch_size=2))
    assert batch_inds_det.tolist()[:2] == [0, 1]
    vec_env.close()


def test_masked_recurrent_ppo_collect_rollouts_passes_formal_fixed_env_seed_channels(monkeypatch):
    device = _cuda_or_skip()
    cfg, env_cfg = _build_small_ppo_config()
    env_cfg = _strict_env_cfg_for_ppo(env_cfg, aux=False)
    prior = EnvironmentPrior(env_cfg)
    fake_model = _FakeRWKVModel(
        num_features=cfg["prior"]["num_features"],
        x_obs_dim=cfg["transformer"]["x_obs_dim"],
        action_dim=cfg["transformer"]["x_action_dim"],
    ).to(device)
    algo, callback, vec_env = build_recurrent_ppo(
        model=fake_model,
        env_prior=prior,
        device=str(device),
        num_features=cfg["prior"]["num_features"],
        n_envs=1,
        n_steps=cfg["optimizer"]["ppo_n_steps"],
        learning_rate=cfg["optimizer"]["learning_rate"],
        batch_size=cfg["optimizer"]["ppo_n_steps"],
        n_epochs=1,
        gamma=cfg["optimizer"]["ppo_gamma"],
        gae_lambda=cfg["optimizer"]["ppo_gae_lambda"],
        clip_range=cfg["optimizer"]["ppo_clip_range"],
        clip_range_vf=cfg["optimizer"]["ppo_clip_range_vf"],
        normalize_advantage=cfg["optimizer"]["ppo_normalize_advantage"],
        ent_coef=cfg["optimizer"]["ppo_ent_coef"],
        vf_coef=cfg["optimizer"]["ppo_vf_coef"],
        max_grad_norm=cfg["optimizer"]["ppo_max_grad_norm"],
        target_kl=cfg["optimizer"]["ppo_target_kl"],
        strict_fixed_env_mode=True,
        env_rng_seeds=[101],
        rollout_rng_seeds=[202],
    )
    algo.ep_info_buffer = []
    algo.ep_success_buffer = []
    algo._last_obs = vec_env.reset()
    algo._last_episode_starts = np.ones((vec_env.num_envs,), dtype=bool)
    algo._last_lstm_states = algo.policy._dummy_states(vec_env.num_envs)
    callback.init_callback(algo)

    recorded = {}
    original_rollout = prior._rollout_family_group_vectorized_with_policy

    def _wrapped_rollout(*args, **kwargs):
        recorded["env_rng_seeds"] = list(kwargs.get("env_rng_seeds") or [])
        recorded["rollout_rng_seeds"] = list(kwargs.get("rollout_rng_seeds") or [])
        return original_rollout(*args, **kwargs)

    monkeypatch.setattr(prior, "_rollout_family_group_vectorized_with_policy", _wrapped_rollout)
    assert algo.collect_rollouts(vec_env, callback, algo.rollout_buffer, n_rollout_steps=algo.n_steps)
    assert recorded["env_rng_seeds"] == [101]
    assert recorded["rollout_rng_seeds"] == [202]
    assert getattr(algo, "_rwkv_last_collect_rollout_env_rng_seeds", None) == (101,)
    assert getattr(algo, "_rwkv_last_collect_rollout_rollout_rng_seeds", None) == (202,)
    vec_env.close()


def test_masked_recurrent_ppo_collect_rollouts_overrides_actor_sampling_when_deterministic(monkeypatch):
    device = _cuda_or_skip()
    cfg, env_cfg = _build_small_ppo_config()
    env_cfg = _strict_env_cfg_for_ppo(env_cfg, aux=False)
    prior = EnvironmentPrior(env_cfg)
    fake_model = _FakeRWKVModel(
        num_features=cfg["prior"]["num_features"],
        x_obs_dim=cfg["transformer"]["x_obs_dim"],
        action_dim=cfg["transformer"]["x_action_dim"],
    ).to(device)
    algo, callback, vec_env = build_recurrent_ppo(
        model=fake_model,
        env_prior=prior,
        device=str(device),
        num_features=cfg["prior"]["num_features"],
        n_envs=1,
        n_steps=cfg["optimizer"]["ppo_n_steps"],
        learning_rate=cfg["optimizer"]["learning_rate"],
        batch_size=cfg["optimizer"]["ppo_n_steps"],
        n_epochs=1,
        gamma=cfg["optimizer"]["ppo_gamma"],
        gae_lambda=cfg["optimizer"]["ppo_gae_lambda"],
        clip_range=cfg["optimizer"]["ppo_clip_range"],
        clip_range_vf=cfg["optimizer"]["ppo_clip_range_vf"],
        normalize_advantage=cfg["optimizer"]["ppo_normalize_advantage"],
        ent_coef=cfg["optimizer"]["ppo_ent_coef"],
        vf_coef=cfg["optimizer"]["ppo_vf_coef"],
        max_grad_norm=cfg["optimizer"]["ppo_max_grad_norm"],
        target_kl=cfg["optimizer"]["ppo_target_kl"],
        deterministic_actor_sampling=True,
    )
    algo.ep_info_buffer = []
    algo.ep_success_buffer = []
    algo._last_obs = vec_env.reset()
    algo._last_episode_starts = np.ones((vec_env.num_envs,), dtype=bool)
    algo._last_lstm_states = algo.policy._dummy_states(vec_env.num_envs)
    callback.init_callback(algo)

    recorded = {}
    original_rollout = prior._rollout_family_group_vectorized_with_policy

    def _wrapped_rollout(*args, **kwargs):
        policy_step_fn = args[1]
        sample_fn = getattr(policy_step_fn, "_policy_actor_sample_fn", None)
        actor_outputs = {
            "action_mean": torch.tensor([[1.0, -2.0]], device=device, dtype=torch.float32),
            "action_std": torch.tensor([[5.0, 7.0]], device=device, dtype=torch.float32),
        }
        noise = torch.tensor([[9.0, 11.0]], device=device, dtype=torch.float32)
        recorded["sample"] = sample_fn(actor_outputs, noise).detach().cpu()
        return original_rollout(*args, **kwargs)

    monkeypatch.setattr(prior, "_rollout_family_group_vectorized_with_policy", _wrapped_rollout)
    assert algo.collect_rollouts(vec_env, callback, algo.rollout_buffer, n_rollout_steps=algo.n_steps)
    assert torch.equal(recorded["sample"], torch.tensor([[1.0, -2.0]], dtype=torch.float32))
    vec_env.close()


def test_environment_prior_ppo_batch_vec_env_step_wait_resets_state_at_sep_when_enabled(monkeypatch):
    device = _cuda_or_skip()
    cfg, env_cfg = _build_small_ppo_config()
    env_cfg = _strict_env_cfg_for_ppo(env_cfg, aux=False)
    prior = EnvironmentPrior(env_cfg)
    fake_model = _FakeRWKVModel(
        num_features=cfg["prior"]["num_features"],
        x_obs_dim=cfg["transformer"]["x_obs_dim"],
        action_dim=cfg["transformer"]["x_action_dim"],
    ).to(device)
    _, _, vec_env = build_recurrent_ppo(
        model=fake_model,
        env_prior=prior,
        device=str(device),
        num_features=cfg["prior"]["num_features"],
        n_envs=1,
        n_steps=cfg["optimizer"]["ppo_n_steps"],
        learning_rate=cfg["optimizer"]["learning_rate"],
        batch_size=cfg["optimizer"]["ppo_n_steps"],
        n_epochs=1,
        gamma=cfg["optimizer"]["ppo_gamma"],
        gae_lambda=cfg["optimizer"]["ppo_gae_lambda"],
        clip_range=cfg["optimizer"]["ppo_clip_range"],
        clip_range_vf=cfg["optimizer"]["ppo_clip_range_vf"],
        normalize_advantage=cfg["optimizer"]["ppo_normalize_advantage"],
        ent_coef=cfg["optimizer"]["ppo_ent_coef"],
        vf_coef=cfg["optimizer"]["ppo_vf_coef"],
        max_grad_norm=cfg["optimizer"]["ppo_max_grad_norm"],
        target_kl=cfg["optimizer"]["ppo_target_kl"],
        reset_env_state_at_sep=True,
    )
    vec_env.reset()
    vec_env._env["terminal_reset_enabled"] = torch.zeros((vec_env.num_envs,), device=device, dtype=torch.bool)
    vec_env._env["init_state_std"] = torch.ones((vec_env.num_envs,), device=device, dtype=torch.float32)
    vec_env._single_eval_pos = torch.tensor([1], device=device, dtype=torch.long)
    vec_env._single_eval_pos_np = np.array([1], dtype=np.int64)

    def _fixed_randn(*args, **kwargs):
        shape = tuple(args[1])
        return torch.full(shape, 3.0, device=kwargs["device"], dtype=kwargs["dtype"])

    monkeypatch.setattr(prior, "_stack_randn_with_generators", _fixed_randn)
    action = np.zeros((1, vec_env.action_dim), dtype=np.float32)
    vec_env.step_async(action)
    _, _, dones, infos = vec_env.step_wait()

    assert bool(np.any(dones)) is False
    assert infos[0]["sep_state_reset"] is True
    assert torch.allclose(vec_env._state_t, torch.full_like(vec_env._state_t, 3.0))
    assert torch.allclose(vec_env._action_t, torch.zeros_like(vec_env._action_t))
    vec_env.close()


def test_masked_recurrent_rollout_buffer_get_gpu_matches_get(monkeypatch):
    device = _cuda_or_skip()
    cfg, env_cfg = _build_small_ppo_config()
    env_cfg = _strict_env_cfg_for_ppo(env_cfg, aux=False)
    prior = EnvironmentPrior(env_cfg)
    fake_model = _FakeRWKVModel(
        num_features=cfg["prior"]["num_features"],
        x_obs_dim=cfg["transformer"]["x_obs_dim"],
        action_dim=cfg["transformer"]["x_action_dim"],
    ).to(device)
    algo, callback, vec_env = build_recurrent_ppo(
        model=fake_model,
        env_prior=prior,
        device=str(device),
        num_features=cfg["prior"]["num_features"],
        n_envs=cfg["optimizer"]["ppo_n_envs"],
        n_steps=cfg["optimizer"]["ppo_n_steps"],
        learning_rate=cfg["optimizer"]["learning_rate"],
        batch_size=cfg["optimizer"]["ppo_batch_size"],
        n_epochs=1,
        gamma=cfg["optimizer"]["ppo_gamma"],
        gae_lambda=cfg["optimizer"]["ppo_gae_lambda"],
        clip_range=cfg["optimizer"]["ppo_clip_range"],
        clip_range_vf=cfg["optimizer"]["ppo_clip_range_vf"],
        normalize_advantage=cfg["optimizer"]["ppo_normalize_advantage"],
        ent_coef=cfg["optimizer"]["ppo_ent_coef"],
        vf_coef=cfg["optimizer"]["ppo_vf_coef"],
        max_grad_norm=cfg["optimizer"]["ppo_max_grad_norm"],
        target_kl=cfg["optimizer"]["ppo_target_kl"],
    )
    algo.ep_info_buffer = []
    algo.ep_success_buffer = []
    algo._last_obs = vec_env.reset()
    algo._last_episode_starts = np.ones((vec_env.num_envs,), dtype=bool)
    algo._last_lstm_states = algo.policy._dummy_states(vec_env.num_envs)
    callback.init_callback(algo)
    assert algo.collect_rollouts(vec_env, callback, algo.rollout_buffer, n_rollout_steps=algo.n_steps)

    monkeypatch.setattr(np.random, "randint", lambda *args, **kwargs: 0)
    cpu_batch = next(algo.rollout_buffer.get(algo.batch_size))
    gpu_batch = next(algo.rollout_buffer.get_gpu(algo.batch_size))

    assert torch.allclose(cpu_batch.observations, gpu_batch.observations, atol=1e-6, rtol=1e-6)
    assert torch.allclose(cpu_batch.actions, gpu_batch.actions, atol=1e-6, rtol=1e-6)
    assert torch.allclose(cpu_batch.old_values, gpu_batch.old_values, atol=1e-6, rtol=1e-6)
    assert torch.allclose(cpu_batch.old_log_prob, gpu_batch.old_log_prob, atol=1e-6, rtol=1e-6)
    assert torch.allclose(cpu_batch.advantages, gpu_batch.advantages, atol=1e-6, rtol=1e-6)
    assert torch.allclose(cpu_batch.returns, gpu_batch.returns, atol=1e-6, rtol=1e-6)
    assert torch.allclose(cpu_batch.objective_masks, gpu_batch.objective_masks, atol=1e-6, rtol=1e-6)
    assert torch.allclose(cpu_batch.rollout_return_means, gpu_batch.rollout_return_means, atol=1e-6, rtol=1e-6)
    assert torch.allclose(cpu_batch.rollout_return_stds, gpu_batch.rollout_return_stds, atol=1e-6, rtol=1e-6)
    assert torch.equal(cpu_batch.mask, gpu_batch.mask)
    assert torch.allclose(cpu_batch.action_masks, gpu_batch.action_masks, atol=1e-6, rtol=1e-6)
    assert torch.allclose(cpu_batch.next_states, gpu_batch.next_states, atol=1e-6, rtol=1e-6)
    assert torch.allclose(cpu_batch.next_state_masks, gpu_batch.next_state_masks, atol=1e-6, rtol=1e-6)
    assert torch.allclose(cpu_batch.episode_starts, gpu_batch.episode_starts, atol=1e-6, rtol=1e-6)
    assert torch.allclose(cpu_batch.lstm_states.pi[0], gpu_batch.lstm_states.pi[0], atol=1e-6, rtol=1e-6)
    assert torch.allclose(cpu_batch.lstm_states.pi[1], gpu_batch.lstm_states.pi[1], atol=1e-6, rtol=1e-6)
    assert torch.allclose(cpu_batch.lstm_states.vf[0], gpu_batch.lstm_states.vf[0], atol=1e-6, rtol=1e-6)
    assert torch.allclose(cpu_batch.lstm_states.vf[1], gpu_batch.lstm_states.vf[1], atol=1e-6, rtol=1e-6)
    vec_env.close()


def test_masked_recurrent_rollout_buffer_get_gpu_preserves_main_loss_and_grad(monkeypatch):
    device = _cuda_or_skip()
    cfg, env_cfg = _build_small_ppo_config()
    env_cfg = _strict_env_cfg_for_ppo(env_cfg, aux=False)
    prior = EnvironmentPrior(env_cfg)
    fake_model = _FakeRWKVModel(
        num_features=cfg["prior"]["num_features"],
        x_obs_dim=cfg["transformer"]["x_obs_dim"],
        action_dim=cfg["transformer"]["x_action_dim"],
    ).to(device)
    algo, callback, vec_env = build_recurrent_ppo(
        model=fake_model,
        env_prior=prior,
        device=str(device),
        num_features=cfg["prior"]["num_features"],
        n_envs=cfg["optimizer"]["ppo_n_envs"],
        n_steps=cfg["optimizer"]["ppo_n_steps"],
        learning_rate=cfg["optimizer"]["learning_rate"],
        batch_size=cfg["optimizer"]["ppo_batch_size"],
        n_epochs=1,
        gamma=cfg["optimizer"]["ppo_gamma"],
        gae_lambda=cfg["optimizer"]["ppo_gae_lambda"],
        clip_range=cfg["optimizer"]["ppo_clip_range"],
        clip_range_vf=cfg["optimizer"]["ppo_clip_range_vf"],
        normalize_advantage=cfg["optimizer"]["ppo_normalize_advantage"],
        ent_coef=cfg["optimizer"]["ppo_ent_coef"],
        vf_coef=cfg["optimizer"]["ppo_vf_coef"],
        max_grad_norm=cfg["optimizer"]["ppo_max_grad_norm"],
        target_kl=cfg["optimizer"]["ppo_target_kl"],
    )
    algo.ep_info_buffer = []
    algo.ep_success_buffer = []
    algo._last_obs = vec_env.reset()
    algo._last_episode_starts = np.ones((vec_env.num_envs,), dtype=bool)
    algo._last_lstm_states = algo.policy._dummy_states(vec_env.num_envs)
    callback.init_callback(algo)
    assert algo.collect_rollouts(vec_env, callback, algo.rollout_buffer, n_rollout_steps=algo.n_steps)

    monkeypatch.setattr(np.random, "randint", lambda *args, **kwargs: 0)
    cpu_batch = next(algo.rollout_buffer.get(algo.batch_size))
    gpu_batch = next(algo.rollout_buffer.get_gpu(algo.batch_size))

    algo.policy.zero_grad(set_to_none=True)
    cpu_loss = _ppo_main_minibatch_loss(algo, cpu_batch)
    cpu_loss.backward()
    cpu_grad = fake_model.token_proj.weight.grad.detach().clone()

    algo.policy.zero_grad(set_to_none=True)
    gpu_loss = _ppo_main_minibatch_loss(algo, gpu_batch)
    gpu_loss.backward()
    gpu_grad = fake_model.token_proj.weight.grad.detach().clone()

    assert torch.allclose(cpu_loss.detach(), gpu_loss.detach(), atol=1e-6, rtol=1e-6)
    assert torch.allclose(cpu_grad, gpu_grad, atol=1e-6, rtol=1e-6)
    vec_env.close()


def test_masked_recurrent_rollout_buffer_get_gpu_flat_preserves_reinforce_style_q_targets(monkeypatch):
    device = _cuda_or_skip()
    cfg, env_cfg = _build_small_ppo_config()
    env_cfg = _strict_env_cfg_for_ppo(env_cfg, aux=False)
    env_cfg["reinforce_aux_enabled"] = True
    env_cfg["normalized_q_value_weight"] = 1.0
    prior = EnvironmentPrior(env_cfg)
    fake_model = _FakeRWKVModel(
        num_features=cfg["prior"]["num_features"],
        x_obs_dim=cfg["transformer"]["x_obs_dim"],
        action_dim=cfg["transformer"]["x_action_dim"],
        normalized_q_head=True,
    ).to(device)
    algo, callback, vec_env = build_recurrent_ppo(
        model=fake_model,
        env_prior=prior,
        device=str(device),
        num_features=cfg["prior"]["num_features"],
        n_envs=cfg["optimizer"]["ppo_n_envs"],
        n_steps=cfg["optimizer"]["ppo_n_steps"],
        learning_rate=cfg["optimizer"]["learning_rate"],
        batch_size=cfg["optimizer"]["ppo_batch_size"],
        n_epochs=1,
        gamma=cfg["optimizer"]["ppo_gamma"],
        gae_lambda=cfg["optimizer"]["ppo_gae_lambda"],
        clip_range=cfg["optimizer"]["ppo_clip_range"],
        clip_range_vf=cfg["optimizer"]["ppo_clip_range_vf"],
        normalize_advantage=cfg["optimizer"]["ppo_normalize_advantage"],
        ent_coef=cfg["optimizer"]["ppo_ent_coef"],
        vf_coef=cfg["optimizer"]["ppo_vf_coef"],
        max_grad_norm=cfg["optimizer"]["ppo_max_grad_norm"],
        target_kl=cfg["optimizer"]["ppo_target_kl"],
    )
    algo.ep_info_buffer = []
    algo.ep_success_buffer = []
    algo._last_obs = vec_env.reset()
    algo._last_episode_starts = np.ones((vec_env.num_envs,), dtype=bool)
    algo._last_lstm_states = algo.policy._dummy_states(vec_env.num_envs)
    callback.init_callback(algo)
    assert algo.collect_rollouts(vec_env, callback, algo.rollout_buffer, n_rollout_steps=algo.n_steps)

    monkeypatch.setattr(np.random, "randint", lambda *args, **kwargs: 0)
    padded_batch = next(algo.rollout_buffer.get(algo.batch_size))
    flat_batch = next(algo.rollout_buffer.get_gpu_flat(algo.batch_size))

    padded_targets = algo._normalize_masked_returns_like_reinforce(
        padded_batch.returns,
        padded_batch.mask > 1e-8,
        n_seq=int(padded_batch.lstm_states.pi[0].shape[1]),
        eps=1e-6,
    )
    flat_targets = algo._normalize_flat_sequence_returns_like_reinforce(
        flat_batch.returns,
        seq_start_indices=flat_batch.seq_start_indices,
        seq_lengths=flat_batch.seq_lengths,
        eps=1e-6,
    )

    assert torch.allclose(
        padded_targets[padded_batch.mask > 1e-8],
        flat_targets,
        atol=1e-6,
        rtol=1e-6,
    )
    vec_env.close()


def test_masked_recurrent_rollout_buffer_get_gpu_flat_preserves_streamed_main_loss_and_grad(monkeypatch):
    device = _cuda_or_skip()
    cfg, env_cfg = _build_small_ppo_config()
    env_cfg = _strict_env_cfg_for_ppo(env_cfg, aux=False)
    prior = EnvironmentPrior(env_cfg)
    fake_model = _FakeRWKVModel(
        num_features=cfg["prior"]["num_features"],
        x_obs_dim=cfg["transformer"]["x_obs_dim"],
        action_dim=cfg["transformer"]["x_action_dim"],
    ).to(device)
    algo, callback, vec_env = build_recurrent_ppo(
        model=fake_model,
        env_prior=prior,
        device=str(device),
        num_features=cfg["prior"]["num_features"],
        n_envs=cfg["optimizer"]["ppo_n_envs"],
        n_steps=cfg["optimizer"]["ppo_n_steps"],
        learning_rate=cfg["optimizer"]["learning_rate"],
        batch_size=cfg["optimizer"]["ppo_batch_size"],
        n_epochs=1,
        gamma=cfg["optimizer"]["ppo_gamma"],
        gae_lambda=cfg["optimizer"]["ppo_gae_lambda"],
        clip_range=cfg["optimizer"]["ppo_clip_range"],
        clip_range_vf=cfg["optimizer"]["ppo_clip_range_vf"],
        normalize_advantage=cfg["optimizer"]["ppo_normalize_advantage"],
        ent_coef=cfg["optimizer"]["ppo_ent_coef"],
        vf_coef=cfg["optimizer"]["ppo_vf_coef"],
        max_grad_norm=cfg["optimizer"]["ppo_max_grad_norm"],
        target_kl=cfg["optimizer"]["ppo_target_kl"],
    )
    algo.ep_info_buffer = []
    algo.ep_success_buffer = []
    algo._last_obs = vec_env.reset()
    algo._last_episode_starts = np.ones((vec_env.num_envs,), dtype=bool)
    algo._last_lstm_states = algo.policy._dummy_states(vec_env.num_envs)
    callback.init_callback(algo)
    assert algo.collect_rollouts(vec_env, callback, algo.rollout_buffer, n_rollout_steps=algo.n_steps)

    monkeypatch.setattr(np.random, "randint", lambda *args, **kwargs: 0)
    padded_batch = next(algo.rollout_buffer.get(algo.batch_size))
    flat_batch = next(algo.rollout_buffer.get_gpu_flat(algo.batch_size))

    algo.policy.zero_grad(set_to_none=True)
    padded_loss = padded_batch.returns.new_zeros(())
    n_seq_total = int(padded_batch.lstm_states.pi[0].shape[1])
    seq_subbatch_size = algo._resolve_sequence_subbatch_size(padded_batch)
    advantages = padded_batch.advantages
    mask = (padded_batch.mask > 1e-8) & (padded_batch.objective_masks > 1e-8)
    if algo.normalize_advantage:
        advantages = _normalize_advantages_with_mask(advantages, mask, eps=1e-8)
    valid_total = mask.sum().clamp_min(1).to(device=padded_batch.returns.device, dtype=padded_batch.returns.dtype)
    for start_seq in range(0, n_seq_total, seq_subbatch_size):
        end_seq = min(n_seq_total, start_seq + seq_subbatch_size)
        sub_rollout_data = _slice_masked_rollout_sequence_batch(
            padded_batch,
            start_seq=start_seq,
            end_seq=end_seq,
        )
        sub_mask = (sub_rollout_data.mask > 1e-8) & (sub_rollout_data.objective_masks > 1e-8)
        sub_mask_f = sub_mask.to(device=sub_rollout_data.returns.device, dtype=sub_rollout_data.returns.dtype)
        sub_advantages = _slice_padded_sequence_tensor(
            advantages,
            n_seq=n_seq_total,
            start_seq=start_seq,
            end_seq=end_seq,
        )
        eval_outputs = algo.policy.evaluate_actions_with_hidden(
            sub_rollout_data.observations,
            sub_rollout_data.actions,
            sub_rollout_data.lstm_states,
            sub_rollout_data.episode_starts,
            action_masks=sub_rollout_data.action_masks,
        )
        value_logits = eval_outputs["value_logits"]
        log_prob = eval_outputs["log_prob"]
        entropy = eval_outputs["entropy"]
        ratio = torch.exp(log_prob - sub_rollout_data.old_log_prob)
        clipped_objective = torch.min(
            sub_advantages * ratio,
            sub_advantages * torch.clamp(ratio, 1 - algo.clip_range(1.0), 1 + algo.clip_range(1.0)),
        )
        policy_loss = -(
            (clipped_objective * sub_mask_f.to(device=clipped_objective.device, dtype=clipped_objective.dtype)).sum()
            / valid_total.to(device=clipped_objective.device, dtype=clipped_objective.dtype)
        )
        value_errors = algo.policy.get_value_bardist()(
            value_logits.to(dtype=torch.float32),
            sub_rollout_data.returns.to(device=value_logits.device, dtype=torch.float32),
        )
        value_loss = (
            (value_errors * sub_mask_f.to(device=value_errors.device, dtype=value_errors.dtype)).sum()
            / valid_total.to(device=value_errors.device, dtype=value_errors.dtype)
        )
        entropy_terms = entropy if entropy is not None else -log_prob
        entropy_loss = -(
            (entropy_terms * sub_mask_f.to(device=entropy_terms.device, dtype=entropy_terms.dtype)).sum()
            / valid_total.to(device=entropy_terms.device, dtype=entropy_terms.dtype)
        )
        padded_loss = padded_loss + (policy_loss + algo.ent_coef * entropy_loss + algo.vf_coef * value_loss)
    padded_loss.backward()
    padded_grad = fake_model.token_proj.weight.grad.detach().clone()

    algo.policy.zero_grad(set_to_none=True)
    flat_loss = _ppo_main_streamed_flat_minibatch_loss(algo, flat_batch)
    flat_loss.backward()
    flat_grad = fake_model.token_proj.weight.grad.detach().clone()

    assert torch.allclose(padded_loss.detach(), flat_loss.detach(), atol=1e-6, rtol=1e-6)
    assert torch.allclose(padded_grad, flat_grad, atol=1e-6, rtol=1e-6)
    vec_env.close()


def test_policy_evaluate_actions_with_hidden_flat_matches_padded_outputs_and_grad(monkeypatch):
    device = _cuda_or_skip()
    cfg, env_cfg = _build_small_ppo_config()
    env_cfg = _strict_env_cfg_for_ppo(env_cfg, aux=False)
    prior = EnvironmentPrior(env_cfg)
    fake_model = _FakeRWKVModel(
        num_features=cfg["prior"]["num_features"],
        x_obs_dim=cfg["transformer"]["x_obs_dim"],
        action_dim=cfg["transformer"]["x_action_dim"],
    ).to(device)
    algo, callback, vec_env = build_recurrent_ppo(
        model=fake_model,
        env_prior=prior,
        device=str(device),
        num_features=cfg["prior"]["num_features"],
        n_envs=cfg["optimizer"]["ppo_n_envs"],
        n_steps=cfg["optimizer"]["ppo_n_steps"],
        learning_rate=cfg["optimizer"]["learning_rate"],
        batch_size=cfg["optimizer"]["ppo_batch_size"],
        n_epochs=1,
        gamma=cfg["optimizer"]["ppo_gamma"],
        gae_lambda=cfg["optimizer"]["ppo_gae_lambda"],
        clip_range=cfg["optimizer"]["ppo_clip_range"],
        clip_range_vf=cfg["optimizer"]["ppo_clip_range_vf"],
        normalize_advantage=cfg["optimizer"]["ppo_normalize_advantage"],
        ent_coef=cfg["optimizer"]["ppo_ent_coef"],
        vf_coef=cfg["optimizer"]["ppo_vf_coef"],
        max_grad_norm=cfg["optimizer"]["ppo_max_grad_norm"],
        target_kl=cfg["optimizer"]["ppo_target_kl"],
    )
    algo.ep_info_buffer = []
    algo.ep_success_buffer = []
    algo._last_obs = vec_env.reset()
    algo._last_episode_starts = np.ones((vec_env.num_envs,), dtype=bool)
    algo._last_lstm_states = algo.policy._dummy_states(vec_env.num_envs)
    callback.init_callback(algo)
    assert algo.collect_rollouts(vec_env, callback, algo.rollout_buffer, n_rollout_steps=algo.n_steps)

    monkeypatch.setattr(np.random, "randint", lambda *args, **kwargs: 0)
    padded_batch = next(algo.rollout_buffer.get(algo.batch_size))
    flat_batch = next(algo.rollout_buffer.get_gpu_flat(algo.batch_size))

    padded_outputs = algo.policy.evaluate_actions_with_hidden(
        padded_batch.observations,
        padded_batch.actions,
        padded_batch.lstm_states,
        padded_batch.episode_starts,
        action_masks=padded_batch.action_masks,
    )
    flat_outputs = algo.policy.evaluate_actions_with_hidden_flat(
        flat_batch.observations,
        flat_batch.actions,
        seq_lengths=flat_batch.seq_lengths,
        action_masks=flat_batch.action_masks,
    )

    mask = padded_batch.mask > 1e-8
    assert torch.allclose(padded_outputs["values"].flatten()[mask], flat_outputs["values"].flatten(), atol=1e-6, rtol=1e-6)
    assert torch.allclose(padded_outputs["log_prob"][mask], flat_outputs["log_prob"], atol=1e-6, rtol=1e-6)
    assert torch.allclose(padded_outputs["entropy"][mask], flat_outputs["entropy"], atol=1e-6, rtol=1e-6)

    algo.policy.zero_grad(set_to_none=True)
    padded_scalar = (
        padded_outputs["values"].flatten()[mask].sum()
        + padded_outputs["log_prob"][mask].sum()
        + padded_outputs["entropy"][mask].sum()
    )
    padded_scalar.backward()
    padded_grad = fake_model.token_proj.weight.grad.detach().clone()

    algo.policy.zero_grad(set_to_none=True)
    flat_scalar = (
        flat_outputs["values"].flatten().sum()
        + flat_outputs["log_prob"].sum()
        + flat_outputs["entropy"].sum()
    )
    flat_scalar.backward()
    flat_grad = fake_model.token_proj.weight.grad.detach().clone()

    assert torch.allclose(padded_grad, flat_grad, atol=1e-6, rtol=1e-6)
    vec_env.close()


def test_official_sequence_hidden_flat_packed_matches_exact_buckets_and_grad():
    device = _cuda_or_skip()
    cfg, env_cfg = _build_small_ppo_config()
    env_cfg = _strict_env_cfg_for_ppo(env_cfg, aux=False)
    prior = EnvironmentPrior(env_cfg)
    fake_model = _FakeRWKVModel(
        num_features=cfg["prior"]["num_features"],
        x_obs_dim=cfg["transformer"]["x_obs_dim"],
        action_dim=cfg["transformer"]["x_action_dim"],
    ).to(device)
    algo, callback, vec_env = build_recurrent_ppo(
        model=fake_model,
        env_prior=prior,
        device=str(device),
        num_features=cfg["prior"]["num_features"],
        n_envs=cfg["optimizer"]["ppo_n_envs"],
        n_steps=cfg["optimizer"]["ppo_n_steps"],
        learning_rate=cfg["optimizer"]["learning_rate"],
        batch_size=cfg["optimizer"]["ppo_batch_size"],
        n_epochs=1,
        gamma=cfg["optimizer"]["ppo_gamma"],
        gae_lambda=cfg["optimizer"]["ppo_gae_lambda"],
        clip_range=cfg["optimizer"]["ppo_clip_range"],
        clip_range_vf=cfg["optimizer"]["ppo_clip_range_vf"],
        normalize_advantage=cfg["optimizer"]["ppo_normalize_advantage"],
        ent_coef=cfg["optimizer"]["ppo_ent_coef"],
        vf_coef=cfg["optimizer"]["ppo_vf_coef"],
        max_grad_norm=cfg["optimizer"]["ppo_max_grad_norm"],
        target_kl=cfg["optimizer"]["ppo_target_kl"],
    )

    torch.manual_seed(0)
    seq_lengths = np.asarray([5, 3, 4, 2], dtype=np.int64)
    obs = torch.randn((int(seq_lengths.sum()), cfg["prior"]["num_features"]), device=device, dtype=torch.float32)

    algo.policy.zero_grad(set_to_none=True)
    hidden_exact = algo.policy._official_sequence_hidden_flat_exact_buckets(obs, seq_lengths=seq_lengths)
    hidden_exact.sum().backward()
    grad_exact = fake_model.token_proj.weight.grad.detach().clone()

    algo.policy.zero_grad(set_to_none=True)
    hidden_packed = algo.policy._official_sequence_hidden_flat_packed(obs, seq_lengths=seq_lengths)
    hidden_packed.sum().backward()
    grad_packed = fake_model.token_proj.weight.grad.detach().clone()

    assert torch.allclose(hidden_exact.detach(), hidden_packed.detach(), atol=1e-6, rtol=1e-6)
    assert torch.allclose(grad_exact, grad_packed, atol=1e-6, rtol=1e-6)
    vec_env.close()


def test_decode_flow_from_hidden_flat_packed_matches_exact_buckets_and_grad():
    device = _cuda_or_skip()
    cfg, env_cfg = _build_small_ppo_config()
    env_cfg = _strict_env_cfg_for_ppo(env_cfg, aux=True)
    prior = EnvironmentPrior(env_cfg)
    fake_model = _FakeRWKVModel(
        num_features=cfg["prior"]["num_features"],
        x_obs_dim=cfg["transformer"]["x_obs_dim"],
        action_dim=cfg["transformer"]["x_action_dim"],
        normalized_q_head=True,
        next_state_flow_dim=8,
    ).to(device)
    algo, callback, vec_env = build_recurrent_ppo(
        model=fake_model,
        env_prior=prior,
        device=str(device),
        num_features=cfg["prior"]["num_features"],
        n_envs=cfg["optimizer"]["ppo_n_envs"],
        n_steps=cfg["optimizer"]["ppo_n_steps"],
        learning_rate=cfg["optimizer"]["learning_rate"],
        batch_size=cfg["optimizer"]["ppo_batch_size"],
        n_epochs=1,
        gamma=cfg["optimizer"]["ppo_gamma"],
        gae_lambda=cfg["optimizer"]["ppo_gae_lambda"],
        clip_range=cfg["optimizer"]["ppo_clip_range"],
        clip_range_vf=cfg["optimizer"]["ppo_clip_range_vf"],
        normalize_advantage=cfg["optimizer"]["ppo_normalize_advantage"],
        ent_coef=cfg["optimizer"]["ppo_ent_coef"],
        vf_coef=cfg["optimizer"]["ppo_vf_coef"],
        max_grad_norm=cfg["optimizer"]["ppo_max_grad_norm"],
        target_kl=cfg["optimizer"]["ppo_target_kl"],
    )

    torch.manual_seed(0)
    seq_lengths = np.asarray([5, 3, 4, 2], dtype=np.int64)
    total_steps = int(seq_lengths.sum())
    hidden = torch.randn((total_steps, fake_model.emsize), device=device, dtype=torch.float32, requires_grad=True)
    actions = torch.randn((total_steps, cfg["transformer"]["x_action_dim"]), device=device, dtype=torch.float32)
    flow_xt = torch.randn((total_steps, 8), device=device, dtype=torch.float32)
    flow_t = torch.rand((total_steps, 1), device=device, dtype=torch.float32)

    algo.policy.zero_grad(set_to_none=True)
    if hidden.grad is not None:
        hidden.grad.zero_()
    flow_exact = algo.policy._decode_flow_from_hidden_flat_exact_buckets(
        hidden,
        actions,
        flow_xt,
        flow_t,
        seq_lengths=seq_lengths,
    )
    flow_exact.sum().backward()
    grad_exact = hidden.grad.detach().clone()

    algo.policy.zero_grad(set_to_none=True)
    hidden.grad = None
    flow_packed = algo.policy._decode_flow_from_hidden_flat_packed(
        hidden,
        actions,
        flow_xt,
        flow_t,
        seq_lengths=seq_lengths,
    )
    flow_packed.sum().backward()
    grad_packed = hidden.grad.detach().clone()

    assert torch.allclose(flow_exact.detach(), flow_packed.detach(), atol=1e-6, rtol=1e-6)
    assert torch.allclose(grad_exact, grad_packed, atol=1e-6, rtol=1e-6)
    vec_env.close()


def test_policy_fused_hidden_and_aux_flat_matches_legacy_decode_and_grad():
    device = _cuda_or_skip()
    cfg, env_cfg = _build_small_ppo_config()
    env_cfg = _strict_env_cfg_for_ppo(env_cfg, aux=True)
    prior = EnvironmentPrior(env_cfg)
    fake_model = _FakeRWKVModel(
        num_features=cfg["prior"]["num_features"],
        x_obs_dim=cfg["transformer"]["x_obs_dim"],
        action_dim=cfg["transformer"]["x_action_dim"],
        normalized_q_head=True,
        next_state_flow_dim=8,
    ).to(device)
    algo, callback, vec_env = build_recurrent_ppo(
        model=fake_model,
        env_prior=prior,
        device=str(device),
        num_features=cfg["prior"]["num_features"],
        n_envs=cfg["optimizer"]["ppo_n_envs"],
        n_steps=cfg["optimizer"]["ppo_n_steps"],
        learning_rate=cfg["optimizer"]["learning_rate"],
        batch_size=cfg["optimizer"]["ppo_batch_size"],
        n_epochs=1,
        gamma=cfg["optimizer"]["ppo_gamma"],
        gae_lambda=cfg["optimizer"]["ppo_gae_lambda"],
        clip_range=cfg["optimizer"]["ppo_clip_range"],
        clip_range_vf=cfg["optimizer"]["ppo_clip_range_vf"],
        normalize_advantage=cfg["optimizer"]["ppo_normalize_advantage"],
        ent_coef=cfg["optimizer"]["ppo_ent_coef"],
        vf_coef=cfg["optimizer"]["ppo_vf_coef"],
        max_grad_norm=cfg["optimizer"]["ppo_max_grad_norm"],
        target_kl=cfg["optimizer"]["ppo_target_kl"],
    )

    torch.manual_seed(0)
    seq_lengths = np.asarray([5, 3, 4, 2], dtype=np.int64)
    total_steps = int(seq_lengths.sum())
    obs = torch.randn((total_steps, cfg["prior"]["num_features"]), device=device, dtype=torch.float32)
    actions = torch.randn((total_steps, cfg["transformer"]["x_action_dim"]), device=device, dtype=torch.float32)
    flow_xt = torch.randn((total_steps, 8), device=device, dtype=torch.float32)
    flow_t = torch.rand((total_steps, 1), device=device, dtype=torch.float32)

    algo.policy.zero_grad(set_to_none=True)
    hidden_legacy = algo.policy._official_sequence_hidden_flat_packed(obs, seq_lengths=seq_lengths)
    aux_legacy = algo.policy.decode_aux_from_hidden(
        hidden_legacy,
        actions=actions,
        flow_matching_xt=flow_xt,
        flow_matching_t=flow_t,
        seq_lengths=seq_lengths,
    )
    legacy_scalar = hidden_legacy.sum() + aux_legacy["normalized_q_logits"].sum() + aux_legacy["next_state_flow"].sum()
    legacy_scalar.backward()
    legacy_grad = fake_model.token_proj.weight.grad.detach().clone()

    algo.policy.zero_grad(set_to_none=True)
    fused_outputs = algo.policy._official_sequence_hidden_and_aux_flat_packed(
        obs,
        seq_lengths=seq_lengths,
        actions=actions,
        include_normalized_q_logits=True,
        flow_matching_xt=flow_xt,
        flow_matching_t=flow_t,
    )
    fused_scalar = (
        fused_outputs["hidden"].sum()
        + fused_outputs["normalized_q_logits"].sum()
        + fused_outputs["next_state_flow"].sum()
    )
    fused_scalar.backward()
    fused_grad = fake_model.token_proj.weight.grad.detach().clone()

    assert torch.allclose(hidden_legacy.detach(), fused_outputs["hidden"].detach(), atol=1e-6, rtol=1e-6)
    assert torch.allclose(
        aux_legacy["normalized_q_logits"].detach(),
        fused_outputs["normalized_q_logits"].detach(),
        atol=1e-6,
        rtol=1e-6,
    )
    assert torch.allclose(
        aux_legacy["next_state_flow"].detach(),
        fused_outputs["next_state_flow"].detach(),
        atol=1e-6,
        rtol=1e-6,
    )
    assert torch.allclose(legacy_grad, fused_grad, atol=1e-6, rtol=1e-6)
    vec_env.close()


def test_flat_aux_losses_with_predecoded_outputs_match_legacy_decode_and_grad():
    device = _cuda_or_skip()
    cfg, env_cfg = _build_small_ppo_config()
    env_cfg = _strict_env_cfg_for_ppo(env_cfg, aux=True)
    prior = EnvironmentPrior(env_cfg)
    fake_model = _FakeRWKVModel(
        num_features=cfg["prior"]["num_features"],
        x_obs_dim=cfg["transformer"]["x_obs_dim"],
        action_dim=cfg["transformer"]["x_action_dim"],
        normalized_q_head=True,
        next_state_flow_dim=8,
    ).to(device)
    algo, callback, vec_env = build_recurrent_ppo(
        model=fake_model,
        env_prior=prior,
        device=str(device),
        num_features=cfg["prior"]["num_features"],
        n_envs=cfg["optimizer"]["ppo_n_envs"],
        n_steps=cfg["optimizer"]["ppo_n_steps"],
        learning_rate=cfg["optimizer"]["learning_rate"],
        batch_size=cfg["optimizer"]["ppo_batch_size"],
        n_epochs=1,
        gamma=cfg["optimizer"]["ppo_gamma"],
        gae_lambda=cfg["optimizer"]["ppo_gae_lambda"],
        clip_range=cfg["optimizer"]["ppo_clip_range"],
        clip_range_vf=cfg["optimizer"]["ppo_clip_range_vf"],
        normalize_advantage=cfg["optimizer"]["ppo_normalize_advantage"],
        ent_coef=cfg["optimizer"]["ppo_ent_coef"],
        vf_coef=cfg["optimizer"]["ppo_vf_coef"],
        max_grad_norm=cfg["optimizer"]["ppo_max_grad_norm"],
        target_kl=cfg["optimizer"]["ppo_target_kl"],
    )

    torch.manual_seed(0)
    seq_lengths = np.asarray([5, 3, 4, 2], dtype=np.int64)
    total_steps = int(seq_lengths.sum())
    obs = torch.randn((total_steps, cfg["prior"]["num_features"]), device=device, dtype=torch.float32)
    actions = torch.randn((total_steps, cfg["transformer"]["x_action_dim"]), device=device, dtype=torch.float32)
    returns = torch.randn((total_steps,), device=device, dtype=torch.float32)
    normalized_q_targets = torch.randn((total_steps,), device=device, dtype=torch.float32)
    next_states = torch.randn((total_steps, 8), device=device, dtype=torch.float32)
    next_state_masks = torch.ones((total_steps, 8), device=device, dtype=torch.float32)
    flow_xt = torch.randn((total_steps, 8), device=device, dtype=torch.float32)
    flow_t = torch.rand((total_steps, 1), device=device, dtype=torch.float32)
    flow_dx = torch.randn((total_steps, 8), device=device, dtype=torch.float32)

    hidden_legacy = algo.policy._official_sequence_hidden_flat_packed(obs, seq_lengths=seq_lengths)
    algo.policy.zero_grad(set_to_none=True)
    legacy_aux_loss, legacy_aux_stats = algo._compute_aux_losses_flat(
        hidden=hidden_legacy,
        actions=actions,
        seq_lengths=seq_lengths,
        normalized_q_targets=normalized_q_targets,
        q_target_denom=returns.new_tensor(float(max(1, int(returns.numel())))),
        flow_matching_xt=flow_xt,
        flow_matching_t=flow_t,
        flow_matching_dx=flow_dx,
        next_states=next_states,
        next_state_masks=next_state_masks,
        flow_denom=next_state_masks.sum().clamp_min(1.0),
    )
    legacy_aux_loss.backward()
    legacy_grad = fake_model.token_proj.weight.grad.detach().clone()

    fused_outputs = algo.policy._official_sequence_hidden_and_aux_flat_packed(
        obs,
        seq_lengths=seq_lengths,
        actions=actions,
        include_normalized_q_logits=True,
        flow_matching_xt=flow_xt,
        flow_matching_t=flow_t,
    )
    algo.policy.zero_grad(set_to_none=True)
    fused_aux_loss, fused_aux_stats = algo._compute_aux_losses_flat(
        hidden=fused_outputs["hidden"],
        actions=actions,
        seq_lengths=seq_lengths,
        normalized_q_logits=fused_outputs["normalized_q_logits"],
        next_state_flow=fused_outputs["next_state_flow"],
        normalized_q_targets=normalized_q_targets,
        q_target_denom=returns.new_tensor(float(max(1, int(returns.numel())))),
        flow_matching_xt=flow_xt,
        flow_matching_t=flow_t,
        flow_matching_dx=flow_dx,
        next_states=next_states,
        next_state_masks=next_state_masks,
        flow_denom=next_state_masks.sum().clamp_min(1.0),
    )
    fused_aux_loss.backward()
    fused_grad = fake_model.token_proj.weight.grad.detach().clone()

    assert torch.allclose(legacy_aux_loss.detach(), fused_aux_loss.detach(), atol=1e-6, rtol=1e-6)
    assert torch.allclose(
        legacy_aux_stats["normalized_q_value_loss"],
        fused_aux_stats["normalized_q_value_loss"],
        atol=1e-6,
        rtol=1e-6,
    )
    assert torch.allclose(
        legacy_aux_stats["next_state_flow_matching_loss"],
        fused_aux_stats["next_state_flow_matching_loss"],
        atol=1e-6,
        rtol=1e-6,
    )
    assert torch.allclose(legacy_grad, fused_grad, atol=1e-6, rtol=1e-6)
    vec_env.close()


def test_flat_aux_losses_match_padded_q_and_flow_loss_and_grad(monkeypatch):
    device = _cuda_or_skip()
    cfg, env_cfg = _build_small_ppo_config()
    env_cfg = _strict_env_cfg_for_ppo(env_cfg, aux=True)
    prior = EnvironmentPrior(env_cfg)
    fake_model = _FakeRWKVModel(
        num_features=cfg["prior"]["num_features"],
        x_obs_dim=cfg["transformer"]["x_obs_dim"],
        action_dim=cfg["transformer"]["x_action_dim"],
        normalized_q_head=True,
        next_state_flow_dim=8,
    ).to(device)
    algo, callback, vec_env = build_recurrent_ppo(
        model=fake_model,
        env_prior=prior,
        device=str(device),
        num_features=cfg["prior"]["num_features"],
        n_envs=cfg["optimizer"]["ppo_n_envs"],
        n_steps=cfg["optimizer"]["ppo_n_steps"],
        learning_rate=cfg["optimizer"]["learning_rate"],
        batch_size=cfg["optimizer"]["ppo_batch_size"],
        n_epochs=1,
        gamma=cfg["optimizer"]["ppo_gamma"],
        gae_lambda=cfg["optimizer"]["ppo_gae_lambda"],
        clip_range=cfg["optimizer"]["ppo_clip_range"],
        clip_range_vf=cfg["optimizer"]["ppo_clip_range_vf"],
        normalize_advantage=cfg["optimizer"]["ppo_normalize_advantage"],
        ent_coef=cfg["optimizer"]["ppo_ent_coef"],
        vf_coef=cfg["optimizer"]["ppo_vf_coef"],
        max_grad_norm=cfg["optimizer"]["ppo_max_grad_norm"],
        target_kl=cfg["optimizer"]["ppo_target_kl"],
    )
    algo.ep_info_buffer = []
    algo.ep_success_buffer = []
    algo._last_obs = vec_env.reset()
    algo._last_episode_starts = np.ones((vec_env.num_envs,), dtype=bool)
    algo._last_lstm_states = algo.policy._dummy_states(vec_env.num_envs)
    callback.init_callback(algo)
    assert algo.collect_rollouts(vec_env, callback, algo.rollout_buffer, n_rollout_steps=algo.n_steps)

    monkeypatch.setattr(np.random, "randint", lambda *args, **kwargs: 0)
    padded_batch = next(algo.rollout_buffer.get(algo.batch_size))
    flat_batch = next(algo.rollout_buffer.get_gpu_flat(algo.batch_size))

    padded_outputs = algo.policy.evaluate_actions_with_hidden(
        padded_batch.observations,
        padded_batch.actions,
        padded_batch.lstm_states,
        padded_batch.episode_starts,
        action_masks=padded_batch.action_masks,
    )
    padded_objective = (padded_batch.mask > 1e-8) & (padded_batch.objective_masks > 1e-8)
    flat_objective = flat_batch.objective_masks > 1e-8
    flat_outputs = algo.policy.evaluate_actions_with_hidden_flat(
        flat_batch.observations,
        flat_batch.actions,
        seq_lengths=flat_batch.seq_lengths,
        action_masks=flat_batch.action_masks,
    )
    padded_q_targets = algo._normalize_masked_returns_like_reinforce(
        _recover_raw_from_value_space(
            padded_batch.returns.detach(),
            value_means=padded_batch.rollout_return_means.detach(),
            value_stds=padded_batch.rollout_return_stds.detach(),
        ),
        padded_objective,
        n_seq=int(padded_batch.lstm_states.pi[0].shape[1]),
        eps=1e-6,
    )
    flat_q_targets = algo._normalize_flat_sequence_returns_like_reinforce(
        _recover_raw_from_value_space(
            flat_batch.returns.detach(),
            value_means=flat_batch.rollout_return_means.detach(),
            value_stds=flat_batch.rollout_return_stds.detach(),
        ),
        seq_start_indices=flat_batch.seq_start_indices,
        seq_lengths=flat_batch.seq_lengths,
        valid_mask_flat=flat_objective,
        eps=1e-6,
    )
    torch.manual_seed(0)
    flat_flow_xt, flat_flow_t, flat_flow_dx = prior._sample_condot_flow_matching_path(flat_batch.next_states.detach())
    padded_flow_xt = algo._pad_flat_sequence_values(
        flat_flow_xt,
        seq_start_indices=flat_batch.seq_start_indices,
    )
    padded_flow_t = algo._pad_flat_sequence_values(
        flat_flow_t,
        seq_start_indices=flat_batch.seq_start_indices,
    )
    padded_flow_dx = algo._pad_flat_sequence_values(
        flat_flow_dx,
        seq_start_indices=flat_batch.seq_start_indices,
    )

    algo.policy.zero_grad(set_to_none=True)
    padded_aux_loss, padded_aux_stats = algo._compute_aux_losses(
        padded_batch,
        eval_outputs=padded_outputs,
        n_seq=int(padded_batch.lstm_states.pi[0].shape[1]),
        normalized_q_targets=padded_q_targets,
        objective_mask=padded_objective,
        q_target_denom=padded_objective.sum().clamp_min(1),
        flow_matching_xt=padded_flow_xt,
        flow_matching_t=padded_flow_t,
        flow_matching_dx=padded_flow_dx,
        flow_denom=(
            padded_batch.next_state_masks.to(dtype=padded_batch.next_states.dtype)
            * (padded_batch.mask > 1e-8).to(dtype=padded_batch.next_states.dtype).unsqueeze(-1)
        ).sum().clamp_min(1.0),
    )
    padded_aux_loss.backward()
    padded_grad = fake_model.token_proj.weight.grad.detach().clone()

    algo.policy.zero_grad(set_to_none=True)
    flat_aux_loss, flat_aux_stats = algo._compute_aux_losses_flat(
        hidden=flat_outputs["hidden"],
        actions=flat_outputs["actions"],
        seq_lengths=flat_batch.seq_lengths,
        normalized_q_targets=flat_q_targets,
        objective_mask=flat_objective,
        q_target_denom=flat_objective.sum().clamp_min(1),
        flow_matching_xt=flat_flow_xt,
        flow_matching_t=flat_flow_t,
        flow_matching_dx=flat_flow_dx,
        next_state_masks=flat_batch.next_state_masks,
        flow_denom=flat_batch.next_state_masks.sum().clamp_min(1.0),
    )
    flat_aux_loss.backward()
    flat_grad = fake_model.token_proj.weight.grad.detach().clone()

    assert torch.allclose(padded_aux_stats["normalized_q_value_loss"], flat_aux_stats["normalized_q_value_loss"], atol=1e-6, rtol=1e-6)
    assert torch.allclose(
        padded_aux_stats["next_state_flow_matching_loss"],
        flat_aux_stats["next_state_flow_matching_loss"],
        atol=1e-4,
        rtol=1e-4,
    )
    assert torch.allclose(padded_grad, flat_grad, atol=1e-6, rtol=1e-6)
    vec_env.close()


def test_flat_q_target_cache_matches_legacy_normalization_with_fragments(monkeypatch):
    device = _cuda_or_skip()
    cfg, env_cfg = _build_small_ppo_config()
    env_cfg = _strict_env_cfg_for_ppo(env_cfg, aux=True)
    env_cfg["next_state_flow_matching_weight"] = 0.0
    fake_model = _FakeRWKVModel(
        num_features=cfg["prior"]["num_features"],
        x_obs_dim=cfg["transformer"]["x_obs_dim"],
        action_dim=cfg["transformer"]["x_action_dim"],
        normalized_q_head=True,
    ).to(device)
    algo, callback, vec_env = build_recurrent_ppo(
        model=fake_model,
        env_prior=EnvironmentPrior(env_cfg),
        device=str(device),
        num_features=cfg["prior"]["num_features"],
        n_envs=cfg["optimizer"]["ppo_n_envs"],
        n_steps=cfg["optimizer"]["ppo_n_steps"],
        learning_rate=cfg["optimizer"]["learning_rate"],
        batch_size=cfg["optimizer"]["ppo_batch_size"],
        n_epochs=1,
        gamma=cfg["optimizer"]["ppo_gamma"],
        gae_lambda=cfg["optimizer"]["ppo_gae_lambda"],
        clip_range=cfg["optimizer"]["ppo_clip_range"],
        clip_range_vf=cfg["optimizer"]["ppo_clip_range_vf"],
        normalize_advantage=cfg["optimizer"]["ppo_normalize_advantage"],
        ent_coef=cfg["optimizer"]["ppo_ent_coef"],
        vf_coef=cfg["optimizer"]["ppo_vf_coef"],
        max_grad_norm=cfg["optimizer"]["ppo_max_grad_norm"],
        target_kl=cfg["optimizer"]["ppo_target_kl"],
    )
    algo.ep_info_buffer = []
    algo.ep_success_buffer = []
    algo._last_obs = vec_env.reset()
    algo._last_episode_starts = np.ones((vec_env.num_envs,), dtype=bool)
    algo._last_lstm_states = algo.policy._dummy_states(vec_env.num_envs)
    callback.init_callback(algo)
    assert algo.collect_rollouts(vec_env, callback, algo.rollout_buffer, n_rollout_steps=algo.n_steps)

    monkeypatch.setattr(np.random, "randint", lambda *args, **kwargs: 1)
    flat_batch = next(algo.rollout_buffer.get_gpu_flat(algo.batch_size))
    legacy_q_targets = algo._normalize_flat_sequence_returns_like_reinforce(
        _recover_raw_from_value_space(
            flat_batch.returns.detach(),
            value_means=flat_batch.rollout_return_means.detach(),
            value_stds=flat_batch.rollout_return_stds.detach(),
        ),
        seq_start_indices=flat_batch.seq_start_indices,
        seq_lengths=flat_batch.seq_lengths,
        valid_mask_flat=flat_batch.objective_masks > 1e-8,
        eps=1e-6,
    )
    assert flat_batch.normalized_q_targets is not None
    assert torch.allclose(flat_batch.normalized_q_targets, legacy_q_targets, atol=1e-4, rtol=1e-5)
    assert torch.count_nonzero(flat_batch.normalized_q_targets[flat_batch.objective_masks <= 1e-8]).item() == 0
    vec_env.close()


def test_rollout_value_target_stats_match_full_context_raw_returns():
    rewards = np.asarray(
        [
            [1.0, 2.0],
            [3.0, 5.0],
            [7.0, 11.0],
        ],
        dtype=np.float32,
    )
    episode_starts = np.asarray(
        [
            [1.0, 1.0],
            [0.0, 0.0],
            [0.0, 0.0],
        ],
        dtype=np.float32,
    )
    raw_returns = _discounted_returns_from_rewards(
        rewards,
        episode_starts=episode_starts,
        dones=np.asarray([True, True], dtype=bool),
        gamma=0.5,
    )
    offsets, scales = _rollout_return_norm_stats_from_raw_returns(raw_returns, eps=1e-6)
    expected_raw_returns = np.asarray(
        [
            [4.25, 7.25],
            [6.5, 10.5],
            [7.0, 11.0],
        ],
        dtype=np.float32,
    )
    expected_offsets = np.asarray([5.9166665, 9.583333], dtype=np.float32)
    expected_scales = np.asarray([1.1960583, 1.6624948], dtype=np.float32)
    assert np.allclose(raw_returns, expected_raw_returns, atol=1e-6, rtol=1e-6)
    assert np.allclose(offsets, expected_offsets, atol=1e-6, rtol=1e-6)
    assert np.allclose(scales, expected_scales, atol=1e-6, rtol=1e-6)


def test_collect_rollouts_preserve_compute_returns_outputs_and_value_stats(monkeypatch):
    device = _cuda_or_skip()
    cfg, env_cfg = _build_small_ppo_config()
    env_cfg = _strict_env_cfg_for_ppo(env_cfg, aux=False)
    prior = EnvironmentPrior(env_cfg)
    fake_model = _FakeRWKVModel(
        num_features=cfg["prior"]["num_features"],
        x_obs_dim=cfg["transformer"]["x_obs_dim"],
        action_dim=cfg["transformer"]["x_action_dim"],
    ).to(device)
    algo, callback, vec_env = build_recurrent_ppo(
        model=fake_model,
        env_prior=prior,
        device=str(device),
        num_features=cfg["prior"]["num_features"],
        n_envs=cfg["optimizer"]["ppo_n_envs"],
        n_steps=cfg["optimizer"]["ppo_n_steps"],
        learning_rate=cfg["optimizer"]["learning_rate"],
        batch_size=cfg["optimizer"]["ppo_batch_size"],
        n_epochs=1,
        gamma=cfg["optimizer"]["ppo_gamma"],
        gae_lambda=cfg["optimizer"]["ppo_gae_lambda"],
        clip_range=cfg["optimizer"]["ppo_clip_range"],
        clip_range_vf=cfg["optimizer"]["ppo_clip_range_vf"],
        normalize_advantage=cfg["optimizer"]["ppo_normalize_advantage"],
        ent_coef=cfg["optimizer"]["ppo_ent_coef"],
        vf_coef=cfg["optimizer"]["ppo_vf_coef"],
        max_grad_norm=cfg["optimizer"]["ppo_max_grad_norm"],
        target_kl=cfg["optimizer"]["ppo_target_kl"],
    )
    algo.ep_info_buffer = []
    algo.ep_success_buffer = []
    algo._last_obs = vec_env.reset()
    algo._last_episode_starts = np.ones((vec_env.num_envs,), dtype=bool)
    algo._last_lstm_states = algo.policy._dummy_states(vec_env.num_envs)
    callback.init_callback(algo)

    def _fake_compute_returns_and_advantage(last_values, dones):
        del last_values, dones
        algo.rollout_buffer.advantages = np.full(
            (algo.rollout_buffer.buffer_size, algo.rollout_buffer.n_envs),
            8.0,
            dtype=np.float32,
        )
        algo.rollout_buffer.returns = np.full(
            (algo.rollout_buffer.buffer_size, algo.rollout_buffer.n_envs),
            11.0,
            dtype=np.float32,
        )
        algo.rollout_buffer.rollout_return_means.fill(7.0)
        algo.rollout_buffer.rollout_return_stds.fill(4.0)

    monkeypatch.setattr(algo.rollout_buffer, "compute_returns_and_advantage", _fake_compute_returns_and_advantage)

    assert algo.collect_rollouts(vec_env, callback, algo.rollout_buffer, n_rollout_steps=algo.n_steps)
    assert np.allclose(algo.rollout_buffer.advantages, 8.0)
    assert np.allclose(algo.rollout_buffer.returns, 11.0)
    assert np.allclose(algo.rollout_buffer.rollout_return_means, 7.0)
    assert np.allclose(algo.rollout_buffer.rollout_return_stds, 4.0)
    vec_env.close()


def test_neutral_value_affine_keeps_main_ppo_loss_identical_to_raw_bar_targets(monkeypatch):
    device = _cuda_or_skip()
    cfg, env_cfg = _build_small_ppo_config()
    env_cfg = _strict_env_cfg_for_ppo(env_cfg, aux=False)
    prior = EnvironmentPrior(env_cfg)
    fake_model = _FakeRWKVModel(
        num_features=cfg["prior"]["num_features"],
        x_obs_dim=cfg["transformer"]["x_obs_dim"],
        action_dim=cfg["transformer"]["x_action_dim"],
    ).to(device)
    algo, callback, vec_env = build_recurrent_ppo(
        model=fake_model,
        env_prior=prior,
        device=str(device),
        num_features=cfg["prior"]["num_features"],
        n_envs=cfg["optimizer"]["ppo_n_envs"],
        n_steps=cfg["optimizer"]["ppo_n_steps"],
        learning_rate=cfg["optimizer"]["learning_rate"],
        batch_size=cfg["optimizer"]["ppo_batch_size"],
        n_epochs=1,
        gamma=cfg["optimizer"]["ppo_gamma"],
        gae_lambda=cfg["optimizer"]["ppo_gae_lambda"],
        clip_range=cfg["optimizer"]["ppo_clip_range"],
        clip_range_vf=cfg["optimizer"]["ppo_clip_range_vf"],
        normalize_advantage=cfg["optimizer"]["ppo_normalize_advantage"],
        ent_coef=cfg["optimizer"]["ppo_ent_coef"],
        vf_coef=cfg["optimizer"]["ppo_vf_coef"],
        max_grad_norm=cfg["optimizer"]["ppo_max_grad_norm"],
        target_kl=cfg["optimizer"]["ppo_target_kl"],
    )
    algo.ep_info_buffer = []
    algo.ep_success_buffer = []
    algo._last_obs = vec_env.reset()
    algo._last_episode_starts = np.ones((vec_env.num_envs,), dtype=bool)
    algo._last_lstm_states = algo.policy._dummy_states(vec_env.num_envs)
    callback.init_callback(algo)
    assert algo.collect_rollouts(vec_env, callback, algo.rollout_buffer, n_rollout_steps=algo.n_steps)

    monkeypatch.setattr(np.random, "randint", lambda *args, **kwargs: 0)
    padded_batch = next(algo.rollout_buffer.get(algo.batch_size))
    new_loss = _ppo_main_minibatch_loss(algo, padded_batch)
    legacy_loss = _ppo_main_minibatch_loss_legacy_raw_value_targets(algo, padded_batch)
    assert torch.allclose(new_loss.detach(), legacy_loss.detach(), atol=1e-6, rtol=1e-6)
    vec_env.close()


def test_main_ppo_losses_ignore_prefix_steps_before_single_eval_pos(monkeypatch):
    device = _cuda_or_skip()
    cfg, env_cfg = _build_small_ppo_config()
    env_cfg = _strict_env_cfg_for_ppo(env_cfg, aux=False)
    prior = EnvironmentPrior(env_cfg)
    fake_model = _FakeRWKVModel(
        num_features=cfg["prior"]["num_features"],
        x_obs_dim=cfg["transformer"]["x_obs_dim"],
        action_dim=cfg["transformer"]["x_action_dim"],
    ).to(device)
    algo, callback, vec_env = build_recurrent_ppo(
        model=fake_model,
        env_prior=prior,
        device=str(device),
        num_features=cfg["prior"]["num_features"],
        n_envs=cfg["optimizer"]["ppo_n_envs"],
        n_steps=cfg["optimizer"]["ppo_n_steps"],
        learning_rate=cfg["optimizer"]["learning_rate"],
        batch_size=cfg["optimizer"]["ppo_batch_size"],
        n_epochs=1,
        gamma=cfg["optimizer"]["ppo_gamma"],
        gae_lambda=cfg["optimizer"]["ppo_gae_lambda"],
        clip_range=cfg["optimizer"]["ppo_clip_range"],
        clip_range_vf=cfg["optimizer"]["ppo_clip_range_vf"],
        normalize_advantage=cfg["optimizer"]["ppo_normalize_advantage"],
        ent_coef=cfg["optimizer"]["ppo_ent_coef"],
        vf_coef=cfg["optimizer"]["ppo_vf_coef"],
        max_grad_norm=cfg["optimizer"]["ppo_max_grad_norm"],
        target_kl=cfg["optimizer"]["ppo_target_kl"],
    )
    algo.ep_info_buffer = []
    algo.ep_success_buffer = []
    algo._last_obs = vec_env.reset()
    algo._last_episode_starts = np.ones((vec_env.num_envs,), dtype=bool)
    algo._last_lstm_states = algo.policy._dummy_states(vec_env.num_envs)
    callback.init_callback(algo)
    monkeypatch.setattr(prior, "_sample_single_eval_pos", lambda n_samples, single_eval_pos: 2)
    assert algo.collect_rollouts(vec_env, callback, algo.rollout_buffer, n_rollout_steps=algo.n_steps)

    monkeypatch.setattr(np.random, "randint", lambda *args, **kwargs: 0)
    padded_batch = next(algo.rollout_buffer.get(algo.batch_size))
    flat_batch = next(algo.rollout_buffer.get_gpu_flat(algo.batch_size))
    padded_prefix = padded_batch.objective_masks <= 1e-8
    flat_prefix = flat_batch.objective_masks <= 1e-8

    padded_loss = _ppo_main_minibatch_loss(algo, padded_batch)
    flat_loss = _ppo_main_streamed_flat_minibatch_loss(algo, flat_batch)

    padded_perturbed = padded_batch._replace(
        returns=torch.where(padded_prefix, padded_batch.returns + 1000.0, padded_batch.returns),
        advantages=torch.where(padded_prefix, padded_batch.advantages - 500.0, padded_batch.advantages),
        old_log_prob=torch.where(padded_prefix, padded_batch.old_log_prob + 25.0, padded_batch.old_log_prob),
        old_values=torch.where(padded_prefix, padded_batch.old_values - 200.0, padded_batch.old_values),
    )
    flat_perturbed = flat_batch._replace(
        returns=torch.where(flat_prefix, flat_batch.returns + 1000.0, flat_batch.returns),
        advantages=torch.where(flat_prefix, flat_batch.advantages - 500.0, flat_batch.advantages),
        old_log_prob=torch.where(flat_prefix, flat_batch.old_log_prob + 25.0, flat_batch.old_log_prob),
        old_values=torch.where(flat_prefix, flat_batch.old_values - 200.0, flat_batch.old_values),
    )

    padded_loss_perturbed = _ppo_main_minibatch_loss(algo, padded_perturbed)
    flat_loss_perturbed = _ppo_main_streamed_flat_minibatch_loss(algo, flat_perturbed)

    assert torch.allclose(padded_loss.detach(), padded_loss_perturbed.detach(), atol=1e-6, rtol=1e-6)
    assert torch.allclose(flat_loss.detach(), flat_loss_perturbed.detach(), atol=1e-6, rtol=1e-6)
    vec_env.close()


def test_q_aux_losses_ignore_prefix_steps_before_single_eval_pos(monkeypatch):
    device = _cuda_or_skip()
    cfg, env_cfg = _build_small_ppo_config()
    env_cfg = _strict_env_cfg_for_ppo(env_cfg, aux=True)
    env_cfg["next_state_flow_matching_weight"] = 0.0
    prior = EnvironmentPrior(env_cfg)
    fake_model = _FakeRWKVModel(
        num_features=cfg["prior"]["num_features"],
        x_obs_dim=cfg["transformer"]["x_obs_dim"],
        action_dim=cfg["transformer"]["x_action_dim"],
        normalized_q_head=True,
    ).to(device)
    algo, callback, vec_env = build_recurrent_ppo(
        model=fake_model,
        env_prior=prior,
        device=str(device),
        num_features=cfg["prior"]["num_features"],
        n_envs=cfg["optimizer"]["ppo_n_envs"],
        n_steps=cfg["optimizer"]["ppo_n_steps"],
        learning_rate=cfg["optimizer"]["learning_rate"],
        batch_size=cfg["optimizer"]["ppo_batch_size"],
        n_epochs=1,
        gamma=cfg["optimizer"]["ppo_gamma"],
        gae_lambda=cfg["optimizer"]["ppo_gae_lambda"],
        clip_range=cfg["optimizer"]["ppo_clip_range"],
        clip_range_vf=cfg["optimizer"]["ppo_clip_range_vf"],
        normalize_advantage=cfg["optimizer"]["ppo_normalize_advantage"],
        ent_coef=cfg["optimizer"]["ppo_ent_coef"],
        vf_coef=cfg["optimizer"]["ppo_vf_coef"],
        max_grad_norm=cfg["optimizer"]["ppo_max_grad_norm"],
        target_kl=cfg["optimizer"]["ppo_target_kl"],
    )
    algo.ep_info_buffer = []
    algo.ep_success_buffer = []
    algo._last_obs = vec_env.reset()
    algo._last_episode_starts = np.ones((vec_env.num_envs,), dtype=bool)
    algo._last_lstm_states = algo.policy._dummy_states(vec_env.num_envs)
    callback.init_callback(algo)
    monkeypatch.setattr(prior, "_sample_single_eval_pos", lambda n_samples, single_eval_pos: 2)
    assert algo.collect_rollouts(vec_env, callback, algo.rollout_buffer, n_rollout_steps=algo.n_steps)

    monkeypatch.setattr(np.random, "randint", lambda *args, **kwargs: 0)
    padded_batch = next(algo.rollout_buffer.get(algo.batch_size))
    flat_batch = next(algo.rollout_buffer.get_gpu_flat(algo.batch_size))

    padded_outputs = algo.policy.evaluate_actions_with_hidden(
        padded_batch.observations,
        padded_batch.actions,
        padded_batch.lstm_states,
        padded_batch.episode_starts,
        action_masks=padded_batch.action_masks,
    )
    flat_outputs = algo.policy.evaluate_actions_with_hidden_flat(
        flat_batch.observations,
        flat_batch.actions,
        seq_lengths=flat_batch.seq_lengths,
        action_masks=flat_batch.action_masks,
        include_normalized_q_logits=True,
    )
    padded_objective = (padded_batch.mask > 1e-8) & (padded_batch.objective_masks > 1e-8)
    flat_objective = flat_batch.objective_masks > 1e-8
    padded_q_targets = padded_batch.normalized_q_targets
    if padded_q_targets is None:
        padded_q_targets = algo._normalize_masked_returns_like_reinforce(
            _recover_raw_from_value_space(
                padded_batch.returns.detach(),
                value_means=padded_batch.rollout_return_means.detach(),
                value_stds=padded_batch.rollout_return_stds.detach(),
            ),
            padded_objective,
            n_seq=int(padded_batch.lstm_states.pi[0].shape[1]),
            eps=1e-6,
        )
    flat_q_targets = flat_batch.normalized_q_targets
    if flat_q_targets is None:
        flat_q_targets = algo._normalize_flat_sequence_returns_like_reinforce(
            _recover_raw_from_value_space(
                flat_batch.returns.detach(),
                value_means=flat_batch.rollout_return_means.detach(),
                value_stds=flat_batch.rollout_return_stds.detach(),
            ),
            seq_start_indices=flat_batch.seq_start_indices,
            seq_lengths=flat_batch.seq_lengths,
            valid_mask_flat=flat_objective,
            eps=1e-6,
        )
    padded_prefix = (padded_batch.mask > 1e-8) & (~padded_objective)
    flat_prefix = ~flat_objective

    padded_aux_loss, padded_aux_stats = algo._compute_aux_losses(
        padded_batch,
        eval_outputs=padded_outputs,
        n_seq=int(padded_batch.lstm_states.pi[0].shape[1]),
        normalized_q_targets=padded_q_targets,
        objective_mask=padded_objective,
        q_target_denom=padded_objective.sum().clamp_min(1),
    )
    flat_aux_loss, flat_aux_stats = algo._compute_aux_losses_flat(
        hidden=flat_outputs["hidden"],
        actions=flat_outputs["actions"],
        seq_lengths=flat_batch.seq_lengths,
        normalized_q_logits=flat_outputs["normalized_q_logits"],
        normalized_q_targets=flat_q_targets,
        objective_mask=flat_objective,
        q_target_denom=flat_objective.sum().clamp_min(1),
    )

    padded_aux_loss_perturbed, padded_aux_stats_perturbed = algo._compute_aux_losses(
        padded_batch,
        eval_outputs=padded_outputs,
        n_seq=int(padded_batch.lstm_states.pi[0].shape[1]),
        normalized_q_targets=torch.where(
            padded_prefix,
            padded_q_targets + 1000.0,
            padded_q_targets,
        ),
        objective_mask=padded_objective,
        q_target_denom=padded_objective.sum().clamp_min(1),
    )
    flat_aux_loss_perturbed, flat_aux_stats_perturbed = algo._compute_aux_losses_flat(
        hidden=flat_outputs["hidden"],
        actions=flat_outputs["actions"],
        seq_lengths=flat_batch.seq_lengths,
        normalized_q_logits=flat_outputs["normalized_q_logits"],
        normalized_q_targets=torch.where(
            flat_prefix,
            flat_q_targets + 1000.0,
            flat_q_targets,
        ),
        objective_mask=flat_objective,
        q_target_denom=flat_objective.sum().clamp_min(1),
    )

    assert torch.allclose(padded_aux_loss.detach(), padded_aux_loss_perturbed.detach(), atol=1e-6, rtol=1e-6)
    assert torch.allclose(flat_aux_loss.detach(), flat_aux_loss_perturbed.detach(), atol=1e-6, rtol=1e-6)
    assert torch.allclose(
        padded_aux_stats["normalized_q_value_loss"],
        padded_aux_stats_perturbed["normalized_q_value_loss"],
        atol=1e-6,
        rtol=1e-6,
    )
    assert torch.allclose(
        flat_aux_stats["normalized_q_value_loss"],
        flat_aux_stats_perturbed["normalized_q_value_loss"],
        atol=1e-6,
        rtol=1e-6,
    )
    vec_env.close()


def test_train_uses_cached_flat_q_targets_without_recomputing_legacy_helper(monkeypatch):
    device = _cuda_or_skip()
    cfg, env_cfg = _build_small_ppo_config()
    env_cfg = _strict_env_cfg_for_ppo(env_cfg, aux=True)
    env_cfg["next_state_flow_matching_weight"] = 0.0
    fake_model = _FakeRWKVModel(
        num_features=cfg["prior"]["num_features"],
        x_obs_dim=cfg["transformer"]["x_obs_dim"],
        action_dim=cfg["transformer"]["x_action_dim"],
        normalized_q_head=True,
    ).to(device)
    algo, callback, vec_env = build_recurrent_ppo(
        model=fake_model,
        env_prior=EnvironmentPrior(env_cfg),
        device=str(device),
        num_features=cfg["prior"]["num_features"],
        n_envs=cfg["optimizer"]["ppo_n_envs"],
        n_steps=cfg["optimizer"]["ppo_n_steps"],
        learning_rate=cfg["optimizer"]["learning_rate"],
        batch_size=cfg["optimizer"]["ppo_batch_size"],
        n_epochs=1,
        gamma=cfg["optimizer"]["ppo_gamma"],
        gae_lambda=cfg["optimizer"]["ppo_gae_lambda"],
        clip_range=cfg["optimizer"]["ppo_clip_range"],
        clip_range_vf=cfg["optimizer"]["ppo_clip_range_vf"],
        normalize_advantage=cfg["optimizer"]["ppo_normalize_advantage"],
        ent_coef=cfg["optimizer"]["ppo_ent_coef"],
        vf_coef=cfg["optimizer"]["ppo_vf_coef"],
        max_grad_norm=cfg["optimizer"]["ppo_max_grad_norm"],
        target_kl=cfg["optimizer"]["ppo_target_kl"],
    )
    algo.ep_info_buffer = []
    algo.ep_success_buffer = []
    algo._last_obs = vec_env.reset()
    algo._last_episode_starts = np.ones((vec_env.num_envs,), dtype=bool)
    algo._last_lstm_states = algo.policy._dummy_states(vec_env.num_envs)
    callback.init_callback(algo)
    assert algo.collect_rollouts(vec_env, callback, algo.rollout_buffer, n_rollout_steps=algo.n_steps)

    calls = 0
    original = algo._normalize_flat_sequence_returns_like_reinforce

    def _wrapped(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(np.random, "randint", lambda *args, **kwargs: 1)
    monkeypatch.setattr(algo, "_normalize_flat_sequence_returns_like_reinforce", _wrapped)
    algo._logger = SimpleNamespace(record=lambda *args, **kwargs: None)
    algo.train()
    assert calls == 0
    vec_env.close()


def test_train_epoch_official_recurrent_ppo_exports_aux_reward_and_weight_metrics():
    class _FakeLogger:
        def __init__(self, values):
            self.name_to_value = dict(values)

    class _FakeAlgo:
        def __init__(self):
            self.env = object()
            self.rollout_buffer = object()
            self.n_steps = 4
            self.num_timesteps = 8
            self.logger = _FakeLogger(
                {
                    "train/loss": 1.25,
                    "train/policy_gradient_loss": 0.75,
                    "train/value_loss": 0.40,
                    "train/entropy_loss": -0.05,
                    "train/normalized_q_value_loss": 0.30,
                    "train/next_state_flow_matching_loss": 0.20,
                    "train/approx_kl": 0.01,
                    "train/clip_fraction": 0.15,
                    "train/explained_variance": 0.5,
                    "train/clip_range": 0.2,
                    "train/clip_range_vf": 0.1,
                    "train/n_updates": 4,
                    "train/update_wall_time_sec": 12.0,
                    "train/outer_batches": 3,
                    "train/subbatches": 7,
                }
            )
            self.ep_info_buffer = [{"r": 1.6, "l": 12}]
            self.ent_coef = 0.0
            self.vf_coef = 0.5
            self._rwkv_aux_q_weight = 1.0
            self._rwkv_aux_flow_weight = 1.0
            self._rwkv_last_reward_component_stats = {
                "reward_mean": 0.35,
                "reward_std": 0.18,
                "reward_return_mean": 1.40,
                "reward_env_mean": 0.40,
                "reward_env_std": 0.15,
                "reward_env_return_mean": 1.60,
                "reward_ctrl_mean": -0.05,
                "reward_ctrl_std": 0.02,
                "reward_ctrl_return_mean": -0.20,
                "reward_survival_mean": 0.03,
                "reward_survival_std": 0.01,
                "reward_survival_return_mean": 0.12,
                "reward_terminal_bonus_mean": 0.02,
                "reward_terminal_bonus_std": 0.05,
                "reward_terminal_bonus_return_mean": 0.08,
                "ep_rew_mean": 1.40,
                "ep_len_mean": 6.0,
                "full_ep_rew_mean": 1.60,
                "full_ep_len_mean": 12.0,
            }

        def collect_rollouts(self, env, callback, rollout_buffer, n_rollout_steps):
            del env, callback, rollout_buffer, n_rollout_steps
            return True

        def _update_current_progress_remaining(self, num_timesteps, total_timesteps_target):
            del num_timesteps, total_timesteps_target
            return None

        def dump_logs(self, epoch_idx):
            del epoch_idx
            return None

        def train(self):
            return None

    model = SimpleNamespace(last_pg_epoch_metrics={"stale": True}, last_ppo_epoch_metrics=None)
    algo = _FakeAlgo()
    loss, nan_share, ignore_share = train_epoch_official_recurrent_ppo(
        model=model,
        ppo_algo=algo,
        ppo_callback=object(),
        ppo_total_timesteps_target=32,
        epoch_idx=1,
    )

    assert loss == pytest.approx(1.25)
    assert nan_share == pytest.approx(0.0)
    assert ignore_share == pytest.approx(0.0)
    assert model.last_pg_epoch_metrics is None
    ppo_diag = model.last_ppo_epoch_metrics
    assert isinstance(ppo_diag, dict)
    assert ppo_diag["normalized_q_value_loss_mean"] == pytest.approx(0.30)
    assert ppo_diag["next_state_flow_matching_loss_mean"] == pytest.approx(0.20)
    assert ppo_diag["reward_mean"] == pytest.approx(0.35)
    assert ppo_diag["reward_return_mean"] == pytest.approx(1.40)
    assert ppo_diag["reward_env_mean"] == pytest.approx(0.40)
    assert ppo_diag["reward_env_std"] == pytest.approx(0.15)
    assert ppo_diag["reward_env_return_mean"] == pytest.approx(1.60)
    assert ppo_diag["reward_ctrl_mean"] == pytest.approx(-0.05)
    assert ppo_diag["reward_ctrl_std"] == pytest.approx(0.02)
    assert ppo_diag["reward_ctrl_return_mean"] == pytest.approx(-0.20)
    assert ppo_diag["reward_survival_mean"] == pytest.approx(0.03)
    assert ppo_diag["reward_terminal_bonus_return_mean"] == pytest.approx(0.08)
    assert ppo_diag["policy_loss_weight"] == pytest.approx(1.0)
    assert ppo_diag["entropy_loss_weight"] == pytest.approx(0.0)
    assert ppo_diag["value_loss_weight"] == pytest.approx(0.5)
    assert ppo_diag["normalized_q_value_weight"] == pytest.approx(1.0)
    assert ppo_diag["next_state_flow_matching_weight"] == pytest.approx(1.0)
    assert ppo_diag["episode_reward_mean"] == pytest.approx(1.40)
    assert ppo_diag["episode_length_mean"] == pytest.approx(6.0)
    assert ppo_diag["full_episode_reward_mean"] == pytest.approx(1.60)
    assert ppo_diag["full_episode_length_mean"] == pytest.approx(12.0)
    ppo_logger_metrics = model.last_ppo_logger_metrics
    assert isinstance(ppo_logger_metrics, dict)
    assert ppo_logger_metrics["train/loss"] == pytest.approx(1.25)
    assert ppo_logger_metrics["rollout/ep_rew_mean"] == pytest.approx(1.40)
    assert ppo_logger_metrics["rollout/ep_len_mean"] == pytest.approx(6.0)
    assert ppo_logger_metrics["rollout/full_ep_rew_mean"] == pytest.approx(1.60)
    assert ppo_logger_metrics["rollout/full_ep_len_mean"] == pytest.approx(12.0)
    assert ppo_logger_metrics["rollout/reward_env_mean"] == pytest.approx(0.40)
    assert ppo_logger_metrics["rollout/reward_ctrl_return_mean"] == pytest.approx(-0.20)
    assert ppo_logger_metrics["time/total_timesteps"] == 8


def test_make_training_callback_writes_ppo_diag_line(tmp_path):
    callback = make_training_callback(
        save_every=10,
        model_string="ppo_diag_test",
        base_path=str(tmp_path),
        report=None,
        config={"model_type": "rlpfn"},
        use_mlflow=False,
        checkpoint_dir=str(tmp_path / "ckpt"),
        classification=False,
        validate=False,
    )

    model = SimpleNamespace(
        losses=[1.25],
        learning_rates=[3e-5],
        wallclock_times=[30.0],
        last_pg_epoch_metrics=None,
        last_ppo_epoch_metrics={
            "policy_total_loss_mean": 1.25,
            "normalized_q_value_loss_mean": 0.30,
            "next_state_flow_matching_loss_mean": 0.20,
            "reward_env_mean": 0.40,
            "reward_ctrl_mean": -0.05,
            "policy_loss_weight": 1.0,
            "entropy_loss_weight": 0.0,
            "value_loss_weight": 0.5,
            "normalized_q_value_weight": 1.0,
            "next_state_flow_matching_weight": 1.0,
        },
        last_ppo_logger_metrics={
            "rollout/ep_len_mean": 165.0,
            "rollout/ep_rew_mean": 0.342,
            "time/fps": 1507,
            "time/iterations": 11,
            "time/time_elapsed": 30601,
            "time/total_timesteps": 46137344,
            "train/loss": 7.5,
            "train/std": 1.0,
            "train/update_wall_time_sec": 2.52e3,
            "train/outer_batches": 128,
            "train/subbatches": 1244,
        },
    )

    callback(model, optimizer=None, scheduler=None, epoch=1)

    log_path = tmp_path / "log" / "ppo_diag_test.log"
    log_text = log_path.read_text()
    assert "Epoch 1 ppo_diag " in log_text
    assert "normalized_q_value_loss_mean=0.3" in log_text
    assert "next_state_flow_matching_loss_mean=0.2" in log_text
    assert "reward_env_mean=0.4" in log_text
    assert "reward_ctrl_mean=-0.05" in log_text
    assert "value_loss_weight=0.5" in log_text
    assert "Epoch 1 ppo_metrics" in log_text
    assert "| rollout/" in log_text
    assert "ep_len_mean" in log_text
    assert "ep_rew_mean" in log_text
    assert "| time/" in log_text
    assert "fps" in log_text
    assert "| train/" in log_text
    assert "update_wall_time_sec" in log_text


def test_ppo_progress_logs_print_without_appending_log_file(tmp_path, monkeypatch):
    class _FakeProgressAlgo:
        def _progress_logging_enabled(self):
            return True

    fake_algo = _FakeProgressAlgo()
    stdout = io.StringIO()
    log_path = tmp_path / "ppo_progress.log"
    fake_algo._rwkv_progress_log_file = str(log_path)

    monkeypatch.setattr("sys.stdout", stdout)
    MaskedRecurrentPPO._emit_progress_log(fake_algo, "[ppo-train-progress] epoch=1/4 outer_batch=1/32")

    printed = stdout.getvalue()
    assert "[ppo-train-progress] epoch=1/4 outer_batch=1/32" in printed
    assert not log_path.exists()


def test_train_epoch_official_recurrent_ppo_tolerates_missing_optional_diag_fields():
    class _FakeLogger:
        def __init__(self, values):
            self.name_to_value = dict(values)

    class _SparseAlgo:
        def __init__(self):
            self.env = object()
            self.rollout_buffer = object()
            self.n_steps = 4
            self.num_timesteps = 8
            self.logger = _FakeLogger({"train/loss": 0.9})
            self.ep_info_buffer = []

        def collect_rollouts(self, env, callback, rollout_buffer, n_rollout_steps):
            del env, callback, rollout_buffer, n_rollout_steps
            return True

        def _update_current_progress_remaining(self, num_timesteps, total_timesteps_target):
            del num_timesteps, total_timesteps_target
            return None

        def dump_logs(self, epoch_idx):
            del epoch_idx
            return None

        def train(self):
            return None

    model = SimpleNamespace(last_pg_epoch_metrics={"stale": True}, last_ppo_epoch_metrics=None)
    loss, nan_share, ignore_share = train_epoch_official_recurrent_ppo(
        model=model,
        ppo_algo=_SparseAlgo(),
        ppo_callback=object(),
        ppo_total_timesteps_target=32,
        epoch_idx=1,
    )

    assert loss == pytest.approx(0.9)
    assert nan_share == pytest.approx(0.0)
    assert ignore_share == pytest.approx(0.0)
    assert model.last_pg_epoch_metrics is None
    ppo_diag = model.last_ppo_epoch_metrics
    assert isinstance(ppo_diag, dict)
    assert ppo_diag["policy_total_loss_mean"] == pytest.approx(0.9)
    assert ppo_diag["policy_gradient_loss_mean"] is None
    assert ppo_diag["normalized_q_value_loss_mean"] is None
    assert ppo_diag["next_state_flow_matching_loss_mean"] is None
    assert ppo_diag["reward_env_mean"] is None
    assert ppo_diag["reward_ctrl_mean"] is None
    assert ppo_diag["policy_loss_weight"] == pytest.approx(1.0)
    assert ppo_diag["entropy_loss_weight"] == pytest.approx(0.0)
    assert ppo_diag["value_loss_weight"] == pytest.approx(0.0)
    assert ppo_diag["normalized_q_value_weight"] == pytest.approx(0.0)
    assert ppo_diag["next_state_flow_matching_weight"] == pytest.approx(0.0)
    ppo_logger_metrics = model.last_ppo_logger_metrics
    assert isinstance(ppo_logger_metrics, dict)
    assert ppo_logger_metrics["train/loss"] == pytest.approx(0.9)
    assert ppo_logger_metrics["time/total_timesteps"] == 8
