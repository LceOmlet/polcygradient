from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from gymnasium import spaces

from ticl.model_configs import get_model_default_config
from ticl.models.tabpfn_bar_distribution import make_standardized_full_support_bar_distribution
from ticl.priors.environment_prior import EnvironmentPrior
from ticl.sb3_recurrent_ppo import (
    MaskedRecurrentFlatBatchSamples,
    MaskedRecurrentRolloutBufferSamples,
    OfficialRWKVRecurrentPPOPolicy,
    RNNStates,
    _slice_flat_sequence_batch_to_padded,
    _slice_masked_rollout_sequence_batch,
    _slice_padded_sequence_tensor,
    build_recurrent_ppo,
    resolve_official_ppo_update_batch_size,
)
from ticl.train import _resolve_official_ppo_rollout_shape


def _cuda_or_skip():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for strict official RWKV PPO tests.")
    return torch.device("cuda")


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
    mask = rollout_data.mask > 1e-8
    valid_total = mask.sum().clamp_min(1).to(device=rollout_data.returns.device, dtype=rollout_data.returns.dtype)
    advantages = rollout_data.advantages
    if algo.normalize_advantage:
        advantages = (advantages - advantages[mask].mean()) / (advantages[mask].std() + 1e-8)
    eval_outputs = algo.policy.evaluate_actions_with_hidden(
        rollout_data.observations,
        rollout_data.actions,
        rollout_data.lstm_states,
        rollout_data.episode_starts,
        action_masks=rollout_data.action_masks,
    )
    values = eval_outputs["values"].flatten()
    log_prob = eval_outputs["log_prob"]
    entropy = eval_outputs["entropy"]
    ratio = torch.exp(log_prob - rollout_data.old_log_prob)
    clipped_objective = torch.min(
        advantages * ratio,
        advantages * torch.clamp(ratio, 1 - algo.clip_range(1.0), 1 + algo.clip_range(1.0)),
    )
    mask_f = mask.to(device=clipped_objective.device, dtype=clipped_objective.dtype)
    policy_loss = -((clipped_objective * mask_f).sum() / valid_total.to(device=clipped_objective.device, dtype=clipped_objective.dtype))
    value_errors = (rollout_data.returns - values) ** 2
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
    valid_total = flat_batch.returns.new_tensor(float(max(1, int(flat_batch.returns.numel()))))
    advantages = flat_batch.advantages
    if algo.normalize_advantage:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
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
        sub_old_values = flat_batch.old_values[flat_start:flat_end]
        sub_returns = flat_batch.returns[flat_start:flat_end]
        eval_outputs = algo.policy.evaluate_actions_with_hidden_flat(
            flat_batch.observations[flat_start:flat_end],
            sub_actions,
            seq_lengths=sub_seq_lengths,
            action_masks=flat_batch.action_masks[flat_start:flat_end],
        )
        values = eval_outputs["values"].flatten()
        log_prob = eval_outputs["log_prob"]
        entropy = eval_outputs["entropy"]
        ratio = torch.exp(log_prob - sub_old_log_prob)
        clipped_objective = torch.min(
            sub_advantages * ratio,
            sub_advantages * torch.clamp(ratio, 1 - algo.clip_range(1.0), 1 + algo.clip_range(1.0)),
        )
        policy_loss = -(clipped_objective.sum() / valid_total.to(device=clipped_objective.device, dtype=clipped_objective.dtype))
        value_errors = (sub_returns - values) ** 2
        value_loss = value_errors.sum() / valid_total.to(device=value_errors.device, dtype=value_errors.dtype)
        entropy_terms = entropy if entropy is not None else -log_prob
        entropy_loss = -(entropy_terms.sum() / valid_total.to(device=entropy_terms.device, dtype=entropy_terms.dtype))
        total_loss = total_loss + (policy_loss + algo.ent_coef * entropy_loss + algo.vf_coef * value_loss)
    return total_loss


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


def test_policy_evaluate_actions_uses_sequence_path_not_step_path():
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
    assert fake_model.rwkv_core.sequence_calls == 1
    assert fake_model.replay_chunk_requests == [(3, 2)]
    assert fake_model.encode_sequence_calls == 1
    assert fake_model.encode_sequence_shapes == [(3, 2, num_features)]


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

    def _wrapped_eval(obs_steps, episode_starts_steps):
        call_shapes.append(tuple(obs_steps.shape))
        if int(obs_steps.shape[0]) > 1:
            raise AssertionError("collect_rollouts should not recompute full rollout values from obs_steps")
        return original_eval(obs_steps, episode_starts_steps)

    monkeypatch.setattr(algo.policy, "evaluate_rollout_values", _wrapped_eval)
    assert algo.collect_rollouts(vec_env, callback, algo.rollout_buffer, n_rollout_steps=algo.n_steps)
    assert call_shapes == [(1, cfg["optimizer"]["ppo_n_envs"], cfg["prior"]["num_features"])]
    assert prior.last_rollout_ppo_trace["streamed_to_sink"] is True
    assert prior.last_rollout_ppo_trace["obs"] is None
    assert prior.last_rollout_ppo_trace["values"] is None
    assert prior.last_rollout_reinforce is None
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
    mask = padded_batch.mask > 1e-8
    if algo.normalize_advantage:
        advantages = (advantages - advantages[mask].mean()) / (advantages[mask].std() + 1e-8)
    valid_total = mask.sum().clamp_min(1).to(device=padded_batch.returns.device, dtype=padded_batch.returns.dtype)
    for start_seq in range(0, n_seq_total, seq_subbatch_size):
        end_seq = min(n_seq_total, start_seq + seq_subbatch_size)
        sub_rollout_data = _slice_masked_rollout_sequence_batch(
            padded_batch,
            start_seq=start_seq,
            end_seq=end_seq,
        )
        sub_mask = sub_rollout_data.mask > 1e-8
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
        values = eval_outputs["values"].flatten()
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
        value_errors = (sub_rollout_data.returns - values) ** 2
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
    flat_outputs = algo.policy.evaluate_actions_with_hidden_flat(
        flat_batch.observations,
        flat_batch.actions,
        seq_lengths=flat_batch.seq_lengths,
        action_masks=flat_batch.action_masks,
    )
    padded_q_targets = algo._normalize_masked_returns_like_reinforce(
        padded_batch.returns.detach(),
        padded_batch.mask > 1e-8,
        n_seq=int(padded_batch.lstm_states.pi[0].shape[1]),
        eps=1e-6,
    )
    flat_q_targets = algo._normalize_flat_sequence_returns_like_reinforce(
        flat_batch.returns.detach(),
        seq_start_indices=flat_batch.seq_start_indices,
        seq_lengths=flat_batch.seq_lengths,
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
        q_target_denom=(padded_batch.mask > 1e-8).sum().clamp_min(1),
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
        q_target_denom=flat_batch.returns.new_tensor(float(max(1, int(flat_batch.returns.numel())))),
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
        flat_batch.returns.detach(),
        seq_start_indices=flat_batch.seq_start_indices,
        seq_lengths=flat_batch.seq_lengths,
        eps=1e-6,
    )
    assert flat_batch.normalized_q_targets is not None
    assert torch.allclose(flat_batch.normalized_q_targets, legacy_q_targets, atol=1e-4, rtol=1e-5)
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
