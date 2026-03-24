from copy import deepcopy
from collections import OrderedDict
import os
import random
import sys
import types

import numpy as np
import pytest
import torch

import ticl.train as train_mod
from ticl.model_builder import get_model
from ticl.model_configs import get_model_default_config
from ticl.models.tabpfn_bar_distribution import make_standardized_full_support_bar_distribution
from ticl.models.rwkv7_pfn import _load_official_rwkv7_demo_rnn
from ticl.priors.environment_prior import EnvironmentPrior
from ticl.rl_validation import evaluate_rlpfn_on_gym_envs
from ticl.train import _build_policy_step_fn, _compute_policy_rollout_chunk_loss


def _seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _ensure_torch_extensions_dir():
    torch_ext_dir = "/tmp/ticl_torch_extensions"
    os.makedirs(torch_ext_dir, exist_ok=True)
    os.environ.setdefault("TORCH_EXTENSIONS_DIR", torch_ext_dir)


def _build_rwkv7_rlpfn_config():
    cfg = deepcopy(get_model_default_config("rlpfn"))
    cfg["transformer"]["backbone"] = "rwkv7"
    cfg["optimizer"]["rl_objective"] = "reinforce"
    return cfg


def _build_small_exact_scm_env_cfg():
    cfg = _build_rwkv7_rlpfn_config()
    env_cfg = deepcopy(cfg["prior"]["environment"])
    env_cfg.update(
        {
            "state_dim": {"distribution": "uniform_int", "min": 8, "max": 8},
            "obs_dim": {"distribution": "uniform_int", "min": 6, "max": 6},
            "action_dim": {"distribution": "uniform_int", "min": 3, "max": 3},
            "noise_dim": {"distribution": "uniform_int", "min": 2, "max": 2},
            "zero_pad_dim": {"distribution": "uniform_int", "min": 0, "max": 0},
            "reward_dropout_randomize": False,
            "reward_dropout_ratio_min": 0.0,
            "reward_dropout_ratio_max": 0.0,
            "action_noise_train_std": 0.05,
            "action_noise_eval_std": 0.03,
            "reinforce_sequence_replay_enabled": False,
            "normalized_q_value_weight": 0.0,
            "next_state_flow_matching_weight": 0.0,
        }
    )
    return cfg, env_cfg


def _run_rwkv7_exact_scm_chunk(
    *,
    device="cpu",
    policy_rollout_checkpoint=False,
    policy_rollout_checkpoint_reentrant=True,
    kv_cache_mode="immutable",
    allow_grad_mutable_cache=False,
    n_samples=8,
    single_eval_pos=4,
    pg_tbptt_window=None,
    tbptt_loss_sink=None,
    reinforce_sequence_replay_enabled=False,
    rwkv_sequence_replay_checkpoint=None,
    rwkv_sequence_replay_batch_chunk_size=None,
    rwkv_sequence_replay_token_budget=None,
    use_reinforce_replay_loss_sink=False,
    collect_x_override=None,
):
    _ensure_torch_extensions_dir()
    _seed_everything(123)
    cfg, env_cfg = _build_small_exact_scm_env_cfg()
    env_cfg["reinforce_sequence_replay_enabled"] = bool(reinforce_sequence_replay_enabled)
    cfg["transformer"]["emsize"] = 64
    cfg["transformer"]["nlayers"] = 2
    cfg["transformer"]["rwkv_head_size"] = 64
    if rwkv_sequence_replay_checkpoint is not None:
        cfg["transformer"]["rwkv_sequence_replay_checkpoint"] = bool(rwkv_sequence_replay_checkpoint)
    if rwkv_sequence_replay_batch_chunk_size is not None:
        cfg["transformer"]["rwkv_sequence_replay_batch_chunk_size"] = rwkv_sequence_replay_batch_chunk_size
    if rwkv_sequence_replay_token_budget is not None:
        cfg["transformer"]["rwkv_sequence_replay_token_budget"] = rwkv_sequence_replay_token_budget
    cfg["optimizer"]["rl_objective"] = "reinforce"

    prior = EnvironmentPrior(env_cfg)
    _, model, *_ = get_model(cfg, device=device, should_train=False, verbose=False)
    if str(device) == "cuda":
        model = model.cuda()
    model.train()
    step_fn = _build_policy_step_fn(
        model,
        num_features=int(cfg["prior"]["num_features"]),
        max_cache_len=int(n_samples),
        kv_cache_mode=kv_cache_mode,
        kv_cache_page_size=None,
        allow_grad_mutable_cache=bool(allow_grad_mutable_cache),
        pg_torch_compile=False,
    )

    _seed_everything(515151)
    h_list = prior._sample_batch_hypers(2)
    for h in h_list:
        h["reward_dropout_enabled"] = False
        h["reward_dropout_randomize"] = False
        h["reward_dropout_ratio"] = 0.0
        h["action_noise_train_std"] = 0.05
        h["action_noise_eval_std"] = 0.03

    model.zero_grad(set_to_none=True)
    sink_call_count = {"n": 0}

    def _replay_loss_sink(loss_root):
        sink_call_count["n"] += 1
        loss_root.backward()

    loss, _, stats = _compute_policy_rollout_chunk_loss(
        env_prior=prior,
        policy_step_fn=step_fn,
        batch_size=2,
        n_samples=int(n_samples),
        num_features=int(cfg["prior"]["num_features"]),
        device=device,
        single_eval_pos=int(single_eval_pos),
        collect_x=(
            bool(use_reinforce_replay_loss_sink)
            if collect_x_override is None
            else bool(collect_x_override)
        ),
        policy_rollout_checkpoint=bool(policy_rollout_checkpoint),
        policy_rollout_checkpoint_reentrant=bool(policy_rollout_checkpoint_reentrant),
        pg_saved_tensors_cpu_offload=False,
        pg_saved_tensors_pin_memory=True,
        pg_tbptt_window=pg_tbptt_window,
        tbptt_loss_sink=tbptt_loss_sink,
        reinforce_replay_loss_sink=(_replay_loss_sink if bool(use_reinforce_replay_loss_sink) else None),
        h_list_override=[dict(h) for h in h_list],
        env_seeds_override=[17, 29],
        rollout_seeds_override=[101, 211],
        rl_objective="reinforce",
    )
    if tbptt_loss_sink is None and bool(loss.requires_grad):
        loss.backward()
    grads = [p.grad.detach().clone() for p in model.parameters() if p.grad is not None]
    stats = {
        k: (v.detach().clone() if torch.is_tensor(v) else v)
        for k, v in stats.items()
    }
    return loss.detach().clone(), stats, grads, int(sink_call_count["n"])


def _run_rwkv7_exact_scm_reinforce_rollout(
    *,
    device="cpu",
    reinforce_sequence_replay_enabled=False,
    rwkv_sequence_replay_checkpoint=None,
    rwkv_sequence_replay_batch_chunk_size=None,
    rwkv_sequence_replay_token_budget=None,
):
    _ensure_torch_extensions_dir()
    _seed_everything(123)
    cfg, env_cfg = _build_small_exact_scm_env_cfg()
    env_cfg["reinforce_sequence_replay_enabled"] = bool(reinforce_sequence_replay_enabled)
    cfg["transformer"]["emsize"] = 64
    cfg["transformer"]["nlayers"] = 2
    cfg["transformer"]["rwkv_head_size"] = 64
    if rwkv_sequence_replay_checkpoint is not None:
        cfg["transformer"]["rwkv_sequence_replay_checkpoint"] = bool(rwkv_sequence_replay_checkpoint)
    if rwkv_sequence_replay_batch_chunk_size is not None:
        cfg["transformer"]["rwkv_sequence_replay_batch_chunk_size"] = rwkv_sequence_replay_batch_chunk_size
    if rwkv_sequence_replay_token_budget is not None:
        cfg["transformer"]["rwkv_sequence_replay_token_budget"] = rwkv_sequence_replay_token_budget
    cfg["optimizer"]["rl_objective"] = "reinforce"

    prior = EnvironmentPrior(env_cfg)
    _, model, *_ = get_model(cfg, device=device, should_train=False, verbose=False)
    if str(device) == "cuda":
        model = model.cuda()
    model.train()
    step_fn = _build_policy_step_fn(
        model,
        num_features=int(cfg["prior"]["num_features"]),
        max_cache_len=8,
        kv_cache_mode="immutable",
        kv_cache_page_size=None,
        allow_grad_mutable_cache=False,
        pg_torch_compile=False,
    )

    _seed_everything(515151)
    h_list = prior._sample_batch_hypers(2)
    for h in h_list:
        h["reward_dropout_enabled"] = False
        h["reward_dropout_randomize"] = False
        h["reward_dropout_ratio"] = 0.0
        h["action_noise_train_std"] = 0.05
        h["action_noise_eval_std"] = 0.03

    loss, rollout, stats = prior.rollout_policy_gradient_loss(
        policy_step_fn=step_fn,
        batch_size=2,
        n_samples=8,
        num_features=int(cfg["prior"]["num_features"]),
        device=device,
        single_eval_pos=4,
        collect_x=True,
        h_list_override=[dict(h) for h in h_list],
        env_seeds_override=[17, 29],
        rollout_seeds_override=[101, 211],
        policy_objective_kind="reinforce",
    )
    replay_meta = getattr(prior, "last_rollout_reinforce", None)
    return {
        "loss": loss.detach().clone(),
        "stats": {
            k: (v.detach().clone() if torch.is_tensor(v) else v)
            for k, v in stats.items()
        },
        "x": None if rollout["x"] is None else rollout["x"].detach().clone(),
        "rewards": rollout["rewards"].detach().clone(),
        "log_probs": prior.last_rollout_reinforce["log_probs"].detach().clone(),
        "sequence_replay_applied": bool(prior.last_rollout_reinforce.get("sequence_replay_applied", False)),
        "single_eval_pos": int(rollout.get("single_eval_pos", 0)),
        "replay_action_steps": (
            None if not isinstance(replay_meta, dict) else replay_meta.get("sequence_replay_action_steps", None)
        ),
        "replay_eval_start": (
            None if not isinstance(replay_meta, dict) else replay_meta.get("sequence_replay_eval_start", None)
        ),
    }


class _FakeBox:
    def __init__(self, low, high):
        self.low = np.asarray(low, dtype=np.float32)
        self.high = np.asarray(high, dtype=np.float32)


class _ScriptedEnv:
    def __init__(self, rollout_lengths, rollout_rewards, obs_dim=400, action_dim=30):
        self._rollout_lengths = [int(x) for x in rollout_lengths]
        self._rollout_rewards = [float(x) for x in rollout_rewards]
        self._obs_dim = int(obs_dim)
        self._rollout_idx = -1
        self._step_idx = 0
        self.action_space = _FakeBox(
            low=-np.ones((int(action_dim),), dtype=np.float32),
            high=np.ones((int(action_dim),), dtype=np.float32),
        )

    def reset(self, seed=None):
        del seed
        self._rollout_idx += 1
        self._step_idx = 0
        obs = np.full((self._obs_dim,), float(self._rollout_idx), dtype=np.float32)
        return obs, {}

    def step(self, action):
        del action
        reward = self._rollout_rewards[self._rollout_idx]
        self._step_idx += 1
        terminated = bool(self._step_idx >= self._rollout_lengths[self._rollout_idx])
        obs = np.full(
            (self._obs_dim,),
            float(self._rollout_idx) + 0.1 * float(self._step_idx),
            dtype=np.float32,
        )
        return obs, reward, terminated, False, {}

    def close(self):
        return None


def _install_fake_gym(monkeypatch, env_factory):
    gym_mod = types.ModuleType("gymnasium")
    spaces_mod = types.ModuleType("gymnasium.spaces")
    spaces_mod.Box = _FakeBox
    gym_mod.spaces = spaces_mod
    gym_mod.make = lambda env_name: env_factory(env_name)
    monkeypatch.setitem(sys.modules, "gymnasium", gym_mod)
    monkeypatch.setitem(sys.modules, "gymnasium.spaces", spaces_mod)


def _assemble_split_token(obs_t, action_t, reward_t, reward_mask_t, phase_t, terminal_t, *, obs_dim: int, action_dim: int):
    batch_size = int(obs_t.shape[0])
    obs_slot_dim = int(obs_dim - 2 - int(phase_t is not None) - int(terminal_t is not None))
    x_token = torch.zeros((1, batch_size, obs_dim + action_dim), dtype=obs_t.dtype)
    x_row = x_token[0]
    obs_copy = min(int(obs_t.shape[-1]), obs_slot_dim)
    if obs_copy > 0:
        x_row[:, :obs_copy] = obs_t[:, :obs_copy]
    x_row[:, obs_slot_dim] = reward_t.reshape(batch_size)
    x_row[:, obs_slot_dim + 1] = reward_mask_t.reshape(batch_size)
    if phase_t is not None:
        x_row[:, obs_slot_dim + 2] = phase_t.reshape(batch_size)
    if terminal_t is not None:
        x_row[:, obs_slot_dim + 3] = terminal_t.reshape(batch_size)
    action_start = obs_slot_dim + 2 + int(phase_t is not None) + int(terminal_t is not None)
    action_copy = min(int(action_t.shape[-1]), action_dim)
    if action_copy > 0:
        x_row[:, action_start: action_start + action_copy] = action_t[:, :action_copy]
    y_token = reward_t.reshape(1, batch_size)
    return x_token, y_token


def test_rwkv7_rlpfn_builder_has_correct_action_head_and_5m_budget():
    _ensure_torch_extensions_dir()
    cfg = _build_rwkv7_rlpfn_config()
    _, model, *_ = get_model(cfg, device="cpu", should_train=False, verbose=False)

    param_count = sum(p.numel() for p in model.parameters())
    assert type(model).__name__ == "RWKV7PFN"
    assert 4_800_000 <= param_count <= 5_600_000
    assert model.policy_action_head_required()
    assert model.has_correct_policy_action_head()
    assert int(model.policy_action_dim) == 30


def test_rwkv7_rlpfn_split_policy_step_matches_generic_policy_step():
    _ensure_torch_extensions_dir()
    _seed_everything(7)
    cfg = _build_rwkv7_rlpfn_config()
    cfg["transformer"]["emsize"] = 64
    cfg["transformer"]["nlayers"] = 2
    cfg["transformer"]["rwkv_head_size"] = 64
    _, model, *_ = get_model(cfg, device="cpu", should_train=False, verbose=False)
    model.eval()

    obs = torch.randn(2, 400)
    action = torch.randn(2, 30)
    reward = torch.randn(2, 1)
    reward_mask = torch.ones(2, 1)
    phase = torch.tensor([[0.0], [1.0]], dtype=torch.float32)
    terminal = torch.tensor([[0.0], [1.0]], dtype=torch.float32)

    x_token, y_token = _assemble_split_token(
        obs,
        action,
        reward,
        reward_mask,
        phase,
        terminal,
        obs_dim=int(model.encoder.obs_dim),
        action_dim=int(model.encoder.action_dim),
    )
    out_generic, state_generic = model.forward_policy_step(x_token, y_token, kv_cache=None)
    out_split, state_split = model.forward_policy_step_split(
        obs,
        action,
        reward,
        reward_mask,
        phase_t=phase,
        terminal_t=terminal,
        kv_cache=None,
    )

    assert torch.allclose(out_generic, out_split, atol=1e-6, rtol=1e-6)
    assert len(state_generic) == len(state_split) == int(cfg["transformer"]["nlayers"])
    for generic_layer, split_layer in zip(state_generic, state_split):
        for generic_tensor, split_tensor in zip(generic_layer, split_layer):
            assert torch.allclose(generic_tensor, split_tensor, atol=1e-6, rtol=1e-6)


def test_rwkv7_rlpfn_policy_step_reuses_state_cache():
    _ensure_torch_extensions_dir()
    _seed_everything(11)
    cfg = _build_rwkv7_rlpfn_config()
    cfg["transformer"]["emsize"] = 64
    cfg["transformer"]["nlayers"] = 2
    cfg["transformer"]["rwkv_head_size"] = 64
    _, model, *_ = get_model(cfg, device="cpu", should_train=False, verbose=False)
    model.eval()
    step_fn = _build_policy_step_fn(
        model,
        num_features=int(cfg["prior"]["num_features"]),
        max_cache_len=16,
        kv_cache_mode="immutable",
        kv_cache_page_size=None,
        allow_grad_mutable_cache=False,
        pg_torch_compile=False,
    )

    env_info = {
        "obs_slot_dim": 400,
        "action_slot_dim": 30,
        "action_dim": 30,
        "phase_t": torch.tensor([[0.0], [1.0]], dtype=torch.float32),
        "terminal_t": torch.tensor([[0.0], [0.0]], dtype=torch.float32),
    }
    obs_1 = torch.randn(2, 400)
    action_1 = torch.randn(2, 30)
    reward_1 = torch.randn(2, 1)
    reward_mask = torch.ones(2, 1)
    obs_2 = torch.randn(2, 400)
    action_2 = torch.randn(2, 30)
    reward_2 = torch.randn(2, 1)

    _, cache_1 = step_fn(obs_1, action_1, reward_1, reward_mask, None, 0, env_info)
    _, cache_with_history = step_fn(obs_2, action_2, reward_2, reward_mask, cache_1, 1, env_info)
    _, cache_without_history = step_fn(obs_2, action_2, reward_2, reward_mask, None, 1, env_info)

    assert isinstance(cache_1, list)
    assert len(cache_1) == int(cfg["transformer"]["nlayers"])
    cache_1_nonzero = any(float(t.detach().abs().sum()) > 0.0 for layer_state in cache_1 for t in layer_state)
    cache_diverged = any(
        not torch.allclose(t_hist, t_fresh)
        for layer_hist, layer_fresh in zip(cache_with_history, cache_without_history)
        for t_hist, t_fresh in zip(layer_hist, layer_fresh)
    )
    assert cache_1_nonzero
    assert cache_diverged


@pytest.mark.skipif(not torch.cuda.is_available(), reason="RWKV head-override step test requires CUDA runtime")
def test_rwkv7_policy_step_head_override_changes_output_without_mutating_model():
    _ensure_torch_extensions_dir()
    _seed_everything(23)
    cfg = _build_rwkv7_rlpfn_config()
    cfg["transformer"]["emsize"] = 64
    cfg["transformer"]["nlayers"] = 2
    cfg["transformer"]["rwkv_head_size"] = 64
    _, model, *_ = get_model(cfg, device="cuda", should_train=False, verbose=False)
    model = model.cuda().eval()
    step_fn = _build_policy_step_fn(
        model,
        num_features=int(cfg["prior"]["num_features"]),
        max_cache_len=16,
        kv_cache_mode="immutable",
        kv_cache_page_size=None,
        allow_grad_mutable_cache=False,
        pg_torch_compile=False,
    )

    env_info = {
        "obs_slot_dim": 400,
        "action_slot_dim": 30,
        "action_dim": 30,
        "phase_t": torch.tensor([[0.0], [1.0]], dtype=torch.float32),
        "terminal_t": torch.tensor([[0.0], [0.0]], dtype=torch.float32),
    }
    obs = torch.randn(2, 400, device="cuda")
    action = torch.randn(2, 30, device="cuda")
    reward = torch.randn(2, 1, device="cuda")
    reward_mask = torch.ones(2, 1, device="cuda")

    base_out, _ = step_fn(obs, action, reward, reward_mask, None, 0, env_info)
    base_out_again, _ = step_fn(obs, action, reward, reward_mask, None, 0, env_info)
    override = OrderedDict(
        (name, torch.zeros_like(param))
        for name, param in model.policy_action_head.named_parameters()
    )
    override["0.bias"] = torch.full_like(override["0.bias"], 0.5)
    override_out, _ = step_fn(
        obs,
        action,
        reward,
        reward_mask,
        None,
        0,
        env_info,
        _policy_action_head_params_override=override,
    )

    assert torch.allclose(base_out, base_out_again, atol=1e-6, rtol=1e-6)
    assert not torch.allclose(base_out, override_out, atol=1e-6, rtol=1e-6)


def test_rwkv7_block_batch1_fastpath_matches_manual_vmap_semantics():
    _ensure_torch_extensions_dir()
    _seed_everything(13)
    cfg = _build_rwkv7_rlpfn_config()
    cfg["transformer"]["emsize"] = 64
    cfg["transformer"]["nlayers"] = 2
    cfg["transformer"]["rwkv_head_size"] = 64
    _, model, *_ = get_model(cfg, device="cpu", should_train=False, verbose=False)
    model.eval()
    block = model.rwkv_core.blocks[0]
    official = _load_official_rwkv7_demo_rnn()

    x = torch.randn(1, int(cfg["transformer"]["emsize"]))
    state = block.init_state(1, device=x.device, dtype=x.dtype)
    v_first = torch.randn_like(x)

    out_fast, state_fast, v_first_fast = block.forward_step(
        x.clone(),
        tuple(t.clone() for t in state),
        v_first.clone(),
    )

    x_ref = block.ln0(x.clone()) if hasattr(block, "ln0") else x.clone()
    att_x_prev, att_kv, ffn_x_prev = tuple(t.clone() for t in state)
    att_in = block.ln1(x_ref)
    att_out_i, att_x_prev_next_i, att_kv_next_i, v_first_next_i = official.time_mixing__(
        int(block.layer_id),
        int(block.n_head),
        int(block.head_size),
        att_in[0],
        att_x_prev[0],
        v_first[0],
        att_kv[0],
        block.att.x_r.squeeze(0).squeeze(0),
        block.att.x_w.squeeze(0).squeeze(0),
        block.att.x_k.squeeze(0).squeeze(0),
        block.att.x_v.squeeze(0).squeeze(0),
        block.att.x_a.squeeze(0).squeeze(0),
        block.att.x_g.squeeze(0).squeeze(0),
        block.att.w0.squeeze(0).squeeze(0),
        block.att.w1,
        block.att.w2,
        block.att.a0.squeeze(0).squeeze(0),
        block.att.a1,
        block.att.a2,
        block.att.v0.squeeze(0).squeeze(0),
        block.att.v1,
        block.att.v2,
        block.att.g1,
        block.att.g2,
        block.att.k_k.squeeze(0).squeeze(0),
        block.att.k_a.squeeze(0).squeeze(0),
        block.att.r_k.reshape(-1),
        block.att.key.weight,
        block.att.value.weight,
        block.att.receptance.weight,
        block.att.output.weight,
        block.att.ln_x.weight,
        block.att.ln_x.bias,
    )
    x_ref = x_ref + att_out_i.unsqueeze(0)
    ffn_in = block.ln2(x_ref)
    ffn_out_i, ffn_x_prev_next_i = official.channel_mixing__(
        ffn_in[0],
        ffn_x_prev[0],
        block.ffn.x_k.squeeze(0).squeeze(0),
        block.ffn.key.weight,
        block.ffn.value.weight,
    )
    x_ref = x_ref + ffn_out_i.unsqueeze(0)
    state_ref = (
        att_x_prev_next_i.unsqueeze(0),
        att_kv_next_i.unsqueeze(0),
        ffn_x_prev_next_i.unsqueeze(0),
    )
    v_first_ref = v_first_next_i.unsqueeze(0)

    assert torch.allclose(out_fast, x_ref, atol=1e-6, rtol=1e-6)
    for tensor_fast, tensor_ref in zip(state_fast, state_ref):
        assert torch.allclose(tensor_fast, tensor_ref, atol=1e-6, rtol=1e-6)
    assert torch.allclose(v_first_fast, v_first_ref, atol=1e-6, rtol=1e-6)


def test_rwkv7_block_batch_gt1_no_grad_fastpath_matches_manual_vmap_semantics():
    _ensure_torch_extensions_dir()
    _seed_everything(131)
    cfg = _build_rwkv7_rlpfn_config()
    cfg["transformer"]["emsize"] = 64
    cfg["transformer"]["nlayers"] = 2
    cfg["transformer"]["rwkv_head_size"] = 64
    _, model, *_ = get_model(cfg, device="cpu", should_train=False, verbose=False)
    model.eval()
    block = model.rwkv_core.blocks[0]
    official = _load_official_rwkv7_demo_rnn()

    batch_size = 3
    x = torch.randn(batch_size, int(cfg["transformer"]["emsize"]))
    state = block.init_state(batch_size, device=x.device, dtype=x.dtype)
    v_first = torch.randn_like(x)

    with torch.no_grad():
        out_fast, state_fast, v_first_fast = block.forward_step(
            x.clone(),
            tuple(t.clone() for t in state),
            v_first.clone(),
        )

    x_ref = block.ln0(x.clone()) if hasattr(block, "ln0") else x.clone()
    att_x_prev, att_kv, ffn_x_prev = tuple(t.clone() for t in state)
    att_in = block.ln1(x_ref)

    def _single_att_step(x_i, x_prev_i, kv_state_i, v_first_i):
        return official.time_mixing__(
            int(block.layer_id),
            int(block.n_head),
            int(block.head_size),
            x_i,
            x_prev_i,
            v_first_i,
            kv_state_i,
            block.att.x_r.squeeze(0).squeeze(0),
            block.att.x_w.squeeze(0).squeeze(0),
            block.att.x_k.squeeze(0).squeeze(0),
            block.att.x_v.squeeze(0).squeeze(0),
            block.att.x_a.squeeze(0).squeeze(0),
            block.att.x_g.squeeze(0).squeeze(0),
            block.att.w0.squeeze(0).squeeze(0),
            block.att.w1,
            block.att.w2,
            block.att.a0.squeeze(0).squeeze(0),
            block.att.a1,
            block.att.a2,
            block.att.v0.squeeze(0).squeeze(0),
            block.att.v1,
            block.att.v2,
            block.att.g1,
            block.att.g2,
            block.att.k_k.squeeze(0).squeeze(0),
            block.att.k_a.squeeze(0).squeeze(0),
            block.att.r_k.reshape(-1),
            block.att.key.weight,
            block.att.value.weight,
            block.att.receptance.weight,
            block.att.output.weight,
            block.att.ln_x.weight,
            block.att.ln_x.bias,
        )

    def _single_ffn_step(x_i, x_prev_i):
        return official.channel_mixing__(
            x_i,
            x_prev_i,
            block.ffn.x_k.squeeze(0).squeeze(0),
            block.ffn.key.weight,
            block.ffn.value.weight,
        )

    att_out_ref, att_x_prev_next_ref, att_kv_next_ref, v_first_next_ref = torch.vmap(
        _single_att_step, in_dims=(0, 0, 0, 0), out_dims=(0, 0, 0, 0)
    )(att_in, att_x_prev, att_kv, v_first)
    x_ref = x_ref + att_out_ref
    ffn_in = block.ln2(x_ref)
    ffn_out_ref, ffn_x_prev_next_ref = torch.vmap(_single_ffn_step, in_dims=(0, 0), out_dims=(0, 0))(
        ffn_in, ffn_x_prev
    )
    x_ref = x_ref + ffn_out_ref

    assert torch.allclose(out_fast, x_ref, atol=1e-6, rtol=1e-6)
    assert torch.allclose(state_fast[0], att_x_prev_next_ref, atol=1e-6, rtol=1e-6)
    assert torch.allclose(state_fast[1], att_kv_next_ref, atol=1e-6, rtol=1e-6)
    assert torch.allclose(state_fast[2], ffn_x_prev_next_ref, atol=1e-6, rtol=1e-6)
    assert torch.allclose(v_first_fast, v_first_next_ref, atol=1e-6, rtol=1e-6)


def test_rwkv7_block_batch1_fastpath_skips_vmap(monkeypatch):
    _ensure_torch_extensions_dir()
    _seed_everything(17)
    cfg = _build_rwkv7_rlpfn_config()
    cfg["transformer"]["emsize"] = 64
    cfg["transformer"]["nlayers"] = 2
    cfg["transformer"]["rwkv_head_size"] = 64
    _, model, *_ = get_model(cfg, device="cpu", should_train=False, verbose=False)
    model.eval()
    block = model.rwkv_core.blocks[0]

    x = torch.randn(1, int(cfg["transformer"]["emsize"]))
    state = block.init_state(1, device=x.device, dtype=x.dtype)
    v_first = torch.randn_like(x)

    orig_vmap = torch.vmap
    vmap_calls = {"count": 0}

    def _counting_vmap(*args, **kwargs):
        vmap_calls["count"] += 1
        return orig_vmap(*args, **kwargs)

    monkeypatch.setattr(torch, "vmap", _counting_vmap)
    out, state_next, v_first_next = block.forward_step(x, state, v_first)

    assert vmap_calls["count"] == 0
    assert tuple(out.shape) == tuple(x.shape)
    assert len(state_next) == 3
    assert tuple(v_first_next.shape) == tuple(v_first.shape)


def test_rwkv7_block_batch_gt1_no_grad_fastpath_skips_vmap(monkeypatch):
    _ensure_torch_extensions_dir()
    _seed_everything(171)
    cfg = _build_rwkv7_rlpfn_config()
    cfg["transformer"]["emsize"] = 64
    cfg["transformer"]["nlayers"] = 2
    cfg["transformer"]["rwkv_head_size"] = 64
    _, model, *_ = get_model(cfg, device="cpu", should_train=False, verbose=False)
    model.eval()
    block = model.rwkv_core.blocks[0]

    x = torch.randn(3, int(cfg["transformer"]["emsize"]))
    state = block.init_state(3, device=x.device, dtype=x.dtype)
    v_first = torch.randn_like(x)

    orig_vmap = torch.vmap
    vmap_calls = {"count": 0}

    def _counting_vmap(*args, **kwargs):
        vmap_calls["count"] += 1
        return orig_vmap(*args, **kwargs)

    monkeypatch.setattr(torch, "vmap", _counting_vmap)
    with torch.no_grad():
        out, state_next, v_first_next = block.forward_step(x, state, v_first)

    assert vmap_calls["count"] == 0
    assert tuple(out.shape) == tuple(x.shape)
    assert len(state_next) == 3
    assert tuple(v_first_next.shape) == tuple(v_first.shape)


def test_rwkv7_core_cuda_batch1_official_eval_fastpath_matches_raw_step():
    if not torch.cuda.is_available():
        return

    _ensure_torch_extensions_dir()
    _seed_everything(19)
    cfg = _build_rwkv7_rlpfn_config()
    cfg["transformer"]["emsize"] = 64
    cfg["transformer"]["nlayers"] = 2
    cfg["transformer"]["rwkv_head_size"] = 64
    _, model, *_ = get_model(cfg, device="cuda", should_train=False, verbose=False)
    model = model.cuda().eval()
    core = model.rwkv_core

    token = torch.randn(1, int(cfg["transformer"]["emsize"]), device="cuda", dtype=torch.float32)

    with torch.no_grad():
        out_fast, state_fast = core.forward_step(token.clone(), None)

        raw_state = core.init_state(1, device=token.device, dtype=token.dtype)
        x_ref = token.clone()
        new_state = []
        v_first = None
        for block, block_state in zip(core.blocks, raw_state):
            x_ref, block_state_next, v_first = block.forward_step(x_ref, block_state, v_first)
            new_state.append(block_state_next)
        x_ref = core.ln_out(x_ref)
        flat_state_ref = core._flatten_official_eval_state(new_state)

    assert isinstance(state_fast, list)
    assert len(state_fast) == int(cfg["transformer"]["nlayers"]) * 3
    assert torch.allclose(out_fast, x_ref, atol=1e-2, rtol=1e-2)
    for fast_tensor, ref_tensor in zip(state_fast, flat_state_ref):
        assert torch.allclose(fast_tensor.to(dtype=ref_tensor.dtype), ref_tensor, atol=1e-2, rtol=1e-2)


def test_rwkv7_rlpfn_exact_scm_reinforce_rollout_backward_is_finite():
    _ensure_torch_extensions_dir()
    _seed_everything(123)
    cfg, env_cfg = _build_small_exact_scm_env_cfg()
    prior = EnvironmentPrior(env_cfg)
    _, model, *_ = get_model(cfg, device="cpu", should_train=False, verbose=False)
    model.train()
    step_fn = _build_policy_step_fn(
        model,
        num_features=int(cfg["prior"]["num_features"]),
        max_cache_len=16,
        kv_cache_mode="immutable",
        kv_cache_page_size=None,
        allow_grad_mutable_cache=False,
        pg_torch_compile=False,
    )

    _seed_everything(515151)
    h_list = prior._sample_batch_hypers(2)
    for h in h_list:
        h["reward_dropout_enabled"] = False
        h["reward_dropout_randomize"] = False
        h["reward_dropout_ratio"] = 0.0
        h["action_noise_train_std"] = 0.05
        h["action_noise_eval_std"] = 0.03

    loss, rollout, stats = prior.rollout_policy_gradient_loss(
        policy_step_fn=step_fn,
        batch_size=2,
        n_samples=8,
        num_features=int(cfg["prior"]["num_features"]),
        device="cpu",
        single_eval_pos=4,
        collect_x=False,
        h_list_override=[dict(h) for h in h_list],
        env_seeds_override=[17, 29],
        rollout_seeds_override=[101, 211],
        policy_objective_kind="reinforce",
    )
    loss.backward()

    policy_grads = [p.grad.detach() for p in model.policy_action_head.parameters() if p.grad is not None]
    backbone_grads = [p.grad.detach() for p in model.rwkv_core.parameters() if p.grad is not None]

    assert torch.isfinite(loss).item()
    assert torch.isfinite(rollout["rewards"]).all().item()
    assert torch.isfinite(stats["objective"]).item()
    assert policy_grads
    assert backbone_grads
    assert all(torch.isfinite(g).all().item() for g in policy_grads)
    assert all(torch.isfinite(g).all().item() for g in backbone_grads)
    assert sum(float(g.abs().sum()) for g in policy_grads) > 0.0
    assert any(float(g.abs().sum()) > 0.0 for g in backbone_grads)


@pytest.mark.parametrize(
    ("head_type", "expected_class_name"),
    [
        ("cfmi_resnet", "CFMIResidualFlowMatchingHead"),
        ("rwkv_two_layer", "RWKVTwoLayerFlowMatchingHead"),
    ],
)
def test_rwkv7_replay_policy_sequence_outputs_emit_aux_predictions(head_type, expected_class_name):
    if not torch.cuda.is_available():
        return
    _ensure_torch_extensions_dir()
    _seed_everything(321)
    cfg = _build_rwkv7_rlpfn_config()
    cfg["prior"]["environment"]["normalized_q_value_weight"] = 0.2
    cfg["prior"]["environment"]["next_state_flow_matching_weight"] = 0.5
    cfg["prior"]["environment"]["next_state_flow_head_type"] = head_type
    cfg["transformer"]["emsize"] = 64
    cfg["transformer"]["nlayers"] = 2
    cfg["transformer"]["rwkv_head_size"] = 64
    _, model, *_ = get_model(cfg, device="cuda", should_train=False, verbose=False)
    model = model.cuda().eval()
    assert model.next_state_flow_head.__class__.__name__ == expected_class_name
    assert model.normalized_q_value_bardist.__class__.__name__ == "FullSupportBarDistribution"

    seq_len = 6
    batch_size = 2
    eval_start = 3
    flow_dim = int(cfg["prior"]["environment"]["obs_slot_dim"])
    with torch.no_grad():
        outputs = model.replay_policy_sequence_outputs(
            torch.randn(seq_len, batch_size, int(cfg["prior"]["num_features"]), device="cuda"),
            torch.randn(seq_len, batch_size, device="cuda"),
            eval_start=eval_start,
            flow_matching_xt=torch.randn(seq_len - eval_start, batch_size, flow_dim, device="cuda"),
            flow_matching_t=torch.rand(seq_len - eval_start, batch_size, 1, device="cuda"),
        )

    assert set(outputs.keys()) == {"action_mean", "normalized_q", "normalized_q_logits", "next_state_flow"}
    assert tuple(outputs["action_mean"].shape) == (seq_len - eval_start, batch_size, int(cfg["transformer"]["x_action_dim"]))
    assert tuple(outputs["normalized_q"].shape) == (seq_len - eval_start, batch_size)
    assert tuple(outputs["normalized_q_logits"].shape) == (
        seq_len - eval_start,
        batch_size,
        model.normalized_q_value_bardist.num_bars,
    )
    assert tuple(outputs["next_state_flow"].shape) == (seq_len - eval_start, batch_size, flow_dim)


def test_transformer_replay_policy_sequence_outputs_emit_cfmi_aux_predictions():
    _seed_everything(654)
    cfg = deepcopy(get_model_default_config("rlpfn"))
    cfg["transformer"]["backbone"] = "transformer"
    cfg["prior"]["environment"]["normalized_q_value_weight"] = 0.2
    cfg["prior"]["environment"]["next_state_flow_matching_weight"] = 0.5
    cfg["prior"]["environment"]["next_state_flow_head_type"] = "cfmi_resnet"
    cfg["transformer"]["emsize"] = 64
    cfg["transformer"]["nlayers"] = 2
    cfg["transformer"]["nhead"] = 4
    _, model, *_ = get_model(cfg, device="cpu", should_train=False, verbose=False)
    model.eval()
    assert model.next_state_flow_head.__class__.__name__ == "CFMIResidualFlowMatchingHead"
    assert model.normalized_q_value_bardist.__class__.__name__ == "FullSupportBarDistribution"

    seq_len = 6
    batch_size = 2
    eval_start = 2
    flow_dim = int(cfg["prior"]["environment"]["obs_slot_dim"])
    with torch.no_grad():
        outputs = model.replay_policy_sequence_outputs(
            torch.randn(seq_len, batch_size, int(cfg["prior"]["num_features"])),
            torch.randn(seq_len, batch_size),
            eval_start=eval_start,
            flow_matching_xt=torch.randn(seq_len - eval_start, batch_size, flow_dim),
            flow_matching_t=torch.rand(seq_len - eval_start, batch_size, 1),
        )

    assert set(outputs.keys()) == {"action_mean", "normalized_q", "normalized_q_logits", "next_state_flow"}
    assert tuple(outputs["action_mean"].shape) == (seq_len - eval_start, batch_size, int(cfg["transformer"]["x_action_dim"]))
    assert tuple(outputs["normalized_q"].shape) == (seq_len - eval_start, batch_size)
    assert tuple(outputs["normalized_q_logits"].shape) == (
        seq_len - eval_start,
        batch_size,
        model.normalized_q_value_bardist.num_bars,
    )
    assert tuple(outputs["next_state_flow"].shape) == (seq_len - eval_start, batch_size, flow_dim)


def test_transformer_rejects_rwkv_two_layer_flow_head():
    _seed_everything(655)
    cfg = deepcopy(get_model_default_config("rlpfn"))
    cfg["transformer"]["backbone"] = "transformer"
    cfg["prior"]["environment"]["next_state_flow_matching_weight"] = 0.5
    cfg["prior"]["environment"]["next_state_flow_head_type"] = "rwkv_two_layer"
    cfg["transformer"]["emsize"] = 64
    cfg["transformer"]["nlayers"] = 2
    cfg["transformer"]["nhead"] = 4
    with pytest.raises(ValueError, match="does not support"):
        get_model(cfg, device="cpu", should_train=False, verbose=False)


def test_reinforce_sequence_replay_loss_from_rollout_includes_aux_losses():
    _seed_everything(999)
    _, env_cfg = _build_small_exact_scm_env_cfg()
    env_cfg["policy_gradient_weight"] = 0.1
    env_cfg["normalized_q_value_weight"] = 0.25
    env_cfg["next_state_flow_matching_weight"] = 0.5
    env_cfg["obs_slot_dim"] = 8
    env_cfg["action_slot_dim"] = 3
    prior = EnvironmentPrior(env_cfg)
    bardist = make_standardized_full_support_bar_distribution()

    class _DummyReplayModel:
        def resolve_replay_batch_chunk_size(self, *, seq_len, total_batch):
            del seq_len
            return int(total_batch)

        def get_normalized_q_value_bardist(self):
            return bardist

    def _replay_tokens(x_tokens, y_tokens, *, eval_start=0):
        del y_tokens
        query = x_tokens[eval_start:]
        return torch.zeros(
            int(query.shape[0]),
            int(query.shape[1]),
            3,
            dtype=query.dtype,
            device=query.device,
        )

    def _replay_outputs(x_tokens, y_tokens, *, eval_start=0, flow_matching_xt=None, flow_matching_t=None):
        del y_tokens, flow_matching_t
        query = x_tokens[eval_start:]
        outputs = {
            "action_mean": torch.zeros(
                int(query.shape[0]),
                int(query.shape[1]),
                3,
                dtype=query.dtype,
                device=query.device,
            ),
            "normalized_q_logits": torch.zeros(
                int(query.shape[0]),
                int(query.shape[1]),
                bardist.num_bars,
                dtype=query.dtype,
                device=query.device,
            ),
        }
        outputs["normalized_q"] = bardist.mean(outputs["normalized_q_logits"])
        if flow_matching_xt is not None:
            outputs["next_state_flow"] = torch.zeros_like(flow_matching_xt)
        return outputs

    def _dummy_step_fn(*args, **kwargs):
        raise AssertionError("direct replay loss test should not call live policy_step")

    _dummy_step_fn._reinforce_sequence_replay_fn = _replay_tokens
    _dummy_step_fn._reinforce_sequence_replay_outputs_fn = _replay_outputs
    _dummy_step_fn._fit_action_dim_fn = lambda x, action_dim: x[..., :action_dim]
    _dummy_step_fn._model_ref = _DummyReplayModel()

    seq_len = 5
    eval_start = 2
    batch_size = 2
    obs_dim = int(env_cfg["obs_slot_dim"])
    action_dim = int(env_cfg["action_slot_dim"])
    rewards = torch.randn(seq_len, batch_size, dtype=torch.float32)
    replay_payload = {
        "reward_in": torch.randn(seq_len, batch_size, dtype=torch.float32),
        "sampled_action": torch.randn(seq_len - eval_start, batch_size, action_dim, dtype=torch.float32),
        "action_std": torch.full((seq_len - eval_start, batch_size), 0.05, dtype=torch.float32),
        "action_mask": torch.ones(batch_size, action_dim, dtype=torch.bool),
        "next_obs": torch.randn(seq_len - eval_start, batch_size, obs_dim, dtype=torch.float32),
        "obs_mask": torch.ones(batch_size, obs_dim, dtype=torch.bool),
        "eval_start": eval_start,
        "full_length": seq_len,
    }
    loss, stats, replay_log_probs = prior.reinforce_sequence_replay_loss_from_rollout(
        policy_step_fn=_dummy_step_fn,
        x_tokens=torch.randn(seq_len, batch_size, 12, dtype=torch.float32),
        rewards=rewards,
        replay_payload=replay_payload,
        single_eval_pos=eval_start,
        fit_action_dim_fn=_dummy_step_fn._fit_action_dim_fn,
    )

    assert torch.isfinite(loss).item()
    assert torch.isfinite(replay_log_probs).all().item()
    assert "normalized_q_value_loss" in stats
    assert "next_state_flow_matching_loss" in stats
    assert "policy_total_loss" in stats
    assert int(stats["normalized_q_value_head_applied"]) == 1
    assert int(stats["next_state_flow_matching_head_applied"]) == 1
    assert torch.allclose(loss.detach(), stats["policy_total_loss"].detach(), atol=1e-6, rtol=1e-6)
    assert float(stats["policy_gradient_weight"]) == 0.1


def test_rwkv7_reinforce_sequence_replay_requires_cuda():
    _ensure_torch_extensions_dir()
    _seed_everything(123)
    _, env_cfg = _build_small_exact_scm_env_cfg()
    env_cfg["reinforce_sequence_replay_enabled"] = True
    prior = EnvironmentPrior(env_cfg)

    def _dummy_step_fn(*args, **kwargs):
        raise AssertionError("CPU replay guard should trigger before rollout executes policy_step_fn")
    _dummy_step_fn._reinforce_sequence_replay_fn = lambda x_tokens, y_tokens: x_tokens

    with pytest.raises(RuntimeError, match="requires CUDA"):
        prior.rollout_policy_gradient_loss(
            policy_step_fn=_dummy_step_fn,
            batch_size=2,
            n_samples=8,
            num_features=434,
            device="cpu",
            single_eval_pos=4,
            collect_x=False,
            policy_objective_kind="reinforce",
        )


def test_rwkv7_anil_requires_cuda():
    _ensure_torch_extensions_dir()
    _seed_everything(123)
    _, env_cfg = _build_small_exact_scm_env_cfg()
    prior = EnvironmentPrior(env_cfg)

    class _DummyReplayModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.policy_action_head = torch.nn.Linear(4, 3)

        def require_policy_action_head(self):
            return True

        def replay_policy_sequence_tokens(self, *args, **kwargs):
            raise AssertionError("CUDA guard should trigger before replay is used")

    def _dummy_step_fn(*args, **kwargs):
        raise AssertionError("CUDA guard should trigger before policy_step_fn is used")

    _dummy_step_fn._model_ref = _DummyReplayModel()
    _dummy_step_fn._fit_action_dim_fn = lambda x, action_dim: x[..., :action_dim]

    _seed_everything(515151)
    h_list = prior._sample_batch_hypers(1)
    for h in h_list:
        h["reward_dropout_enabled"] = False
        h["reward_dropout_randomize"] = False
        h["reward_dropout_ratio"] = 0.0

    with pytest.raises(RuntimeError, match="requires CUDA"):
        _compute_policy_rollout_chunk_loss(
            env_prior=prior,
            policy_step_fn=_dummy_step_fn,
            batch_size=1,
            n_samples=8,
            num_features=16,
            device="cpu",
            single_eval_pos=4,
            collect_x=False,
            policy_rollout_checkpoint=False,
            policy_rollout_checkpoint_reentrant=True,
            pg_saved_tensors_cpu_offload=False,
            pg_saved_tensors_pin_memory=True,
            pg_tbptt_window=None,
            h_list_override=[dict(h) for h in h_list],
            env_seeds_override=[17],
            rollout_seeds_override=[101],
            rl_objective="anil",
        )


def test_anil_rejects_tbptt_before_attempting_cuda_or_rollout():
    _ensure_torch_extensions_dir()
    _, env_cfg = _build_small_exact_scm_env_cfg()
    prior = EnvironmentPrior(env_cfg)

    class _DummyReplayModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.policy_action_head = torch.nn.Linear(4, 3)

        def require_policy_action_head(self):
            return True

        def replay_policy_sequence_tokens(self, *args, **kwargs):
            raise AssertionError("TBPTT guard should trigger before replay is used")

    def _dummy_step_fn(*args, **kwargs):
        raise AssertionError("TBPTT guard should trigger before rollout executes policy_step_fn")

    _dummy_step_fn._model_ref = _DummyReplayModel()
    _dummy_step_fn._fit_action_dim_fn = lambda x, action_dim: x[..., :action_dim]

    with pytest.raises(RuntimeError, match="requires pg_tbptt_window=None"):
        _compute_policy_rollout_chunk_loss(
            env_prior=prior,
            policy_step_fn=_dummy_step_fn,
            batch_size=1,
            n_samples=8,
            num_features=16,
            device="cpu",
            single_eval_pos=4,
            collect_x=False,
            policy_rollout_checkpoint=False,
            policy_rollout_checkpoint_reentrant=True,
            pg_saved_tensors_cpu_offload=False,
            pg_saved_tensors_pin_memory=True,
            pg_tbptt_window=4,
            h_list_override=[dict(h) for h in prior._sample_batch_hypers(1)],
            env_seeds_override=[17],
            rollout_seeds_override=[101],
            rl_objective="anil",
        )


def test_anil_rejects_rollout_checkpoint_before_attempting_cuda_or_rollout():
    _ensure_torch_extensions_dir()
    _, env_cfg = _build_small_exact_scm_env_cfg()
    prior = EnvironmentPrior(env_cfg)

    class _DummyReplayModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.policy_action_head = torch.nn.Linear(4, 3)

        def require_policy_action_head(self):
            return True

        def replay_policy_sequence_tokens(self, *args, **kwargs):
            raise AssertionError("Checkpoint guard should trigger before replay is used")

    def _dummy_step_fn(*args, **kwargs):
        raise AssertionError("Checkpoint guard should trigger before rollout executes policy_step_fn")

    _dummy_step_fn._model_ref = _DummyReplayModel()
    _dummy_step_fn._fit_action_dim_fn = lambda x, action_dim: x[..., :action_dim]

    with pytest.raises(RuntimeError, match="does not support policy_rollout_checkpoint"):
        _compute_policy_rollout_chunk_loss(
            env_prior=prior,
            policy_step_fn=_dummy_step_fn,
            batch_size=1,
            n_samples=8,
            num_features=16,
            device="cpu",
            single_eval_pos=4,
            collect_x=False,
            policy_rollout_checkpoint=True,
            policy_rollout_checkpoint_reentrant=True,
            pg_saved_tensors_cpu_offload=False,
            pg_saved_tensors_pin_memory=True,
            pg_tbptt_window=None,
            h_list_override=[dict(h) for h in prior._sample_batch_hypers(1)],
            env_seeds_override=[17],
            rollout_seeds_override=[101],
            rl_objective="anil",
        )


def test_anil_head_adaptation_updates_functional_head_without_mutating_base_params():
    class _DummyReplayModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.policy_action_head = torch.nn.Linear(2, 1, bias=False)

        def require_policy_action_head(self):
            return True

        def replay_policy_sequence_tokens(
            self,
            x_tokens,
            y_tokens,
            *,
            eval_start=0,
            policy_action_head_params_override=None,
        ):
            del y_tokens
            features = x_tokens[int(eval_start):, :, :2]
            if policy_action_head_params_override is not None:
                weight = policy_action_head_params_override["weight"]
                return torch.nn.functional.linear(features, weight, None)
            return self.policy_action_head(features)

    class _FakePrior:
        def reinforce_sequence_replay_loss_from_rollout(
            self,
            *,
            policy_step_fn,
            x_tokens,
            rewards,
            replay_payload,
            single_eval_pos,
            baseline_mode,
            fit_action_dim_fn,
        ):
            del rewards, baseline_mode, fit_action_dim_fn
            preds = policy_step_fn._reinforce_sequence_replay_fn(
                x_tokens,
                replay_payload["reward_in"],
                eval_start=int(single_eval_pos),
            ).squeeze(-1)
            target = replay_payload["target"]
            loss = ((preds - target) ** 2).mean()
            stats = {
                "objective": torch.tensor(1.0, dtype=torch.float32),
                "reward_mean": torch.tensor(0.0, dtype=torch.float32),
                "reward_std": torch.tensor(1.0, dtype=torch.float32),
            }
            return loss, stats, None

    model = _DummyReplayModel()

    def _base_step_fn(*args, **kwargs):
        raise AssertionError("adapt helper should use replay path, not live policy_step")

    _base_step_fn._model_ref = model
    _base_step_fn._fit_action_dim_fn = lambda x, action_dim: x[..., :action_dim]

    support_rollout = {
        "x": torch.tensor(
            [
                [[1.0, -1.0]],
                [[0.5, 2.0]],
                [[-1.5, 0.25]],
            ],
            dtype=torch.float32,
        ),
        "rewards": torch.zeros((3, 1), dtype=torch.float32),
        "single_eval_pos": 1,
    }
    support_payload = {
        "reward_in": torch.zeros((3, 1), dtype=torch.float32),
        "target": torch.tensor([[1.0], [-0.5]], dtype=torch.float32),
    }
    init_weight = model.policy_action_head.weight.detach().clone()

    adapted_head, support_stats = train_mod._adapt_anil_policy_action_head(
        _FakePrior(),
        _base_step_fn,
        support_rollout=support_rollout,
        support_replay_payload=support_payload,
        anil_inner_steps=1,
        anil_inner_learning_rate=0.25,
    )

    assert "weight" in adapted_head
    assert not torch.allclose(adapted_head["weight"].detach(), init_weight, atol=1e-6, rtol=1e-6)
    assert torch.allclose(model.policy_action_head.weight.detach(), init_weight, atol=1e-6, rtol=1e-6)
    assert torch.isfinite(torch.as_tensor(support_stats["objective"])).item()


def test_rwkv7_reinforce_sequence_replay_runs_live_rollout_under_no_grad():
    if not torch.cuda.is_available():
        return

    _ensure_torch_extensions_dir()
    _seed_everything(123)
    cfg, env_cfg = _build_small_exact_scm_env_cfg()
    env_cfg["reinforce_sequence_replay_enabled"] = True
    cfg["transformer"]["emsize"] = 64
    cfg["transformer"]["nlayers"] = 2
    cfg["transformer"]["rwkv_head_size"] = 64
    cfg["optimizer"]["rl_objective"] = "reinforce"

    prior = EnvironmentPrior(env_cfg)
    _, model, *_ = get_model(cfg, device="cuda", should_train=False, verbose=False)
    model = model.cuda().train()
    base_step_fn = _build_policy_step_fn(
        model,
        num_features=int(cfg["prior"]["num_features"]),
        max_cache_len=8,
        kv_cache_mode="immutable",
        kv_cache_page_size=None,
        allow_grad_mutable_cache=False,
        pg_torch_compile=False,
    )

    grad_enabled_history = []

    def wrapped_step_fn(*args, **kwargs):
        grad_enabled_history.append(bool(torch.is_grad_enabled()))
        return base_step_fn(*args, **kwargs)

    wrapped_step_fn._reinforce_sequence_replay_fn = base_step_fn._reinforce_sequence_replay_fn
    wrapped_step_fn._fit_action_dim_fn = getattr(base_step_fn, "_fit_action_dim_fn", None)

    _seed_everything(515151)
    h_list = prior._sample_batch_hypers(2)
    for h in h_list:
        h["reward_dropout_enabled"] = False
        h["reward_dropout_randomize"] = False
        h["reward_dropout_ratio"] = 0.0
        h["action_noise_train_std"] = 0.05
        h["action_noise_eval_std"] = 0.03

    loss, rollout, stats = prior.rollout_policy_gradient_loss(
        policy_step_fn=wrapped_step_fn,
        batch_size=2,
        n_samples=8,
        num_features=int(cfg["prior"]["num_features"]),
        device="cuda",
        single_eval_pos=4,
        collect_x=False,
        h_list_override=[dict(h) for h in h_list],
        env_seeds_override=[17, 29],
        rollout_seeds_override=[101, 211],
        policy_objective_kind="reinforce",
    )

    assert grad_enabled_history
    assert not any(grad_enabled_history)
    assert torch.isfinite(loss).item()
    assert torch.isfinite(rollout["rewards"]).all().item()
    assert int(stats.get("reinforce_sequence_replay_applied", 0)) == 1


def test_rwkv7_reinforce_sequence_replay_backward_is_finite():
    if not torch.cuda.is_available():
        return

    _ensure_torch_extensions_dir()
    loss_replay, stats_replay, grads_replay, _ = _run_rwkv7_exact_scm_chunk(
        device="cuda",
        policy_rollout_checkpoint=False,
        kv_cache_mode="immutable",
        allow_grad_mutable_cache=False,
        reinforce_sequence_replay_enabled=True,
    )

    assert torch.isfinite(loss_replay).item()
    assert torch.isfinite(stats_replay["objective"]).item()
    assert torch.isfinite(stats_replay["reward_mean"]).item()
    assert torch.isfinite(stats_replay["reward_std"]).item()
    assert int(stats_replay.get("reinforce_sequence_replay_applied", 0)) == 1
    assert grads_replay
    assert all(torch.isfinite(g).all().item() for g in grads_replay)
    assert any(float(g.abs().sum()) > 0.0 for g in grads_replay)
    for key in (
        "reinforce_action_dim_mean",
        "reinforce_action_dim_min",
        "reinforce_action_dim_max",
        "reinforce_action_std_mean",
        "reinforce_action_std_min",
        "reinforce_action_std_max",
        "reinforce_logprob_log_std_mean",
        "reinforce_logprob_log_std_min",
        "reinforce_logprob_log_std_max",
        "reinforce_logprob_z2_mean",
        "reinforce_logprob_z2_max",
    ):
        assert key in stats_replay, key
        assert torch.isfinite(torch.as_tensor(stats_replay[key])).item(), key


def test_rwkv7_reinforce_sequence_replay_preserves_rollout_under_fixed_seeds():
    if not torch.cuda.is_available():
        return

    base = _run_rwkv7_exact_scm_reinforce_rollout(
        device="cuda",
        reinforce_sequence_replay_enabled=False,
    )
    replay = _run_rwkv7_exact_scm_reinforce_rollout(
        device="cuda",
        reinforce_sequence_replay_enabled=True,
    )

    assert torch.equal(base["x"], replay["x"])
    assert torch.equal(base["rewards"], replay["rewards"])
    eval_start = int(base["single_eval_pos"])
    assert torch.equal(base["log_probs"][eval_start:], replay["log_probs"][eval_start:])
    for key in ("objective", "reward_mean", "reward_std"):
        assert torch.equal(base["stats"][key], replay["stats"][key]), key
    assert replay["sequence_replay_applied"]
    assert torch.isfinite(replay["log_probs"]).all().item()


def test_rwkv7_reinforce_sequence_replay_keeps_only_eval_suffix_aux_tensors():
    if not torch.cuda.is_available():
        return

    replay = _run_rwkv7_exact_scm_reinforce_rollout(
        device="cuda",
        reinforce_sequence_replay_enabled=True,
    )

    assert replay["sequence_replay_applied"]
    assert replay["replay_eval_start"] == replay["single_eval_pos"]
    assert replay["replay_action_steps"] == int(replay["rewards"].shape[0]) - int(replay["single_eval_pos"])


def test_rwkv7_default_config_enables_sequence_replay_checkpoint():
    cfg = _build_rwkv7_rlpfn_config()
    assert cfg["transformer"]["rwkv_sequence_replay_checkpoint"] is True
    assert cfg["transformer"]["rwkv_sequence_replay_batch_chunk_size"] == 64
    assert cfg["transformer"]["rwkv_sequence_replay_token_budget"] == 262144


def test_rwkv7_default_model_enables_sequence_replay_checkpoint():
    if not torch.cuda.is_available():
        return

    _ensure_torch_extensions_dir()
    cfg = _build_rwkv7_rlpfn_config()
    _, model, *_ = get_model(cfg, device="cuda", should_train=False, verbose=False)
    assert bool(model.rwkv_core.sequence_replay_checkpoint) is True
    assert int(model.rwkv_sequence_replay_batch_chunk_size) == 64
    assert int(model.rwkv_sequence_replay_token_budget) == 262144


def test_rwkv7_default_replay_chunk_resolver_scales_with_sequence_length():
    if not torch.cuda.is_available():
        return

    _ensure_torch_extensions_dir()
    cfg = _build_rwkv7_rlpfn_config()
    _, model, *_ = get_model(cfg, device="cuda", should_train=False, verbose=False)
    assert int(model.resolve_replay_batch_chunk_size(seq_len=512, total_batch=1024)) == 64
    assert int(model.resolve_replay_batch_chunk_size(seq_len=4096, total_batch=1024)) == 32


def test_rwkv7_cuda_stepwise_core_matches_official_sequence_core():
    if not torch.cuda.is_available():
        return

    _ensure_torch_extensions_dir()
    _seed_everything(123)
    cfg = _build_rwkv7_rlpfn_config()
    cfg["transformer"]["emsize"] = 64
    cfg["transformer"]["nlayers"] = 2
    cfg["transformer"]["rwkv_head_size"] = 64
    _, model, *_ = get_model(cfg, device="cuda", should_train=False, verbose=False)
    model = model.cuda().train()

    seq_len = 8
    batch_size = 2
    num_features = int(cfg["prior"]["num_features"])
    x = torch.randn(seq_len, batch_size, num_features, device="cuda")
    y = torch.randn(seq_len, batch_size, device="cuda")

    tokens = model._encode_train_token(x, y)
    tokens = model._cast_token_for_rwkv_core(tokens)
    hidden_seq = model.rwkv_core.forward_tokens_sequence_only(tokens)
    hidden_step, _ = model.rwkv_core.forward_tokens(tokens, None)

    assert torch.equal(hidden_seq, hidden_step)


def test_rwkv7_reinforce_sequence_replay_gradients_match_stepwise_reinforce():
    if not torch.cuda.is_available():
        return

    _ensure_torch_extensions_dir()
    loss_base, stats_base, grads_base, _ = _run_rwkv7_exact_scm_chunk(
        device="cuda",
        policy_rollout_checkpoint=False,
        kv_cache_mode="immutable",
        allow_grad_mutable_cache=False,
        reinforce_sequence_replay_enabled=False,
    )
    loss_replay, stats_replay, grads_replay, _ = _run_rwkv7_exact_scm_chunk(
        device="cuda",
        policy_rollout_checkpoint=False,
        kv_cache_mode="immutable",
        allow_grad_mutable_cache=False,
        reinforce_sequence_replay_enabled=True,
    )

    assert torch.equal(loss_base, loss_replay)
    for key in ("objective", "reward_mean", "reward_std"):
        assert torch.equal(stats_base[key], stats_replay[key]), key
    assert len(grads_base) == len(grads_replay)

    max_abs_diff = 0.0
    mean_abs_diff = 0.0
    for grad_base, grad_replay in zip(grads_base, grads_replay):
        diff = (grad_base - grad_replay).abs()
        max_abs_diff = max(max_abs_diff, float(diff.max()))
        mean_abs_diff += float(diff.mean())
    mean_abs_diff /= float(max(1, len(grads_base)))
    assert max_abs_diff <= 6e-3
    assert mean_abs_diff <= 5e-6


def test_rwkv7_reinforce_sequence_replay_checkpoint_matches_no_checkpoint():
    if not torch.cuda.is_available():
        return

    _ensure_torch_extensions_dir()
    loss_base, stats_base, grads_base, _ = _run_rwkv7_exact_scm_chunk(
        device="cuda",
        policy_rollout_checkpoint=False,
        kv_cache_mode="immutable",
        allow_grad_mutable_cache=False,
        reinforce_sequence_replay_enabled=True,
        rwkv_sequence_replay_checkpoint=False,
    )
    loss_ckpt, stats_ckpt, grads_ckpt, _ = _run_rwkv7_exact_scm_chunk(
        device="cuda",
        policy_rollout_checkpoint=False,
        kv_cache_mode="immutable",
        allow_grad_mutable_cache=False,
        reinforce_sequence_replay_enabled=True,
        rwkv_sequence_replay_checkpoint=True,
    )

    assert torch.equal(loss_base, loss_ckpt)
    for key in ("objective", "reward_mean", "reward_std"):
        assert torch.equal(stats_base[key], stats_ckpt[key]), key
    assert len(grads_base) == len(grads_ckpt)

    max_abs_diff = 0.0
    mean_abs_diff = 0.0
    for grad_base, grad_ckpt in zip(grads_base, grads_ckpt):
        diff = (grad_base - grad_ckpt).abs()
        max_abs_diff = max(max_abs_diff, float(diff.max()))
        mean_abs_diff += float(diff.mean())
    mean_abs_diff /= float(max(1, len(grads_base)))
    assert max_abs_diff <= 6e-3
    assert mean_abs_diff <= 5e-6


def test_rwkv7_reinforce_sequence_replay_batch_microbatch_matches_full_batch():
    if not torch.cuda.is_available():
        return

    _ensure_torch_extensions_dir()
    loss_full, stats_full, grads_full, _ = _run_rwkv7_exact_scm_chunk(
        device="cuda",
        policy_rollout_checkpoint=False,
        kv_cache_mode="immutable",
        allow_grad_mutable_cache=False,
        reinforce_sequence_replay_enabled=True,
        rwkv_sequence_replay_batch_chunk_size=None,
    )
    loss_micro, stats_micro, grads_micro, _ = _run_rwkv7_exact_scm_chunk(
        device="cuda",
        policy_rollout_checkpoint=False,
        kv_cache_mode="immutable",
        allow_grad_mutable_cache=False,
        reinforce_sequence_replay_enabled=True,
        rwkv_sequence_replay_batch_chunk_size=1,
    )

    assert torch.equal(loss_full, loss_micro)
    for key in ("objective", "reward_mean", "reward_std"):
        assert torch.equal(stats_full[key], stats_micro[key]), key
    assert len(grads_full) == len(grads_micro)

    max_abs_diff = 0.0
    mean_abs_diff = 0.0
    for grad_full, grad_micro in zip(grads_full, grads_micro):
        diff = (grad_full - grad_micro).abs()
        max_abs_diff = max(max_abs_diff, float(diff.max()))
        mean_abs_diff += float(diff.mean())
    mean_abs_diff /= float(max(1, len(grads_full)))
    assert max_abs_diff <= 6e-3
    assert mean_abs_diff <= 5e-6


def test_rwkv7_reinforce_sequence_replay_loss_sink_matches_no_sink():
    if not torch.cuda.is_available():
        return

    _ensure_torch_extensions_dir()
    loss_base, stats_base, grads_base, sink_calls_base = _run_rwkv7_exact_scm_chunk(
        device="cuda",
        policy_rollout_checkpoint=False,
        kv_cache_mode="immutable",
        allow_grad_mutable_cache=False,
        reinforce_sequence_replay_enabled=True,
        rwkv_sequence_replay_batch_chunk_size=1,
        use_reinforce_replay_loss_sink=False,
    )
    loss_sink, stats_sink, grads_sink, sink_calls_stream = _run_rwkv7_exact_scm_chunk(
        device="cuda",
        policy_rollout_checkpoint=False,
        kv_cache_mode="immutable",
        allow_grad_mutable_cache=False,
        reinforce_sequence_replay_enabled=True,
        rwkv_sequence_replay_batch_chunk_size=1,
        use_reinforce_replay_loss_sink=True,
    )

    assert torch.equal(loss_base, loss_sink)
    for key in ("objective", "reward_mean", "reward_std"):
        assert torch.equal(stats_base[key], stats_sink[key]), key
    assert len(grads_base) == len(grads_sink)
    assert sink_calls_base == 0
    assert sink_calls_stream > 1

    max_abs_diff = 0.0
    mean_abs_diff = 0.0
    for grad_base, grad_sink in zip(grads_base, grads_sink):
        diff = (grad_base - grad_sink).abs()
        max_abs_diff = max(max_abs_diff, float(diff.max()))
        mean_abs_diff += float(diff.mean())
    mean_abs_diff /= float(max(1, len(grads_base)))
    assert max_abs_diff <= 6e-3
    assert mean_abs_diff <= 5e-6


def test_rwkv7_reinforce_sequence_replay_loss_sink_works_without_collect_x():
    if not torch.cuda.is_available():
        return

    _ensure_torch_extensions_dir()
    loss_sink, stats_sink, grads_sink, sink_calls_stream = _run_rwkv7_exact_scm_chunk(
        device="cuda",
        policy_rollout_checkpoint=False,
        kv_cache_mode="immutable",
        allow_grad_mutable_cache=False,
        reinforce_sequence_replay_enabled=True,
        rwkv_sequence_replay_batch_chunk_size=1,
        use_reinforce_replay_loss_sink=True,
        collect_x_override=False,
    )

    assert torch.isfinite(loss_sink).item()
    for key in ("objective", "reward_mean", "reward_std"):
        assert torch.isfinite(stats_sink[key]).item(), key
    assert grads_sink
    assert sink_calls_stream > 1


def test_rwkv7_default_sequence_replay_path_invokes_checkpoint(monkeypatch):
    if not torch.cuda.is_available():
        return

    call_count = {"n": 0}
    original_checkpoint = torch.utils.checkpoint.checkpoint

    def _wrapped_checkpoint(function, *args, **kwargs):
        call_count["n"] += 1
        return original_checkpoint(function, *args, **kwargs)

    monkeypatch.setattr(torch.utils.checkpoint, "checkpoint", _wrapped_checkpoint)
    loss, stats, grads, _ = _run_rwkv7_exact_scm_chunk(
        device="cuda",
        policy_rollout_checkpoint=False,
        kv_cache_mode="immutable",
        allow_grad_mutable_cache=False,
        reinforce_sequence_replay_enabled=True,
        rwkv_sequence_replay_checkpoint=None,
    )

    assert torch.isfinite(loss).item()
    assert torch.isfinite(stats["objective"]).item()
    assert grads
    assert int(call_count["n"]) > 0


def test_rwkv7_rlpfn_official_cuda_sequence_forward_runs():
    if not torch.cuda.is_available():
        return

    _ensure_torch_extensions_dir()
    _seed_everything(321)
    cfg = _build_rwkv7_rlpfn_config()
    cfg["transformer"]["emsize"] = 64
    cfg["transformer"]["nlayers"] = 2
    cfg["transformer"]["rwkv_head_size"] = 64
    _, model, *_ = get_model(cfg, device="cuda", should_train=False, verbose=False)
    model = model.cuda().eval()

    seq_len = 15
    batch_size = 2
    num_features = int(cfg["prior"]["num_features"])
    x = torch.randn(seq_len, batch_size, num_features, device="cuda")
    y = torch.randn(seq_len, batch_size, device="cuda")

    out = model((x, y), single_eval_pos=7)
    assert out.is_cuda
    assert tuple(out.shape) == (seq_len - 7, batch_size, int(model.policy_action_dim))


def test_rwkv7_rlpfn_validation_uses_split_policy_step_state_cache(monkeypatch):
    _ensure_torch_extensions_dir()
    _install_fake_gym(
        monkeypatch,
        lambda env_name: _ScriptedEnv([2, 2], [1.0, 1.0]),
    )
    cfg = _build_rwkv7_rlpfn_config()
    cfg["transformer"]["emsize"] = 64
    cfg["transformer"]["nlayers"] = 2
    cfg["transformer"]["rwkv_head_size"] = 64
    _, model, *_ = get_model(cfg, device="cpu", should_train=False, verbose=False)
    model.eval()

    import ticl.rl_validation as rl_validation_mod

    def _forbid_old_scoring(*args, **kwargs):
        raise AssertionError("RWKV validation must not fall back to legacy candidate scoring")

    monkeypatch.setattr(rl_validation_mod, "_score_candidate_action_jobs", _forbid_old_scoring)

    split_calls = {"count": 0}
    split_cache_seen = []

    original_split = model.forward_policy_step_split

    def _wrapped_split(
        self,
        obs_t,
        action_t,
        reward_t,
        reward_mask_t,
        phase_t=None,
        terminal_t=None,
        kv_cache=None,
        max_cache_len=None,
        kv_cache_mode="auto",
        kv_cache_page_size=None,
        allow_grad_mutable_cache=False,
        allow_grad_inplace_paged_cache=False,
    ):
        split_calls["count"] += 1
        split_cache_seen.append(kv_cache is not None)
        return original_split(
            obs_t,
            action_t,
            reward_t,
            reward_mask_t,
            phase_t=phase_t,
            terminal_t=terminal_t,
            kv_cache=kv_cache,
            max_cache_len=max_cache_len,
            kv_cache_mode=kv_cache_mode,
            kv_cache_page_size=kv_cache_page_size,
            allow_grad_mutable_cache=allow_grad_mutable_cache,
            allow_grad_inplace_paged_cache=allow_grad_inplace_paged_cache,
        )

    def _forbid_generic(
        self,
        x_token,
        y_token,
        kv_cache=None,
        max_cache_len=None,
        kv_cache_mode="auto",
        kv_cache_page_size=None,
        allow_grad_mutable_cache=False,
        allow_grad_inplace_paged_cache=False,
    ):
        raise AssertionError("RWKV validation should use split policy-step fastpath, not generic policy_step")

    monkeypatch.setattr(model, "forward_policy_step_split", types.MethodType(_wrapped_split, model))
    monkeypatch.setattr(model, "forward_policy_step", types.MethodType(_forbid_generic, model))

    val_cfg = {
        "device": "cpu",
        "prior": {
            "num_features": int(cfg["prior"]["num_features"]),
            "environment": {
                "obs_slot_dim": 400,
                "action_slot_dim": 30,
                "terminal_reset_enabled": True,
                "init_action_std": 0.0,
                "action_noise_train_std": 0.0,
                "action_noise_eval_std": 0.0,
                "reinforce_action_transform": "none",
                "reinforce_reward_transform": "none",
            },
        },
        "optimizer": {
            "pg_kv_cache_mode": "auto",
            "pg_kv_cache_page_size": None,
        },
        "orchestration": {
            "rl_validate_envs": "DummyEnv-vRWKV",
            "rl_validate_episodes": 1,
            "rl_validate_max_steps": 8,
            "rl_validate_action_candidates": 1,
            "rl_validate_seed": 1,
            "rl_validate_context_lower_bound": 1,
        },
    }

    mean_ret, per_env = evaluate_rlpfn_on_gym_envs(model=model, config=val_cfg)

    assert np.isfinite(mean_ret)
    assert per_env["DummyEnv-vRWKV"]["return"] == 2.0
    assert split_calls["count"] > 0
    assert split_cache_seen[0] is False
    assert any(split_cache_seen[1:])


def test_rwkv7_reinforce_current_accel_stack_matches_noaccel_strictly():
    loss_base, stats_base, grads_base, _ = _run_rwkv7_exact_scm_chunk(
        policy_rollout_checkpoint=False,
        kv_cache_mode="immutable",
        allow_grad_mutable_cache=False,
    )
    loss_paged, stats_paged, grads_paged, _ = _run_rwkv7_exact_scm_chunk(
        policy_rollout_checkpoint=False,
        kv_cache_mode="paged",
        allow_grad_mutable_cache=True,
    )
    loss_ckpt, stats_ckpt, grads_ckpt, _ = _run_rwkv7_exact_scm_chunk(
        policy_rollout_checkpoint=True,
        policy_rollout_checkpoint_reentrant=True,
        kv_cache_mode="paged",
        allow_grad_mutable_cache=True,
    )

    assert torch.equal(loss_base, loss_paged)
    assert torch.equal(loss_base, loss_ckpt)
    for key in ("objective", "reward_mean", "reward_std"):
        assert torch.equal(stats_base[key], stats_paged[key]), key
        assert torch.equal(stats_base[key], stats_ckpt[key]), key
    assert len(grads_base) == len(grads_paged) == len(grads_ckpt)
    assert all(torch.equal(g_base, g_paged) for g_base, g_paged in zip(grads_base, grads_paged))
    assert all(torch.equal(g_base, g_ckpt) for g_base, g_ckpt in zip(grads_base, grads_ckpt))


def test_rwkv7_reinforce_tbptt_streaming_matches_buffered_gradients():
    loss_buffered, stats_buffered, grads_buffered, _ = _run_rwkv7_exact_scm_chunk(
        policy_rollout_checkpoint=False,
        kv_cache_mode="paged",
        allow_grad_mutable_cache=True,
        n_samples=12,
        single_eval_pos=4,
        pg_tbptt_window=4,
        tbptt_loss_sink=None,
    )

    streamed_roots = []
    _, stats_streamed, grads_streamed, _ = _run_rwkv7_exact_scm_chunk(
        policy_rollout_checkpoint=False,
        kv_cache_mode="paged",
        allow_grad_mutable_cache=True,
        n_samples=12,
        single_eval_pos=4,
        pg_tbptt_window=4,
        tbptt_loss_sink=lambda loss_root: (streamed_roots.append(float(loss_root.detach())), loss_root.backward()),
    )

    assert torch.isfinite(loss_buffered)
    assert len(streamed_roots) > 1
    for key in ("objective", "reward_mean", "reward_std"):
        assert torch.equal(stats_buffered[key], stats_streamed[key]), key
    assert len(grads_buffered) == len(grads_streamed)
    max_grad_diff = max(float((g_buf - g_stream).abs().max()) for g_buf, g_stream in zip(grads_buffered, grads_streamed))
    assert max_grad_diff < 1e-6


@pytest.mark.skipif(not torch.cuda.is_available(), reason="ANIL RWKV replay path currently requires CUDA")
def test_rwkv7_anil_support_query_share_task_but_use_distinct_rollout_seeds(monkeypatch):
    _ensure_torch_extensions_dir()
    _seed_everything(321)
    cfg, env_cfg = _build_small_exact_scm_env_cfg()
    cfg["optimizer"]["rl_objective"] = "anil"
    cfg["transformer"]["emsize"] = 64
    cfg["transformer"]["nlayers"] = 2
    cfg["transformer"]["rwkv_head_size"] = 64
    prior = EnvironmentPrior(env_cfg)
    _, model, *_ = get_model(cfg, device="cuda", should_train=False, verbose=False)
    model = model.cuda().train()
    step_fn = _build_policy_step_fn(
        model,
        num_features=int(cfg["prior"]["num_features"]),
        max_cache_len=6,
        kv_cache_mode="immutable",
        kv_cache_page_size=None,
        allow_grad_mutable_cache=False,
        pg_torch_compile=False,
    )

    _seed_everything(515151)
    h_list = prior._sample_batch_hypers(2)
    for h in h_list:
        h["reward_dropout_enabled"] = False
        h["reward_dropout_randomize"] = False
        h["reward_dropout_ratio"] = 0.0
        h["action_noise_train_std"] = 0.05
        h["action_noise_eval_std"] = 0.03

    real_collect = train_mod._collect_anil_replay_rollout
    recorded = []

    def _recording_collect(*args, **kwargs):
        recorded.append(
            {
                "h_id": id(kwargs["h"]),
                "env_seed": int(kwargs["env_seed"]),
                "rollout_seed": int(kwargs["rollout_seed"]),
            }
        )
        return real_collect(*args, **kwargs)

    monkeypatch.setattr(train_mod, "_collect_anil_replay_rollout", _recording_collect)
    model.zero_grad(set_to_none=True)
    loss, _, stats = _compute_policy_rollout_chunk_loss(
        env_prior=prior,
        policy_step_fn=step_fn,
        batch_size=2,
        n_samples=6,
        num_features=int(cfg["prior"]["num_features"]),
        device="cuda",
        single_eval_pos=3,
        collect_x=False,
        policy_rollout_checkpoint=False,
        policy_rollout_checkpoint_reentrant=True,
        pg_saved_tensors_cpu_offload=False,
        pg_saved_tensors_pin_memory=True,
        pg_tbptt_window=None,
        h_list_override=[dict(h) for h in h_list],
        env_seeds_override=[17, 29],
        rollout_seeds_override=[101, 211],
        rl_objective="anil",
        anil_inner_steps=1,
        anil_inner_learning_rate=0.1,
    )
    loss.backward()

    assert torch.isfinite(loss).item()
    assert torch.isfinite(stats["objective"]).item()
    assert int(stats["anil_task_count"]) == 2
    assert len(recorded) == 4
    for task_idx in range(2):
        support_meta = recorded[2 * task_idx]
        query_meta = recorded[2 * task_idx + 1]
        assert support_meta["h_id"] == query_meta["h_id"]
        assert support_meta["env_seed"] == query_meta["env_seed"]
        assert support_meta["rollout_seed"] != query_meta["rollout_seed"]
        assert query_meta["rollout_seed"] == train_mod._derive_secondary_rollout_seed(
            support_meta["rollout_seed"]
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="ANIL RWKV replay path currently requires CUDA")
def test_rwkv7_anil_query_loss_sink_matches_no_sink():
    _ensure_torch_extensions_dir()
    _seed_everything(654)
    cfg, env_cfg = _build_small_exact_scm_env_cfg()
    cfg["optimizer"]["rl_objective"] = "anil"
    cfg["transformer"]["emsize"] = 64
    cfg["transformer"]["nlayers"] = 2
    cfg["transformer"]["rwkv_head_size"] = 64

    def _run_anil(sink_enabled: bool):
        prior = EnvironmentPrior(deepcopy(env_cfg))
        _, model, *_ = get_model(cfg, device="cuda", should_train=False, verbose=False)
        model = model.cuda().train()
        step_fn = _build_policy_step_fn(
            model,
            num_features=int(cfg["prior"]["num_features"]),
            max_cache_len=6,
            kv_cache_mode="immutable",
            kv_cache_page_size=None,
            allow_grad_mutable_cache=False,
            pg_torch_compile=False,
        )
        _seed_everything(919191)
        h_list = prior._sample_batch_hypers(1)
        for h in h_list:
            h["reward_dropout_enabled"] = False
            h["reward_dropout_randomize"] = False
            h["reward_dropout_ratio"] = 0.0
            h["action_noise_train_std"] = 0.05
            h["action_noise_eval_std"] = 0.03
        model.zero_grad(set_to_none=True)
        sink_calls = {"n": 0}

        def _sink(loss_root):
            sink_calls["n"] += 1
            loss_root.backward()

        loss, _, stats = _compute_policy_rollout_chunk_loss(
            env_prior=prior,
            policy_step_fn=step_fn,
            batch_size=1,
            n_samples=6,
            num_features=int(cfg["prior"]["num_features"]),
            device="cuda",
            single_eval_pos=3,
            collect_x=False,
            policy_rollout_checkpoint=False,
            policy_rollout_checkpoint_reentrant=True,
            pg_saved_tensors_cpu_offload=False,
            pg_saved_tensors_pin_memory=True,
            pg_tbptt_window=None,
            h_list_override=[dict(h) for h in h_list],
            env_seeds_override=[17],
            rollout_seeds_override=[101],
            rl_objective="anil",
            anil_inner_steps=1,
            anil_inner_learning_rate=0.1,
            anil_query_loss_sink=_sink if sink_enabled else None,
        )
        if (not sink_enabled) and bool(loss.requires_grad):
            loss.backward()
        grads = [p.grad.detach().clone() for p in model.parameters() if p.grad is not None]
        return loss.detach().clone(), stats, grads, int(sink_calls["n"])

    loss_base, stats_base, grads_base, sink_calls_base = _run_anil(sink_enabled=False)
    loss_sink, stats_sink, grads_sink, sink_calls_sink = _run_anil(sink_enabled=True)

    assert sink_calls_base == 0
    assert sink_calls_sink == 1
    assert torch.allclose(loss_base, loss_sink, atol=1e-5, rtol=1e-5)
    for key in ("objective", "reward_mean", "reward_std", "anil_support_objective"):
        assert torch.allclose(stats_base[key], stats_sink[key], atol=1e-5, rtol=1e-5), key
    assert len(grads_base) == len(grads_sink)
    max_grad_diff = max(float((g_base - g_sink).abs().max()) for g_base, g_sink in zip(grads_base, grads_sink))
    assert max_grad_diff < 1e-5
