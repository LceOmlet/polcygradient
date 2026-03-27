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
from ticl.priors.maintained_fast_runner import dispatch_policy_rollout
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
            "ctrl_reward_weight": 0.0,
            "ctrl_reward_enable_prob": 0.0,
            "survival_reward_weight": 0.0,
            "survival_reward_enable_prob": 0.0,
            "action_noise_train_std": 0.05,
            "action_noise_eval_std": 0.03,
            "reinforce_sequence_replay_enabled": False,
            "normalized_q_value_weight": 0.0,
            "next_state_flow_matching_weight": 0.0,
            "reinforce_sequence_replay_share_context_forward": False,
        }
    )
    return cfg, env_cfg


def _resolve_dim_upper_bound(spec):
    if isinstance(spec, dict):
        if "max" in spec:
            return int(spec["max"])
        choice_values = spec.get("choice_values", None)
        if isinstance(choice_values, (list, tuple)) and len(choice_values) > 0:
            return int(max(choice_values))
        if "value" in spec:
            return int(spec["value"])
    return int(spec)


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


def test_rwkv7_rlpfn_builder_has_correct_action_head_and_default_budget():
    _ensure_torch_extensions_dir()
    cfg = _build_rwkv7_rlpfn_config()
    _, model, *_ = get_model(cfg, device="cpu", should_train=False, verbose=False)

    param_count = sum(p.numel() for p in model.parameters())
    assert type(model).__name__ == "RWKV7PFN"
    assert 30_000_000 <= param_count <= 45_000_000
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
    assert "mean.bias" in override
    override["mean.bias"] = torch.full_like(override["mean.bias"], 0.5)
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

    assert isinstance(base_out, dict)
    assert isinstance(override_out, dict)
    assert torch.allclose(base_out["action_mean"], base_out_again["action_mean"], atol=1e-6, rtol=1e-6)
    assert not torch.allclose(base_out["action_mean"], override_out["action_mean"], atol=1e-6, rtol=1e-6)


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
    batch_size = 1
    eval_start = 3
    flow_dim = _resolve_dim_upper_bound(cfg["prior"]["environment"]["state_dim"])
    with torch.no_grad():
        outputs = model.replay_policy_sequence_outputs(
            torch.randn(seq_len, batch_size, int(cfg["prior"]["num_features"]), device="cuda"),
            torch.randn(seq_len, batch_size, device="cuda"),
            eval_start=eval_start,
            flow_matching_xt=torch.randn(seq_len - eval_start, batch_size, flow_dim, device="cuda"),
            flow_matching_t=torch.rand(seq_len - eval_start, batch_size, 1, device="cuda"),
        )

    assert {"action_mean", "normalized_q", "normalized_q_logits", "next_state_flow"}.issubset(outputs.keys())
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
    batch_size = 1
    eval_start = 2
    flow_dim = _resolve_dim_upper_bound(cfg["prior"]["environment"]["state_dim"])
    with torch.no_grad():
        outputs = model.replay_policy_sequence_outputs(
            torch.randn(seq_len, batch_size, int(cfg["prior"]["num_features"])),
            torch.randn(seq_len, batch_size),
            eval_start=eval_start,
            flow_matching_xt=torch.randn(seq_len - eval_start, batch_size, flow_dim),
            flow_matching_t=torch.rand(seq_len - eval_start, batch_size, 1),
        )

    assert {"action_mean", "normalized_q", "normalized_q_logits", "next_state_flow"}.issubset(outputs.keys())
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


def test_model_builder_uses_full_state_dim_for_next_state_flow_head():
    _seed_everything(6551)
    cfg = deepcopy(get_model_default_config("rlpfn"))
    cfg["transformer"]["backbone"] = "transformer"
    cfg["transformer"]["emsize"] = 64
    cfg["transformer"]["nlayers"] = 2
    cfg["transformer"]["nhead"] = 4
    cfg["prior"]["environment"]["state_dim"] = {"distribution": "uniform_int", "min": 11, "max": 11}
    cfg["prior"]["environment"]["obs_dim"] = {"distribution": "uniform_int", "min": 5, "max": 5}
    cfg["prior"]["environment"]["obs_slot_dim"] = 5
    cfg["prior"]["environment"]["next_state_flow_matching_weight"] = 0.5
    cfg["prior"]["environment"]["next_state_flow_head_type"] = "cfmi_resnet"
    _, model, *_ = get_model(cfg, device="cpu", should_train=False, verbose=False)
    assert int(model.next_state_flow_dim) == 11


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
    replay_output_calls = []
    q_query_calls = []
    flow_query_calls = []

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

    def _replay_outputs(
        x_tokens,
        y_tokens,
        *,
        eval_start=0,
        flow_matching_xt=None,
        flow_matching_t=None,
    ):
        del y_tokens, flow_matching_xt, flow_matching_t
        replay_output_calls.append(
            {
                "seq_len": int(x_tokens.shape[0]),
                "query_len": int(x_tokens.shape[0] - eval_start),
            }
        )
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
        return outputs

    def _init_kv_cache(x_tokens, y_tokens):
        del y_tokens
        return int(x_tokens.shape[0])

    def _append_train_token_to_kv(x_token, y_token, kv_cache):
        del y_token
        if kv_cache is None:
            kv_cache = 0
        return int(kv_cache) + int(x_token.shape[0])

    def _predict_query_replay_outputs_with_kv(
        x_query,
        kv_cache,
        *,
        action_query=None,
        flow_matching_xt=None,
        flow_matching_t=None,
    ):
        del action_query, flow_matching_t
        call_info = {
            "prefix_len": int(kv_cache),
            "flow_shape": None if flow_matching_xt is None else tuple(flow_matching_xt.shape),
            "query_cols": [
                tuple(torch.nonzero(x_query[0, batch_idx], as_tuple=False).reshape(-1).tolist())
                for batch_idx in range(int(x_query.shape[1]))
            ],
        }
        if flow_matching_xt is None:
            q_query_calls.append(call_info)
        else:
            flow_query_calls.append(call_info)
        outputs = {
            "normalized_q_logits": torch.zeros(
                1,
                int(x_query.shape[1]),
                bardist.num_bars,
                dtype=torch.float32,
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
    _dummy_step_fn._reinforce_sequence_init_kv_cache_fn = _init_kv_cache
    _dummy_step_fn._reinforce_sequence_append_train_token_to_kv_fn = _append_train_token_to_kv
    _dummy_step_fn._reinforce_sequence_predict_query_outputs_with_kv_fn = (
        _predict_query_replay_outputs_with_kv
    )
    _dummy_step_fn._fit_action_dim_fn = lambda x, action_dim: x[..., :action_dim]
    _dummy_step_fn._model_ref = _DummyReplayModel()

    seq_len = 5
    eval_start = 2
    batch_size = 2
    state_dim = _resolve_dim_upper_bound(env_cfg["state_dim"])
    action_dim = int(env_cfg["action_slot_dim"])
    rewards = torch.randn(seq_len, batch_size, dtype=torch.float32)
    sampled_action_eval = torch.tensor(
        [
            [[1.0, 2.0, 3.0], [4.0, 5.0, 0.0]],
            [[6.0, 7.0, 8.0], [9.0, 10.0, 0.0]],
            [[11.0, 12.0, 13.0], [14.0, 15.0, 0.0]],
        ],
        dtype=torch.float32,
    )
    flow_action_prefix = torch.tensor(
        [
            [[21.0, 22.0, 23.0], [24.0, 25.0, 0.0]],
            [[31.0, 32.0, 33.0], [34.0, 35.0, 0.0]],
        ],
        dtype=torch.float32,
    )
    action_query_cols = torch.tensor(
        [
            [7, 8, 9],
            [6, 7, -1],
        ],
        dtype=torch.long,
    )
    replay_payload = {
        "reward_in": torch.randn(seq_len, batch_size, dtype=torch.float32),
        "sampled_action": sampled_action_eval,
        "action_std": torch.full((seq_len - eval_start, batch_size), 0.05, dtype=torch.float32),
        "action_mask": torch.ones(batch_size, action_dim, dtype=torch.bool),
        "action_query_cols": action_query_cols,
        "next_state": torch.randn(seq_len - eval_start, batch_size, state_dim, dtype=torch.float32),
        "state_mask": torch.ones(batch_size, state_dim, dtype=torch.bool),
        "flow_action": flow_action_prefix,
        "flow_next_state": torch.randn(eval_start, batch_size, state_dim, dtype=torch.float32),
        "flow_state_mask": torch.ones(batch_size, state_dim, dtype=torch.bool),
        "flow_query_start": 0,
        "flow_query_stop": eval_start,
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
    assert len(replay_output_calls) == 1
    assert all(call["seq_len"] == seq_len for call in replay_output_calls)
    assert all(call["query_len"] == (seq_len - eval_start) for call in replay_output_calls)
    assert len(q_query_calls) == (seq_len - eval_start)
    assert [call["prefix_len"] for call in q_query_calls] == [eval_start + 1, eval_start + 2, eval_start + 3]
    assert len(flow_query_calls) == eval_start
    assert [call["prefix_len"] for call in flow_query_calls] == [1, 2]
    assert [call["query_cols"] for call in q_query_calls] == [[(7, 8, 9), (6, 7)]] * (seq_len - eval_start)
    assert [call["query_cols"] for call in flow_query_calls] == [[(7, 8, 9), (6, 7)]] * eval_start
    assert all(
        call["flow_shape"] == (1, 2, state_dim)
        for call in flow_query_calls
    )
    assert int(stats["reinforce_sequence_replay_flow_steps"]) == eval_start
    assert int(stats["reinforce_sequence_replay_policy_aux_shared_context_forward"]) == 0
    assert int(stats["reinforce_sequence_replay_aux_query_pass_enabled"]) == 1
    assert int(stats["reinforce_sequence_replay_q_flow_shared_aux_query_pass"]) == 1
    assert "reinforce_sequence_replay_shared_backbone_pass" not in stats
    assert "reinforce_sequence_replay_shared_aux_backbone_pass" not in stats


def test_reinforce_sequence_replay_loss_sink_stages_policy_q_and_flow():
    _seed_everything(9992)
    _, env_cfg = _build_small_exact_scm_env_cfg()
    env_cfg["policy_gradient_weight"] = 0.1
    env_cfg["normalized_q_value_weight"] = 0.25
    env_cfg["next_state_flow_matching_weight"] = 0.5
    env_cfg["obs_slot_dim"] = 8
    env_cfg["action_slot_dim"] = 3
    prior = EnvironmentPrior(env_cfg)
    bardist = make_standardized_full_support_bar_distribution()
    event_log = []

    class _DummyReplayModel:
        def resolve_replay_batch_chunk_size(self, *, seq_len, total_batch):
            del seq_len
            return int(total_batch)

        def get_normalized_q_value_bardist(self):
            return bardist

    def _replay_outputs(
        x_tokens,
        y_tokens,
        *,
        eval_start=0,
        flow_matching_xt=None,
        flow_matching_t=None,
    ):
        del y_tokens, flow_matching_xt, flow_matching_t
        event_log.append("policy_forward")
        query = x_tokens[eval_start:]
        base = query[..., :3]
        return {
            "action_mean": base,
        }

    def _predict_query_replay_outputs_with_kv(
        x_query,
        kv_cache,
        *,
        action_query=None,
        flow_matching_xt=None,
        flow_matching_t=None,
    ):
        q_steps = int(x_query.shape[0])
        q_batch = int(x_query.shape[1])
        del x_query, action_query, flow_matching_t
        hidden = kv_cache.unsqueeze(0)
        outputs = {}
        if flow_matching_xt is None:
            event_log.append("q_forward")
            logits = hidden.expand(q_steps, q_batch, bardist.num_bars)
            outputs["normalized_q_logits"] = logits
            outputs["normalized_q"] = bardist.mean(logits)
        else:
            event_log.append("flow_forward")
            outputs["next_state_flow"] = hidden.expand_as(flow_matching_xt)
        return outputs

    def _dummy_step_fn(*args, **kwargs):
        raise AssertionError("loss-sink staging test should not call live policy_step")

    _dummy_step_fn._reinforce_sequence_replay_fn = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("loss-sink staging test should use replay outputs path")
    )
    _dummy_step_fn._reinforce_sequence_replay_outputs_fn = _replay_outputs
    _dummy_step_fn._reinforce_sequence_init_kv_cache_fn = (
        lambda x_tokens, y_tokens: x_tokens.sum(dim=0, keepdim=False).sum(dim=-1, keepdim=True)
    )
    _dummy_step_fn._reinforce_sequence_append_train_token_to_kv_fn = (
        lambda x_token, y_token, kv_cache: (
            x_token.sum(dim=0, keepdim=False).sum(dim=-1, keepdim=True)
            if kv_cache is None
            else kv_cache + x_token.sum(dim=0, keepdim=False).sum(dim=-1, keepdim=True)
        )
    )
    _dummy_step_fn._reinforce_sequence_predict_query_outputs_with_kv_fn = (
        _predict_query_replay_outputs_with_kv
    )
    _dummy_step_fn._fit_action_dim_fn = lambda x, action_dim: x[..., :action_dim]
    _dummy_step_fn._model_ref = _DummyReplayModel()

    seq_len = 5
    eval_start = 2
    batch_size = 1
    state_dim = _resolve_dim_upper_bound(env_cfg["state_dim"])
    action_dim = int(env_cfg["action_slot_dim"])
    rewards = torch.randn(seq_len, batch_size, dtype=torch.float32)
    x_tokens = torch.randn(seq_len, batch_size, 12, dtype=torch.float32, requires_grad=True)
    replay_payload = {
        "reward_in": torch.randn(seq_len, batch_size, dtype=torch.float32),
        "sampled_action": torch.randn(seq_len - eval_start, batch_size, action_dim, dtype=torch.float32),
        "action_std": torch.full((seq_len - eval_start, batch_size), 0.05, dtype=torch.float32),
        "action_mask": torch.ones(batch_size, action_dim, dtype=torch.bool),
        "next_state": torch.randn(seq_len - eval_start, batch_size, state_dim, dtype=torch.float32),
        "state_mask": torch.ones(batch_size, state_dim, dtype=torch.bool),
        "flow_action": torch.randn(eval_start, batch_size, action_dim, dtype=torch.float32),
        "flow_next_state": torch.randn(eval_start, batch_size, state_dim, dtype=torch.float32),
        "flow_state_mask": torch.ones(batch_size, state_dim, dtype=torch.bool),
        "flow_query_start": 0,
        "flow_query_stop": eval_start,
        "eval_start": eval_start,
        "full_length": seq_len,
    }
    sink_values = []

    def _loss_sink(loss_root):
        event_log.append("sink")
        sink_values.append(float(loss_root.detach().cpu()))
        loss_root.backward()

    loss, stats, replay_log_probs = prior.reinforce_sequence_replay_loss_from_rollout(
        policy_step_fn=_dummy_step_fn,
        x_tokens=x_tokens,
        rewards=rewards,
        replay_payload=replay_payload,
        single_eval_pos=eval_start,
        fit_action_dim_fn=_dummy_step_fn._fit_action_dim_fn,
        loss_sink=_loss_sink,
    )

    assert torch.isfinite(loss).item()
    assert torch.isfinite(replay_log_probs).all().item()
    assert len(sink_values) == 2
    assert all(np.isfinite(v) for v in sink_values)
    assert event_log[0] == "policy_forward"
    assert event_log[1] == "sink"
    assert "q_forward" in event_log[2:]
    assert "flow_forward" in event_log[2:]
    assert event_log[-1] == "sink"
    assert x_tokens.grad is not None
    assert torch.isfinite(x_tokens.grad).all().item()
    assert float(x_tokens.grad.abs().sum()) > 0.0
    assert torch.allclose(loss.detach(), stats["policy_total_loss"].detach(), atol=1e-6, rtol=1e-6)


def test_reinforce_sequence_replay_q_flow_shared_aux_query_pass_sinks_policy_then_aux():
    _seed_everything(9993)
    _, env_cfg = _build_small_exact_scm_env_cfg()
    env_cfg["policy_gradient_weight"] = 0.1
    env_cfg["normalized_q_value_weight"] = 0.25
    env_cfg["next_state_flow_matching_weight"] = 0.5
    env_cfg["obs_slot_dim"] = 8
    env_cfg["action_slot_dim"] = 3
    prior = EnvironmentPrior(env_cfg)
    bardist = make_standardized_full_support_bar_distribution()
    event_log = []

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

    def _init_kv_cache(x_tokens, y_tokens):
        del y_tokens
        return x_tokens.sum(dim=0).sum(dim=-1, keepdim=True)

    def _append_train_token_to_kv(x_token, y_token, kv_cache):
        del y_token
        token_state = x_token.sum(dim=0).sum(dim=-1, keepdim=True)
        return token_state if kv_cache is None else (kv_cache + token_state)

    def _replay_outputs(
        x_tokens,
        y_tokens,
        *,
        eval_start=0,
        flow_matching_xt=None,
        flow_matching_t=None,
    ):
        del y_tokens, flow_matching_xt, flow_matching_t
        event_log.append("policy_forward")
        query = x_tokens[eval_start:]
        hidden = query[..., :1]
        return {
            "action_mean": hidden.expand(int(query.shape[0]), int(query.shape[1]), 3),
            "action_std": torch.full(
                (int(query.shape[0]), int(query.shape[1]), 3),
                0.05,
                dtype=x_tokens.dtype,
                device=x_tokens.device,
            ),
        }

    def _predict_query_replay_outputs_with_kv(
        x_query,
        kv_cache,
        *,
        action_query=None,
        flow_matching_xt=None,
        flow_matching_t=None,
    ):
        del x_query, action_query, flow_matching_t
        hidden = kv_cache.unsqueeze(0)
        outputs = {}
        if flow_matching_xt is None:
            event_log.append("q_forward")
            logits = hidden.expand(1, int(hidden.shape[1]), bardist.num_bars)
            outputs["normalized_q_logits"] = logits
            outputs["normalized_q"] = bardist.mean(logits)
        else:
            event_log.append("flow_forward")
            outputs["next_state_flow"] = hidden.expand_as(flow_matching_xt)
        return outputs

    def _dummy_step_fn(*args, **kwargs):
        raise AssertionError("shared-aux sink test should not call live policy_step")

    _dummy_step_fn._reinforce_sequence_replay_fn = _replay_tokens
    _dummy_step_fn._reinforce_sequence_replay_outputs_fn = _replay_outputs
    _dummy_step_fn._reinforce_sequence_init_kv_cache_fn = _init_kv_cache
    _dummy_step_fn._reinforce_sequence_append_train_token_to_kv_fn = _append_train_token_to_kv
    _dummy_step_fn._reinforce_sequence_predict_query_outputs_with_kv_fn = (
        _predict_query_replay_outputs_with_kv
    )
    _dummy_step_fn._fit_action_dim_fn = lambda x, action_dim: x[..., :action_dim]
    _dummy_step_fn._model_ref = _DummyReplayModel()

    seq_len = 5
    eval_start = 2
    batch_size = 1
    state_dim = _resolve_dim_upper_bound(env_cfg["state_dim"])
    action_dim = int(env_cfg["action_slot_dim"])
    rewards = torch.randn(seq_len, batch_size, dtype=torch.float32)
    x_tokens = torch.randn(seq_len, batch_size, 12, dtype=torch.float32, requires_grad=True)
    replay_payload = {
        "reward_in": torch.randn(seq_len, batch_size, dtype=torch.float32),
        "sampled_action": torch.randn(seq_len - eval_start, batch_size, action_dim, dtype=torch.float32),
        "action_std": torch.full((seq_len - eval_start, batch_size), 0.05, dtype=torch.float32),
        "action_mask": torch.ones(batch_size, action_dim, dtype=torch.bool),
        "next_state": torch.randn(seq_len - eval_start, batch_size, state_dim, dtype=torch.float32),
        "state_mask": torch.ones(batch_size, state_dim, dtype=torch.bool),
        "flow_action": torch.randn(eval_start, batch_size, action_dim, dtype=torch.float32),
        "flow_next_state": torch.randn(eval_start, batch_size, state_dim, dtype=torch.float32),
        "flow_state_mask": torch.ones(batch_size, state_dim, dtype=torch.bool),
        "flow_query_start": 0,
        "flow_query_stop": eval_start,
        "eval_start": eval_start,
        "full_length": seq_len,
    }
    sink_values = []

    def _loss_sink(loss_root):
        event_log.append("sink")
        sink_values.append(float(loss_root.detach().cpu()))
        loss_root.backward()

    loss, stats, replay_log_probs = prior.reinforce_sequence_replay_loss_from_rollout(
        policy_step_fn=_dummy_step_fn,
        x_tokens=x_tokens,
        rewards=rewards,
        replay_payload=replay_payload,
        single_eval_pos=eval_start,
        fit_action_dim_fn=_dummy_step_fn._fit_action_dim_fn,
        loss_sink=_loss_sink,
    )

    assert torch.isfinite(loss).item()
    assert torch.isfinite(replay_log_probs).all().item()
    assert len(sink_values) == 2
    assert event_log.count("sink") == 2
    assert event_log[0] == "policy_forward"
    assert event_log[1] == "sink"
    assert event_log[-1] == "sink"
    assert "q_forward" in event_log
    assert "flow_forward" in event_log
    assert int(stats["reinforce_sequence_replay_policy_aux_shared_context_forward"]) == 0
    assert int(stats["reinforce_sequence_replay_aux_query_pass_enabled"]) == 1
    assert int(stats["reinforce_sequence_replay_q_flow_shared_aux_query_pass"]) == 1
    assert "reinforce_sequence_replay_shared_backbone_pass" not in stats
    assert "reinforce_sequence_replay_shared_aux_backbone_pass" not in stats
    assert x_tokens.grad is not None
    assert torch.isfinite(x_tokens.grad).all().item()
    assert float(x_tokens.grad.abs().sum()) > 0.0
    assert torch.allclose(loss.detach(), stats["policy_total_loss"].detach(), atol=1e-6, rtol=1e-6)


def test_reinforce_sequence_replay_prefers_official_sequence_aux_helper():
    _seed_everything(9994)
    _, env_cfg = _build_small_exact_scm_env_cfg()
    env_cfg["policy_gradient_weight"] = 0.1
    env_cfg["normalized_q_value_weight"] = 0.25
    env_cfg["next_state_flow_matching_weight"] = 0.5
    env_cfg["obs_slot_dim"] = 8
    env_cfg["action_slot_dim"] = 3
    prior = EnvironmentPrior(env_cfg)
    bardist = make_standardized_full_support_bar_distribution()
    event_log = []

    class _DummyReplayModel:
        def resolve_replay_batch_chunk_size(self, *, seq_len, total_batch):
            del seq_len
            return int(total_batch)

        def get_normalized_q_value_bardist(self):
            return bardist

    def _replay_outputs(
        x_tokens,
        y_tokens,
        *,
        eval_start=0,
        flow_matching_xt=None,
        flow_matching_t=None,
    ):
        del y_tokens, flow_matching_xt, flow_matching_t
        event_log.append("policy_forward")
        query = x_tokens[eval_start:]
        return {
            "action_mean": torch.zeros(
                int(query.shape[0]),
                int(query.shape[1]),
                3,
                dtype=query.dtype,
                device=query.device,
            ),
        }

    def _replay_aux_sequence_outputs(
        x_tokens,
        y_tokens,
        *,
        full_length,
        q_query_tokens=None,
        q_query_start=0,
        q_query_stop=None,
        q_action_query=None,
        flow_query_tokens=None,
        flow_query_start=0,
        flow_query_stop=None,
        flow_action_query=None,
        flow_matching_xt=None,
        flow_matching_t=None,
    ):
        del x_tokens, y_tokens, full_length, q_query_stop, flow_query_stop, flow_matching_t
        event_log.append("official_aux_sequence")
        assert q_query_tokens is not None
        assert q_action_query is not None
        assert flow_query_tokens is not None
        assert flow_action_query is not None
        outputs = {
            "q": {
                "normalized_q_logits": torch.zeros(
                    int(q_query_tokens.shape[0]),
                    int(q_query_tokens.shape[1]),
                    bardist.num_bars,
                    dtype=torch.float32,
                ),
            },
            "flow": {
                "next_state_flow": torch.zeros_like(flow_matching_xt),
            },
        }
        outputs["q"]["normalized_q"] = bardist.mean(outputs["q"]["normalized_q_logits"])
        assert int(q_query_start) == 2
        assert int(flow_query_start) == 0
        return outputs

    def _dummy_step_fn(*args, **kwargs):
        raise AssertionError("official-sequence helper test should not call live policy_step")

    _dummy_step_fn._reinforce_sequence_replay_fn = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("official-sequence helper test should use replay outputs path")
    )
    _dummy_step_fn._reinforce_sequence_replay_outputs_fn = _replay_outputs
    _dummy_step_fn._reinforce_sequence_replay_aux_sequence_outputs_fn = (
        _replay_aux_sequence_outputs
    )
    _dummy_step_fn._reinforce_sequence_stream_replay_aux_outputs_with_kv_fn = (
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("official sequence helper should bypass streaming aux fallback")
        )
    )
    _dummy_step_fn._reinforce_sequence_init_kv_cache_fn = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("official sequence helper should bypass init_kv_cache fallback")
    )
    _dummy_step_fn._reinforce_sequence_append_train_token_to_kv_fn = (
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("official sequence helper should bypass append_train_token_to_kv fallback")
        )
    )
    _dummy_step_fn._reinforce_sequence_predict_query_outputs_with_kv_fn = (
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("official sequence helper should bypass predict_query_replay_outputs_with_kv fallback")
        )
    )
    _dummy_step_fn._fit_action_dim_fn = lambda x, action_dim: x[..., :action_dim]
    _dummy_step_fn._model_ref = _DummyReplayModel()

    seq_len = 5
    eval_start = 2
    batch_size = 1
    state_dim = _resolve_dim_upper_bound(env_cfg["state_dim"])
    action_dim = int(env_cfg["action_slot_dim"])
    rewards = torch.randn(seq_len, batch_size, dtype=torch.float32)
    replay_payload = {
        "reward_in": torch.randn(seq_len, batch_size, dtype=torch.float32),
        "sampled_action": torch.randn(seq_len - eval_start, batch_size, action_dim, dtype=torch.float32),
        "action_std": torch.full((seq_len - eval_start, batch_size), 0.05, dtype=torch.float32),
        "action_mask": torch.ones(batch_size, action_dim, dtype=torch.bool),
        "next_state": torch.randn(seq_len - eval_start, batch_size, state_dim, dtype=torch.float32),
        "state_mask": torch.ones(batch_size, state_dim, dtype=torch.bool),
        "flow_action": torch.randn(eval_start, batch_size, action_dim, dtype=torch.float32),
        "flow_next_state": torch.randn(eval_start, batch_size, state_dim, dtype=torch.float32),
        "flow_state_mask": torch.ones(batch_size, state_dim, dtype=torch.bool),
        "flow_query_start": 0,
        "flow_query_stop": eval_start,
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
    assert event_log == ["policy_forward", "official_aux_sequence"]
    assert int(stats["reinforce_sequence_replay_policy_aux_shared_context_forward"]) == 0
    assert int(stats["reinforce_sequence_replay_aux_query_pass_enabled"]) == 1
    assert int(stats["reinforce_sequence_replay_q_flow_shared_aux_query_pass"]) == 1
    assert "reinforce_sequence_replay_shared_backbone_pass" not in stats
    assert "reinforce_sequence_replay_shared_aux_backbone_pass" not in stats


def test_reinforce_sequence_replay_prefers_shared_context_forward_helper():
    _seed_everything(99941)
    _, env_cfg = _build_small_exact_scm_env_cfg()
    env_cfg["policy_gradient_weight"] = 0.1
    env_cfg["normalized_q_value_weight"] = 0.25
    env_cfg["next_state_flow_matching_weight"] = 0.5
    env_cfg["obs_slot_dim"] = 8
    env_cfg["action_slot_dim"] = 3
    env_cfg["reinforce_sequence_replay_share_context_forward"] = True
    prior = EnvironmentPrior(env_cfg)
    bardist = make_standardized_full_support_bar_distribution()
    event_log = []

    class _DummyReplayModel:
        def resolve_replay_batch_chunk_size(self, *, seq_len, total_batch):
            del seq_len
            return int(total_batch)

        def get_normalized_q_value_bardist(self):
            return bardist

    def _shared_policy_and_aux_outputs(
        x_tokens,
        y_tokens,
        *,
        full_length,
        eval_start=0,
        q_query_tokens=None,
        q_query_start=0,
        q_query_stop=None,
        q_action_query=None,
        flow_query_tokens=None,
        flow_query_start=0,
        flow_query_stop=None,
        flow_action_query=None,
        flow_matching_xt=None,
        flow_matching_t=None,
    ):
        del y_tokens, full_length, q_query_stop, flow_query_stop, flow_matching_t
        event_log.append("shared_context_forward")
        assert q_query_tokens is not None
        assert q_action_query is not None
        assert flow_query_tokens is not None
        assert flow_action_query is not None
        assert int(q_query_start) == int(eval_start)
        assert int(flow_query_start) == 0
        policy_query = x_tokens[eval_start:]
        action_mean = policy_query[..., :3]
        action_std = torch.full_like(action_mean, 0.05)
        q_logits = q_action_query.mean(dim=-1, keepdim=True).expand(
            int(q_action_query.shape[0]),
            int(q_action_query.shape[1]),
            bardist.num_bars,
        )
        return {
            "policy": {
                "action_mean": action_mean,
                "action_std": action_std,
            },
            "q": {
                "normalized_q_logits": q_logits,
                "normalized_q": bardist.mean(q_logits),
            },
            "flow": {
                "next_state_flow": torch.zeros_like(flow_matching_xt),
            },
        }

    def _dummy_step_fn(*args, **kwargs):
        raise AssertionError("shared-context-forward test should not call live policy_step")

    _dummy_step_fn._reinforce_sequence_replay_fn = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("shared-context-forward test should not use live replay fallback")
    )
    _dummy_step_fn._reinforce_sequence_replay_outputs_fn = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("shared-context-forward test should bypass legacy policy replay")
    )
    _dummy_step_fn._reinforce_sequence_replay_aux_sequence_outputs_fn = (
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("shared-context-forward test should bypass legacy aux replay")
        )
    )
    _dummy_step_fn._reinforce_sequence_encode_train_tokens_fn = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("shared-context-forward test should bypass shared train-token encoding")
    )
    _dummy_step_fn._reinforce_sequence_replay_outputs_from_train_tokens_fn = (
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("shared-context-forward test should bypass policy-from-train-tokens replay")
        )
    )
    _dummy_step_fn._reinforce_sequence_replay_aux_sequence_outputs_from_train_tokens_fn = (
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("shared-context-forward test should bypass aux-from-train-tokens replay")
        )
    )
    _dummy_step_fn._reinforce_sequence_stream_replay_aux_outputs_with_kv_fn = (
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("shared-context-forward test should bypass aux-only streaming fallback")
        )
    )
    _dummy_step_fn._reinforce_sequence_replay_policy_and_aux_sequence_outputs_fn = (
        _shared_policy_and_aux_outputs
    )
    _dummy_step_fn._reinforce_sequence_stream_policy_and_aux_outputs_with_kv_fn = (
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("shared-context-forward test should prefer official shared replay helper")
        )
    )
    _dummy_step_fn._fit_action_dim_fn = lambda x, action_dim: x[..., :action_dim]
    _dummy_step_fn._model_ref = _DummyReplayModel()

    seq_len = 5
    eval_start = 2
    batch_size = 1
    state_dim = _resolve_dim_upper_bound(env_cfg["state_dim"])
    action_dim = int(env_cfg["action_slot_dim"])
    rewards = torch.randn(seq_len, batch_size, dtype=torch.float32)
    replay_payload = {
        "reward_in": torch.randn(seq_len, batch_size, dtype=torch.float32),
        "sampled_action": torch.randn(seq_len - eval_start, batch_size, action_dim, dtype=torch.float32),
        "action_std": torch.full((seq_len - eval_start, batch_size), 0.05, dtype=torch.float32),
        "action_mask": torch.ones(batch_size, action_dim, dtype=torch.bool),
        "next_state": torch.randn(seq_len - eval_start, batch_size, state_dim, dtype=torch.float32),
        "state_mask": torch.ones(batch_size, state_dim, dtype=torch.bool),
        "flow_action": torch.randn(eval_start, batch_size, action_dim, dtype=torch.float32),
        "flow_next_state": torch.randn(eval_start, batch_size, state_dim, dtype=torch.float32),
        "flow_state_mask": torch.ones(batch_size, state_dim, dtype=torch.bool),
        "flow_query_start": 0,
        "flow_query_stop": eval_start,
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
    assert event_log == ["shared_context_forward"]
    assert int(stats["reinforce_sequence_replay_policy_aux_shared_context_forward"]) == 1
    assert int(stats["reinforce_sequence_replay_aux_query_pass_enabled"]) == 1
    assert int(stats["reinforce_sequence_replay_q_flow_shared_aux_query_pass"]) == 1
    assert int(stats["reinforce_sequence_replay_shared_train_token_encoding"]) == 0
    assert "reinforce_sequence_replay_shared_backbone_pass" not in stats
    assert "reinforce_sequence_replay_shared_aux_backbone_pass" not in stats


def test_reinforce_sequence_replay_prefers_strict_aux_sequence_helper_over_packed_and_streaming():
    _seed_everything(999411)
    _, env_cfg = _build_small_exact_scm_env_cfg()
    env_cfg["policy_gradient_weight"] = 0.1
    env_cfg["normalized_q_value_weight"] = 0.25
    env_cfg["next_state_flow_matching_weight"] = 0.5
    env_cfg["obs_slot_dim"] = 8
    env_cfg["action_slot_dim"] = 3
    prior = EnvironmentPrior(env_cfg)
    bardist = make_standardized_full_support_bar_distribution()
    event_log = []

    class _DummyReplayModel:
        def resolve_replay_batch_chunk_size(self, *, seq_len, total_batch):
            del seq_len
            return int(total_batch)

        def get_normalized_q_value_bardist(self):
            return bardist

    def _replay_outputs(x_tokens, y_tokens, *, eval_start=0, flow_matching_xt=None, flow_matching_t=None):
        del y_tokens, flow_matching_xt, flow_matching_t
        event_log.append("policy_forward")
        return {"action_mean": x_tokens[eval_start:, :, :3]}

    def _strict_aux_outputs(
        x_tokens,
        y_tokens,
        *,
        full_length,
        q_query_tokens=None,
        q_query_start=0,
        q_query_stop=None,
        q_action_query=None,
        flow_query_tokens=None,
        flow_query_start=0,
        flow_query_stop=None,
        flow_action_query=None,
        flow_matching_xt=None,
        flow_matching_t=None,
    ):
        del x_tokens, y_tokens, full_length, q_query_tokens, q_query_start, q_query_stop
        del flow_query_tokens, flow_query_start, flow_query_stop, flow_matching_t
        event_log.append("strict_aux_sequence")
        q_logits = q_action_query.mean(dim=-1, keepdim=True).expand(
            int(q_action_query.shape[0]),
            int(q_action_query.shape[1]),
            bardist.num_bars,
        )
        return {
            "q": {
                "normalized_q_logits": q_logits,
                "normalized_q": bardist.mean(q_logits),
            },
            "flow": {
                "next_state_flow": torch.zeros_like(flow_matching_xt) + flow_action_query.mean(dim=-1, keepdim=True),
            },
        }

    def _dummy_step_fn(*args, **kwargs):
        raise AssertionError("strict aux helper test should not call live policy_step")

    _dummy_step_fn._reinforce_sequence_replay_fn = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("strict aux helper test should use replay outputs path")
    )
    _dummy_step_fn._reinforce_sequence_replay_outputs_fn = _replay_outputs
    _dummy_step_fn._reinforce_sequence_replay_aux_strict_sequence_outputs_fn = _strict_aux_outputs
    _dummy_step_fn._reinforce_sequence_replay_aux_sequence_outputs_fn = (
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("strict aux helper should bypass packed aux replay")
        )
    )
    _dummy_step_fn._reinforce_sequence_stream_replay_aux_outputs_with_kv_fn = (
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("strict aux helper should bypass streaming aux replay")
        )
    )
    _dummy_step_fn._reinforce_sequence_init_kv_cache_fn = lambda *args, **kwargs: None
    _dummy_step_fn._reinforce_sequence_append_train_token_to_kv_fn = lambda *args, **kwargs: None
    _dummy_step_fn._reinforce_sequence_predict_query_outputs_with_kv_fn = lambda *args, **kwargs: {}
    _dummy_step_fn._fit_action_dim_fn = lambda x, action_dim: x[..., :action_dim]
    _dummy_step_fn._model_ref = _DummyReplayModel()

    seq_len = 5
    eval_start = 2
    batch_size = 1
    state_dim = _resolve_dim_upper_bound(env_cfg["state_dim"])
    action_dim = int(env_cfg["action_slot_dim"])
    rewards = torch.randn(seq_len, batch_size, dtype=torch.float32)
    replay_payload = {
        "reward_in": torch.randn(seq_len, batch_size, dtype=torch.float32),
        "sampled_action": torch.randn(seq_len - eval_start, batch_size, action_dim, dtype=torch.float32),
        "action_std": torch.full((seq_len - eval_start, batch_size), 0.05, dtype=torch.float32),
        "action_mask": torch.ones(batch_size, action_dim, dtype=torch.bool),
        "next_state": torch.randn(seq_len - eval_start, batch_size, state_dim, dtype=torch.float32),
        "state_mask": torch.ones(batch_size, state_dim, dtype=torch.bool),
        "flow_action": torch.randn(eval_start, batch_size, action_dim, dtype=torch.float32),
        "flow_next_state": torch.randn(eval_start, batch_size, state_dim, dtype=torch.float32),
        "flow_state_mask": torch.ones(batch_size, state_dim, dtype=torch.bool),
        "flow_query_start": 0,
        "flow_query_stop": eval_start,
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
    assert event_log == ["policy_forward", "strict_aux_sequence"]
    assert int(stats["reinforce_sequence_replay_policy_aux_shared_context_forward"]) == 0


def test_build_policy_step_fn_prefers_actor_only_official_replay_helpers():
    class _DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.policy_action_head = torch.nn.Linear(3, 3)
            self.event_log = []

        def forward_policy_step(self, x_token, y_token, **kwargs):
            del y_token, kwargs
            return {"action_mean": x_token[..., :3], "action_std": torch.ones_like(x_token[..., :3])}, None

        def replay_policy_sequence_tokens(self, x_tokens, y_tokens, *, eval_start=0, policy_action_head_params_override=None):
            del x_tokens, y_tokens, eval_start, policy_action_head_params_override
            self.event_log.append("legacy_tokens")
            return torch.zeros(1, 1, 3)

        def replay_policy_actor_outputs(self, x_tokens, y_tokens, *, eval_start=0, policy_action_head_params_override=None):
            del y_tokens, eval_start, policy_action_head_params_override
            self.event_log.append("actor_only")
            return {
                "action_mean": x_tokens[..., :3].clone(),
                "action_std": torch.ones_like(x_tokens[..., :3]),
            }

        def replay_policy_sequence_outputs(self, x_tokens, y_tokens, *, eval_start=0, action_query=None, policy_action_head_params_override=None, flow_matching_xt=None, flow_matching_t=None):
            del x_tokens, y_tokens, eval_start, action_query, policy_action_head_params_override, flow_matching_xt, flow_matching_t
            self.event_log.append("legacy_outputs")
            raise AssertionError("training should prefer actor-only replay outputs helper")

        def encode_train_sequence_tokens(self, x_tokens, y_tokens):
            del y_tokens
            return x_tokens.clone()

        def replay_policy_actor_outputs_from_train_tokens(self, train_tokens, *, eval_start=0, policy_action_head_params_override=None):
            del eval_start, policy_action_head_params_override
            self.event_log.append("actor_only_train_tokens")
            return {
                "action_mean": train_tokens[..., :3].clone(),
                "action_std": torch.ones_like(train_tokens[..., :3]),
            }

        def replay_policy_sequence_outputs_from_train_tokens(self, train_tokens, *, eval_start=0, action_query=None, policy_action_head_params_override=None, flow_matching_xt=None, flow_matching_t=None):
            del train_tokens, eval_start, action_query, policy_action_head_params_override, flow_matching_xt, flow_matching_t
            self.event_log.append("legacy_outputs_train_tokens")
            raise AssertionError("training should prefer actor-only replay outputs-from-train-tokens helper")

    model = _DummyModel()
    step_fn = _build_policy_step_fn(
        model,
        num_features=6,
        max_cache_len=4,
        kv_cache_mode="immutable",
        kv_cache_page_size=None,
        allow_grad_mutable_cache=False,
        pg_torch_compile=False,
    )

    replay_outputs = step_fn._reinforce_sequence_replay_outputs_fn(
        torch.randn(4, 2, 6),
        torch.randn(4, 2),
        eval_start=1,
    )
    assert tuple(replay_outputs["action_mean"].shape) == (4, 2, 3)
    assert callable(getattr(model, "replay_policy_actor_outputs_from_train_tokens", None))
    assert model.event_log == ["actor_only"]


def test_rwkv7_replay_aux_strict_sequence_outputs_from_train_tokens_uses_one_query_per_context():
    _seed_everything(10123)
    cfg = _build_rwkv7_rlpfn_config()
    cfg["transformer"]["emsize"] = 64
    cfg["transformer"]["nlayers"] = 2
    cfg["transformer"]["rwkv_head_size"] = 64
    cfg["transformer"]["rwkv_sequence_replay_checkpoint"] = False
    cfg["prior"]["environment"]["normalized_q_value_weight"] = 1.0
    cfg["prior"]["environment"]["next_state_flow_matching_weight"] = 1.0
    cfg["prior"]["environment"]["state_dim"] = {"distribution": "uniform_int", "min": 8, "max": 8}
    _, model, *_ = get_model(cfg, device="cpu", should_train=False, verbose=False)
    model.train()

    recorded_shapes = []

    def _fake_forward_tokens_sequence_only(tokens):
        recorded_shapes.append(tuple(tokens.shape))
        return tokens

    def _fake_decode_aux_replay_outputs(
        hidden_q,
        *,
        action_query=None,
        include_normalized_q_logits=False,
        flow_matching_xt=None,
        flow_matching_t=None,
    ):
        del action_query, flow_matching_t
        outputs = {}
        if include_normalized_q_logits:
            outputs["normalized_q_logits"] = hidden_q.mean(dim=-1, keepdim=True).expand(
                int(hidden_q.shape[0]),
                int(hidden_q.shape[1]),
                model.get_normalized_q_value_bardist().num_bars,
            )
        if flow_matching_xt is not None:
            outputs["next_state_flow"] = flow_matching_xt.clone()
        return outputs

    model.rwkv_core.forward_tokens_sequence_only = _fake_forward_tokens_sequence_only
    model._decode_aux_replay_outputs = _fake_decode_aux_replay_outputs

    train_tokens = torch.randn(4, 2, model.emsize, dtype=torch.float32)
    q_query_tokens = torch.randn(2, 2, cfg["prior"]["num_features"], dtype=torch.float32)
    flow_query_tokens = torch.randn(1, 2, cfg["prior"]["num_features"], dtype=torch.float32)
    flow_matching_xt = torch.randn(1, 2, 8, dtype=torch.float32)
    flow_matching_t = torch.rand(1, 2, 1, dtype=torch.float32)

    outputs = model.replay_aux_strict_sequence_outputs_from_train_tokens(
        train_tokens,
        full_length=4,
        q_query_tokens=q_query_tokens,
        q_query_start=2,
        q_query_stop=4,
        q_action_query=torch.randn(2, 2, 3, dtype=torch.float32),
        flow_query_tokens=flow_query_tokens,
        flow_query_start=0,
        flow_query_stop=1,
        flow_action_query=torch.randn(1, 2, 3, dtype=torch.float32),
        flow_matching_xt=flow_matching_xt,
        flow_matching_t=flow_matching_t,
    )

    assert tuple(outputs["q"]["normalized_q_logits"].shape[:2]) == (2, 2)
    assert tuple(outputs["flow"]["next_state_flow"].shape) == tuple(flow_matching_xt.shape)
    assert recorded_shapes == [
        (4, 2, model.emsize),
        (5, 2, model.emsize),
        (2, 2, model.emsize),
    ]


def test_rwkv7_replay_aux_strict_sequence_outputs_from_train_tokens_streams_output_chunks():
    _seed_everything(10123)
    cfg = _build_rwkv7_rlpfn_config()
    cfg["transformer"]["emsize"] = 64
    cfg["transformer"]["nlayers"] = 2
    cfg["transformer"]["rwkv_head_size"] = 64
    cfg["transformer"]["rwkv_sequence_replay_checkpoint"] = False
    cfg["prior"]["environment"]["normalized_q_value_weight"] = 1.0
    cfg["prior"]["environment"]["next_state_flow_matching_weight"] = 1.0
    cfg["prior"]["environment"]["next_state_flow_head_type"] = "mlp"
    cfg["prior"]["environment"]["state_dim"] = {"distribution": "uniform_int", "min": 8, "max": 8}
    _, model, *_ = get_model(cfg, device="cpu", should_train=False, verbose=False)
    model.eval()

    def _fake_forward_tokens_sequence_only(tokens):
        return tokens.cumsum(dim=0)

    def _fake_decode_aux_replay_outputs(
        hidden_q,
        *,
        action_query=None,
        include_normalized_q_logits=False,
        flow_matching_xt=None,
        flow_matching_t=None,
    ):
        outputs = {}
        if include_normalized_q_logits:
            base = hidden_q.mean(dim=-1, keepdim=True)
            if action_query is not None:
                base = base + action_query.mean(dim=-1, keepdim=True)
            outputs["normalized_q_logits"] = base.expand(
                int(hidden_q.shape[0]),
                int(hidden_q.shape[1]),
                model.get_normalized_q_value_bardist().num_bars,
            )
        if flow_matching_xt is not None:
            outputs["next_state_flow"] = flow_matching_xt.clone()
        return outputs

    model.rwkv_core.forward_tokens_sequence_only = _fake_forward_tokens_sequence_only
    model._decode_aux_replay_outputs = _fake_decode_aux_replay_outputs

    train_tokens = torch.randn(4, 2, model.emsize, dtype=torch.float32)
    q_query_tokens = torch.randn(2, 2, cfg["prior"]["num_features"], dtype=torch.float32)
    flow_query_tokens = torch.randn(1, 2, cfg["prior"]["num_features"], dtype=torch.float32)
    q_action_query = torch.randn(2, 2, 3, dtype=torch.float32)
    flow_matching_xt = torch.randn(1, 2, 8, dtype=torch.float32)
    flow_matching_t = torch.rand(1, 2, 1, dtype=torch.float32)

    full = model.replay_aux_strict_sequence_outputs_from_train_tokens(
        train_tokens,
        full_length=4,
        q_query_tokens=q_query_tokens,
        q_query_start=2,
        q_query_stop=4,
        q_action_query=q_action_query,
        flow_query_tokens=flow_query_tokens,
        flow_query_start=0,
        flow_query_stop=1,
        flow_action_query=torch.randn(1, 2, 3, dtype=torch.float32),
        flow_matching_xt=flow_matching_xt,
        flow_matching_t=flow_matching_t,
    )

    streamed_records = []
    streamed_outputs = {"q": {}, "flow": {}}

    def _output_chunk_sink(section_name, local_start, local_stop, chunk_outputs):
        streamed_records.append((str(section_name), int(local_start), int(local_stop)))
        for key, value in chunk_outputs.items():
            streamed_outputs[section_name].setdefault(key, []).append(value.detach().clone())

    streamed = model.replay_aux_strict_sequence_outputs_from_train_tokens(
        train_tokens,
        full_length=4,
        q_query_tokens=q_query_tokens,
        q_query_start=2,
        q_query_stop=4,
        q_action_query=q_action_query,
        flow_query_tokens=flow_query_tokens,
        flow_query_start=0,
        flow_query_stop=1,
        flow_action_query=torch.randn(1, 2, 3, dtype=torch.float32),
        flow_matching_xt=flow_matching_xt,
        flow_matching_t=flow_matching_t,
        output_chunk_sink=_output_chunk_sink,
    )

    assert streamed.get("_streamed_via_sink", False) is True
    assert streamed["q"] == {}
    assert streamed["flow"] == {}
    assert streamed_records == [("q", 0, 1), ("q", 1, 2), ("flow", 0, 1)]
    torch.testing.assert_close(
        torch.cat(streamed_outputs["q"]["normalized_q_logits"], dim=0),
        full["q"]["normalized_q_logits"],
        atol=1e-6,
        rtol=1e-6,
    )
    torch.testing.assert_close(
        torch.cat(streamed_outputs["flow"]["next_state_flow"], dim=0),
        full["flow"]["next_state_flow"],
        atol=1e-6,
        rtol=1e-6,
    )


def test_rwkv7_replay_aux_strict_sequence_outputs_streams_output_chunks():
    _seed_everything(10125)
    cfg = _build_rwkv7_rlpfn_config()
    cfg["transformer"]["emsize"] = 64
    cfg["transformer"]["nlayers"] = 2
    cfg["transformer"]["rwkv_head_size"] = 64
    cfg["transformer"]["rwkv_sequence_replay_checkpoint"] = False
    cfg["prior"]["environment"]["normalized_q_value_weight"] = 1.0
    cfg["prior"]["environment"]["next_state_flow_matching_weight"] = 1.0
    cfg["prior"]["environment"]["next_state_flow_head_type"] = "mlp"
    cfg["prior"]["environment"]["state_dim"] = {"distribution": "uniform_int", "min": 8, "max": 8}
    _, model, *_ = get_model(cfg, device="cpu", should_train=False, verbose=False)
    model.eval()

    def _fake_forward_tokens_sequence_only(tokens):
        return tokens.cumsum(dim=0)

    def _fake_decode_aux_replay_outputs(
        hidden_q,
        *,
        action_query=None,
        include_normalized_q_logits=False,
        flow_matching_xt=None,
        flow_matching_t=None,
    ):
        outputs = {}
        if include_normalized_q_logits:
            base = hidden_q.mean(dim=-1, keepdim=True)
            if action_query is not None:
                base = base + action_query.mean(dim=-1, keepdim=True)
            outputs["normalized_q_logits"] = base.expand(
                int(hidden_q.shape[0]),
                int(hidden_q.shape[1]),
                model.get_normalized_q_value_bardist().num_bars,
            )
        if flow_matching_xt is not None:
            outputs["next_state_flow"] = flow_matching_xt.clone()
        return outputs

    model.rwkv_core.forward_tokens_sequence_only = _fake_forward_tokens_sequence_only
    model._decode_aux_replay_outputs = _fake_decode_aux_replay_outputs

    x_tokens = torch.randn(4, 2, cfg["prior"]["num_features"], dtype=torch.float32)
    y_tokens = torch.randn(4, 2, dtype=torch.float32)
    q_query_tokens = torch.randn(2, 2, cfg["prior"]["num_features"], dtype=torch.float32)
    flow_query_tokens = torch.randn(1, 2, cfg["prior"]["num_features"], dtype=torch.float32)
    q_action_query = torch.randn(2, 2, 3, dtype=torch.float32)
    flow_matching_xt = torch.randn(1, 2, 8, dtype=torch.float32)
    flow_matching_t = torch.rand(1, 2, 1, dtype=torch.float32)

    full = model.replay_aux_strict_sequence_outputs(
        x_tokens,
        y_tokens,
        full_length=4,
        q_query_tokens=q_query_tokens,
        q_query_start=2,
        q_query_stop=4,
        q_action_query=q_action_query,
        flow_query_tokens=flow_query_tokens,
        flow_query_start=0,
        flow_query_stop=1,
        flow_action_query=torch.randn(1, 2, 3, dtype=torch.float32),
        flow_matching_xt=flow_matching_xt,
        flow_matching_t=flow_matching_t,
    )

    streamed_records = []
    streamed_outputs = {"q": {}, "flow": {}}

    def _output_chunk_sink(section_name, local_start, local_stop, chunk_outputs):
        streamed_records.append((str(section_name), int(local_start), int(local_stop)))
        for key, value in chunk_outputs.items():
            streamed_outputs[section_name].setdefault(key, []).append(value.detach().clone())

    streamed = model.replay_aux_strict_sequence_outputs(
        x_tokens,
        y_tokens,
        full_length=4,
        q_query_tokens=q_query_tokens,
        q_query_start=2,
        q_query_stop=4,
        q_action_query=q_action_query,
        flow_query_tokens=flow_query_tokens,
        flow_query_start=0,
        flow_query_stop=1,
        flow_action_query=torch.randn(1, 2, 3, dtype=torch.float32),
        flow_matching_xt=flow_matching_xt,
        flow_matching_t=flow_matching_t,
        output_chunk_sink=_output_chunk_sink,
    )

    assert streamed.get("_streamed_via_sink", False) is True
    assert streamed["q"] == {}
    assert streamed["flow"] == {}
    assert streamed_records == [("q", 0, 1), ("q", 1, 2), ("flow", 0, 1)]
    torch.testing.assert_close(
        torch.cat(streamed_outputs["q"]["normalized_q_logits"], dim=0),
        full["q"]["normalized_q_logits"],
        atol=1e-6,
        rtol=1e-6,
    )
    torch.testing.assert_close(
        torch.cat(streamed_outputs["flow"]["next_state_flow"], dim=0),
        full["flow"]["next_state_flow"],
        atol=1e-6,
        rtol=1e-6,
    )


def test_rwkv7_stream_replay_aux_outputs_with_kv_matches_strict_under_prefix_sum_core(monkeypatch):
    _seed_everything(10124)
    cfg = _build_rwkv7_rlpfn_config()
    cfg["transformer"]["emsize"] = 64
    cfg["transformer"]["nlayers"] = 2
    cfg["transformer"]["rwkv_head_size"] = 64
    cfg["transformer"]["rwkv_sequence_replay_checkpoint"] = False
    cfg["prior"]["environment"]["normalized_q_value_weight"] = 1.0
    cfg["prior"]["environment"]["next_state_flow_matching_weight"] = 1.0
    cfg["prior"]["environment"]["next_state_flow_head_type"] = "mlp"
    cfg["prior"]["environment"]["state_dim"] = {"distribution": "uniform_int", "min": 8, "max": 8}
    _, model, *_ = get_model(cfg, device="cpu", should_train=False, verbose=False)
    model.eval()

    bardist = model.get_normalized_q_value_bardist()
    emsize = int(model.emsize)
    num_features = int(cfg["prior"]["num_features"])

    monkeypatch.setattr(
        model,
        "resolve_replay_batch_chunk_size",
        lambda *, seq_len, total_batch: int(total_batch),
    )
    monkeypatch.setattr(model, "_cast_token_for_rwkv_core", lambda tokens: tokens)

    def _encode_train_token(x_tokens, y_tokens):
        y_term = y_tokens.unsqueeze(-1).expand(int(x_tokens.shape[0]), int(x_tokens.shape[1]), emsize)
        return x_tokens[..., :emsize] + y_term

    def _encode_query_token(x_tokens):
        return x_tokens[..., :emsize]

    def _forward_tokens_sequence_only(tokens):
        return tokens.cumsum(dim=0)

    def _init_kv_cache(x_train, y_train):
        return _encode_train_token(x_train, y_train).sum(dim=0)

    def _append_train_token_to_kv(
        x_token,
        y_token,
        kv_cache,
        max_cache_len=None,
        kv_cache_mode="auto",
        kv_cache_page_size=None,
        allow_grad_mutable_cache=False,
        allow_grad_inplace_paged_cache=False,
    ):
        del max_cache_len, kv_cache_mode, kv_cache_page_size
        del allow_grad_mutable_cache, allow_grad_inplace_paged_cache
        token = _encode_train_token(x_token, y_token)[0]
        return token if kv_cache is None else (kv_cache + token)

    def _forward_query_hidden_with_kv(x_query, kv_cache):
        query = _encode_query_token(x_query)
        base = torch.zeros_like(query) if kv_cache is None else kv_cache.unsqueeze(0)
        return base + query

    def _decode_aux_replay_outputs(
        hidden_q,
        *,
        action_query=None,
        include_normalized_q_logits=False,
        flow_matching_xt=None,
        flow_matching_t=None,
    ):
        base = hidden_q.mean(dim=-1, keepdim=True)
        if action_query is not None:
            base = base + action_query.mean(dim=-1, keepdim=True)
        if flow_matching_t is not None:
            base = base + flow_matching_t.to(dtype=hidden_q.dtype).mean(dim=-1, keepdim=True)
        outputs = {}
        if include_normalized_q_logits:
            logits = base.expand(int(hidden_q.shape[0]), int(hidden_q.shape[1]), bardist.num_bars)
            outputs["normalized_q_logits"] = logits
            outputs["normalized_q"] = bardist.mean(logits)
        if flow_matching_xt is not None:
            outputs["next_state_flow"] = flow_matching_xt + base
        return outputs

    monkeypatch.setattr(model, "_encode_train_token", _encode_train_token)
    monkeypatch.setattr(model, "_encode_query_token", _encode_query_token)
    monkeypatch.setattr(model.rwkv_core, "forward_tokens_sequence_only", _forward_tokens_sequence_only)
    monkeypatch.setattr(model, "init_kv_cache", _init_kv_cache)
    monkeypatch.setattr(model, "append_train_token_to_kv", _append_train_token_to_kv)
    monkeypatch.setattr(model, "_forward_query_hidden_with_kv", _forward_query_hidden_with_kv)
    monkeypatch.setattr(model, "_decode_aux_replay_outputs", _decode_aux_replay_outputs)

    seq_len = 6
    batch_size = 2
    q_query_start = 3
    q_query_stop = 6
    flow_query_start = 0
    flow_query_stop = 3
    q_steps = q_query_stop - q_query_start
    flow_steps = flow_query_stop - flow_query_start
    flow_dim = 8
    action_dim = 3

    x_tokens = torch.randn(seq_len, batch_size, num_features, dtype=torch.float32)
    y_tokens = torch.randn(seq_len, batch_size, dtype=torch.float32)
    train_tokens = model.encode_train_sequence_tokens(x_tokens, y_tokens)
    q_query_tokens = torch.randn(q_steps, batch_size, num_features, dtype=torch.float32)
    flow_query_tokens = torch.randn(flow_steps, batch_size, num_features, dtype=torch.float32)
    q_action_query = torch.randn(q_steps, batch_size, action_dim, dtype=torch.float32)
    flow_action_query = torch.randn(flow_steps, batch_size, action_dim, dtype=torch.float32)
    flow_matching_xt = torch.randn(flow_steps, batch_size, flow_dim, dtype=torch.float32)
    flow_matching_t = torch.rand(flow_steps, batch_size, 1, dtype=torch.float32)

    with torch.no_grad():
        strict = model.replay_aux_strict_sequence_outputs_from_train_tokens(
            train_tokens,
            full_length=seq_len,
            q_query_tokens=q_query_tokens,
            q_query_start=q_query_start,
            q_query_stop=q_query_stop,
            q_action_query=q_action_query,
            flow_query_tokens=flow_query_tokens,
            flow_query_start=flow_query_start,
            flow_query_stop=flow_query_stop,
            flow_action_query=flow_action_query,
            flow_matching_xt=flow_matching_xt,
            flow_matching_t=flow_matching_t,
        )
        streaming = model.stream_replay_aux_outputs_with_kv(
            x_tokens,
            y_tokens,
            full_length=seq_len,
            q_query_tokens=q_query_tokens,
            q_query_start=q_query_start,
            q_query_stop=q_query_stop,
            q_action_query=q_action_query,
            flow_query_tokens=flow_query_tokens,
            flow_query_start=flow_query_start,
            flow_query_stop=flow_query_stop,
            flow_action_query=flow_action_query,
            flow_matching_xt=flow_matching_xt,
            flow_matching_t=flow_matching_t,
        )

    torch.testing.assert_close(
        streaming["q"]["normalized_q_logits"],
        strict["q"]["normalized_q_logits"],
        atol=1e-6,
        rtol=1e-6,
    )
    torch.testing.assert_close(
        streaming["flow"]["next_state_flow"],
        strict["flow"]["next_state_flow"],
        atol=1e-6,
        rtol=1e-6,
    )


def test_reinforce_sequence_replay_loss_sink_disables_shared_context_forward_to_split_graphs():
    _seed_everything(999412)
    _, env_cfg = _build_small_exact_scm_env_cfg()
    env_cfg["policy_gradient_weight"] = 0.1
    env_cfg["normalized_q_value_weight"] = 0.25
    env_cfg["next_state_flow_matching_weight"] = 0.5
    env_cfg["obs_slot_dim"] = 8
    env_cfg["action_slot_dim"] = 3
    env_cfg["reinforce_sequence_replay_share_context_forward"] = True
    prior = EnvironmentPrior(env_cfg)
    bardist = make_standardized_full_support_bar_distribution()
    event_log = []

    class _DummyReplayModel:
        def resolve_replay_batch_chunk_size(self, *, seq_len, total_batch):
            del seq_len
            return int(total_batch)

        def get_normalized_q_value_bardist(self):
            return bardist

    def _replay_outputs(
        x_tokens,
        y_tokens,
        *,
        eval_start=0,
        flow_matching_xt=None,
        flow_matching_t=None,
    ):
        del y_tokens, flow_matching_xt, flow_matching_t
        event_log.append("policy_forward")
        query = x_tokens[eval_start:]
        return {
            "action_mean": query[..., :3],
        }

    def _packed_aux_outputs(
        x_tokens,
        y_tokens,
        *,
        full_length,
        q_query_tokens=None,
        q_query_start=0,
        q_query_stop=None,
        q_action_query=None,
        flow_query_tokens=None,
        flow_query_start=0,
        flow_query_stop=None,
        flow_action_query=None,
        flow_matching_xt=None,
        flow_matching_t=None,
    ):
        del y_tokens, full_length, q_query_tokens, q_query_start, q_query_stop
        del flow_query_tokens, flow_query_start, flow_query_stop, flow_matching_t
        event_log.append("packed_aux_sequence")
        grad_anchor = x_tokens.mean()
        q_logits = q_action_query.mean(dim=-1, keepdim=True).expand(
            int(q_action_query.shape[0]),
            int(q_action_query.shape[1]),
            bardist.num_bars,
        ) + grad_anchor
        return {
            "q": {
                "normalized_q_logits": q_logits,
                "normalized_q": bardist.mean(q_logits),
            },
            "flow": {
                "next_state_flow": (
                    torch.zeros_like(flow_matching_xt)
                    + flow_action_query.mean(dim=-1, keepdim=True)
                    + grad_anchor
                ),
            },
        }

    def _dummy_step_fn(*args, **kwargs):
        raise AssertionError("loss-sink split-graph test should not call live policy_step")

    _dummy_step_fn._reinforce_sequence_replay_fn = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("loss-sink split-graph test should use replay outputs path")
    )
    _dummy_step_fn._reinforce_sequence_replay_outputs_fn = _replay_outputs
    _dummy_step_fn._reinforce_sequence_replay_aux_strict_sequence_outputs_fn = (
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("loss-sink split-graph test should bypass strict aux replay")
        )
    )
    _dummy_step_fn._reinforce_sequence_replay_aux_sequence_outputs_fn = _packed_aux_outputs
    _dummy_step_fn._reinforce_sequence_replay_aux_sequence_outputs_from_train_tokens_fn = (
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("loss-sink split-graph test should bypass aux-from-train-tokens replay")
        )
    )
    _dummy_step_fn._reinforce_sequence_encode_train_tokens_fn = (
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("loss-sink split-graph test should not share train-token encoding")
        )
    )
    _dummy_step_fn._reinforce_sequence_replay_outputs_from_train_tokens_fn = (
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("loss-sink split-graph test should bypass policy-from-train-tokens replay")
        )
    )
    _dummy_step_fn._reinforce_sequence_stream_policy_and_aux_outputs_with_kv_fn = (
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("loss-sink split-graph test should disable shared context forward")
        )
    )
    _dummy_step_fn._reinforce_sequence_stream_replay_aux_outputs_with_kv_fn = (
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("loss-sink split-graph test should not fall back to streaming aux replay")
        )
    )
    _dummy_step_fn._fit_action_dim_fn = lambda x, action_dim: x[..., :action_dim]
    _dummy_step_fn._model_ref = _DummyReplayModel()

    seq_len = 5
    eval_start = 2
    batch_size = 1
    state_dim = _resolve_dim_upper_bound(env_cfg["state_dim"])
    action_dim = int(env_cfg["action_slot_dim"])
    rewards = torch.randn(seq_len, batch_size, dtype=torch.float32)
    x_tokens = torch.randn(seq_len, batch_size, 12, dtype=torch.float32, requires_grad=True)
    replay_payload = {
        "reward_in": torch.randn(seq_len, batch_size, dtype=torch.float32),
        "sampled_action": torch.randn(seq_len - eval_start, batch_size, action_dim, dtype=torch.float32),
        "action_std": torch.full((seq_len - eval_start, batch_size), 0.05, dtype=torch.float32),
        "action_mask": torch.ones(batch_size, action_dim, dtype=torch.bool),
        "next_state": torch.randn(seq_len - eval_start, batch_size, state_dim, dtype=torch.float32),
        "state_mask": torch.ones(batch_size, state_dim, dtype=torch.bool),
        "flow_action": torch.randn(eval_start, batch_size, action_dim, dtype=torch.float32),
        "flow_next_state": torch.randn(eval_start, batch_size, state_dim, dtype=torch.float32),
        "flow_state_mask": torch.ones(batch_size, state_dim, dtype=torch.bool),
        "flow_query_start": 0,
        "flow_query_stop": eval_start,
        "eval_start": eval_start,
        "full_length": seq_len,
    }
    sink_values = []

    def _loss_sink(loss_root):
        event_log.append("sink")
        sink_values.append(float(loss_root.detach().cpu()))
        loss_root.backward()

    loss, stats, replay_log_probs = prior.reinforce_sequence_replay_loss_from_rollout(
        policy_step_fn=_dummy_step_fn,
        x_tokens=x_tokens,
        rewards=rewards,
        replay_payload=replay_payload,
        single_eval_pos=eval_start,
        fit_action_dim_fn=_dummy_step_fn._fit_action_dim_fn,
        loss_sink=_loss_sink,
    )

    assert torch.isfinite(loss).item()
    assert torch.isfinite(replay_log_probs).all().item()
    assert len(sink_values) == 2
    assert event_log == ["policy_forward", "sink", "packed_aux_sequence", "sink"]
    assert int(stats["reinforce_sequence_replay_policy_aux_shared_context_forward"]) == 0
    assert int(stats["reinforce_sequence_replay_shared_train_token_encoding"]) == 0
    assert x_tokens.grad is not None
    assert torch.isfinite(x_tokens.grad).all().item()
    assert float(x_tokens.grad.abs().sum()) > 0.0


def test_reinforce_sequence_replay_loss_sink_streams_strict_aux_chunks_when_supported():
    _seed_everything(999414)
    _, env_cfg = _build_small_exact_scm_env_cfg()
    env_cfg["policy_gradient_weight"] = 0.1
    env_cfg["normalized_q_value_weight"] = 0.25
    env_cfg["next_state_flow_matching_weight"] = 0.5
    env_cfg["obs_slot_dim"] = 8
    env_cfg["action_slot_dim"] = 3
    prior = EnvironmentPrior(env_cfg)
    bardist = make_standardized_full_support_bar_distribution()
    event_log = []

    class _DummyReplayModel:
        def resolve_replay_batch_chunk_size(self, *, seq_len, total_batch):
            del seq_len
            return int(total_batch)

        def get_normalized_q_value_bardist(self):
            return bardist

    def _replay_outputs(
        x_tokens,
        y_tokens,
        *,
        eval_start=0,
        flow_matching_xt=None,
        flow_matching_t=None,
    ):
        del y_tokens, flow_matching_xt, flow_matching_t
        event_log.append("policy_forward")
        query = x_tokens[eval_start:]
        return {
            "action_mean": query[..., :3],
        }

    def _strict_aux_outputs(
        x_tokens,
        y_tokens,
        *,
        full_length,
        q_query_tokens=None,
        q_query_start=0,
        q_query_stop=None,
        q_action_query=None,
        flow_query_tokens=None,
        flow_query_start=0,
        flow_query_stop=None,
        flow_action_query=None,
        flow_matching_xt=None,
        flow_matching_t=None,
        output_chunk_sink=None,
    ):
        del y_tokens, full_length, q_query_start, q_query_stop, flow_query_start, flow_query_stop
        del flow_action_query, flow_matching_t
        event_log.append("strict_aux_stream")
        if callable(output_chunk_sink):
            q_base = x_tokens[: int(q_query_tokens.shape[0]), :, :1]
            q_logits = q_base.expand(
                int(q_query_tokens.shape[0]),
                int(q_query_tokens.shape[1]),
                bardist.num_bars,
            ) + q_action_query.mean(dim=-1, keepdim=True)
            output_chunk_sink("q", 0, int(q_logits.shape[0]), {"normalized_q_logits": q_logits})
            flow_pred = flow_matching_xt + x_tokens[: int(flow_matching_xt.shape[0]), :, : int(flow_matching_xt.shape[-1])]
            output_chunk_sink("flow", 0, int(flow_matching_xt.shape[0]), {"next_state_flow": flow_pred})
            return {
                "_streamed_via_sink": True,
                "q": {},
                "flow": {},
            }
        raise AssertionError("expected output_chunk_sink to be provided")

    _dummy_step_fn = types.SimpleNamespace(
        _model_ref=_DummyReplayModel(),
        _reinforce_sequence_replay_fn=lambda x_tokens, y_tokens, eval_start=0: x_tokens[eval_start:, :, :3],
        _reinforce_sequence_replay_outputs_fn=_replay_outputs,
        _reinforce_sequence_replay_aux_strict_sequence_outputs_fn=_strict_aux_outputs,
        _policy_actor_log_prob_fn=None,
        _policy_actor_decomp_stats_fn=None,
        _fit_action_dim_fn=lambda x, target_dim: x[:, :target_dim],
    )

    seq_len = 5
    eval_start = 3
    batch_size = 1
    action_dim = 3
    state_dim = 4
    x_tokens = torch.randn(seq_len, batch_size, 8 + 1 + 1 + 1 + action_dim, dtype=torch.float32, requires_grad=True)
    rewards = torch.randn(seq_len, batch_size, dtype=torch.float32)
    replay_payload = {
        "reward_in": torch.randn(seq_len, batch_size, dtype=torch.float32),
        "sampled_action": torch.randn(seq_len - eval_start, batch_size, action_dim, dtype=torch.float32),
        "action_std": torch.full((seq_len - eval_start, batch_size), 0.05, dtype=torch.float32),
        "action_mask": torch.ones(batch_size, action_dim, dtype=torch.bool),
        "next_state": torch.randn(seq_len - eval_start, batch_size, state_dim, dtype=torch.float32),
        "state_mask": torch.ones(batch_size, state_dim, dtype=torch.bool),
        "flow_action": torch.randn(eval_start, batch_size, action_dim, dtype=torch.float32),
        "flow_next_state": torch.randn(eval_start, batch_size, state_dim, dtype=torch.float32),
        "flow_state_mask": torch.ones(batch_size, state_dim, dtype=torch.bool),
        "flow_query_start": 0,
        "flow_query_stop": eval_start,
        "eval_start": eval_start,
        "full_length": seq_len,
    }
    sink_events = []

    def _loss_sink(loss_root):
        sink_events.append(float(loss_root.detach().cpu()))
        loss_root.backward()

    loss, stats, replay_log_probs = prior.reinforce_sequence_replay_loss_from_rollout(
        policy_step_fn=_dummy_step_fn,
        x_tokens=x_tokens,
        rewards=rewards,
        replay_payload=replay_payload,
        single_eval_pos=eval_start,
        fit_action_dim_fn=_dummy_step_fn._fit_action_dim_fn,
        loss_sink=_loss_sink,
    )

    assert torch.isfinite(loss).item()
    assert torch.isfinite(replay_log_probs).all().item()
    assert event_log == ["policy_forward", "strict_aux_stream"]
    assert len(sink_events) == 3
    assert int(stats["reinforce_sequence_replay_aux_query_pass_enabled"]) == 1


def test_reinforce_sequence_replay_loss_sink_with_payload_keeps_shared_context_forward():
    _seed_everything(999413)
    _, env_cfg = _build_small_exact_scm_env_cfg()
    env_cfg["policy_gradient_weight"] = 0.1
    env_cfg["normalized_q_value_weight"] = 0.25
    env_cfg["next_state_flow_matching_weight"] = 0.5
    env_cfg["obs_slot_dim"] = 8
    env_cfg["action_slot_dim"] = 3
    env_cfg["reinforce_sequence_replay_share_context_forward"] = True
    prior = EnvironmentPrior(env_cfg)
    bardist = make_standardized_full_support_bar_distribution()
    event_log = []

    class _DummyReplayModel:
        def resolve_replay_batch_chunk_size(self, *, seq_len, total_batch):
            del seq_len
            return int(total_batch)

        def get_normalized_q_value_bardist(self):
            return bardist

    def _shared_policy_and_aux_outputs(
        x_tokens,
        y_tokens,
        *,
        full_length,
        eval_start=0,
        q_query_tokens=None,
        q_query_start=0,
        q_query_stop=None,
        q_action_query=None,
        flow_query_tokens=None,
        flow_query_start=0,
        flow_query_stop=None,
        flow_action_query=None,
        flow_matching_xt=None,
        flow_matching_t=None,
    ):
        del y_tokens, full_length, q_query_tokens, q_query_start, q_query_stop
        del flow_query_tokens, flow_query_start, flow_query_stop, flow_matching_t
        event_log.append("shared_context_forward")
        grad_anchor = x_tokens.mean()
        policy_query = x_tokens[eval_start:]
        action_mean = policy_query[..., :3] + grad_anchor
        action_std = torch.full_like(action_mean, 0.05)
        q_logits = q_action_query.mean(dim=-1, keepdim=True).expand(
            int(q_action_query.shape[0]),
            int(q_action_query.shape[1]),
            bardist.num_bars,
        ) + grad_anchor
        flow_pred = (
            torch.zeros_like(flow_matching_xt)
            + flow_action_query.mean(dim=-1, keepdim=True)
            + grad_anchor
        )
        return {
            "policy": {
                "action_mean": action_mean,
                "action_std": action_std,
            },
            "q": {
                "normalized_q_logits": q_logits,
                "normalized_q": bardist.mean(q_logits),
            },
            "flow": {
                "next_state_flow": flow_pred,
            },
        }

    def _dummy_step_fn(*args, **kwargs):
        raise AssertionError("shared payload-sink test should not call live policy_step")

    _dummy_step_fn._reinforce_sequence_replay_fn = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("shared payload-sink test should bypass live replay fallback")
    )
    _dummy_step_fn._reinforce_sequence_replay_outputs_fn = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("shared payload-sink test should bypass legacy policy replay")
    )
    _dummy_step_fn._reinforce_sequence_replay_aux_sequence_outputs_fn = (
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("shared payload-sink test should bypass legacy aux replay")
        )
    )
    _dummy_step_fn._reinforce_sequence_encode_train_tokens_fn = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("shared payload-sink test should bypass shared train-token encoding")
    )
    _dummy_step_fn._reinforce_sequence_replay_outputs_from_train_tokens_fn = (
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("shared payload-sink test should bypass policy-from-train-tokens replay")
        )
    )
    _dummy_step_fn._reinforce_sequence_replay_aux_sequence_outputs_from_train_tokens_fn = (
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("shared payload-sink test should bypass aux-from-train-tokens replay")
        )
    )
    _dummy_step_fn._reinforce_sequence_stream_replay_aux_outputs_with_kv_fn = (
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("shared payload-sink test should bypass aux-only streaming fallback")
        )
    )
    _dummy_step_fn._reinforce_sequence_replay_policy_and_aux_sequence_outputs_fn = (
        _shared_policy_and_aux_outputs
    )
    _dummy_step_fn._reinforce_sequence_stream_policy_and_aux_outputs_with_kv_fn = (
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("shared payload-sink test should prefer official shared replay helper")
        )
    )
    _dummy_step_fn._fit_action_dim_fn = lambda x, action_dim: x[..., :action_dim]
    _dummy_step_fn._model_ref = _DummyReplayModel()

    seq_len = 5
    eval_start = 2
    batch_size = 1
    state_dim = _resolve_dim_upper_bound(env_cfg["state_dim"])
    action_dim = int(env_cfg["action_slot_dim"])
    rewards = torch.randn(seq_len, batch_size, dtype=torch.float32)
    x_tokens = torch.randn(seq_len, batch_size, 12, dtype=torch.float32, requires_grad=True)
    replay_payload = {
        "reward_in": torch.randn(seq_len, batch_size, dtype=torch.float32),
        "sampled_action": torch.randn(seq_len - eval_start, batch_size, action_dim, dtype=torch.float32),
        "action_std": torch.full((seq_len - eval_start, batch_size), 0.05, dtype=torch.float32),
        "action_mask": torch.ones(batch_size, action_dim, dtype=torch.bool),
        "next_state": torch.randn(seq_len - eval_start, batch_size, state_dim, dtype=torch.float32),
        "state_mask": torch.ones(batch_size, state_dim, dtype=torch.bool),
        "flow_action": torch.randn(eval_start, batch_size, action_dim, dtype=torch.float32),
        "flow_next_state": torch.randn(eval_start, batch_size, state_dim, dtype=torch.float32),
        "flow_state_mask": torch.ones(batch_size, state_dim, dtype=torch.bool),
        "flow_query_start": 0,
        "flow_query_stop": eval_start,
        "eval_start": eval_start,
        "full_length": seq_len,
    }
    sink_payloads = []

    def _loss_sink(loss_root):
        if isinstance(loss_root, dict):
            sink_payloads.append(bool(loss_root.get("retain_graph", False)))
            loss_tensor = loss_root.get("loss", None)
        else:
            sink_payloads.append(False)
            loss_tensor = loss_root
        if torch.is_tensor(loss_tensor) and bool(loss_tensor.requires_grad):
            event_log.append("sink")
            loss_tensor.backward(retain_graph=bool(sink_payloads[-1]))
        return None

    _loss_sink._ticl_accepts_replay_payload = True

    loss, stats, replay_log_probs = prior.reinforce_sequence_replay_loss_from_rollout(
        policy_step_fn=_dummy_step_fn,
        x_tokens=x_tokens,
        rewards=rewards,
        replay_payload=replay_payload,
        single_eval_pos=eval_start,
        fit_action_dim_fn=_dummy_step_fn._fit_action_dim_fn,
        loss_sink=_loss_sink,
    )

    assert torch.isfinite(loss).item()
    assert torch.isfinite(replay_log_probs).all().item()
    assert event_log == ["shared_context_forward", "sink"]
    assert sink_payloads == [False]
    assert int(stats["reinforce_sequence_replay_policy_aux_shared_context_forward"]) == 1
    assert x_tokens.grad is not None
    assert torch.isfinite(x_tokens.grad).all().item()
    assert float(x_tokens.grad.abs().sum()) > 0.0


def test_reinforce_sequence_replay_shared_context_forward_pads_narrow_flow_targets():
    _seed_everything(999411)
    _, env_cfg = _build_small_exact_scm_env_cfg()
    env_cfg["policy_gradient_weight"] = 0.1
    env_cfg["normalized_q_value_weight"] = 0.25
    env_cfg["next_state_flow_matching_weight"] = 0.5
    env_cfg["obs_slot_dim"] = 8
    env_cfg["action_slot_dim"] = 3
    env_cfg["state_dim"] = {"distribution": "uniform_int", "min": 7, "max": 8}
    env_cfg["reinforce_sequence_replay_share_context_forward"] = True
    prior = EnvironmentPrior(env_cfg)
    bardist = make_standardized_full_support_bar_distribution()

    class _DummyReplayModel:
        next_state_flow_dim = 8

        def resolve_replay_batch_chunk_size(self, *, seq_len, total_batch):
            del seq_len
            return int(total_batch)

        def get_normalized_q_value_bardist(self):
            return bardist

    def _shared_policy_and_aux_outputs(
        x_tokens,
        y_tokens,
        *,
        full_length,
        eval_start=0,
        q_query_tokens=None,
        q_query_start=0,
        q_query_stop=None,
        q_action_query=None,
        flow_query_tokens=None,
        flow_query_start=0,
        flow_query_stop=None,
        flow_action_query=None,
        flow_matching_xt=None,
        flow_matching_t=None,
    ):
        del y_tokens, full_length, q_query_start, q_query_stop, flow_query_start, flow_query_stop, flow_matching_t
        assert q_query_tokens is not None
        assert flow_query_tokens is not None
        assert q_action_query is not None
        assert flow_action_query is not None
        assert tuple(flow_matching_xt.shape) == (2, 1, 8)
        return {
            "policy": {
                "action_mean": x_tokens[eval_start:, :, :3],
                "action_std": torch.full_like(x_tokens[eval_start:, :, :3], 0.05),
            },
            "q": {
                "normalized_q_logits": torch.zeros(
                    int(q_action_query.shape[0]),
                    int(q_action_query.shape[1]),
                    bardist.num_bars,
                    dtype=torch.float32,
                    device=q_action_query.device,
                ),
                "normalized_q": torch.zeros(
                    int(q_action_query.shape[0]),
                    int(q_action_query.shape[1]),
                    dtype=torch.float32,
                    device=q_action_query.device,
                ),
            },
            "flow": {
                "next_state_flow": torch.zeros_like(flow_matching_xt),
            },
        }

    def _dummy_step_fn(*args, **kwargs):
        raise AssertionError("padding test should not call live policy_step")

    _dummy_step_fn._reinforce_sequence_replay_fn = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("padding test should bypass live replay fallback")
    )
    _dummy_step_fn._reinforce_sequence_stream_policy_and_aux_outputs_with_kv_fn = (
        _shared_policy_and_aux_outputs
    )
    _dummy_step_fn._fit_action_dim_fn = lambda x, action_dim: x[..., :action_dim]
    _dummy_step_fn._model_ref = _DummyReplayModel()

    seq_len = 4
    eval_start = 2
    batch_size = 1
    rewards = torch.randn(seq_len, batch_size, dtype=torch.float32)
    replay_payload = {
        "reward_in": torch.randn(seq_len, batch_size, dtype=torch.float32),
        "sampled_action": torch.randn(seq_len - eval_start, batch_size, 3, dtype=torch.float32),
        "action_std": torch.full((seq_len - eval_start, batch_size), 0.05, dtype=torch.float32),
        "action_mask": torch.ones(batch_size, 3, dtype=torch.bool),
        "flow_action": torch.randn(eval_start, batch_size, 3, dtype=torch.float32),
        "flow_next_state": torch.randn(eval_start, batch_size, 7, dtype=torch.float32),
        "flow_state_mask": torch.ones(batch_size, 7, dtype=torch.bool),
        "flow_query_start": 0,
        "flow_query_stop": eval_start,
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
    assert int(stats["reinforce_sequence_replay_policy_aux_shared_context_forward"]) == 1


def test_reinforce_sequence_replay_uses_model_resolved_backward_microbatch():
    _seed_everything(9991)
    _, env_cfg = _build_small_exact_scm_env_cfg()
    env_cfg["policy_gradient_weight"] = 0.1
    prior = EnvironmentPrior(env_cfg)

    class _DummyReplayModel:
        def resolve_replay_batch_chunk_size(self, *, seq_len, total_batch):
            del seq_len
            return int(total_batch)

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

    def _dummy_step_fn(*args, **kwargs):
        raise AssertionError("chunk-size test should not call live policy_step")

    _dummy_step_fn._reinforce_sequence_replay_fn = _replay_tokens
    _dummy_step_fn._fit_action_dim_fn = lambda x, action_dim: x[..., :action_dim]
    _dummy_step_fn._model_ref = _DummyReplayModel()

    seq_len = 4
    eval_start = 2
    batch_size = 16
    action_dim = 3
    rewards = torch.randn(seq_len, batch_size, dtype=torch.float32)
    replay_payload = {
        "reward_in": torch.randn(seq_len, batch_size, dtype=torch.float32),
        "sampled_action": torch.randn(seq_len - eval_start, batch_size, action_dim, dtype=torch.float32),
        "action_std": torch.full((seq_len - eval_start, batch_size), 0.05, dtype=torch.float32),
        "action_mask": torch.ones(batch_size, action_dim, dtype=torch.bool),
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
    assert int(stats["reinforce_sequence_replay_batch_chunk"]) == 16
    assert int(stats["reinforce_sequence_replay_chunk_count"]) == 1


def test_transformer_replay_policy_sequence_outputs_allow_aux_predictions_without_action_query():
    _seed_everything(656)
    cfg = deepcopy(get_model_default_config("rlpfn"))
    cfg["transformer"]["backbone"] = "transformer"
    cfg["prior"]["environment"]["normalized_q_value_weight"] = 0.2
    cfg["prior"]["environment"]["next_state_flow_matching_weight"] = 0.5
    cfg["prior"]["environment"]["next_state_flow_head_type"] = "mlp"
    cfg["transformer"]["emsize"] = 64
    cfg["transformer"]["nlayers"] = 2
    cfg["transformer"]["nhead"] = 4
    _, model, *_ = get_model(cfg, device="cpu", should_train=False, verbose=False)
    model.eval()

    seq_len = 6
    batch_size = 2
    eval_start = 2
    flow_dim = _resolve_dim_upper_bound(cfg["prior"]["environment"]["state_dim"])
    with torch.no_grad():
        outputs = model.replay_policy_sequence_outputs(
            torch.randn(seq_len, batch_size, int(cfg["prior"]["num_features"])),
            torch.randn(seq_len, batch_size),
            eval_start=eval_start,
            flow_matching_xt=torch.randn(seq_len - eval_start, batch_size, flow_dim),
            flow_matching_t=torch.rand(seq_len - eval_start, batch_size, 1),
        )
    assert tuple(outputs["next_state_flow"].shape) == (seq_len - eval_start, batch_size, flow_dim)


@pytest.mark.parametrize("backbone", ["transformer", "rwkv7"])
def test_predict_query_replay_outputs_with_kv_emits_aux_shapes(backbone):
    _ensure_torch_extensions_dir()
    _seed_everything(657)
    cfg = deepcopy(get_model_default_config("rlpfn"))
    cfg["transformer"]["backbone"] = backbone
    cfg["prior"]["environment"]["normalized_q_value_weight"] = 0.2
    cfg["prior"]["environment"]["next_state_flow_matching_weight"] = 0.5
    cfg["prior"]["environment"]["next_state_flow_head_type"] = "mlp"
    cfg["transformer"]["emsize"] = 64
    cfg["transformer"]["nlayers"] = 2
    if backbone == "transformer":
        cfg["transformer"]["nhead"] = 4
    else:
        cfg["transformer"]["rwkv_head_size"] = 64
    _, model, *_ = get_model(cfg, device="cpu", should_train=False, verbose=False)
    model.eval()

    seq_len = 6
    batch_size = 2
    prefix_len = 3
    flow_dim = _resolve_dim_upper_bound(cfg["prior"]["environment"]["state_dim"])
    x_tokens = torch.randn(seq_len, batch_size, int(cfg["prior"]["num_features"]))
    y_tokens = torch.randn(seq_len, batch_size)
    x_query = torch.zeros(1, batch_size, int(cfg["prior"]["num_features"]))
    kv_cache = model.init_kv_cache(x_tokens[:prefix_len], y_tokens[:prefix_len])
    with torch.no_grad():
        outputs = model.predict_query_replay_outputs_with_kv(
            x_query,
            kv_cache,
            flow_matching_xt=torch.randn(1, batch_size, flow_dim),
            flow_matching_t=torch.rand(1, batch_size, 1),
        )
    assert tuple(outputs["action_mean"].shape[:2]) == (1, batch_size)
    assert tuple(outputs["normalized_q"].shape) == (1, batch_size)
    assert tuple(outputs["normalized_q_logits"].shape) == (
        1,
        batch_size,
        model.normalized_q_value_bardist.num_bars,
    )
    assert tuple(outputs["next_state_flow"].shape) == (1, batch_size, flow_dim)


def test_rwkv7_replay_aux_sequence_outputs_skips_policy_decode(monkeypatch):
    _ensure_torch_extensions_dir()
    _seed_everything(658)
    cfg = deepcopy(get_model_default_config("rlpfn"))
    cfg["transformer"]["backbone"] = "rwkv7"
    cfg["prior"]["environment"]["normalized_q_value_weight"] = 0.2
    cfg["prior"]["environment"]["next_state_flow_matching_weight"] = 0.5
    cfg["prior"]["environment"]["next_state_flow_head_type"] = "mlp"
    cfg["transformer"]["emsize"] = 64
    cfg["transformer"]["nlayers"] = 2
    cfg["transformer"]["rwkv_head_size"] = 64
    _, model, *_ = get_model(cfg, device="cpu", should_train=False, verbose=False)
    model.eval()

    monkeypatch.setattr(
        model,
        "_decode_policy_actor_outputs",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("aux sequence helper should not decode policy actor outputs")
        ),
    )
    monkeypatch.setattr(model.rwkv_core, "forward_tokens_sequence_only", lambda tokens: tokens)

    seq_len = 6
    batch_size = 2
    eval_start = 3
    action_dim = int(cfg["prior"]["environment"]["action_slot_dim"])
    flow_dim = _resolve_dim_upper_bound(cfg["prior"]["environment"]["state_dim"])
    x_tokens = torch.randn(seq_len, batch_size, int(cfg["prior"]["num_features"]))
    y_tokens = torch.randn(seq_len, batch_size)
    q_query_tokens = torch.zeros(seq_len - eval_start, batch_size, int(cfg["prior"]["num_features"]))
    flow_query_tokens = torch.zeros(eval_start, batch_size, int(cfg["prior"]["num_features"]))

    with torch.no_grad():
        outputs = model.replay_aux_sequence_outputs(
            x_tokens,
            y_tokens,
            full_length=seq_len,
            q_query_tokens=q_query_tokens,
            q_query_start=eval_start,
            q_query_stop=seq_len,
            q_action_query=torch.randn(seq_len - eval_start, batch_size, action_dim),
            flow_query_tokens=flow_query_tokens,
            flow_query_start=0,
            flow_query_stop=eval_start,
            flow_action_query=torch.randn(eval_start, batch_size, action_dim),
            flow_matching_xt=torch.randn(eval_start, batch_size, flow_dim),
            flow_matching_t=torch.rand(eval_start, batch_size, 1),
        )

    assert tuple(outputs["q"]["normalized_q_logits"].shape) == (
        seq_len - eval_start,
        batch_size,
        model.normalized_q_value_bardist.num_bars,
    )
    assert "action_mean" not in outputs["q"]
    assert tuple(outputs["flow"]["next_state_flow"].shape) == (eval_start, batch_size, flow_dim)
    assert "action_mean" not in outputs["flow"]


def test_rwkv7_replay_policy_sequence_outputs_from_train_tokens_matches_legacy(monkeypatch):
    _ensure_torch_extensions_dir()
    _seed_everything(6581)
    cfg = deepcopy(get_model_default_config("rlpfn"))
    cfg["transformer"]["backbone"] = "rwkv7"
    cfg["prior"]["environment"]["normalized_q_value_weight"] = 0.2
    cfg["prior"]["environment"]["next_state_flow_matching_weight"] = 0.5
    cfg["prior"]["environment"]["next_state_flow_head_type"] = "mlp"
    cfg["transformer"]["emsize"] = 64
    cfg["transformer"]["nlayers"] = 2
    cfg["transformer"]["rwkv_head_size"] = 64
    _, model, *_ = get_model(cfg, device="cpu", should_train=False, verbose=False)
    model.eval()
    monkeypatch.setattr(model.rwkv_core, "forward_tokens_sequence_only", lambda tokens: tokens)

    seq_len = 6
    batch_size = 2
    eval_start = 3
    action_dim = int(cfg["prior"]["environment"]["action_slot_dim"])
    flow_dim = _resolve_dim_upper_bound(cfg["prior"]["environment"]["state_dim"])
    x_tokens = torch.randn(seq_len, batch_size, int(cfg["prior"]["num_features"]))
    y_tokens = torch.randn(seq_len, batch_size)
    action_query = torch.randn(seq_len - eval_start, batch_size, action_dim)
    flow_matching_xt = torch.randn(seq_len - eval_start, batch_size, flow_dim)
    flow_matching_t = torch.rand(seq_len - eval_start, batch_size, 1)

    with torch.no_grad():
        train_tokens = model.encode_train_sequence_tokens(x_tokens, y_tokens)
        legacy = model.replay_policy_sequence_outputs(
            x_tokens,
            y_tokens,
            eval_start=eval_start,
            action_query=action_query,
            flow_matching_xt=flow_matching_xt,
            flow_matching_t=flow_matching_t,
        )
        shared = model.replay_policy_sequence_outputs_from_train_tokens(
            train_tokens,
            eval_start=eval_start,
            action_query=action_query,
            flow_matching_xt=flow_matching_xt,
            flow_matching_t=flow_matching_t,
        )

    assert legacy.keys() == shared.keys()
    for key in legacy:
        torch.testing.assert_close(shared[key], legacy[key], atol=1e-6, rtol=1e-6)


def test_rwkv7_replay_aux_sequence_outputs_from_train_tokens_matches_legacy(monkeypatch):
    _ensure_torch_extensions_dir()
    _seed_everything(6582)
    cfg = deepcopy(get_model_default_config("rlpfn"))
    cfg["transformer"]["backbone"] = "rwkv7"
    cfg["prior"]["environment"]["normalized_q_value_weight"] = 0.2
    cfg["prior"]["environment"]["next_state_flow_matching_weight"] = 0.5
    cfg["prior"]["environment"]["next_state_flow_head_type"] = "mlp"
    cfg["transformer"]["emsize"] = 64
    cfg["transformer"]["nlayers"] = 2
    cfg["transformer"]["rwkv_head_size"] = 64
    _, model, *_ = get_model(cfg, device="cpu", should_train=False, verbose=False)
    model.eval()
    monkeypatch.setattr(model.rwkv_core, "forward_tokens_sequence_only", lambda tokens: tokens)

    seq_len = 6
    batch_size = 2
    eval_start = 3
    action_dim = int(cfg["prior"]["environment"]["action_slot_dim"])
    flow_dim = _resolve_dim_upper_bound(cfg["prior"]["environment"]["state_dim"])
    x_tokens = torch.randn(seq_len, batch_size, int(cfg["prior"]["num_features"]))
    y_tokens = torch.randn(seq_len, batch_size)
    train_tokens = model.encode_train_sequence_tokens(x_tokens, y_tokens)
    q_query_tokens = torch.zeros(seq_len - eval_start, batch_size, int(cfg["prior"]["num_features"]))
    flow_query_tokens = torch.zeros(eval_start, batch_size, int(cfg["prior"]["num_features"]))
    q_action_query = torch.randn(seq_len - eval_start, batch_size, action_dim)
    flow_action_query = torch.randn(eval_start, batch_size, action_dim)
    flow_matching_xt = torch.randn(eval_start, batch_size, flow_dim)
    flow_matching_t = torch.rand(eval_start, batch_size, 1)

    with torch.no_grad():
        legacy = model.replay_aux_sequence_outputs(
            x_tokens,
            y_tokens,
            full_length=seq_len,
            q_query_tokens=q_query_tokens,
            q_query_start=eval_start,
            q_query_stop=seq_len,
            q_action_query=q_action_query,
            flow_query_tokens=flow_query_tokens,
            flow_query_start=0,
            flow_query_stop=eval_start,
            flow_action_query=flow_action_query,
            flow_matching_xt=flow_matching_xt,
            flow_matching_t=flow_matching_t,
        )
        shared = model.replay_aux_sequence_outputs_from_train_tokens(
            train_tokens,
            full_length=seq_len,
            q_query_tokens=q_query_tokens,
            q_query_start=eval_start,
            q_query_stop=seq_len,
            q_action_query=q_action_query,
            flow_query_tokens=flow_query_tokens,
            flow_query_start=0,
            flow_query_stop=eval_start,
            flow_action_query=flow_action_query,
            flow_matching_xt=flow_matching_xt,
            flow_matching_t=flow_matching_t,
        )

    torch.testing.assert_close(
        shared["q"]["normalized_q_logits"],
        legacy["q"]["normalized_q_logits"],
        atol=1e-6,
        rtol=1e-6,
    )
    torch.testing.assert_close(
        shared["flow"]["next_state_flow"],
        legacy["flow"]["next_state_flow"],
        atol=1e-6,
        rtol=1e-6,
    )


def test_rwkv7_stream_replay_policy_and_aux_outputs_with_kv_routes_single_context_traversal(monkeypatch):
    _ensure_torch_extensions_dir()
    _seed_everything(65821)
    cfg = deepcopy(get_model_default_config("rlpfn"))
    cfg["transformer"]["backbone"] = "rwkv7"
    cfg["prior"]["environment"]["normalized_q_value_weight"] = 0.2
    cfg["prior"]["environment"]["next_state_flow_matching_weight"] = 0.5
    cfg["prior"]["environment"]["next_state_flow_head_type"] = "mlp"
    cfg["transformer"]["emsize"] = 64
    cfg["transformer"]["nlayers"] = 2
    cfg["transformer"]["rwkv_head_size"] = 64
    _, model, *_ = get_model(cfg, device="cpu", should_train=False, verbose=False)
    model.eval()

    action_dim = int(cfg["prior"]["environment"]["action_slot_dim"])
    flow_dim = _resolve_dim_upper_bound(cfg["prior"]["environment"]["state_dim"])
    eval_start = 3
    seq_len = 5
    batch_size = 2
    q_steps = seq_len - eval_start
    flow_steps = 2
    q_query_start = eval_start
    flow_query_start = 1
    flow_query_stop = flow_query_start + flow_steps
    q_query_tokens = torch.zeros(q_steps, batch_size, int(cfg["prior"]["num_features"]))
    flow_query_tokens = torch.zeros(flow_steps, batch_size, int(cfg["prior"]["num_features"]))
    q_action_query = torch.randn(q_steps, batch_size, action_dim)
    flow_action_query = torch.randn(flow_steps, batch_size, action_dim)
    flow_matching_xt = torch.randn(flow_steps, batch_size, flow_dim)
    flow_matching_t = torch.rand(flow_steps, batch_size, 1)

    x_tokens = torch.zeros(seq_len, batch_size, int(cfg["prior"]["num_features"]), dtype=torch.float32)
    for token_idx in range(seq_len):
        x_tokens[token_idx, :, 0] = float(token_idx + 1)
    y_tokens = torch.zeros(seq_len, batch_size, dtype=torch.float32)

    event_log = []

    monkeypatch.setattr(
        model,
        "resolve_replay_batch_chunk_size",
        lambda *, seq_len, total_batch: int(total_batch),
    )

    def _init_kv_cache(x_prefix, y_prefix):
        del y_prefix
        event_log.append(("init", int(x_prefix.shape[0])))
        if int(x_prefix.shape[0]) == 0:
            return None
        return x_prefix[:, :, :1].sum(dim=0)

    def _forward_policy_step_actor(
        x_token,
        y_token,
        kv_cache=None,
        policy_action_head_params_override=None,
    ):
        del y_token, policy_action_head_params_override
        token_value = x_token[0, :, :1]
        next_cache = token_value if kv_cache is None else (kv_cache + token_value)
        event_log.append(("policy", int(token_value[0, 0].item())))
        return {
            "action_mean": next_cache.unsqueeze(0).expand(1, int(next_cache.shape[0]), action_dim),
            "action_std": torch.full(
                (1, int(next_cache.shape[0]), action_dim),
                0.05,
                dtype=x_token.dtype,
                device=x_token.device,
            ),
        }, next_cache

    def _append_train_token_to_kv(
        x_token,
        y_token,
        kv_cache,
        max_cache_len=None,
        kv_cache_mode="auto",
        kv_cache_page_size=None,
        allow_grad_mutable_cache=False,
        allow_grad_inplace_paged_cache=False,
    ):
        del y_token, max_cache_len, kv_cache_mode, kv_cache_page_size
        del allow_grad_mutable_cache, allow_grad_inplace_paged_cache
        token_value = x_token[0, :, :1]
        next_cache = token_value if kv_cache is None else (kv_cache + token_value)
        event_log.append(("append", int(token_value[0, 0].item())))
        return next_cache

    def _predict_query_aux_outputs_with_kv(
        x_query,
        kv_cache,
        *,
        action_query=None,
        include_normalized_q_logits=False,
        flow_matching_xt=None,
        flow_matching_t=None,
    ):
        del x_query, action_query, flow_matching_t
        if include_normalized_q_logits:
            event_log.append(("q", int(kv_cache[0, 0].item())))
            logits = kv_cache.unsqueeze(0).expand(
                1,
                int(kv_cache.shape[0]),
                model.normalized_q_value_bardist.num_bars,
            )
            return {
                "normalized_q_logits": logits,
                "normalized_q": model.normalized_q_value_bardist.mean(logits),
            }
        event_log.append(("flow", int(kv_cache[0, 0].item())))
        return {
            "next_state_flow": kv_cache.unsqueeze(0).expand_as(flow_matching_xt),
        }

    monkeypatch.setattr(model, "init_kv_cache", _init_kv_cache)
    monkeypatch.setattr(model, "append_train_token_to_kv", _append_train_token_to_kv)
    monkeypatch.setattr(model, "forward_policy_step_actor", _forward_policy_step_actor)
    monkeypatch.setattr(model, "predict_query_aux_outputs_with_kv", _predict_query_aux_outputs_with_kv)

    with torch.no_grad():
        outputs = model.stream_replay_policy_and_aux_outputs_with_kv(
            x_tokens,
            y_tokens,
            full_length=seq_len,
            eval_start=eval_start,
            q_query_tokens=q_query_tokens,
            q_query_start=q_query_start,
            q_query_stop=seq_len,
            q_action_query=q_action_query,
            flow_query_tokens=flow_query_tokens,
            flow_query_start=flow_query_start,
            flow_query_stop=flow_query_stop,
            flow_action_query=flow_action_query,
            flow_matching_xt=flow_matching_xt,
            flow_matching_t=flow_matching_t,
        )

    assert event_log == [
        ("init", 1),
        ("append", 2),
        ("flow", 3),
        ("append", 3),
        ("flow", 6),
        ("policy", 4),
        ("q", 10),
        ("policy", 5),
        ("q", 15),
    ]
    assert tuple(outputs["policy"]["action_mean"].shape) == (q_steps, batch_size, action_dim)
    assert tuple(outputs["q"]["normalized_q_logits"].shape) == (
        q_steps,
        batch_size,
        model.normalized_q_value_bardist.num_bars,
    )
    assert tuple(outputs["flow"]["next_state_flow"].shape) == (flow_steps, batch_size, flow_dim)
    torch.testing.assert_close(
        outputs["policy"]["action_mean"][:, :, 0],
        torch.tensor([[10.0, 10.0], [15.0, 15.0]]),
        atol=1e-6,
        rtol=1e-6,
    )
    torch.testing.assert_close(
        outputs["flow"]["next_state_flow"][:, :, 0],
        torch.tensor([[3.0, 3.0], [6.0, 6.0]]),
        atol=1e-6,
        rtol=1e-6,
    )


def test_rwkv7_replay_policy_and_aux_sequence_outputs_from_train_tokens_uses_single_official_forward(monkeypatch):
    _ensure_torch_extensions_dir()
    _seed_everything(65822)
    cfg = deepcopy(get_model_default_config("rlpfn"))
    cfg["transformer"]["backbone"] = "rwkv7"
    cfg["prior"]["environment"]["normalized_q_value_weight"] = 0.2
    cfg["prior"]["environment"]["next_state_flow_matching_weight"] = 0.5
    cfg["prior"]["environment"]["next_state_flow_head_type"] = "mlp"
    cfg["prior"]["environment"]["reinforce_sequence_replay_share_context_forward"] = True
    cfg["transformer"]["emsize"] = 64
    cfg["transformer"]["nlayers"] = 2
    cfg["transformer"]["rwkv_head_size"] = 64
    _, model, *_ = get_model(cfg, device="cpu", should_train=False, verbose=False)
    model.eval()

    action_dim = int(cfg["prior"]["environment"]["action_slot_dim"])
    flow_dim = _resolve_dim_upper_bound(cfg["prior"]["environment"]["state_dim"])
    eval_start = 3
    seq_len = 5
    batch_size = 2
    q_steps = seq_len - eval_start
    flow_steps = eval_start
    q_query_tokens = torch.zeros(q_steps, batch_size, int(cfg["prior"]["num_features"]))
    flow_query_tokens = torch.zeros(flow_steps, batch_size, int(cfg["prior"]["num_features"]))
    q_action_query = torch.randn(q_steps, batch_size, action_dim)
    flow_action_query = torch.randn(flow_steps, batch_size, action_dim)
    flow_matching_xt = torch.randn(flow_steps, batch_size, flow_dim)
    flow_matching_t = torch.rand(flow_steps, batch_size, 1)
    x_tokens = torch.randn(seq_len, batch_size, int(cfg["prior"]["num_features"]), dtype=torch.float32)
    y_tokens = torch.randn(seq_len, batch_size, dtype=torch.float32)
    train_tokens = model.encode_train_sequence_tokens(x_tokens, y_tokens)

    call_count = {"n": 0}
    def _counting_forward(tokens):
        call_count["n"] += 1
        return tokens

    monkeypatch.setattr(
        model,
        "resolve_replay_batch_chunk_size",
        lambda *, seq_len, total_batch: int(total_batch),
    )
    monkeypatch.setattr(model.rwkv_core, "forward_tokens_sequence_only", _counting_forward)

    with torch.no_grad():
        outputs = model.replay_policy_and_aux_sequence_outputs_from_train_tokens(
            train_tokens,
            full_length=seq_len,
            eval_start=eval_start,
            q_query_tokens=q_query_tokens,
            q_query_start=eval_start,
            q_query_stop=seq_len,
            q_action_query=q_action_query,
            flow_query_tokens=flow_query_tokens,
            flow_query_start=0,
            flow_query_stop=eval_start,
            flow_action_query=flow_action_query,
            flow_matching_xt=flow_matching_xt,
            flow_matching_t=flow_matching_t,
        )

    assert call_count["n"] == 1
    assert tuple(outputs["policy"]["action_mean"].shape) == (q_steps, batch_size, action_dim)
    assert tuple(outputs["q"]["normalized_q_logits"].shape) == (
        q_steps,
        batch_size,
        model.normalized_q_value_bardist.num_bars,
    )
    assert tuple(outputs["flow"]["next_state_flow"].shape) == (flow_steps, batch_size, flow_dim)


def test_reinforce_sequence_replay_shared_context_forward_matches_separate_paths():
    _seed_everything(99942)
    _, base_env_cfg = _build_small_exact_scm_env_cfg()
    base_env_cfg["policy_gradient_weight"] = 0.1
    base_env_cfg["normalized_q_value_weight"] = 0.25
    base_env_cfg["next_state_flow_matching_weight"] = 0.5
    base_env_cfg["obs_slot_dim"] = 8
    base_env_cfg["action_slot_dim"] = 3
    bardist = make_standardized_full_support_bar_distribution()

    shared_env_cfg = deepcopy(base_env_cfg)
    shared_env_cfg["reinforce_sequence_replay_share_context_forward"] = True
    shared_prior = EnvironmentPrior(shared_env_cfg)

    legacy_env_cfg = deepcopy(base_env_cfg)
    legacy_env_cfg["reinforce_sequence_replay_share_context_forward"] = False
    legacy_env_cfg["reinforce_sequence_replay_share_train_token_encoding"] = False
    legacy_prior = EnvironmentPrior(legacy_env_cfg)

    class _DummyReplayModel:
        def resolve_replay_batch_chunk_size(self, *, seq_len, total_batch):
            del seq_len
            return int(total_batch)

        def get_normalized_q_value_bardist(self):
            return bardist

    def _policy_outputs(x_tokens, *, eval_start):
        query = x_tokens[eval_start:]
        action_mean = (0.1 * query[..., :3]) + 0.02
        action_std = torch.full_like(action_mean, 0.07)
        return {
            "action_mean": action_mean,
            "action_std": action_std,
        }

    def _q_outputs(q_action_query):
        logits = q_action_query.sum(dim=-1, keepdim=True).expand(
            int(q_action_query.shape[0]),
            int(q_action_query.shape[1]),
            bardist.num_bars,
        )
        return {
            "normalized_q_logits": logits,
            "normalized_q": bardist.mean(logits),
        }

    def _flow_outputs(flow_action_query, flow_matching_xt):
        base = flow_action_query.mean(dim=-1, keepdim=True)
        return {
            "next_state_flow": torch.zeros_like(flow_matching_xt) + base.expand_as(flow_matching_xt),
        }

    def _make_shared_step_fn():
        def _dummy_step_fn(*args, **kwargs):
            raise AssertionError("shared compare test should not call live policy_step")

        def _shared_outputs(
            x_tokens,
            y_tokens,
            *,
            full_length,
            eval_start=0,
            q_query_tokens=None,
            q_query_start=0,
            q_query_stop=None,
            q_action_query=None,
            flow_query_tokens=None,
            flow_query_start=0,
            flow_query_stop=None,
            flow_action_query=None,
            flow_matching_xt=None,
            flow_matching_t=None,
        ):
            del y_tokens, full_length, q_query_tokens, q_query_start, q_query_stop
            del flow_query_tokens, flow_query_start, flow_query_stop, flow_matching_t
            outputs = {
                "policy": _policy_outputs(x_tokens, eval_start=eval_start),
                "q": _q_outputs(q_action_query),
                "flow": _flow_outputs(flow_action_query, flow_matching_xt),
            }
            return outputs

        _dummy_step_fn._reinforce_sequence_replay_fn = lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("shared compare test should bypass live replay fallback")
        )
        _dummy_step_fn._reinforce_sequence_replay_outputs_fn = lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("shared compare test should bypass legacy policy replay")
        )
        _dummy_step_fn._reinforce_sequence_replay_aux_sequence_outputs_fn = (
            lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError("shared compare test should bypass legacy aux replay")
            )
        )
        _dummy_step_fn._reinforce_sequence_encode_train_tokens_fn = lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("shared compare test should bypass shared train-token encoding")
        )
        _dummy_step_fn._reinforce_sequence_replay_outputs_from_train_tokens_fn = (
            lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError("shared compare test should bypass policy-from-train-tokens replay")
            )
        )
        _dummy_step_fn._reinforce_sequence_replay_aux_sequence_outputs_from_train_tokens_fn = (
            lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError("shared compare test should bypass aux-from-train-tokens replay")
            )
        )
        _dummy_step_fn._reinforce_sequence_stream_replay_aux_outputs_with_kv_fn = (
            lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError("shared compare test should bypass aux-only streaming fallback")
            )
        )
        _dummy_step_fn._reinforce_sequence_replay_policy_and_aux_sequence_outputs_fn = _shared_outputs
        _dummy_step_fn._reinforce_sequence_stream_policy_and_aux_outputs_with_kv_fn = (
            lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError("shared compare test should prefer official shared replay helper")
            )
        )
        _dummy_step_fn._fit_action_dim_fn = lambda x, action_dim: x[..., :action_dim]
        _dummy_step_fn._model_ref = _DummyReplayModel()
        return _dummy_step_fn

    def _make_legacy_step_fn():
        def _dummy_step_fn(*args, **kwargs):
            raise AssertionError("legacy compare test should not call live policy_step")

        def _legacy_policy_outputs(
            x_tokens,
            y_tokens,
            *,
            eval_start=0,
            flow_matching_xt=None,
            flow_matching_t=None,
        ):
            del y_tokens, flow_matching_xt, flow_matching_t
            return _policy_outputs(x_tokens, eval_start=eval_start)

        def _legacy_aux_outputs(
            x_tokens,
            y_tokens,
            *,
            full_length,
            q_query_tokens=None,
            q_query_start=0,
            q_query_stop=None,
            q_action_query=None,
            flow_query_tokens=None,
            flow_query_start=0,
            flow_query_stop=None,
            flow_action_query=None,
            flow_matching_xt=None,
            flow_matching_t=None,
        ):
            del x_tokens, y_tokens, full_length, q_query_tokens, q_query_start, q_query_stop
            del flow_query_tokens, flow_query_start, flow_query_stop, flow_matching_t
            return {
                "q": _q_outputs(q_action_query),
                "flow": _flow_outputs(flow_action_query, flow_matching_xt),
            }

        _dummy_step_fn._reinforce_sequence_replay_fn = lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("legacy compare test should use replay outputs path")
        )
        _dummy_step_fn._reinforce_sequence_replay_outputs_fn = _legacy_policy_outputs
        _dummy_step_fn._reinforce_sequence_replay_aux_sequence_outputs_fn = _legacy_aux_outputs
        _dummy_step_fn._reinforce_sequence_init_kv_cache_fn = lambda *args, **kwargs: None
        _dummy_step_fn._reinforce_sequence_append_train_token_to_kv_fn = lambda *args, **kwargs: None
        _dummy_step_fn._reinforce_sequence_predict_query_outputs_with_kv_fn = lambda *args, **kwargs: {}
        _dummy_step_fn._fit_action_dim_fn = lambda x, action_dim: x[..., :action_dim]
        _dummy_step_fn._model_ref = _DummyReplayModel()
        return _dummy_step_fn

    seq_len = 5
    eval_start = 2
    batch_size = 2
    state_dim = _resolve_dim_upper_bound(base_env_cfg["state_dim"])
    action_dim = int(base_env_cfg["action_slot_dim"])
    x_tokens = torch.randn(seq_len, batch_size, 12, dtype=torch.float32)
    rewards = torch.randn(seq_len, batch_size, dtype=torch.float32)
    replay_payload = {
        "reward_in": torch.randn(seq_len, batch_size, dtype=torch.float32),
        "sampled_action": torch.randn(seq_len - eval_start, batch_size, action_dim, dtype=torch.float32),
        "action_std": torch.full((seq_len - eval_start, batch_size), 0.05, dtype=torch.float32),
        "action_mask": torch.ones(batch_size, action_dim, dtype=torch.bool),
        "next_state": torch.randn(seq_len - eval_start, batch_size, state_dim, dtype=torch.float32),
        "state_mask": torch.ones(batch_size, state_dim, dtype=torch.bool),
        "flow_action": torch.randn(eval_start, batch_size, action_dim, dtype=torch.float32),
        "flow_next_state": torch.randn(eval_start, batch_size, state_dim, dtype=torch.float32),
        "flow_state_mask": torch.ones(batch_size, state_dim, dtype=torch.bool),
        "flow_query_start": 0,
        "flow_query_stop": eval_start,
        "eval_start": eval_start,
        "full_length": seq_len,
    }

    _seed_everything(99943)
    shared_loss, shared_stats, shared_log_probs = shared_prior.reinforce_sequence_replay_loss_from_rollout(
        policy_step_fn=_make_shared_step_fn(),
        x_tokens=x_tokens,
        rewards=rewards,
        replay_payload=replay_payload,
        single_eval_pos=eval_start,
        fit_action_dim_fn=lambda x, action_dim: x[..., :action_dim],
    )
    _seed_everything(99943)
    legacy_loss, legacy_stats, legacy_log_probs = legacy_prior.reinforce_sequence_replay_loss_from_rollout(
        policy_step_fn=_make_legacy_step_fn(),
        x_tokens=x_tokens,
        rewards=rewards,
        replay_payload=replay_payload,
        single_eval_pos=eval_start,
        fit_action_dim_fn=lambda x, action_dim: x[..., :action_dim],
    )

    torch.testing.assert_close(shared_loss, legacy_loss, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(shared_log_probs, legacy_log_probs, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(
        shared_stats["policy_total_loss"],
        legacy_stats["policy_total_loss"],
        atol=1e-6,
        rtol=1e-6,
    )
    torch.testing.assert_close(
        shared_stats["normalized_q_value_loss"],
        legacy_stats["normalized_q_value_loss"],
        atol=1e-6,
        rtol=1e-6,
    )
    torch.testing.assert_close(
        shared_stats["next_state_flow_matching_loss"],
        legacy_stats["next_state_flow_matching_loss"],
        atol=1e-6,
        rtol=1e-6,
    )
    assert int(shared_stats["reinforce_sequence_replay_policy_aux_shared_context_forward"]) == 1
    assert int(legacy_stats["reinforce_sequence_replay_policy_aux_shared_context_forward"]) == 0
    assert int(shared_stats["reinforce_sequence_replay_q_flow_shared_aux_query_pass"]) == 1
    assert int(legacy_stats["reinforce_sequence_replay_q_flow_shared_aux_query_pass"]) == 1
    assert "reinforce_sequence_replay_shared_backbone_pass" not in shared_stats
    assert "reinforce_sequence_replay_shared_aux_backbone_pass" not in shared_stats


def test_reinforce_sequence_replay_prefers_shared_train_token_encoding_when_available():
    _seed_everything(9995)
    _, env_cfg = _build_small_exact_scm_env_cfg()
    env_cfg["policy_gradient_weight"] = 0.1
    env_cfg["normalized_q_value_weight"] = 0.25
    env_cfg["next_state_flow_matching_weight"] = 0.5
    env_cfg["obs_slot_dim"] = 8
    env_cfg["action_slot_dim"] = 3
    prior = EnvironmentPrior(env_cfg)
    bardist = make_standardized_full_support_bar_distribution()
    event_log = []

    class _DummyReplayModel:
        def resolve_replay_batch_chunk_size(self, *, seq_len, total_batch):
            del seq_len
            return int(total_batch)

        def get_normalized_q_value_bardist(self):
            return bardist

    def _encode_train_tokens(x_tokens, y_tokens):
        event_log.append("encode_shared")
        return x_tokens + y_tokens.unsqueeze(-1)

    def _replay_outputs_from_train_tokens(
        train_tokens,
        *,
        eval_start=0,
        action_query=None,
        flow_matching_xt=None,
        flow_matching_t=None,
    ):
        del action_query, flow_matching_xt, flow_matching_t
        event_log.append("policy_from_train_tokens")
        query = train_tokens[eval_start:]
        return {
            "action_mean": torch.zeros(
                int(query.shape[0]),
                int(query.shape[1]),
                3,
                dtype=query.dtype,
                device=query.device,
            ),
        }

    def _replay_aux_from_train_tokens(
        train_tokens,
        *,
        full_length,
        q_query_tokens=None,
        q_query_start=0,
        q_query_stop=None,
        q_action_query=None,
        flow_query_tokens=None,
        flow_query_start=0,
        flow_query_stop=None,
        flow_action_query=None,
        flow_matching_xt=None,
        flow_matching_t=None,
    ):
        del train_tokens, full_length, q_query_stop, flow_query_stop, flow_matching_t
        event_log.append("aux_from_train_tokens")
        assert q_query_tokens is not None
        assert q_action_query is not None
        assert flow_query_tokens is not None
        assert flow_action_query is not None
        assert int(q_query_start) == 2
        assert int(flow_query_start) == 0
        outputs = {
            "q": {
                "normalized_q_logits": torch.zeros(
                    int(q_query_tokens.shape[0]),
                    int(q_query_tokens.shape[1]),
                    bardist.num_bars,
                    dtype=torch.float32,
                ),
            },
            "flow": {
                "next_state_flow": torch.zeros_like(flow_matching_xt),
            },
        }
        outputs["q"]["normalized_q"] = bardist.mean(outputs["q"]["normalized_q_logits"])
        return outputs

    def _dummy_step_fn(*args, **kwargs):
        raise AssertionError("shared-encoding test should not call live policy_step")

    _dummy_step_fn._reinforce_sequence_replay_fn = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("shared-encoding test should use replay outputs path")
    )
    _dummy_step_fn._reinforce_sequence_replay_outputs_fn = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("legacy replay outputs path should be bypassed")
    )
    _dummy_step_fn._reinforce_sequence_replay_aux_sequence_outputs_fn = (
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("legacy aux sequence path should be bypassed")
        )
    )
    _dummy_step_fn._reinforce_sequence_encode_train_tokens_fn = _encode_train_tokens
    _dummy_step_fn._reinforce_sequence_replay_outputs_from_train_tokens_fn = (
        _replay_outputs_from_train_tokens
    )
    _dummy_step_fn._reinforce_sequence_replay_aux_sequence_outputs_from_train_tokens_fn = (
        _replay_aux_from_train_tokens
    )
    _dummy_step_fn._reinforce_sequence_stream_replay_aux_outputs_with_kv_fn = (
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("shared encoded train-token path should bypass streaming aux fallback")
        )
    )
    _dummy_step_fn._reinforce_sequence_init_kv_cache_fn = lambda *args, **kwargs: None
    _dummy_step_fn._reinforce_sequence_append_train_token_to_kv_fn = lambda *args, **kwargs: None
    _dummy_step_fn._reinforce_sequence_predict_query_outputs_with_kv_fn = lambda *args, **kwargs: {}
    _dummy_step_fn._fit_action_dim_fn = lambda x, action_dim: x[..., :action_dim]
    _dummy_step_fn._model_ref = _DummyReplayModel()

    seq_len = 5
    eval_start = 2
    batch_size = 1
    state_dim = _resolve_dim_upper_bound(env_cfg["state_dim"])
    action_dim = int(env_cfg["action_slot_dim"])
    rewards = torch.randn(seq_len, batch_size, dtype=torch.float32)
    replay_payload = {
        "reward_in": torch.randn(seq_len, batch_size, dtype=torch.float32),
        "sampled_action": torch.randn(seq_len - eval_start, batch_size, action_dim, dtype=torch.float32),
        "action_std": torch.full((seq_len - eval_start, batch_size), 0.05, dtype=torch.float32),
        "action_mask": torch.ones(batch_size, action_dim, dtype=torch.bool),
        "next_state": torch.randn(seq_len - eval_start, batch_size, state_dim, dtype=torch.float32),
        "state_mask": torch.ones(batch_size, state_dim, dtype=torch.bool),
        "flow_action": torch.randn(eval_start, batch_size, action_dim, dtype=torch.float32),
        "flow_next_state": torch.randn(eval_start, batch_size, state_dim, dtype=torch.float32),
        "flow_state_mask": torch.ones(batch_size, state_dim, dtype=torch.bool),
        "flow_query_start": 0,
        "flow_query_stop": eval_start,
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
    assert event_log == ["encode_shared", "policy_from_train_tokens", "aux_from_train_tokens"]
    assert int(stats["reinforce_sequence_replay_shared_train_token_encoding"]) == 1


def test_reinforce_sequence_replay_can_disable_shared_train_token_encoding_for_legacy_compare():
    _seed_everything(9996)
    _, env_cfg = _build_small_exact_scm_env_cfg()
    env_cfg["policy_gradient_weight"] = 0.1
    env_cfg["normalized_q_value_weight"] = 0.25
    env_cfg["next_state_flow_matching_weight"] = 0.5
    env_cfg["obs_slot_dim"] = 8
    env_cfg["action_slot_dim"] = 3
    env_cfg["reinforce_sequence_replay_share_train_token_encoding"] = False
    prior = EnvironmentPrior(env_cfg)
    bardist = make_standardized_full_support_bar_distribution()
    event_log = []

    class _DummyReplayModel:
        def resolve_replay_batch_chunk_size(self, *, seq_len, total_batch):
            del seq_len
            return int(total_batch)

        def get_normalized_q_value_bardist(self):
            return bardist

    def _legacy_replay_outputs(
        x_tokens,
        y_tokens,
        *,
        eval_start=0,
        flow_matching_xt=None,
        flow_matching_t=None,
    ):
        del y_tokens, flow_matching_xt, flow_matching_t
        event_log.append("policy_legacy")
        query = x_tokens[eval_start:]
        return {
            "action_mean": torch.zeros(
                int(query.shape[0]),
                int(query.shape[1]),
                3,
                dtype=query.dtype,
                device=query.device,
            ),
        }

    def _legacy_replay_aux(
        x_tokens,
        y_tokens,
        *,
        full_length,
        q_query_tokens=None,
        q_query_start=0,
        q_query_stop=None,
        q_action_query=None,
        flow_query_tokens=None,
        flow_query_start=0,
        flow_query_stop=None,
        flow_action_query=None,
        flow_matching_xt=None,
        flow_matching_t=None,
    ):
        del x_tokens, y_tokens, full_length, q_query_stop, flow_query_stop, flow_matching_t
        event_log.append("aux_legacy")
        assert q_query_tokens is not None
        assert q_action_query is not None
        assert flow_query_tokens is not None
        assert flow_action_query is not None
        assert int(q_query_start) == 2
        assert int(flow_query_start) == 0
        outputs = {
            "q": {
                "normalized_q_logits": torch.zeros(
                    int(q_query_tokens.shape[0]),
                    int(q_query_tokens.shape[1]),
                    bardist.num_bars,
                    dtype=torch.float32,
                ),
            },
            "flow": {
                "next_state_flow": torch.zeros_like(flow_matching_xt),
            },
        }
        outputs["q"]["normalized_q"] = bardist.mean(outputs["q"]["normalized_q_logits"])
        return outputs

    def _dummy_step_fn(*args, **kwargs):
        raise AssertionError("legacy-compare test should not call live policy_step")

    _dummy_step_fn._reinforce_sequence_replay_fn = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("legacy-compare test should use replay outputs path")
    )
    _dummy_step_fn._reinforce_sequence_replay_outputs_fn = _legacy_replay_outputs
    _dummy_step_fn._reinforce_sequence_replay_aux_sequence_outputs_fn = _legacy_replay_aux
    _dummy_step_fn._reinforce_sequence_encode_train_tokens_fn = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("shared encoding should stay disabled for legacy compare")
    )
    _dummy_step_fn._reinforce_sequence_replay_outputs_from_train_tokens_fn = (
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("shared encoded policy replay should stay disabled for legacy compare")
        )
    )
    _dummy_step_fn._reinforce_sequence_replay_aux_sequence_outputs_from_train_tokens_fn = (
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("shared encoded aux replay should stay disabled for legacy compare")
        )
    )
    _dummy_step_fn._reinforce_sequence_stream_replay_aux_outputs_with_kv_fn = (
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("legacy compare should not fall through to streaming aux fallback")
        )
    )
    _dummy_step_fn._reinforce_sequence_init_kv_cache_fn = lambda *args, **kwargs: None
    _dummy_step_fn._reinforce_sequence_append_train_token_to_kv_fn = lambda *args, **kwargs: None
    _dummy_step_fn._reinforce_sequence_predict_query_outputs_with_kv_fn = lambda *args, **kwargs: {}
    _dummy_step_fn._fit_action_dim_fn = lambda x, action_dim: x[..., :action_dim]
    _dummy_step_fn._model_ref = _DummyReplayModel()

    seq_len = 5
    eval_start = 2
    batch_size = 1
    state_dim = _resolve_dim_upper_bound(env_cfg["state_dim"])
    action_dim = int(env_cfg["action_slot_dim"])
    rewards = torch.randn(seq_len, batch_size, dtype=torch.float32)
    replay_payload = {
        "reward_in": torch.randn(seq_len, batch_size, dtype=torch.float32),
        "sampled_action": torch.randn(seq_len - eval_start, batch_size, action_dim, dtype=torch.float32),
        "action_std": torch.full((seq_len - eval_start, batch_size), 0.05, dtype=torch.float32),
        "action_mask": torch.ones(batch_size, action_dim, dtype=torch.bool),
        "next_state": torch.randn(seq_len - eval_start, batch_size, state_dim, dtype=torch.float32),
        "state_mask": torch.ones(batch_size, state_dim, dtype=torch.bool),
        "flow_action": torch.randn(eval_start, batch_size, action_dim, dtype=torch.float32),
        "flow_next_state": torch.randn(eval_start, batch_size, state_dim, dtype=torch.float32),
        "flow_state_mask": torch.ones(batch_size, state_dim, dtype=torch.bool),
        "flow_query_start": 0,
        "flow_query_stop": eval_start,
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
    assert event_log == ["policy_legacy", "aux_legacy"]
    assert int(stats["reinforce_sequence_replay_shared_train_token_encoding"]) == 0


def test_dispatch_policy_rollout_omits_legacy_replay_targets_by_default():
    _seed_everything(6579)
    _, env_cfg = _build_small_exact_scm_env_cfg()
    prior = EnvironmentPrior(env_cfg)

    def _dummy_replay_fn(x_tokens, y_tokens, *, eval_start=0):
        del y_tokens
        return torch.zeros(
            int(x_tokens.shape[0] - eval_start),
            int(x_tokens.shape[1]),
            3,
            dtype=x_tokens.dtype,
            device=x_tokens.device,
        )

    def _dummy_step_fn(*args, **kwargs):
        raise AssertionError("dispatch test should use the monkeypatched rollout")

    _dummy_step_fn._reinforce_sequence_replay_fn = _dummy_replay_fn
    _dummy_step_fn._fit_action_dim_fn = lambda x, action_dim: x[..., :action_dim]

    seq_len = 5
    batch_size = 1
    num_features = 12
    eval_start = 2
    action_dim = 3
    state_dim = 8
    x_buffer = torch.empty(seq_len, batch_size, num_features, dtype=torch.float32)
    reward_buffer = torch.empty(seq_len, batch_size, dtype=torch.float32)
    infos = [None] * batch_size

    def _fake_rollout_family_group_vectorized_with_policy(*args, **kwargs):
        del args, kwargs
        x_group = torch.randn(seq_len, batch_size, num_features, dtype=torch.float32)
        y_group = torch.randn(seq_len, batch_size, dtype=torch.float32)
        prior.last_rollout_reinforce = {
            "log_probs": torch.zeros(seq_len, batch_size, dtype=torch.float32),
            "log_prob_score": None,
        }
        prior.last_rollout_reinforce_replay = {
            "reward_in": torch.randn(seq_len, batch_size, dtype=torch.float32),
            "sampled_action": torch.randn(seq_len - eval_start, batch_size, action_dim, dtype=torch.float32),
            "action_std": torch.full((seq_len - eval_start, batch_size), 0.1, dtype=torch.float32),
            "action_mask": torch.ones(batch_size, action_dim, dtype=torch.bool),
            "action_query_cols": torch.tensor([[7, 8, 9]], dtype=torch.long),
            "next_state": None,
            "state_mask": None,
            "flow_action": torch.randn(eval_start, batch_size, action_dim, dtype=torch.float32),
            "flow_next_state": torch.randn(eval_start, batch_size, state_dim, dtype=torch.float32),
            "flow_state_mask": torch.ones(batch_size, state_dim, dtype=torch.bool),
            "flow_query_start": 0,
            "flow_query_stop": eval_start,
            "eval_start": eval_start,
            "full_length": seq_len,
        }
        prior.last_rollout_reward_components = None
        prior.last_rollout_eval_terminal_counts = torch.zeros(batch_size, dtype=torch.float32)
        prior.last_rollout_policy_trace = None
        prior.last_rollout_terminal_stats = {}
        prior.last_rollout_profile = None
        return x_group, y_group, [None] * batch_size

    prior._rollout_family_group_vectorized_with_policy = _fake_rollout_family_group_vectorized_with_policy

    ctx = {
        "policy_step_fn": _dummy_step_fn,
        "batch_size": batch_size,
        "n_samples": seq_len,
        "num_features": num_features,
        "single_eval_pos": eval_start,
        "device": "cpu",
        "collect_x": True,
        "collect_runtime_info": False,
        "tbptt_window": None,
        "tbptt_reward_sink_supports_aux": False,
        "store_rewards": True,
        "policy_objective_kind": "reinforce",
        "_policy_collect_log_probs": False,
        "_policy_collect_action_trace": False,
        "_policy_collect_reinforce_replay": True,
        "_policy_detach_action_in_env": None,
        "_policy_disable_log_probs": True,
        "_policy_force_no_grad": True,
        "_policy_defer_reinforce_replay": True,
        "alpha_grad_trace_roots_only": False,
        "backend": "torch_vectorized",
        "strict_rng_match": False,
        "grouping_mode": "family",
        "h_list": [dict(env_cfg)],
        "env_seeds": None,
        "rollout_seeds": None,
        "make_alpha_grad_tbptt_group_sink": lambda group_indices: None,
        "alpha_grad_outer_merge_state": {
            "enabled": False,
            "window_buckets": None,
            "next_flush": 0,
            "expected_group_count": 0,
        },
        "x": x_buffer,
        "rewards": reward_buffer,
        "reinforce_log_probs": None,
        "infos": infos,
    }

    rollout = dispatch_policy_rollout(prior, ctx)
    assert isinstance(rollout, dict)

    replay_payload = prior.last_rollout_reinforce_replay
    assert isinstance(replay_payload, dict)
    assert replay_payload.get("next_state", None) is None
    assert replay_payload.get("state_mask", None) is None
    assert tuple(replay_payload["flow_action"].shape) == (eval_start, batch_size, action_dim)
    assert tuple(replay_payload["flow_next_state"].shape) == (eval_start, batch_size, state_dim)
    assert tuple(replay_payload["flow_state_mask"].shape) == (batch_size, state_dim)


def test_dispatch_policy_rollout_preserves_full_state_replay_payload():
    _seed_everything(658)
    _, env_cfg = _build_small_exact_scm_env_cfg()
    env_cfg["reinforce_sequence_replay_store_legacy_targets"] = True
    prior = EnvironmentPrior(env_cfg)

    def _dummy_replay_fn(x_tokens, y_tokens, *, eval_start=0):
        del y_tokens
        return torch.zeros(
            int(x_tokens.shape[0] - eval_start),
            int(x_tokens.shape[1]),
            3,
            dtype=x_tokens.dtype,
            device=x_tokens.device,
        )

    def _dummy_step_fn(*args, **kwargs):
        raise AssertionError("dispatch test should use the monkeypatched rollout")

    _dummy_step_fn._reinforce_sequence_replay_fn = _dummy_replay_fn
    _dummy_step_fn._fit_action_dim_fn = lambda x, action_dim: x[..., :action_dim]

    seq_len = 5
    batch_size = 1
    num_features = 12
    eval_start = 2
    action_dim = 3
    state_dim = 8
    x_buffer = torch.empty(seq_len, batch_size, num_features, dtype=torch.float32)
    reward_buffer = torch.empty(seq_len, batch_size, dtype=torch.float32)
    infos = [None] * batch_size

    def _fake_rollout_family_group_vectorized_with_policy(*args, **kwargs):
        del args, kwargs
        x_group = torch.randn(seq_len, batch_size, num_features, dtype=torch.float32)
        y_group = torch.randn(seq_len, batch_size, dtype=torch.float32)
        prior.last_rollout_reinforce = {
            "log_probs": torch.zeros(seq_len, batch_size, dtype=torch.float32),
            "log_prob_score": None,
        }
        prior.last_rollout_reinforce_replay = {
            "reward_in": torch.randn(seq_len, batch_size, dtype=torch.float32),
            "sampled_action": torch.randn(seq_len - eval_start, batch_size, action_dim, dtype=torch.float32),
            "action_std": torch.full((seq_len - eval_start, batch_size), 0.1, dtype=torch.float32),
            "action_mask": torch.ones(batch_size, action_dim, dtype=torch.bool),
            "action_query_cols": torch.tensor([[7, 8, 9]], dtype=torch.long),
            "next_state": torch.randn(seq_len - eval_start, batch_size, state_dim, dtype=torch.float32),
            "state_mask": torch.ones(batch_size, state_dim, dtype=torch.bool),
            "flow_action": torch.randn(eval_start, batch_size, action_dim, dtype=torch.float32),
            "flow_next_state": torch.randn(eval_start, batch_size, state_dim, dtype=torch.float32),
            "flow_state_mask": torch.ones(batch_size, state_dim, dtype=torch.bool),
            "flow_query_start": 0,
            "flow_query_stop": eval_start,
            "eval_start": eval_start,
            "full_length": seq_len,
        }
        prior.last_rollout_reward_components = None
        prior.last_rollout_eval_terminal_counts = torch.zeros(batch_size, dtype=torch.float32)
        prior.last_rollout_policy_trace = None
        prior.last_rollout_terminal_stats = {}
        prior.last_rollout_profile = None
        return x_group, y_group, [None] * batch_size

    prior._rollout_family_group_vectorized_with_policy = _fake_rollout_family_group_vectorized_with_policy

    ctx = {
        "policy_step_fn": _dummy_step_fn,
        "batch_size": batch_size,
        "n_samples": seq_len,
        "num_features": num_features,
        "single_eval_pos": eval_start,
        "device": "cpu",
        "collect_x": True,
        "collect_runtime_info": False,
        "tbptt_window": None,
        "tbptt_reward_sink_supports_aux": False,
        "store_rewards": True,
        "policy_objective_kind": "reinforce",
        "_policy_collect_log_probs": False,
        "_policy_collect_action_trace": False,
        "_policy_collect_reinforce_replay": True,
        "_policy_detach_action_in_env": None,
        "_policy_disable_log_probs": True,
        "_policy_force_no_grad": True,
        "_policy_defer_reinforce_replay": True,
        "alpha_grad_trace_roots_only": False,
        "backend": "torch_vectorized",
        "strict_rng_match": False,
        "grouping_mode": "family",
        "h_list": [dict(env_cfg)],
        "env_seeds": None,
        "rollout_seeds": None,
        "make_alpha_grad_tbptt_group_sink": lambda group_indices: None,
        "alpha_grad_outer_merge_state": {
            "enabled": False,
            "window_buckets": None,
            "next_flush": 0,
            "expected_group_count": 0,
        },
        "x": x_buffer,
        "rewards": reward_buffer,
        "reinforce_log_probs": None,
        "infos": infos,
    }

    rollout = dispatch_policy_rollout(prior, ctx)
    assert isinstance(rollout, dict)

    replay_payload = prior.last_rollout_reinforce_replay
    assert isinstance(replay_payload, dict)
    assert torch.is_tensor(replay_payload.get("next_state", None))
    assert torch.is_tensor(replay_payload.get("state_mask", None))
    assert tuple(replay_payload["next_state"].shape) == (seq_len - eval_start, batch_size, state_dim)
    assert tuple(replay_payload["state_mask"].shape) == (batch_size, state_dim)
    assert torch.equal(replay_payload["action_query_cols"], torch.tensor([[7, 8, 9]], dtype=torch.long))
    assert tuple(replay_payload["flow_action"].shape) == (eval_start, batch_size, action_dim)
    assert tuple(replay_payload["flow_next_state"].shape) == (eval_start, batch_size, state_dim)
    assert tuple(replay_payload["flow_state_mask"].shape) == (batch_size, state_dim)
    assert int(replay_payload["flow_query_start"]) == 0
    assert int(replay_payload["flow_query_stop"]) == eval_start


def test_rollout_with_policy_flow_replay_works_without_legacy_targets():
    _seed_everything(660)
    cfg, env_cfg = _build_small_exact_scm_env_cfg()
    env_cfg["reinforce_sequence_replay_enabled"] = True
    env_cfg["normalized_q_value_weight"] = 0.0
    env_cfg["next_state_flow_matching_weight"] = 0.5
    env_cfg["reinforce_sequence_replay_store_legacy_targets"] = False
    prior = EnvironmentPrior(env_cfg)

    def _concretize_env_value(value):
        if not isinstance(value, dict):
            return value
        if "value" in value:
            return value["value"]
        choice_values = value.get("choice_values", None)
        if isinstance(choice_values, (list, tuple)) and len(choice_values) > 0:
            return choice_values[0]
        if "max" in value:
            return value["max"]
        if "min" in value:
            return value["min"]
        if "lower_bound" in value:
            return value["lower_bound"]
        return value

    env_h = {key: _concretize_env_value(value) for key, value in env_cfg.items()}

    batch_size = 1
    n_samples = 4
    single_eval_pos = 1
    num_features = int(cfg["prior"]["num_features"])
    action_dim = _resolve_dim_upper_bound(env_cfg["action_dim"])
    state_dim = _resolve_dim_upper_bound(env_cfg["state_dim"])

    def _dummy_policy_step_fn(obs_t, action_t, reward_t, reward_mask_t, cache, t, env_info):
        del action_t, reward_t, reward_mask_t, cache, t, env_info
        action_mean = torch.zeros(
            int(obs_t.shape[0]),
            action_dim,
            device=obs_t.device,
            dtype=obs_t.dtype,
        )
        return {
            "action_mean": action_mean,
            "action_std": torch.full_like(action_mean, 0.1),
        }

    rollout = prior.rollout_with_policy(
        policy_step_fn=_dummy_policy_step_fn,
        batch_size=batch_size,
        n_samples=n_samples,
        num_features=num_features,
        device="cpu",
        single_eval_pos=single_eval_pos,
        collect_x=True,
        collect_runtime_info=False,
        store_rewards=True,
        policy_objective_kind="reinforce",
        h_list_override=[env_h],
        env_seeds_override=[123],
        rollout_seeds_override=[456],
        _policy_collect_reinforce_replay=True,
        _policy_disable_log_probs=True,
        _policy_force_no_grad=True,
        _policy_defer_reinforce_replay=True,
    )

    assert isinstance(rollout, dict)
    replay_payload = prior.last_rollout_reinforce_replay
    assert isinstance(replay_payload, dict)
    assert replay_payload.get("next_obs", None) is None
    assert replay_payload.get("obs_mask", None) is None
    assert replay_payload.get("next_state", None) is None
    assert replay_payload.get("state_mask", None) is None
    assert tuple(replay_payload["flow_action"].shape) == (single_eval_pos, batch_size, action_dim)
    assert tuple(replay_payload["flow_next_state"].shape) == (single_eval_pos, batch_size, state_dim)
    assert tuple(replay_payload["flow_state_mask"].shape) == (batch_size, state_dim)


def test_rollout_with_policy_flow_replay_pads_to_state_upper_bound():
    _seed_everything(661)
    cfg, env_cfg = _build_small_exact_scm_env_cfg()
    env_cfg["reinforce_sequence_replay_enabled"] = True
    env_cfg["normalized_q_value_weight"] = 0.0
    env_cfg["next_state_flow_matching_weight"] = 0.5
    env_cfg["reinforce_sequence_replay_store_legacy_targets"] = False
    env_cfg["state_dim"] = {"distribution": "uniform_int", "min": 7, "max": 8}
    prior = EnvironmentPrior(env_cfg)

    def _concretize_env_value(value):
        if not isinstance(value, dict):
            return value
        if "value" in value:
            return value["value"]
        choice_values = value.get("choice_values", None)
        if isinstance(choice_values, (list, tuple)) and len(choice_values) > 0:
            return choice_values[0]
        if "max" in value:
            return value["max"]
        if "min" in value:
            return value["min"]
        if "lower_bound" in value:
            return value["lower_bound"]
        return value

    env_h = {key: _concretize_env_value(value) for key, value in env_cfg.items()}
    env_h["state_dim"] = 7

    batch_size = 1
    n_samples = 4
    single_eval_pos = 2
    num_features = int(cfg["prior"]["num_features"])
    action_dim = _resolve_dim_upper_bound(env_cfg["action_dim"])
    state_dim_upper = _resolve_dim_upper_bound(env_cfg["state_dim"])

    def _dummy_policy_step_fn(obs_t, action_t, reward_t, reward_mask_t, cache, t, env_info):
        del action_t, reward_t, reward_mask_t, cache, t, env_info
        action_mean = torch.zeros(
            int(obs_t.shape[0]),
            action_dim,
            device=obs_t.device,
            dtype=obs_t.dtype,
        )
        return {
            "action_mean": action_mean,
            "action_std": torch.full_like(action_mean, 0.1),
        }

    rollout = prior.rollout_with_policy(
        policy_step_fn=_dummy_policy_step_fn,
        batch_size=batch_size,
        n_samples=n_samples,
        num_features=num_features,
        device="cpu",
        single_eval_pos=single_eval_pos,
        collect_x=True,
        collect_runtime_info=False,
        store_rewards=True,
        policy_objective_kind="reinforce",
        h_list_override=[env_h],
        env_seeds_override=[123],
        rollout_seeds_override=[456],
        _policy_collect_reinforce_replay=True,
        _policy_disable_log_probs=True,
        _policy_force_no_grad=True,
        _policy_defer_reinforce_replay=True,
    )

    assert isinstance(rollout, dict)
    replay_payload = prior.last_rollout_reinforce_replay
    assert isinstance(replay_payload, dict)
    assert tuple(replay_payload["flow_next_state"].shape) == (single_eval_pos, batch_size, state_dim_upper)
    assert tuple(replay_payload["flow_state_mask"].shape) == (batch_size, state_dim_upper)
    assert bool(replay_payload["flow_state_mask"][0, 6].item()) is True
    assert bool(replay_payload["flow_state_mask"][0, 7].item()) is False


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
            terminal_counts=None,
        ):
            del rewards, baseline_mode, fit_action_dim_fn, terminal_counts
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
    assert torch.allclose(
        base["log_probs"][eval_start:],
        replay["log_probs"],
        atol=3e-3,
        rtol=1e-6,
    )
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

    assert torch.allclose(loss_base, loss_replay, atol=1e-4, rtol=1e-6)
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

    assert torch.allclose(loss_base, loss_sink, atol=1e-4, rtol=1e-6)
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
    original_split_actor = model.forward_policy_step_split_actor

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

    def _wrapped_split_actor(
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
        policy_action_head_params_override=None,
    ):
        split_calls["count"] += 1
        split_cache_seen.append(kv_cache is not None)
        return original_split_actor(
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
            policy_action_head_params_override=policy_action_head_params_override,
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
    monkeypatch.setattr(model, "forward_policy_step_split_actor", types.MethodType(_wrapped_split_actor, model))
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
        _seed_everything(919191)
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
