import random
import numpy as np
import pytest
import torch

import ticl.train as train_mod
from ticl.model_configs import get_prior_config
from ticl.models.encoders import Linear
import ticl.models.layer as layer_mod
from ticl.models.tabpfn import TabPFN
from ticl.priors.environment_prior import EnvironmentPrior
from ticl.train import (
    _apply_aev5_next_step_corridor_,
    _build_policy_step_fn,
    _compute_policy_rollout_chunk_loss,
    _is_oom_exception,
    _set_policy_inner_recompute_attn,
    train_epoch_policy_gradient,
)


def _seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _build_tiny_policy_model(recompute_attn=False):
    model = TabPFN(
        n_out=1,
        n_features=16,
        emsize=32,
        nhead=1,
        nhid_factor=2,
        nlayers=2,
        dropout=0.0,
        y_encoder_layer=Linear(1, emsize=32),
        classification_task=False,
        y_encoder="linear",
        single_eval_causal=True,
        recompute_attn=bool(recompute_attn),
    )
    model.train()
    return model


def _fixed_env_cfg():
    cfg = dict(get_prior_config()["prior"]["environment"])
    cfg["family"] = {"distribution": "meta_choice", "choice_values": ["scm"]}
    cfg["state_dim"] = {"distribution": "uniform_int", "min": 8, "max": 8}
    cfg["obs_dim"] = {"distribution": "uniform_int", "min": 6, "max": 6}
    cfg["action_dim"] = {"distribution": "uniform_int", "min": 3, "max": 3}
    cfg["noise_dim"] = {"distribution": "uniform_int", "min": 4, "max": 4}
    cfg["zero_pad_dim"] = {"distribution": "uniform_int", "min": 2, "max": 2}
    cfg["obs_slot_dim"] = 12
    cfg["action_slot_dim"] = 2
    cfg["num_layers"] = {"distribution": "uniform_int", "min": 2, "max": 2}
    cfg["prior_mlp_hidden_dim"] = {"distribution": "uniform_int", "min": 12, "max": 12}
    cfg["prior_mlp_activations"] = "tanh"
    cfg["reward_dropout_enabled"] = False
    cfg["init_state_std"] = 0.1
    cfg["init_action_std"] = 0.1
    cfg["state_noise_std"] = 0.0
    cfg["action_noise_train_std"] = 0.0
    cfg["action_noise_eval_std"] = 0.0
    cfg["init_std"] = 0.05
    cfg["noise_std"] = 0.0
    return cfg


def _run_chunk(
    policy_rollout_checkpoint,
    policy_rollout_checkpoint_reentrant=False,
    pg_saved_tensors_cpu_offload=False,
    pg_torch_compile=False,
    kv_cache_mode="immutable",
    kv_cache_page_size=None,
    allow_grad_mutable_cache=False,
    pg_tbptt_window=None,
    env_cfg=None,
    batch_size=1,
    n_samples=12,
    single_eval_pos=6,
    recompute_attn=False,
):
    _seed_everything(20260227)
    model = _build_tiny_policy_model(recompute_attn=recompute_attn)
    prior = EnvironmentPrior(dict(env_cfg) if env_cfg is not None else _fixed_env_cfg())
    step_fn = _build_policy_step_fn(
        model,
        num_features=16,
        max_cache_len=n_samples,
        kv_cache_mode=kv_cache_mode,
        kv_cache_page_size=kv_cache_page_size,
        allow_grad_mutable_cache=allow_grad_mutable_cache,
        pg_torch_compile=pg_torch_compile,
        pg_torch_compile_backend="eager",
        pg_torch_compile_mode="reduce-overhead",
        pg_torch_compile_fullgraph=False,
        pg_torch_compile_dynamic=False,
    )
    loss, _, stats = _compute_policy_rollout_chunk_loss(
        env_prior=prior,
        policy_step_fn=step_fn,
        batch_size=batch_size,
        n_samples=n_samples,
        num_features=16,
        device="cpu",
        single_eval_pos=single_eval_pos,
        collect_x=False,
        policy_rollout_checkpoint=policy_rollout_checkpoint,
        policy_rollout_checkpoint_reentrant=policy_rollout_checkpoint_reentrant,
        pg_saved_tensors_cpu_offload=pg_saved_tensors_cpu_offload,
        pg_saved_tensors_pin_memory=True,
        pg_tbptt_window=pg_tbptt_window,
    )
    model.zero_grad(set_to_none=True)
    loss.backward()
    grads = [p.grad.detach().clone() for p in model.parameters() if p.grad is not None]
    stats_detached = {
        k: (v.detach().clone() if torch.is_tensor(v) else torch.as_tensor(v))
        for k, v in stats.items()
    }
    return loss.detach().clone(), stats_detached, grads


def test_train_keeps_aggregate_k_gradients_fixed_when_adaptive_batch_size_is_enabled(monkeypatch):
    observed_aggregate = []
    observed_epochs = []

    def fake_train_epoch(
        model,
        aggregate_k_gradients,
        using_dist,
        scaler,
        dl,
        device,
        optimizer,
        criterion,
        n_out,
        epoch_idx=None,
        progress_bar=False,
        epoch_profiler=None,
        epoch_start_time=None,
        train_profiler_log_every_batches=0,
        verbose=False,
        kernel_profiler=None,
    ):
        observed_aggregate.append(int(aggregate_k_gradients))
        observed_epochs.append(int(epoch_idx))
        return 1.0, 0.0, 0.0

    monkeypatch.setattr(train_mod, "train_epoch", fake_train_epoch)

    class _DummyDL:
        def __len__(self):
            return 1

    model = torch.nn.Linear(1, 1)
    model.n_out = 1
    criterion = torch.nn.MSELoss()

    total_loss, _, _, final_epoch = train_mod.train(
        dl=_DummyDL(),
        model=model,
        criterion=criterion,
        epochs=60,
        learning_rate=1e-3,
        min_lr=1e-4,
        warmup_epochs=1,
        device="cpu",
        aggregate_k_gradients=1,
        adaptive_batch_size=True,
        learning_rate_schedule="cosine",
        verbose=False,
        rl_objective="supervised",
    )

    assert float(total_loss) == 1.0
    assert int(final_epoch) == 60
    assert observed_epochs == list(range(1, 61))
    assert observed_aggregate == [1] * 60


def test_policy_rollout_checkpoint_matches_no_checkpoint_semantics():
    loss_base, stats_base, grads_base = _run_chunk(policy_rollout_checkpoint=False)
    loss_ckpt, stats_ckpt, grads_ckpt = _run_chunk(policy_rollout_checkpoint=True)

    assert torch.allclose(loss_base, loss_ckpt, atol=1e-6, rtol=1e-5)
    assert torch.allclose(stats_base["objective"], stats_ckpt["objective"], atol=1e-6, rtol=1e-5)
    assert torch.allclose(stats_base["reward_mean"], stats_ckpt["reward_mean"], atol=1e-6, rtol=1e-5)
    assert torch.allclose(stats_base["reward_std"], stats_ckpt["reward_std"], atol=1e-6, rtol=1e-5)
    assert len(grads_base) == len(grads_ckpt)
    for g_base, g_ckpt in zip(grads_base, grads_ckpt):
        assert torch.allclose(g_base, g_ckpt, atol=1e-6, rtol=1e-5)


def test_policy_rollout_saved_tensors_offload_matches_baseline_semantics():
    loss_base, stats_base, grads_base = _run_chunk(
        policy_rollout_checkpoint=False,
        pg_saved_tensors_cpu_offload=False,
    )
    loss_offload, stats_offload, grads_offload = _run_chunk(
        policy_rollout_checkpoint=False,
        pg_saved_tensors_cpu_offload=True,
    )

    assert torch.allclose(loss_base, loss_offload, atol=1e-6, rtol=1e-5)
    assert torch.allclose(stats_base["objective"], stats_offload["objective"], atol=1e-6, rtol=1e-5)
    assert torch.allclose(stats_base["reward_mean"], stats_offload["reward_mean"], atol=1e-6, rtol=1e-5)
    assert torch.allclose(stats_base["reward_std"], stats_offload["reward_std"], atol=1e-6, rtol=1e-5)
    assert len(grads_base) == len(grads_offload)
    for g_base, g_offload in zip(grads_base, grads_offload):
        assert torch.allclose(g_base, g_offload, atol=1e-6, rtol=1e-5)


def test_policy_rollout_checkpoint_and_offload_match_baseline_semantics():
    loss_base, stats_base, grads_base = _run_chunk(
        policy_rollout_checkpoint=False,
        pg_saved_tensors_cpu_offload=False,
    )
    loss_combo, stats_combo, grads_combo = _run_chunk(
        policy_rollout_checkpoint=True,
        pg_saved_tensors_cpu_offload=True,
    )

    assert torch.allclose(loss_base, loss_combo, atol=1e-6, rtol=1e-5)
    assert torch.allclose(stats_base["objective"], stats_combo["objective"], atol=1e-6, rtol=1e-5)
    assert torch.allclose(stats_base["reward_mean"], stats_combo["reward_mean"], atol=1e-6, rtol=1e-5)
    assert torch.allclose(stats_base["reward_std"], stats_combo["reward_std"], atol=1e-6, rtol=1e-5)
    assert len(grads_base) == len(grads_combo)
    for g_base, g_combo in zip(grads_base, grads_combo):
        assert torch.allclose(g_base, g_combo, atol=1e-6, rtol=1e-5)


def test_policy_rollout_checkpoint_preserves_aev5_next_stats_and_semantics():
    env_cfg = _fixed_env_cfg()
    env_cfg["anti_explosion_vanishing_v5_enabled"] = False
    env_cfg["anti_explosion_vanishing_v5_next_enabled"] = True
    env_cfg["anti_explosion_vanishing_v5_next_state_gain_lo"] = 0.985
    env_cfg["anti_explosion_vanishing_v5_next_state_gain_hi"] = 1.035
    env_cfg["anti_explosion_vanishing_v5_next_state_rms_lo"] = 4e-3
    env_cfg["anti_explosion_vanishing_v5_next_state_rms_hi"] = 9e-2
    env_cfg["anti_explosion_vanishing_v5_next_loss_target_std"] = 0.25
    env_cfg["anti_explosion_vanishing_v5_next_loss_scale_lo"] = 0.5
    env_cfg["anti_explosion_vanishing_v5_next_loss_scale_hi"] = 4.0

    loss_base, stats_base, grads_base = _run_chunk(
        policy_rollout_checkpoint=False,
        env_cfg=env_cfg,
        pg_tbptt_window=4,
        n_samples=12,
        single_eval_pos=6,
    )
    loss_ckpt, stats_ckpt, grads_ckpt = _run_chunk(
        policy_rollout_checkpoint=True,
        env_cfg=env_cfg,
        pg_tbptt_window=4,
        n_samples=12,
        single_eval_pos=6,
    )

    assert torch.allclose(loss_base, loss_ckpt, atol=1e-6, rtol=1e-5)
    for key in (
        "objective",
        "reward_mean",
        "reward_std",
        "objective_with_aev5_next",
        "aev5_next_loss_mul",
        "aev5_next_scale",
        "aev5_next_scale_raw",
        "aev5_next_reward_std_ref",
        "aev5_next_gain_mean",
        "aev5_next_gain_std",
        "aev5_next_gain_min",
        "aev5_next_gain_max",
        "aev5_next_update_rms_mean",
        "aev5_next_update_rms_std",
        "aev5_next_scale_mean",
        "aev5_next_scale_max",
        "aev5_next_high_clip_share",
        "aev5_next_low_active_share",
        "aev5_next_low_boost_share",
        "aev5_next_corridor_trigger_share",
        "aev5_next_bias_thermostat_abs_offset",
    ):
        assert torch.allclose(stats_base[key], stats_ckpt[key], atol=1e-6, rtol=1e-5)
    assert float(stats_base["aev5_next_enabled"]) == float(stats_ckpt["aev5_next_enabled"]) == 1.0
    assert len(grads_base) == len(grads_ckpt)
    for g_base, g_ckpt in zip(grads_base, grads_ckpt):
        assert torch.allclose(g_base, g_ckpt, atol=1e-6, rtol=1e-5)


def test_policy_rollout_tbptt_exposes_consistent_aev5_next_semantics():
    env_cfg = _fixed_env_cfg()
    env_cfg["anti_explosion_vanishing_v5_enabled"] = False
    env_cfg["anti_explosion_vanishing_v5_next_enabled"] = True
    env_cfg["anti_explosion_vanishing_v5_next_state_gain_lo"] = 0.985
    env_cfg["anti_explosion_vanishing_v5_next_state_gain_hi"] = 1.035
    env_cfg["anti_explosion_vanishing_v5_next_state_rms_lo"] = 4e-3
    env_cfg["anti_explosion_vanishing_v5_next_state_rms_hi"] = 9e-2
    env_cfg["anti_explosion_vanishing_v5_next_loss_target_std"] = 0.25
    env_cfg["anti_explosion_vanishing_v5_next_loss_scale_lo"] = 0.5
    env_cfg["anti_explosion_vanishing_v5_next_loss_scale_hi"] = 4.0

    loss_full, stats_full, grads_full = _run_chunk(
        policy_rollout_checkpoint=False,
        env_cfg=env_cfg,
        pg_tbptt_window=None,
        n_samples=12,
        single_eval_pos=6,
    )
    loss_tbptt, stats_tbptt, grads_tbptt = _run_chunk(
        policy_rollout_checkpoint=False,
        env_cfg=env_cfg,
        pg_tbptt_window=4,
        n_samples=12,
        single_eval_pos=6,
    )

    assert torch.allclose(loss_full, loss_tbptt, atol=1e-6, rtol=1e-5)
    for key in (
        "objective",
        "reward_mean",
        "reward_std",
    ):
        assert torch.allclose(stats_full[key], stats_tbptt[key], atol=1e-6, rtol=1e-5)
    for key in (
        "objective_with_aev5_next",
        "aev5_next_loss_mul",
        "aev5_next_scale",
        "aev5_next_scale_raw",
        "aev5_next_reward_std_ref",
        "aev5_next_corridor_trigger_share",
        "aev5_next_bias_thermostat_abs_offset",
    ):
        assert torch.isfinite(stats_tbptt[key])
    assert float(stats_tbptt["aev5_next_scale"]) == float(stats_tbptt["aev5_next_loss_mul"])
    if abs(float(stats_tbptt["aev5_next_loss_mul"]) - 1.0) > 1e-6:
        assert not torch.allclose(
            stats_tbptt["objective_with_aev5_next"],
            stats_tbptt["objective"],
            atol=1e-6,
            rtol=1e-5,
        )
    assert len(grads_full) == len(grads_tbptt)


def test_aev5_next_step_corridor_boosts_and_clips_uniformly():
    model = torch.nn.Linear(4, 2, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[1.0, -2.0, 3.0, -4.0], [0.5, -0.5, 1.5, -1.5]]))

    model.weight.grad = torch.full_like(model.weight, 1e-5)
    stats_low = {
        "grad_rms": float(model.weight.grad.pow(2).mean().sqrt().item()),
    }
    cfg = {
        "enabled": True,
        "step_grad_rms_lo": 1e-4,
        "step_grad_rms_hi": 1e-2,
        "step_reward_std_gate": 0.05,
        "step_low_boost_cap": 4.0,
        "eps": 1e-6,
    }
    grad_before = model.weight.grad.detach().clone()
    out_low = _apply_aev5_next_step_corridor_(model, grad_stats=stats_low, reward_std=0.2, cfg=cfg)
    grad_after_low = model.weight.grad.detach().clone()

    assert out_low["enabled"] is True
    assert out_low["high_clip"] is False
    assert out_low["low_active"] is True
    assert out_low["low_boost"] is True
    assert out_low["triggered"] is True
    assert out_low["abs_offset"] > 0.0
    assert out_low["scale"] > 1.0
    assert torch.allclose(grad_after_low, grad_before * float(out_low["scale"]), atol=1e-9, rtol=1e-6)

    model.weight.grad = torch.full_like(model.weight, 5e-2)
    stats_high = {
        "grad_rms": float(model.weight.grad.pow(2).mean().sqrt().item()),
    }
    grad_before_high = model.weight.grad.detach().clone()
    out_high = _apply_aev5_next_step_corridor_(model, grad_stats=stats_high, reward_std=0.2, cfg=cfg)
    grad_after_high = model.weight.grad.detach().clone()

    assert out_high["enabled"] is True
    assert out_high["high_clip"] is True
    assert out_high["low_active"] is False
    assert out_high["low_boost"] is False
    assert out_high["triggered"] is True
    assert out_high["abs_offset"] > 0.0
    assert 0.0 < out_high["scale"] < 1.0
    assert torch.allclose(grad_after_high, grad_before_high * float(out_high["scale"]), atol=1e-9, rtol=1e-6)


def test_policy_rollout_compile_matches_baseline_semantics():
    loss_base, stats_base, grads_base = _run_chunk(
        policy_rollout_checkpoint=False,
        pg_saved_tensors_cpu_offload=False,
        pg_torch_compile=False,
    )
    loss_comp, stats_comp, grads_comp = _run_chunk(
        policy_rollout_checkpoint=False,
        pg_saved_tensors_cpu_offload=False,
        pg_torch_compile=True,
    )

    assert torch.allclose(loss_base, loss_comp, atol=1e-6, rtol=1e-5)
    assert torch.allclose(stats_base["objective"], stats_comp["objective"], atol=1e-6, rtol=1e-5)
    assert torch.allclose(stats_base["reward_mean"], stats_comp["reward_mean"], atol=1e-6, rtol=1e-5)
    assert torch.allclose(stats_base["reward_std"], stats_comp["reward_std"], atol=1e-6, rtol=1e-5)
    assert len(grads_base) == len(grads_comp)
    for g_base, g_comp in zip(grads_base, grads_comp):
        assert torch.allclose(g_base, g_comp, atol=1e-6, rtol=1e-5)


def test_policy_rollout_checkpoint_static_mutable_kv_matches_immutable_semantics():
    loss_base, stats_base, grads_base = _run_chunk(
        policy_rollout_checkpoint=True,
        policy_rollout_checkpoint_reentrant=False,
        kv_cache_mode="immutable",
        allow_grad_mutable_cache=False,
    )
    loss_static, stats_static, grads_static = _run_chunk(
        policy_rollout_checkpoint=True,
        policy_rollout_checkpoint_reentrant=False,
        kv_cache_mode="static",
        allow_grad_mutable_cache=True,
    )

    assert torch.allclose(loss_base, loss_static, atol=1e-6, rtol=1e-5)
    assert torch.allclose(stats_base["objective"], stats_static["objective"], atol=1e-6, rtol=1e-5)
    assert torch.allclose(stats_base["reward_mean"], stats_static["reward_mean"], atol=1e-6, rtol=1e-5)
    assert torch.allclose(stats_base["reward_std"], stats_static["reward_std"], atol=1e-6, rtol=1e-5)
    assert len(grads_base) == len(grads_static)
    for g_base, g_static in zip(grads_base, grads_static):
        assert torch.allclose(g_base, g_static, atol=1e-6, rtol=1e-5)


def test_policy_rollout_checkpoint_reentrant_paged_mutable_kv_matches_immutable_semantics():
    loss_base, stats_base, grads_base = _run_chunk(
        policy_rollout_checkpoint=True,
        policy_rollout_checkpoint_reentrant=True,
        kv_cache_mode="immutable",
        allow_grad_mutable_cache=False,
    )
    loss_paged, stats_paged, grads_paged = _run_chunk(
        policy_rollout_checkpoint=True,
        policy_rollout_checkpoint_reentrant=True,
        kv_cache_mode="paged",
        kv_cache_page_size=1,
        allow_grad_mutable_cache=True,
    )

    assert torch.allclose(loss_base, loss_paged, atol=1e-6, rtol=1e-5)
    assert torch.allclose(stats_base["objective"], stats_paged["objective"], atol=1e-6, rtol=1e-5)
    assert torch.allclose(stats_base["reward_mean"], stats_paged["reward_mean"], atol=1e-6, rtol=1e-5)
    assert torch.allclose(stats_base["reward_std"], stats_paged["reward_std"], atol=1e-6, rtol=1e-5)
    assert len(grads_base) == len(grads_paged)
    for g_base, g_paged in zip(grads_base, grads_paged):
        assert torch.allclose(g_base, g_paged, atol=1e-6, rtol=1e-5)


def test_policy_rollout_checkpoint_reentrant_paged_mutable_kv_page4_matches_immutable_semantics():
    loss_base, stats_base, grads_base = _run_chunk(
        policy_rollout_checkpoint=True,
        policy_rollout_checkpoint_reentrant=True,
        kv_cache_mode="immutable",
        allow_grad_mutable_cache=False,
    )
    loss_paged, stats_paged, grads_paged = _run_chunk(
        policy_rollout_checkpoint=True,
        policy_rollout_checkpoint_reentrant=True,
        kv_cache_mode="paged",
        kv_cache_page_size=4,
        allow_grad_mutable_cache=True,
    )

    assert torch.allclose(loss_base, loss_paged, atol=1e-6, rtol=1e-5)
    assert torch.allclose(stats_base["objective"], stats_paged["objective"], atol=1e-6, rtol=1e-5)
    assert torch.allclose(stats_base["reward_mean"], stats_paged["reward_mean"], atol=1e-6, rtol=1e-5)
    assert torch.allclose(stats_base["reward_std"], stats_paged["reward_std"], atol=1e-6, rtol=1e-5)
    assert len(grads_base) == len(grads_paged)
    for g_base, g_paged in zip(grads_base, grads_paged):
        assert torch.allclose(g_base, g_paged, atol=1e-6, rtol=1e-5)


def test_policy_rollout_checkpoint_reentrant_matches_nonreentrant_semantics():
    loss_reent, stats_reent, grads_reent = _run_chunk(
        policy_rollout_checkpoint=True,
        policy_rollout_checkpoint_reentrant=True,
    )
    loss_non, stats_non, grads_non = _run_chunk(
        policy_rollout_checkpoint=True,
        policy_rollout_checkpoint_reentrant=False,
    )

    assert torch.allclose(loss_reent, loss_non, atol=1e-6, rtol=1e-5)
    assert torch.allclose(stats_reent["objective"], stats_non["objective"], atol=1e-6, rtol=1e-5)
    assert torch.allclose(stats_reent["reward_mean"], stats_non["reward_mean"], atol=1e-6, rtol=1e-5)
    assert torch.allclose(stats_reent["reward_std"], stats_non["reward_std"], atol=1e-6, rtol=1e-5)
    assert len(grads_reent) == len(grads_non)
    for g_reent, g_non in zip(grads_reent, grads_non):
        assert torch.allclose(g_reent, g_non, atol=1e-6, rtol=1e-5)


def test_set_policy_inner_recompute_attn_disables_layers_and_is_idempotent():
    model = _build_tiny_policy_model(recompute_attn=True)
    layers = list(model.transformer_encoder.layers)
    assert len(layers) > 0
    assert all(bool(layer.recompute_attn) for layer in layers)

    changed, total = _set_policy_inner_recompute_attn(model, enabled=False)
    assert total == len(layers)
    assert changed == len(layers)
    assert all(not bool(layer.recompute_attn) for layer in layers)

    changed_repeat, total_repeat = _set_policy_inner_recompute_attn(model, enabled=False)
    assert total_repeat == len(layers)
    assert changed_repeat == 0
    assert all(not bool(layer.recompute_attn) for layer in layers)


def test_policy_rollout_checkpoint_recompute_attn_toggle_matches_semantics():
    loss_on, stats_on, grads_on = _run_chunk(
        policy_rollout_checkpoint=True,
        policy_rollout_checkpoint_reentrant=False,
        kv_cache_mode="static",
        allow_grad_mutable_cache=True,
        recompute_attn=True,
    )
    loss_off, stats_off, grads_off = _run_chunk(
        policy_rollout_checkpoint=True,
        policy_rollout_checkpoint_reentrant=False,
        kv_cache_mode="static",
        allow_grad_mutable_cache=True,
        recompute_attn=False,
    )

    assert torch.allclose(loss_on, loss_off, atol=1e-6, rtol=1e-5)
    assert torch.allclose(stats_on["objective"], stats_off["objective"], atol=1e-6, rtol=1e-5)
    assert torch.allclose(stats_on["reward_mean"], stats_off["reward_mean"], atol=1e-6, rtol=1e-5)
    assert torch.allclose(stats_on["reward_std"], stats_off["reward_std"], atol=1e-6, rtol=1e-5)
    assert len(grads_on) == len(grads_off)
    for g_on, g_off in zip(grads_on, grads_off):
        assert torch.allclose(g_on, g_off, atol=1e-6, rtol=1e-5)


def test_policy_step_fn_detaches_reward_and_mask_before_pfn():
    _seed_everything(20260227)
    model = _build_tiny_policy_model()
    step_fn = _build_policy_step_fn(
        model,
        num_features=16,
        max_cache_len=12,
        kv_cache_mode="immutable",
        kv_cache_page_size=None,
        allow_grad_mutable_cache=False,
        pg_torch_compile=False,
    )

    obs_t = torch.randn(2, 6, dtype=torch.float32)
    action_t = torch.randn(2, 3, dtype=torch.float32)
    reward_t = torch.randn(2, 1, dtype=torch.float32, requires_grad=True)
    reward_mask_t = torch.randn(2, 1, dtype=torch.float32, requires_grad=True)
    env_info = {"obs_slot_dim": 12, "action_slot_dim": 2, "action_dim": 3}

    action_next, _ = step_fn(obs_t, action_t, reward_t, reward_mask_t, None, 0, env_info)
    loss = action_next.sum()
    grad_reward, grad_reward_mask = torch.autograd.grad(
        loss,
        (reward_t, reward_mask_t),
        allow_unused=True,
    )

    assert grad_reward is None
    assert grad_reward_mask is None


def test_policy_step_fn_prefers_split_fastpath_and_falls_back_on_slot_mismatch():
    class _DummyEncoder:
        obs_dim = 8   # obs slots + reward/mask
        action_dim = 4

    class _DummyModel:
        def __init__(self):
            self.x_encoder_type = "split_obs_action"
            self.encoder = _DummyEncoder()
            self.calls = {"split": 0, "legacy": 0}

        def forward_policy_step_split(
            self,
            obs_t,
            action_t,
            reward_t,
            reward_mask_t,
            kv_cache=None,
            **kwargs,
        ):
            del action_t, reward_t, reward_mask_t, kwargs
            self.calls["split"] += 1
            out = torch.full((1, obs_t.shape[0], 4), 7.0, dtype=obs_t.dtype, device=obs_t.device)
            return out, kv_cache

        def forward_policy_step(
            self,
            x_token,
            y_token,
            kv_cache=None,
            **kwargs,
        ):
            del y_token, kwargs
            self.calls["legacy"] += 1
            out = torch.full((1, x_token.shape[1], 4), 3.0, dtype=x_token.dtype, device=x_token.device)
            return out, kv_cache

    model = _DummyModel()
    step_fn = _build_policy_step_fn(
        model,
        num_features=16,
        max_cache_len=12,
        kv_cache_mode="immutable",
        kv_cache_page_size=None,
        allow_grad_mutable_cache=False,
        pg_torch_compile=False,
    )

    obs_t = torch.randn(2, 6, dtype=torch.float32)
    action_t = torch.randn(2, 4, dtype=torch.float32)
    reward_t = torch.randn(2, 1, dtype=torch.float32, requires_grad=True)
    reward_mask_t = torch.randn(2, 1, dtype=torch.float32, requires_grad=True)

    env_info_match = {"obs_slot_dim": 6, "action_slot_dim": 4, "action_dim": 4}
    action_next_match, _ = step_fn(obs_t, action_t, reward_t, reward_mask_t, None, 0, env_info_match)
    assert torch.allclose(action_next_match, torch.full((2, 4), 7.0))
    assert model.calls["split"] == 1
    assert model.calls["legacy"] == 0

    env_info_mismatch = {"obs_slot_dim": 5, "action_slot_dim": 4, "action_dim": 4}
    action_next_mismatch, _ = step_fn(obs_t, action_t, reward_t, reward_mask_t, None, 1, env_info_mismatch)
    assert torch.allclose(action_next_mismatch, torch.full((2, 4), 3.0))
    assert model.calls["split"] == 1
    assert model.calls["legacy"] == 1


def test_policy_rollout_chunk_torch_vectorized_matches_serial_semantics():
    env_cfg_serial = _fixed_env_cfg()
    env_cfg_serial["batch_parallel_backend"] = "python_thread"
    env_cfg_serial["batch_parallel_workers"] = 1
    env_cfg_serial["batch_vectorized_strict_rng_match"] = True

    env_cfg_vec = dict(env_cfg_serial)
    env_cfg_vec["batch_parallel_backend"] = "torch_vectorized"

    loss_serial, stats_serial, grads_serial = _run_chunk(
        policy_rollout_checkpoint=False,
        pg_saved_tensors_cpu_offload=False,
        env_cfg=env_cfg_serial,
        batch_size=4,
        n_samples=12,
        single_eval_pos=6,
    )
    loss_vec, stats_vec, grads_vec = _run_chunk(
        policy_rollout_checkpoint=False,
        pg_saved_tensors_cpu_offload=False,
        env_cfg=env_cfg_vec,
        batch_size=4,
        n_samples=12,
        single_eval_pos=6,
    )

    assert torch.allclose(loss_serial, loss_vec, atol=1e-6, rtol=1e-5)
    assert torch.allclose(stats_serial["objective"], stats_vec["objective"], atol=1e-6, rtol=1e-5)
    assert torch.allclose(stats_serial["reward_mean"], stats_vec["reward_mean"], atol=1e-6, rtol=1e-5)
    assert torch.allclose(stats_serial["reward_std"], stats_vec["reward_std"], atol=1e-6, rtol=1e-5)
    assert len(grads_serial) == len(grads_vec)
    for g_serial, g_vec in zip(grads_serial, grads_vec):
        assert torch.allclose(g_serial, g_vec, atol=1e-6, rtol=1e-5)


class _ChunkDebugEnvPrior:
    @staticmethod
    def _sample_single_eval_pos(n_samples, single_eval_pos):
        if single_eval_pos is not None:
            return int(single_eval_pos)
        return max(1, int(n_samples) // 2)


class _ChunkDebugDL:
    def __init__(self, batch_size, n_samples, num_features, steps=1):
        self.batch_size = int(batch_size)
        self.n_samples = int(n_samples)
        self.num_features = int(num_features)
        self._steps = int(steps)
        self.epoch_count = 0

    def __len__(self):
        return self._steps


def _run_train_epoch_once_for_memory_probe(
    *,
    policy_rollout_checkpoint,
    policy_rollout_checkpoint_reentrant=False,
    pg_grad_mutable_kv_cache=True,
    pg_saved_tensors_cpu_offload=False,
    pg_tbptt_window=None,
    n_samples=24,
    batch_size=2,
    recompute_attn=True,
):
    _seed_everything(20260302 + int(policy_rollout_checkpoint) + int(n_samples))
    model = _build_tiny_policy_model(recompute_attn=recompute_attn)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    dl = _ChunkDebugDL(batch_size=batch_size, n_samples=n_samples, num_features=16, steps=1)
    env_prior = EnvironmentPrior(_fixed_env_cfg())

    saved_bytes = 0

    def _pack(x):
        nonlocal saved_bytes
        if torch.is_tensor(x):
            saved_bytes += x.numel() * x.element_size()
        return x

    def _unpack(x):
        return x

    with torch.autograd.graph.saved_tensors_hooks(_pack, _unpack):
        loss, _, _ = train_epoch_policy_gradient(
            model=model,
            aggregate_k_gradients=1,
            using_dist=False,
            scaler=None,
            dl=dl,
            device="cpu",
            optimizer=optimizer,
            env_prior=env_prior,
            policy_rollout_chunk_size=1,
            policy_rollout_checkpoint=bool(policy_rollout_checkpoint),
            policy_rollout_checkpoint_reentrant=bool(policy_rollout_checkpoint_reentrant),
            pg_grad_mutable_kv_cache=bool(pg_grad_mutable_kv_cache),
            pg_saved_tensors_cpu_offload=bool(pg_saved_tensors_cpu_offload),
            pg_saved_tensors_pin_memory=False,
            pg_tbptt_window=pg_tbptt_window,
            pg_torch_compile=False,
            progress_bar=False,
        )
    assert np.isfinite(float(loss))
    return float(saved_bytes) / float(1024 ** 2)


def _run_train_epoch_once_for_tbptt_smoke(*, rl_objective="policy_gradient", n_samples=12, tbptt_window=4):
    _seed_everything(20260310 + int(n_samples))
    model = _build_tiny_policy_model(recompute_attn=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    params_before = [p.detach().clone() for p in model.parameters()]
    dl = _ChunkDebugDL(batch_size=1, n_samples=n_samples, num_features=16, steps=1)
    env_prior = EnvironmentPrior(_fixed_env_cfg())
    loss, _, _ = train_epoch_policy_gradient(
        model=model,
        aggregate_k_gradients=1,
        using_dist=False,
        scaler=None,
        dl=dl,
        device="cpu",
        optimizer=optimizer,
        env_prior=env_prior,
        policy_rollout_chunk_size=1,
        policy_rollout_checkpoint=True,
        policy_rollout_checkpoint_reentrant=True,
        pg_grad_mutable_kv_cache=True,
        pg_saved_tensors_cpu_offload=False,
        pg_saved_tensors_pin_memory=False,
        pg_tbptt_window=tbptt_window,
        pg_torch_compile=False,
        progress_bar=False,
        rl_objective=rl_objective,
    )
    assert np.isfinite(float(loss))
    params_after = [p.detach() for p in model.parameters()]
    any_param_changed = any(
        not torch.allclose(before, after)
        for before, after in zip(params_before, params_after)
    )
    assert any_param_changed
    assert optimizer.state
    return float(loss)


def test_policy_rollout_chunk_size_one_runs_per_column():
    _seed_everything(20260301)
    model = _build_tiny_policy_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    dl = _ChunkDebugDL(batch_size=5, n_samples=12, num_features=16, steps=1)
    env_prior = _ChunkDebugEnvPrior()

    import ticl.train as train_mod
    calls = []
    orig = train_mod._compute_policy_rollout_chunk_loss

    def _fake_compute(*, env_prior, policy_step_fn, batch_size, n_samples, num_features, device,
                      single_eval_pos, collect_x, policy_rollout_checkpoint,
                      policy_rollout_checkpoint_reentrant, pg_saved_tensors_cpu_offload,
                      pg_saved_tensors_pin_memory, pg_tbptt_window, tbptt_loss_sink=None, rl_objective="policy_gradient"):
        del env_prior, policy_step_fn, n_samples, num_features, single_eval_pos, collect_x
        del policy_rollout_checkpoint, policy_rollout_checkpoint_reentrant
        del pg_saved_tensors_cpu_offload, pg_saved_tensors_pin_memory, pg_tbptt_window, tbptt_loss_sink
        calls.append(int(batch_size))
        # Keep a valid autograd path for train_epoch_policy_gradient.
        anchor = next(model.parameters()).sum() * 0.0
        stats = {
            "objective": torch.zeros((), device=device),
            "reward_mean": torch.zeros((), device=device),
            "reward_std": torch.zeros((), device=device),
        }
        return anchor, None, stats

    train_mod._compute_policy_rollout_chunk_loss = _fake_compute
    try:
        loss, _, _ = train_epoch_policy_gradient(
            model=model,
            aggregate_k_gradients=1,
            using_dist=False,
            scaler=None,
            dl=dl,
            device="cpu",
            optimizer=optimizer,
            env_prior=env_prior,
            policy_rollout_chunk_size=1,
            policy_rollout_checkpoint=False,
            policy_rollout_checkpoint_reentrant=False,
            pg_grad_mutable_kv_cache=False,
            pg_saved_tensors_cpu_offload=False,
            pg_saved_tensors_pin_memory=False,
            pg_torch_compile=False,
            progress_bar=False,
        )
    finally:
        train_mod._compute_policy_rollout_chunk_loss = orig

    assert torch.isfinite(torch.tensor(loss))
    assert calls == [1, 1, 1, 1, 1]


@pytest.mark.parametrize("rl_objective", ["policy_gradient", "first_policy_gradient"])
def test_policy_rollout_tbptt_streaming_backward_smoke(rl_objective):
    loss = _run_train_epoch_once_for_tbptt_smoke(
        rl_objective=rl_objective,
        n_samples=64,
        tbptt_window=32,
    )
    assert np.isfinite(loss)


def test_policy_env_replay_steps_runs_multiple_inner_updates_per_batch():
    _seed_everything(20260315)
    model = _build_tiny_policy_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    dl = _ChunkDebugDL(batch_size=2, n_samples=12, num_features=16, steps=1)

    class _ReplayDebugEnvPrior:
        def __init__(self):
            self.eval_calls = 0
            self.hyper_calls = 0
            self.seed_calls = 0

        def _sample_single_eval_pos(self, n_samples, single_eval_pos):
            del single_eval_pos
            self.eval_calls += 1
            return max(1, int(n_samples) // 2)

        def _sample_batch_hypers(self, batch_size):
            self.hyper_calls += 1
            return [
                {
                    "reward_dropout_enabled": False,
                    "reward_dropout_randomize": False,
                    "reward_dropout_ratio": 0.0,
                    "obs_slot_dim": 12,
                    "action_slot_dim": 2,
                }
                for _ in range(int(batch_size))
            ]

        def _sample_seed_list(self, batch_size):
            self.seed_calls += 1
            start = 1000 + (self.seed_calls * 10)
            return [start + i for i in range(int(batch_size))]

    env_prior = _ReplayDebugEnvPrior()

    import ticl.train as train_mod
    calls = []
    step_count = {"n": 0}
    orig_compute = train_mod._compute_policy_rollout_chunk_loss
    orig_step = optimizer.step

    def _count_step(*args, **kwargs):
        step_count["n"] += 1
        return orig_step(*args, **kwargs)

    def _fake_compute(*, env_prior, policy_step_fn, batch_size, n_samples, num_features, device,
                      single_eval_pos, collect_x, policy_rollout_checkpoint,
                      policy_rollout_checkpoint_reentrant, pg_saved_tensors_cpu_offload,
                      pg_saved_tensors_pin_memory, pg_tbptt_window, tbptt_loss_sink=None,
                      h_list_override=None, env_seeds_override=None, rollout_seeds_override=None, rl_objective="policy_gradient"):
        del env_prior, policy_step_fn, n_samples, num_features, collect_x
        del policy_rollout_checkpoint, policy_rollout_checkpoint_reentrant
        del pg_saved_tensors_cpu_offload, pg_saved_tensors_pin_memory
        del pg_tbptt_window, tbptt_loss_sink, rollout_seeds_override
        calls.append(
            {
                "batch_size": int(batch_size),
                "single_eval_pos": int(single_eval_pos),
                "h_list_override": h_list_override,
                "env_seeds_override": env_seeds_override,
            }
        )
        anchor = next(model.parameters()).sum() * 0.0
        stats = {
            "objective": torch.zeros((), device=device),
            "reward_mean": torch.zeros((), device=device),
            "reward_std": torch.zeros((), device=device),
        }
        return anchor, None, stats

    optimizer.step = _count_step
    train_mod._compute_policy_rollout_chunk_loss = _fake_compute
    try:
        loss, _, _ = train_epoch_policy_gradient(
            model=model,
            aggregate_k_gradients=1,
            using_dist=False,
            scaler=None,
            dl=dl,
            device="cpu",
            optimizer=optimizer,
            env_prior=env_prior,
            policy_rollout_chunk_size=2,
            policy_rollout_checkpoint=False,
            policy_rollout_checkpoint_reentrant=False,
            pg_grad_mutable_kv_cache=False,
            pg_saved_tensors_cpu_offload=False,
            pg_saved_tensors_pin_memory=False,
            pg_torch_compile=False,
            pg_env_replay_steps=3,
            progress_bar=False,
        )
    finally:
        optimizer.step = orig_step
        train_mod._compute_policy_rollout_chunk_loss = orig_compute

    assert torch.isfinite(torch.tensor(loss))
    assert env_prior.eval_calls == 1
    assert env_prior.hyper_calls == 1
    assert env_prior.seed_calls == 1
    # One outer batch step, with 3 inner rollout->update cycles.
    assert step_count["n"] == 3
    assert len(calls) == 3
    assert all(call["batch_size"] == 2 for call in calls)
    assert all(call["single_eval_pos"] == calls[0]["single_eval_pos"] for call in calls)
    assert all(call["h_list_override"] == calls[0]["h_list_override"] for call in calls)
    assert all(call["env_seeds_override"] == calls[0]["env_seeds_override"] for call in calls)


def test_policy_env_replay_steps_requires_no_grad_accumulation():
    _seed_everything(20260316)
    model = _build_tiny_policy_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    dl = _ChunkDebugDL(batch_size=2, n_samples=12, num_features=16, steps=1)
    env_prior = _ChunkDebugEnvPrior()

    raised = False
    try:
        train_epoch_policy_gradient(
            model=model,
            aggregate_k_gradients=2,
            using_dist=False,
            scaler=None,
            dl=dl,
            device="cpu",
            optimizer=optimizer,
            env_prior=env_prior,
            policy_rollout_chunk_size=2,
            policy_rollout_checkpoint=False,
            policy_rollout_checkpoint_reentrant=False,
            pg_grad_mutable_kv_cache=False,
            pg_saved_tensors_cpu_offload=False,
            pg_saved_tensors_pin_memory=False,
            pg_torch_compile=False,
            pg_env_replay_steps=3,
            progress_bar=False,
        )
    except ValueError as e:
        raised = True
        assert "aggregate_k_gradients == 1" in str(e)
    assert raised


def test_policy_rollout_chunk_autotune_grows_after_oom_recovery():
    _seed_everything(20260313)
    model = _build_tiny_policy_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    dl = _ChunkDebugDL(batch_size=4, n_samples=12, num_features=16, steps=4)
    env_prior = _ChunkDebugEnvPrior()

    import ticl.train as train_mod
    calls = []
    orig = train_mod._compute_policy_rollout_chunk_loss
    oom_sizes_once = {4, 2}

    def _fake_compute(*, env_prior, policy_step_fn, batch_size, n_samples, num_features, device,
                      single_eval_pos, collect_x, policy_rollout_checkpoint,
                      policy_rollout_checkpoint_reentrant, pg_saved_tensors_cpu_offload,
                      pg_saved_tensors_pin_memory, pg_tbptt_window, tbptt_loss_sink=None, rl_objective="policy_gradient"):
        del env_prior, policy_step_fn, n_samples, num_features, single_eval_pos, collect_x
        del policy_rollout_checkpoint, policy_rollout_checkpoint_reentrant
        del pg_saved_tensors_cpu_offload, pg_saved_tensors_pin_memory, pg_tbptt_window, tbptt_loss_sink
        calls.append(int(batch_size))
        if int(batch_size) in oom_sizes_once:
            oom_sizes_once.remove(int(batch_size))
            raise RuntimeError("CUDA error: out of memory")
        anchor = next(model.parameters()).sum() * 0.0
        stats = {
            "objective": torch.zeros((), device=device),
            "reward_mean": torch.zeros((), device=device),
            "reward_std": torch.zeros((), device=device),
        }
        return anchor, None, stats

    train_mod._compute_policy_rollout_chunk_loss = _fake_compute
    try:
        loss, _, _ = train_epoch_policy_gradient(
            model=model,
            aggregate_k_gradients=1,
            using_dist=False,
            scaler=None,
            dl=dl,
            device="cpu",
            optimizer=optimizer,
            env_prior=env_prior,
            policy_rollout_chunk_size=None,
            policy_rollout_chunk_autotune=True,
            policy_rollout_chunk_grow_every=1,
            policy_rollout_chunk_grow_factor=2.0,
            policy_rollout_checkpoint=False,
            policy_rollout_checkpoint_reentrant=False,
            pg_grad_mutable_kv_cache=False,
            pg_saved_tensors_cpu_offload=False,
            pg_saved_tensors_pin_memory=False,
            pg_torch_compile=False,
            progress_bar=False,
        )
    finally:
        train_mod._compute_policy_rollout_chunk_loss = orig

    assert torch.isfinite(torch.tensor(loss))
    assert 4 in calls
    assert 1 in calls
    # One `2` comes from OOM reduction path; additional `2` calls indicate growth back after recovery.
    assert calls.count(2) >= 2


def test_policy_rollout_oom_can_reduce_tbptt_before_chunk_size():
    _seed_everything(20260314)
    model = _build_tiny_policy_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    dl = _ChunkDebugDL(batch_size=4, n_samples=12, num_features=16, steps=2)
    env_prior = _ChunkDebugEnvPrior()

    import ticl.train as train_mod
    calls = []
    orig = train_mod._compute_policy_rollout_chunk_loss
    first_oom = {"raised": False}

    def _fake_compute(*, env_prior, policy_step_fn, batch_size, n_samples, num_features, device,
                      single_eval_pos, collect_x, policy_rollout_checkpoint,
                      policy_rollout_checkpoint_reentrant, pg_saved_tensors_cpu_offload,
                      pg_saved_tensors_pin_memory, pg_tbptt_window, tbptt_loss_sink=None, rl_objective="policy_gradient"):
        del env_prior, policy_step_fn, n_samples, num_features, single_eval_pos, collect_x
        del policy_rollout_checkpoint, policy_rollout_checkpoint_reentrant
        del pg_saved_tensors_cpu_offload, pg_saved_tensors_pin_memory
        calls.append((int(batch_size), None if pg_tbptt_window is None else int(pg_tbptt_window)))
        if (not first_oom["raised"]) and pg_tbptt_window is None:
            first_oom["raised"] = True
            raise RuntimeError("CUDA error: out of memory")
        anchor = next(model.parameters()).sum() * 0.0
        stats = {
            "objective": torch.zeros((), device=device),
            "reward_mean": torch.zeros((), device=device),
            "reward_std": torch.zeros((), device=device),
        }
        return anchor, None, stats

    train_mod._compute_policy_rollout_chunk_loss = _fake_compute
    try:
        loss, _, _ = train_epoch_policy_gradient(
            model=model,
            aggregate_k_gradients=1,
            using_dist=False,
            scaler=None,
            dl=dl,
            device="cpu",
            optimizer=optimizer,
            env_prior=env_prior,
            policy_rollout_chunk_size=4,
            policy_rollout_chunk_autotune=False,
            policy_rollout_checkpoint=False,
            policy_rollout_checkpoint_reentrant=False,
            pg_grad_mutable_kv_cache=False,
            pg_saved_tensors_cpu_offload=False,
            pg_saved_tensors_pin_memory=False,
            pg_oom_debug_raise=False,
            pg_tbptt_window=None,
            pg_oom_reduce_tbptt_first=True,
            pg_torch_compile=False,
            progress_bar=False,
        )
    finally:
        train_mod._compute_policy_rollout_chunk_loss = orig

    assert torch.isfinite(torch.tensor(loss))
    # Keep batch-parallel chunk width unchanged; reduce TBPTT window after OOM.
    assert calls[0] == (4, None)
    assert any(call == (4, 6) for call in calls[1:])
    assert all(call[0] == 4 for call in calls)


def test_is_oom_exception_detects_accelerator_and_runtime_oom_forms():
    assert _is_oom_exception(RuntimeError("CUDA error: out of memory"))
    assert _is_oom_exception(RuntimeError("cudaErrorMemoryAllocation"))
    accel_cls = getattr(torch, "AcceleratorError", RuntimeError)
    assert _is_oom_exception(accel_cls("CUDA error: out of memory"))
    assert not _is_oom_exception(RuntimeError("mat1 and mat2 shapes cannot be multiplied"))


def test_policy_rollout_runtime_oom_during_backward_is_caught_and_skipped():
    _seed_everything(20260311)
    model = _build_tiny_policy_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    dl = _ChunkDebugDL(batch_size=2, n_samples=12, num_features=16, steps=1)
    env_prior = _ChunkDebugEnvPrior()

    import ticl.train as train_mod
    orig_compute = train_mod._compute_policy_rollout_chunk_loss
    orig_backward = torch.autograd.backward

    def _fake_compute(*, env_prior, policy_step_fn, batch_size, n_samples, num_features, device,
                      single_eval_pos, collect_x, policy_rollout_checkpoint,
                      policy_rollout_checkpoint_reentrant, pg_saved_tensors_cpu_offload,
                      pg_saved_tensors_pin_memory, pg_tbptt_window, tbptt_loss_sink=None, rl_objective="policy_gradient"):
        del env_prior, policy_step_fn, batch_size, n_samples, num_features, single_eval_pos, collect_x
        del policy_rollout_checkpoint, policy_rollout_checkpoint_reentrant
        del pg_saved_tensors_cpu_offload, pg_saved_tensors_pin_memory, pg_tbptt_window, tbptt_loss_sink
        anchor = next(model.parameters()).sum() * 0.0
        stats = {
            "objective": torch.zeros((), device=device),
            "reward_mean": torch.zeros((), device=device),
            "reward_std": torch.zeros((), device=device),
        }
        return anchor, None, stats

    def _fake_backward(*args, **kwargs):
        del args, kwargs
        raise RuntimeError("CUDA error: out of memory")

    train_mod._compute_policy_rollout_chunk_loss = _fake_compute
    torch.autograd.backward = _fake_backward
    try:
        loss, _, _ = train_epoch_policy_gradient(
            model=model,
            aggregate_k_gradients=1,
            using_dist=False,
            scaler=None,
            dl=dl,
            device="cpu",
            optimizer=optimizer,
            env_prior=env_prior,
            policy_rollout_chunk_size=2,
            policy_rollout_checkpoint=False,
            policy_rollout_checkpoint_reentrant=True,
            pg_grad_mutable_kv_cache=False,
            pg_saved_tensors_cpu_offload=False,
            pg_saved_tensors_pin_memory=False,
            pg_oom_debug_raise=False,
            pg_torch_compile=False,
            progress_bar=False,
        )
    finally:
        train_mod._compute_policy_rollout_chunk_loss = orig_compute
        torch.autograd.backward = orig_backward

    assert np.isinf(float(loss))


def test_policy_rollout_runtime_oom_debug_raise_propagates_exception():
    _seed_everything(20260312)
    model = _build_tiny_policy_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    dl = _ChunkDebugDL(batch_size=2, n_samples=12, num_features=16, steps=1)
    env_prior = _ChunkDebugEnvPrior()

    import ticl.train as train_mod
    orig_compute = train_mod._compute_policy_rollout_chunk_loss
    orig_backward = torch.autograd.backward

    def _fake_compute(*, env_prior, policy_step_fn, batch_size, n_samples, num_features, device,
                      single_eval_pos, collect_x, policy_rollout_checkpoint,
                      policy_rollout_checkpoint_reentrant, pg_saved_tensors_cpu_offload,
                      pg_saved_tensors_pin_memory, pg_tbptt_window, tbptt_loss_sink=None, rl_objective="policy_gradient"):
        del env_prior, policy_step_fn, batch_size, n_samples, num_features, single_eval_pos, collect_x
        del policy_rollout_checkpoint, policy_rollout_checkpoint_reentrant
        del pg_saved_tensors_cpu_offload, pg_saved_tensors_pin_memory, pg_tbptt_window, tbptt_loss_sink
        anchor = next(model.parameters()).sum() * 0.0
        stats = {
            "objective": torch.zeros((), device=device),
            "reward_mean": torch.zeros((), device=device),
            "reward_std": torch.zeros((), device=device),
        }
        return anchor, None, stats

    def _fake_backward(*args, **kwargs):
        del args, kwargs
        raise RuntimeError("CUDA error: out of memory")

    train_mod._compute_policy_rollout_chunk_loss = _fake_compute
    torch.autograd.backward = _fake_backward
    try:
        raised = False
        try:
            train_epoch_policy_gradient(
                model=model,
                aggregate_k_gradients=1,
                using_dist=False,
                scaler=None,
                dl=dl,
                device="cpu",
                optimizer=optimizer,
                env_prior=env_prior,
                policy_rollout_chunk_size=2,
                policy_rollout_checkpoint=False,
                policy_rollout_checkpoint_reentrant=True,
                pg_grad_mutable_kv_cache=False,
                pg_saved_tensors_cpu_offload=False,
                pg_saved_tensors_pin_memory=False,
                pg_oom_debug_raise=True,
                pg_torch_compile=False,
                progress_bar=False,
            )
        except RuntimeError as e:
            raised = True
            assert "out of memory" in str(e).lower()
        assert raised
    finally:
        train_mod._compute_policy_rollout_chunk_loss = orig_compute
        torch.autograd.backward = orig_backward


def _layer_recompute_flags(model):
    return [bool(layer.recompute_attn) for layer in model.transformer_encoder.layers]


def test_policy_rollout_checkpoint_temporarily_disables_inner_recompute_attn_and_restores():
    _seed_everything(20260301)
    model = _build_tiny_policy_model(recompute_attn=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    dl = _ChunkDebugDL(batch_size=3, n_samples=12, num_features=16, steps=1)
    env_prior = _ChunkDebugEnvPrior()

    import ticl.train as train_mod
    seen = []
    orig = train_mod._compute_policy_rollout_chunk_loss

    def _fake_compute(*, env_prior, policy_step_fn, batch_size, n_samples, num_features, device,
                      single_eval_pos, collect_x, policy_rollout_checkpoint,
                      policy_rollout_checkpoint_reentrant, pg_saved_tensors_cpu_offload,
                      pg_saved_tensors_pin_memory, pg_tbptt_window, tbptt_loss_sink=None, rl_objective="policy_gradient"):
        del env_prior, policy_step_fn, n_samples, num_features, single_eval_pos, collect_x
        del policy_rollout_checkpoint, policy_rollout_checkpoint_reentrant
        del pg_saved_tensors_cpu_offload, pg_saved_tensors_pin_memory, pg_tbptt_window, tbptt_loss_sink
        del batch_size
        seen.append(_layer_recompute_flags(model))
        anchor = next(model.parameters()).sum() * 0.0
        stats = {
            "objective": torch.zeros((), device=device),
            "reward_mean": torch.zeros((), device=device),
            "reward_std": torch.zeros((), device=device),
        }
        return anchor, None, stats

    train_mod._compute_policy_rollout_chunk_loss = _fake_compute
    try:
        train_epoch_policy_gradient(
            model=model,
            aggregate_k_gradients=1,
            using_dist=False,
            scaler=None,
            dl=dl,
            device="cpu",
            optimizer=optimizer,
            env_prior=env_prior,
            policy_rollout_chunk_size=1,
            policy_rollout_checkpoint=True,
            policy_rollout_checkpoint_reentrant=False,
            pg_grad_mutable_kv_cache=False,
            pg_saved_tensors_cpu_offload=False,
            pg_saved_tensors_pin_memory=False,
            pg_torch_compile=False,
            progress_bar=False,
        )
    finally:
        train_mod._compute_policy_rollout_chunk_loss = orig

    assert len(seen) == 3
    for flags in seen:
        assert all(flag is False for flag in flags)
    assert all(flag is True for flag in _layer_recompute_flags(model))


def test_policy_rollout_checkpoint_disables_layer_forward_step_checkpoint_calls():
    _seed_everything(20260303)
    model = _build_tiny_policy_model(recompute_attn=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    dl = _ChunkDebugDL(batch_size=2, n_samples=16, num_features=16, steps=1)
    env_prior = EnvironmentPrior(_fixed_env_cfg())

    original_checkpoint = layer_mod.checkpoint
    call_count = {"n": 0}

    def _counting_checkpoint(*args, **kwargs):
        call_count["n"] += 1
        return original_checkpoint(*args, **kwargs)

    layer_mod.checkpoint = _counting_checkpoint
    try:
        train_epoch_policy_gradient(
            model=model,
            aggregate_k_gradients=1,
            using_dist=False,
            scaler=None,
            dl=dl,
            device="cpu",
            optimizer=optimizer,
            env_prior=env_prior,
            policy_rollout_chunk_size=1,
            policy_rollout_checkpoint=False,
            policy_rollout_checkpoint_reentrant=False,
            pg_grad_mutable_kv_cache=True,
            pg_saved_tensors_cpu_offload=False,
            pg_saved_tensors_pin_memory=False,
            pg_torch_compile=False,
            progress_bar=False,
        )
        calls_without_rollout_ckpt = int(call_count["n"])
        call_count["n"] = 0

        train_epoch_policy_gradient(
            model=model,
            aggregate_k_gradients=1,
            using_dist=False,
            scaler=None,
            dl=dl,
            device="cpu",
            optimizer=optimizer,
            env_prior=env_prior,
            policy_rollout_chunk_size=1,
            policy_rollout_checkpoint=True,
            policy_rollout_checkpoint_reentrant=False,
            pg_grad_mutable_kv_cache=True,
            pg_saved_tensors_cpu_offload=False,
            pg_saved_tensors_pin_memory=False,
            pg_torch_compile=False,
            progress_bar=False,
        )
        calls_with_rollout_ckpt = int(call_count["n"])
    finally:
        layer_mod.checkpoint = original_checkpoint

    assert calls_without_rollout_ckpt > 0
    assert calls_with_rollout_ckpt == 0


def test_policy_rollout_checkpoint_significantly_reduces_saved_tensor_bytes():
    mem_without = _run_train_epoch_once_for_memory_probe(
        policy_rollout_checkpoint=False,
        n_samples=24,
        batch_size=2,
        recompute_attn=True,
    )
    mem_with = _run_train_epoch_once_for_memory_probe(
        policy_rollout_checkpoint=True,
        n_samples=24,
        batch_size=2,
        recompute_attn=True,
    )

    assert mem_without > 0.0
    assert mem_with <= (mem_without * 0.10)


def test_rollout_reentrant_and_nonreentrant_paths_are_memory_probe_executable():
    mem_nonreent_mutable = _run_train_epoch_once_for_memory_probe(
        policy_rollout_checkpoint=True,
        policy_rollout_checkpoint_reentrant=False,
        pg_grad_mutable_kv_cache=True,
        pg_saved_tensors_cpu_offload=False,
        n_samples=24,
        batch_size=2,
        recompute_attn=True,
    )
    mem_reent_immutable = _run_train_epoch_once_for_memory_probe(
        policy_rollout_checkpoint=True,
        policy_rollout_checkpoint_reentrant=True,
        pg_grad_mutable_kv_cache=False,
        pg_saved_tensors_cpu_offload=False,
        n_samples=24,
        batch_size=2,
        recompute_attn=True,
    )

    assert np.isfinite(mem_nonreent_mutable)
    assert np.isfinite(mem_reent_immutable)


def test_reentrant_immutable_checkpoint_matches_nonreentrant_mutable_semantics():
    loss_old, stats_old, grads_old = _run_chunk(
        policy_rollout_checkpoint=True,
        policy_rollout_checkpoint_reentrant=False,
        kv_cache_mode="static",
        allow_grad_mutable_cache=True,
    )
    loss_new, stats_new, grads_new = _run_chunk(
        policy_rollout_checkpoint=True,
        policy_rollout_checkpoint_reentrant=True,
        kv_cache_mode="immutable",
        allow_grad_mutable_cache=False,
    )

    assert torch.allclose(loss_old, loss_new, atol=1e-6, rtol=1e-5)
    assert torch.allclose(stats_old["objective"], stats_new["objective"], atol=1e-6, rtol=1e-5)
    assert torch.allclose(stats_old["reward_mean"], stats_new["reward_mean"], atol=1e-6, rtol=1e-5)
    assert torch.allclose(stats_old["reward_std"], stats_new["reward_std"], atol=1e-6, rtol=1e-5)
    assert len(grads_old) == len(grads_new)
    for g_old, g_new in zip(grads_old, grads_new):
        assert torch.allclose(g_old, g_new, atol=1e-6, rtol=1e-5)


def test_tbptt_window_equal_horizon_matches_full_horizon_semantics():
    loss_full, stats_full, grads_full = _run_chunk(
        policy_rollout_checkpoint=True,
        policy_rollout_checkpoint_reentrant=True,
        kv_cache_mode="paged",
        kv_cache_page_size=1,
        allow_grad_mutable_cache=True,
        batch_size=2,
        n_samples=12,
        single_eval_pos=6,
        pg_tbptt_window=None,
    )
    loss_tbptt, stats_tbptt, grads_tbptt = _run_chunk(
        policy_rollout_checkpoint=True,
        policy_rollout_checkpoint_reentrant=True,
        kv_cache_mode="paged",
        kv_cache_page_size=1,
        allow_grad_mutable_cache=True,
        batch_size=2,
        n_samples=12,
        single_eval_pos=6,
        pg_tbptt_window=12,
    )

    assert torch.allclose(loss_full, loss_tbptt, atol=1e-6, rtol=1e-5)
    assert torch.allclose(stats_full["objective"], stats_tbptt["objective"], atol=1e-6, rtol=1e-5)
    assert torch.allclose(stats_full["reward_mean"], stats_tbptt["reward_mean"], atol=1e-6, rtol=1e-5)
    assert torch.allclose(stats_full["reward_std"], stats_tbptt["reward_std"], atol=1e-6, rtol=1e-5)
    assert len(grads_full) == len(grads_tbptt)
    for g_full, g_tbptt in zip(grads_full, grads_tbptt):
        assert torch.allclose(g_full, g_tbptt, atol=1e-6, rtol=1e-5)


def test_tbptt_window_reduces_saved_tensor_bytes_trend():
    mem_full = _run_train_epoch_once_for_memory_probe(
        policy_rollout_checkpoint=False,
        n_samples=24,
        batch_size=2,
        recompute_attn=False,
        pg_tbptt_window=None,
    )
    mem_tbptt = _run_train_epoch_once_for_memory_probe(
        policy_rollout_checkpoint=False,
        n_samples=24,
        batch_size=2,
        recompute_attn=False,
        pg_tbptt_window=6,
    )

    assert np.isfinite(mem_full)
    assert np.isfinite(mem_tbptt)
    assert mem_tbptt < mem_full


def test_pg_phase_start_is_written_before_rollout_finishes(tmp_path):
    _seed_everything(20260318)
    model = _build_tiny_policy_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    dl = _ChunkDebugDL(batch_size=2, n_samples=12, num_features=16, steps=1)
    env_prior = _ChunkDebugEnvPrior()
    log_path = tmp_path / "phase.log"
    traceback_calls = []

    orig_compute = train_mod._compute_policy_rollout_chunk_loss
    orig_print_exc = train_mod.traceback.print_exc

    def _fake_compute(*, env_prior, policy_step_fn, batch_size, n_samples, num_features, device,
                      single_eval_pos, collect_x, policy_rollout_checkpoint,
                      policy_rollout_checkpoint_reentrant, pg_saved_tensors_cpu_offload,
                      pg_saved_tensors_pin_memory, pg_tbptt_window, tbptt_loss_sink=None, rl_objective="policy_gradient"):
        del env_prior, policy_step_fn, batch_size, n_samples, num_features, device
        del single_eval_pos, collect_x, policy_rollout_checkpoint, policy_rollout_checkpoint_reentrant
        del pg_saved_tensors_cpu_offload, pg_saved_tensors_pin_memory, pg_tbptt_window, tbptt_loss_sink, rl_objective
        raise KeyboardInterrupt()

    def _fake_print_exc(*args, **kwargs):
        del args, kwargs
        traceback_calls.append(True)

    train_mod._compute_policy_rollout_chunk_loss = _fake_compute
    train_mod.traceback.print_exc = _fake_print_exc
    try:
        with pytest.raises(KeyboardInterrupt):
            train_epoch_policy_gradient(
                model=model,
                aggregate_k_gradients=1,
                using_dist=False,
                scaler=None,
                dl=dl,
                device="cpu",
                optimizer=optimizer,
                env_prior=env_prior,
                policy_rollout_chunk_size=2,
                policy_rollout_checkpoint=False,
                policy_rollout_checkpoint_reentrant=False,
                pg_grad_mutable_kv_cache=False,
                pg_saved_tensors_cpu_offload=False,
                pg_saved_tensors_pin_memory=False,
                pg_torch_compile=False,
                pg_phase_log_every_batches=1,
                pg_phase_log_file=str(log_path),
                progress_bar=False,
            )
    finally:
        train_mod._compute_policy_rollout_chunk_loss = orig_compute
        train_mod.traceback.print_exc = orig_print_exc

    lines = log_path.read_text().splitlines()
    assert any(line.startswith("[pg-phase-start]") for line in lines)
    assert not any(line.startswith("[pg-phase]") for line in lines)
    start_line = next(line for line in lines if line.startswith("[pg-phase-start]"))
    assert "epoch=0" in start_line
    assert "batch=0" in start_line
    assert "status=start" in start_line
    assert "rl_objective=policy_gradient" in start_line
    assert traceback_calls == [True]


def test_pg_phase_start_and_finish_are_both_written(tmp_path):
    _seed_everything(20260319)
    model = _build_tiny_policy_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    dl = _ChunkDebugDL(batch_size=2, n_samples=12, num_features=16, steps=1)
    env_prior = _ChunkDebugEnvPrior()
    log_path = tmp_path / "phase.log"

    orig_compute = train_mod._compute_policy_rollout_chunk_loss

    def _fake_compute(*, env_prior, policy_step_fn, batch_size, n_samples, num_features, device,
                      single_eval_pos, collect_x, policy_rollout_checkpoint,
                      policy_rollout_checkpoint_reentrant, pg_saved_tensors_cpu_offload,
                      pg_saved_tensors_pin_memory, pg_tbptt_window, tbptt_loss_sink=None, rl_objective="policy_gradient"):
        del env_prior, policy_step_fn, n_samples, num_features, single_eval_pos, collect_x
        del policy_rollout_checkpoint, policy_rollout_checkpoint_reentrant
        del pg_saved_tensors_cpu_offload, pg_saved_tensors_pin_memory, pg_tbptt_window, tbptt_loss_sink, rl_objective
        anchor = next(model.parameters()).sum() * 0.0
        stats = {
            "objective": torch.zeros((), device=device),
            "reward_mean": torch.zeros((), device=device),
            "reward_std": torch.zeros((), device=device),
        }
        return anchor, None, stats

    train_mod._compute_policy_rollout_chunk_loss = _fake_compute
    try:
        loss, _, _ = train_epoch_policy_gradient(
            model=model,
            aggregate_k_gradients=1,
            using_dist=False,
            scaler=None,
            dl=dl,
            device="cpu",
            optimizer=optimizer,
            env_prior=env_prior,
            policy_rollout_chunk_size=2,
            policy_rollout_checkpoint=False,
            policy_rollout_checkpoint_reentrant=False,
            pg_grad_mutable_kv_cache=False,
            pg_saved_tensors_cpu_offload=False,
            pg_saved_tensors_pin_memory=False,
            pg_torch_compile=False,
            pg_phase_log_every_batches=1,
            pg_phase_log_file=str(log_path),
            progress_bar=False,
        )
    finally:
        train_mod._compute_policy_rollout_chunk_loss = orig_compute

    assert torch.isfinite(torch.tensor(loss))
    lines = log_path.read_text().splitlines()
    start_lines = [line for line in lines if line.startswith("[pg-phase-start]")]
    finish_lines = [line for line in lines if line.startswith("[pg-phase]")]
    assert len(start_lines) == 1
    assert len(finish_lines) == 1
    assert "status=start" in start_lines[0]
    assert "status=ok" in finish_lines[0]
