import torch
from torch.utils.checkpoint import checkpoint

from ticl.models.encoders import Linear
from ticl.models.tabpfn import TabPFN


def _build_model(recompute_attn=False):
    model = TabPFN(
        n_out=1,
        n_features=12,
        emsize=32,
        nhead=1,
        nhid_factor=2,
        nlayers=2,
        dropout=0.0,
        recompute_attn=bool(recompute_attn),
        y_encoder_layer=Linear(1, emsize=32),
        classification_task=False,
        y_encoder="linear",
        single_eval_causal=True,
    )
    model.eval()
    return model


def test_tabpfn_forward_with_kv_matches_full_forward():
    torch.manual_seed(123)
    model = _build_model()

    x = torch.randn(7, 3, 12)
    y = torch.randn(7, 3)
    single_eval_pos = 5

    out_full = model((x, y), single_eval_pos=single_eval_pos)
    out_kv = model.forward_with_kv((x, y), single_eval_pos=single_eval_pos)

    assert out_full.shape == out_kv.shape
    assert torch.allclose(out_full, out_kv, atol=1e-5, rtol=1e-4)


def test_tabpfn_kv_incremental_append_matches_full_forward():
    torch.manual_seed(321)
    model = _build_model()

    x = torch.randn(6, 2, 12)
    y = torch.randn(6, 2)

    # Build cache from first 4 training tokens.
    kv_cache = model.init_kv_cache(x[:4], y[:4])

    # Query token at index 4 (as test token with train prefix length 4).
    out_q1_kv = model.predict_query_with_kv(x[4:5], kv_cache)
    out_q1_full = model((x[:5], y[:5]), single_eval_pos=4)
    assert torch.allclose(out_q1_kv, out_q1_full, atol=1e-5, rtol=1e-4)

    # Append realized token 4 as new train token, then query token 5.
    kv_cache = model.append_train_token_to_kv(x[4:5], y[4:5], kv_cache)
    out_q2_kv = model.predict_query_with_kv(x[5:6], kv_cache)
    out_q2_full = model((x[:6], y[:6]), single_eval_pos=5)
    assert torch.allclose(out_q2_kv, out_q2_full, atol=1e-5, rtol=1e-4)


def test_tabpfn_forward_policy_step_updates_cache():
    torch.manual_seed(7)
    model = _build_model()

    x1 = torch.randn(1, 2, 12)
    y1 = torch.randn(1, 2)
    out1, kv1 = model.forward_policy_step(x1, y1, kv_cache=None)

    assert out1.shape == (1, 2, 1)
    assert isinstance(kv1, list)
    assert len(kv1) == len(model.transformer_encoder.layers)
    for layer_cache in kv1:
        assert layer_cache["k"].shape[2] == 1
        assert layer_cache["v"].shape[2] == 1

    x2 = torch.randn(1, 2, 12)
    y2 = torch.randn(1, 2)
    out2, kv2 = model.forward_policy_step(x2, y2, kv_cache=kv1)

    assert out2.shape == (1, 2, 1)
    for layer_cache in kv2:
        assert layer_cache["k"].shape[2] == 2
        assert layer_cache["v"].shape[2] == 2


def test_tabpfn_forward_policy_step_with_max_cache_len_matches_legacy_cache():
    torch.manual_seed(23)
    model_cap = _build_model()
    model_legacy = _build_model()
    model_legacy.load_state_dict(model_cap.state_dict())

    steps = 6
    x_tokens = torch.randn(steps, 2, 12)
    y_tokens = torch.randn(steps, 2)

    cache_cap = None
    cache_legacy = None
    for t in range(steps):
        out_cap, cache_cap = model_cap.forward_policy_step(
            x_tokens[t: t + 1],
            y_tokens[t: t + 1],
            kv_cache=cache_cap,
            max_cache_len=steps,
        )
        out_legacy, cache_legacy = model_legacy.forward_policy_step(
            x_tokens[t: t + 1],
            y_tokens[t: t + 1],
            kv_cache=cache_legacy,
        )
        assert torch.allclose(out_cap, out_legacy, atol=1e-5, rtol=1e-4)
        for layer_cap, layer_legacy in zip(cache_cap, cache_legacy):
            assert torch.allclose(layer_cap["k"], layer_legacy["k"], atol=1e-5, rtol=1e-4)
            assert torch.allclose(layer_cap["v"], layer_legacy["v"], atol=1e-5, rtol=1e-4)


def test_tabpfn_forward_policy_step_preallocates_kv_store_without_reallocation():
    torch.manual_seed(29)
    model = _build_model()

    steps = 5
    x_tokens = torch.randn(steps, 2, 12)
    y_tokens = torch.randn(steps, 2)

    with torch.no_grad():
        cache = None
        first_layer_k_ptr = None
        first_layer_v_ptr = None
        for t in range(steps):
            _, cache = model.forward_policy_step(
                x_tokens[t: t + 1],
                y_tokens[t: t + 1],
                kv_cache=cache,
                max_cache_len=steps,
            )
            for layer_cache in cache:
                assert "k_store" in layer_cache and "v_store" in layer_cache
                assert int(layer_cache["valid_len"]) == t + 1
                assert layer_cache["k_store"].shape[2] == steps
                assert layer_cache["v_store"].shape[2] == steps
                assert layer_cache["k"].shape[2] == t + 1
                assert layer_cache["v"].shape[2] == t + 1

            layer0 = cache[0]
            if first_layer_k_ptr is None:
                first_layer_k_ptr = int(layer0["k_store"].data_ptr())
                first_layer_v_ptr = int(layer0["v_store"].data_ptr())
            else:
                assert int(layer0["k_store"].data_ptr()) == first_layer_k_ptr
                assert int(layer0["v_store"].data_ptr()) == first_layer_v_ptr


def test_tabpfn_forward_policy_step_paged_cache_matches_legacy_no_grad():
    torch.manual_seed(31)
    model_paged = _build_model()
    model_legacy = _build_model()
    model_legacy.load_state_dict(model_paged.state_dict())

    steps = 7
    x_tokens = torch.randn(steps, 2, 12)
    y_tokens = torch.randn(steps, 2)

    with torch.no_grad():
        cache_paged = None
        cache_legacy = None
        for t in range(steps):
            out_paged, cache_paged = model_paged.forward_policy_step(
                x_tokens[t: t + 1],
                y_tokens[t: t + 1],
                kv_cache=cache_paged,
                max_cache_len=steps,
                kv_cache_mode="paged",
                kv_cache_page_size=3,
            )
            out_legacy, cache_legacy = model_legacy.forward_policy_step(
                x_tokens[t: t + 1],
                y_tokens[t: t + 1],
                kv_cache=cache_legacy,
            )
            assert torch.allclose(out_paged, out_legacy, atol=1e-5, rtol=1e-4)
            for layer_paged, layer_legacy in zip(cache_paged, cache_legacy):
                assert layer_paged["cache_mode"] == "paged"
                assert layer_paged["k_pages"] is not None
                assert layer_paged["v_pages"] is not None
                assert layer_paged["k_store"] is None
                assert layer_paged["v_store"] is None
                assert torch.allclose(layer_paged["k"], layer_legacy["k"], atol=1e-5, rtol=1e-4)
                assert torch.allclose(layer_paged["v"], layer_legacy["v"], atol=1e-5, rtol=1e-4)


def test_tabpfn_forward_policy_step_with_max_cache_len_matches_legacy_backward():
    torch.manual_seed(37)
    model_cap = _build_model()
    model_legacy = _build_model()
    model_legacy.load_state_dict(model_cap.state_dict())

    steps = 6
    x_tokens = torch.randn(steps, 2, 12)
    y_tokens = torch.randn(steps, 2)

    cache_cap = None
    cache_legacy = None
    outs_cap = []
    outs_legacy = []
    for t in range(steps):
        out_cap, cache_cap = model_cap.forward_policy_step(
            x_tokens[t: t + 1],
            y_tokens[t: t + 1],
            kv_cache=cache_cap,
            max_cache_len=steps,
            kv_cache_mode="static",
        )
        out_legacy, cache_legacy = model_legacy.forward_policy_step(
            x_tokens[t: t + 1],
            y_tokens[t: t + 1],
            kv_cache=cache_legacy,
        )
        outs_cap.append(out_cap)
        outs_legacy.append(out_legacy)

    out_cap_all = torch.cat(outs_cap, dim=0)
    out_legacy_all = torch.cat(outs_legacy, dim=0)
    assert torch.allclose(out_cap_all, out_legacy_all, atol=1e-5, rtol=1e-4)

    loss_cap = out_cap_all.pow(2).mean()
    loss_legacy = out_legacy_all.pow(2).mean()

    model_cap.zero_grad(set_to_none=True)
    model_legacy.zero_grad(set_to_none=True)
    loss_cap.backward()
    loss_legacy.backward()

    for layer_cache in cache_cap:
        assert layer_cache["cache_mode"] == "immutable"
        assert layer_cache.get("k_store", None) is None
        assert layer_cache.get("v_store", None) is None
        assert layer_cache.get("k_pages", None) is None
        assert layer_cache.get("v_pages", None) is None

    for (name_cap, p_cap), (name_legacy, p_legacy) in zip(model_cap.named_parameters(), model_legacy.named_parameters()):
        assert name_cap == name_legacy
        assert p_cap.grad is not None
        assert p_legacy.grad is not None
        assert torch.allclose(p_cap.grad, p_legacy.grad, atol=1e-5, rtol=1e-4), name_cap


def test_tabpfn_forward_policy_step_recompute_attn_matches_no_recompute_semantics():
    torch.manual_seed(41)
    model_base = _build_model(recompute_attn=False)
    model_ckpt = _build_model(recompute_attn=True)
    model_ckpt.load_state_dict(model_base.state_dict())
    model_base.train()
    model_ckpt.train()

    steps = 6
    x_tokens = torch.randn(steps, 2, 12)
    y_tokens = torch.randn(steps, 2)

    cache_base = None
    cache_ckpt = None
    outs_base = []
    outs_ckpt = []
    for t in range(steps):
        out_base, cache_base = model_base.forward_policy_step(
            x_tokens[t: t + 1],
            y_tokens[t: t + 1],
            kv_cache=cache_base,
            max_cache_len=steps,
            kv_cache_mode="immutable",
        )
        out_ckpt, cache_ckpt = model_ckpt.forward_policy_step(
            x_tokens[t: t + 1],
            y_tokens[t: t + 1],
            kv_cache=cache_ckpt,
            max_cache_len=steps,
            kv_cache_mode="immutable",
        )
        outs_base.append(out_base)
        outs_ckpt.append(out_ckpt)

    out_base_all = torch.cat(outs_base, dim=0)
    out_ckpt_all = torch.cat(outs_ckpt, dim=0)
    assert torch.allclose(out_base_all, out_ckpt_all, atol=1e-5, rtol=1e-4)

    loss_base = out_base_all.pow(2).mean()
    loss_ckpt = out_ckpt_all.pow(2).mean()

    model_base.zero_grad(set_to_none=True)
    model_ckpt.zero_grad(set_to_none=True)
    loss_base.backward()
    loss_ckpt.backward()

    for (name_base, p_base), (name_ckpt, p_ckpt) in zip(model_base.named_parameters(), model_ckpt.named_parameters()):
        assert name_base == name_ckpt
        assert p_base.grad is not None
        assert p_ckpt.grad is not None
        assert torch.allclose(p_base.grad, p_ckpt.grad, atol=1e-5, rtol=1e-4), name_base


def test_tabpfn_forward_policy_step_static_mutable_checkpoint_matches_immutable_semantics():
    torch.manual_seed(43)
    model_static = _build_model()
    model_immutable = _build_model()
    model_immutable.load_state_dict(model_static.state_dict())
    model_static.train()
    model_immutable.train()

    steps = 6
    x_tokens = torch.randn(steps, 2, 12)
    y_tokens = torch.randn(steps, 2)

    def _loss_for_model(model, kv_cache_mode, allow_grad_mutable_cache):
        def _rollout_loss(dummy):
            cache = None
            outs = []
            for t in range(steps):
                out, cache = model.forward_policy_step(
                    x_tokens[t: t + 1],
                    y_tokens[t: t + 1],
                    kv_cache=cache,
                    max_cache_len=steps,
                    kv_cache_mode=kv_cache_mode,
                    allow_grad_mutable_cache=allow_grad_mutable_cache,
                )
                outs.append(out)
            out_all = torch.cat(outs, dim=0)
            return out_all.pow(2).mean() + dummy * 0.0

        dummy = torch.ones((), requires_grad=True)
        return checkpoint(_rollout_loss, dummy, use_reentrant=False)

    loss_static = _loss_for_model(model_static, "static", True)
    loss_immutable = _loss_for_model(model_immutable, "immutable", False)

    assert torch.allclose(loss_static.detach(), loss_immutable.detach(), atol=1e-5, rtol=1e-4)

    model_static.zero_grad(set_to_none=True)
    model_immutable.zero_grad(set_to_none=True)
    loss_static.backward()
    loss_immutable.backward()

    for (name_static, p_static), (name_immutable, p_immutable) in zip(
        model_static.named_parameters(),
        model_immutable.named_parameters(),
    ):
        assert name_static == name_immutable
        assert p_static.grad is not None
        assert p_immutable.grad is not None
        assert torch.allclose(p_static.grad, p_immutable.grad, atol=1e-5, rtol=1e-4), name_static
