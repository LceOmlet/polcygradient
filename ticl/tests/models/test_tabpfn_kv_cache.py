import os

import torch
import pytest
from torch.utils.checkpoint import checkpoint

import ticl.models.layer as layer_mod
from ticl.models.encoders import Linear
from ticl.models.perfeature_tabpfn import PerFeatureTabPFN
from ticl.models.tabpfn import TabPFN
from ticl.priors.environment_prior import EnvironmentPrior


def _build_model(recompute_attn=False, nhead=1, emsize=32):
    model = TabPFN(
        n_out=1,
        n_features=12,
        emsize=emsize,
        nhead=nhead,
        nhid_factor=2,
        nlayers=2,
        dropout=0.0,
        recompute_attn=bool(recompute_attn),
        y_encoder_layer=Linear(1, emsize=emsize),
        classification_task=False,
        y_encoder="linear",
        single_eval_causal=True,
    )
    model.eval()
    return model


def _build_perfeature_model(recompute_attn=False, nhead=1, emsize=32):
    model = PerFeatureTabPFN(
        n_out=1,
        n_features=12,
        emsize=emsize,
        nhead=nhead,
        nhid_factor=2,
        nlayers=2,
        dropout=0.0,
        recompute_attn=bool(recompute_attn),
        y_encoder_layer=Linear(1, emsize=emsize),
        classification_task=False,
        y_encoder="linear",
        single_eval_causal=True,
        x_encoder_type="single",
        features_per_group=3,
    )
    model.eval()
    return model


def _materialize_kv_from_layer_cache(layer_cache):
    if (
        isinstance(layer_cache, dict)
        and str(layer_cache.get("cache_layout", "")) == "per_feature_item_microchunked"
        and isinstance(layer_cache.get("item_chunk_caches", None), list)
    ):
        k_chunks = []
        v_chunks = []
        for cache_chunk in layer_cache["item_chunk_caches"]:
            chunk_k, chunk_v = _materialize_kv_from_layer_cache(cache_chunk)
            k_chunks.append(chunk_k)
            v_chunks.append(chunk_v)
        if not k_chunks or not v_chunks:
            raise AssertionError("Chunked per-feature cache is empty.")
        return torch.cat(k_chunks, dim=0), torch.cat(v_chunks, dim=0)

    k = layer_cache.get("k", None)
    v = layer_cache.get("v", None)
    if k is not None and v is not None:
        return k, v

    k_pages = layer_cache.get("k_pages", None)
    v_pages = layer_cache.get("v_pages", None)
    if k_pages is None or v_pages is None:
        raise AssertionError("Expected dense k/v tensors or paged k_pages/v_pages in layer cache.")

    valid_len = int(layer_cache.get("valid_len", 0))
    k_prefix = layer_cache.get("k_prefix", None)
    v_prefix = layer_cache.get("v_prefix", None)
    prefix_len = 0
    if k_prefix is not None and v_prefix is not None:
        prefix_len = int(k_prefix.shape[2])
        if prefix_len > valid_len:
            prefix_len = valid_len
    remaining = int(max(0, valid_len))
    k_chunks = []
    v_chunks = []
    if prefix_len > 0:
        prefix_k = k_prefix[:, :, :prefix_len, :]
        prefix_v = v_prefix[:, :, :prefix_len, :]
        target_heads = None
        if isinstance(k_pages, list) and len(k_pages) > 0:
            target_heads = int(k_pages[0].shape[1])
        elif k is not None:
            target_heads = int(k.shape[1])
        if target_heads is not None and int(prefix_k.shape[1]) == 1 and int(target_heads) > 1:
            prefix_k = prefix_k.expand(prefix_k.shape[0], target_heads, prefix_k.shape[2], prefix_k.shape[3])
            prefix_v = prefix_v.expand(prefix_v.shape[0], target_heads, prefix_v.shape[2], prefix_v.shape[3])
        k_chunks.append(prefix_k)
        v_chunks.append(prefix_v)
        remaining -= prefix_len
    for k_page, v_page in zip(k_pages, v_pages):
        if remaining <= 0:
            break
        take = int(min(int(k_page.shape[2]), remaining))
        if take <= 0:
            continue
        k_chunks.append(k_page[:, :, :take, :])
        v_chunks.append(v_page[:, :, :take, :])
        remaining -= take
    if remaining != 0:
        raise AssertionError("Paged cache pages do not cover valid_len.")
    if not k_chunks or not v_chunks:
        raise AssertionError("Paged cache pages are empty.")
    return torch.cat(k_chunks, dim=2), torch.cat(v_chunks, dim=2)


def _perfeature_layer_forward_step_reference(
    layer,
    src_step,
    kv_cache=None,
    append_to_cache=True,
    max_cache_len=None,
    kv_cache_mode="auto",
    kv_cache_page_size=None,
    allow_grad_mutable_cache=False,
    allow_grad_inplace_paged_cache=False,
):
    keep_seq_dim = bool(src_step.ndim == 4)
    state = src_step.squeeze(0) if keep_seq_dim else src_step
    feature_state = layer._feature_forward(state)
    batch_size, num_groups, dim = feature_state.shape
    item_in = feature_state.reshape(1, batch_size * num_groups, dim)
    item_out, item_cache = layer.item_block.forward_step(
        item_in,
        kv_cache=kv_cache,
        append_to_cache=append_to_cache,
        max_cache_len=max_cache_len,
        kv_cache_mode=kv_cache_mode,
        kv_cache_page_size=kv_cache_page_size,
        allow_grad_mutable_cache=allow_grad_mutable_cache,
        allow_grad_inplace_paged_cache=allow_grad_inplace_paged_cache,
    )
    out = item_out.reshape(batch_size, num_groups, dim)
    if keep_seq_dim:
        out = out.unsqueeze(0)
    return out, item_cache


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


def test_perfeature_tabpfn_forward_with_kv_matches_full_forward():
    torch.manual_seed(123)
    model = _build_perfeature_model()

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


def test_perfeature_tabpfn_kv_incremental_append_matches_full_forward():
    torch.manual_seed(321)
    model = _build_perfeature_model()

    x = torch.randn(6, 2, 12)
    y = torch.randn(6, 2)

    kv_cache = model.init_kv_cache(x[:4], y[:4])

    out_q1_kv = model.predict_query_with_kv(x[4:5], kv_cache)
    out_q1_full = model((x[:5], y[:5]), single_eval_pos=4)
    assert torch.allclose(out_q1_kv, out_q1_full, atol=1e-5, rtol=1e-4)

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


def test_perfeature_tabpfn_forward_policy_step_updates_cache():
    torch.manual_seed(7)
    model = _build_perfeature_model()

    x1 = torch.randn(1, 2, 12)
    y1 = torch.randn(1, 2)
    out1, kv1 = model.forward_policy_step(x1, y1, kv_cache=None)

    assert out1.shape == (1, 2, 1)
    assert isinstance(kv1, list)
    assert len(kv1) == len(model.transformer_encoder.layers)
    for layer_cache in kv1:
        assert isinstance(layer_cache, dict)
        k_live, v_live = _materialize_kv_from_layer_cache(layer_cache)
        assert k_live.shape[2] == 1
        assert v_live.shape[2] == 1

    x2 = torch.randn(1, 2, 12)
    y2 = torch.randn(1, 2)
    out2, kv2 = model.forward_policy_step(x2, y2, kv_cache=kv1)

    assert out2.shape == (1, 2, 1)
    for layer_cache in kv2:
        k_live, v_live = _materialize_kv_from_layer_cache(layer_cache)
        assert k_live.shape[2] == 2
        assert v_live.shape[2] == 2


def test_perfeature_policy_step_profile_reports_feature_and_item(monkeypatch):
    monkeypatch.setenv("TICL_POLICY_STEP_PROFILE", "1")
    torch.manual_seed(1234)
    model = _build_perfeature_model(recompute_attn=False, nhead=1, emsize=16)
    model.train()

    x = torch.randn(1, 2, 12)
    y = torch.randn(1, 2)
    _, kv_cache = model.forward_policy_step(x, y, kv_cache=None)
    profile_first = model.consume_policy_step_profile()

    assert isinstance(profile_first, dict)
    assert int(profile_first.get("calls", 0) or 0) == 1
    assert float(profile_first.get("perfeature_feature_wall_s", 0.0) or 0.0) > 0.0
    assert float(profile_first.get("perfeature_item_wall_s", 0.0) or 0.0) > 0.0
    assert int(profile_first.get("perfeature_item_saved_nonparam_unique_storage_bytes_last", 0) or 0) > 0
    assert int(profile_first.get("transformer_layer_attnff_saved_nonparam_unique_storage_bytes_last", 0) or 0) >= 0
    assert int(profile_first.get("transformer_layer_finalize_saved_nonparam_unique_storage_bytes_last", 0) or 0) >= 0
    assert "transformer_layer_flash_prefix_merge_wall_s" in profile_first
    assert "transformer_layer_flash_prefix_prefix_saved_q_nonparam_unique_storage_bytes_last" in profile_first
    assert "transformer_layer_flash_prefix_prefix_saved_other_nonparam_unique_storage_bytes_last" in profile_first

    x_next = torch.randn(1, 2, 12)
    y_next = torch.randn(1, 2)
    _, _ = model.forward_policy_step(x_next, y_next, kv_cache=kv_cache)
    profile_second = model.consume_policy_step_profile()

    assert isinstance(profile_second, dict)
    assert int(profile_second.get("calls", 0) or 0) == 1
    assert int(profile_second.get("perfeature_item_saved_nonparam_unique_storage_bytes_last", 0) or 0) > 0


def test_perfeature_policy_step_profile_reports_item_microchunk_stats(monkeypatch):
    monkeypatch.setenv("TICL_POLICY_STEP_PROFILE", "1")
    torch.manual_seed(2234)
    model = _build_perfeature_model(recompute_attn=False, nhead=1, emsize=16)
    model.train()

    for layer in model.transformer_encoder.layers:
        layer.item_step_max_columns = 8
        layer.item_step_max_columns_warmup = 8
        layer.item_step_warmup_valid_len = 0

    x = torch.randn(1, 8, 12)
    y = torch.randn(1, 8)
    _, _ = model.forward_policy_step(
        x,
        y,
        kv_cache=None,
        max_cache_len=8,
        kv_cache_mode="paged",
        kv_cache_page_size=3,
        allow_grad_mutable_cache=True,
    )
    profile = model.consume_policy_step_profile()

    assert isinstance(profile, dict)
    assert int(profile.get("perfeature_item_chunk_calls", 0) or 0) > 0
    assert float(profile.get("perfeature_item_chunk_wall_s", 0.0) or 0.0) > 0.0
    assert int(profile.get("perfeature_item_chunk_batch_max", 0) or 0) > 0
    assert int(profile.get("perfeature_item_chunk_saved_nonparam_unique_storage_bytes_max", 0) or 0) >= 0
    assert int(profile.get("perfeature_item_chunk_cache_unique_storage_bytes_max", 0) or 0) > 0


def test_perfeature_tabpfn_forward_policy_step_paged_cache_matches_legacy_no_grad():
    torch.manual_seed(31)
    model_paged = _build_perfeature_model()
    model_legacy = _build_perfeature_model()
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
                paged_k, paged_v = _materialize_kv_from_layer_cache(layer_paged)
                legacy_k, legacy_v = _materialize_kv_from_layer_cache(layer_legacy)
                assert torch.allclose(paged_k, legacy_k, atol=1e-5, rtol=1e-4)
                assert torch.allclose(paged_v, legacy_v, atol=1e-5, rtol=1e-4)


def test_perfeature_layer_forward_step_internal_microbatch_preserves_paged_cache_semantics():
    torch.manual_seed(41)
    model = _build_perfeature_model(recompute_attn=True)
    layer = model.transformer_encoder.layers[0]
    layer.item_step_max_columns = 8

    x1 = torch.randn(1, 8, 12)
    y1 = torch.randn(1, 8)
    x2 = torch.randn(1, 8, 12)
    y2 = torch.randn(1, 8)

    token1 = model._apply_input_ln(model.encoder(x1) + model.y_encoder(y1.unsqueeze(-1)).unsqueeze(-2))
    token2 = model._apply_input_ln(model.encoder(x2) + model.y_encoder(y2.unsqueeze(-1)).unsqueeze(-2))

    out_live_1, cache_live_1 = layer.forward_step(
        token1,
        kv_cache=None,
        max_cache_len=8,
        kv_cache_mode="paged",
        kv_cache_page_size=3,
        allow_grad_mutable_cache=True,
    )
    out_ref_1, cache_ref_1 = _perfeature_layer_forward_step_reference(
        layer,
        token1,
        kv_cache=None,
        max_cache_len=8,
        kv_cache_mode="paged",
        kv_cache_page_size=3,
        allow_grad_mutable_cache=True,
    )

    assert torch.allclose(out_live_1, out_ref_1, atol=1e-5, rtol=1e-4)
    live_k_1, live_v_1 = _materialize_kv_from_layer_cache(cache_live_1)
    ref_k_1, ref_v_1 = _materialize_kv_from_layer_cache(cache_ref_1)
    assert torch.allclose(live_k_1, ref_k_1, atol=1e-5, rtol=1e-4)
    assert torch.allclose(live_v_1, ref_v_1, atol=1e-5, rtol=1e-4)

    out_live_2, cache_live_2 = layer.forward_step(
        token2,
        kv_cache=cache_live_1,
        max_cache_len=8,
        kv_cache_mode="paged",
        kv_cache_page_size=3,
        allow_grad_mutable_cache=True,
    )
    out_ref_2, cache_ref_2 = _perfeature_layer_forward_step_reference(
        layer,
        token2,
        kv_cache=cache_ref_1,
        max_cache_len=8,
        kv_cache_mode="paged",
        kv_cache_page_size=3,
        allow_grad_mutable_cache=True,
    )

    assert torch.allclose(out_live_2, out_ref_2, atol=1e-5, rtol=1e-4)
    live_k_2, live_v_2 = _materialize_kv_from_layer_cache(cache_live_2)
    ref_k_2, ref_v_2 = _materialize_kv_from_layer_cache(cache_ref_2)
    assert torch.allclose(live_k_2, ref_k_2, atol=1e-5, rtol=1e-4)
    assert torch.allclose(live_v_2, ref_v_2, atol=1e-5, rtol=1e-4)


def test_perfeature_item_microbatch_warmup_threshold_relaxes_only_early_steps():
    torch.manual_seed(42)
    model = _build_perfeature_model(recompute_attn=False)
    layer = model.transformer_encoder.layers[0]
    layer.item_step_max_columns = 256
    layer.item_step_max_columns_warmup = 512
    layer.item_step_warmup_valid_len = 32

    # 12 features grouped by 3 -> 4 groups; 64 batch would normally require
    # microchunking at 256 columns, but the warmup path should keep it whole.
    assert layer._item_step_batch_microchunk_size(64, 4, valid_len=0) == 64
    assert layer._item_step_batch_microchunk_size(64, 4, valid_len=31) == 64
    assert layer._item_step_batch_microchunk_size(64, 4, valid_len=32) == 64

    # 20 groups makes the base threshold small enough that the warmup path and
    # steady-state path diverge in a way that exercises the heuristic.
    assert layer._item_step_batch_microchunk_size(64, 20, valid_len=0) == 25
    assert layer._item_step_batch_microchunk_size(64, 20, valid_len=31) == 25
    assert layer._item_step_batch_microchunk_size(64, 20, valid_len=32) == 12


def test_perfeature_item_block_cow_grow_cap_preserves_outputs_and_cache_contents():
    torch.manual_seed(43)
    model_cap = _build_perfeature_model(recompute_attn=False)
    model_ref = _build_perfeature_model(recompute_attn=False)
    model_ref.load_state_dict(model_cap.state_dict())

    for layer in model_cap.transformer_encoder.layers:
        layer.item_block.paged_cow_grow_max_tokens = 2

    steps = 5
    x_tokens = torch.randn(steps, 2, 12)
    y_tokens = torch.randn(steps, 2)

    cache_cap = None
    cache_ref = None
    for t in range(steps):
        out_cap, cache_cap = model_cap.forward_policy_step(
            x_tokens[t: t + 1],
            y_tokens[t: t + 1],
            kv_cache=cache_cap,
            max_cache_len=steps,
            kv_cache_mode="paged",
            kv_cache_page_size=4,
            allow_grad_mutable_cache=True,
        )
        out_ref, cache_ref = model_ref.forward_policy_step(
            x_tokens[t: t + 1],
            y_tokens[t: t + 1],
            kv_cache=cache_ref,
            max_cache_len=steps,
            kv_cache_mode="paged",
            kv_cache_page_size=4,
            allow_grad_mutable_cache=True,
        )
        assert torch.allclose(out_cap, out_ref, atol=1e-5, rtol=1e-4)
        for layer_cap, layer_ref in zip(cache_cap, cache_ref):
            cap_k, cap_v = _materialize_kv_from_layer_cache(layer_cap)
            ref_k, ref_v = _materialize_kv_from_layer_cache(layer_ref)
            assert torch.allclose(cap_k, ref_k, atol=1e-5, rtol=1e-4)
            assert torch.allclose(cap_v, ref_v, atol=1e-5, rtol=1e-4)
            if (
                isinstance(layer_cap, dict)
                and str(layer_cap.get("cache_layout", "")) == "per_feature_item_microchunked"
            ):
                for chunk_cache in layer_cap.get("item_chunk_caches", []):
                    if isinstance(chunk_cache, dict) and isinstance(chunk_cache.get("k_pages", None), list):
                        assert all(int(page.shape[2]) <= 2 for page in chunk_cache["k_pages"])


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
                paged_k, paged_v = _materialize_kv_from_layer_cache(layer_paged)
                legacy_k, legacy_v = _materialize_kv_from_layer_cache(layer_legacy)
                assert torch.allclose(paged_k, legacy_k, atol=1e-5, rtol=1e-4)
                assert torch.allclose(paged_v, legacy_v, atol=1e-5, rtol=1e-4)


def test_tbptt_detach_prefix_compaction_preserves_paged_cache_semantics():
    torch.manual_seed(41)
    model_base = _build_model()
    model_compact = _build_model()
    model_compact.load_state_dict(model_base.state_dict())
    model_base.train()
    model_compact.train()

    steps = 7
    x_tokens = torch.randn(steps, 2, 12)
    y_tokens = torch.randn(steps, 2)

    cache_base = None
    cache_compact = None
    detach_after = 6
    with torch.enable_grad():
        for t in range(steps):
            out_base, cache_base = model_base.forward_policy_step(
                x_tokens[t: t + 1],
                y_tokens[t: t + 1],
                kv_cache=cache_base,
                max_cache_len=steps,
                kv_cache_mode="paged",
                kv_cache_page_size=3,
                allow_grad_mutable_cache=True,
            )
            out_compact, cache_compact = model_compact.forward_policy_step(
                x_tokens[t: t + 1],
                y_tokens[t: t + 1],
                kv_cache=cache_compact,
                max_cache_len=steps,
                kv_cache_mode="paged",
                kv_cache_page_size=3,
                allow_grad_mutable_cache=True,
            )
            assert torch.allclose(out_base, out_compact, atol=1e-5, rtol=1e-4)
            if t == (detach_after - 1):
                orig_page_counts = [len(layer_cache["k_pages"]) for layer_cache in cache_compact]
                cache_base = EnvironmentPrior._detach_policy_cache(cache_base, clone_tensors=False)
                cache_compact = EnvironmentPrior._detach_policy_cache(cache_compact, clone_tensors=False)
                for orig_pages, layer_cache in zip(orig_page_counts, cache_compact):
                    assert layer_cache["cache_mode"] == "paged"
                    assert isinstance(layer_cache["k_pages"], list)
                    assert len(layer_cache["k_pages"]) <= int(orig_pages)
                    assert len(layer_cache["v_pages"]) <= int(orig_pages)
                    if int(orig_pages) > 1:
                        assert int(layer_cache.get("prefix_base_len", 0)) > 0
                        assert len(layer_cache["k_pages"]) == 1
                        assert len(layer_cache["v_pages"]) == 1

        for layer_base, layer_compact in zip(cache_base, cache_compact):
            base_k, base_v = _materialize_kv_from_layer_cache(layer_base)
            compact_k, compact_v = _materialize_kv_from_layer_cache(layer_compact)
            assert torch.allclose(base_k, compact_k, atol=1e-5, rtol=1e-4)
            assert torch.allclose(base_v, compact_v, atol=1e-5, rtol=1e-4)


def test_tbptt_detach_prefix_compaction_matches_live_paged_rollout():
    torch.manual_seed(123)
    model_live = _build_model()
    model_detached = _build_model()
    model_detached.load_state_dict(model_live.state_dict())
    model_live.train()
    model_detached.train()

    steps = 12
    detach_after = 9
    x_tokens = torch.randn(steps, 2, 12)
    y_tokens = torch.randn(steps, 2)

    cache_live = None
    cache_detached = None
    with torch.enable_grad():
        for t in range(steps):
            out_live, cache_live = model_live.forward_policy_step(
                x_tokens[t: t + 1],
                y_tokens[t: t + 1],
                kv_cache=cache_live,
                max_cache_len=steps,
                kv_cache_mode="paged",
                kv_cache_page_size=1,
                allow_grad_mutable_cache=True,
            )
            out_detached, cache_detached = model_detached.forward_policy_step(
                x_tokens[t: t + 1],
                y_tokens[t: t + 1],
                kv_cache=cache_detached,
                max_cache_len=steps,
                kv_cache_mode="paged",
                kv_cache_page_size=1,
                allow_grad_mutable_cache=True,
            )
            assert torch.allclose(out_live, out_detached, atol=1e-5, rtol=1e-4)
            if t == (detach_after - 1):
                cache_detached = EnvironmentPrior._detach_policy_cache(cache_detached, clone_tensors=False)
                for layer_cache in cache_detached:
                    assert int(layer_cache.get("prefix_base_len", 0)) > 0
                    assert len(layer_cache["k_pages"]) == 1
                    assert len(layer_cache["v_pages"]) == 1

        for layer_live, layer_detached in zip(cache_live, cache_detached):
            live_k, live_v = _materialize_kv_from_layer_cache(layer_live)
            detached_k, detached_v = _materialize_kv_from_layer_cache(layer_detached)
            assert torch.allclose(live_k, detached_k, atol=1e-5, rtol=1e-4)
            assert torch.allclose(live_v, detached_v, atol=1e-5, rtol=1e-4)


def test_tbptt_detach_prefix_compaction_matches_live_rollout_with_tail_clone_append():
    old_enabled = layer_mod._TAIL_FREEZE_CLONE_APPEND_ENABLED
    old_guard = layer_mod._TAIL_FREEZE_CLONE_APPEND_GUARD_ENABLED
    layer_mod._TAIL_FREEZE_CLONE_APPEND_ENABLED = True
    layer_mod._TAIL_FREEZE_CLONE_APPEND_GUARD_ENABLED = False
    try:
        torch.manual_seed(124)
        model_live = _build_model()
        model_detached = _build_model()
        model_detached.load_state_dict(model_live.state_dict())
        model_live.train()
        model_detached.train()

        steps = 12
        detach_after = 9
        x_tokens = torch.randn(steps, 2, 12)
        y_tokens = torch.randn(steps, 2)

        cache_live = None
        cache_detached = None
        with torch.enable_grad():
            for t in range(steps):
                out_live, cache_live = model_live.forward_policy_step(
                    x_tokens[t: t + 1],
                    y_tokens[t: t + 1],
                    kv_cache=cache_live,
                    max_cache_len=steps,
                    kv_cache_mode="paged",
                    kv_cache_page_size=1,
                    allow_grad_mutable_cache=True,
                )
                out_detached, cache_detached = model_detached.forward_policy_step(
                    x_tokens[t: t + 1],
                    y_tokens[t: t + 1],
                    kv_cache=cache_detached,
                    max_cache_len=steps,
                    kv_cache_mode="paged",
                    kv_cache_page_size=1,
                    allow_grad_mutable_cache=True,
                )
                assert torch.allclose(out_live, out_detached, atol=1e-5, rtol=1e-4)
                if t == (detach_after - 1):
                    cache_detached = EnvironmentPrior._detach_policy_cache(cache_detached, clone_tensors=False)
                    for layer_cache in cache_detached:
                        assert int(layer_cache.get("prefix_base_len", 0)) > 0
                        assert len(layer_cache["k_pages"]) == 1
                        assert len(layer_cache["v_pages"]) == 1

            for layer_live, layer_detached in zip(cache_live, cache_detached):
                live_k, live_v = _materialize_kv_from_layer_cache(layer_live)
                detached_k, detached_v = _materialize_kv_from_layer_cache(layer_detached)
                assert torch.allclose(live_k, detached_k, atol=1e-5, rtol=1e-4)
                assert torch.allclose(live_v, detached_v, atol=1e-5, rtol=1e-4)
    finally:
        layer_mod._TAIL_FREEZE_CLONE_APPEND_ENABLED = old_enabled
        layer_mod._TAIL_FREEZE_CLONE_APPEND_GUARD_ENABLED = old_guard


def _run_inplace_prefix_compaction_rollout(build_model_fn):
    torch.manual_seed(125)
    model_live = build_model_fn()
    model_detached = build_model_fn()
    model_detached.load_state_dict(model_live.state_dict())
    model_live.train()
    model_detached.train()

    steps = 20
    detach_after = 9
    x_tokens = torch.randn(steps, 2, 12)
    y_tokens = torch.randn(steps, 2)

    cache_live = None
    cache_detached = None
    with torch.enable_grad():
        for t in range(steps):
            out_live, cache_live = model_live.forward_policy_step(
                x_tokens[t: t + 1],
                y_tokens[t: t + 1],
                kv_cache=cache_live,
                max_cache_len=steps,
                kv_cache_mode="paged",
                kv_cache_page_size=1,
                allow_grad_mutable_cache=True,
                allow_grad_inplace_paged_cache=True,
            )
            out_detached, cache_detached = model_detached.forward_policy_step(
                x_tokens[t: t + 1],
                y_tokens[t: t + 1],
                kv_cache=cache_detached,
                max_cache_len=steps,
                kv_cache_mode="paged",
                kv_cache_page_size=1,
                allow_grad_mutable_cache=True,
                allow_grad_inplace_paged_cache=True,
            )
            assert torch.allclose(out_live, out_detached, atol=1e-5, rtol=1e-4)
            if t == (detach_after - 1):
                cache_detached = EnvironmentPrior._detach_policy_cache(cache_detached, clone_tensors=False)
                for layer_cache in cache_detached:
                    assert int(layer_cache.get("prefix_base_len", 0)) > 0
                    assert len(layer_cache["k_pages"]) == 1
                    assert len(layer_cache["v_pages"]) == 1

        for layer_live, layer_detached in zip(cache_live, cache_detached):
            live_k, live_v = _materialize_kv_from_layer_cache(layer_live)
            detached_k, detached_v = _materialize_kv_from_layer_cache(layer_detached)
            assert torch.allclose(live_k, detached_k, atol=1e-5, rtol=1e-4)
            assert torch.allclose(live_v, detached_v, atol=1e-5, rtol=1e-4)


def test_tbptt_detach_prefix_compaction_matches_live_rollout_with_inplace_paged_cache():
    _run_inplace_prefix_compaction_rollout(_build_model)


def test_perfeature_tbptt_detach_prefix_compaction_matches_live_rollout_with_inplace_paged_cache():
    _run_inplace_prefix_compaction_rollout(_build_perfeature_model)


@pytest.mark.parametrize(
    "build_model_fn",
    [_build_model, _build_perfeature_model],
)
def test_detached_prefix_streaming_paths_avoid_dense_prefix_tail_cat(build_model_fn, monkeypatch):
    torch.manual_seed(126)
    monkeypatch.setenv("TICL_POLICY_PREFIX_STREAMING_TRAIN", "1")
    model = build_model_fn()
    model.train()

    steps = 16
    detach_after = 12
    x_tokens = torch.randn(steps + 1, 2, 12)
    y_tokens = torch.randn(steps + 1, 2)

    cache = None
    with torch.enable_grad():
        for t in range(detach_after):
            _, cache = model.forward_policy_step(
                x_tokens[t: t + 1],
                y_tokens[t: t + 1],
                kv_cache=cache,
                max_cache_len=steps + 1,
                kv_cache_mode="paged",
                kv_cache_page_size=1,
                allow_grad_mutable_cache=True,
            )
        cache = EnvironmentPrior._detach_policy_cache(cache, clone_tensors=False)
        for layer_cache in cache:
            assert int(layer_cache.get("prefix_base_len", 0)) > 0
            assert torch.is_tensor(layer_cache.get("k_prefix", None))
            assert torch.is_tensor(layer_cache.get("v_prefix", None))

        def _forbidden(*args, **kwargs):
            raise AssertionError("prefix+tail dense cat path should not be used here")

        monkeypatch.setattr(
            layer_mod.TransformerEncoderLayer,
            "_combine_prefix_with_paged_tail",
            staticmethod(_forbidden),
        )

        out_step, cache = model.forward_policy_step(
            x_tokens[detach_after: detach_after + 1],
            y_tokens[detach_after: detach_after + 1],
            kv_cache=cache,
            max_cache_len=steps + 1,
            kv_cache_mode="paged",
            kv_cache_page_size=1,
            allow_grad_mutable_cache=True,
        )
        out_query = model.predict_query_with_kv(
            x_tokens[detach_after + 1: detach_after + 2],
            cache,
        )

    assert torch.isfinite(out_step).all()
    assert torch.isfinite(out_query).all()


def test_tbptt_detach_immutable_prefix_head_sharing_mean_preserves_rollout_and_query():
    torch.manual_seed(127)
    model_base = _build_model(nhead=4, emsize=32)
    model_shared = _build_model(nhead=4, emsize=32)
    model_shared.load_state_dict(model_base.state_dict())
    model_base.train()
    model_shared.train()

    steps = 16
    detach_after = 12
    x_tokens = torch.randn(steps + 1, 2, 12)
    y_tokens = torch.randn(steps + 1, 2)

    cache_base = None
    cache_shared = None
    with torch.enable_grad():
        for t in range(detach_after):
            _, cache_base = model_base.forward_policy_step(
                x_tokens[t: t + 1],
                y_tokens[t: t + 1],
                kv_cache=cache_base,
                max_cache_len=steps + 1,
                kv_cache_mode="paged",
                kv_cache_page_size=1,
                allow_grad_mutable_cache=True,
            )
            _, cache_shared = model_shared.forward_policy_step(
                x_tokens[t: t + 1],
                y_tokens[t: t + 1],
                kv_cache=cache_shared,
                max_cache_len=steps + 1,
                kv_cache_mode="paged",
                kv_cache_page_size=1,
                allow_grad_mutable_cache=True,
            )

        cache_base = EnvironmentPrior._detach_policy_cache(cache_base, clone_tensors=False)
        old_mode = os.environ.get("TICL_POLICY_IMMUTABLE_PREFIX_HEAD_SHARING", None)
        try:
            os.environ["TICL_POLICY_IMMUTABLE_PREFIX_HEAD_SHARING"] = "mean"
            cache_shared = EnvironmentPrior._detach_policy_cache(cache_shared, clone_tensors=False)
        finally:
            if old_mode is None:
                os.environ.pop("TICL_POLICY_IMMUTABLE_PREFIX_HEAD_SHARING", None)
            else:
                os.environ["TICL_POLICY_IMMUTABLE_PREFIX_HEAD_SHARING"] = old_mode

        for layer_cache in cache_shared:
            assert layer_cache.get("prefix_head_sharing", "off") == "mean"
            assert torch.is_tensor(layer_cache.get("k_prefix", None))
            assert torch.is_tensor(layer_cache.get("v_prefix", None))
            assert int(layer_cache["k_prefix"].shape[1]) == 1
            assert int(layer_cache["v_prefix"].shape[1]) == 1

        out_base, cache_base = model_base.forward_policy_step(
            x_tokens[detach_after: detach_after + 1],
            y_tokens[detach_after: detach_after + 1],
            kv_cache=cache_base,
            max_cache_len=steps + 1,
            kv_cache_mode="paged",
            kv_cache_page_size=1,
            allow_grad_mutable_cache=True,
        )
        out_shared, cache_shared = model_shared.forward_policy_step(
            x_tokens[detach_after: detach_after + 1],
            y_tokens[detach_after: detach_after + 1],
            kv_cache=cache_shared,
            max_cache_len=steps + 1,
            kv_cache_mode="paged",
            kv_cache_page_size=1,
            allow_grad_mutable_cache=True,
        )
        query_base = model_base.predict_query_with_kv(
            x_tokens[detach_after + 1: detach_after + 2],
            cache_base,
        )
        query_shared = model_shared.predict_query_with_kv(
            x_tokens[detach_after + 1: detach_after + 2],
            cache_shared,
        )

    assert torch.isfinite(out_base).all()
    assert torch.isfinite(out_shared).all()
    assert torch.isfinite(query_base).all()
    assert torch.isfinite(query_shared).all()
    for layer_cache in cache_shared:
        shared_k, shared_v = _materialize_kv_from_layer_cache(layer_cache)
        assert torch.isfinite(shared_k).all()
        assert torch.isfinite(shared_v).all()


def test_tbptt_detach_head_sharing_mode_persists_before_prefix_materializes():
    torch.manual_seed(911)
    model = _build_model(nhead=4, emsize=32)
    model.train()

    steps = 12
    detach_after = 3
    x_tokens = torch.randn(steps, 2, 12)
    y_tokens = torch.randn(steps, 2)

    cache = None
    with torch.enable_grad():
        for t in range(detach_after):
            _, cache = model.forward_policy_step(
                x_tokens[t: t + 1],
                y_tokens[t: t + 1],
                kv_cache=cache,
                max_cache_len=steps,
                kv_cache_mode="paged",
                kv_cache_page_size=4,
                allow_grad_mutable_cache=True,
            )

        for layer_cache in cache:
            assert layer_cache.get("k_prefix", None) is None
            assert layer_cache.get("v_prefix", None) is None

        old_mode = os.environ.get("TICL_POLICY_IMMUTABLE_PREFIX_HEAD_SHARING", None)
        try:
            os.environ["TICL_POLICY_IMMUTABLE_PREFIX_HEAD_SHARING"] = "mean"
            cache = EnvironmentPrior._detach_policy_cache(cache, clone_tensors=False)
        finally:
            if old_mode is None:
                os.environ.pop("TICL_POLICY_IMMUTABLE_PREFIX_HEAD_SHARING", None)
            else:
                os.environ["TICL_POLICY_IMMUTABLE_PREFIX_HEAD_SHARING"] = old_mode

        for layer_cache in cache:
            assert layer_cache.get("prefix_head_sharing", "off") == "mean"
            assert layer_cache.get("k_prefix", None) is None
            assert layer_cache.get("v_prefix", None) is None

        prefix_materialized = False
        for t in range(detach_after, steps):
            _, cache = model.forward_policy_step(
                x_tokens[t: t + 1],
                y_tokens[t: t + 1],
                kv_cache=cache,
                max_cache_len=steps,
                kv_cache_mode="paged",
                kv_cache_page_size=4,
                allow_grad_mutable_cache=True,
            )
            layer_prefixes = [
                layer_cache
                for layer_cache in cache
                if torch.is_tensor(layer_cache.get("k_prefix", None))
                and torch.is_tensor(layer_cache.get("v_prefix", None))
            ]
            if not layer_prefixes:
                continue
            prefix_materialized = True
            for layer_cache in layer_prefixes:
                assert layer_cache.get("prefix_head_sharing", "off") == "mean"
                assert int(layer_cache["k_prefix"].shape[1]) == 1
                assert int(layer_cache["v_prefix"].shape[1]) == 1
            break

    assert prefix_materialized


def test_forward_policy_step_finalize_compile_preserves_semantics(monkeypatch):
    if not callable(getattr(torch, "compile", None)):
        return

    torch.manual_seed(43)
    model_base = _build_model()
    monkeypatch.setenv("TICL_POLICY_FINALIZE_TORCH_COMPILE", "1")
    model_compiled = _build_model()
    model_compiled.load_state_dict(model_base.state_dict())
    model_base.train()
    model_compiled.train()

    steps = 4
    x_tokens = torch.randn(steps, 2, 12)
    y_tokens = torch.randn(steps, 2)

    cache_base = None
    cache_compiled = None
    outs_base = []
    outs_compiled = []
    with torch.enable_grad():
        for t in range(steps):
            out_base, cache_base = model_base.forward_policy_step(
                x_tokens[t: t + 1],
                y_tokens[t: t + 1],
                kv_cache=cache_base,
            )
            out_compiled, cache_compiled = model_compiled.forward_policy_step(
                x_tokens[t: t + 1],
                y_tokens[t: t + 1],
                kv_cache=cache_compiled,
            )
            outs_base.append(out_base)
            outs_compiled.append(out_compiled)

    out_base_all = torch.cat(outs_base, dim=0)
    out_compiled_all = torch.cat(outs_compiled, dim=0)
    assert torch.allclose(out_base_all, out_compiled_all, atol=1e-5, rtol=1e-4)

    model_base.zero_grad(set_to_none=True)
    model_compiled.zero_grad(set_to_none=True)
    loss_base = out_base_all.square().mean()
    loss_compiled = out_compiled_all.square().mean()
    loss_base.backward()
    loss_compiled.backward()

    grads_base = [p.grad.detach().clone() for p in model_base.parameters() if p.grad is not None]
    grads_compiled = [p.grad.detach().clone() for p in model_compiled.parameters() if p.grad is not None]
    assert len(grads_base) == len(grads_compiled)
    for grad_base, grad_compiled in zip(grads_base, grads_compiled):
        assert torch.allclose(grad_base, grad_compiled, atol=1e-5, rtol=1e-4)


def test_forward_policy_step_finalize_default_eager_fastpath_preserves_semantics(monkeypatch):
    torch.manual_seed(44)
    monkeypatch.setenv("TICL_POLICY_FINALIZE_2D_ZERO_DROPOUT_POSTNORM_GELU_FASTPATH", "0")
    model_base = _build_model()
    monkeypatch.setenv("TICL_POLICY_FINALIZE_2D_ZERO_DROPOUT_POSTNORM_GELU_FASTPATH", "1")
    model_fast = _build_model()
    model_fast.load_state_dict(model_base.state_dict())
    model_base.train()
    model_fast.train()

    assert not model_base.transformer_encoder.layers[0]._finalize_2d_zero_dropout_postnorm_gelu_active()
    assert model_fast.transformer_encoder.layers[0]._finalize_2d_zero_dropout_postnorm_gelu_active()

    steps = 4
    x_tokens = torch.randn(steps, 2, 12)
    y_tokens = torch.randn(steps, 2)

    cache_base = None
    cache_fast = None
    outs_base = []
    outs_fast = []
    with torch.enable_grad():
        for t in range(steps):
            out_base, cache_base = model_base.forward_policy_step(
                x_tokens[t: t + 1],
                y_tokens[t: t + 1],
                kv_cache=cache_base,
            )
            out_fast, cache_fast = model_fast.forward_policy_step(
                x_tokens[t: t + 1],
                y_tokens[t: t + 1],
                kv_cache=cache_fast,
            )
            outs_base.append(out_base)
            outs_fast.append(out_fast)

    out_base_all = torch.cat(outs_base, dim=0)
    out_fast_all = torch.cat(outs_fast, dim=0)
    assert torch.allclose(out_base_all, out_fast_all, atol=1e-5, rtol=1e-4)

    model_base.zero_grad(set_to_none=True)
    model_fast.zero_grad(set_to_none=True)
    loss_base = out_base_all.square().mean()
    loss_fast = out_fast_all.square().mean()
    loss_base.backward()
    loss_fast.backward()

    grads_base = [p.grad.detach().clone() for p in model_base.parameters() if p.grad is not None]
    grads_fast = [p.grad.detach().clone() for p in model_fast.parameters() if p.grad is not None]
    assert len(grads_base) == len(grads_fast)
    for grad_base, grad_fast in zip(grads_base, grads_fast):
        assert torch.allclose(grad_base, grad_fast, atol=1e-5, rtol=1e-4)


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


def test_perfeature_forward_policy_step_paged_recompute_attn_matches_no_recompute_semantics(monkeypatch):
    monkeypatch.setenv("TICL_POLICY_PAGED_RECOMPUTE_ATTN", "1")
    torch.manual_seed(411)
    model_base = _build_perfeature_model(recompute_attn=False, nhead=1, emsize=16)
    model_ckpt = _build_perfeature_model(recompute_attn=True, nhead=1, emsize=16)
    model_ckpt.load_state_dict(model_base.state_dict())
    model_base.train()
    model_ckpt.train()

    steps = 5
    x_tokens = torch.randn(steps, 2, 12)
    y_tokens = torch.randn(steps, 2)

    cache_base = None
    cache_ckpt = None
    outs_base = []
    outs_ckpt = []
    for t in range(steps):
        out_base, cache_base = model_base.forward_policy_step(
            x_tokens[t : t + 1],
            y_tokens[t : t + 1],
            kv_cache=cache_base,
            max_cache_len=steps,
            kv_cache_mode="paged",
            kv_cache_page_size=3,
            allow_grad_mutable_cache=True,
        )
        out_ckpt, cache_ckpt = model_ckpt.forward_policy_step(
            x_tokens[t : t + 1],
            y_tokens[t : t + 1],
            kv_cache=cache_ckpt,
            max_cache_len=steps,
            kv_cache_mode="paged",
            kv_cache_page_size=3,
            allow_grad_mutable_cache=True,
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

    for (name_base, p_base), (name_ckpt, p_ckpt) in zip(
        model_base.named_parameters(),
        model_ckpt.named_parameters(),
    ):
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
