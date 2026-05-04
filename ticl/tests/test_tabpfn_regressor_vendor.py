from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ticl.models.tabpfn_regressor_vendor import (
    FullSupportBarDistribution,
    TabPFNOfficialRegressorHarness,
    _compute_regression_loss,
    _ranked_probability_score_loss_from_bar_logits,
    make_standardized_full_support_bar_distribution,
    transform_borders_one,
    translate_probs_across_borders,
)


_VENDOR_ROOT = Path(__file__).resolve().parents[1] / "models" / "vendor" / "tabpfn_v7_1_1"
_UPSTREAM_ROOT = _VENDOR_ROOT / "upstream" / "src" / "tabpfn"


def _load_module_from_path(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _extract_functions(path: Path, names: set[str], namespace: dict[str, Any]) -> dict[str, Any]:
    tree = ast.parse(path.read_text(), filename=str(path))
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    module = ast.Module(body=selected, type_ignores=[])
    compiled = compile(module, str(path), "exec")
    out = dict(namespace)
    exec(compiled, out)  # noqa: S102
    return out


def test_vendor_bar_distribution_matches_upstream_snapshot_cpu_fp32():
    upstream_mod = _load_module_from_path(
        _UPSTREAM_ROOT / "architectures" / "base" / "bar_distribution.py",
        "tabpfn_upstream_bar_distribution",
    )
    borders = torch.linspace(-5.0, 5.0, 101, dtype=torch.float32)
    runtime = FullSupportBarDistribution(borders.clone())
    upstream = upstream_mod.FullSupportBarDistribution(borders.clone())

    torch.manual_seed(0)
    logits = torch.randn(2, 4, 100, dtype=torch.float32)
    targets = torch.tensor(
        [[-4.75, -0.1, float("nan"), 4.95], [0.0, 1.25, -3.5, 4.5]],
        dtype=torch.float32,
    )
    ys = torch.tensor([-4.9, -1.5, 0.0, 3.2, 4.9], dtype=torch.float32)

    runtime_loss = runtime(logits, targets)
    upstream_loss = upstream(logits, targets)
    assert float((runtime_loss - upstream_loss).abs().max()) <= 1e-6

    for method_name in ("mean", "median", "mode"):
        runtime_value = getattr(runtime, method_name)(logits)
        upstream_value = getattr(upstream, method_name)(logits)
        assert float((runtime_value - upstream_value).abs().max()) <= 1e-6

    runtime_pdf = runtime.pdf(logits, targets.nan_to_num(0.0))
    upstream_pdf = upstream.pdf(logits, targets.nan_to_num(0.0))
    assert float((runtime_pdf - upstream_pdf).abs().max()) <= 1e-6

    runtime_cdf = runtime.cdf(logits, ys)
    upstream_cdf = upstream.cdf(logits, ys)
    assert float((runtime_cdf - upstream_cdf).abs().max()) <= 1e-6

    runtime_icdf = runtime.icdf(logits, 0.73)
    upstream_icdf = upstream.icdf(logits, 0.73)
    assert float((runtime_icdf - upstream_icdf).abs().max()) <= 1e-6


def test_vendor_regression_loss_matches_upstream_snapshot_cpu_fp32():
    extracted = _extract_functions(
        _UPSTREAM_ROOT / "finetuning" / "finetuned_regressor.py",
        {"_compute_regression_loss", "_ranked_probability_score_loss_from_bar_logits"},
        {"torch": torch, "Literal": __import__("typing").Literal, "Any": Any},
    )
    bardist = make_standardized_full_support_bar_distribution(num_buckets=9, value_range=3.0)

    torch.manual_seed(1)
    logits = torch.randn(3, 5, 9, dtype=torch.float32)
    targets = torch.tensor(
        [
            [0.2, -0.3, float("nan"), 0.5, 1.1],
            [0.0, -1.5, 1.6, float("nan"), -0.2],
            [2.1, -2.4, 0.1, 0.7, 0.9],
        ],
        dtype=torch.float32,
    )

    runtime_rps = _ranked_probability_score_loss_from_bar_logits(
        logits_BQL=logits,
        targets_BQ=targets,
        bardist_loss_fn=bardist,
        loss_type="crps",
    )
    upstream_rps = extracted["_ranked_probability_score_loss_from_bar_logits"](
        logits_BQL=logits,
        targets_BQ=targets,
        bardist_loss_fn=bardist,
        loss_type="crps",
    )
    assert float((runtime_rps - upstream_rps).abs()) <= 1e-6

    runtime_total = _compute_regression_loss(
        logits_BQL=logits,
        targets_BQ=targets,
        bardist_loss_fn=bardist,
        ce_loss_weight=0.0,
        crps_loss_weight=1.0,
        crls_loss_weight=0.0,
        mse_loss_weight=1.0,
        mse_loss_clip=None,
        mae_loss_weight=0.0,
        mae_loss_clip=None,
    )
    upstream_total = extracted["_compute_regression_loss"](
        logits_BQL=logits,
        targets_BQ=targets,
        bardist_loss_fn=bardist,
        ce_loss_weight=0.0,
        crps_loss_weight=1.0,
        crls_loss_weight=0.0,
        mse_loss_weight=1.0,
        mse_loss_clip=None,
        mae_loss_weight=0.0,
        mae_loss_clip=None,
    )
    assert float((runtime_total - upstream_total).abs()) <= 1e-6


def test_vendor_utils_match_upstream_snapshot_cpu_fp32():
    extracted = _extract_functions(
        _UPSTREAM_ROOT / "utils.py",
        {
            "_repair_borders",
            "_cancel_nan_borders",
            "_map_to_bucket_ix",
            "_cdf",
            "translate_probs_across_borders",
            "transform_borders_one",
        },
        {
            "np": np,
            "npt": np.typing,
            "torch": torch,
            "TransformerMixin": TransformerMixin,
            "Pipeline": Pipeline,
            "Literal": __import__("typing").Literal,
            "REGRESSION_NAN_BORDER_LIMIT_LOWER": -1e3,
            "REGRESSION_NAN_BORDER_LIMIT_UPPER": 1e3,
        },
    )

    torch.manual_seed(2)
    logits = torch.randn(4, 7, dtype=torch.float32)
    frm = torch.linspace(-2.0, 2.0, 8, dtype=torch.float32)
    to = torch.linspace(-3.0, 3.0, 10, dtype=torch.float32)
    runtime_probs = translate_probs_across_borders(logits, frm=frm, to=to)
    upstream_probs = extracted["translate_probs_across_borders"](logits, frm=frm, to=to)
    assert float((runtime_probs - upstream_probs).abs().max()) <= 1e-6

    scaler = StandardScaler().fit(np.array([[-3.0], [-1.0], [0.0], [2.5], [5.0]], dtype=np.float64))
    borders = np.linspace(-5.0, 5.0, 11, dtype=np.float64)
    runtime_transform = transform_borders_one(
        borders,
        scaler,
        repair_nan_borders_after_transform=True,
    )
    upstream_transform = extracted["transform_borders_one"](
        borders,
        scaler,
        repair_nan_borders_after_transform=True,
    )
    assert runtime_transform[1] == upstream_transform[1]
    assert runtime_transform[0] is None and upstream_transform[0] is None
    np.testing.assert_allclose(runtime_transform[2], upstream_transform[2], atol=1e-12, rtol=0.0)


def test_tabpfn_official_regressor_harness_head_and_decode_contract():
    extracted = _extract_functions(
        _UPSTREAM_ROOT / "regressor.py",
        {"_logits_to_output"},
        {"torch": torch, "np": np, "FullSupportBarDistribution": FullSupportBarDistribution},
    )
    harness = TabPFNOfficialRegressorHarness(emsize=8, num_buckets=11, value_range=4.0)
    state_keys = set(harness.state_dict().keys())
    assert "output_projection.0.weight" in state_keys
    assert "output_projection.2.weight" in state_keys

    y_train = torch.tensor([1.0, 2.0, 4.0, 8.0, 16.0], dtype=torch.float32)
    harness.set_target_statistics(y_train)
    assert harness.raw_space_bardist_.num_bars == 11
    assert harness.znorm_space_bardist_.num_bars == 11

    torch.manual_seed(3)
    hidden = torch.randn(2, 3, 8, dtype=torch.float32)
    logits = harness.forward_logits(hidden)
    full = harness.predict_full(logits=logits, quantiles=[0.25, 0.5, 0.75], use_raw_space=True)

    upstream_mean = extracted["_logits_to_output"](
        output_type="mean",
        logits=logits,
        criterion=harness.raw_space_bardist_,
        quantiles=[0.25, 0.5, 0.75],
    )
    upstream_quantiles = extracted["_logits_to_output"](
        output_type="quantiles",
        logits=logits,
        criterion=harness.raw_space_bardist_,
        quantiles=[0.25, 0.5, 0.75],
    )

    np.testing.assert_allclose(full["mean"].detach().cpu().numpy(), upstream_mean, atol=1e-6, rtol=0.0)
    for got, expected in zip(full["quantiles"], upstream_quantiles):
        np.testing.assert_allclose(got.detach().cpu().numpy(), expected, atol=1e-6, rtol=0.0)


def test_tabpfn_official_regressor_harness_constant_target_contract():
    harness = TabPFNOfficialRegressorHarness(emsize=6, num_buckets=9, value_range=3.0)
    harness.set_target_statistics(torch.tensor([7.5, 7.5, 7.5], dtype=torch.float32))
    assert harness.is_constant_target_ is True

    logits = torch.randn(4, 2, 9, dtype=torch.float32)
    full = harness.predict_full(logits=logits, quantiles=[0.1, 0.9], use_raw_space=True)
    assert tuple(full["logits"].shape) == (4, 2, 1)
    assert full["criterion"] is harness.znorm_space_bardist_
    assert torch.allclose(full["mean"], torch.full((4, 2), 7.5))
    assert torch.allclose(full["median"], torch.full((4, 2), 7.5))
    assert torch.allclose(full["mode"], torch.full((4, 2), 7.5))
    assert len(full["quantiles"]) == 2
    assert torch.allclose(full["quantiles"][0], torch.full((4, 2), 7.5))
