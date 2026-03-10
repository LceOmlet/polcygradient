import math
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
import torch


_ACCEPTANCE_ENV = "TICL_RUN_BIG_BATCH_ACCEPTANCE"


def _require_big_batch_acceptance():
    if str(os.environ.get(_ACCEPTANCE_ENV, "0")).strip().lower() not in {"1", "true", "yes", "on"}:
        pytest.skip(f"set {_ACCEPTANCE_ENV}=1 to run big-batch acceptance tests")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for big-batch acceptance tests")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _run_rlpfn_cli(extra_args):
    repo_root = _repo_root()
    env = os.environ.copy()
    env.setdefault("MLFLOW_HOSTNAME", "")
    cmd = [
        sys.executable,
        "-m",
        "ticl.fit_model",
        "rlpfn",
        "--epochs",
        "1",
        "--num-steps",
        "1",
        "--seed-everything",
        "True",
        *list(extra_args),
    ]
    proc = subprocess.run(
        cmd,
        cwd=str(repo_root),
        env=env,
        capture_output=True,
        text=True,
        timeout=1800,
    )
    if proc.returncode != 0:
        raise AssertionError(
            "rlpfn CLI acceptance run failed\n"
            f"cmd: {' '.join(cmd)}\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}"
        )
    combined = f"{proc.stdout}\n{proc.stderr}"
    match = re.search(r"Policy phase log file:\s+(.+)", combined)
    if match is None:
        raise AssertionError(f"failed to locate phase log path in CLI output\n{combined}")
    log_ref = match.group(1).strip()
    log_path = (repo_root / log_ref).resolve() if not os.path.isabs(log_ref) else Path(log_ref)
    if not log_path.exists():
        raise AssertionError(f"phase log file not found: {log_path}")
    phase_lines = [line.strip() for line in log_path.read_text().splitlines() if line.startswith("[pg-phase]")]
    if not phase_lines:
        raise AssertionError(f"no [pg-phase] line found in {log_path}")
    return log_path, phase_lines[-1]


def _parse_phase_line(line):
    out = {}
    for token in line.split():
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        out[key] = value
    return out


def _float_field(fields, key):
    if key not in fields:
        raise AssertionError(f"missing field {key} in phase line: {fields}")
    try:
        value = float(fields[key])
    except Exception as exc:
        raise AssertionError(f"field {key} is not a float: {fields[key]}") from exc
    if not math.isfinite(value):
        raise AssertionError(f"field {key} is not finite: {value}")
    return value


def test_rlpfn_default_big_batch_phase_log_acceptance():
    _require_big_batch_acceptance()
    log_path, line = _run_rlpfn_cli([])
    fields = _parse_phase_line(line)

    assert fields["status"] == "ok"
    assert fields["chunk"] == "1024"
    assert fields["tbptt"] == "32"

    reward_mean = _float_field(fields, "reward_mean")
    reward_std = _float_field(fields, "reward_std")
    objective_total_v5n = _float_field(fields, "objective_total_v5n")
    aev5n_loss_mul = _float_field(fields, "aev5n_loss_mul")
    aev5n_scale = _float_field(fields, "aev5n_scale")
    aev5n_std_ref = _float_field(fields, "aev5n_std_ref")
    aev5n_bias_state = _float_field(fields, "aev5n_bias_state")
    aev5n_bias_therm = _float_field(fields, "aev5n_bias_therm")
    aev5n_bias_step = _float_field(fields, "aev5n_bias_step")
    lip_mclip = _float_field(fields, "lip_mclip")
    lip_mtail = _float_field(fields, "lip_mtail")
    lip_mproj = _float_field(fields, "lip_mproj")
    lip_oclip = _float_field(fields, "lip_oclip")
    lip_oproj = _float_field(fields, "lip_oproj")

    assert reward_std > 0.0
    assert aev5n_loss_mul > 0.0
    assert abs(aev5n_scale - aev5n_loss_mul) <= 5e-4
    assert abs(aev5n_std_ref - reward_std) <= 5e-4
    assert aev5n_bias_state >= 0.0
    assert aev5n_bias_therm >= 0.0
    assert aev5n_bias_step >= 0.0
    assert lip_mclip >= 0.0
    assert lip_mtail >= 0.0
    assert lip_mproj >= 0.0
    assert lip_oclip >= 0.0
    assert lip_oproj >= 0.0

    if abs(aev5n_loss_mul - 1.0) > 1e-3:
        assert abs(objective_total_v5n - reward_mean) > max(1e-5, 1e-2 * max(abs(reward_mean), 1e-6)), (
            f"phase log regression in {log_path}: "
            f"objective_total_v5n={objective_total_v5n}, reward_mean={reward_mean}, aev5n_loss_mul={aev5n_loss_mul}"
        )


def test_rlpfn_forced_v5_next_phase_log_thermostat_acceptance():
    _require_big_batch_acceptance()
    log_path, line = _run_rlpfn_cli(
        [
            "--anti-explosion-vanishing-v5-next-loss-target-std",
            "1e-3",
            "--anti-explosion-vanishing-v5-next-loss-scale-lo",
            "1e-2",
            "--anti-explosion-vanishing-v5-next-loss-scale-hi",
            "1.0",
        ]
    )
    fields = _parse_phase_line(line)

    assert fields["status"] == "ok"
    assert fields["chunk"] == "1024"
    assert fields["tbptt"] == "32"

    reward_mean = _float_field(fields, "reward_mean")
    objective_total_v5n = _float_field(fields, "objective_total_v5n")
    aev5n_loss_mul = _float_field(fields, "aev5n_loss_mul")
    aev5n_scale = _float_field(fields, "aev5n_scale")
    aev5n_std_ref = _float_field(fields, "aev5n_std_ref")
    aev5n_bias_therm = _float_field(fields, "aev5n_bias_therm")
    lip_mproj = _float_field(fields, "lip_mproj")

    assert aev5n_std_ref > 0.0
    assert 0.0 < aev5n_loss_mul < 0.5
    assert abs(aev5n_scale - aev5n_loss_mul) <= 5e-4
    assert aev5n_bias_therm > 0.0
    assert lip_mproj >= 0.0
    assert abs(objective_total_v5n - reward_mean) > max(1e-5, 1e-2 * max(abs(reward_mean), 1e-6)), (
        f"forced thermostat regression in {log_path}: "
        f"objective_total_v5n={objective_total_v5n}, reward_mean={reward_mean}, aev5n_loss_mul={aev5n_loss_mul}"
    )
