#!/usr/bin/env python
"""Read-only reward/action transform semantics audit for vectorized prior rollout.

This audit is intentionally not a distribution intervention.  It checks one
specific computation-chain contract:

    frozen h -> vectorized environment group -> action transform / reward transform

The current maintained selected lists use ``none`` for both transforms, so this
script also builds controlled in-memory h variants to expose whether the
family-vectorized path would respect future per-h transform sources.
"""

from __future__ import annotations

import argparse
import copy
import csv
import inspect
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from ticl.model_configs import get_model_default_config  # noqa: E402
from ticl.priors.environment_prior import EnvironmentPrior  # noqa: E402


ART = Path("/home/chen/RLPFN/artifacts")
DEFAULT_SELECTED_CSV = (
    ART
    / "phase2_profile_v2p_dimcoupled_reward_balance_probe_0516"
    / "selected_prior_envs.csv"
)
DEFAULT_OUTPUT_DIR = ART / "phase2_profile_v2p_reward_action_semantics_audit_0516"


def _jsonable(value: Any) -> Any:
    if torch.is_tensor(value):
        if value.numel() == 1:
            return _jsonable(value.detach().cpu().item())
        return [_jsonable(v) for v in value.detach().cpu().reshape(-1).tolist()]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            return str(value)
        return value
    return value


def _load_selected_rows(selected_csv: Path, rule: str) -> list[dict[str, str]]:
    with selected_csv.open("r", encoding="utf-8", newline="") as handle:
        rows = [dict(row) for row in csv.DictReader(handle) if str(row.get("rule")) == str(rule)]
    if not rows:
        raise ValueError(f"No rows for rule={rule!r} in {selected_csv}")
    return rows


def _load_h(row: dict[str, str]) -> dict[str, Any]:
    path = Path(str(row["full_frozen_h_json"])).expanduser()
    with path.open("r", encoding="utf-8") as handle:
        h = json.load(handle)
    if not isinstance(h, dict):
        raise ValueError(f"Expected frozen h dict in {path}")
    return h


def _select_same_family_rows(
    *,
    prior: EnvironmentPrior,
    rows: list[dict[str, str]],
    n_envs: int,
) -> tuple[list[dict[str, Any]], list[int], list[dict[str, str]]]:
    loaded: list[tuple[dict[str, Any], int, dict[str, str], tuple[Any, ...]]] = []
    for row in rows:
        h = _load_h(row)
        sig = (
            prior._normalize_family(h.get("family", "scm")),
            bool(prior._resolve_reference_semantics_enabled(h)),
        )
        seed = int(float(row.get("env_seed", row.get("seed", 0)) or 0))
        loaded.append((h, seed, row, sig))
    sig_counts = Counter(item[3] for item in loaded)
    target_sig, target_count = sig_counts.most_common(1)[0]
    if int(target_count) < int(n_envs):
        raise ValueError(f"Need {n_envs} rows in one family/ref group, got {target_count} for {target_sig}")
    selected = [item for item in loaded if item[3] == target_sig][: int(n_envs)]
    return [copy.deepcopy(item[0]) for item in selected], [int(item[1]) for item in selected], [item[2] for item in selected]


def _current_vectorized_source_flags() -> dict[str, Any]:
    source = inspect.getsource(EnvironmentPrior._rollout_family_group_vectorized_with_policy)
    outer_source = inspect.getsource(EnvironmentPrior._environment_group_signature)
    pre_rollout_generators = source.split("rollout_generators =", 1)[0]
    return {
        "outer_group_signature_includes_reward_transform": (
            "_resolve_reinforce_reward_transform(h)" in outer_source
        ),
        "outer_group_signature_includes_action_transform": (
            "_resolve_reinforce_action_transform(h)" in outer_source
        ),
        "family_group_signature_includes_reward_transform": (
            "_resolve_reinforce_reward_transform(h)" in pre_rollout_generators
        ),
        "family_group_signature_includes_action_transform": (
            "_resolve_reinforce_action_transform(h)" in pre_rollout_generators
        ),
        "reward_compose_uses_config_transform": (
            "mode=self._resolve_reinforce_reward_transform(self.config)" in source
        ),
    }


def _mode(h: dict[str, Any], key: str, default: str = "none") -> str:
    return str(h.get(key, default)).strip().lower()


def _group_h_list_for_current_source(
    *,
    prior: EnvironmentPrior,
    h_list: list[dict[str, Any]],
    env_seeds: list[int],
    flags: dict[str, Any],
) -> list[tuple[list[int], list[dict[str, Any]], list[int]]]:
    # Mirror the official maintained dispatch first.  In family mode it calls
    # `_environment_group_signature`, which is the public semantic boundary for
    # homogeneous rollout groups before `_rollout_family_group_vectorized...`.
    groups: dict[tuple[Any, ...], list[tuple[int, dict[str, Any], int]]] = defaultdict(list)
    for idx, (h, seed) in enumerate(zip(h_list, env_seeds)):
        sig = prior._environment_group_signature(h, "family")
        groups[sig].append((idx, h, int(seed)))
    return [
        ([idx for idx, _, _ in items], [h for _, h, _ in items], [seed for _, _, seed in items])
        for items in groups.values()
    ]


def _transform_reward_per_h(
    *,
    prior: EnvironmentPrior,
    h_list: list[dict[str, Any]],
    reward_raw: torch.Tensor,
    reward_clip: torch.Tensor,
    rms_eps: torch.Tensor,
    tanh_c: torch.Tensor,
    tanh_bound: torch.Tensor,
) -> torch.Tensor:
    out = []
    for idx, h in enumerate(h_list):
        out.append(
            prior._transform_exact_scm_base_reward(
                reward_raw[idx : idx + 1],
                reward_clip=reward_clip[idx : idx + 1],
                mode=prior._resolve_reinforce_reward_transform(h),
                rms_eps=rms_eps[idx : idx + 1],
                tanh_c=tanh_c[idx : idx + 1],
                tanh_bound=tanh_bound[idx : idx + 1],
            )
        )
    return torch.cat(out, dim=0)


def _transform_action_per_h(
    *,
    prior: EnvironmentPrior,
    h_list: list[dict[str, Any]],
    action_raw: torch.Tensor,
    action_mask: torch.Tensor,
    action_rms_eps: torch.Tensor,
    action_clip_bound: torch.Tensor,
) -> torch.Tensor:
    out = []
    for idx, h in enumerate(h_list):
        out.append(
            prior._transform_reinforce_action(
                action_raw[idx : idx + 1],
                mode=prior._resolve_reinforce_action_transform(h),
                rms_eps=action_rms_eps[idx : idx + 1],
                clip_bound=action_clip_bound[idx : idx + 1],
                mask=action_mask[idx : idx + 1],
            )
        )
    return torch.cat(out, dim=0)


def _make_case(base_h: list[dict[str, Any]], case_name: str) -> list[dict[str, Any]]:
    h_list = [copy.deepcopy(h) for h in base_h]
    for h in h_list:
        h["reward_dropout_randomize"] = False
    if case_name == "baseline_selected":
        return h_list
    if case_name == "homogeneous_reward_clip":
        for h in h_list:
            h["reinforce_reward_transform"] = "clip"
            h["reinforce_reward_tanh_bound"] = 0.5
        return h_list
    if case_name == "mixed_reward_second_clip":
        for idx, h in enumerate(h_list):
            h["reinforce_reward_transform"] = "none" if idx == 0 else "clip"
            h["reinforce_reward_tanh_bound"] = 0.5
        return h_list
    if case_name == "homogeneous_action_clip":
        for h in h_list:
            h["reinforce_action_transform"] = "clip"
            h["reinforce_action_clip_bound"] = 0.75
        return h_list
    if case_name == "mixed_action_second_clip":
        for idx, h in enumerate(h_list):
            h["reinforce_action_transform"] = "none" if idx == 0 else "clip"
            h["reinforce_action_clip_bound"] = 0.75
        return h_list
    raise ValueError(f"Unknown case: {case_name}")


def _audit_case(
    *,
    prior: EnvironmentPrior,
    base_h: list[dict[str, Any]],
    env_seeds: list[int],
    case_name: str,
    device: torch.device,
    flags: dict[str, Any],
) -> dict[str, Any]:
    h_list = _make_case(base_h, case_name)
    batch_size = len(h_list)
    groups = _group_h_list_for_current_source(prior=prior, h_list=h_list, env_seeds=env_seeds, flags=flags)

    max_reward_raw = torch.linspace(-3.0, 3.0, steps=batch_size, device=device, dtype=torch.float32)
    if batch_size >= 4:
        max_reward_raw[0] = -3.0
        max_reward_raw[1] = -0.75
        max_reward_raw[-2] = 0.75
        max_reward_raw[-1] = 3.0
    reward_actual = torch.empty((batch_size,), device=device, dtype=torch.float32)
    action_actual_rows: list[torch.Tensor | None] = [None] * batch_size
    reward_clip = torch.empty((batch_size,), device=device, dtype=torch.float32)
    reward_rms_eps = torch.empty((batch_size,), device=device, dtype=torch.float32)
    reward_tanh_c = torch.empty((batch_size,), device=device, dtype=torch.float32)
    reward_tanh_bound = torch.empty((batch_size,), device=device, dtype=torch.float32)
    action_rms_eps = torch.empty((batch_size,), device=device, dtype=torch.float32)
    action_clip_bound = torch.empty((batch_size,), device=device, dtype=torch.float32)
    action_dim_per_sample = torch.empty((batch_size,), device=device, dtype=torch.long)
    max_action_dim = 1
    group_summaries = []

    for group_indices, group_h, group_seeds in groups:
        env = prior._sample_environment_family_coarse_batch(
            group_h,
            device=device,
            rng_seeds=group_seeds,
            build_policy_generator=False,
            preserve_skipped_generator_rng=True,
            _disable_rejection=True,
        )
        idx_t = torch.as_tensor(group_indices, device=device, dtype=torch.long)
        reward_clip.index_copy_(0, idx_t, env["reward_clip"])
        reward_rms_eps.index_copy_(0, idx_t, env["reinforce_reward_rms_eps"])
        reward_tanh_c.index_copy_(0, idx_t, env["reinforce_reward_tanh_c"])
        reward_tanh_bound.index_copy_(0, idx_t, env["reinforce_reward_tanh_bound"])
        action_rms_eps.index_copy_(0, idx_t, env["reinforce_action_rms_eps"])
        action_clip_bound.index_copy_(0, idx_t, env["reinforce_action_clip_bound"])
        action_dim_per_sample.index_copy_(0, idx_t, env["action_dim_per_sample"])
        max_action_dim = max(max_action_dim, int(env["action_dim"]))

        group_reward_raw = max_reward_raw.index_select(0, idx_t)
        if bool(flags["reward_compose_uses_config_transform"]):
            reward_mode = prior._resolve_reinforce_reward_transform(prior.config)
            reward_mode_source = "config"
        else:
            reward_mode = env.get("reinforce_reward_transform", "none")
            reward_mode_source = "env_group"
        group_reward = prior._transform_exact_scm_base_reward(
            group_reward_raw,
            reward_clip=env["reward_clip"],
            mode=reward_mode,
            rms_eps=env["reinforce_reward_rms_eps"],
            tanh_c=env["reinforce_reward_tanh_c"],
            tanh_bound=env["reinforce_reward_tanh_bound"],
        )
        reward_actual.index_copy_(0, idx_t, group_reward)

        action_dim = int(env["action_dim"])
        raw_row = torch.full((len(group_h), action_dim), 1.5, device=device, dtype=torch.float32)
        if action_dim > 1:
            raw_row[:, 0] = -1.5
        mask = (
            torch.arange(action_dim, device=device).unsqueeze(0)
            < env["action_dim_per_sample"].unsqueeze(1)
        ).to(dtype=torch.float32)
        group_action = prior._transform_reinforce_action(
            raw_row,
            mode=env.get("reinforce_action_transform", "none"),
            rms_eps=env["reinforce_action_rms_eps"],
            clip_bound=env["reinforce_action_clip_bound"],
            mask=mask,
        )
        for local_idx, global_idx in enumerate(group_indices):
            action_actual_rows[int(global_idx)] = group_action[local_idx].detach()

        group_summaries.append(
            {
                "indices": list(map(int, group_indices)),
                "env_reward_transform": str(env.get("reinforce_reward_transform", "none")),
                "reward_mode_source": reward_mode_source,
                "reward_mode_used": str(reward_mode),
                "env_action_transform": str(env.get("reinforce_action_transform", "none")),
                "h_reward_modes": [prior._resolve_reinforce_reward_transform(h) for h in group_h],
                "h_action_modes": [prior._resolve_reinforce_action_transform(h) for h in group_h],
            }
        )

    padded_action_rows = []
    for row in action_actual_rows:
        if row is None:
            raise RuntimeError("internal audit error: missing action row")
        padded = torch.zeros((max_action_dim,), device=device, dtype=torch.float32)
        padded[: int(row.numel())] = row
        padded_action_rows.append(padded)
    action_actual = torch.stack(padded_action_rows, dim=0)
    action_raw_global = torch.full_like(action_actual, 1.5)
    if action_actual.shape[-1] > 1:
        action_raw_global[:, 0] = -1.5
    action_mask_global = (
        torch.arange(action_actual.shape[-1], device=device).unsqueeze(0)
        < action_dim_per_sample.unsqueeze(1)
    ).to(dtype=torch.float32)
    reward_expected = _transform_reward_per_h(
        prior=prior,
        h_list=h_list,
        reward_raw=max_reward_raw,
        reward_clip=reward_clip,
        rms_eps=reward_rms_eps,
        tanh_c=reward_tanh_c,
        tanh_bound=reward_tanh_bound,
    )
    action_expected = _transform_action_per_h(
        prior=prior,
        h_list=h_list,
        action_raw=action_raw_global,
        action_mask=action_mask_global,
        action_rms_eps=action_rms_eps,
        action_clip_bound=action_clip_bound,
    )

    reward_abs = (reward_actual - reward_expected).abs()
    action_abs = ((action_actual - action_expected).abs() * action_mask_global).reshape(batch_size, -1)
    action_row_max = action_abs.max(dim=1).values
    reward_pass = bool(torch.allclose(reward_actual, reward_expected, atol=1e-6, rtol=1e-6))
    action_pass = bool(torch.all(action_row_max <= 1e-6).item())
    return {
        "case": case_name,
        "groups": group_summaries,
        "h_reward_mode_counts": dict(Counter(prior._resolve_reinforce_reward_transform(h) for h in h_list)),
        "h_action_mode_counts": dict(Counter(prior._resolve_reinforce_action_transform(h) for h in h_list)),
        "reward_actual": reward_actual.detach().cpu().tolist(),
        "reward_expected_per_h": reward_expected.detach().cpu().tolist(),
        "reward_max_abs_diff": float(reward_abs.max().detach().cpu().item()),
        "reward_mismatch_indices": [
            int(i) for i, v in enumerate(reward_abs.detach().cpu().tolist()) if float(v) > 1e-6
        ],
        "reward_pass": reward_pass,
        "action_max_abs_diff": float(action_row_max.max().detach().cpu().item()),
        "action_mismatch_indices": [
            int(i) for i, v in enumerate(action_row_max.detach().cpu().tolist()) if float(v) > 1e-6
        ],
        "action_pass": action_pass,
        "pass": bool(reward_pass and action_pass),
    }


def _write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Reward/action vectorized semantics audit",
        "",
        f"- selected_csv: `{report['selected_csv']}`",
        f"- rule: `{report['rule']}`",
        f"- source_flags: `{report['source_flags']}`",
        f"- baseline_current_selected_transform_safe: `{report['baseline_current_selected_transform_safe']}`",
        f"- controlled_semantic_bug_found: `{report['controlled_semantic_bug_found']}`",
        "",
        "| case | pass | reward max diff | action max diff | reward mismatches | action mismatches |",
        "| --- | --- | ---: | ---: | --- | --- |",
    ]
    for case in report["cases"]:
        lines.append(
            "| {case} | {passed} | {rd:.6g} | {ad:.6g} | {rm} | {am} |".format(
                case=case["case"],
                passed=case["pass"],
                rd=float(case["reward_max_abs_diff"]),
                ad=float(case["action_max_abs_diff"]),
                rm=case["reward_mismatch_indices"],
                am=case["action_mismatch_indices"],
            )
        )
    lines.extend(
        [
            "",
            "Interpretation:",
            "",
            "- `baseline_selected` checks the current maintained selected rows.",
            "- Controlled cases are in-memory probes of future non-`none` h sources; they should pass if the vectorized path is semantically per-h.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selected-csv", type=Path, default=DEFAULT_SELECTED_CSV)
    parser.add_argument("--rule", type=str, default="gym_q90_obs_action_range")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--n-envs", type=int, default=4)
    args = parser.parse_args()

    device = torch.device(str(args.device))
    cfg = copy.deepcopy(get_model_default_config("rlpfn"))
    prior = EnvironmentPrior(copy.deepcopy(cfg["prior"]["environment"]))
    rows = _load_selected_rows(args.selected_csv, args.rule)
    base_h, env_seeds, selected_rows = _select_same_family_rows(prior=prior, rows=rows, n_envs=int(args.n_envs))
    flags = _current_vectorized_source_flags()

    all_reward_modes = Counter()
    all_action_modes = Counter()
    for row in rows:
        h = _load_h(row)
        all_reward_modes[prior._resolve_reinforce_reward_transform(h)] += 1
        all_action_modes[prior._resolve_reinforce_action_transform(h)] += 1

    case_names = [
        "baseline_selected",
        "homogeneous_reward_clip",
        "mixed_reward_second_clip",
        "homogeneous_action_clip",
        "mixed_action_second_clip",
    ]
    cases = [
        _audit_case(
            prior=prior,
            base_h=base_h,
            env_seeds=env_seeds,
            case_name=case_name,
            device=device,
            flags=flags,
        )
        for case_name in case_names
    ]
    controlled_cases = [case for case in cases if case["case"] != "baseline_selected"]
    controlled_bug_found = any(not bool(case["pass"]) for case in controlled_cases)
    report = {
        "analysis_entry": "phase2_vectorized_reward_action_semantics_audit",
        "selected_csv": str(args.selected_csv),
        "rule": str(args.rule),
        "n_envs": int(args.n_envs),
        "device": str(device),
        "source_flags": flags,
        "selected_h_paths": [str(row.get("full_frozen_h_json", "")) for row in selected_rows],
        "selected_env_seeds": env_seeds,
        "all_selected_reward_transform_counts": dict(all_reward_modes),
        "all_selected_action_transform_counts": dict(all_action_modes),
        "baseline_current_selected_transform_safe": (
            set(all_reward_modes.keys()) == {"none"} and set(all_action_modes.keys()) == {"none"}
        ),
        "controlled_semantic_bug_found": bool(controlled_bug_found),
        "cases": cases,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "reward_action_semantics_audit.json"
    md_path = args.output_dir / "reward_action_semantics_audit.md"
    json_path.write_text(json.dumps(_jsonable(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_markdown(_jsonable(report), md_path)
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")
    print(json.dumps(_jsonable({
        "baseline_safe": report["baseline_current_selected_transform_safe"],
        "controlled_bug_found": report["controlled_semantic_bug_found"],
        "source_flags": flags,
        "case_pass": {case["case"]: case["pass"] for case in cases},
    }), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
