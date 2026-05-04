import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_OUTPUT_DIR = (
    "/home/chen/RLPFN/artifacts/"
    "phase2_gpu0_pretrain_gate_selection_0429"
)
DEFAULT_POPULATION_FEATURES = (
    "/home/chen/RLPFN/artifacts/"
    "phase2_gpu0_population_trainability_feature_analysis_0429/"
    "population_gpu0_trainability_features.csv"
)
DEFAULT_ENRICHED_POPULATION = (
    "/home/chen/RLPFN/artifacts/"
    "phase2_prior_generation_rule_audit_0429/"
    "enriched_population.csv"
)


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def _mass_share(env_sums: pd.Series) -> float | None:
    vals = pd.to_numeric(env_sums, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    denom = float(np.abs(vals).sum())
    if denom <= 1e-12:
        return None
    return float(np.abs(vals).max() / denom)


def _finite_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _load_h(path: str) -> dict[str, Any]:
    return json.loads(Path(path).expanduser().read_text(encoding="utf-8"))


def _ctrl_enabled(h: dict[str, Any]) -> bool:
    return bool(
        float(h.get("_ctrl_reward_enable_u", 1.0)) < float(h.get("ctrl_reward_enable_prob", 0.0))
    )


def _component_proxy_row(row: pd.Series, *, n_steps: int) -> dict[str, Any]:
    h = _load_h(str(row["full_frozen_h_json"]))
    ctrl_enabled = _ctrl_enabled(h)
    ctrl_weight = float(h.get("ctrl_reward_weight") or 0.0)
    # Canonical pretrain probes use approximately unit-std stochastic actions. For the
    # current exact-SCM ctrl implementation, observed sidecars match weight * horizon
    # rather than weight * horizon * action_dim, so keep this deliberately simple.
    ctrl_sum_proxy = -ctrl_weight * float(n_steps) if ctrl_enabled else 0.0
    env_sum_proxy = _finite_float(row.get("env_sum_mean"))
    if env_sum_proxy is None:
        env_sum_proxy = 0.0
    denom = abs(ctrl_sum_proxy) + abs(env_sum_proxy)
    ctrl_abs_share = abs(ctrl_sum_proxy) / denom if denom > 1e-12 else 0.0
    total_sum_proxy = env_sum_proxy + ctrl_sum_proxy
    return {
        "ctrl_enabled_proxy": bool(ctrl_enabled),
        "ctrl_weight_proxy": float(ctrl_weight),
        "ctrl_sum_proxy": float(ctrl_sum_proxy),
        "env_sum_proxy": float(env_sum_proxy),
        "total_sum_with_ctrl_proxy": float(total_sum_proxy),
        "ctrl_abs_share_total_proxy": float(ctrl_abs_share),
    }


def _group_stats(group: pd.DataFrame) -> dict[str, Any]:
    vals = pd.to_numeric(group["env_sum_mean"], errors="coerce").replace([np.inf, -np.inf], np.nan)
    total_vals = pd.to_numeric(group["total_sum_with_ctrl_proxy"], errors="coerce").replace([np.inf, -np.inf], np.nan)
    ctrl_vals = pd.to_numeric(group["ctrl_sum_proxy"], errors="coerce").replace([np.inf, -np.inf], np.nan)
    neg = vals[vals < 0.0]
    total_neg = total_vals[total_vals < 0.0]
    neg_share = None
    if float(np.abs(neg).sum()) > 1e-12:
        neg_share = float(np.abs(neg).max() / np.abs(neg).sum())
    total_neg_share = None
    if float(np.abs(total_neg).sum()) > 1e-12:
        total_neg_share = float(np.abs(total_neg).max() / np.abs(total_neg).sum())
    return {
        "n": int(len(group)),
        "seeds": [int(v) for v in group["seed"].tolist()],
        "env_sum_mean": float(vals.mean()),
        "env_sum_min": float(vals.min()),
        "env_sum_max": float(vals.max()),
        "env_sum_std": float(vals.std(ddof=0)),
        "negative_count": int((vals < 0.0).sum()),
        "worst_abs_share_all_proxy": _mass_share(vals),
        "worst_abs_share_negative_proxy": neg_share,
        "total_sum_with_ctrl_mean_proxy": float(total_vals.mean()),
        "total_sum_with_ctrl_min_proxy": float(total_vals.min()),
        "worst_abs_share_total_with_ctrl_proxy": _mass_share(total_vals),
        "worst_abs_share_negative_total_with_ctrl_proxy": total_neg_share,
        "worst_ctrl_abs_share_total_proxy": float(group["ctrl_abs_share_total_proxy"].max()),
        "mean_ctrl_abs_share_total_proxy": float(group["ctrl_abs_share_total_proxy"].mean()),
        "ctrl_enabled_count_proxy": int(group["ctrl_enabled_proxy"].sum()),
        "ctrl_sum_proxy_mean": float(ctrl_vals.mean()),
        "ctrl_sum_proxy_min": float(ctrl_vals.min()),
        "mean_bad_q90_fraction": float(group["bad_q90_fraction"].mean()),
        "mean_low_positive_tail_fraction": float(group["low_positive_tail_fraction"].mean()),
        "mean_pos_frac": float(group["pos_frac_mean"].mean()),
        "mean_reward_std": float(group["param_reward_std"].mean()),
        "mean_return_std": float(group["param_return_std"].mean()),
        "mean_top_tail_gap": float(group["top_tail_gap_mean"].mean()),
        "mean_reward_sens": float(group["param_reward_sens"].mean()),
        "mean_state_sens": float(group["param_state_sens"].mean()),
        "mean_obs_rank": float(group["param_obs_rank"].mean()),
        "mean_G5": float(group["param_G5"].mean()),
    }


def _write_group_csv(group: pd.DataFrame, output_path: Path, rule_name: str) -> None:
    out = group.copy()
    out.insert(0, "rule", rule_name)
    out.to_csv(output_path, index=False)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="GPU0-only pretrain gate selector from canonical population probes."
    )
    parser.add_argument("--population-features", default=DEFAULT_POPULATION_FEATURES)
    parser.add_argument("--enriched-population", default=DEFAULT_ENRICHED_POPULATION)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--max-worst-abs-share", type=float, default=0.20)
    parser.add_argument("--max-worst-total-with-ctrl-abs-share", type=float, default=0.30)
    parser.add_argument("--max-ctrl-abs-share", type=float, default=0.50)
    parser.add_argument("--max-reward-state-to-action-ratio", type=float, default=None)
    parser.add_argument("--min-reward-action-gain-fraction", type=float, default=None)
    parser.add_argument("--n-steps", type=int, default=256)
    parser.add_argument("--bad-q90-high", type=float, default=0.80)
    parser.add_argument("--positive-frac-low", type=float, default=0.20)
    parser.add_argument("--strict-positive-frac-low", type=float, default=0.10)
    args = parser.parse_args(argv)

    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.population_features)
    enriched = pd.read_csv(args.enriched_population)
    if "full_frozen_h_json" not in enriched.columns:
        raise ValueError(f"{args.enriched_population} must contain full_frozen_h_json.")
    merge_cols = ["seed", "full_frozen_h_json"]
    df = df.merge(enriched[merge_cols], on="seed", how="left", validate="one_to_one")
    missing_paths = int(df["full_frozen_h_json"].isna().sum())
    if missing_paths:
        raise ValueError(f"Missing full_frozen_h_json for {missing_paths} selected population rows.")
    component_rows = [
        _component_proxy_row(row, n_steps=int(args.n_steps))
        for _, row in df.iterrows()
    ]
    component_df = pd.DataFrame(component_rows)
    df = pd.concat([df.reset_index(drop=True), component_df.reset_index(drop=True)], axis=1)

    for col in [
        "param_reward_std",
        "param_return_std",
        "top_tail_gap_mean",
        "param_amp_geom_ratio",
        "param_obs_ratio",
        "param_obs_top",
        "param_state_dim",
        "param_reward_noise_gain",
        "param_reward_action_gain",
        "param_reward_state_gain",
        "param_action_dim",
        "param_obs_std",
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["reward_state_to_action_ratio_proxy"] = (
        df["param_reward_state_gain"] / df["param_reward_action_gain"].clip(lower=1e-12)
    )

    reward_std_hi = float(df["param_reward_std"].quantile(2.0 / 3.0))
    return_std_hi = float(df["param_return_std"].quantile(2.0 / 3.0))
    top_tail_gap_hi = float(df["top_tail_gap_mean"].quantile(2.0 / 3.0))

    high_vol_count = (
        (df["param_reward_std"] >= reward_std_hi).astype(int)
        + (df["param_return_std"] >= return_std_hi).astype(int)
        + (df["top_tail_gap_mean"] >= top_tail_gap_hi).astype(int)
    )
    df["gate_high_volatility_2of3"] = high_vol_count >= 2
    df["gate_structural_no_tail"] = (
        (df["bad_q90_fraction"] >= float(args.bad_q90_high))
        & (df["pos_frac_mean"] < float(args.strict_positive_frac_low))
    )
    df["gate_low_tail"] = (
        (df["low_positive_tail_fraction"] >= float(args.bad_q90_high))
        | (df["pos_frac_mean"] < float(args.positive_frac_low))
    )
    df["gate_ctrl_dominance_proxy"] = (
        pd.to_numeric(df["ctrl_abs_share_total_proxy"], errors="coerce")
        > float(args.max_ctrl_abs_share)
    )
    df["gate_topology_action_coupling_weak"] = False
    if args.max_reward_state_to_action_ratio is not None:
        df["gate_topology_action_coupling_weak"] = (
            df["gate_topology_action_coupling_weak"]
            | (
                pd.to_numeric(df["reward_state_to_action_ratio_proxy"], errors="coerce")
                > float(args.max_reward_state_to_action_ratio)
            )
        )
    if args.min_reward_action_gain_fraction is not None:
        df["gate_topology_action_coupling_weak"] = (
            df["gate_topology_action_coupling_weak"]
            | (
                pd.to_numeric(df["param_reward_action_gain"], errors="coerce")
                < float(args.min_reward_action_gain_fraction)
            )
        )
    df["gate_env_reject"] = (
        df["gate_structural_no_tail"]
        | (df["gate_low_tail"] & df["gate_high_volatility_2of3"])
        | df["gate_ctrl_dominance_proxy"]
        | df["gate_topology_action_coupling_weak"]
    )

    # Diagnostic-only structural patterns from the GPU0 root-cause report.
    df["diagnostic_amp_noise_obs"] = (
        (df["param_amp_geom_ratio"] > 0.66)
        & (df["param_reward_noise_gain"] <= 0.12)
        & (df["param_obs_std"] <= 0.99)
    )
    df["diagnostic_obs_geometry_dim"] = (
        (df["param_obs_ratio"] > 0.59)
        & (df["param_obs_top"] <= 0.75)
        & (df["param_state_dim"] > 220)
    )

    selected_groups: list[dict[str, Any]] = []
    group_csvs: list[str] = []
    candidates = df[~df["gate_env_reject"]].copy()
    candidates = candidates.sort_values(["seed"], kind="mergesort").reset_index(drop=True)
    used: set[int] = set()
    group_idx = 0
    # Deterministic first-by-seed sliding selection. No high-return sorting/cherry-picking.
    for start in range(0, len(candidates)):
        remaining = candidates[~candidates["seed"].astype(int).isin(used)]
        if len(remaining) < int(args.n_envs):
            break
        group = remaining.head(int(args.n_envs)).copy()
        stats = _group_stats(group)
        if (
            stats["worst_abs_share_all_proxy"] is not None
            and float(stats["worst_abs_share_all_proxy"]) <= float(args.max_worst_abs_share)
            and stats["worst_abs_share_total_with_ctrl_proxy"] is not None
            and float(stats["worst_abs_share_total_with_ctrl_proxy"])
            <= float(args.max_worst_total_with_ctrl_abs_share)
            and float(stats["worst_ctrl_abs_share_total_proxy"]) <= float(args.max_ctrl_abs_share)
        ):
            rule_name = f"gpu0_pretrain_gate_group{group_idx:02d}"
            path = out_dir / f"{rule_name}.csv"
            _write_group_csv(group, path, rule_name)
            group_csvs.append(str(path))
            selected_groups.append({"rule": rule_name, "csv": str(path), **stats})
            used.update(int(v) for v in group["seed"].tolist())
            group_idx += 1
        else:
            # Drop only the current worst contributor and retry; this makes the group-level
            # concentration gate explicit without changing env-level gate semantics.
            vals = pd.to_numeric(group["env_sum_mean"], errors="coerce").abs()
            total_vals = pd.to_numeric(group["total_sum_with_ctrl_proxy"], errors="coerce").abs()
            ctrl_vals = pd.to_numeric(group["ctrl_abs_share_total_proxy"], errors="coerce")
            combined = vals.fillna(0.0) + total_vals.fillna(0.0) + 10.0 * ctrl_vals.fillna(0.0)
            drop_seed = int(group.loc[combined.idxmax(), "seed"])
            used.add(drop_seed)
        if group_idx >= 4:
            break

    all_env_csv = out_dir / "env_gate_table.csv"
    rejected_csv = out_dir / "env_rejected.csv"
    passed_csv = out_dir / "env_passed.csv"
    df.to_csv(all_env_csv, index=False)
    df[df["gate_env_reject"]].to_csv(rejected_csv, index=False)
    df[~df["gate_env_reject"]].to_csv(passed_csv, index=False)

    report = {
        "contract": (
            "GPU0-only generation/pretrain gate selection. Env-level no-tail/high-volatility "
            "rules are hard filters; diagnostic structural columns are report-only. Group-level "
            "selection rejects batches whose canonical reward mass is dominated by one env. "
            "No generator default semantics are changed."
        ),
        "input_population_features": str(Path(args.population_features).resolve()),
        "input_enriched_population": str(Path(args.enriched_population).resolve()),
        "output_dir": str(out_dir),
        "thresholds": {
            "bad_q90_high": float(args.bad_q90_high),
            "positive_frac_low": float(args.positive_frac_low),
            "strict_positive_frac_low": float(args.strict_positive_frac_low),
            "reward_std_high_tertile": reward_std_hi,
            "return_std_high_tertile": return_std_hi,
            "top_tail_gap_high_tertile": top_tail_gap_hi,
            "max_worst_abs_share_all_proxy": float(args.max_worst_abs_share),
            "max_worst_total_with_ctrl_abs_share": float(args.max_worst_total_with_ctrl_abs_share),
            "max_ctrl_abs_share": float(args.max_ctrl_abs_share),
            "max_reward_state_to_action_ratio": (
                None
                if args.max_reward_state_to_action_ratio is None
                else float(args.max_reward_state_to_action_ratio)
            ),
            "min_reward_action_gain_fraction": (
                None
                if args.min_reward_action_gain_fraction is None
                else float(args.min_reward_action_gain_fraction)
            ),
            "ctrl_proxy_n_steps": int(args.n_steps),
        },
        "counts": {
            "population": int(len(df)),
            "env_rejected": int(df["gate_env_reject"].sum()),
            "env_passed": int((~df["gate_env_reject"]).sum()),
            "structural_no_tail": int(df["gate_structural_no_tail"].sum()),
            "low_tail": int(df["gate_low_tail"].sum()),
            "high_volatility_2of3": int(df["gate_high_volatility_2of3"].sum()),
            "ctrl_dominance_proxy": int(df["gate_ctrl_dominance_proxy"].sum()),
            "topology_action_coupling_weak": int(df["gate_topology_action_coupling_weak"].sum()),
            "diagnostic_amp_noise_obs": int(df["diagnostic_amp_noise_obs"].sum()),
            "diagnostic_obs_geometry_dim": int(df["diagnostic_obs_geometry_dim"].sum()),
            "groups_selected": int(len(selected_groups)),
        },
        "files": {
            "env_gate_table_csv": str(all_env_csv),
            "env_rejected_csv": str(rejected_csv),
            "env_passed_csv": str(passed_csv),
            "group_csvs": group_csvs,
        },
        "selected_groups": selected_groups,
    }
    (out_dir / "selection_report.json").write_text(
        json.dumps(_json_safe(report), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(_json_safe(report), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
