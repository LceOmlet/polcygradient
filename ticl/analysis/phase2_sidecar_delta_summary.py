import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _float_or_nan(value) -> float:
    if value is None:
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def summarize_sidecars(output_dir: str | Path, *, early_updates: int = 3, late_updates: int = 3) -> dict:
    outdir = Path(output_dir).expanduser().resolve()
    sidecar_dir = outdir / "semantic_sidecars"
    rows = []
    for path in sorted(sidecar_dir.glob("prior_update_*_envs.jsonl")):
        update_idx = int(path.stem.split("_")[2])
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                rows.append(
                    {
                        "update_idx": update_idx,
                        "env_idx": int(row["env_idx"]),
                        "fixed_env_seed": int(row.get("env_group_env_seed", -1)),
                        "reward_env_sum": _float_or_nan(row.get("reward_env_sum")),
                        "reward_env_q90": _float_or_nan(row.get("reward_env_q90")),
                        "reward_env_positive_fraction": _float_or_nan(row.get("reward_env_positive_fraction")),
                        "done_count": _float_or_nan(row.get("done_count")),
                        "first_done_step": _float_or_nan(row.get("first_done_step")),
                        "action_norm_mean": _float_or_nan(row.get("action_norm_mean")),
                        "obs_per_dim_rms_mean": _float_or_nan(row.get("obs_per_dim_rms_mean")),
                        "next_state_step_l2_delta_to_rms_ratio": _float_or_nan(
                            row.get("next_state_step_l2_delta_to_rms_ratio")
                        ),
                    }
                )
    if not rows:
        raise FileNotFoundError(f"No prior semantic sidecar rows found under {sidecar_dir}")
    df = pd.DataFrame(rows)
    max_update = int(df["update_idx"].max())
    early_cut = min(int(early_updates), max_update + 1)
    late_start = max(0, max_update - int(late_updates) + 1)
    early = (
        df[df["update_idx"] < early_cut]
        .groupby(["env_idx", "fixed_env_seed"])
        .mean(numeric_only=True)
        .add_prefix("early_")
    )
    late = (
        df[df["update_idx"] >= late_start]
        .groupby(["env_idx", "fixed_env_seed"])
        .mean(numeric_only=True)
        .add_prefix("late_")
    )
    merged = early.join(late, how="inner").reset_index()
    for key in [
        "reward_env_sum",
        "reward_env_q90",
        "reward_env_positive_fraction",
        "done_count",
        "first_done_step",
        "action_norm_mean",
        "obs_per_dim_rms_mean",
        "next_state_step_l2_delta_to_rms_ratio",
    ]:
        merged[f"delta_{key}"] = merged[f"late_{key}"] - merged[f"early_{key}"]

    summary = {
        "output_dir": str(outdir),
        "updates": int(max_update + 1),
        "n_envs": int(merged["env_idx"].nunique()),
        "early_updates": int(early_cut),
        "late_updates": int(max_update - late_start + 1),
        "mean_delta_reward_env_sum": float(merged["delta_reward_env_sum"].mean()),
        "median_delta_reward_env_sum": float(merged["delta_reward_env_sum"].median()),
        "worst_delta_reward_env_sum": float(merged["delta_reward_env_sum"].min()),
        "negative_delta_count": int((merged["delta_reward_env_sum"] < 0).sum()),
        "delta_lt_minus5_count": int((merged["delta_reward_env_sum"] < -5).sum()),
        "delta_lt_minus10_count": int((merged["delta_reward_env_sum"] < -10).sum()),
        "worst_envs": merged.sort_values("delta_reward_env_sum")
        .head(10)[
            [
                "env_idx",
                "fixed_env_seed",
                "delta_reward_env_sum",
                "early_reward_env_sum",
                "late_reward_env_sum",
                "delta_reward_env_q90",
                "delta_reward_env_positive_fraction",
                "delta_done_count",
                "delta_first_done_step",
                "delta_action_norm_mean",
            ]
        ]
        .replace({np.nan: None})
        .to_dict("records"),
    }
    merged.sort_values("delta_reward_env_sum").to_csv(outdir / "per_env_delta_summary.csv", index=False)
    (outdir / "delta_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize fixed-group per-env reward deltas from semantic sidecars.")
    parser.add_argument("output_dir", type=str)
    parser.add_argument("--early-updates", type=int, default=3)
    parser.add_argument("--late-updates", type=int, default=3)
    args = parser.parse_args()
    print(
        json.dumps(
            summarize_sidecars(args.output_dir, early_updates=args.early_updates, late_updates=args.late_updates),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
