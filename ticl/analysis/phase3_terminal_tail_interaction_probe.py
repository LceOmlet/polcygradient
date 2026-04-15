import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _corr(xs: list[float], ys: list[float]) -> float:
    if len(xs) != len(ys):
        raise ValueError("Correlation inputs must have matching lengths.")
    if len(xs) <= 1:
        return 0.0
    x = np.asarray(xs, dtype=np.float32)
    y = np.asarray(ys, dtype=np.float32)
    finite = np.isfinite(x) & np.isfinite(y)
    if int(finite.sum()) <= 1:
        return 0.0
    x = x[finite]
    y = y[finite]
    x = x - float(x.mean())
    y = y - float(y.mean())
    denom = float(np.linalg.norm(x) * np.linalg.norm(y))
    if denom <= 0.0:
        return 0.0
    return float(np.dot(x, y) / denom)


def _topk_mass_share(items: list[dict[str, Any]], *, mass_key: str, rank_key: str, k: int) -> dict[str, Any]:
    ranked = sorted(items, key=lambda item: float(item[rank_key]), reverse=True)
    top = ranked[: max(1, min(int(k), len(ranked)))]
    total_mass = float(sum(float(item[mass_key]) for item in items))
    top_mass = float(sum(float(item[mass_key]) for item in top))
    return {
        "k": int(k),
        "rank_key": str(rank_key),
        "mass_key": str(mass_key),
        "env_indices": [int(item["env_index"]) for item in top],
        "mean_rank_value": float(sum(float(item[rank_key]) for item in top) / max(1, len(top))),
        "mass_share": float(top_mass / total_mass) if total_mass > 0.0 else 0.0,
    }


def _summary(items: list[dict[str, Any]], mass_key: str) -> dict[str, Any]:
    masses = [float(item[mass_key]) for item in items]
    terminal_resets = [float(item["objective_terminal_reset_count"]) for item in items]
    pre_gap = [float(item["pre_vs_zero_suffix_gap"]) for item in items]
    return {
        "corr_vs_terminal_resets": _corr(masses, terminal_resets),
        "corr_vs_pre_suffix_gap": _corr(masses, pre_gap),
        "mass_total": float(sum(masses)),
        "top2_pre_gap_capture": _topk_mass_share(items, mass_key=mass_key, rank_key="pre_vs_zero_suffix_gap", k=2),
        "top4_pre_gap_capture": _topk_mass_share(items, mass_key=mass_key, rank_key="pre_vs_zero_suffix_gap", k=4),
    }


def _bucket_row(row: dict[str, Any], tail_len_key: str) -> dict[str, Any]:
    tail_rows = row["tail_len_rows"][tail_len_key]
    base_mass = float(row["base_positive_mass"])
    first_tail = float(tail_rows["first_terminal_tail"]["removed_positive_mass"])
    post_tail = float(tail_rows["post_reset_terminal_tail"]["removed_positive_mass"])
    first_only = float(tail_rows["first_episode_only"]["removed_positive_mass"])
    post_only = float(tail_rows["post_reset_only"]["removed_positive_mass"])
    first_non_tail = max(0.0, first_only - first_tail)
    post_non_tail = max(0.0, post_only - post_tail)
    terminal_tail = float(tail_rows["terminal_tail"]["removed_positive_mass"])
    return {
        "env_index": int(row["env_index"]),
        "objective_terminal_reset_count": int(row["objective_terminal_reset_count"]),
        "pre_vs_zero_suffix_gap": float(row["pre_vs_zero_suffix_gap"]),
        "base_positive_mass": base_mass,
        "first_episode_tail_mass": first_tail,
        "first_episode_non_tail_mass": first_non_tail,
        "post_reset_tail_mass": post_tail,
        "post_reset_non_tail_mass": post_non_tail,
        "terminal_tail_mass": terminal_tail,
    }


def _dominant_bucket(summaries: dict[str, dict[str, Any]]) -> dict[str, Any]:
    ranked = sorted(
        summaries.items(),
        key=lambda item: (float(item[1]["corr_vs_terminal_resets"]), -float(item[1]["corr_vs_pre_suffix_gap"])),
        reverse=True,
    )
    name, summary = ranked[0]
    return {
        "bucket": str(name),
        "corr_vs_terminal_resets": float(summary["corr_vs_terminal_resets"]),
        "corr_vs_pre_suffix_gap": float(summary["corr_vs_pre_suffix_gap"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Phase 3 tail interaction probe (reuse terminal-tail mask artifact, no rerun)."
    )
    parser.add_argument(
        "--tail-mask-probe-json",
        type=str,
        default="/home/chen/RLPFN/artifacts/phase3_terminal_tail_mask_probe.json",
    )
    parser.add_argument("--tail-len", type=int, default=8)
    parser.add_argument("--output-json", type=str, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output_path = Path(args.output_json).expanduser().resolve()
    if output_path.exists() and not bool(args.overwrite):
        print(output_path.read_text())
        return 0

    probe_path = Path(args.tail_mask_probe_json).expanduser().resolve()
    payload = json.loads(probe_path.read_text())
    tail_len_key = str(int(args.tail_len))
    if tail_len_key not in payload["summaries"]:
        raise KeyError(f"tail_len {args.tail_len} not found in {probe_path}")

    env_rows = [_bucket_row(row, tail_len_key) for row in payload["env_rows"]]
    bucket_keys = [
        "first_episode_tail_mass",
        "first_episode_non_tail_mass",
        "post_reset_tail_mass",
        "post_reset_non_tail_mass",
        "terminal_tail_mass",
    ]
    summaries = {key: _summary(env_rows, key) for key in bucket_keys}
    dominance = _dominant_bucket(summaries)

    base_total = float(sum(float(row["base_positive_mass"]) for row in env_rows))
    share = {
        key: float(summaries[key]["mass_total"] / base_total) if base_total > 0.0 else 0.0 for key in bucket_keys
    }

    result = {
        "audit_entry": "phase3_terminal_tail_interaction_probe",
        "config": {
            "tail_mask_probe_json": str(probe_path),
            "tail_len": int(args.tail_len),
        },
        "bucket_summaries": summaries,
        "bucket_share_of_base": share,
        "dominant_bias_bucket": dominance,
        "conclusions": {
            "dominant_bucket": dominance["bucket"],
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(output_path.read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
