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


def _summary(items: list[dict[str, Any]], *, mass_key: str) -> dict[str, Any]:
    masses = [float(item[mass_key]) for item in items]
    terminal_resets = [float(item["objective_terminal_reset_count"]) for item in items]
    pre_gap = [float(item["pre_vs_zero_suffix_gap"]) for item in items]
    return {
        "corr_vs_terminal_resets": _corr(masses, terminal_resets),
        "corr_vs_pre_suffix_gap": _corr(masses, pre_gap),
        "mass_total": float(sum(masses)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Phase 3 verify whether first-episode-only ablation removes heldout bias (no rerun)."
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

    base_items = []
    for row in payload["env_rows"]:
        base_items.append(
            {
                "objective_terminal_reset_count": row["objective_terminal_reset_count"],
                "pre_vs_zero_suffix_gap": row["pre_vs_zero_suffix_gap"],
                "base_positive_mass": row["base_positive_mass"],
                "first_episode_remaining_mass": row["tail_len_rows"][tail_len_key]["first_episode_only"][
                    "remaining_positive_mass"
                ],
            }
        )

    base_summary = _summary(base_items, mass_key="base_positive_mass")
    ablated_summary = _summary(base_items, mass_key="first_episode_remaining_mass")

    result = {
        "audit_entry": "phase3_first_episode_ablation_verify",
        "config": {
            "tail_mask_probe_json": str(probe_path),
            "tail_len": int(args.tail_len),
        },
        "base_summary": base_summary,
        "first_episode_ablation_summary": ablated_summary,
        "deltas": {
            "corr_reset_delta": float(ablated_summary["corr_vs_terminal_resets"] - base_summary["corr_vs_terminal_resets"]),
            "corr_pre_delta": float(ablated_summary["corr_vs_pre_suffix_gap"] - base_summary["corr_vs_pre_suffix_gap"]),
        },
        "conclusions": {
            "first_episode_ablation_reduces_reset_bias": bool(
                ablated_summary["corr_vs_terminal_resets"] < base_summary["corr_vs_terminal_resets"]
            ),
            "first_episode_ablation_improves_pre_gap_alignment": bool(
                ablated_summary["corr_vs_pre_suffix_gap"] > base_summary["corr_vs_pre_suffix_gap"]
            ),
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(output_path.read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
