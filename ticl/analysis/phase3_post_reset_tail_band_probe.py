import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


TAIL_LENS = (1, 2, 4, 8)


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


def _tail_removed(row: dict[str, Any], tail_len: int) -> float:
    return float(
        row["tail_len_rows"][str(int(tail_len))]["post_reset_terminal_tail"]["removed_positive_mass"]
    )


def _band_mass(row: dict[str, Any], lower: int, upper: int) -> float:
    upper_mass = _tail_removed(row, upper)
    lower_mass = _tail_removed(row, lower)
    return max(0.0, upper_mass - lower_mass)


def _dominant_band(summaries: dict[str, dict[str, Any]]) -> dict[str, Any]:
    ranked = sorted(
        summaries.items(),
        key=lambda item: (float(item[1]["corr_vs_terminal_resets"]), -float(item[1]["corr_vs_pre_suffix_gap"])),
        reverse=True,
    )
    name, summary = ranked[0]
    return {
        "band": str(name),
        "corr_vs_terminal_resets": float(summary["corr_vs_terminal_resets"]),
        "corr_vs_pre_suffix_gap": float(summary["corr_vs_pre_suffix_gap"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Phase 3 post-reset terminal-tail band probe (no rerun)."
    )
    parser.add_argument(
        "--tail-mask-probe-json",
        type=str,
        default="/home/chen/RLPFN/artifacts/phase3_terminal_tail_mask_probe.json",
    )
    parser.add_argument("--output-json", type=str, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output_path = Path(args.output_json).expanduser().resolve()
    if output_path.exists() and not bool(args.overwrite):
        print(output_path.read_text())
        return 0

    probe_path = Path(args.tail_mask_probe_json).expanduser().resolve()
    payload = json.loads(probe_path.read_text())
    for tail_len in TAIL_LENS:
        if str(int(tail_len)) not in payload["summaries"]:
            raise KeyError(f"tail_len {tail_len} not found in {probe_path}")

    env_items = []
    for row in payload["env_rows"]:
        band_1 = _tail_removed(row, 1)
        band_2 = _band_mass(row, 1, 2)
        band_4 = _band_mass(row, 2, 4)
        band_8 = _band_mass(row, 4, 8)
        env_items.append(
            {
                "objective_terminal_reset_count": row["objective_terminal_reset_count"],
                "pre_vs_zero_suffix_gap": row["pre_vs_zero_suffix_gap"],
                "post_reset_tail_band_1": band_1,
                "post_reset_tail_band_2": band_2,
                "post_reset_tail_band_4": band_4,
                "post_reset_tail_band_8": band_8,
            }
        )

    summaries = {
        "tail_band_1": _summary(env_items, mass_key="post_reset_tail_band_1"),
        "tail_band_2": _summary(env_items, mass_key="post_reset_tail_band_2"),
        "tail_band_4": _summary(env_items, mass_key="post_reset_tail_band_4"),
        "tail_band_8": _summary(env_items, mass_key="post_reset_tail_band_8"),
    }
    dominant = _dominant_band(summaries)

    result = {
        "audit_entry": "phase3_post_reset_tail_band_probe",
        "config": {"tail_mask_probe_json": str(probe_path), "tail_lens": list(TAIL_LENS)},
        "band_summaries": summaries,
        "dominant_band": dominant,
        "conclusions": {"dominant_band": dominant["band"]},
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(output_path.read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
