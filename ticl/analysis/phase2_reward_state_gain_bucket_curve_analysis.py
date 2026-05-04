import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


HORIZONS = (2, 3, 4, 5)


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return str(value)


def _safe_quantile(values: np.ndarray, q: float) -> float:
    if values.size == 0:
        return float("nan")
    return float(np.quantile(values.astype(np.float64, copy=False), float(q)))


def _vector_stats(values: np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return {
            "count": 0.0,
            "mean": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "q10": float("nan"),
            "q25": float("nan"),
            "q50": float("nan"),
            "q75": float("nan"),
            "q90": float("nan"),
            "max": float("nan"),
        }
    return {
        "count": float(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "q10": _safe_quantile(arr, 0.10),
        "q25": _safe_quantile(arr, 0.25),
        "q50": _safe_quantile(arr, 0.50),
        "q75": _safe_quantile(arr, 0.75),
        "q90": _safe_quantile(arr, 0.90),
        "max": float(np.max(arr)),
    }


def _bucket_labels(edges: np.ndarray) -> list[str]:
    labels = []
    for idx in range(int(edges.size) - 1):
        labels.append(f"[{edges[idx]:.3f}, {edges[idx + 1]:.3f}]")
    return labels


def _assign_quantile_buckets(values: np.ndarray, bucket_count: int) -> tuple[np.ndarray, np.ndarray]:
    quantiles = np.linspace(0.0, 1.0, int(bucket_count) + 1)
    edges = np.quantile(values.astype(np.float64, copy=False), quantiles)
    edges = np.maximum.accumulate(edges)
    if np.allclose(edges, edges[0]):
        return np.zeros_like(values, dtype=np.int64), np.asarray([edges[0], edges[0]], dtype=np.float64)
    bucket_ids = np.searchsorted(edges[1:-1], values, side="right").astype(np.int64)
    return bucket_ids, edges


def _curve_from_row(row: dict[str, Any], prefix: str) -> np.ndarray:
    return np.asarray([float(row[f"horizon_{k}_{prefix}"]) for k in HORIZONS], dtype=np.float64)


def _normalize_curve(curve: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    base = float(curve[0]) if curve.size > 0 else float("nan")
    if not np.isfinite(base) or abs(base) <= float(eps):
        return np.full_like(curve, np.nan, dtype=np.float64)
    return curve / base


def _distance(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    if int(mask.sum()) <= 0:
        return float("nan")
    diff = a[mask] - b[mask]
    return float(np.sqrt(np.mean(np.square(diff))))


def _nanmean_or_nan(values: np.ndarray, axis: int = 0) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size <= 0:
        return np.asarray([], dtype=np.float64)
    valid = np.any(np.isfinite(arr), axis=axis)
    with np.errstate(invalid="ignore"):
        mean = np.nanmean(arr, axis=axis)
    if np.ndim(mean) == 0:
        return np.asarray(float(mean) if bool(valid) else float("nan"), dtype=np.float64)
    mean = np.asarray(mean, dtype=np.float64)
    mean[~np.asarray(valid, dtype=bool)] = np.nan
    return mean


def _nanmedian_or_nan(values: np.ndarray, axis: int = 0) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size <= 0:
        return np.asarray([], dtype=np.float64)
    valid = np.any(np.isfinite(arr), axis=axis)
    with np.errstate(invalid="ignore"):
        median = np.nanmedian(arr, axis=axis)
    if np.ndim(median) == 0:
        return np.asarray(float(median) if bool(valid) else float("nan"), dtype=np.float64)
    median = np.asarray(median, dtype=np.float64)
    median[~np.asarray(valid, dtype=bool)] = np.nan
    return median


def _load_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _extract_ant_curve(report_path: Path) -> dict[str, Any]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    diag = {int(item["horizon_step"]): item for item in report["horizon_diagnostics"]}
    curve = np.asarray([float(diag[k]["bias_corrected_identity_score"]) for k in HORIZONS], dtype=np.float64)
    return {
        "env_id": str(report["env_id"]),
        "report_json": str(report_path),
        "corrected_identity_curve": curve,
        "normalized_curve": _normalize_curve(curve),
    }


def _bucket_summary(
    rows: list[dict[str, Any]],
    bucket_ids: np.ndarray,
    edges: np.ndarray,
    ant_refs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    labels = _bucket_labels(edges)
    results = []
    max_bucket = int(bucket_ids.max()) if bucket_ids.size > 0 else -1
    for bucket_id in range(max_bucket + 1):
        subset = [row for row, bid in zip(rows, bucket_ids) if int(bid) == int(bucket_id)]
        gains = np.asarray([float(row["topology_reward_state_input_gain_fraction"]) for row in subset], dtype=np.float64)
        curves = np.stack(
            [_curve_from_row(row, "bias_corrected_identity_score") for row in subset],
            axis=0,
        )
        norm_curves = np.stack([_normalize_curve(curve) for curve in curves], axis=0)
        mean_curve = np.mean(curves, axis=0)
        median_curve = np.median(curves, axis=0)
        mean_norm_curve = _nanmean_or_nan(norm_curves, axis=0)
        median_norm_curve = _nanmedian_or_nan(norm_curves, axis=0)
        normalized_valid_row_count = int(np.sum(np.all(np.isfinite(norm_curves), axis=1)))
        by_activation: dict[str, int] = {}
        for row in subset:
            by_activation[str(row["activation"])] = by_activation.get(str(row["activation"]), 0) + 1
        ant_distances = []
        for ant in ant_refs:
            ant_distances.append(
                {
                    "env_id": str(ant["env_id"]),
                    "normalized_curve_rmse": _distance(mean_norm_curve, ant["normalized_curve"]),
                    "raw_curve_rmse": _distance(mean_curve, ant["corrected_identity_curve"]),
                }
            )
        results.append(
            {
                "bucket_id": int(bucket_id),
                "bucket_label": str(labels[int(bucket_id)]) if int(bucket_id) < len(labels) else f"bucket_{bucket_id}",
                "row_count": int(len(subset)),
                "normalized_curve_valid_row_count": normalized_valid_row_count,
                "reward_state_input_gain_fraction_stats": _vector_stats(gains),
                "mean_corrected_identity_curve": {
                    f"G{k}": float(mean_curve[idx]) for idx, k in enumerate(HORIZONS)
                },
                "median_corrected_identity_curve": {
                    f"G{k}": float(median_curve[idx]) for idx, k in enumerate(HORIZONS)
                },
                "mean_normalized_curve": {
                    f"G{k}/G2": float(mean_norm_curve[idx]) for idx, k in enumerate(HORIZONS)
                },
                "median_normalized_curve": {
                    f"G{k}/G2": float(median_norm_curve[idx]) for idx, k in enumerate(HORIZONS)
                },
                "activation_counts": by_activation,
                "ant_curve_distances": ant_distances,
            }
        )
    return results


def _write_bucket_curve_csv(path: Path, buckets: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "bucket_id",
        "bucket_label",
        "row_count",
        "gain_q25",
        "gain_q50",
        "gain_q75",
        "G2_mean",
        "G3_mean",
        "G4_mean",
        "G5_mean",
        "G2_ratio",
        "G3_ratio",
        "G4_ratio",
        "G5_ratio",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for bucket in buckets:
            gain_stats = bucket["reward_state_input_gain_fraction_stats"]
            writer.writerow(
                {
                    "bucket_id": bucket["bucket_id"],
                    "bucket_label": bucket["bucket_label"],
                    "row_count": bucket["row_count"],
                    "gain_q25": gain_stats["q25"],
                    "gain_q50": gain_stats["q50"],
                    "gain_q75": gain_stats["q75"],
                    "G2_mean": bucket["mean_corrected_identity_curve"]["G2"],
                    "G3_mean": bucket["mean_corrected_identity_curve"]["G3"],
                    "G4_mean": bucket["mean_corrected_identity_curve"]["G4"],
                    "G5_mean": bucket["mean_corrected_identity_curve"]["G5"],
                    "G2_ratio": bucket["mean_normalized_curve"]["G2/G2"],
                    "G3_ratio": bucket["mean_normalized_curve"]["G3/G2"],
                    "G4_ratio": bucket["mean_normalized_curve"]["G4/G2"],
                    "G5_ratio": bucket["mean_normalized_curve"]["G5/G2"],
                }
            )


def _write_plot(path: Path, buckets: list[dict[str, Any]], ant_refs: list[dict[str, Any]]) -> str | None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return None

    horizons = list(HORIZONS)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for bucket in buckets:
        label = f"B{int(bucket['bucket_id']) + 1} {bucket['bucket_label']}"
        raw_curve = [bucket["mean_corrected_identity_curve"][f"G{k}"] for k in horizons]
        norm_curve = [bucket["mean_normalized_curve"][f"G{k}/G2"] for k in horizons]
        axes[0].plot(horizons, raw_curve, marker="o", label=label)
        axes[1].plot(horizons, norm_curve, marker="o", label=label)
    for ant in ant_refs:
        label = str(ant["env_id"])
        axes[0].plot(horizons, ant["corrected_identity_curve"], linestyle="--", linewidth=2.0, label=label)
        axes[1].plot(horizons, ant["normalized_curve"], linestyle="--", linewidth=2.0, label=label)
    axes[0].set_title("Corrected Identity")
    axes[1].set_title("Normalized Decay Relative to G2")
    for ax in axes:
        ax.set_xlabel("Horizon k")
        ax.set_xticks(horizons)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("bias-corrected identity")
    axes[1].set_ylabel("ratio")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return str(path)


def run_analysis(args: argparse.Namespace) -> dict[str, Any]:
    rows_path = Path(args.rows_with_topology_jsonl).expanduser().resolve()
    rows = _load_rows(rows_path)
    gain_values = np.asarray([float(row["topology_reward_state_input_gain_fraction"]) for row in rows], dtype=np.float64)
    bucket_ids, edges = _assign_quantile_buckets(gain_values, int(args.bucket_count))
    ant_refs = [
        _extract_ant_curve(Path(path).expanduser().resolve()) for path in args.ant_report_jsons
    ]
    buckets = _bucket_summary(rows, bucket_ids, edges, ant_refs)

    closest_by_ant = []
    for ant in ant_refs:
        ranked = [
            bucket
            for bucket in sorted(
                buckets,
                key=lambda bucket: float(
                    next(
                        item["normalized_curve_rmse"]
                        for item in bucket["ant_curve_distances"]
                        if str(item["env_id"]) == str(ant["env_id"])
                    )
                ),
            )
            if np.isfinite(
                next(
                    item["normalized_curve_rmse"]
                    for item in bucket["ant_curve_distances"]
                    if str(item["env_id"]) == str(ant["env_id"])
                )
            )
        ]
        if not ranked:
            continue
        closest_by_ant.append(
            {
                "env_id": str(ant["env_id"]),
                "closest_bucket_by_normalized_curve_rmse": ranked[0]["bucket_id"],
                "closest_bucket_label": ranked[0]["bucket_label"],
                "high_bucket_id": max(bucket["bucket_id"] for bucket in buckets),
                "high_bucket_label": buckets[-1]["bucket_label"],
                "high_bucket_rmse": next(
                    item["normalized_curve_rmse"]
                    for item in buckets[-1]["ant_curve_distances"]
                    if str(item["env_id"]) == str(ant["env_id"])
                ),
                "closest_bucket_rmse": next(
                    item["normalized_curve_rmse"]
                    for item in ranked[0]["ant_curve_distances"]
                    if str(item["env_id"]) == str(ant["env_id"])
                ),
            }
        )

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "bucket_curves.csv"
    _write_bucket_curve_csv(csv_path, buckets)
    plot_path = _write_plot(output_dir / "bucket_curves.png", buckets, ant_refs)

    summary = {
        "audit_entry": "phase2_reward_state_gain_bucket_curve_analysis",
        "rows_with_topology_jsonl": str(rows_path),
        "row_count": int(len(rows)),
        "bucket_feature": "topology_reward_state_input_gain_fraction",
        "bucket_count": int(args.bucket_count),
        "bucket_edges": [float(v) for v in np.asarray(edges, dtype=np.float64)],
        "horizons": list(HORIZONS),
        "bucket_summaries": buckets,
        "ant_references": ant_refs,
        "closest_bucket_by_ant_normalized_decay_rmse": closest_by_ant,
        "bucket_curves_csv": str(csv_path),
        "bucket_curves_plot": plot_path,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(_json_safe(summary), sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(f"SUMMARY_WRITTEN {summary_path}", flush=True)
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Bucket exact-SCM environments by reward_state_input_gain_fraction and compare G2-G5 curves to Ant."
    )
    parser.add_argument("--rows-with-topology-jsonl", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--bucket-count", type=int, default=5)
    parser.add_argument(
        "--ant-report-jsons",
        type=str,
        nargs="+",
        required=True,
    )
    return parser


def main() -> None:
    run_analysis(build_arg_parser().parse_args())


if __name__ == "__main__":
    main()
