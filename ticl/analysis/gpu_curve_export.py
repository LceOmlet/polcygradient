#!/usr/bin/env python3
import argparse
import csv
import json
import os


def _read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _stage_label(ts, stage_rows):
    # Single-step use-case: stage rows are usually tiny, linear scan is enough.
    labels = []
    for r in stage_rows:
        t0 = float(r.get("t_start_unix", 0.0))
        t1 = float(r.get("t_end_unix", 0.0))
        if t0 <= ts <= t1:
            labels.append(str(r.get("stage", "unknown")))
    if not labels:
        return ""
    return "|".join(labels)


def main():
    parser = argparse.ArgumentParser(description="Export observer samples as curve-friendly CSV.")
    parser.add_argument("--samples-jsonl", required=True, type=str)
    parser.add_argument("--stages-jsonl", type=str, default=None)
    parser.add_argument("--output-csv", required=True, type=str)
    args = parser.parse_args()

    samples = _read_jsonl(args.samples_jsonl)
    stage_rows = []
    if args.stages_jsonl and os.path.exists(args.stages_jsonl):
        stage_rows = _read_jsonl(args.stages_jsonl)

    out_dir = os.path.dirname(args.output_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "timestamp_unix",
                "gpu_util_percent",
                "gpu_mem_util_percent",
                "gpu_power_w",
                "process_mem_mib",
                "process_mem_share_percent",
                "stage_label",
            ]
        )
        for s in samples:
            ts = float(s.get("timestamp_unix", 0.0))
            writer.writerow(
                [
                    ts,
                    s.get("gpu_util_percent"),
                    s.get("gpu_mem_util_percent"),
                    s.get("gpu_power_w"),
                    s.get("process_mem_mib"),
                    s.get("process_mem_share_percent"),
                    _stage_label(ts, stage_rows),
                ]
            )

    print(f"Exported curve CSV -> {args.output_csv}")


if __name__ == "__main__":
    main()
