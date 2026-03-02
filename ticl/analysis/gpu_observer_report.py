#!/usr/bin/env python3
import argparse
import json
import os
import statistics
import time
from collections import defaultdict


def _read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _num_stats(vals):
    data = [float(v) for v in vals if v is not None]
    if not data:
        return {"avg": None, "p95": None, "max": None, "min": None}
    data_sorted = sorted(data)
    p95 = data_sorted[int(round(0.95 * (len(data_sorted) - 1)))]
    return {
        "avg": float(statistics.mean(data)),
        "p95": float(p95),
        "max": float(max(data)),
        "min": float(min(data)),
    }


def summarize_samples(rows):
    if not rows:
        return {"samples": 0}
    util = [r.get("gpu_util_percent") for r in rows]
    mem = [r.get("process_mem_mib") for r in rows]
    mem_share = [r.get("process_mem_share_percent") for r in rows]
    process_sm_util = [r.get("process_sm_util_percent") for r in rows]
    process_mem_util = [r.get("process_mem_util_percent") for r in rows]
    process_util_available = [bool(r.get("process_util_available", False)) for r in rows]
    power = [r.get("gpu_power_w") for r in rows]
    t0 = float(rows[0].get("timestamp_unix", 0.0))
    t1 = float(rows[-1].get("timestamp_unix", t0))
    return {
        "samples": int(len(rows)),
        "duration_sec": float(max(0.0, t1 - t0)),
        "gpu_util": _num_stats(util),
        "process_mem_mib": _num_stats(mem),
        "process_mem_share_percent": _num_stats(mem_share),
        "process_sm_util_percent": _num_stats(process_sm_util),
        "process_mem_util_percent": _num_stats(process_mem_util),
        "process_util_available_ratio": float(
            (sum(1 for x in process_util_available if x) / len(process_util_available))
            if process_util_available else 0.0
        ),
        "gpu_power_w": _num_stats(power),
    }


def summarize_stages(rows):
    if not rows:
        return {"stage_rows": 0, "stages": {}, "batch_total_duration_sec": None}

    by_stage = defaultdict(list)
    by_batch = defaultdict(float)
    for r in rows:
        stage = str(r.get("stage", "unknown"))
        dur = r.get("duration_sec", None)
        by_stage[stage].append(r)
        if dur is not None:
            key = (int(r.get("epoch", -1)), int(r.get("batch", -1)))
            by_batch[key] += float(dur)

    stage_stats = {}
    for stage, recs in by_stage.items():
        stage_stats[stage] = {
            "count": int(len(recs)),
            "duration_sec": _num_stats([x.get("duration_sec") for x in recs]),
            "cuda_elapsed_ms": _num_stats([x.get("cuda_elapsed_ms") for x in recs]),
            "cuda_busy_ratio": _num_stats([x.get("cuda_busy_ratio") for x in recs]),
            "gpu_util_avg": _num_stats([x.get("gpu_util_avg") for x in recs]),
            "process_mem_avg_mib": _num_stats([x.get("process_mem_avg_mib") for x in recs]),
            "process_sm_util_avg": _num_stats([x.get("process_sm_util_avg") for x in recs]),
            "process_mem_util_avg": _num_stats([x.get("process_mem_util_avg") for x in recs]),
            "process_util_available_ratio": float(
                sum(1 for x in recs if bool(x.get("process_util_available", False))) / len(recs)
            ),
        }

    batch_total = list(by_batch.values())
    batch_total_stats = _num_stats(batch_total)
    return {
        "stage_rows": int(len(rows)),
        "stages": stage_stats,
        "batch_total_duration_sec": batch_total_stats,
    }


def append_skyline_record(path, record):
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=True))
        f.write("\n")


def main():
    parser = argparse.ArgumentParser(description="Summarize GPU observer JSONL and maintain skyline records.")
    parser.add_argument("--samples-jsonl", required=True, type=str, help="Path to GPU observer sample JSONL.")
    parser.add_argument("--stages-jsonl", type=str, default=None, help="Path to stage observer JSONL (optional).")
    parser.add_argument("--workload-id", type=str, default="default", help="Skyline workload key.")
    parser.add_argument("--run-tag", type=str, default=None, help="Optional run tag.")
    parser.add_argument("--skyline-jsonl", type=str, default=None, help="If set, append summarized record here.")
    parser.add_argument("--summary-json", type=str, default=None, help="If set, write summary JSON here.")
    args = parser.parse_args()

    samples = _read_jsonl(args.samples_jsonl)
    stages = _read_jsonl(args.stages_jsonl) if args.stages_jsonl and os.path.exists(args.stages_jsonl) else []

    summary = {
        "timestamp_unix": float(time.time()),
        "workload_id": str(args.workload_id),
        "run_tag": args.run_tag,
        "samples_summary": summarize_samples(samples),
        "stages_summary": summarize_stages(stages),
        "source_files": {
            "samples_jsonl": args.samples_jsonl,
            "stages_jsonl": args.stages_jsonl,
        },
    }

    print(json.dumps(summary, ensure_ascii=True, indent=2))

    if args.summary_json:
        out_dir = os.path.dirname(args.summary_json)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.summary_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=True, indent=2)

    if args.skyline_jsonl:
        append_skyline_record(args.skyline_jsonl, summary)
        print(f"Appended skyline record -> {args.skyline_jsonl}")


if __name__ == "__main__":
    main()
