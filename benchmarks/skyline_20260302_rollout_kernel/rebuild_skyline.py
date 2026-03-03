#!/usr/bin/env python3
"""
Rebuild rollout-kernel skyline from saved run directories.

This script is intentionally strict about run validity to avoid mixing
pre-stability and post-stability cohorts.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional


PG_PHASE_RE = re.compile(
    r"\[pg-phase\].*?"
    r"rollout_s=(?P<rollout_s>[-+0-9.eE]+)\s+"
    r"backward_s=(?P<backward_s>[-+0-9.eE]+)\s+"
    r"step_s=(?P<step_s>[-+0-9.eE]+)\s+"
    r"backward_calls=(?P<backward_calls>\d+)\s+"
    r"chunk=(?P<chunk>\S+)\s+"
    r"tbptt=(?P<tbptt>\S+)\s+"
    r"status=(?P<status>[A-Za-z0-9_]+)"
    r"(?P<tail>.*)$"
)
WALLCLOCK_RE = re.compile(r"Wallclock time:\s*([0-9.]+)s")
MEAN_LOSS_RE = re.compile(r"mean loss\s+([A-Za-z0-9+.\-eE]+)")
VALID_SKIPPED_RE = re.compile(r"valid/skipped\s+(\d+)/(\d+)")
TOKENS_PER_SEC_RE = re.compile(r"tokens/s=([0-9.]+)")
HOST_RSS_RE = re.compile(r"Maximum resident set size \(kbytes\):\s*(\d+)")
BATCH_SIZE_RE = re.compile(r"(?:^|\s)--batch-size(?:=|\s+)(\d+)(?:\s|$)")

POLICY_AUTODTYPE_RE = re.compile(r"Policy autocast dtype:\s*([A-Za-z0-9_]+)")
POLICY_FINALIZE2D_RE = re.compile(r"Policy finalize 2D fastpath:\s*(True|False)")
POLICY_KV_MODE_RE = re.compile(r"Policy KV cache mode:\s*([A-Za-z0-9_]+)")
POLICY_KV_PAGE_RE = re.compile(r"Policy KV cache page size:\s*(\d+)")
POLICY_DENSE_CAP_RE = re.compile(r"Policy paged-attn dense page cap:\s*(\d+)")
POLICY_CKPT_RE = re.compile(r"Policy rollout checkpoint:\s*(True|False)")
POLICY_CKPT_REENTRANT_RE = re.compile(r"Policy rollout checkpoint reentrant:\s*(True|False)")
POLICY_GRAD_MUTABLE_RE = re.compile(r"Policy grad-mutable KV cache:\s*(True|False)")
POLICY_COMPILE_RE = re.compile(r"Policy torch.compile:\s*(True|False)")
SEED_SET_RE = re.compile(r"^Seed set to\s+(\d+)\s*$")
COMMAND_TIMED_RE = re.compile(r'Command being timed:\s+"(.+)"')
CMD_TXT_CMD_RE = re.compile(r"^\[benchmark\]\s+cmd=(.+)$")
CMD_TXT_GIT_RE = re.compile(r"^\[benchmark\]\s+git_head=(.+)$")


def _to_float(v: str) -> Optional[float]:
    try:
        out = float(v)
    except Exception:
        return None
    if math.isnan(out) or math.isinf(out):
        return out
    return float(out)


def _to_int(v: str) -> Optional[int]:
    try:
        return int(v)
    except Exception:
        return None


def _to_bool(v: str) -> Optional[bool]:
    vv = str(v).strip().lower()
    if vv in {"true", "1", "yes", "on"}:
        return True
    if vv in {"false", "0", "no", "off"}:
        return False
    return None


def _parse_tail_metrics(tail: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if not tail:
        return out
    for token in tail.strip().split():
        if "=" not in token:
            continue
        k, v = token.split("=", 1)
        if not k:
            continue
        out[k] = v
    return out


def _tail_float(metrics: Dict[str, str], key: str) -> Optional[float]:
    raw = metrics.get(key, None)
    if raw is None:
        return None
    return _to_float(raw)


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


@dataclass
class RunRecord:
    run_id: str
    run_dir: str
    git_head: Optional[str]
    command: Optional[str]
    batch_size: Optional[int]
    seeded: bool
    autocast_dtype: Optional[str]
    finalize_2d_fastpath: Optional[bool]
    kv_mode: Optional[str]
    kv_page_size: Optional[int]
    dense_page_cap: Optional[int]
    rollout_checkpoint: Optional[bool]
    rollout_checkpoint_reentrant: Optional[bool]
    grad_mutable_kv: Optional[bool]
    torch_compile: Optional[bool]
    wallclock_s: Optional[float]
    mean_loss: Optional[float]
    valid_steps: Optional[int]
    skipped_steps: Optional[int]
    rollout_s: Optional[float]
    backward_s: Optional[float]
    step_s: Optional[float]
    backward_calls: Optional[int]
    chunk: Optional[int]
    tbptt: Optional[int]
    status: Optional[str]
    tokens_per_sec: Optional[float]
    host_max_rss_kb: Optional[int]
    rollout_policy_wall_ms: Optional[float]
    rollout_transition_wall_ms: Optional[float]
    rollout_transition_y_share: Optional[float]
    rollout_transition_x_share: Optional[float]
    policy_step_total_ms: Optional[float]
    policy_step_transformer_share: Optional[float]
    policy_step_tf_layer_total_ms: Optional[float]
    policy_step_tf_layer_attnff_share: Optional[float]
    policy_step_tf_layer_attn_core_share: Optional[float]
    policy_step_tf_layer_finalize_share: Optional[float]
    pg_loss_sig: Optional[str]

    @property
    def strict_valid(self) -> bool:
        if self.status != "ok":
            return False
        if self.valid_steps is None or self.skipped_steps is None:
            return False
        if not (self.valid_steps > 0 and self.skipped_steps == 0):
            return False
        if self.mean_loss is None:
            return False
        if math.isinf(self.mean_loss) or math.isnan(self.mean_loss):
            return False
        return True

    @property
    def wallclock_per_chunk_s(self) -> Optional[float]:
        if self.wallclock_s is None:
            return None
        if self.chunk is None or self.chunk <= 0:
            return None
        return float(self.wallclock_s / float(self.chunk))

    @property
    def wallclock_per_batch_s(self) -> Optional[float]:
        if self.wallclock_s is None:
            return None
        if self.batch_size is None or self.batch_size <= 0:
            return None
        return float(self.wallclock_s / float(self.batch_size))

    @property
    def cohort_key(self) -> str:
        parts = [
            f"seeded={int(bool(self.seeded))}",
            f"dtype={self.autocast_dtype or 'na'}",
            f"kv={self.kv_mode or 'na'}",
            f"page={self.kv_page_size if self.kv_page_size is not None else 'na'}",
            f"densecap={self.dense_page_cap if self.dense_page_cap is not None else 'na'}",
            f"ckpt={self.rollout_checkpoint if self.rollout_checkpoint is not None else 'na'}",
            f"reentrant={self.rollout_checkpoint_reentrant if self.rollout_checkpoint_reentrant is not None else 'na'}",
            f"mutable={self.grad_mutable_kv if self.grad_mutable_kv is not None else 'na'}",
            f"compile={self.torch_compile if self.torch_compile is not None else 'na'}",
            f"chunk={self.chunk if self.chunk is not None else 'na'}",
            f"tbptt={self.tbptt if self.tbptt is not None else 'na'}",
            f"fin2d={self.finalize_2d_fastpath if self.finalize_2d_fastpath is not None else 'na'}",
            f"pgsig={self.pg_loss_sig or 'na'}",
        ]
        return "|".join(parts)


def _parse_run_dir(run_dir: Path) -> Optional[RunRecord]:
    run_log = run_dir / "run.log"
    if not run_log.exists():
        return None

    text = _read_text(run_log)
    lines = text.splitlines()

    git_head = None
    command = None
    cmd_txt = run_dir / "cmd.txt"
    if cmd_txt.exists():
        for line in _read_text(cmd_txt).splitlines():
            m_git = CMD_TXT_GIT_RE.match(line.strip())
            if m_git:
                git_head = m_git.group(1).strip()
            m_cmd = CMD_TXT_CMD_RE.match(line.strip())
            if m_cmd:
                command = m_cmd.group(1).strip()

    for line in lines:
        m_cmd_timed = COMMAND_TIMED_RE.search(line)
        if m_cmd_timed:
            command = m_cmd_timed.group(1).strip()

    batch_size = None
    if command:
        m_bs = BATCH_SIZE_RE.search(command)
        if m_bs:
            batch_size = _to_int(m_bs.group(1))

    seeded = any(SEED_SET_RE.match(line.strip()) for line in lines)

    autocast_dtype = None
    finalize_2d_fastpath = None
    kv_mode = None
    kv_page_size = None
    dense_page_cap = None
    rollout_checkpoint = None
    rollout_checkpoint_reentrant = None
    grad_mutable_kv = None
    torch_compile = None

    for line in lines:
        if autocast_dtype is None:
            m = POLICY_AUTODTYPE_RE.search(line)
            if m:
                autocast_dtype = m.group(1).strip().lower()
        if finalize_2d_fastpath is None:
            m = POLICY_FINALIZE2D_RE.search(line)
            if m:
                finalize_2d_fastpath = _to_bool(m.group(1))
        if kv_mode is None:
            m = POLICY_KV_MODE_RE.search(line)
            if m:
                kv_mode = m.group(1).strip().lower()
        if kv_page_size is None:
            m = POLICY_KV_PAGE_RE.search(line)
            if m:
                kv_page_size = _to_int(m.group(1))
        if dense_page_cap is None:
            m = POLICY_DENSE_CAP_RE.search(line)
            if m:
                dense_page_cap = _to_int(m.group(1))
        if rollout_checkpoint is None:
            m = POLICY_CKPT_RE.search(line)
            if m:
                rollout_checkpoint = _to_bool(m.group(1))
        if rollout_checkpoint_reentrant is None:
            m = POLICY_CKPT_REENTRANT_RE.search(line)
            if m:
                rollout_checkpoint_reentrant = _to_bool(m.group(1))
        if grad_mutable_kv is None:
            m = POLICY_GRAD_MUTABLE_RE.search(line)
            if m:
                grad_mutable_kv = _to_bool(m.group(1))
        if torch_compile is None:
            m = POLICY_COMPILE_RE.search(line)
            if m:
                torch_compile = _to_bool(m.group(1))

    wallclock_s = None
    mean_loss = None
    valid_steps = None
    skipped_steps = None
    tokens_per_sec = None
    host_max_rss_kb = None

    last_pg = None
    for line in lines:
        if wallclock_s is None:
            m = WALLCLOCK_RE.search(line)
            if m:
                wallclock_s = _to_float(m.group(1))
        if mean_loss is None:
            m = MEAN_LOSS_RE.search(line)
            if m:
                mean_loss = _to_float(m.group(1))
        if valid_steps is None or skipped_steps is None:
            m = VALID_SKIPPED_RE.search(line)
            if m:
                valid_steps = _to_int(m.group(1))
                skipped_steps = _to_int(m.group(2))
        if tokens_per_sec is None:
            m = TOKENS_PER_SEC_RE.search(line)
            if m:
                tokens_per_sec = _to_float(m.group(1))
        if host_max_rss_kb is None:
            m = HOST_RSS_RE.search(line)
            if m:
                host_max_rss_kb = _to_int(m.group(1))
        m_pg = PG_PHASE_RE.search(line)
        if m_pg:
            last_pg = m_pg

    rollout_s = None
    backward_s = None
    step_s = None
    backward_calls = None
    chunk = None
    tbptt = None
    status = None
    rollout_policy_wall_ms = None
    rollout_transition_wall_ms = None
    rollout_transition_y_share = None
    rollout_transition_x_share = None
    policy_step_total_ms = None
    policy_step_transformer_share = None
    policy_step_tf_layer_total_ms = None
    policy_step_tf_layer_attnff_share = None
    policy_step_tf_layer_attn_core_share = None
    policy_step_tf_layer_finalize_share = None
    pg_loss_sig = None

    if last_pg is not None:
        rollout_s = _to_float(last_pg.group("rollout_s"))
        backward_s = _to_float(last_pg.group("backward_s"))
        step_s = _to_float(last_pg.group("step_s"))
        backward_calls = _to_int(last_pg.group("backward_calls"))
        chunk = _to_int(last_pg.group("chunk"))
        tbptt = _to_int(last_pg.group("tbptt"))
        status = last_pg.group("status")
        tail_metrics = _parse_tail_metrics(last_pg.group("tail"))
        rollout_policy_wall_ms = _tail_float(tail_metrics, "rollout_policy_wall_ms")
        rollout_transition_wall_ms = _tail_float(tail_metrics, "rollout_transition_wall_ms")
        rollout_transition_y_share = _tail_float(tail_metrics, "rollout_transition_y_share")
        rollout_transition_x_share = _tail_float(tail_metrics, "rollout_transition_x_share")
        policy_step_total_ms = _tail_float(tail_metrics, "policy_step_total_ms")
        policy_step_transformer_share = _tail_float(tail_metrics, "policy_step_transformer_share")
        policy_step_tf_layer_total_ms = _tail_float(tail_metrics, "policy_step_tf_layer_total_ms")
        policy_step_tf_layer_attnff_share = _tail_float(tail_metrics, "policy_step_tf_layer_attnff_share")
        policy_step_tf_layer_attn_core_share = _tail_float(tail_metrics, "policy_step_tf_layer_attn_core_share")
        policy_step_tf_layer_finalize_share = _tail_float(tail_metrics, "policy_step_tf_layer_finalize_share")
        pg_loss_sig = tail_metrics.get("pg_loss_sig", None)

    return RunRecord(
        run_id=run_dir.name,
        run_dir=str(run_dir),
        git_head=git_head,
        command=command,
        batch_size=batch_size,
        seeded=seeded,
        autocast_dtype=autocast_dtype,
        finalize_2d_fastpath=finalize_2d_fastpath,
        kv_mode=kv_mode,
        kv_page_size=kv_page_size,
        dense_page_cap=dense_page_cap,
        rollout_checkpoint=rollout_checkpoint,
        rollout_checkpoint_reentrant=rollout_checkpoint_reentrant,
        grad_mutable_kv=grad_mutable_kv,
        torch_compile=torch_compile,
        wallclock_s=wallclock_s,
        mean_loss=mean_loss,
        valid_steps=valid_steps,
        skipped_steps=skipped_steps,
        rollout_s=rollout_s,
        backward_s=backward_s,
        step_s=step_s,
        backward_calls=backward_calls,
        chunk=chunk,
        tbptt=tbptt,
        status=status,
        tokens_per_sec=tokens_per_sec,
        host_max_rss_kb=host_max_rss_kb,
        rollout_policy_wall_ms=rollout_policy_wall_ms,
        rollout_transition_wall_ms=rollout_transition_wall_ms,
        rollout_transition_y_share=rollout_transition_y_share,
        rollout_transition_x_share=rollout_transition_x_share,
        policy_step_total_ms=policy_step_total_ms,
        policy_step_transformer_share=policy_step_transformer_share,
        policy_step_tf_layer_total_ms=policy_step_tf_layer_total_ms,
        policy_step_tf_layer_attnff_share=policy_step_tf_layer_attnff_share,
        policy_step_tf_layer_attn_core_share=policy_step_tf_layer_attn_core_share,
        policy_step_tf_layer_finalize_share=policy_step_tf_layer_finalize_share,
        pg_loss_sig=pg_loss_sig,
    )


def _fmt(v: Optional[float], nd: int = 2) -> str:
    if v is None:
        return "na"
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return str(v)
    return f"{v:.{nd}f}"


def _status_counts(records: List[RunRecord]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for r in records:
        s = r.status or "unknown"
        out[s] = out.get(s, 0) + 1
    return out


def _write_jsonl(path: Path, records: List[RunRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            payload = asdict(r)
            payload["strict_valid"] = bool(r.strict_valid)
            payload["wallclock_per_chunk_s"] = r.wallclock_per_chunk_s
            payload["wallclock_per_batch_s"] = r.wallclock_per_batch_s
            payload["cohort_key"] = r.cohort_key
            f.write(json.dumps(payload, ensure_ascii=True))
            f.write("\n")


def _render_table(records: List[RunRecord], title: str, max_rows: int) -> List[str]:
    lines = [f"## {title}", "", "| rank | run_id | status | valid/skipped | wallclock(s) | wall/batch(s) | rollout/backward(s) | chunk | wall/chunk | dtype | notes |", "|---:|---|---|---|---:|---:|---|---:|---:|---|---|"]
    sorted_rows = sorted(
        [r for r in records if r.wallclock_s is not None],
        key=lambda x: (x.wallclock_s if x.wallclock_s is not None else 1e18),
    )[:max_rows]
    for i, r in enumerate(sorted_rows, start=1):
        valid_skipped = f"{r.valid_steps if r.valid_steps is not None else 'na'}/{r.skipped_steps if r.skipped_steps is not None else 'na'}"
        rb = f"{_fmt(r.rollout_s)}/{_fmt(r.backward_s)}"
        note_bits = []
        if r.policy_step_tf_layer_finalize_share is not None:
            note_bits.append(f"fin={_fmt(100.0 * r.policy_step_tf_layer_finalize_share, 1)}%")
        if r.rollout_transition_y_share is not None and r.rollout_transition_x_share is not None:
            note_bits.append(
                f"tr(y/x)={_fmt(100.0 * r.rollout_transition_y_share, 1)}/{_fmt(100.0 * r.rollout_transition_x_share, 1)}%"
            )
        if r.pg_loss_sig:
            note_bits.append(f"pg={r.pg_loss_sig}")
        notes = ", ".join(note_bits) if note_bits else "-"
        lines.append(
            f"| {i} | `{r.run_id}` | `{r.status or 'na'}` | `{valid_skipped}` | `{_fmt(r.wallclock_s)}` | `{_fmt(r.wallclock_per_batch_s, 4)}` | `{rb}` | `{r.chunk if r.chunk is not None else 'na'}` | `{_fmt(r.wallclock_per_chunk_s)}` | `{r.autocast_dtype or 'na'}` | {notes} |"
        )
    if not sorted_rows:
        lines.append("| - | - | - | - | - | - | - | - | - | - | - |")
    lines.append("")
    return lines


def _render_cohort_summary(records: List[RunRecord], max_rows: int) -> List[str]:
    by_cohort: Dict[str, List[RunRecord]] = {}
    for r in records:
        by_cohort.setdefault(r.cohort_key, []).append(r)
    cohort_rows = []
    for key, rows in by_cohort.items():
        valid_rows = [r for r in rows if r.strict_valid and r.wallclock_s is not None]
        any_rows = [r for r in rows if r.wallclock_s is not None]
        best_valid = min(valid_rows, key=lambda x: x.wallclock_s) if valid_rows else None
        best_any = min(any_rows, key=lambda x: x.wallclock_s) if any_rows else None
        cohort_rows.append(
            (
                key,
                len(rows),
                len(valid_rows),
                None if best_valid is None else best_valid.wallclock_s,
                None if best_any is None else best_any.wallclock_s,
                None if best_any is None else best_any.run_id,
                None if best_valid is None else best_valid.run_id,
            )
        )
    cohort_rows.sort(key=lambda x: (-(x[2]), x[3] if x[3] is not None else 1e18, x[4] if x[4] is not None else 1e18))

    lines = [
        "## Cohort Ladder",
        "",
        "| rank | cohort_key | runs | strict_valid | best_strict_valid(s) | best_any_status(s) | best_valid_run | best_any_run |",
        "|---:|---|---:|---:|---:|---:|---|---|",
    ]
    for i, row in enumerate(cohort_rows[:max_rows], start=1):
        lines.append(
            "| {i} | `{k}` | `{runs}` | `{valids}` | `{best_valid}` | `{best_any}` | `{best_valid_run}` | `{best_any_run}` |".format(
                i=i,
                k=row[0],
                runs=row[1],
                valids=row[2],
                best_valid=_fmt(row[3]),
                best_any=_fmt(row[4]),
                best_valid_run=row[6] or "na",
                best_any_run=row[5] or "na",
            )
        )
    if not cohort_rows:
        lines.append("| - | - | - | - | - | - | - | - |")
    lines.append("")
    return lines


def _write_markdown(path: Path, records: List[RunRecord], max_rows: int) -> None:
    total = len(records)
    strict_valid = [r for r in records if r.strict_valid]
    non_strict = [r for r in records if not r.strict_valid]
    status_counts = _status_counts(records)

    lines: List[str] = []
    lines.append("# Rebuilt Skyline Ladder (post-stability)")
    lines.append("")
    lines.append("## Validity Gate")
    lines.append("")
    lines.append("- `strict_valid = True` iff all conditions hold:")
    lines.append("  - `status == ok` in `[pg-phase]`")
    lines.append("  - `valid/skipped` is `>=1/0`")
    lines.append("  - `mean loss` is finite")
    lines.append("- Skyline cohorts are also separated by `pg_loss_sig` to avoid mixing different loss forms.")
    lines.append("- Purpose: prevent pre-stability `grad_norm_nonfinite/inf` runs from polluting skyline comparisons.")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Runs scanned: `{total}`")
    lines.append(f"- Strict-valid runs: `{len(strict_valid)}`")
    lines.append(f"- Non-strict/diagnostic runs: `{len(non_strict)}`")
    lines.append(f"- Status counts: `{json.dumps(status_counts, ensure_ascii=True)}`")
    lines.append("")

    lines.extend(_render_table(strict_valid, "Strict-Valid Skyline", max_rows=max_rows))
    lines.extend(_render_table(non_strict, "Diagnostic Ladder (Invalid/Skipped/OOM)", max_rows=max_rows))
    lines.extend(_render_cohort_summary(records, max_rows=max_rows))

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild rollout-kernel skyline with strict validity gating.")
    parser.add_argument(
        "--bench-dir",
        type=str,
        default="benchmarks/skyline_20260302_rollout_kernel",
        help="Benchmark root directory containing run subdirectories with run.log.",
    )
    parser.add_argument(
        "--out-jsonl",
        type=str,
        default="benchmarks/skyline_20260302_rollout_kernel/REBUILT_RUNS.jsonl",
        help="Output JSONL path for parsed run records.",
    )
    parser.add_argument(
        "--out-md",
        type=str,
        default="benchmarks/skyline_20260302_rollout_kernel/REBUILT_SKYLINE.md",
        help="Output markdown path for rebuilt skyline ladder.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=20,
        help="Maximum rows per markdown table.",
    )
    args = parser.parse_args()

    bench_dir = Path(args.bench_dir)
    if not bench_dir.exists():
        raise SystemExit(f"bench dir not found: {bench_dir}")

    records: List[RunRecord] = []
    for child in sorted(bench_dir.iterdir()):
        if not child.is_dir():
            continue
        rec = _parse_run_dir(child)
        if rec is None:
            continue
        records.append(rec)

    _write_jsonl(Path(args.out_jsonl), records)
    _write_markdown(Path(args.out_md), records, max_rows=max(1, int(args.max_rows)))

    strict_valid = [r for r in records if r.strict_valid and r.wallclock_s is not None]
    best_strict = min(strict_valid, key=lambda x: x.wallclock_s) if strict_valid else None
    print(f"scanned_runs={len(records)} strict_valid_runs={len(strict_valid)}")
    if best_strict is not None:
        print(
            "best_strict_valid="
            f"{best_strict.run_id} wallclock_s={best_strict.wallclock_s:.2f} "
            f"status={best_strict.status} valid={best_strict.valid_steps}/{best_strict.skipped_steps}"
        )
    print(f"wrote_jsonl={args.out_jsonl}")
    print(f"wrote_markdown={args.out_md}")


if __name__ == "__main__":
    main()
