import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_PHASE2_SCALE_SELECTOR_TRACE_JSON = (
    "/home/chen/RLPFN/artifacts/phase3_pair2_phase2_scale_target_present_selector_trace.json"
)
DEFAULT_OUTPUT_JSON = (
    "/home/chen/RLPFN/artifacts/phase3_pair2_phase2_scale_selector_reidentified.json"
)


def _load_json(path: str) -> dict[str, Any]:
    return json.loads(Path(path).expanduser().resolve().read_text())


def _sorted_positions(row: dict[str, Any]) -> list[int]:
    return sorted(int(v) for v in list(row.get("target_episode_positions_present", [])))


def _contiguous_span(positions: list[int]) -> dict[str, int] | None:
    if not positions:
        return None
    return {
        "position_start": int(min(positions)),
        "position_end": int(max(positions)),
        "token_count": int(len(positions)),
    }


def _candidate_rows(trace_payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = list(trace_payload.get("rows", []))
    candidates: list[dict[str, Any]] = []
    for row in rows:
        if int(row.get("target_env_objective_token_count", 0)) <= 0:
            continue
        if int(row.get("target_episode_token_count", 0)) <= 0:
            continue
        positions = _sorted_positions(row)
        span = _contiguous_span(positions)
        if span is None:
            continue
        candidates.append(
            {
                "epoch_idx": int(row["epoch_idx"]),
                "outer_batch_idx": int(row["outer_batch_idx"]),
                "target_env_index": int(row["target_env_index"]),
                "target_objective_episode_index": int(row["target_objective_episode_index"]),
                "target_status": str(row["target_status"]),
                "target_episode_token_count": int(row["target_episode_token_count"]),
                "target_env_objective_token_count": int(row["target_env_objective_token_count"]),
                "target_env_objective_episode_indices_present": [
                    int(v) for v in list(row.get("target_env_objective_episode_indices_present", []))
                ],
                "target_episode_positions_present": positions,
                "candidate_position_start": int(span["position_start"]),
                "candidate_position_end": int(span["position_end"]),
                "candidate_token_count": int(span["token_count"]),
            }
        )
    return candidates


def _pick_canonical_selector(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    if not candidates:
        raise ValueError("No Phase-2-scale selector candidates were found in the trace.")
    ordered = sorted(
        candidates,
        key=lambda row: (
            int(row["outer_batch_idx"]),
            int(row["epoch_idx"]),
            -int(row["candidate_token_count"]),
            int(row["candidate_position_start"]),
        ),
    )
    return dict(ordered[0])


def build_phase2_scale_selector_reidentified(
    *,
    selector_trace_json: str,
) -> dict[str, Any]:
    trace_payload = _load_json(selector_trace_json)
    candidates = _candidate_rows(trace_payload)
    canonical = _pick_canonical_selector(candidates)
    summary = dict(trace_payload.get("summary", {}))
    selector = dict(trace_payload.get("selector", {}))
    return {
        "probe_entry": "phase3_pair2_phase2_scale_selector_reidentified",
        "inputs": {
            "selector_trace_json": str(Path(selector_trace_json).expanduser().resolve()),
        },
        "phase2_scale_trace_summary": {
            "row_count": int(summary.get("row_count", 0)),
            "target_appeared": bool(summary.get("target_appeared", False)),
            "first_match": summary.get("first_match", None),
            "requested_selector": {
                "target_env_index": None if selector.get("target_env_index") is None else int(selector["target_env_index"]),
                "target_objective_episode_index": None
                if selector.get("target_objective_episode_index") is None
                else int(selector["target_objective_episode_index"]),
                "target_objective_position_start": None
                if selector.get("target_objective_position_start") is None
                else int(selector["target_objective_position_start"]),
                "target_objective_position_end": None
                if selector.get("target_objective_position_end") is None
                else int(selector["target_objective_position_end"]),
            },
        },
        "candidate_rows": candidates,
        "canonical_selector": {
            "epoch_idx": int(canonical["epoch_idx"]),
            "outer_batch_idx": int(canonical["outer_batch_idx"]),
            "target_env_index": int(canonical["target_env_index"]),
            "target_objective_episode_index": int(canonical["target_objective_episode_index"]),
            "target_objective_position_start": int(canonical["candidate_position_start"]),
            "target_objective_position_end": int(canonical["candidate_position_end"]),
            "token_count": int(canonical["candidate_token_count"]),
            "selection_reason": (
                "Phase-2-scale trace contains no exact hit for the stale requested range. "
                "Pick the same env/episode on the earliest outer batch where that episode is present, "
                "and shrink the selector to the actual contiguous position span present there."
            ),
        },
        "conclusions": {
            "phase2_scale_requested_selector_is_absent": bool(not summary.get("target_appeared", False)),
            "phase2_scale_candidate_selector_found": True,
            "candidate_preserves_env_and_episode_identity": bool(
                int(canonical["target_env_index"]) == int(selector["target_env_index"])
                and int(canonical["target_objective_episode_index"]) == int(selector["target_objective_episode_index"])
            ),
            "candidate_requires_position_shrink": bool(
                int(canonical["candidate_position_start"]) != int(selector["target_objective_position_start"])
                or int(canonical["candidate_position_end"]) != int(selector["target_objective_position_end"])
            ),
        },
        "recommendation": {
            "next_focus": "phase2_scale_target_present_recapture",
            "reason": (
                "The trusted Phase-2-scale selector should be re-captured using the reidentified present span, "
                "not the stale larger-scale range."
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reidentify the trusted Phase-2-scale env12 train-side selector from the selector trace."
    )
    parser.add_argument("--selector-trace-json", type=str, default=DEFAULT_PHASE2_SCALE_SELECTOR_TRACE_JSON)
    parser.add_argument("--output-json", type=str, default=DEFAULT_OUTPUT_JSON)
    args = parser.parse_args()

    report = build_phase2_scale_selector_reidentified(
        selector_trace_json=args.selector_trace_json,
    )
    output_path = Path(args.output_json).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
