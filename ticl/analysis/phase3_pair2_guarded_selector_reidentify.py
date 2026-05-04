import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_GUARDED_SELECTOR_TRACE_JSON = (
    "/home/chen/RLPFN/artifacts/phase3_pair2_guarded_target_present_selector_trace.json"
)
DEFAULT_SOURCE_COMPARE_PACK_JSON = (
    "/home/chen/RLPFN/artifacts/phase3_env12_mid_episode_extension_compare_pack.json"
)
DEFAULT_OUTPUT_JSON = (
    "/home/chen/RLPFN/artifacts/phase3_pair2_guarded_selector_reidentified.json"
)


def _load_json(path: str) -> dict[str, Any]:
    return json.loads(Path(path).expanduser().resolve().read_text())


def _as_int_list(values: Any) -> list[int]:
    return [int(v) for v in list(values)]


def _require_contiguous(values: list[int], *, name: str) -> None:
    if not values:
        raise ValueError(f"Expected non-empty contiguous {name}.")
    ordered = sorted(int(v) for v in values)
    expected = list(range(int(ordered[0]), int(ordered[-1]) + 1))
    if ordered != expected:
        raise ValueError(f"Expected contiguous {name}, got {ordered!r}.")


def _load_source_anchor(source_compare_pack_json: str) -> dict[str, Any]:
    payload = _load_json(source_compare_pack_json)
    if str(payload.get("audit_entry", "")) != "phase3_env12_mid_episode_extension_compare_pack":
        raise ValueError("Expected phase3_env12_mid_episode_extension_compare_pack artifact.")
    extension_block = dict(payload.get("comparison", {}).get("extension_block", {}))
    global_positions = _as_int_list(extension_block.get("local_indices", []))
    _require_contiguous(global_positions, name="source objective-global positions")
    return {
        "source_compare_pack_json": str(Path(source_compare_pack_json).expanduser().resolve()),
        "target_env_index": int(extension_block.get("env_index", 12)),
        "objective_global_positions": global_positions,
        "objective_global_position_start": int(global_positions[0]),
        "objective_global_position_end": int(global_positions[-1]),
        "token_count": int(len(global_positions)),
        "global_steps": _as_int_list(extension_block.get("global_steps", [])),
    }


def _load_guarded_env_row(guarded_selector_trace_json: str, *, target_env_index: int) -> dict[str, Any]:
    payload = _load_json(guarded_selector_trace_json)
    if str(payload.get("probe_entry", "")) != "phase3_pair2_guarded_target_present_selector_trace":
        raise ValueError("Expected phase3_pair2_guarded_target_present_selector_trace artifact.")
    baseline = dict(payload.get("baseline", {}))
    selector_trace = dict(baseline.get("selector_trace", {}))
    env_present_rows = list(selector_trace.get("env_present_rows", []))
    matching_rows = [
        dict(row)
        for row in env_present_rows
        if int(row.get("objective_env_indices_present", [-1])[0]) == int(target_env_index)
    ]
    if len(matching_rows) != 1:
        raise ValueError(
            "Expected exactly one env-present row for the guarded target env, got "
            f"{len(matching_rows)}."
        )
    return matching_rows[0]


def _map_global_positions_to_current_selector(
    *,
    env_present_row: dict[str, Any],
    objective_global_positions: list[int],
    target_env_index: int,
) -> dict[str, Any]:
    _require_contiguous(objective_global_positions, name="guarded semantic anchor")
    spans = [
        {
            "objective_episode_index": int(seg["objective_episode_index"]),
            "objective_position_min": int(seg["objective_position_min"]),
            "objective_position_max": int(seg["objective_position_max"]),
            "token_count": int(seg["token_count"]),
        }
        for seg in list(env_present_row.get("target_env_objective_episode_spans", []))
    ]
    if not spans:
        raise ValueError("Guarded trace row has no target-env episode spans.")

    cursor = 0
    mapped_rows: list[dict[str, int]] = []
    anchor_set = {int(v) for v in objective_global_positions}
    for seg in spans:
        token_count = int(seg["token_count"])
        global_start = int(cursor)
        global_end = int(cursor + token_count - 1)
        overlap = sorted(v for v in anchor_set if int(global_start) <= int(v) <= int(global_end))
        if overlap:
            for global_position in overlap:
                mapped_rows.append(
                    {
                        "outer_batch_idx": int(env_present_row["outer_batch_idx"]),
                        "target_env_index": int(target_env_index),
                        "objective_global_position": int(global_position),
                        "objective_episode_index": int(seg["objective_episode_index"]),
                        "objective_episode_position": int(
                            int(seg["objective_position_min"]) + int(global_position) - int(global_start)
                        ),
                    }
                )
        cursor += token_count

    mapped_global_positions = [int(row["objective_global_position"]) for row in mapped_rows]
    if mapped_global_positions != list(objective_global_positions):
        raise ValueError(
            "Guarded trace did not preserve the full semantic anchor. "
            f"Expected {objective_global_positions!r}, got {mapped_global_positions!r}."
        )

    episode_indices = sorted({int(row["objective_episode_index"]) for row in mapped_rows})
    if len(episode_indices) != 1:
        raise ValueError(
            "Semantic anchor spans multiple guarded episodes; expected a single contiguous selector, got "
            f"{episode_indices!r}."
        )
    episode_positions = [int(row["objective_episode_position"]) for row in mapped_rows]
    _require_contiguous(episode_positions, name="guarded episode positions")
    return {
        "outer_batch_idx": int(env_present_row["outer_batch_idx"]),
        "target_env_index": int(target_env_index),
        "target_objective_episode_index": int(episode_indices[0]),
        "target_objective_position_start": int(min(episode_positions)),
        "target_objective_position_end": int(max(episode_positions)),
        "token_count": int(len(mapped_rows)),
        "mapped_rows": mapped_rows,
    }


def build_pair2_guarded_selector_reidentified(
    *,
    guarded_selector_trace_json: str,
    source_compare_pack_json: str,
) -> dict[str, Any]:
    source_anchor = _load_source_anchor(source_compare_pack_json)
    env_present_row = _load_guarded_env_row(
        guarded_selector_trace_json,
        target_env_index=int(source_anchor["target_env_index"]),
    )
    reidentified = _map_global_positions_to_current_selector(
        env_present_row=env_present_row,
        objective_global_positions=list(source_anchor["objective_global_positions"]),
        target_env_index=int(source_anchor["target_env_index"]),
    )
    requested_selector = _load_json(guarded_selector_trace_json)["target_selector"]
    return {
        "probe_entry": "phase3_pair2_guarded_selector_reidentified",
        "inputs": {
            "guarded_selector_trace_json": str(Path(guarded_selector_trace_json).expanduser().resolve()),
            "source_compare_pack_json": str(Path(source_compare_pack_json).expanduser().resolve()),
        },
        "source_semantic_anchor": source_anchor,
        "guarded_trace_env_row": {
            "outer_batch_idx": int(env_present_row["outer_batch_idx"]),
            "target_status": str(env_present_row["target_status"]),
            "objective_env_indices_present": _as_int_list(env_present_row["objective_env_indices_present"]),
            "target_env_objective_token_count": int(env_present_row["target_env_objective_token_count"]),
            "target_env_objective_episode_indices_present": _as_int_list(
                env_present_row["target_env_objective_episode_indices_present"]
            ),
            "target_env_objective_episode_spans": [
                {
                    "objective_episode_index": int(seg["objective_episode_index"]),
                    "objective_position_min": int(seg["objective_position_min"]),
                    "objective_position_max": int(seg["objective_position_max"]),
                    "token_count": int(seg["token_count"]),
                }
                for seg in list(env_present_row["target_env_objective_episode_spans"])
            ],
        },
        "requested_selector": {
            "target_outer_batch_idx": requested_selector.get("target_outer_batch_idx", None),
            "target_env_index": int(requested_selector["target_env_index"]),
            "target_objective_episode_index": int(requested_selector["target_objective_episode_index"]),
            "target_objective_position_start": int(requested_selector["target_objective_position_start"]),
            "target_objective_position_end": int(requested_selector["target_objective_position_end"]),
        },
        "reidentified_selector": reidentified,
        "conclusions": {
            "guarded_requested_selector_is_stale": True,
            "semantic_anchor_is_objective_global_positions": True,
            "reidentified_selector_found_under_same_guarded_contract": True,
            "reidentified_selector_preserves_target_env_identity": bool(
                int(reidentified["target_env_index"]) == int(source_anchor["target_env_index"])
            ),
            "reidentified_selector_requires_episode_shift": bool(
                int(reidentified["target_objective_episode_index"])
                != int(requested_selector["target_objective_episode_index"])
            ),
            "reidentified_selector_requires_position_shift": bool(
                int(reidentified["target_objective_position_start"])
                != int(requested_selector["target_objective_position_start"])
                or int(reidentified["target_objective_position_end"])
                != int(requested_selector["target_objective_position_end"])
            ),
            "reidentified_selector_requires_outer_batch_lock": bool(
                requested_selector.get("target_outer_batch_idx", None)
                != reidentified["outer_batch_idx"]
            ),
            "semantic_anchor_maps_to_single_current_episode": True,
        },
        "recommendation": {
            "next_focus": "repair_selector_definition",
            "reason": (
                "Use the reidentified guarded-contract selector for both the runtime gate and the target-side "
                "snapshot capture before treating any pair2 target-side A/B as efficacy evidence."
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Reidentify the current guarded-contract pair2 selector from the stable env12 semantic anchor."
    )
    parser.add_argument("--guarded-selector-trace-json", type=str, default=DEFAULT_GUARDED_SELECTOR_TRACE_JSON)
    parser.add_argument("--source-compare-pack-json", type=str, default=DEFAULT_SOURCE_COMPARE_PACK_JSON)
    parser.add_argument("--output-json", type=str, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output_path = Path(args.output_json).expanduser().resolve()
    if output_path.exists() and not bool(args.overwrite):
        print(output_path.read_text())
        return 0

    report = build_pair2_guarded_selector_reidentified(
        guarded_selector_trace_json=args.guarded_selector_trace_json,
        source_compare_pack_json=args.source_compare_pack_json,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
