import argparse
import json
from pathlib import Path
from typing import Any


def _load_scope_payload(path: str) -> dict[str, Any]:
    payload = json.loads(Path(path).expanduser().resolve().read_text())
    if str(payload.get("audit_entry", "")) != "phase3_env12_local56_boundary_scope_counterfactual_probe":
        raise ValueError("Expected phase3_env12_local56_boundary_scope_counterfactual_probe artifact.")
    return payload


def _decompose_boundary_magnitude_delta(
    *,
    current_mean_raw: float,
    current_std_raw: float,
    counter_mean_raw: float,
    counter_std_raw: float,
) -> dict[str, float]:
    current_mag = float(current_mean_raw / current_std_raw)
    counter_mag = float(counter_mean_raw / counter_std_raw)
    mean_only = float((counter_mean_raw - current_mean_raw) / current_std_raw)
    std_only = float(current_mean_raw * ((1.0 / counter_std_raw) - (1.0 / current_std_raw)))
    interaction = float((counter_mean_raw - current_mean_raw) * ((1.0 / counter_std_raw) - (1.0 / current_std_raw)))
    total = float(counter_mag - current_mag)
    return {
        "current_boundary_magnitude": current_mag,
        "counterfactual_boundary_magnitude": counter_mag,
        "boundary_magnitude_delta": total,
        "mean_only_delta": mean_only,
        "std_only_delta": std_only,
        "interaction_delta": interaction,
        "reconstruction_error": float((mean_only + std_only + interaction) - total),
    }


def _dominant_driver(mean_only: float, std_only: float, interaction: float) -> str:
    worsening_terms = {
        "mean_too_large": float(max(0.0, mean_only)),
        "std_too_small": float(max(0.0, std_only)),
        "interaction": float(max(0.0, interaction)),
    }
    label, value = max(worsening_terms.items(), key=lambda item: (float(item[1]), str(item[0])))
    if value <= 0.0:
        return "none"
    return str(label)


def _build_probe(scope_payload: dict[str, Any]) -> dict[str, Any]:
    scope_rows = [dict(row) for row in list(scope_payload.get("scope_rows", []))]
    row_by_name = {str(row["scope_name"]): dict(row) for row in scope_rows}
    current = dict(row_by_name["current_full_rollout"])
    comparisons = []
    for scope_name in ["objective_all_tokens", "objective_episode0_only", "objective_episode1_only"]:
        row = dict(row_by_name[scope_name])
        decomp = _decompose_boundary_magnitude_delta(
            current_mean_raw=float(current["mean_raw"]),
            current_std_raw=float(current["std_raw"]),
            counter_mean_raw=float(row["mean_raw"]),
            counter_std_raw=float(row["std_raw"]),
        )
        comparisons.append(
            {
                "scope_name": str(scope_name),
                "current_mean_raw": float(current["mean_raw"]),
                "current_std_raw": float(current["std_raw"]),
                "counter_mean_raw": float(row["mean_raw"]),
                "counter_std_raw": float(row["std_raw"]),
                "current_mean_over_std": float(current["mean_over_std"]),
                "counter_mean_over_std": float(row["mean_over_std"]),
                "current_boundary_shift_norm": float(current["local56_boundary_shift_norm"]),
                "counter_boundary_shift_norm": float(row["local56_boundary_shift_norm"]),
                **decomp,
                "dominant_worsening_driver": _dominant_driver(
                    decomp["mean_only_delta"],
                    decomp["std_only_delta"],
                    decomp["interaction_delta"],
                ),
            }
        )

    objective_all = next(row for row in comparisons if str(row["scope_name"]) == "objective_all_tokens")
    episode0 = next(row for row in comparisons if str(row["scope_name"]) == "objective_episode0_only")
    episode1 = next(row for row in comparisons if str(row["scope_name"]) == "objective_episode1_only")

    return {
        "audit_entry": "phase3_env12_local56_boundary_mean_std_interaction_probe",
        "config": {
            "source_boundary_scope_counterfactual_json": str(
                Path(
                    "/home/chen/RLPFN/artifacts/phase3_env12_local56_boundary_scope_counterfactual_probe.json"
                ).expanduser().resolve()
            )
        },
        "comparison_rows": comparisons,
        "conclusions": {
            "objective_all_worsening_is_std_dominated": bool(
                str(objective_all["dominant_worsening_driver"]) == "std_too_small"
                and float(objective_all["std_only_delta"]) > 0.0
                and float(objective_all["mean_only_delta"]) < 0.0
            ),
            "episode0_worsening_is_std_dominated": bool(
                str(episode0["dominant_worsening_driver"]) == "std_too_small"
                and float(episode0["std_only_delta"]) > 0.0
                and float(episode0["mean_only_delta"]) < 0.0
            ),
            "episode1_worsening_is_std_dominated": bool(
                str(episode1["dominant_worsening_driver"]) == "std_too_small"
                and float(episode1["std_only_delta"]) > 0.0
                and float(episode1["mean_only_delta"]) < 0.0
            ),
            "mean_change_alone_would_shrink_penalty_in_all_objective_scopes": bool(
                all(float(row["mean_only_delta"]) < 0.0 for row in comparisons)
            ),
            "std_change_alone_would_worsen_penalty_in_all_objective_scopes": bool(
                all(float(row["std_only_delta"]) > 0.0 for row in comparisons)
            ),
            "interaction_offsets_part_of_std_worsening_in_all_objective_scopes": bool(
                all(float(row["interaction_delta"]) < 0.0 for row in comparisons)
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Decompose local56 boundary worsening into mean-only, std-only, and interaction terms."
    )
    parser.add_argument(
        "--boundary-scope-counterfactual-json",
        type=str,
        default="/home/chen/RLPFN/artifacts/phase3_env12_local56_boundary_scope_counterfactual_probe.json",
    )
    parser.add_argument("--output-json", type=str, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output_path = Path(args.output_json).expanduser().resolve()
    if output_path.exists() and not bool(args.overwrite):
        print(output_path.read_text())
        return 0

    scope_payload = _load_scope_payload(args.boundary_scope_counterfactual_json)
    result = _build_probe(scope_payload)
    result["config"]["source_boundary_scope_counterfactual_json"] = str(
        Path(args.boundary_scope_counterfactual_json).expanduser().resolve()
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(output_path.read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
