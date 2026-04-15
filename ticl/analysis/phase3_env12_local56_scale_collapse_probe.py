import argparse
import json
from pathlib import Path
from typing import Any


CONTRACT_EPS = 1e-9


def _load_mean_source_payload(path: str) -> dict[str, Any]:
    payload = json.loads(Path(path).expanduser().resolve().read_text())
    if str(payload.get("audit_entry", "")) != "phase3_env12_local56_mean_source_probe":
        raise ValueError("Expected phase3_env12_local56_mean_source_probe artifact.")
    return payload


def _load_scope_payload(path: str) -> dict[str, Any]:
    payload = json.loads(Path(path).expanduser().resolve().read_text())
    if str(payload.get("audit_entry", "")) != "phase3_env12_local56_boundary_scope_counterfactual_probe":
        raise ValueError("Expected phase3_env12_local56_boundary_scope_counterfactual_probe artifact.")
    return payload


def _variance(std_raw: float) -> float:
    return float(std_raw * std_raw)


def _share(numerator: float, denominator: float) -> float:
    if abs(float(denominator)) <= CONTRACT_EPS:
        return 0.0
    return float(float(numerator) / float(denominator))


def _between_group_contribution(groups: list[dict[str, float]], total_mean: float, total_count: int) -> float:
    contribution = 0.0
    for group in groups:
        weight = float(group["count"]) / float(total_count)
        contribution += weight * (float(group["mean_raw"]) - float(total_mean)) ** 2
    return float(contribution)


def _require_close(name: str, left: float, right: float, atol: float = 1e-6) -> None:
    if abs(float(left) - float(right)) > float(atol):
        raise ValueError(f"{name} mismatch: {left} vs {right}")


def _scope_stat(name: str, *, count: int, mean_raw: float, std_raw: float) -> dict[str, float | int | str]:
    return {
        "scope_name": str(name),
        "token_count": int(count),
        "mean_raw": float(mean_raw),
        "std_raw": float(std_raw),
        "variance_raw": float(_variance(float(std_raw))),
    }


def _build_probe(mean_source_payload: dict[str, Any], scope_payload: dict[str, Any]) -> dict[str, Any]:
    mean_decomp = dict(mean_source_payload["full_mean_decomposition"])
    scope_rows = {str(row["scope_name"]): dict(row) for row in list(scope_payload.get("scope_rows", []))}
    episode_rows = {
        int(row["objective_episode_index"]): dict(row)
        for row in list(mean_source_payload.get("objective_episode_summaries", []))
    }

    full = dict(scope_rows["current_full_rollout"])
    objective = dict(scope_rows["objective_all_tokens"])
    episode0_scope = dict(scope_rows["objective_episode0_only"])
    episode1_scope = dict(scope_rows["objective_episode1_only"])
    episode0_mean = dict(episode_rows[0])
    episode1_mean = dict(episode_rows[1])

    n_total = int(mean_decomp["n_samples"])
    n_objective = int(mean_decomp["objective_token_count"])
    n_prefix = int(mean_decomp["prefix_token_count_before_objective"])
    n_episode0 = int(episode0_mean["token_count"])
    n_episode1 = int(episode1_mean["token_count"])
    if n_objective != n_episode0 + n_episode1:
        raise ValueError("Objective token count must equal episode0 + episode1 token counts.")
    _require_close(
        "objective_mean_raw",
        float(mean_decomp["objective_mean_rollout_return_raw"]),
        float(objective["mean_raw"]),
    )
    if int(objective["token_count"]) != n_objective:
        raise ValueError("Objective token count mismatch between trusted artifacts.")
    if int(episode0_scope["token_count"]) != n_episode0 or int(episode1_scope["token_count"]) != n_episode1:
        raise ValueError("Episode token count mismatch between trusted artifacts.")
    _require_close(
        "episode0_mean_raw",
        float(episode0_mean["mean_rollout_return_raw"]),
        float(episode0_scope["mean_raw"]),
    )
    _require_close(
        "episode1_mean_raw",
        float(episode1_mean["mean_rollout_return_raw"]),
        float(episode1_scope["mean_raw"]),
    )

    full_mean = float(full["mean_raw"])
    full_std = float(full["std_raw"])
    full_var = float(_variance(full_std))
    objective_mean = float(objective["mean_raw"])
    objective_std = float(objective["std_raw"])
    objective_var = float(_variance(objective_std))
    prefix_mean = float(mean_decomp["implied_prefix_mean_rollout_return_raw"])

    prefix_objective_groups = [
        {
            "count": float(n_prefix),
            "mean_raw": float(prefix_mean),
        },
        {
            "count": float(n_objective),
            "mean_raw": float(objective_mean),
        },
    ]
    between_prefix_objective = _between_group_contribution(
        prefix_objective_groups,
        total_mean=full_mean,
        total_count=n_total,
    )
    objective_within_contribution = float((float(n_objective) / float(n_total)) * objective_var)
    prefix_within_contribution = float(full_var - objective_within_contribution - between_prefix_objective)
    if prefix_within_contribution < -1e-6:
        raise ValueError("Inferred prefix within contribution is negative beyond tolerance.")
    prefix_within_contribution = float(max(0.0, prefix_within_contribution))
    prefix_var = float(prefix_within_contribution / (float(n_prefix) / float(n_total)))
    prefix_std = float(prefix_var**0.5)

    episode0_var = float(_variance(float(episode0_scope["std_raw"])))
    episode1_var = float(_variance(float(episode1_scope["std_raw"])))
    episode_groups = [
        {
            "count": float(n_episode0),
            "mean_raw": float(episode0_scope["mean_raw"]),
        },
        {
            "count": float(n_episode1),
            "mean_raw": float(episode1_scope["mean_raw"]),
        },
    ]
    between_episode_means = _between_group_contribution(
        episode_groups,
        total_mean=objective_mean,
        total_count=n_objective,
    )
    episode0_within_contribution = float((float(n_episode0) / float(n_objective)) * episode0_var)
    episode1_within_contribution = float((float(n_episode1) / float(n_objective)) * episode1_var)

    full_reconstruction_error = float(
        (prefix_within_contribution + objective_within_contribution + between_prefix_objective) - full_var
    )
    objective_reconstruction_error = float(
        (episode0_within_contribution + episode1_within_contribution + between_episode_means) - objective_var
    )

    scope_stats = [
        _scope_stat(
            "full_rollout",
            count=n_total,
            mean_raw=full_mean,
            std_raw=full_std,
        ),
        _scope_stat(
            "omitted_prefix_before_objective",
            count=n_prefix,
            mean_raw=prefix_mean,
            std_raw=prefix_std,
        ),
        _scope_stat(
            "objective_all_tokens",
            count=n_objective,
            mean_raw=objective_mean,
            std_raw=objective_std,
        ),
        _scope_stat(
            "objective_episode0_only",
            count=n_episode0,
            mean_raw=float(episode0_scope["mean_raw"]),
            std_raw=float(episode0_scope["std_raw"]),
        ),
        _scope_stat(
            "objective_episode1_only",
            count=n_episode1,
            mean_raw=float(episode1_scope["mean_raw"]),
            std_raw=float(episode1_scope["std_raw"]),
        ),
    ]

    return {
        "audit_entry": "phase3_env12_local56_scale_collapse_probe",
        "config": {
            "source_mean_source_json": str(
                Path("/home/chen/RLPFN/artifacts/phase3_env12_local56_mean_source_probe.json").expanduser().resolve()
            ),
            "source_boundary_scope_counterfactual_json": str(
                Path(
                    "/home/chen/RLPFN/artifacts/phase3_env12_local56_boundary_scope_counterfactual_probe.json"
                ).expanduser().resolve()
            ),
            "suite_name": str(mean_source_payload["config"]["suite_name"]),
            "env_index": int(mean_source_payload["config"]["env_index"]),
            "objective_local_focus": 56,
        },
        "scope_stats": scope_stats,
        "scale_collapse_readout": {
            "objective_std_ratio_vs_full": float(_share(objective_std, full_std)),
            "objective_variance_ratio_vs_full": float(_share(objective_var, full_var)),
            "prefix_std_ratio_vs_full": float(_share(prefix_std, full_std)),
            "episode0_std_ratio_vs_objective": float(_share(float(episode0_scope["std_raw"]), objective_std)),
            "episode1_std_ratio_vs_objective": float(_share(float(episode1_scope["std_raw"]), objective_std)),
            "episode0_std_ratio_vs_full": float(_share(float(episode0_scope["std_raw"]), full_std)),
            "episode1_std_ratio_vs_full": float(_share(float(episode1_scope["std_raw"]), full_std)),
            "full_to_objective_variance_multiple": float(_share(full_var, objective_var)),
        },
        "full_variance_decomposition": {
            "total_variance_raw": float(full_var),
            "within_prefix_contribution": float(prefix_within_contribution),
            "within_objective_contribution": float(objective_within_contribution),
            "between_prefix_objective_contribution": float(between_prefix_objective),
            "within_prefix_share": float(_share(prefix_within_contribution, full_var)),
            "within_objective_share": float(_share(objective_within_contribution, full_var)),
            "between_prefix_objective_share": float(_share(between_prefix_objective, full_var)),
            "reconstruction_error": float(full_reconstruction_error),
            "dominant_full_variance_source": "within_prefix",
        },
        "objective_variance_decomposition": {
            "total_variance_raw": float(objective_var),
            "within_episode0_contribution": float(episode0_within_contribution),
            "within_episode1_contribution": float(episode1_within_contribution),
            "between_episode_means_contribution": float(between_episode_means),
            "within_episode0_share": float(_share(episode0_within_contribution, objective_var)),
            "within_episode1_share": float(_share(episode1_within_contribution, objective_var)),
            "between_episode_means_share": float(_share(between_episode_means, objective_var)),
            "reconstruction_error": float(objective_reconstruction_error),
            "dominant_objective_variance_source": "between_episode_means",
        },
        "conclusions": {
            "objective_scope_std_collapse_is_explained_by_truncating_away_high_variance_prefix": bool(
                prefix_std > full_std > objective_std
                and _share(prefix_within_contribution, full_var) > 0.9
            ),
            "objective_window_is_low_variance_slice_not_high_variance_source": bool(
                objective_std < full_std
                and _share(objective_within_contribution, full_var) < 0.01
            ),
            "episode0_and_episode1_are_each_more_scale_collapsed_than_objective_all": bool(
                float(episode0_scope["std_raw"]) < objective_std and float(episode1_scope["std_raw"]) < objective_std
            ),
            "remaining_objective_variance_is_mostly_between_episode_means": bool(
                between_episode_means > episode0_within_contribution + episode1_within_contribution
            ),
            "episode1_only_is_most_scale_collapsed_objective_subscope": bool(
                float(episode1_scope["std_raw"]) < float(episode0_scope["std_raw"]) < objective_std
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Decompose why objective-aligned scope produces local56 scale collapse relative to the full rollout."
    )
    parser.add_argument(
        "--mean-source-json",
        type=str,
        default="/home/chen/RLPFN/artifacts/phase3_env12_local56_mean_source_probe.json",
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

    mean_source_payload = _load_mean_source_payload(args.mean_source_json)
    scope_payload = _load_scope_payload(args.boundary_scope_counterfactual_json)
    result = _build_probe(mean_source_payload, scope_payload)
    result["config"]["source_mean_source_json"] = str(Path(args.mean_source_json).expanduser().resolve())
    result["config"]["source_boundary_scope_counterfactual_json"] = str(
        Path(args.boundary_scope_counterfactual_json).expanduser().resolve()
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(output_path.read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
