from __future__ import annotations

import csv
import hashlib
import json
import random
from pathlib import Path


BASE_SEED = 20260330
ROOT = Path(__file__).resolve().parent
INPUT_PATH = ROOT / "input_table.csv"
SAMPLES_PATH = ROOT / "random3_samples.csv"
COLUMN_PATH = ROOT / "random3_column.csv"
LATEX_PATH = ROOT / "random3_column.tex"
SUMMARY_PATH = ROOT / "random3_summary.json"


def dataset_seed(dataset: str) -> int:
    digest = hashlib.sha256(f"{BASE_SEED}:{dataset}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def main() -> None:
    with INPUT_PATH.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        method_names = [name for name in reader.fieldnames if name != "Dataset"]
        rows = list(reader)

    sample_rows: list[dict[str, str]] = []
    column_rows: list[dict[str, str]] = []

    for row in rows:
        dataset = row["Dataset"]
        available = [
            (method, float(row[method]))
            for method in method_names
            if row[method] not in {"", "N/A"}
        ]
        if len(available) < 3:
            raise ValueError(f"{dataset} has fewer than 3 available methods.")

        rng = random.Random(dataset_seed(dataset))
        chosen = rng.sample(available, 3)
        average = sum(value for _, value in chosen) / 3.0
        average_display = f"{average:.3f}"

        sample_rows.append(
            {
                "Dataset": dataset,
                "Method1": chosen[0][0],
                "Value1": f"{chosen[0][1]:.3f}",
                "Method2": chosen[1][0],
                "Value2": f"{chosen[1][1]:.3f}",
                "Method3": chosen[2][0],
                "Value3": f"{chosen[2][1]:.3f}",
                "AverageRaw": f"{average:.6f}",
                "AverageDisplay": average_display,
            }
        )
        column_rows.append({"Dataset": dataset, "Random3Avg": average_display})

    with SAMPLES_PATH.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "Dataset",
                "Method1",
                "Value1",
                "Method2",
                "Value2",
                "Method3",
                "Value3",
                "AverageRaw",
                "AverageDisplay",
            ],
        )
        writer.writeheader()
        writer.writerows(sample_rows)

    with COLUMN_PATH.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["Dataset", "Random3Avg"])
        writer.writeheader()
        writer.writerows(column_rows)

    latex_lines = [
        f"% Generated from input_table.csv with base seed {BASE_SEED}",
        "% Suggested header: \\textbf{Random-3 Avg}",
    ]
    latex_lines.extend(
        f"{row['Dataset']} & {row['Random3Avg']} \\\\"
        for row in column_rows
    )
    LATEX_PATH.write_text("\n".join(latex_lines) + "\n", encoding="utf-8")

    summary = {
        "base_seed": BASE_SEED,
        "num_datasets": len(column_rows),
        "input_file": str(INPUT_PATH),
        "sample_file": str(SAMPLES_PATH),
        "column_file": str(COLUMN_PATH),
        "latex_file": str(LATEX_PATH),
    }
    SUMMARY_PATH.write_text(
        json.dumps(summary, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
