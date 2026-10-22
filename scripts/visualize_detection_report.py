"""Create detection charts from batch_predict_media_folder.py JSON output.

Example
-------
python scripts/visualize_detection_report.py F:/BirdCamera/recordings/bird_detection_report.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path, help="JSON report created by batch_predict_media_folder.py.")
    parser.add_argument(
        "--output-dir", type=Path, help="Directory for PNG charts (defaults beside the report)."
    )
    return parser.parse_args()


def save_detection_counts(report: dict[str, object], output_dir: Path) -> None:
    class_names = [name for name in report["class_names"] if name != "No Bird"]
    counts = report["counts"]
    values = [counts.get(name, 0) for name in class_names]
    figure, axis = plt.subplots(figsize=(max(9, len(class_names) * 0.8), 6))
    axis.bar(class_names, values, color="#4C78A8")
    axis.set(title="Detected bird species", xlabel="Species", ylabel="Detected files")
    axis.tick_params(axis="x", rotation=45, labelsize=10)
    for index, value in enumerate(values):
        axis.text(index, value, str(value), ha="center", va="bottom")
    figure.tight_layout()
    figure.savefig(output_dir / "detections_by_species.png", dpi=180)
    plt.close(figure)


def save_stacked_chart(
    grouped_counts: dict[str, Counter[str]],
    categories: list[str],
    title: str,
    xlabel: str,
    output: Path,
) -> None:
    species = sorted({label for counts in grouped_counts.values() for label in counts})
    if not species:
        return
    figure, axis = plt.subplots(figsize=(max(9, len(categories) * 1.15), 6))
    bottom = [0] * len(categories)
    for label in species:
        values = [grouped_counts[category][label] for category in categories]
        axis.bar(categories, values, bottom=bottom, label=label)
        bottom = [current + value for current, value in zip(bottom, values)]
    axis.set(title=title, xlabel=xlabel, ylabel="Detected files")
    axis.legend(title="Species", bbox_to_anchor=(1.02, 1), loc="upper left")
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def save_time_charts(report: dict[str, object], output_dir: Path) -> None:
    by_day_period: defaultdict[str, Counter[str]] = defaultdict(Counter)
    by_month: defaultdict[str, Counter[str]] = defaultdict(Counter)
    periods = ["Night (00–05)", "Morning (06–11)", "Afternoon (12–17)", "Evening (18–23)"]
    month_labels = [datetime(2000, month, 1).strftime("%b") for month in range(1, 13)]
    for result in report["results"]:
        label = result["label"]
        if label == "No Bird":
            continue
        timestamp = datetime.fromisoformat(result["timestamp"])
        period = periods[min(timestamp.hour // 6, 3)]
        by_day_period[period][label] += 1
        by_month[month_labels[timestamp.month - 1]][label] += 1
    save_stacked_chart(
        by_day_period, periods, "Bird detections by time of day", "Time of day",
        output_dir / "detections_by_time_of_day.png",
    )
    save_stacked_chart(
        by_month, month_labels, "Bird detections by month", "Month",
        output_dir / "detections_by_month.png",
    )


def main() -> None:
    arguments = parse_arguments()
    report_path = arguments.report.resolve()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    output_dir = (arguments.output_dir or report_path.parent / "visualizations").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    save_detection_counts(report, output_dir)
    save_time_charts(report, output_dir)
    print(f"Saved visualizations to: {output_dir}")


if __name__ == "__main__":
    main()
