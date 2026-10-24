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


UNCERTAIN_LABEL = "Uncertain / No Bird"
CONFIDENCE_THRESHOLD = 0.30
ENGLISH_SPECIES_NAMES = {
    "Blaumeise": "Blue tit",
    "Eichelhäher": "Eurasian jay",
    "Feldsperling": "Eurasian tree sparrow",
    "Kleiber": "Eurasian nuthatch",
    "Kohlmeise": "Great tit",
    "Rotkehlchen": "European robin",
    "Specht": "Woodpecker",
    "Sumpfmeise": "Marsh tit",
    "Tannenmeise": "Coal tit",
    "Taube": "Common wood pigeon",
}
SPECIES_COLORS = {
    "Great tit": "#1f77b4",
    "Blue tit": "#a6bddb",
    "Coal tit": "#ff7f0e",
    "European robin": "#fdbf6f",
    "Eurasian jay": "#2ca02c",
    "Eurasian nuthatch": "#98df8a",
    "Marsh tit": "#d62728",
    "Common wood pigeon": "#fb9a99",
    "Eurasian tree sparrow": "#9467bd",
    "Woodpecker": "#c5b0d5",
    UNCERTAIN_LABEL: "#6c757d",
}
SUBTITLE = (
    "Model predictions only. Model accuracy on held-out real images: 94.8%; "
    "counts may vary by species and conditions."
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path, help="JSON report created by batch_predict_media_folder.py.")
    parser.add_argument(
        "--output-dir", type=Path, help="Directory for PNG charts (defaults beside the report)."
    )
    return parser.parse_args()


def display_label(result: dict[str, object]) -> str:
    """Return the chart category for one saved prediction."""
    label = str(result["label"])
    confidence = float(result["confidence"])
    if label == "No Bird" or confidence < CONFIDENCE_THRESHOLD:
        return UNCERTAIN_LABEL
    return ENGLISH_SPECIES_NAMES.get(label, label)


def add_title(figure, title: str) -> None:
    figure.suptitle(title, fontsize=17, y=0.98)
    figure.text(0.5, 0.93, SUBTITLE, ha="center", fontsize=9, color="#444444")


def species_order(grouped_counts: dict[str, Counter[str]]) -> list[str]:
    totals: Counter[str] = Counter()
    for counts in grouped_counts.values():
        totals.update(counts)
    names = [name for name, _ in totals.most_common() if name != UNCERTAIN_LABEL]
    if totals[UNCERTAIN_LABEL]:
        names.append(UNCERTAIN_LABEL)
    return names


def save_detection_counts(report: dict[str, object], output_dir: Path) -> None:
    counts: Counter[str] = Counter(display_label(result) for result in report["results"])
    labels = [name for name, _ in counts.most_common() if name != UNCERTAIN_LABEL]
    if counts[UNCERTAIN_LABEL]:
        labels.append(UNCERTAIN_LABEL)
    values = [counts[name] for name in labels]
    figure, axis = plt.subplots(figsize=(12, 6.4))
    bars = axis.barh(labels, values, color=[SPECIES_COLORS.get(name, "#4C78A8") for name in labels])
    axis.invert_yaxis()
    axis.set(xlabel="Number of images where the AI detected this species")
    axis.grid(axis="x", alpha=0.45)
    axis.set_axisbelow(True)
    for bar, value in zip(bars, values):
        axis.text(value, bar.get_y() + bar.get_height() / 2, f" {value:,}", va="center")
    add_title(figure, "How often did the AI detect each bird species?")
    figure.tight_layout(rect=(0, 0, 1, 0.88))
    figure.savefig(output_dir / "detections_by_species.png", dpi=180)
    plt.close(figure)


def save_stacked_chart(
    grouped_counts: dict[str, Counter[str]],
    categories: list[str],
    title: str,
    xlabel: str,
    output: Path,
    *,
    rotate_labels: bool = False,
    label_stride: int = 1,
) -> None:
    species = species_order(grouped_counts)
    if not species:
        return
    figure, axis = plt.subplots(figsize=(max(13, min(18, len(categories) * 0.32)), 6.8))
    bottom = [0] * len(categories)
    for label in species:
        values = [grouped_counts[category][label] for category in categories]
        axis.bar(categories, values, bottom=bottom, label=label, color=SPECIES_COLORS.get(label))
        bottom = [current + value for current, value in zip(bottom, values)]
    axis.set(xlabel=xlabel, ylabel="Number of images where the AI detected a bird")
    axis.grid(axis="y", alpha=0.45)
    axis.set_axisbelow(True)
    if rotate_labels:
        axis.tick_params(axis="x", rotation=45, labelsize=8)
    if label_stride > 1:
        axis.set_xticks(range(0, len(categories), label_stride), categories[::label_stride])
    axis.legend(title="Predicted species", bbox_to_anchor=(1.02, 1), loc="upper left")
    add_title(figure, title)
    figure.tight_layout(rect=(0, 0, 0.84, 0.88))
    figure.savefig(output, dpi=180)
    plt.close(figure)


def save_time_charts(report: dict[str, object], output_dir: Path) -> None:
    by_hour: defaultdict[str, Counter[str]] = defaultdict(Counter)
    by_month: defaultdict[str, Counter[str]] = defaultdict(Counter)
    by_day: defaultdict[str, Counter[str]] = defaultdict(Counter)
    timestamps = [datetime.fromisoformat(result["timestamp"]) for result in report["results"]]
    season_start_year = min(timestamp.year for timestamp in timestamps)

    def is_in_season(timestamp: datetime) -> bool:
        return (
            timestamp.year == season_start_year and timestamp.month >= 10
        ) or (
            timestamp.year == season_start_year + 1 and timestamp.month <= 4
        )

    for result in report["results"]:
        label = display_label(result)
        timestamp = datetime.fromisoformat(result["timestamp"])
        if not is_in_season(timestamp):
            continue
        by_hour[f"{timestamp.hour:02d}:00"][label] += 1
        by_month[timestamp.strftime("%Y-%m")][label] += 1
        by_day[timestamp.strftime("%Y-%m-%d")][label] += 1
    month_labels = sorted(by_month)
    day_labels = sorted(by_day)
    save_stacked_chart(
        by_hour,
        [f"{hour:02d}:00" for hour in range(24)],
        "Bird detections by time of day",
        "Time of day",
        output_dir / "detections_by_time_of_day.png",
        rotate_labels=True,
    )
    save_stacked_chart(
        by_month,
        month_labels,
        "Bird detections by month",
        "Month",
        output_dir / "detections_by_month.png",
        rotate_labels=True,
    )
    save_stacked_chart(
        by_day,
        day_labels,
        "Bird detections by day",
        "Date",
        output_dir / "detections_by_day.png",
        rotate_labels=True,
        label_stride=7,
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
