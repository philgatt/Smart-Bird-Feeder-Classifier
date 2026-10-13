"""Create an interim image set with exact duplicates removed."""

from __future__ import annotations

from pathlib import Path

from bird_classifier.data.preprocessing import copy_unique_images


PROJECT_ROOT = Path(__file__).resolve().parents[1]


if __name__ == "__main__":
    copy_unique_images(
        input_dir=PROJECT_ROOT / "data" / "raw" / "images",
        output_dir=PROJECT_ROOT / "data" / "interim" / "deduplicated_images",
        report_path=PROJECT_ROOT / "data" / "interim" / "duplicate_report.csv",
    )
