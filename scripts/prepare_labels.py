"""Create the training-label table from a Label Studio export."""

from __future__ import annotations

import argparse
from pathlib import Path

from bird_classifier.data.label_processing import prepare_training_labels


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--labels-path",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "labels" / "labels_raw.csv",
        help="Label Studio CSV export to process.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "training_labels.csv",
        help="Destination for the cleaned, encoded CSV.",
    )
    parser.add_argument(
        "--encoder-path",
        type=Path,
        default=PROJECT_ROOT / "models" / "label_encoder.joblib",
        help="Destination for the fitted LabelEncoder.",
    )
    parser.add_argument(
        "--exclude-label",
        action="append",
        default=["Tannenmeise"],
        help="Bird label to leave out. May be supplied more than once.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing processed CSV or encoder file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    labels = prepare_training_labels(
        args.labels_path,
        args.output_path,
        args.encoder_path,
        excluded_labels=tuple(args.exclude_label),
        overwrite=args.overwrite,
    )
    print(f"Saved {len(labels)} labelled images to {args.output_path}")
    print(labels["split"].value_counts().to_string())
    print("\nClass counts:\n" + labels["choice"].value_counts().to_string())


if __name__ == "__main__":
    main()
