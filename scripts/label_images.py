"""Launch the local UI for assigning labels to bird images."""

import argparse
from pathlib import Path

from bird_classifier.data.label_images import launch


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--image-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "images",
        help="Directory containing images to label.",
    )
    parser.add_argument(
        "--labels-csv",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "labels" / "labels.csv",
        help="CSV file where labels are saved.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    launch(image_root=args.image_root, labels_csv=args.labels_csv)
