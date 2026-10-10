"""Extract three reproducible random frames from each raw video."""

from __future__ import annotations

import argparse
from pathlib import Path

from bird_classifier.data.extract_frames import extract_random_frames


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "videos",
        help="Directory containing source videos (default: data/raw/videos).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "interim" / "candidate_images",
        help="Directory for extracted JPEGs and the frame manifest.",
    )
    parser.add_argument("--frames-per-video", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true", help="Replace existing output files.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    extracted = extract_random_frames(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        frames_per_video=args.frames_per_video,
        seed=args.seed,
        overwrite=args.overwrite,
    )
    print(f"Extracted {len(extracted)} frames from {len({item.source_video for item in extracted})} videos.")
