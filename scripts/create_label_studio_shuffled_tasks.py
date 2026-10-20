"""Create a randomly ordered Label Studio task import for local image files.

The defaults match a Label Studio local-files root configured as ``F:/`` and
the bird image folder ``F:/video vogel``.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

DEFAULT_IMAGE_DIRECTORY = Path("F:/video vogel")
DEFAULT_LOCAL_FILES_PATH = "video vogel"
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIRECTORY)
    parser.add_argument(
        "--local-files-path",
        default=DEFAULT_LOCAL_FILES_PATH,
        help="Path relative to LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output JSON file (default: shuffled_tasks.json inside --image-dir).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Optional seed for a repeatable order. Omit for a fresh random order.",
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    image_directory = arguments.image_dir.resolve()
    if not image_directory.is_dir():
        raise NotADirectoryError(f"Image folder not found: {image_directory}")
    output_path = (arguments.output or image_directory / "shuffled_tasks.json").resolve()
    images = [
        path.name
        for path in image_directory.iterdir()
        if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS
    ]
    random.Random(arguments.seed).shuffle(images)
    local_files_path = arguments.local_files_path.strip("/\\")
    tasks = [
        {"data": {"image": f"/data/local-files/?d={local_files_path}/{image_name}"}}
        for image_name in images
    ]
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(tasks, handle, indent=2)
    print(f"Generated {len(tasks):,} randomly ordered Label Studio tasks: {output_path}")


if __name__ == "__main__":
    main()
