"""Prepare raw image data for labeling and model training."""

from __future__ import annotations

import csv
import hashlib
import shutil
from pathlib import Path
from typing import Final
import torch
from bird_classifier.data.dataloader import BirdDataset, create_dataloader
import pandas as pd
import numpy as np
import os
import json


IMAGE_EXTENSIONS: Final = {".jpg", ".jpeg", ".png", ".webp"}
REPORT_FIELDS: Final = ("source_image", "sha256", "kept_image", "is_duplicate")


def file_hash(path: Path) -> str:
    """Return a SHA-256 hash for a file's contents."""
    hasher = hashlib.sha256()

    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            hasher.update(chunk)

    return hasher.hexdigest()


def find_image_files(input_dir: Path) -> list[Path]:
    """Return supported image files below ``input_dir`` in a stable order."""
    return sorted(
        (path for path in input_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS),
        key=lambda path: path.as_posix().casefold(),
    )


def copy_unique_images(input_dir: Path, output_dir: Path, report_path: Path) -> None:
    """Copy one image from each exact-duplicate group into an interim directory.

    Raw images are never changed. Every scanned source image is recorded in
    ``report_path`` and points to the retained file for its content-hash group.
    """
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {input_dir}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}. "
            "Choose an empty directory to avoid mixing runs."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    seen_hashes: dict[str, Path] = {}

    with report_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=REPORT_FIELDS, quoting=csv.QUOTE_ALL)
        writer.writeheader()

        for image_path in find_image_files(input_dir):
            relative_path = image_path.relative_to(input_dir)
            image_hash = file_hash(image_path)
            kept_path = seen_hashes.get(image_hash)
            is_duplicate = kept_path is not None

            if kept_path is None:
                image_number = len(seen_hashes) + 1
                kept_path = output_dir / f"{image_number}{image_path.suffix.lower()}"
                shutil.copy2(image_path, kept_path)
                seen_hashes[image_hash] = kept_path

            writer.writerow(
                {
                    "source_image": relative_path.as_posix(),
                    "sha256": image_hash,
                    "kept_image": kept_path.relative_to(output_dir).as_posix(),
                    "is_duplicate": is_duplicate,
                }
            )

def get_mean_std(
    dataset,
    statistics_path=Path("data/processed/image_statistics.json"),
    force_overwrite=False,
):
    statistics_path = Path(statistics_path)

    if statistics_path.is_file() and not force_overwrite:
        with statistics_path.open("r", encoding="utf-8") as file:
            statistics = json.load(file)

        mean = torch.tensor(statistics["mean"], dtype=torch.float32)
        std = torch.tensor(statistics["std"], dtype=torch.float32)

        return mean, std

    dataloader = create_dataloader(dataset, batch_size=32, shuffle=False)

    pixel_sum = torch.zeros(3)
    pixel_squared_sum = torch.zeros(3)
    pixel_count = 0

    for images, _ in dataloader:
        pixel_sum += images.sum(dim=(0, 2, 3))
        pixel_squared_sum += (images**2).sum(dim=(0, 2, 3))
        pixel_count += images.shape[0] * images.shape[2] * images.shape[3]

    mean = pixel_sum / pixel_count
    variance = (pixel_squared_sum / pixel_count - mean**2).clamp_min(0)
    std = variance.sqrt()

    statistics_path.parent.mkdir(parents=True, exist_ok=True)

    with statistics_path.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "mean": mean.tolist(),
                "std": std.tolist(),
            },
            file,
            indent=2,
        )

    return mean, std

def get_data_split(image_dir, labels_path, split) -> tuple[list[str], np.ndarray]:
    """Get the image paths and encoded labels for the specified split."""
    labels = pd.read_csv(labels_path) 

    split_data = labels.loc[labels["split"] == split]

    image_paths = [
        image_dir / image_filename
        for image_filename in split_data["image"]
    ]

    encoded_labels = split_data["choice_encoded"].to_numpy(dtype=np.int64)

    return image_paths, encoded_labels