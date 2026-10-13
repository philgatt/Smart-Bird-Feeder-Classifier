"""Prepare Label Studio exports for bird-classifier training."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Final

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


LABEL_STUDIO_PREFIX: Final = re.compile(r"^[0-9a-fA-F]{8}-")
MANUAL_LABEL_CORRECTIONS: Final = {"128.jpg": "Sumpfmeise"}


def original_image_filename(label_studio_path: str) -> str:
    """Return the original filename from a Label Studio upload path."""
    filename = Path(label_studio_path).name
    return LABEL_STUDIO_PREFIX.sub("", filename)


def load_clean_labels(
    labels_path: Path,
    *,
    excluded_labels: tuple[str, ...] = ("Tannenmeise",),
) -> pd.DataFrame:
    """Load a Label Studio CSV and retain clean filenames and bird labels."""
    labels = pd.read_csv(labels_path)
    required_columns = {"image", "choice"}
    missing_columns = required_columns.difference(labels.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Label export is missing required column(s): {missing}")

    labels = labels.loc[:, ["image", "choice"]].copy()
    labels["image"] = labels["image"].map(original_image_filename)

    for image_name, choice in MANUAL_LABEL_CORRECTIONS.items():
        labels.loc[labels["image"] == image_name, "choice"] = choice

    labels = labels.dropna(subset=["image", "choice"])
    labels["choice"] = labels["choice"].astype(str).str.strip()
    labels = labels.loc[labels["choice"] != ""]
    labels = labels.loc[~labels["choice"].isin(excluded_labels)].copy()

    if labels.empty:
        raise ValueError("No labelled images remain after cleaning the Label Studio export.")
    if labels["image"].duplicated().any():
        duplicates = labels.loc[labels["image"].duplicated(), "image"].tolist()
        raise ValueError(f"Each image must have one label; duplicate image rows: {duplicates}")

    return labels


def encode_and_split_labels(
    labels: pd.DataFrame,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, LabelEncoder]:
    """Add encoded targets and a reproducible, stratified train/test split."""
    labels = labels.copy()
    encoder = LabelEncoder()
    labels["choice_encoded"] = encoder.fit_transform(labels["choice"])

    train_data, test_data = train_test_split(
        labels,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
        stratify=labels["choice"],
    )

    labels["split"] = pd.NA
    labels.loc[train_data.index, "split"] = "train"
    labels.loc[test_data.index, "split"] = "test"
    return labels, encoder


def prepare_training_labels(
    labels_path: Path,
    output_path: Path,
    encoder_path: Path,
    *,
    excluded_labels: tuple[str, ...] = ("Tannenmeise",),
    overwrite: bool = False,
) -> pd.DataFrame:
    """Clean, encode, split, and save a Label Studio export for training."""
    existing_outputs = [path for path in (output_path, encoder_path) if path.exists()]
    if existing_outputs and not overwrite:
        output_names = ", ".join(str(path) for path in existing_outputs)
        raise FileExistsError(
            f"Refusing to overwrite existing output(s): {output_names}. "
            "Pass overwrite=True to replace them."
        )

    labels = load_clean_labels(labels_path, excluded_labels=excluded_labels)
    labels, encoder = encode_and_split_labels(labels)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    labels.to_csv(output_path, index=False)

    encoder_path.parent.mkdir(parents=True, exist_ok=True)
    import joblib

    joblib.dump(encoder, encoder_path)
    return labels
