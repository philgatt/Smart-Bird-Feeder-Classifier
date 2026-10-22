"""Classify every supported image or video in a folder and write a JSON report.

Examples
--------
python scripts/batch_predict_media_folder.py F:/BirdCamera/recordings
python scripts/batch_predict_media_folder.py F:/BirdCamera/recordings --recursive --model model.pt
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from time import perf_counter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from torchvision import transforms  # noqa: E402

from bird_classifier.models.ConVNeXt_tiny import create_convnext_tiny  # noqa: E402
from bird_classifier.models.mobilenet_v3_large import create_mobilenet_v3_large  # noqa: E402

IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}
VIDEO_EXTENSIONS = {".avi", ".mkv", ".mov", ".mp4", ".wmv"}
SUPPORTED_EXTENSIONS = IMAGE_EXTENSIONS | VIDEO_EXTENSIONS
IMAGE_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
DEFAULT_MODEL_PATH = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "synthetic_all_species_with_no_bird_convnext_cross_validation"
    / "model.pt"
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path, help="Folder containing images and/or videos.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH, help="Model checkpoint.")
    parser.add_argument(
        "--output",
        type=Path,
        help="JSON report path (default: <folder>/bird_detection_report.json).",
    )
    parser.add_argument("--recursive", action="store_true", help="Include files in subfolders.")
    parser.add_argument(
        "--every-n-frames", type=int, default=3, help="Classify every Nth video frame (default: 3)."
    )
    parser.add_argument(
        "--min-species-seconds",
        type=float,
        default=3.0,
        help="Seconds of confident species evidence required for a video (default: 3).",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.35,
        help="Minimum species confidence for video evidence (default: 0.35).",
    )
    return parser.parse_args()


def checkpoint_architecture(checkpoint: dict[str, object]) -> str:
    architecture = checkpoint.get("architecture")
    if isinstance(architecture, str):
        return architecture
    state_dict = checkpoint.get("model_state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError("Checkpoint does not contain model_state_dict")
    if "classifier.3.weight" in state_dict:
        return "mobilenet-v3-large"
    if "classifier.2.weight" in state_dict:
        return "convnext-tiny"
    raise ValueError("Could not identify the checkpoint architecture")


def load_model(model_path: Path) -> tuple[torch.nn.Module, tuple[str, ...], torch.device, str]:
    if not model_path.is_file():
        raise FileNotFoundError(f"Model checkpoint was not found: {model_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    class_names = tuple(checkpoint["class_names"])
    architecture = checkpoint_architecture(checkpoint)
    if architecture == "convnext-tiny":
        model = create_convnext_tiny(len(class_names), pretrained=False)
    elif architecture == "mobilenet-v3-large":
        model = create_mobilenet_v3_large(len(class_names), pretrained=False)
    else:
        raise ValueError(f"Unsupported checkpoint architecture: {architecture}")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    return model, class_names, device, architecture


def timestamp_for(path: Path) -> tuple[datetime, str]:
    """Use a camera timestamp in the filename when available, otherwise file modification time."""
    pattern = r"(20\d{2})[-_/]?(\d{2})[-_/]?(\d{2})[^\d]?(\d{2})[:_-]?(\d{2})[:_-]?(\d{2})"
    match = re.search(pattern, path.stem)
    if match:
        try:
            return datetime(*map(int, match.groups())), "filename"
        except ValueError:
            pass
    return datetime.fromtimestamp(path.stat().st_mtime), "file_modified_time"


def predict_probabilities(
    frame: np.ndarray,
    model: torch.nn.Module,
    device: torch.device,
    transform: transforms.Compose,
) -> np.ndarray:
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    tensor = transform(Image.fromarray(rgb_frame)).unsqueeze(0).to(device)
    with torch.inference_mode(), torch.autocast(
        device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"
    ):
        return model(tensor).softmax(dim=1)[0].cpu().numpy()


def classify_image(
    path: Path,
    model: torch.nn.Module,
    class_names: tuple[str, ...],
    device: torch.device,
    transform: transforms.Compose,
) -> tuple[str, float, float | None]:
    frame = cv2.imread(str(path))
    if frame is None:
        raise ValueError("OpenCV could not read image")
    probabilities = predict_probabilities(frame, model, device, transform)
    index = int(probabilities.argmax())
    return class_names[index], float(probabilities[index]), None


def classify_video(
    path: Path,
    model: torch.nn.Module,
    class_names: tuple[str, ...],
    device: torch.device,
    transform: transforms.Compose,
    every_n_frames: int,
    min_species_seconds: float,
    min_confidence: float,
) -> tuple[str, float, float | None]:
    video = cv2.VideoCapture(str(path))
    if not video.isOpened():
        video.release()
        raise ValueError("OpenCV could not open video")
    fps = video.get(cv2.CAP_PROP_FPS) or 30.0
    seconds_per_sample = every_n_frames / fps
    no_bird_index = class_names.index("No Bird") if "No Bird" in class_names else None
    confident_seconds: Counter[str] = Counter()
    confidence_sums: Counter[str] = Counter()
    sample_count = 0
    try:
        frame_index = 0
        while True:
            success, frame = video.read()
            if not success:
                break
            if frame_index % every_n_frames == 0:
                probabilities = predict_probabilities(frame, model, device, transform)
                index = int(probabilities.argmax())
                confidence = float(probabilities[index])
                if index != no_bird_index and confidence >= min_confidence:
                    label = class_names[index]
                    confident_seconds[label] += seconds_per_sample
                    confidence_sums[label] += confidence * seconds_per_sample
                sample_count += 1
            frame_index += 1
    finally:
        video.release()
    eligible = [
        label for label, seconds in confident_seconds.items() if seconds >= min_species_seconds
    ]
    if eligible:
        label = max(
            eligible,
            key=lambda name: (
                confident_seconds[name],
                confidence_sums[name] / confident_seconds[name],
            ),
        )
        confidence = confidence_sums[label] / confident_seconds[label]
        return label, confidence, sample_count * seconds_per_sample
    return "No Bird", 0.0, sample_count * seconds_per_sample


def main() -> None:
    arguments = parse_arguments()
    if arguments.every_n_frames < 1:
        raise ValueError("--every-n-frames must be at least 1")
    if not 0 <= arguments.min_confidence <= 1:
        raise ValueError("--min-confidence must be between 0 and 1")
    folder = arguments.folder.resolve()
    if not folder.is_dir():
        raise NotADirectoryError(f"Folder was not found: {folder}")
    output_path = (arguments.output or folder / "bird_detection_report.json").resolve()
    paths = sorted(
        path for path in (folder.rglob("*") if arguments.recursive else folder.iterdir())
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if not paths:
        raise ValueError("No supported images or videos were found")

    model_path = arguments.model.resolve()
    model, class_names, device, architecture = load_model(model_path)
    transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    print(
        f"Using {architecture} on {device.type.upper()} for {len(paths):,} files.",
        flush=True,
    )
    print("Progress is shown every 25 files.", flush=True)
    results: list[dict[str, object]] = []
    counts: Counter[str] = Counter()
    errors: list[dict[str, str]] = []
    started_at = perf_counter()
    for number, path in enumerate(paths, start=1):
        if number == 1 or number % 25 == 0 or number == len(paths):
            elapsed_seconds = perf_counter() - started_at
            files_per_second = number / elapsed_seconds if elapsed_seconds else 0.0
            remaining_seconds = (
                (len(paths) - number) / files_per_second if files_per_second else 0.0
            )
            print(
                f"[{number:,}/{len(paths):,}] {number / len(paths):.1%} | "
                f"elapsed {elapsed_seconds / 60:.1f} min | "
                f"estimated remaining {remaining_seconds / 60:.1f} min | {path.name}",
                flush=True,
            )
        try:
            if path.suffix.lower() in IMAGE_EXTENSIONS:
                label, confidence, duration = classify_image(
                    path, model, class_names, device, transform
                )
                media_type = "image"
            else:
                label, confidence, duration = classify_video(
                    path, model, class_names, device, transform, arguments.every_n_frames,
                    arguments.min_species_seconds, arguments.min_confidence,
                )
                media_type = "video"
            timestamp, timestamp_source = timestamp_for(path)
            counts[label] += 1
            results.append(
                {
                    "path": str(path), "media_type": media_type, "label": label,
                    "confidence": round(confidence, 6), "duration_seconds": duration,
                    "timestamp": timestamp.isoformat(), "timestamp_source": timestamp_source,
                }
            )
        except Exception as error:  # keep processing the rest of the folder
            errors.append({"path": str(path), "error": str(error)})
            print(f"  Could not read {path.name}: {error}", flush=True)

    report = {
        "format_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "source_folder": str(folder),
        "model_path": str(model_path),
        "architecture": architecture,
        "class_names": class_names,
        "settings": {
            "every_n_frames": arguments.every_n_frames,
            "min_species_seconds": arguments.min_species_seconds,
            "min_confidence": arguments.min_confidence,
        },
        "counts": dict(sorted(counts.items())),
        "results": results,
        "errors": errors,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved report: {output_path}", flush=True)
    for label in class_names:
        print(f"{label}: {counts[label]}", flush=True)
    if errors:
        print(f"Unreadable files: {len(errors)}", flush=True)


if __name__ == "__main__":
    main()
