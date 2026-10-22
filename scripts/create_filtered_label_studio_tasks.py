"""Create a Label Studio task file that omits high-confidence common-tit predictions.

The source Label Studio project is never modified. Existing annotated images are
left out of the new task file, so annotations remain safely in the original
project while the new project contains only work that still needs labelling.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

import torch  # noqa: E402
from PIL import Image, UnidentifiedImageError  # noqa: E402
from torchvision import transforms  # noqa: E402

from bird_classifier.models.ConVNeXt_tiny import create_convnext_tiny  # noqa: E402


DEFAULT_MODEL = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "synthetic_all_species_with_no_bird_convnext_experiment"
    / "model.pt"
)
IMAGE_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def image_name_from_url(url: str) -> str:
    """Return the local filename from a Label Studio local-file URL."""
    parsed = urlparse(url)
    relative_path = parse_qs(parsed.query).get("d", [""])[0]
    if not relative_path:
        raise ValueError(f"The Label Studio image URL has no local-file path: {url}")
    return Path(unquote(relative_path)).name


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True, help="Folder containing the images.")
    parser.add_argument(
        "--tasks",
        type=Path,
        required=True,
        help="Original shuffled Label Studio task JSON file.",
    )
    parser.add_argument(
        "--label-export",
        type=Path,
        required=True,
        help="Fresh CSV export of the existing project; rows with a choice are preserved there.",
    )
    parser.add_argument("--output", type=Path, required=True, help="Filtered task JSON to import into a new project.")
    parser.add_argument(
        "--audit-csv",
        type=Path,
        required=True,
        help="CSV listing every high-confidence image excluded as a common tit.",
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.95,
        help="Only omit a target class at or above this probability (default: 0.95).",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    return parser.parse_args()


def load_labelled_filenames(export_path: Path) -> set[str]:
    with export_path.open(newline="", encoding="utf-8-sig") as source:
        rows = csv.DictReader(source)
        required = {"choice", "image"}
        if not rows.fieldnames or not required.issubset(rows.fieldnames):
            raise ValueError(f"Expected columns {sorted(required)} in {export_path}")
        return {
            image_name_from_url(row["image"])
            for row in rows
            if (row.get("choice") or "").strip()
        }


def load_model(model_path: Path) -> tuple[torch.nn.Module, tuple[str, ...], torch.device]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    class_names = tuple(checkpoint["class_names"])
    model = create_convnext_tiny(len(class_names), pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, class_names, device


def main() -> None:
    args = parse_arguments()
    if not 0 < args.threshold <= 1:
        raise ValueError("--threshold must be in (0, 1].")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive.")
    if not args.images.is_dir() or not args.model.is_file():
        raise FileNotFoundError("The image folder or model checkpoint does not exist.")

    tasks = json.loads(args.tasks.read_text(encoding="utf-8"))
    if not isinstance(tasks, list):
        raise ValueError("The task file must contain a JSON list.")
    labelled_filenames = load_labelled_filenames(args.label_export)
    remaining_tasks = []
    pending: list[tuple[dict, str]] = []
    for task in tasks:
        try:
            filename = image_name_from_url(task["data"]["image"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Invalid task entry: {task}") from error
        if filename not in labelled_filenames:
            pending.append((task, filename))

    model, class_names, device = load_model(args.model)
    transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    target_classes = {"Blaumeise", "Kohlmeise"}
    excluded_rows: list[dict[str, str]] = []
    unreadable = 0
    print(
        f"Using {device.type.upper()} on {len(pending):,} unlabelled tasks; "
        f"{len(labelled_filenames):,} annotated tasks will remain only in the original project."
    )

    for start in range(0, len(pending), args.batch_size):
        batch_entries = pending[start : start + args.batch_size]
        images: list[torch.Tensor] = []
        readable_entries: list[tuple[dict, str]] = []
        for task, filename in batch_entries:
            try:
                with Image.open(args.images / filename) as image:
                    images.append(transform(image.convert("RGB")))
                readable_entries.append((task, filename))
            except (FileNotFoundError, UnidentifiedImageError, OSError):
                # Preserve undecodable images for manual review instead of discarding them.
                remaining_tasks.append(task)
                unreadable += 1
        if images:
            batch = torch.stack(images).to(device)
            with torch.inference_mode(), torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"
            ):
                probabilities = model(batch).softmax(dim=1)
            scores, indices = probabilities.max(dim=1)
            for (task, filename), score, index in zip(readable_entries, scores, indices):
                prediction = class_names[int(index)]
                confidence = float(score)
                if prediction in target_classes and confidence >= args.threshold:
                    excluded_rows.append(
                        {
                            "filename": filename,
                            "prediction": prediction,
                            "confidence": f"{confidence:.6f}",
                        }
                    )
                else:
                    remaining_tasks.append(task)
        completed = min(start + args.batch_size, len(pending))
        print(f"Processed {completed:,}/{len(pending):,}", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(remaining_tasks, indent=2), encoding="utf-8")
    args.audit_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.audit_csv.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=("filename", "prediction", "confidence"))
        writer.writeheader()
        writer.writerows(excluded_rows)
    print(
        f"Done: kept {len(remaining_tasks):,} tasks, excluded {len(excluded_rows):,} high-confidence "
        f"Blaumeise/Kohlmeise tasks, preserved {len(labelled_filenames):,} already-labelled tasks, "
        f"and kept {unreadable:,} unreadable tasks for manual review."
    )


if __name__ == "__main__":
    main()
