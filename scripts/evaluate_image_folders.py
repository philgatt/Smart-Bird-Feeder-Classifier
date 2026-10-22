"""Evaluate a ConvNeXt or MobileNet checkpoint against labelled image folders."""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

import torch  # noqa: E402
from PIL import Image  # noqa: E402
from torchvision import transforms  # noqa: E402

from bird_classifier.models.ConVNeXt_tiny import create_convnext_tiny  # noqa: E402
from bird_classifier.models.mobilenet_v3_large import create_mobilenet_v3_large  # noqa: E402


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
IMAGE_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def parse_folder(value: str) -> tuple[str, Path]:
    try:
        label, folder = value.split("=", maxsplit=1)
    except ValueError as error:
        raise argparse.ArgumentTypeError("Use LABEL=FOLDER for each image folder.") from error
    folder_path = Path(folder)
    if not label or not folder_path.is_dir():
        raise argparse.ArgumentTypeError(f"Invalid labelled folder: {value}")
    return label, folder_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="Classifier .pt checkpoint.")
    parser.add_argument(
        "--architecture",
        choices=("convnext-tiny", "mobilenet-v3-large"),
        default="convnext-tiny",
        help="Checkpoint architecture (default: convnext-tiny).",
    )
    parser.add_argument("folders", type=parse_folder, nargs="+", metavar="LABEL=FOLDER")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.model, map_location=device, weights_only=False)
    class_names = tuple(checkpoint["class_names"])
    if args.architecture == "mobilenet-v3-large":
        model = create_mobilenet_v3_large(len(class_names), pretrained=False).to(device)
    else:
        model = create_convnext_tiny(len(class_names), pretrained=False).to(device)
    mean = IMAGENET_MEAN
    standard_deviation = IMAGENET_STD
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean, standard_deviation),
        ]
    )

    total = 0
    correct = 0
    print(f"Model: {args.model.name}; device: {device.type.upper()}")
    print()
    for true_label, folder in args.folders:
        files = sorted(path for path in folder.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)
        predictions: Counter[str] = Counter()
        for path in files:
            with Image.open(path) as image:
                tensor = transform(image.convert("RGB")).unsqueeze(0).to(device)
            with torch.inference_mode(), torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"
            ):
                predicted_index = int(model(tensor).argmax(dim=1).item())
            prediction = class_names[predicted_index]
            predictions[prediction] += 1
            total += 1
            correct += prediction == true_label

        hits = predictions[true_label]
        recall = hits / len(files) if files else 0.0
        print(f"{true_label}: {hits}/{len(files)} correct ({recall:.1%})")
        print("  " + ", ".join(f"{label}: {count}" for label, count in predictions.most_common()))

    print()
    print(f"Combined accuracy: {correct}/{total} ({correct / total:.1%})" if total else "No images found.")


if __name__ == "__main__":
    main()
