"""Train a ConvNeXt-Tiny or MobileNetV3-Large bird-camera classifier.

The script reserves 20% of each core real class as an untouched final test set.
For the extended class set, core classes train on real images only, while the
three rare classes train on synthetic images only and every corresponding real
image is held out for final testing.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from bird_classifier.models.ConVNeXt_tiny import (  # noqa: E402
    create_convnext_tiny,
    unfreeze_convnext_backbone,
)
from bird_classifier.models.mobilenet_v3_large import (  # noqa: E402
    create_mobilenet_v3_large,
    unfreeze_mobilenet_v3_large_backbone,
)

SYNTHETIC_ROOT = PROJECT_ROOT / "data" / "raw" / "images" / "labeled" / "synthetic"
REAL_ROOT = PROJECT_ROOT / "data" / "raw" / "images" / "labeled" / "real"
TORCH_CACHE_ROOT = PROJECT_ROOT / "models" / "torch_cache"
RESIZED_IMAGE_CACHE_ROOT = PROJECT_ROOT / "data" / "interim" / "training_image_cache_224"
RESIZED_IMAGE_CACHE_MARKER = RESIZED_IMAGE_CACHE_ROOT / ".complete"

# The core deployment classifier uses well-covered classes.  The extended set
# adds the three species with useful synthetic sets but comparatively little
# real data, so it remains an explicitly separate model choice.
CORE_CLASS_NAMES = (
    "Blaumeise",
    "Eichelhäher",
    "Feldsperling",
    "Kleiber",
    "Kohlmeise",
    "No Bird",
    "Rotkehlchen",
    "Specht",
    "Sumpfmeise",
    "Tannenmeise",
    "Taube",
)
EXTENDED_ONLY_CLASS_NAMES = ("Amsel", "Gruenfink", "Stieglitz")
ALL_REAL_CLASS_NAMES = (
    "Amsel",
    "Bergfink",
    "Blaumeise",
    "Buchfink",
    "Eichelhäher",
    "Erlenzeisig",
    "Feldsperling",
    "Gimpel",
    "Gruenfink",
    "Heckenbraunelle",
    "Kleiber",
    "Kohlmeise",
    "No Bird",
    "Rotkehlchen",
    "Specht",
    "Stieglitz",
    "Sumpfmeise",
    "Tannenmeise",
    "Taube",
)
CLASS_NAMES = CORE_CLASS_NAMES
TRAINING_DATA_MODE = "combined"
ARCHITECTURE = "convnext-tiny"
OUTPUT_ROOT = PROJECT_ROOT / "data" / "interim" / "combined_synthetic_real_convnext_tiny"
REAL_FOLDER_NAMES = {"No Bird": "NoBird"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
SEED = 42
REAL_TEST_FRACTION = 0.20
VALIDATION_FRACTION = 0.20
IMAGE_SIZE = 224
EPOCHS = int(os.environ.get("BIRD_COMBINED_EPOCHS", "30"))
BATCH_SIZE = int(os.environ.get("BIRD_COMBINED_BATCH_SIZE", "32"))
# On Windows, spawned loader workers can stall while opening camera images.
# The reusable 224px cache makes an in-process loader fast and reliable.
NUM_WORKERS = int(os.environ.get("BIRD_COMBINED_NUM_WORKERS", "0"))
CACHE_RESIZED_IMAGES = os.environ.get("BIRD_CACHE_RESIZED_IMAGES", "true").lower() not in {
    "0",
    "false",
    "no",
}
FREEZE_BACKBONE_EPOCHS = 5
EARLY_STOPPING_PATIENCE = 6
EARLY_STOPPING_MIN_DELTA = 0.002
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class Sample:
    path: Path
    label: int
    source: str


class BirdDataset(Dataset):
    def __init__(self, samples: list[Sample], transform: transforms.Compose) -> None:
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[index]
        with Image.open(sample.path) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, torch.tensor(sample.label, dtype=torch.long)


def class_folder(class_name: str, source: str) -> str:
    """Map the display class name to each source directory's historical name."""
    return REAL_FOLDER_NAMES.get(class_name, class_name) if source == "real" else class_name


def collect_samples(root: Path, source: str) -> list[Sample]:
    samples: list[Sample] = []
    for label, class_name in enumerate(CLASS_NAMES):
        directory = root / class_folder(class_name, source)
        paths = sorted(path for path in directory.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)
        if not paths:
            raise RuntimeError(f"No usable {source} images found for {class_name}: {directory}")
        samples.extend(Sample(path, label, source) for path in paths)
    return samples


def split_by_class(
    samples: list[Sample], fraction: float, generator: random.Random
) -> tuple[list[Sample], list[Sample]]:
    first, second = [], []
    for label in range(len(CLASS_NAMES)):
        class_samples = [sample for sample in samples if sample.label == label]
        generator.shuffle(class_samples)
        second_count = max(1, round(len(class_samples) * fraction))
        second.extend(class_samples[:second_count])
        first.extend(class_samples[second_count:])
    return first, second


def split_real_samples_for_experiment(
    samples: list[Sample],
    synthetic_only_labels: set[int],
    fraction: float,
    generator: random.Random,
) -> tuple[list[Sample], list[Sample]]:
    """Reserve all real images for classes trained only from synthetic data."""
    training, test = [], []
    for label in range(len(CLASS_NAMES)):
        class_samples = [sample for sample in samples if sample.label == label]
        generator.shuffle(class_samples)
        if label in synthetic_only_labels:
            test.extend(class_samples)
            continue
        test_count = max(1, round(len(class_samples) * fraction))
        test.extend(class_samples[:test_count])
        training.extend(class_samples[test_count:])
    return training, test


def count_by_class(samples: list[Sample]) -> dict[str, int]:
    return {class_name: sum(sample.label == label for sample in samples) for label, class_name in enumerate(CLASS_NAMES)}


def count_by_source(samples: list[Sample]) -> dict[str, int]:
    return {source: sum(sample.source == source for sample in samples) for source in ("synthetic", "real")}


def cache_resized_samples(samples: list[Sample]) -> list[Sample]:
    """Cache 224px RGB images once so each epoch avoids full-frame JPEG decoding."""
    RESIZED_IMAGE_CACHE_ROOT.mkdir(parents=True, exist_ok=True)

    def cached_path(sample: Sample) -> Path:
        fingerprint = hashlib.sha1(str(sample.path).encode("utf-8")).hexdigest()
        return RESIZED_IMAGE_CACHE_ROOT / f"{sample.source}_{sample.label}_{fingerprint}.jpg"

    cache_signature = hashlib.sha1(
        "\n".join(str(sample.path) for sample in samples).encode("utf-8")
    ).hexdigest()
    if (
        RESIZED_IMAGE_CACHE_MARKER.exists()
        and RESIZED_IMAGE_CACHE_MARKER.read_text(encoding="utf-8").strip() == cache_signature
    ):
        return [Sample(cached_path(sample), sample.label, sample.source) for sample in samples]

    def build_cache_item(sample: Sample) -> None:
        target = cached_path(sample)
        if target.exists():
            return
        with Image.open(sample.path) as image:
            image.convert("RGB").resize(
                (IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR
            ).save(target, format="JPEG", quality=92)

    print(
        f"Preparing or extending the 224px image cache for {len(samples)} images.",
        flush=True,
    )
    # Cache generation is a one-time I/O task. Keep it parallel even though
    # training itself uses zero Windows DataLoader workers for reliability.
    workers = min(8, os.cpu_count() or 1)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for index, _ in enumerate(executor.map(build_cache_item, samples), start=1):
            if index % 250 == 0 or index == len(samples):
                print(f"Image cache: {index}/{len(samples)}", flush=True)
    RESIZED_IMAGE_CACHE_MARKER.write_text(f"{cache_signature}\n", encoding="utf-8")
    return [Sample(cached_path(sample), sample.label, sample.source) for sample in samples]


def make_transforms() -> tuple[transforms.Compose, transforms.Compose]:
    train_transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(degrees=8, translate=(0.04, 0.04), scale=(0.9, 1.1)),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    evaluation_transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return train_transform, evaluation_transform


def class_weights(samples: list[Sample], device: torch.device) -> torch.Tensor:
    counts = np.bincount([sample.label for sample in samples], minlength=len(CLASS_NAMES))
    return torch.tensor(counts.sum() / (len(CLASS_NAMES) * counts), dtype=torch.float32, device=device)


def evaluate(
    model: nn.Module, loader: DataLoader, samples: list[Sample], loss_function: nn.Module, device: torch.device
) -> tuple[float, dict[str, object], list[dict[str, str]]]:
    model.eval()
    total_loss = 0.0
    predictions: list[int] = []
    confidences: list[float] = []
    with torch.inference_mode():
        for images, labels in loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                logits = model(images)
                loss = loss_function(logits, labels)
            probabilities = logits.softmax(dim=1).cpu().numpy()
            total_loss += loss.item() * len(labels)
            predictions.extend(np.argmax(probabilities, axis=1).tolist())
            confidences.extend(np.max(probabilities, axis=1).tolist())

    actual = [sample.label for sample in samples]
    per_class = {}
    for label, class_name in enumerate(CLASS_NAMES):
        class_total = sum(value == label for value in actual)
        correct = sum(a == p == label for a, p in zip(actual, predictions, strict=True))
        per_class[class_name] = {"correct": correct, "total": class_total, "recall": correct / class_total if class_total else None}
    accuracy = sum(a == p for a, p in zip(actual, predictions, strict=True)) / len(actual)
    rows = [
        {
            "path": str(sample.path),
            "source": sample.source,
            "actual": CLASS_NAMES[sample.label],
            "prediction": CLASS_NAMES[prediction],
            "confidence": f"{confidence:.6f}",
        }
        for sample, prediction, confidence in zip(samples, predictions, confidences, strict=True)
    ]
    return total_loss / len(samples), {"accuracy": accuracy, "per_class": per_class}, rows


def save_loss_plot(train_losses: list[float], validation_losses: list[float]) -> Path:
    path = OUTPUT_ROOT / "loss_curve.png"
    width, height, margin = 900, 560, 70
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((margin, 20), f"{ARCHITECTURE}: {TRAINING_DATA_MODE} training loss", fill="black")
    draw.line((margin, height - margin, width - margin, height - margin), fill="black", width=2)
    draw.line((margin, margin, margin, height - margin), fill="black", width=2)
    minimum = min(0.0, min(train_losses + validation_losses))
    maximum = max(train_losses + validation_losses)
    loss_range = max(maximum - minimum, 1e-6)

    def point(index: int, loss: float) -> tuple[int, int]:
        return (
            round(margin + index * (width - 2 * margin) / max(len(train_losses) - 1, 1)),
            round(height - margin - (loss - minimum) * (height - 2 * margin) / loss_range),
        )

    for losses, colour in ((train_losses, "royalblue"), (validation_losses, "firebrick")):
        points = [point(index, value) for index, value in enumerate(losses)]
        if len(points) > 1:
            draw.line(points, fill=colour, width=3)
        for x, y in points:
            draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=colour)
    draw.text((width - 285, 20), "Blue: train loss   Red: validation loss", fill="black")
    image.save(path)
    return path


def parse_arguments() -> argparse.Namespace:
    legacy_mode = os.environ.get("BIRD_TRAINING_DATA_MODE", "combined").lower()
    legacy_mode = {"real_only": "real"}.get(legacy_mode, legacy_mode)
    parser = argparse.ArgumentParser(
        description="Train a bird classifier on synthetic, real, or combined data."
    )
    parser.add_argument(
        "--data-source",
        choices=("synthetic", "real", "combined"),
        default=legacy_mode,
        help=(
            "Training data for the core class set. The extended class set always uses "
            "real core classes and synthetic-only rare classes."
        ),
    )
    parser.add_argument(
        "--architecture",
        choices=("convnext-tiny", "mobilenet-v3-large"),
        default="convnext-tiny",
        help="Model architecture to train.",
    )
    parser.add_argument(
        "--class-set",
        choices=("core", "extended"),
        default="core",
        help="Core classes, or core plus Amsel, Gruenfink, and Stieglitz.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Optional directory for checkpoints and reports.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=EPOCHS,
        help=f"Maximum training epochs (default: {EPOCHS}).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help=f"Images per training batch (default: {BATCH_SIZE}).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=NUM_WORKERS,
        help=f"Data-loader worker processes; use 0 on Windows for reliability (default: {NUM_WORKERS}).",
    )
    parser.add_argument(
        "--no-image-cache",
        action="store_true",
        help="Do not use the reusable 224px image cache.",
    )
    return parser.parse_args()


def default_output_root(data_source: str, architecture: str, class_set: str) -> Path:
    """Keep the existing deployed checkpoint location stable for the default run."""
    if (data_source, architecture, class_set) == ("combined", "convnext-tiny", "core"):
        return PROJECT_ROOT / "data" / "interim" / "combined_synthetic_real_convnext_tiny"
    safe_architecture = architecture.replace("-", "_")
    return PROJECT_ROOT / "data" / "interim" / f"{data_source}_{class_set}_{safe_architecture}"


def main() -> None:
    global ARCHITECTURE, BATCH_SIZE, CACHE_RESIZED_IMAGES, CLASS_NAMES, EPOCHS, NUM_WORKERS, OUTPUT_ROOT, TRAINING_DATA_MODE
    args = parse_arguments()
    TRAINING_DATA_MODE = args.data_source
    ARCHITECTURE = args.architecture
    if args.epochs < 1:
        raise ValueError("--epochs must be at least 1.")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1.")
    if args.num_workers < 0:
        raise ValueError("--num-workers cannot be negative.")
    EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    NUM_WORKERS = args.num_workers
    CACHE_RESIZED_IMAGES = not args.no_image_cache
    CLASS_NAMES = (
        CORE_CLASS_NAMES
        if args.class_set == "core"
        else CORE_CLASS_NAMES + EXTENDED_ONLY_CLASS_NAMES
    )
    synthetic_only_class_names = (
        EXTENDED_ONLY_CLASS_NAMES if args.class_set == "extended" else ()
    )
    synthetic_only_labels = {
        CLASS_NAMES.index(class_name) for class_name in synthetic_only_class_names
    }
    OUTPUT_ROOT = args.output_dir or default_output_root(
        TRAINING_DATA_MODE, ARCHITECTURE, args.class_set
    )
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("This training run requires CUDA.")
    # Input dimensions are fixed at 224x224, so cuDNN can safely select its
    # fastest convolution kernels once and reuse them for every batch.
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    TORCH_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    torch.hub.set_dir(str(TORCH_CACHE_ROOT))
    generator = random.Random(SEED)
    real_samples = collect_samples(REAL_ROOT, "real")
    real_training_pool, real_test_samples = split_real_samples_for_experiment(
        real_samples,
        synthetic_only_labels,
        REAL_TEST_FRACTION,
        generator,
    )
    if args.class_set == "extended":
        synthetic_samples = collect_samples(SYNTHETIC_ROOT, "synthetic")
        synthetic_rare_samples = [
            sample for sample in synthetic_samples if sample.label in synthetic_only_labels
        ]
        training_pool = real_training_pool + synthetic_rare_samples
        TRAINING_DATA_MODE = "hybrid-real-core-synthetic-rare"
    elif TRAINING_DATA_MODE == "combined":
        training_pool = collect_samples(SYNTHETIC_ROOT, "synthetic") + real_training_pool
    elif TRAINING_DATA_MODE == "real":
        training_pool = real_training_pool
    else:
        training_pool = collect_samples(SYNTHETIC_ROOT, "synthetic")
    training_samples, validation_samples = split_by_class(training_pool, VALIDATION_FRACTION, generator)
    if CACHE_RESIZED_IMAGES:
        all_samples = training_samples + validation_samples + real_test_samples
        cached = cache_resized_samples(all_samples)
        training_samples = cached[: len(training_samples)]
        validation_samples = cached[len(training_samples) : len(training_samples) + len(validation_samples)]
        real_test_samples = cached[len(training_samples) + len(validation_samples) :]
    train_transform, evaluation_transform = make_transforms()

    loader_options = {
        "batch_size": BATCH_SIZE,
        "pin_memory": True,
        "num_workers": NUM_WORKERS,
        "persistent_workers": NUM_WORKERS > 0,
    }
    if NUM_WORKERS > 0:
        loader_options["prefetch_factor"] = 4
    train_loader = DataLoader(BirdDataset(training_samples, train_transform), shuffle=True, **loader_options)
    validation_loader = DataLoader(BirdDataset(validation_samples, evaluation_transform), **loader_options)
    test_loader = DataLoader(BirdDataset(real_test_samples, evaluation_transform), **loader_options)
    if ARCHITECTURE == "convnext-tiny":
        model = create_convnext_tiny(len(CLASS_NAMES), pretrained=True, freeze_backbone=True).to(device)
        unfreeze_backbone = unfreeze_convnext_backbone
    else:
        model = create_mobilenet_v3_large(
            len(CLASS_NAMES), pretrained=True, freeze_backbone=True
        ).to(device)
        unfreeze_backbone = unfreeze_mobilenet_v3_large_backbone
    loss_function = nn.CrossEntropyLoss(weight=class_weights(training_samples, device))
    optimizer = torch.optim.AdamW((parameter for parameter in model.parameters() if parameter.requires_grad), lr=3e-4, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda")
    print(
        f"Training on {torch.cuda.get_device_name(device)} | "
        f"batch size {BATCH_SIZE} | data-loader workers {NUM_WORKERS}",
        flush=True,
    )
    best_path = OUTPUT_ROOT / "best_validation_model.pt"
    best_loss, best_epoch, stale_epochs = float("inf"), 0, 0
    train_losses: list[float] = []
    validation_losses: list[float] = []

    for epoch in range(EPOCHS):
        epoch_started = time.perf_counter()
        if epoch == FREEZE_BACKBONE_EPOCHS:
            unfreeze_backbone(model)
            optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=1e-4)
            print(f"Unfroze {ARCHITECTURE} backbone.")
        model.train()
        total_loss = total_images = 0
        for images, labels in train_loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                loss = loss_function(model(images), labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item() * len(labels)
            total_images += len(labels)
        train_loss = total_loss / total_images
        validation_loss, validation_metrics, _ = evaluate(model, validation_loader, validation_samples, loss_function, device)
        train_losses.append(train_loss)
        validation_losses.append(validation_loss)
        print(
            f"Epoch {epoch + 1:02d}/{EPOCHS} | train loss: {train_loss:.4f} | "
            f"validation loss: {validation_loss:.4f} | "
            f"validation accuracy: {validation_metrics['accuracy']:.3f} | "
            f"time: {time.perf_counter() - epoch_started:.1f}s",
            flush=True,
        )
        if validation_loss < best_loss - EARLY_STOPPING_MIN_DELTA:
            best_loss, best_epoch, stale_epochs = validation_loss, epoch + 1, 0
            torch.save({"architecture": ARCHITECTURE, "epoch": best_epoch, "validation_loss": best_loss, "model_state_dict": model.state_dict()}, best_path)
        else:
            stale_epochs += 1
            if stale_epochs >= EARLY_STOPPING_PATIENCE:
                print("Early stopping triggered.")
                break

    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=False)["model_state_dict"])
    _, validation_metrics, _ = evaluate(model, validation_loader, validation_samples, loss_function, device)
    _, real_test_metrics, prediction_rows = evaluate(model, test_loader, real_test_samples, loss_function, device)
    loss_plot = save_loss_plot(train_losses, validation_losses)
    report = {
        "architecture": ARCHITECTURE,
        "training_data_mode": TRAINING_DATA_MODE,
        "class_set": args.class_set,
        "class_names": CLASS_NAMES,
        "seed": SEED,
        "device": str(device),
        "excluded_real_classes": sorted(set(ALL_REAL_CLASS_NAMES).difference(CLASS_NAMES)),
        "synthetic_only_training_classes": list(synthetic_only_class_names),
        "real_test_policy": {
            "core_classes": f"{REAL_TEST_FRACTION:.0%} class-balanced held-out split",
            "synthetic_only_classes": "all labelled real images held out",
        },
        "splits": {
            "training": {"total": len(training_samples), "by_class": count_by_class(training_samples), "by_source": count_by_source(training_samples)},
            "validation": {"total": len(validation_samples), "by_class": count_by_class(validation_samples), "by_source": count_by_source(validation_samples)},
            "held_out_real_test": {"total": len(real_test_samples), "by_class": count_by_class(real_test_samples)},
        },
        "early_stopping": {"best_epoch": best_epoch, "best_validation_loss": best_loss, "epochs_completed": len(train_losses)},
        "validation_metrics": validation_metrics,
        "held_out_real_test_metrics": real_test_metrics,
        "loss_history": {"train": train_losses, "validation": validation_losses},
        "loss_plot": str(loss_plot),
    }
    (OUTPUT_ROOT / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with (OUTPUT_ROOT / "held_out_real_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=prediction_rows[0].keys())
        writer.writeheader()
        writer.writerows(prediction_rows)
    torch.save({"architecture": ARCHITECTURE, "model_state_dict": model.state_dict(), "class_names": CLASS_NAMES, "training_data": TRAINING_DATA_MODE, "class_set": args.class_set, "held_out_real_test_fraction": REAL_TEST_FRACTION, "synthetic_only_training_classes": list(synthetic_only_class_names)}, OUTPUT_ROOT / "model.pt")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
