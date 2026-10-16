"""Train a bird-species classifier on the prepared labelled dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn

from bird_classifier.data.dataloader import BirdDataset, create_dataloader
from bird_classifier.data.preprocessing import get_data_split, get_mean_std
from bird_classifier.models.cnn import SimpleCNN
from bird_classifier.models.ConVNeXt_tiny import (
    create_convnext_tiny,
    unfreeze_convnext_backbone,
)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        choices=("convnext-tiny", "simple-cnn"),
        default="convnext-tiny",
        help="Classifier architecture to train (default: convnext-tiny).",
    )
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Override the architecture-specific default learning rate.",
    )
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Train ConvNeXt-Tiny from random weights instead of ImageNet weights.",
    )
    parser.add_argument(
        "--torch-cache-dir",
        type=Path,
        default=Path("models/torch_cache"),
        help="Directory for downloaded pretrained weights (default: models/torch_cache).",
    )
    parser.add_argument(
        "--freeze-backbone-epochs",
        type=int,
        default=5,
        help="For pretrained ConvNeXt, train only its new head for this many epochs.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Checkpoint path. Defaults to models/<model>.pt.",
    )
    return parser.parse_args()


def evaluate(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_function: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct_predictions = 0
    total_images = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device, dtype=torch.long)
            logits = model(images)
            loss = loss_function(logits, labels)
            total_loss += loss.item() * images.shape[0]
            correct_predictions += (logits.argmax(dim=1) == labels).sum().item()
            total_images += images.shape[0]

    return total_loss / total_images, correct_predictions / total_images


def build_model(args: argparse.Namespace, num_classes: int) -> nn.Module:
    if args.model == "simple-cnn":
        return SimpleCNN(in_channels=3, num_classes=num_classes)

    use_pretrained_weights = not args.no_pretrained
    return create_convnext_tiny(
        num_classes=num_classes,
        pretrained=use_pretrained_weights,
        freeze_backbone=use_pretrained_weights and args.freeze_backbone_epochs > 0,
    )


def create_optimizer(model: nn.Module, learning_rate: float) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=learning_rate,
        weight_decay=1e-4,
    )


def main() -> None:
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.freeze_backbone_epochs < 0:
        raise ValueError(
            "epochs and batch-size must be positive; freeze-backbone-epochs cannot be negative"
        )

    train_paths, train_labels = get_data_split(
        image_dir=Path("data/interim/deduplicated_images"),
        labels_path=Path("data/processed/training_labels.csv"),
        split="train",
    )
    validation_paths, validation_labels = get_data_split(
        image_dir=Path("data/interim/deduplicated_images"),
        labels_path=Path("data/processed/training_labels.csv"),
        split="test",
    )
    num_classes = len(np.unique(train_labels))

    if args.model == "convnext-tiny" and not args.no_pretrained:
        # ImageNet normalization must match the pretrained feature extractor.
        train_mean = np.asarray(IMAGENET_MEAN, dtype=np.float32)
        train_std = np.asarray(IMAGENET_STD, dtype=np.float32)
    else:
        unnormalized_train_dataset = BirdDataset(train_paths, train_labels)
        train_mean, train_std = get_mean_std(unnormalized_train_dataset)

    train_dataset = BirdDataset(train_paths, train_labels, train_std, train_mean)
    validation_dataset = BirdDataset(validation_paths, validation_labels, train_std, train_mean)
    train_loader = create_dataloader(train_dataset, batch_size=args.batch_size, shuffle=True)
    validation_loader = create_dataloader(
        validation_dataset, batch_size=args.batch_size, shuffle=False
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.model == "convnext-tiny" and not args.no_pretrained:
        # Keep the ImageNet checkpoint in the project drive instead of Torch's
        # default cache location on C:.
        args.torch_cache_dir.mkdir(parents=True, exist_ok=True)
        torch.hub.set_dir(str(args.torch_cache_dir))
    model = build_model(args, num_classes).to(device)
    learning_rate = args.learning_rate or (3e-5 if args.model == "convnext-tiny" else 1e-3)
    optimizer = create_optimizer(model, learning_rate)
    loss_function = nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        if (
            args.model == "convnext-tiny"
            and not args.no_pretrained
            and epoch == args.freeze_backbone_epochs
        ):
            unfreeze_convnext_backbone(model)
            optimizer = create_optimizer(model, learning_rate)
            print("Unfroze the ConvNeXt feature extractor for full fine-tuning.")

        model.train()
        total_loss = 0.0
        correct_predictions = 0
        total_images = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device, dtype=torch.long)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = loss_function(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.shape[0]
            correct_predictions += (logits.argmax(dim=1) == labels).sum().item()
            total_images += images.shape[0]

        train_loss = total_loss / total_images
        train_accuracy = correct_predictions / total_images
        validation_loss, validation_accuracy = evaluate(
            model, validation_loader, loss_function, device
        )
        print(
            f"Epoch {epoch + 1}/{args.epochs} | "
            f"train loss: {train_loss:.4f} | train accuracy: {train_accuracy:.2%} | "
            f"validation loss: {validation_loss:.4f} | "
            f"validation accuracy: {validation_accuracy:.2%}"
        )

    output_path = args.output or Path("models") / f"{args.model}.pt"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_name": args.model,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "mean": np.asarray(train_mean).tolist(),
            "std": np.asarray(train_std).tolist(),
            "num_classes": num_classes,
        },
        output_path,
    )
    print(f"Saved checkpoint to {output_path}")


if __name__ == "__main__":
    main()
