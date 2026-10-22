"""MobileNetV3-Small factory for lightweight bird-presence inference."""

from __future__ import annotations

from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small


def create_mobilenet_v3_small(
    num_classes: int,
    *,
    pretrained: bool = True,
    freeze_backbone: bool = False,
):
    """Create MobileNetV3-Small with a classification head for ``num_classes``."""
    if num_classes < 1:
        raise ValueError("num_classes must be at least 1")

    weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
    model = mobilenet_v3_small(weights=weights)
    classifier = model.classifier[-1]
    model.classifier[-1] = type(classifier)(classifier.in_features, num_classes)

    if freeze_backbone:
        for parameter in model.features.parameters():
            parameter.requires_grad = False
    return model


def unfreeze_mobilenet_v3_small_backbone(model) -> None:
    """Make MobileNetV3-Small's feature extractor trainable after warm-up."""
    for parameter in model.features.parameters():
        parameter.requires_grad = True
