"""MobileNetV3-Large classifier factory for mobile-oriented bird inference."""

from __future__ import annotations

from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large


def create_mobilenet_v3_large(
    num_classes: int,
    *,
    pretrained: bool = True,
    freeze_backbone: bool = False,
):
    """Create MobileNetV3-Large with a classification head for ``num_classes``."""
    if num_classes < 1:
        raise ValueError("num_classes must be at least 1")

    weights = MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained else None
    model = mobilenet_v3_large(weights=weights)
    classifier = model.classifier[-1]
    model.classifier[-1] = type(classifier)(classifier.in_features, num_classes)

    if freeze_backbone:
        for parameter in model.features.parameters():
            parameter.requires_grad = False
    return model


def unfreeze_mobilenet_v3_large_backbone(model) -> None:
    """Make MobileNetV3's feature extractor trainable after warm-up."""
    for parameter in model.features.parameters():
        parameter.requires_grad = True
