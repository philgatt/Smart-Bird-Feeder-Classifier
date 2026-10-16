"""ConvNeXt-Tiny classifier factory for bird-species classification.

The model is intentionally device-agnostic: create it here, then move it to
CUDA/CPU in the training script with ``model.to(device)``.
"""

from __future__ import annotations

from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny


def create_convnext_tiny(
    num_classes: int,
    *,
    pretrained: bool = True,
    freeze_backbone: bool = False,
):
    """Create a ConvNeXt-Tiny model with a classifier sized for this dataset.

    Args:
        num_classes: Number of output species/classes. Must be positive.
        pretrained: Load ImageNet-1K weights. Set to ``False`` for offline
            smoke tests or when training from scratch.
        freeze_backbone: Freeze the feature extractor, leaving only the new
            classification head trainable. This is useful for the first few
            fine-tuning epochs on a small dataset.

    Returns:
        A torchvision ConvNeXt-Tiny model that outputs ``num_classes`` logits.
    """
    if num_classes < 1:
        raise ValueError("num_classes must be at least 1")

    weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
    model = convnext_tiny(weights=weights)

    # Torchvision's ConvNeXt classifier is LayerNorm -> Flatten -> Linear.
    # Replace only its final Linear layer; retain the pretrained backbone and
    # normalization layers intact.
    classifier = model.classifier[-1]
    model.classifier[-1] = type(classifier)(classifier.in_features, num_classes)

    if freeze_backbone:
        for parameter in model.features.parameters():
            parameter.requires_grad = False

    return model


def unfreeze_convnext_backbone(model) -> None:
    """Make all ConvNeXt feature-extractor parameters trainable again."""
    for parameter in model.features.parameters():
        parameter.requires_grad = True
