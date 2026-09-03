import torch

from models.detector import UAVDetector


def main():
    device = torch.device("cpu")

    print("Creating model...")
    model = UAVDetector(
        num_classes=10,
        pretrained=False,
        fpn_channels=256,
    )

    model = model.to(device)
    model.eval()

    print("Creating input...")
    x = torch.randn(
        2,
        3,
        640,
        640,
        device=device,
    )

    print(f"Input: {x.shape}")

    with torch.no_grad():
        backbone_features = model.backbone(x)

    expected_backbone_shapes = [
        (2, 160, 160, 96),
        (2, 80, 80, 192),
        (2, 40, 40, 384),
        (2, 20, 20, 768),
    ]

    print("\nBackbone outputs:")

    for i, (feature, expected) in enumerate(
        zip(backbone_features, expected_backbone_shapes)
    ):
        print(f"  Stage {i}: {feature.shape}")

        assert tuple(feature.shape) == expected, (
            f"Backbone stage {i} shape mismatch: "
            f"got {tuple(feature.shape)}, expected {expected}"
        )

    with torch.no_grad():
        fpn_features = model.fpn(backbone_features)

    expected_fpn_shapes = [
        (2, 256, 160, 160),
        (2, 256, 80, 80),
        (2, 256, 40, 40),
        (2, 256, 20, 20),
    ]

    print("\nFPN outputs:")

    for i, (feature, expected) in enumerate(
        zip(fpn_features, expected_fpn_shapes)
    ):
        print(f"  P{i + 3}: {feature.shape}")

        assert tuple(feature.shape) == expected, (
            f"FPN P{i + 3} shape mismatch: "
            f"got {tuple(feature.shape)}, expected {expected}"
        )

    with torch.no_grad():
        outputs = model.head(fpn_features)

    expected_head_shapes = [
        (
            (2, 10, 160, 160),
            (2, 4, 160, 160),
            (2, 1, 160, 160),
        ),
        (
            (2, 10, 80, 80),
            (2, 4, 80, 80),
            (2, 1, 80, 80),
        ),
        (
            (2, 10, 40, 40),
            (2, 4, 40, 40),
            (2, 1, 40, 40),
        ),
        (
            (2, 10, 20, 20),
            (2, 4, 20, 20),
            (2, 1, 20, 20),
        ),
    ]

    print("\nFCOS outputs:")

    for i, (output, expected) in enumerate(
        zip(outputs, expected_head_shapes)
    ):
        cls_shape = tuple(output["cls_logits"].shape)
        bbox_shape = tuple(output["bbox_reg"].shape)
        ctr_shape = tuple(output["centerness"].shape)

        print(f"  P{i + 3}:")
        print(f"    cls:        {cls_shape}")
        print(f"    bbox:       {bbox_shape}")
        print(f"    centerness: {ctr_shape}")

        assert cls_shape == expected[0], (
            f"P{i + 3} classification shape mismatch: "
            f"got {cls_shape}, expected {expected[0]}"
        )

        assert bbox_shape == expected[1], (
            f"P{i + 3} bbox shape mismatch: "
            f"got {bbox_shape}, expected {expected[1]}"
        )

        assert ctr_shape == expected[2], (
            f"P{i + 3} centerness shape mismatch: "
            f"got {ctr_shape}, expected {expected[2]}"
        )

    print("\nPASS: forward pass shapes OK")


if __name__ == "__main__":
    main()