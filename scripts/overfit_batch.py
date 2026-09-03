import torch
from torch.utils.data import DataLoader, Subset

from datasets.visdrone import VisDroneDataset
from datasets.transforms import Letterbox

from models.detector import UAVDetector
from models.head.fcos_target import FCOSTargetAssigner
from models.losses.fcos_loss import FCOSLoss


DATA_ROOT = "data/raw/VisDrone2019-DET"

NUM_IMAGES = 6
NUM_ITERS = 250
LR = 1e-3


class OverfitTransform:
    """
    Letterbox only.

    No random augmentation is used because this script
    is specifically for the 6-image overfit test.
    """

    def __init__(self, size=640):
        self.letterbox = Letterbox(size=size)

    def __call__(self, image, boxes, labels):
        image, boxes = self.letterbox(
            image,
            boxes,
        )

        return image, boxes, labels


def collate_fn(batch):
    images = torch.stack(
        [item[0] for item in batch]
    )

    targets = [
        item[1]
        for item in batch
    ]

    return images, targets


def main():

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Device: {device}")

    # ---------------------------------------------------------
    # Dataset
    # ---------------------------------------------------------

    dataset = VisDroneDataset(
        root=DATA_ROOT,
        split="train",
        transform=OverfitTransform(size=640),
    )

    dataset = Subset(
        dataset,
        range(NUM_IMAGES),
    )

    loader = DataLoader(
        dataset,
        batch_size=NUM_IMAGES,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    images, raw_targets = next(iter(loader))

    images = images.to(device)

    print(f"Images: {images.shape}")

    for i, target in enumerate(raw_targets):
        print(
            f"Image {i}: "
            f"{target['boxes'].shape[0]} boxes"
        )

    # ---------------------------------------------------------
    # Model
    # ---------------------------------------------------------

    model = UAVDetector(
        num_classes=10,
        pretrained=False,
        fpn_channels=256,
    )

    model = model.to(device)
    model.train()

    # ---------------------------------------------------------
    # Loss + target assigner
    # ---------------------------------------------------------

    target_assigner = FCOSTargetAssigner()

    loss_fn = FCOSLoss(
        lambda_cls=1.0,
        lambda_loc=2.0,
        lambda_ctr=1.0,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
    )

    # ---------------------------------------------------------
    # Determine FPN feature shapes
    # ---------------------------------------------------------

    with torch.no_grad():

        features = model.backbone(images)

        fpn_features = model.fpn(
            features
        )

        feature_shapes = [
            feature.shape[-2:]
            for feature in fpn_features
        ]

    # ---------------------------------------------------------
    # Create FCOS targets
    # ---------------------------------------------------------

    fcos_targets = []

    for target in raw_targets:

        boxes = torch.as_tensor(
            target["boxes"],
            dtype=torch.float32,
            device=device,)

        labels = torch.as_tensor(
            target["labels"],
            dtype=torch.long,
            device=device,)

        image_targets = target_assigner(
            boxes,
            labels,
            feature_shapes,
        )

        fcos_targets.append(
            image_targets
        )

    total_positive = sum(
        (
            level_target[1] >= 0
        ).sum().item()
        for image_targets in fcos_targets
        for level_target in image_targets
    )

    print(
        f"Total positive locations: "
        f"{total_positive}"
    )

    if total_positive == 0:
        raise RuntimeError(
            "No positive FCOS targets found."
        )

    # ---------------------------------------------------------
    # Initial loss
    # ---------------------------------------------------------

    model.train()

    with torch.no_grad():

        predictions = model(images)

        initial_loss, initial_info = loss_fn(
            predictions,
            fcos_targets,
        )

    initial_loss_value = initial_loss.item()

    print(
        f"Initial loss: "
        f"{initial_loss_value:.6f}"
    )

    # ---------------------------------------------------------
    # Overfit loop
    # ---------------------------------------------------------

    for iteration in range(
        1,
        NUM_ITERS + 1,
    ):

        optimizer.zero_grad()

        predictions = model(images)

        loss, info = loss_fn(
            predictions,
            fcos_targets,
        )

        loss.backward()

        optimizer.step()

        if (
            iteration == 1
            or iteration % 10 == 0
        ):

            print(
                f"Iter {iteration:03d} | "
                f"loss={loss.item():.6f} | "
                f"cls={info['cls_loss']:.6f} | "
                f"box={info['box_loss']:.6f} | "
                f"ctr={info['centerness_loss']:.6f} | "
                f"pos={info['num_positive']}"
            )

    # ---------------------------------------------------------
    # Final result
    # ---------------------------------------------------------

    final_loss = loss.item()

    reduction = (
        1.0
        - final_loss / initial_loss_value
    ) * 100.0

    print()

    print(
        f"Initial loss: "
        f"{initial_loss_value:.6f}"
    )

    print(
        f"Final loss:   "
        f"{final_loss:.6f}"
    )

    print(
        f"Loss reduction: "
        f"{reduction:.2f}%"
    )

    if final_loss < initial_loss_value * 0.10:

        print()

        print(
            "PASS: 6-image overfit loss < 10% "
            "of initial loss"
        )

    else:

        print()

        print(
            "FAIL: 6-image overfit did not reach "
            "10% of initial loss"
        )


if __name__ == "__main__":
    main()