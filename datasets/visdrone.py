from pathlib import Path

import cv2
import torch
from torch.utils.data import Dataset


CATEGORY_MAP = {
    1: 0,  # pedestrian
    2: 1,  # people
    3: 2,  # bicycle
    4: 3,  # car
    5: 4,  # van
    6: 5,  # truck
    7: 6,  # tricycle
    8: 7,  # awning-tricycle
    9: 8,  # bus
    10: 9,  # motor
}


class VisDroneDataset(Dataset):
    def __init__(self, root, split="train", transform=None):
        self.root = Path(root)
        self.split = split
        self.transform = transform

        self.image_dir = (
            self.root / f"VisDrone2019-DET-{split}" / "images"
        )
        self.annotation_dir = (
            self.root / f"VisDrone2019-DET-{split}" / "annotations"
        )

        self.images = sorted(self.image_dir.glob("*.jpg"))

        if len(self.images) == 0:
            raise RuntimeError(
                f"No images found in {self.image_dir}"
            )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_path = self.images[index]

        image = cv2.imread(str(image_path))

        if image is None:
            raise RuntimeError(
                f"Could not read image: {image_path}"
            )

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        annotation_path = (
            self.annotation_dir / f"{image_path.stem}.txt"
        )

        boxes = []
        labels = []

        with open(annotation_path, "r") as f:
            for line in f:
                parts = line.strip().split(",")

                if len(parts) < 8:
                    continue

                x, y, w, h, score, category, truncation, occlusion = map(
                    int, parts[:8]
                )

                # Ignore region and "others"
                if category not in CATEGORY_MAP:
                    continue

                if w <= 0 or h <= 0:
                    continue

                x1 = x
                y1 = y
                x2 = x + w
                y2 = y + h

                boxes.append([x1, y1, x2, y2])
                labels.append(CATEGORY_MAP[category])

        if self.transform is not None:
            image, boxes = self.transform(image, boxes)
        
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.long),
            "image_id": torch.tensor(index),
        }

        return image, target