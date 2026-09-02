from pathlib import Path

import cv2

from datasets.visdrone import VisDroneDataset


CLASS_NAMES = [
    "pedestrian",
    "people",
    "bicycle",
    "car",
    "van",
    "truck",
    "tricycle",
    "awning-tricycle",
    "bus",
    "motor",
]


dataset = VisDroneDataset(
    root="data/raw/VisDrone2019-DET",
    split="train",
)

image, target = dataset[0]

# Convert PyTorch tensor [C, H, W] back to OpenCV image [H, W, C]
image = image.permute(1, 2, 0).numpy()
image = (image * 255).clip(0, 255).astype("uint8")

# RGB -> BGR for OpenCV
image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

boxes = target["boxes"].numpy()
labels = target["labels"].numpy()

for box, label in zip(boxes, labels):
    x1, y1, x2, y2 = box.astype(int)

    cv2.rectangle(
        image,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        2,
    )

    cv2.putText(
        image,
        CLASS_NAMES[label],
        (x1, max(y1 - 5, 15)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        1,
    )


output_dir = Path("results/data_visualization")
output_dir.mkdir(parents=True, exist_ok=True)

output_path = output_dir / "train_sample_0.png"

cv2.imwrite(str(output_path), image)

print(f"Saved visualization to: {output_path}")