from collections import Counter
from pathlib import Path
import json

import cv2

from datasets.visdrone import CATEGORY_MAP


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


ROOT = Path("data/raw/VisDrone2019-DET")
SPLIT = "train"

image_dir = ROOT / f"VisDrone2019-DET-{SPLIT}" / "images"
annotation_dir = ROOT / f"VisDrone2019-DET-{SPLIT}" / "annotations"

images = sorted(image_dir.glob("*.jpg"))

class_counts = Counter()

total_objects = 0
objects_per_image = []

box_widths = []
box_heights = []
box_areas = []

small_objects = 0

for image_path in images:
    annotation_path = annotation_dir / f"{image_path.stem}.txt"

    image_objects = 0

    with open(annotation_path, "r") as f:
        for line in f:
            parts = line.strip().split(",")

            if len(parts) < 8:
                continue

            x, y, w, h, score, category, truncation, occlusion = map(
                int, parts[:8]
            )

            if category not in CATEGORY_MAP:
                continue

            if w <= 0 or h <= 0:
                continue

            class_counts[category] += 1

            total_objects += 1
            image_objects += 1

            box_widths.append(w)
            box_heights.append(h)
            box_areas.append(w * h)

            # Simple area-based small-object statistic.
            if w * h < 32 * 32:
                small_objects += 1

    objects_per_image.append(image_objects)


stats = {
    "split": SPLIT,
    "num_images": len(images),
    "total_objects": total_objects,
    "avg_objects_per_image": (
        total_objects / len(images) if images else 0
    ),
    "small_objects": small_objects,
    "small_object_fraction": (
        small_objects / total_objects if total_objects else 0
    ),
    "class_counts": {
        CLASS_NAMES[CATEGORY_MAP[k]]: v
        for k, v in sorted(class_counts.items())
    },
    "box_width": {
        "min": min(box_widths) if box_widths else 0,
        "max": max(box_widths) if box_widths else 0,
        "mean": (
            sum(box_widths) / len(box_widths)
            if box_widths else 0
        ),
    },
    "box_height": {
        "min": min(box_heights) if box_heights else 0,
        "max": max(box_heights) if box_heights else 0,
        "mean": (
            sum(box_heights) / len(box_heights)
            if box_heights else 0
        ),
    },
}


output_dir = Path("data/statistics")
output_dir.mkdir(parents=True, exist_ok=True)

output_path = output_dir / "dataset_statistics.json"

with open(output_path, "w") as f:
    json.dump(stats, f, indent=2)

print("Dataset statistics")
print("------------------")
print("Images:", len(images))
print("Objects:", total_objects)
print("Average objects/image:", stats["avg_objects_per_image"])
print("Small-object fraction:", stats["small_object_fraction"])
print()
print("Classes:")

for name, count in stats["class_counts"].items():
    print(f"  {name:20s}: {count}")

print()
print(f"Saved to: {output_path}")