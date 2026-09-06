from pathlib import Path
import cv2


ROOT = Path("data/raw/VisDrone2019-DET")

SPLITS = ["train", "val", "test-dev"]

for split in SPLITS:
    image_dir = ROOT / f"VisDrone2019-DET-{split}" / "images"
    annotation_dir = ROOT / f"VisDrone2019-DET-{split}" / "annotations"

    print(f"\n========== {split} ==========")

    count = 0

    for image_path in sorted(image_dir.glob("*.jpg")):
        annotation_path = annotation_dir / f"{image_path.stem}.txt"

        image = cv2.imread(str(image_path))

        if image is None:
            continue

        height, width = image.shape[:2]

        with open(annotation_path, "r") as f:
            for line_number, line in enumerate(f, start=1):
                parts = line.strip().split(",")

                if len(parts) < 8:
                    continue

                x, y, w, h, score, category, trunc, occ = map(
                    int, parts[:8]
                )

                x2 = x + w
                y2 = y + h

                invalid = (
                    w <= 0
                    or h <= 0
                    or x < 0
                    or y < 0
                    or x2 > width
                    or y2 > height
                )

                if invalid:
                    count += 1

                    print(
                        f"\nInvalid box #{count}"
                    )
                    print(f"Image: {image_path.name}")
                    print(f"Annotation: {annotation_path}")
                    print(f"Line: {line_number}")
                    print(f"Image size: {width} x {height}")
                    print(f"Raw annotation: {line.strip()}")
                    print(
                        f"Box: x={x}, y={y}, "
                        f"w={w}, h={h}"
                    )
                    print(
                        f"Box extent: "
                        f"x2={x2}, y2={y2}"
                    )

    print(f"\nTotal invalid boxes in {split}: {count}")