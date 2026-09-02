"""Validate a VisDrone2019-DET split."""

import argparse
import sys
from pathlib import Path


def parse_line(line):
    parts = line.strip().split(",")

    if len(parts) < 8:
        return None

    x, y, w, h, score, cat, trunc, occ = map(int, parts[:8])

    return {
        "x": x,
        "y": y,
        "w": w,
        "h": h,
        "score": score,
        "cat": cat,
        "trunc": trunc,
        "occ": occ,
    }


def verify(root: Path, split: str):
    img_dir = root / f"VisDrone2019-DET-{split}" / "images"
    ann_dir = root / f"VisDrone2019-DET-{split}" / "annotations"

    report = {
        "images": 0,
        "annotations": 0,
        "missing_images": [],
        "missing_annotations": [],
        "malformed_lines": [],
        "invalid_boxes": [],
        "categories_seen": set(),
    }

    images = sorted(img_dir.glob("*.jpg"))
    report["images"] = len(images)

    for img_path in images:

        ann_path = ann_dir / (img_path.stem + ".txt")

        if not ann_path.exists():
            report["missing_annotations"].append(img_path.name)
            continue

        import cv2

        im = cv2.imread(str(img_path))

        if im is None:
            report["missing_images"].append(img_path.name)
            continue

        H, W = im.shape[:2]

        with open(ann_path) as f:

            for i, line in enumerate(f):

                if not line.strip():
                    continue

                rec = parse_line(line)

                if rec is None:
                    report["malformed_lines"].append(
                        f"{ann_path.name}:{i}"
                    )
                    continue

                report["categories_seen"].add(rec["cat"])

                if rec["w"] <= 0 or rec["h"] <= 0:
                    report["invalid_boxes"].append(
                        f"{ann_path.name}:{i} negative/zero dim"
                    )

                if (
                    rec["x"] < 0
                    or rec["y"] < 0
                    or rec["x"] + rec["w"] > W
                    or rec["y"] + rec["h"] > H
                ):
                    report["invalid_boxes"].append(
                        f"{ann_path.name}:{i} out of image bounds"
                    )

        report["annotations"] += 1

    print(f"--- VisDrone {split} report ---")
    print(f"images found:            {report['images']}")
    print(f"annotations matched:     {report['annotations']}")
    print(
        f"missing annotations:     "
        f"{len(report['missing_annotations'])}"
    )
    print(
        f"missing/corrupt images:  "
        f"{len(report['missing_images'])}"
    )
    print(
        f"malformed lines:         "
        f"{len(report['malformed_lines'])}"
    )
    print(
        f"invalid boxes:           "
        f"{len(report['invalid_boxes'])}"
    )
    print(
        f"category ids seen:       "
        f"{sorted(report['categories_seen'])}"
    )

    ok = not (
        report["missing_images"]
        or report["missing_annotations"]
        or report["malformed_lines"]
        or report["invalid_boxes"]
    )

    print("RESULT:", "PASS" if ok else "FAIL — see above")

    return ok


if __name__ == "__main__":

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--root",
        required=True,
    )

    ap.add_argument(
        "--split",
        required=True,
        choices=["train", "val", "test-dev"],
    )

    args = ap.parse_args()

    ok = verify(
        Path(args.root),
        args.split,
    )

    sys.exit(0 if ok else 1)