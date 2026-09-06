"""Metric computation utilities for the UAV token-pruning evaluation pipeline.

This module is intentionally decoupled from any repository-specific model or
dataset code. It operates purely on:
  * COCO-format ground truth (a `pycocotools.coco.COCO` object)
  * COCO-format prediction dicts: {"image_id", "category_id", "bbox", "score"}
  * plain Python/NumPy arrays for box conversions and latency numbers

No assumptions are made about FCOS decoding, stride values, or VisDrone
category IDs -- those live in `evaluate.py` as clearly marked adapters.
"""

from __future__ import annotations

import contextlib
import io
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


@dataclass(frozen=True)
class DetectionMetrics:
    map50: float
    map50_95: float
    precision: float
    recall: float

    def as_dict(self) -> Dict[str, float]:
        return {
            "map50": self.map50,
            "map50_95": self.map50_95,
            "precision": self.precision,
            "recall": self.recall,
        }


@dataclass(frozen=True)
class LatencyMetrics:
    mean_ms: float
    p50_ms: float
    p95_ms: float
    fps: float

    def as_dict(self) -> Dict[str, float]:
        return {
            "mean_latency_ms": self.mean_ms,
            "p50_latency_ms": self.p50_ms,
            "p95_latency_ms": self.p95_ms,
            "fps": self.fps,
        }


def xyxy_to_xywh(boxes: np.ndarray) -> np.ndarray:
    """Convert [x1, y1, x2, y2] boxes to COCO-style [x, y, w, h].

    Args:
        boxes: array of shape (N, 4) in xyxy format.

    Returns:
        Array of shape (N, 4) in xywh format. Does NOT filter degenerate
        boxes -- call `filter_degenerate_boxes` first if you need that.

    Raises:
        ValueError: if `boxes` is not shaped (N, 4).
    """
    boxes = np.asarray(boxes, dtype=np.float64)
    if boxes.size == 0:
        return boxes.reshape(0, 4)
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError(f"Expected shape (N, 4), got {boxes.shape}")
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    w = x2 - x1
    h = y2 - y1
    return np.stack([x1, y1, w, h], axis=1)


def filter_degenerate_boxes(
    boxes_xywh: np.ndarray, *extra_arrays: np.ndarray
) -> Tuple[np.ndarray, ...]:
    """Drop boxes with non-positive width or height.

    Args:
        boxes_xywh: array of shape (N, 4) in xywh format.
        *extra_arrays: any number of arrays of shape (N, ...) filtered in
            lockstep with `boxes_xywh` (e.g. scores, labels).

    Returns:
        Tuple of (filtered_boxes, *filtered_extra_arrays), in the order
        they were passed in.
    """
    boxes_xywh = np.asarray(boxes_xywh, dtype=np.float64)
    if boxes_xywh.size == 0:
        return (boxes_xywh.reshape(0, 4),) + tuple(
            np.asarray(a).reshape(0, *np.asarray(a).shape[1:]) for a in extra_arrays
        )
    valid = (boxes_xywh[:, 2] > 0) & (boxes_xywh[:, 3] > 0)
    filtered = [boxes_xywh[valid]]
    for arr in extra_arrays:
        arr = np.asarray(arr)
        filtered.append(arr[valid])
    return tuple(filtered)


def build_coco_predictions(
    image_id: int,
    boxes_xyxy: np.ndarray,
    scores: np.ndarray,
    category_ids: np.ndarray,
) -> List[Dict]:
    """Package one image's detections into COCO-format prediction dicts.

    Args:
        image_id: COCO image id these detections belong to.
        boxes_xyxy: (N, 4) array in xyxy format, in ORIGINAL image coordinates.
        scores: (N,) array of confidence scores.
        category_ids: (N,) array of COCO category ids (already mapped from
            model class indices -- see `map_category_ids` and the
            `decode_fcos_predictions` adapter in evaluate.py).

    Returns:
        List of dicts matching the COCO detection format. Degenerate boxes
        (width <= 0 or height <= 0) are silently dropped.

    Raises:
        ValueError: if the input array lengths don't match.
    """
    boxes_xyxy = np.asarray(boxes_xyxy, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    category_ids = np.asarray(category_ids)

    if not (len(boxes_xyxy) == len(scores) == len(category_ids)):
        raise ValueError(
            "boxes, scores, and category_ids must have matching lengths: "
            f"{len(boxes_xyxy)}, {len(scores)}, {len(category_ids)}"
        )
    if len(boxes_xyxy) == 0:
        return []

    boxes_xywh = xyxy_to_xywh(boxes_xyxy)
    boxes_xywh, scores, category_ids = filter_degenerate_boxes(
        boxes_xywh, scores, category_ids
    )

    predictions = []
    for box, score, cat_id in zip(boxes_xywh, scores, category_ids):
        predictions.append(
            {
                "image_id": int(image_id),
                "category_id": int(cat_id),
                "bbox": [float(box[0]), float(box[1]), float(box[2]), float(box[3])],
                "score": float(score),
            }
        )
    return predictions


def map_category_ids(
    class_indices: np.ndarray, class_to_category_id: Dict[int, int]
) -> np.ndarray:
    """Map model class indices to COCO/VisDrone category ids.

    This is a pure lookup -- the actual VisDrone class-index-to-category-id
    mapping is repository-specific and must be supplied by the caller (see
    the `CLASS_TO_CATEGORY_ID` adapter constant in evaluate.py).

    Args:
        class_indices: (N,) array of integer class indices produced by the
            model / decoding step.
        class_to_category_id: mapping from class index to COCO category id.

    Returns:
        (N,) int64 array of category ids.

    Raises:
        KeyError: if a class index has no entry in `class_to_category_id`.
    """
    class_indices = np.asarray(class_indices)
    if class_indices.size == 0:
        return np.zeros((0,), dtype=np.int64)
    try:
        return np.array(
            [class_to_category_id[int(idx)] for idx in class_indices], dtype=np.int64
        )
    except KeyError as exc:
        raise KeyError(
            f"Class index {exc.args[0]} has no entry in class_to_category_id. "
            "Update the mapping to cover every class your model can predict."
        ) from exc


def evaluate_map(coco_gt: COCO, predictions: Sequence[Dict]) -> Tuple[float, float]:
    """Run pycocotools COCOeval and return (mAP@0.50, mAP@0.50:0.95).

    Args:
        coco_gt: ground-truth COCO object (see the `build_coco_ground_truth`
            adapter in evaluate.py).
        predictions: list of COCO-format prediction dicts.

    Returns:
        (map50, map50_95). Returns (0.0, 0.0) without calling pycocotools if
        `predictions` is empty (pycocotools' `loadRes` errors on an empty list).
    """
    if len(predictions) == 0:
        return 0.0, 0.0

    with contextlib.redirect_stdout(io.StringIO()):
        coco_dt = coco_gt.loadRes(list(predictions))
        coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

    map50_95 = float(coco_eval.stats[0])
    map50 = float(coco_eval.stats[1])
    return map50, map50_95


def _xywh_to_xyxy(box: Sequence[float]) -> Tuple[float, float, float, float]:
    x, y, w, h = box
    return (x, y, x + w, y + h)


def _box_iou(
    box_a: Tuple[float, float, float, float], box_b: Tuple[float, float, float, float]
) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area
    return inter_area / union if union > 0 else 0.0


def compute_precision_recall(
    coco_gt: COCO, predictions: Sequence[Dict], iou_threshold: float = 0.5
) -> Tuple[float, float]:
    """Compute a single precision/recall pair at a fixed IoU threshold.

    COCO's AP metric is computed from a full precision-recall curve, so
    there's no single canonical "precision"/"recall" scalar inside
    `COCOeval` itself. This function instead does per-image, per-class
    greedy IoU matching between the (already score-filtered) `predictions`
    and ground truth:

        precision = TP / (TP + FP)
        recall    = TP / (TP + FN)

    These numbers are only meaningful together with whatever score
    threshold produced `predictions` -- lowering the threshold trades
    recall for precision.

    Args:
        coco_gt: ground-truth COCO object.
        predictions: COCO-format prediction dicts, already filtered to the
            confidence threshold you want precision/recall reported at.
        iou_threshold: IoU at/above which a prediction counts as matching a
            ground-truth box.

    Returns:
        (precision, recall). Returns (0.0, 0.0) if there is neither a
        prediction nor a ground-truth box in the dataset.
    """
    preds_by_image: Dict[int, List[Dict]] = {}
    for pred in predictions:
        preds_by_image.setdefault(pred["image_id"], []).append(pred)

    total_tp = total_fp = total_fn = 0

    for image_id in coco_gt.getImgIds():
        gt_anns = coco_gt.loadAnns(coco_gt.getAnnIds(imgIds=image_id))
        image_preds = sorted(
            preds_by_image.get(image_id, []), key=lambda p: p["score"], reverse=True
        )

        gt_by_cat: Dict[int, List[Dict]] = {}
        for ann in gt_anns:
            gt_by_cat.setdefault(ann["category_id"], []).append(ann)
        preds_by_cat: Dict[int, List[Dict]] = {}
        for pred in image_preds:
            preds_by_cat.setdefault(pred["category_id"], []).append(pred)

        for cat_id in set(gt_by_cat) | set(preds_by_cat):
            gts = gt_by_cat.get(cat_id, [])
            preds = preds_by_cat.get(cat_id, [])
            gt_boxes = [_xywh_to_xyxy(g["bbox"]) for g in gts]
            matched_gt = [False] * len(gts)

            for pred in preds:
                pred_box = _xywh_to_xyxy(pred["bbox"])
                best_iou, best_idx = 0.0, -1
                for idx, gt_box in enumerate(gt_boxes):
                    if matched_gt[idx]:
                        continue
                    iou = _box_iou(pred_box, gt_box)
                    if iou > best_iou:
                        best_iou, best_idx = iou, idx
                if best_idx >= 0 and best_iou >= iou_threshold:
                    matched_gt[best_idx] = True
                    total_tp += 1
                else:
                    total_fp += 1

            total_fn += matched_gt.count(False)

    denom_p = total_tp + total_fp
    denom_r = total_tp + total_fn
    precision = total_tp / denom_p if denom_p > 0 else 0.0
    recall = total_tp / denom_r if denom_r > 0 else 0.0
    return precision, recall


def compute_detection_metrics(
    coco_gt: COCO, predictions: Sequence[Dict], iou_threshold: float = 0.5
) -> DetectionMetrics:
    """Combine `evaluate_map` and `compute_precision_recall` into one result."""
    map50, map50_95 = evaluate_map(coco_gt, predictions)
    precision, recall = compute_precision_recall(coco_gt, predictions, iou_threshold)
    return DetectionMetrics(
        map50=map50, map50_95=map50_95, precision=precision, recall=recall
    )


class LatencyTracker:
    """Accumulates per-iteration timings for named pipeline stages.

    Usage:
        tracker = LatencyTracker()
        for _ in range(iterations):
            tracker.record("preprocessing", pre_ms)
            tracker.record("model_inference", infer_ms)
            tracker.record("postprocessing", post_ms)
        summary = tracker.summarize("model_inference")
    """

    def __init__(self) -> None:
        self._timings: Dict[str, List[float]] = {}

    def record(self, stage: str, milliseconds: float) -> None:
        self._timings.setdefault(stage, []).append(float(milliseconds))

    def raw(self, stage: str) -> List[float]:
        return list(self._timings.get(stage, []))

    def summarize(self, stage: str) -> LatencyMetrics:
        """Compute mean/P50/P95/FPS for a recorded stage.

        Raises:
            ValueError: if no timings were recorded for `stage`.
        """
        samples = self._timings.get(stage)
        if not samples:
            raise ValueError(f"No timings recorded for stage '{stage}'")
        return compute_latency_metrics(samples)


def compute_latency_metrics(samples_ms: Sequence[float]) -> LatencyMetrics:
    """Compute mean/P50/P95/FPS from a list of per-iteration latencies (ms).

    Raises:
        ValueError: if `samples_ms` is empty.
    """
    if len(samples_ms) == 0:
        raise ValueError("samples_ms must be non-empty")
    arr = np.asarray(samples_ms, dtype=np.float64)
    mean_ms = float(np.mean(arr))
    p50_ms = float(np.percentile(arr, 50))
    p95_ms = float(np.percentile(arr, 95))
    fps = 1000.0 / mean_ms if mean_ms > 0 else 0.0
    return LatencyMetrics(mean_ms=mean_ms, p50_ms=p50_ms, p95_ms=p95_ms, fps=fps)