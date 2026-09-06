"""Unit tests for the evaluation package. Uses only synthetic data -- no
VisDrone dataset or trained checkpoint required.
"""

from __future__ import annotations

import numpy as np
import pytest
from pycocotools.coco import COCO

from evaluation.evaluate import class_aware_nms
from evaluation.metrics import (
    build_coco_predictions,
    compute_latency_metrics,
    compute_precision_recall,
    filter_degenerate_boxes,
    map_category_ids,
    xyxy_to_xywh,
)


# ---------------------------------------------------------------------------
# 1. xyxy -> xywh
# ---------------------------------------------------------------------------
def test_xyxy_to_xywh_basic():
    boxes = np.array([[10.0, 20.0, 30.0, 50.0]])
    result = xyxy_to_xywh(boxes)
    np.testing.assert_allclose(result, [[10.0, 20.0, 20.0, 30.0]])


def test_xyxy_to_xywh_empty():
    boxes = np.zeros((0, 4))
    result = xyxy_to_xywh(boxes)
    assert result.shape == (0, 4)


def test_xyxy_to_xywh_wrong_shape_raises():
    with pytest.raises(ValueError):
        xyxy_to_xywh(np.array([[1.0, 2.0, 3.0]]))


# ---------------------------------------------------------------------------
# 2. invalid / degenerate boxes
# ---------------------------------------------------------------------------
def test_filter_degenerate_boxes_drops_zero_width_height():
    boxes_xywh = np.array(
        [
            [0.0, 0.0, 10.0, 10.0],  # valid
            [0.0, 0.0, 0.0, 10.0],  # zero width
            [0.0, 0.0, 10.0, 0.0],  # zero height
            [0.0, 0.0, -5.0, 10.0],  # negative width
        ]
    )
    scores = np.array([0.9, 0.8, 0.7, 0.6])
    filtered_boxes, filtered_scores = filter_degenerate_boxes(boxes_xywh, scores)
    assert filtered_boxes.shape == (1, 4)
    np.testing.assert_allclose(filtered_boxes[0], [0.0, 0.0, 10.0, 10.0])
    np.testing.assert_allclose(filtered_scores, [0.9])


def test_filter_degenerate_boxes_empty_input():
    boxes_xywh = np.zeros((0, 4))
    scores = np.zeros((0,))
    filtered_boxes, filtered_scores = filter_degenerate_boxes(boxes_xywh, scores)
    assert filtered_boxes.shape == (0, 4)
    assert filtered_scores.shape == (0,)


# ---------------------------------------------------------------------------
# 3. prediction format
# ---------------------------------------------------------------------------
def test_build_coco_predictions_format():
    boxes_xyxy = np.array([[0.0, 0.0, 10.0, 20.0]])
    scores = np.array([0.87])
    category_ids = np.array([3])

    predictions = build_coco_predictions(
        image_id=42, boxes_xyxy=boxes_xyxy, scores=scores, category_ids=category_ids
    )

    assert len(predictions) == 1
    pred = predictions[0]
    assert set(pred.keys()) == {"image_id", "category_id", "bbox", "score"}
    assert pred["image_id"] == 42
    assert pred["category_id"] == 3
    assert pred["bbox"] == [0.0, 0.0, 10.0, 20.0]
    assert pred["bbox"][2] > 0 and pred["bbox"][3] > 0
    assert pytest.approx(pred["score"]) == 0.87


def test_build_coco_predictions_drops_degenerate():
    boxes_xyxy = np.array([[0.0, 0.0, 10.0, 20.0], [5.0, 5.0, 5.0, 20.0]])  # 2nd: zero width
    scores = np.array([0.9, 0.5])
    category_ids = np.array([1, 1])

    predictions = build_coco_predictions(0, boxes_xyxy, scores, category_ids)
    assert len(predictions) == 1


def test_build_coco_predictions_mismatched_lengths_raises():
    with pytest.raises(ValueError):
        build_coco_predictions(
            image_id=0,
            boxes_xyxy=np.zeros((2, 4)),
            scores=np.zeros((1,)),
            category_ids=np.zeros((2,)),
        )


# ---------------------------------------------------------------------------
# 4. empty detections
# ---------------------------------------------------------------------------
def test_build_coco_predictions_empty():
    predictions = build_coco_predictions(
        image_id=0,
        boxes_xyxy=np.zeros((0, 4)),
        scores=np.zeros((0,)),
        category_ids=np.zeros((0,)),
    )
    assert predictions == []


def test_class_aware_nms_empty_input():
    boxes = np.zeros((0, 4))
    scores = np.zeros((0,))
    classes = np.zeros((0,), dtype=np.int64)
    out_boxes, out_scores, out_classes = class_aware_nms(boxes, scores, classes, 0.5, 100)
    assert len(out_boxes) == 0
    assert len(out_scores) == 0
    assert len(out_classes) == 0


# ---------------------------------------------------------------------------
# 5. NMS behavior
# ---------------------------------------------------------------------------
def test_class_aware_nms_suppresses_overlapping_same_class():
    # Two heavily-overlapping boxes of the same class -> one should be suppressed.
    boxes = np.array(
        [
            [0.0, 0.0, 10.0, 10.0],
            [0.5, 0.5, 10.5, 10.5],  # near-identical box, lower score
        ]
    )
    scores = np.array([0.9, 0.8])
    classes = np.array([0, 0])

    out_boxes, out_scores, out_classes = class_aware_nms(boxes, scores, classes, 0.5, 100)

    assert len(out_boxes) == 1
    np.testing.assert_allclose(out_scores, [0.9])


def test_class_aware_nms_keeps_overlapping_different_classes():
    # Same overlapping boxes but different classes -> both should survive.
    boxes = np.array(
        [
            [0.0, 0.0, 10.0, 10.0],
            [0.5, 0.5, 10.5, 10.5],
        ]
    )
    scores = np.array([0.9, 0.8])
    classes = np.array([0, 1])

    out_boxes, out_scores, out_classes = class_aware_nms(boxes, scores, classes, 0.5, 100)

    assert len(out_boxes) == 2


def test_class_aware_nms_respects_max_detections():
    n = 10
    boxes = np.array([[i * 20.0, 0.0, i * 20.0 + 10.0, 10.0] for i in range(n)])
    scores = np.linspace(0.9, 0.1, n)
    classes = np.zeros(n, dtype=np.int64)

    out_boxes, out_scores, out_classes = class_aware_nms(boxes, scores, classes, 0.5, 3)

    assert len(out_boxes) == 3
    # highest-scoring boxes are non-overlapping here, so all top-3 survive
    np.testing.assert_allclose(out_scores, scores[:3])


# ---------------------------------------------------------------------------
# 6. percentile latency calculation
# ---------------------------------------------------------------------------
def test_compute_latency_metrics_basic():
    samples = [10.0, 20.0, 30.0, 40.0, 50.0]
    metrics = compute_latency_metrics(samples)
    assert pytest.approx(metrics.mean_ms) == 30.0
    assert pytest.approx(metrics.p50_ms) == 30.0
    assert pytest.approx(metrics.fps, rel=1e-6) == 1000.0 / 30.0


def test_compute_latency_metrics_p95_matches_numpy():
    samples = list(range(1, 101))  # 1..100 ms
    metrics = compute_latency_metrics(samples)
    expected_p95 = float(np.percentile(samples, 95))
    assert pytest.approx(metrics.p95_ms) == expected_p95


def test_compute_latency_metrics_empty_raises():
    with pytest.raises(ValueError):
        compute_latency_metrics([])


# ---------------------------------------------------------------------------
# 7. category ID conversion
# ---------------------------------------------------------------------------
def test_map_category_ids_basic():
    class_indices = np.array([0, 2, 1])
    mapping = {0: 10, 1: 11, 2: 12}
    result = map_category_ids(class_indices, mapping)
    np.testing.assert_array_equal(result, [10, 12, 11])


def test_map_category_ids_missing_key_raises():
    class_indices = np.array([0, 5])
    mapping = {0: 10}
    with pytest.raises(KeyError):
        map_category_ids(class_indices, mapping)


def test_map_category_ids_empty():
    result = map_category_ids(np.array([]), {})
    assert result.shape == (0,)


# ---------------------------------------------------------------------------
# Bonus: precision/recall matching, purely from synthetic COCO-format data
# (exercises the pycocotools-adjacent matching logic without needing a real
# dataset or model).
# ---------------------------------------------------------------------------
def _make_synthetic_coco_gt():
    coco_gt = COCO()
    coco_gt.dataset = {
        "images": [{"id": 1, "width": 100, "height": 100, "file_name": "img1.jpg"}],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [10.0, 10.0, 20.0, 20.0],
                "area": 400.0,
                "iscrowd": 0,
            },
            {
                "id": 2,
                "image_id": 1,
                "category_id": 1,
                "bbox": [50.0, 50.0, 20.0, 20.0],
                "area": 400.0,
                "iscrowd": 0,
            },
        ],
        "categories": [{"id": 1, "name": "pedestrian"}],
    }
    coco_gt.createIndex()
    return coco_gt


def test_compute_precision_recall_perfect_match():
    coco_gt = _make_synthetic_coco_gt()
    predictions = [
        {"image_id": 1, "category_id": 1, "bbox": [10.0, 10.0, 20.0, 20.0], "score": 0.95},
        {"image_id": 1, "category_id": 1, "bbox": [50.0, 50.0, 20.0, 20.0], "score": 0.90},
    ]
    precision, recall = compute_precision_recall(coco_gt, predictions, iou_threshold=0.5)
    assert pytest.approx(precision) == 1.0
    assert pytest.approx(recall) == 1.0


def test_compute_precision_recall_false_positive_and_miss():
    coco_gt = _make_synthetic_coco_gt()
    predictions = [
        {"image_id": 1, "category_id": 1, "bbox": [10.0, 10.0, 20.0, 20.0], "score": 0.95},
        # spurious detection far from any GT box
        {"image_id": 1, "category_id": 1, "bbox": [80.0, 80.0, 10.0, 10.0], "score": 0.60},
        # second GT box never detected
    ]
    precision, recall = compute_precision_recall(coco_gt, predictions, iou_threshold=0.5)
    assert pytest.approx(precision) == 0.5  # 1 TP, 1 FP
    assert pytest.approx(recall) == 0.5  # 1 TP, 1 FN


def test_compute_precision_recall_no_predictions():
    coco_gt = _make_synthetic_coco_gt()
    precision, recall = compute_precision_recall(coco_gt, [], iou_threshold=0.5)
    assert precision == 0.0
    assert recall == 0.0