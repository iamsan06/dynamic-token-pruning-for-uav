"""Evaluation package for the UAV token-pruning detector.

Public API re-exported here for convenience:
    from evaluation import compute_detection_metrics, LatencyTracker
"""

from evaluation.metrics import (
    DetectionMetrics,
    LatencyMetrics,
    LatencyTracker,
    compute_detection_metrics,
    compute_latency_metrics,
    xyxy_to_xywh,
)

__all__ = [
    "DetectionMetrics",
    "LatencyMetrics",
    "LatencyTracker",
    "compute_detection_metrics",
    "compute_latency_metrics",
    "xyxy_to_xywh",
]