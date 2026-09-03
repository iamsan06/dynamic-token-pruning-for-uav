import numpy as np

from datasets.transforms import TrainTransforms


def test_train_transform():
    image = np.zeros((540, 960, 3), dtype=np.uint8)

    boxes = np.array([
        [100, 100, 200, 200],
        [400, 200, 500, 350],
    ], dtype=np.float32)

    labels = np.array([0, 1], dtype=np.int64)

    transform = TrainTransforms(size=640)

    output_image, output_boxes, output_labels = transform(
        image, boxes, labels
    )

    assert output_image.shape == (640, 640, 3)
    assert output_boxes.ndim == 2
    assert output_boxes.shape[1] == 4
    assert output_labels.ndim == 1
    assert output_labels.shape[0] == output_boxes.shape[0]

    assert np.all(output_boxes[:, 0] >= 0)
    assert np.all(output_boxes[:, 1] >= 0)
    assert np.all(output_boxes[:, 2] <= 640)
    assert np.all(output_boxes[:, 3] <= 640)