import numpy as np

from datasets.transforms import Letterbox


def test_letterbox_output_shape():
    image = np.zeros((540, 960, 3), dtype=np.uint8)

    boxes = np.array([
        [100, 100, 200, 200]
    ], dtype=np.float32)

    transform = Letterbox(size=640)

    output_image, output_boxes = transform(image, boxes)

    assert output_image.shape == (640, 640, 3)
    assert output_boxes.shape == (1, 4)


def test_letterbox_box_coordinates():
    image = np.zeros((540, 960, 3), dtype=np.uint8)

    boxes = np.array([
        [100, 100, 200, 200]
    ], dtype=np.float32)

    transform = Letterbox(size=640)

    _, output_boxes = transform(image, boxes)

    # scale = 640 / 960 = 2/3
    # vertical padding = (640 - 360) / 2 = 140

    expected = np.array([
        [66.6667, 206.6667, 133.3333, 273.3333]
    ], dtype=np.float32)

    assert np.allclose(output_boxes, expected, atol=1e-3)