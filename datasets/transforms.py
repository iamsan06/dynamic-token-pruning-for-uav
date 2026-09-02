import cv2
import numpy as np
import random

class Letterbox:
    def __init__(self, size=640, pad_value=114):
        self.size = size
        self.pad_value = pad_value

    def __call__(self, image, boxes):
        """
        Resize image while preserving aspect ratio and pad to size x size.

        Args:
            image: RGB uint8 numpy array, shape [H, W, 3]
            boxes: numpy array, shape [N, 4], format [x1, y1, x2, y2]

        Returns:
            image: RGB uint8 numpy array, shape [size, size, 3]
            boxes: numpy array, shape [N, 4]
        """

        original_h, original_w = image.shape[:2]

        scale = min(
            self.size / original_w,
            self.size / original_h
        )

        new_w = int(round(original_w * scale))
        new_h = int(round(original_h * scale))

        resized = cv2.resize(
            image,
            (new_w, new_h),
            interpolation=cv2.INTER_LINEAR
        )

        canvas = np.full(
            (self.size, self.size, 3),
            self.pad_value,
            dtype=np.uint8
        )

        pad_x = (self.size - new_w) // 2
        pad_y = (self.size - new_h) // 2

        canvas[
            pad_y:pad_y + new_h,
            pad_x:pad_x + new_w
        ] = resized

        boxes = np.asarray(boxes, dtype=np.float32).copy()

        if len(boxes) > 0:
            boxes[:, [0, 2]] *= scale
            boxes[:, [1, 3]] *= scale

            boxes[:, [0, 2]] += pad_x
            boxes[:, [1, 3]] += pad_y

        return canvas, boxes


class TrainTransforms:
    def __init__(self, size=640):
        self.size = size
        self.letterbox = Letterbox(size=size)

    def __call__(self, image, boxes):
        image = image.copy()
        boxes = np.asarray(boxes, dtype=np.float32).copy()

        # Horizontal flip
        if random.random() < 0.5:
            image = np.ascontiguousarray(image[:, ::-1])

            if len(boxes) > 0:
                width = image.shape[1]
                x1 = boxes[:, 0].copy()
                x2 = boxes[:, 2].copy()

                boxes[:, 0] = width - x2
                boxes[:, 2] = width - x1

        # Scale jitter
        if random.random() < 0.4:
            scale = random.uniform(0.8, 1.2)

            h, w = image.shape[:2]

            new_w = max(1, int(w * scale))
            new_h = max(1, int(h * scale))

            image = cv2.resize(
                image,
                (new_w, new_h),
                interpolation=cv2.INTER_LINEAR
            )

            if len(boxes) > 0:
                boxes *= scale

        # Rotation
        if random.random() < 0.3:
            angle = random.uniform(-15.0, 15.0)

            h, w = image.shape[:2]
            center = (w / 2, h / 2)

            matrix = cv2.getRotationMatrix2D(
                center,
                angle,
                1.0
            )

            image = cv2.warpAffine(
                image,
                matrix,
                (w, h),
                borderValue=(114, 114, 114)
            )

            if len(boxes) > 0:
                corners = np.stack([
                    boxes[:, [0, 1]],
                    boxes[:, [2, 1]],
                    boxes[:, [2, 3]],
                    boxes[:, [0, 3]],
                ], axis=1)

                ones = np.ones(
                    (corners.shape[0], corners.shape[1], 1)
                )

                corners_h = np.concatenate(
                    [corners, ones],
                    axis=2
                )

                rotated = corners_h @ matrix.T

                x_min = rotated[:, :, 0].min(axis=1)
                y_min = rotated[:, :, 1].min(axis=1)
                x_max = rotated[:, :, 0].max(axis=1)
                y_max = rotated[:, :, 1].max(axis=1)

                boxes = np.stack(
                    [x_min, y_min, x_max, y_max],
                    axis=1
                )

        # Brightness / contrast
        if random.random() < 0.3:
            alpha = random.uniform(0.8, 1.2)
            beta = random.uniform(-20, 20)

            image = cv2.convertScaleAbs(
                image,
                alpha=alpha,
                beta=beta
            )

        # Final letterbox
        image, boxes = self.letterbox(image, boxes)

        # Clip boxes to image boundaries
        if len(boxes) > 0:
            boxes[:, [0, 2]] = np.clip(
                boxes[:, [0, 2]],
                0,
                self.size
            )

            boxes[:, [1, 3]] = np.clip(
                boxes[:, [1, 3]],
                0,
                self.size
            )

            # Remove boxes that became invalid
            valid = (
                (boxes[:, 2] > boxes[:, 0]) &
                (boxes[:, 3] > boxes[:, 1])
            )

            boxes = boxes[valid]

        return image, boxes