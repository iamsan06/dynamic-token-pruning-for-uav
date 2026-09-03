import torch


class FCOSTargetAssigner:
    """
    FCOS target assignment for one image.

    FPN levels:
        P3: stride 4,  object range [0, 64]
        P4: stride 8,  object range [64, 128]
        P5: stride 16, object range [128, 256]
        P6: stride 32, object range [256, inf]

    Bounding boxes are expected as:
        [x1, y1, x2, y2]

    Labels are integer class IDs:
        0 ... num_classes-1
    """

    def __init__(
        self,
        strides=(4, 8, 16, 32),
        object_sizes=(
            (0, 64),
            (64, 128),
            (128, 256),
            (256, float("inf")),
        ),
        center_sampling_radius=1.5,
    ):
        self.strides = strides
        self.object_sizes = object_sizes
        self.center_sampling_radius = center_sampling_radius

    def _locations(self, height, width, stride, device):
        """
        Generate feature-map locations in input-image coordinates.

        Each location corresponds to the center of one feature cell.
        """

        shifts_x = (
            torch.arange(
                width,
                device=device,
                dtype=torch.float32,
            )
            + 0.5
        ) * stride

        shifts_y = (
            torch.arange(
                height,
                device=device,
                dtype=torch.float32,
            )
            + 0.5
        ) * stride

        yy, xx = torch.meshgrid(
            shifts_y,
            shifts_x,
            indexing="ij",
        )

        return torch.stack(
            [xx.reshape(-1), yy.reshape(-1)],
            dim=1,
        )

    def _assign_level(
        self,
        height,
        width,
        stride,
        size_range,
        boxes,
        labels,
    ):
        device = boxes.device

        locations = self._locations(
            height,
            width,
            stride,
            device,
        )

        num_locations = locations.shape[0]

        # No objects in the image.
        if boxes.shape[0] == 0:
            return (
                locations,
                torch.full(
                    (num_locations,),
                    -1,
                    dtype=torch.long,
                    device=device,
                ),
                torch.zeros(
                    (num_locations, 4),
                    dtype=torch.float32,
                    device=device,
                ),
                torch.zeros(
                    (num_locations,),
                    dtype=torch.float32,
                    device=device,
                ),
            )

        x = locations[:, 0]
        y = locations[:, 1]

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        # Distances from every location to every GT box.
        left = x[:, None] - x1[None, :]
        top = y[:, None] - y1[None, :]
        right = x2[None, :] - x[:, None]
        bottom = y2[None, :] - y[:, None]

        ltrb = torch.stack(
            [left, top, right, bottom],
            dim=2,
        )

        # A location must be inside the GT box.
        inside_box = ltrb.min(dim=2).values > 0

        # Object size = maximum of l,t,r,b.
        max_regression = ltrb.max(dim=2).values

        min_size, max_size = size_range

        size_ok = (
            (max_regression >= min_size)
            & (max_regression < max_size)
        )

        # ---------------------------------------------------------
        # Center sampling
        # ---------------------------------------------------------

        radius = self.center_sampling_radius * stride

        center_x = (
            (x1 + x2) / 2.0
        )

        center_y = (
            (y1 + y2) / 2.0
        )

        center_x1 = torch.maximum(
            center_x - radius,
            x1,
        )

        center_y1 = torch.maximum(
            center_y - radius,
            y1,
        )

        center_x2 = torch.minimum(
            center_x + radius,
            x2,
        )

        center_y2 = torch.minimum(
            center_y + radius,
            y2,
        )

        inside_center = (
            (x[:, None] >= center_x1[None, :])
            & (x[:, None] <= center_x2[None, :])
            & (y[:, None] >= center_y1[None, :])
            & (y[:, None] <= center_y2[None, :])
        )

        candidate = (
            inside_box
            & size_ok
            & inside_center
        )

        # ---------------------------------------------------------
        # Resolve locations matching multiple GT boxes.
        #
        # FCOS assigns the smallest-area matching GT box.
        # ---------------------------------------------------------

        areas = (
            (x2 - x1)
            * (y2 - y1)
        )

        areas = areas[None, :].expand(
            num_locations,
            -1,
        )

        areas = areas.masked_fill(
            ~candidate,
            float("inf"),
        )

        min_area, matched_gt = areas.min(dim=1)

        positive = torch.isfinite(min_area)

        cls_targets = torch.full(
            (num_locations,),
            -1,
            dtype=torch.long,
            device=device,
        )

        bbox_targets = torch.zeros(
            (num_locations, 4),
            dtype=torch.float32,
            device=device,
        )

        centerness_targets = torch.zeros(
            (num_locations,),
            dtype=torch.float32,
            device=device,
        )

        if positive.any():
            positive_indices = positive.nonzero(
                as_tuple=False
            ).squeeze(1)

            matched = matched_gt[positive_indices]

            cls_targets[positive_indices] = labels[matched]

            bbox_targets[positive_indices] = (
                ltrb[
                    positive_indices,
                    matched,
                ]
            )

            pos_ltrb = bbox_targets[
                positive_indices
            ]

            left_right = (
                pos_ltrb[:, [0, 2]]
            )

            top_bottom = (
                pos_ltrb[:, [1, 3]]
            )

            lr_ratio = (
                left_right.min(dim=1).values
                / left_right.max(dim=1).values.clamp(min=1e-8)
            )

            tb_ratio = (
                top_bottom.min(dim=1).values
                / top_bottom.max(dim=1).values.clamp(min=1e-8)
            )

            centerness_targets[
                positive_indices
            ] = torch.sqrt(
                lr_ratio * tb_ratio
            )

        return (
            locations,
            cls_targets,
            bbox_targets,
            centerness_targets,
        )

    def __call__(
        self,
        boxes,
        labels,
        feature_shapes,
    ):
        """
        Args:
            boxes:
                [N, 4]

            labels:
                [N]

            feature_shapes:
                List of (H, W), one for each FPN level.

        Returns:
            List of dictionaries, one per FPN level.
        """

        if boxes.ndim != 2 or boxes.shape[1] != 4:
            raise ValueError(
                f"boxes must have shape [N,4], got {boxes.shape}"
            )

        if labels.ndim != 1:
            raise ValueError(
                f"labels must have shape [N], got {labels.shape}"
            )

        if boxes.shape[0] != labels.shape[0]:
            raise ValueError(
                "boxes and labels must contain "
                "the same number of objects"
            )

        if len(feature_shapes) != len(self.strides):
            raise ValueError(
                "feature_shapes and strides must have "
                "the same length"
            )

        outputs = []

        for (
            feature_shape,
            stride,
            size_range,
        ) in zip(
            feature_shapes,
            self.strides,
            self.object_sizes,
        ):
            height, width = feature_shape

            outputs.append(
                self._assign_level(
                    height,
                    width,
                    stride,
                    size_range,
                    boxes,
                    labels,
                )
            )

        return outputs