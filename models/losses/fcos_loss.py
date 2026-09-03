import torch
import torch.nn as nn

from models.losses.focal_loss import SigmoidFocalLoss
from models.losses.box_loss import DIoULoss
from models.losses.centerness_loss import CenternessLoss


class FCOSLoss(nn.Module):
    def __init__(
        self,
        lambda_cls=1.0,
        lambda_loc=2.0,
        lambda_ctr=1.0,
    ):
        super().__init__()

        self.lambda_cls = lambda_cls
        self.lambda_loc = lambda_loc
        self.lambda_ctr = lambda_ctr

        self.cls_loss = SigmoidFocalLoss(
            alpha=0.25,
            gamma=2.0,
        )

        self.box_loss = DIoULoss()
        self.centerness_loss = CenternessLoss()

    def forward(
        self,
        predictions,
        targets,
    ):
        """
        predictions:
            List of 4 FPN-level prediction dictionaries.

            predictions[level]["cls_logits"]:
                [B, C, H, W]

            predictions[level]["bbox_reg"]:
                [B, 4, H, W]

            predictions[level]["centerness"]:
                [B, 1, H, W]

        targets:
            List with one entry per image.

            targets[image][level] is:

                (
                    locations,
                    cls_targets,
                    bbox_targets,
                    centerness_targets,
                )

        Returns:
            total_loss, loss_info
        """

        batch_size = predictions[0]["cls_logits"].shape[0]

        if len(targets) != batch_size:
            raise ValueError(
                f"Expected {batch_size} image targets, "
                f"got {len(targets)}"
            )

        total_cls_loss = 0.0
        total_box_loss = 0.0
        total_ctr_loss = 0.0
        total_positive = 0

        # ---------------------------------------------------------
        # Process each FPN level
        # ---------------------------------------------------------

        for level, prediction in enumerate(predictions):

            cls_logits = prediction["cls_logits"]
            bbox_reg = prediction["bbox_reg"]
            centerness = prediction["centerness"]

            B, num_classes, H, W = cls_logits.shape

            # -----------------------------------------------------
            # Flatten predictions
            # -----------------------------------------------------

            cls_logits = cls_logits.permute(
                0, 2, 3, 1
            ).reshape(
                B,
                H * W,
                num_classes,
            )

            bbox_reg = bbox_reg.permute(
                0, 2, 3, 1
            ).reshape(
                B,
                H * W,
                4,
            )

            centerness = centerness.permute(
                0, 2, 3, 1
            ).reshape(
                B,
                H * W,
            )

            level_cls_loss = 0.0
            level_box_loss = 0.0
            level_ctr_loss = 0.0

            level_positive = 0

            # -----------------------------------------------------
            # Process every image independently
            # -----------------------------------------------------

            for image_idx in range(batch_size):

                (
                    locations,
                    cls_targets,
                    bbox_targets,
                    ctr_targets,
                ) = targets[image_idx][level]

                positive = cls_targets >= 0

                num_positive = positive.sum().item()
                level_positive += num_positive

                # -------------------------------------------------
                # Classification targets
                #
                # -1 = background
                #  0..9 = object class
                #
                # Background therefore becomes an all-zero
                # one-hot target.
                # -------------------------------------------------

                cls_target_one_hot = torch.zeros(
                    H * W,
                    num_classes,
                    device=cls_logits.device,
                    dtype=cls_logits.dtype,
                )

                if positive.any():
                    positive_indices = positive.nonzero(
                        as_tuple=False
                    ).squeeze(1)

                    cls_target_one_hot[
                        positive_indices,
                        cls_targets[positive],
                    ] = 1.0

                level_cls_loss += self.cls_loss(
                    cls_logits[image_idx],
                    cls_target_one_hot,
                )

                # -------------------------------------------------
                # Regression + centerness
                # -------------------------------------------------

                if positive.any():

                    positive_indices = positive.nonzero(
                        as_tuple=False
                    ).squeeze(1)

                    pred_ltrb = bbox_reg[
                        image_idx,
                        positive_indices,
                    ]

                    target_ltrb = bbox_targets[
                        positive_indices
                    ]

                    positive_locations = locations[
                        positive_indices
                    ]

                    pred_boxes = self._decode_boxes(
                        positive_locations,
                        pred_ltrb,
                    )

                    target_boxes = self._decode_boxes(
                        positive_locations,
                        target_ltrb,
                    )

                    level_box_loss += self.box_loss(
                        pred_boxes,
                        target_boxes,
                    )

                    pred_ctr = centerness[
                        image_idx,
                        positive_indices,
                    ]

                    target_ctr = ctr_targets[
                        positive_indices
                    ]

                    level_ctr_loss += self.centerness_loss(
                        pred_ctr,
                        target_ctr,
                    )

            total_cls_loss += level_cls_loss
            total_box_loss += level_box_loss
            total_ctr_loss += level_ctr_loss
            total_positive += level_positive

        # ---------------------------------------------------------
        # Avoid invalid regression loss when no objects exist.
        # ---------------------------------------------------------

        if total_positive == 0:
            zero = (
                predictions[0]["cls_logits"].sum() * 0.0
            )

            return zero, {
                "cls_loss": float(total_cls_loss.detach()),
                "box_loss": 0.0,
                "centerness_loss": 0.0,
                "num_positive": 0,
            }

        total_loss = (
            self.lambda_cls * total_cls_loss
            + self.lambda_loc * total_box_loss
            + self.lambda_ctr * total_ctr_loss
        )

        return total_loss, {
            "cls_loss": float(
                total_cls_loss.detach()
            ),
            "box_loss": float(
                total_box_loss.detach()
            ),
            "centerness_loss": float(
                total_ctr_loss.detach()
            ),
            "num_positive": total_positive,
        }

    @staticmethod
    def _decode_boxes(
        locations,
        ltrb,
    ):
        """
        Convert:

            left, top, right, bottom

        into:

            x1, y1, x2, y2
        """

        x = locations[:, 0]
        y = locations[:, 1]

        left = ltrb[:, 0]
        top = ltrb[:, 1]
        right = ltrb[:, 2]
        bottom = ltrb[:, 3]

        x1 = x - left
        y1 = y - top
        x2 = x + right
        y2 = y + bottom

        return torch.stack(
            [x1, y1, x2, y2],
            dim=1,
        )