import torch
import torch.nn as nn


class FCOSHead(nn.Module):
    """
    FCOS detection head shared across all FPN levels.

    Input:
        List of FPN feature maps:
        [
            [B, 256, H3, W3],
            [B, 256, H4, W4],
            [B, 256, H5, W5],
            [B, 256, H6, W6],
        ]

    Output:
        A list of dictionaries, one per FPN level:
        {
            "cls_logits": [B, num_classes, H, W],
            "bbox_reg":   [B, 4, H, W],
            "centerness": [B, 1, H, W],
        }
    """

    def __init__(
        self,
        in_channels=256,
        num_classes=10,
        num_convs=4,
        prior_prob=0.01,
    ):
        super().__init__()

        self.num_classes = num_classes

        # Shared convolutional tower.
        cls_tower = []

        for i in range(num_convs):
            cls_tower.append(
                nn.Conv2d(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                )
            )
            cls_tower.append(nn.ReLU(inplace=True))

        self.cls_tower = nn.Sequential(*cls_tower)

        # Shared regression tower.
        bbox_tower = []

        for i in range(num_convs):
            bbox_tower.append(
                nn.Conv2d(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                )
            )
            bbox_tower.append(nn.ReLU(inplace=True))

        self.bbox_tower = nn.Sequential(*bbox_tower)

        # Prediction layers.
        self.cls_logits = nn.Conv2d(
            in_channels,
            num_classes,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.bbox_reg = nn.Conv2d(
            in_channels,
            4,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.centerness = nn.Conv2d(
            in_channels,
            1,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self._initialize_weights(prior_prob)

    def _initialize_weights(self, prior_prob):
        """
        FCOS-style initialization.
        """

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(
                    module.weight,
                    std=0.01,
                )

                if module.bias is not None:
                    nn.init.constant_(
                        module.bias,
                        0,
                    )

        # Initialize classification bias so that the initial
        # probability of a positive class is small.
        bias_value = -torch.log(
            torch.tensor(
                (1.0 - prior_prob) / prior_prob
            )
        )

        nn.init.constant_(
            self.cls_logits.bias,
            bias_value.item(),
        )

    def forward(self, features):
        """
        Args:
            features:
                List of four FPN feature maps.

        Returns:
            List of four dictionaries containing:
                cls_logits
                bbox_reg
                centerness
        """

        outputs = []

        for feature in features:
            cls_features = self.cls_tower(feature)
            bbox_features = self.bbox_tower(feature)

            cls_logits = self.cls_logits(cls_features)

            bbox_reg = self.bbox_reg(bbox_features)

            # Regression distances must be positive.
            #
            # FIX: torch.relu(bbox_reg) previously produced two
            # compounding problems:
            #   1. ~50% of output units start with a negative
            #      pre-activation (small std=0.01 init, zero bias)
            #      and are permanently dead under ReLU -- zero
            #      gradient forever, since ReLU'(x)=0 for x<0.
            #   2. The surviving positive units start at ~1e-2,
            #      four orders of magnitude below the pixel-scale
            #      l/t/r/b targets (tens to hundreds of pixels),
            #      with no exponential/scale term to bridge that
            #      gap.
            #
            # Together these collapsed every predicted box to a
            # near-zero-area point at its feature location, so
            # DIoU loss floored near 1.0 and never improved even
            # though gradients were technically flowing.
            #
            # torch.exp has no dead zone (gradient is nonzero
            # everywhere) and naturally represents a wide positive
            # range from a near-zero input. The clamp keeps the
            # exponential numerically stable early in training
            # (bounds output to ~[0.0025, 403] px, which comfortably
            # covers all four FPN levels' regression ranges).
            bbox_reg = torch.exp(bbox_reg.clamp(min=-6.0, max=6.0))

            centerness = self.centerness(bbox_features)

            outputs.append(
                {
                    "cls_logits": cls_logits,
                    "bbox_reg": bbox_reg,
                    "centerness": centerness,
                }
            )

        return outputs