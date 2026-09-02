import torch
import torch.nn as nn
import torch.nn.functional as F


class FPN(nn.Module):
    def __init__(
        self,
        in_channels=(96, 192, 384, 768),
        out_channels=256,
    ):
        super().__init__()

        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(c, out_channels, kernel_size=1)
            for c in in_channels
        ])

        self.output_convs = nn.ModuleList([
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
            )
            for _ in in_channels
        ])

    def forward(self, features):
        """
        Args:
            features:
                Swin outputs in NHWC format:
                [
                    [B, H/4,  W/4,  96],
                    [B, H/8,  W/8, 192],
                    [B, H/16, W/16, 384],
                    [B, H/32, W/32, 768],
                ]

        Returns:
            FPN features in NCHW format.
        """

        features = [
            x.permute(0, 3, 1, 2).contiguous()
            for x in features
        ]

        laterals = [
            conv(x)
            for conv, x in zip(
                self.lateral_convs,
                features,
            )
        ]

        for i in range(len(laterals) - 1, 0, -1):
            laterals[i - 1] = (
                laterals[i - 1]
                + F.interpolate(
                    laterals[i],
                    size=laterals[i - 1].shape[-2:],
                    mode="nearest",
                )
            )

        outputs = [
            conv(x)
            for conv, x in zip(
                self.output_convs,
                laterals,
            )
        ]

        return outputs