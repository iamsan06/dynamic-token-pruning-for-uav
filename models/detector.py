import torch
import torch.nn as nn

from models.backbone.swin import SwinBackbone
from models.neck.fpn import FPN
from models.head.fcos_head import FCOSHead


class UAVDetector(nn.Module):
    """
    Full UAV object detector:

        Image
          ↓
        Swin-T
          ↓
        FPN
          ↓
        FCOS Head

    Input:
        [B, 3, 640, 640]

    Outputs:
        Four FPN levels containing:
            cls_logits
            bbox_reg
            centerness
    """

    def __init__(
        self,
        num_classes=10,
        pretrained=True,
        fpn_channels=256,
    ):
        super().__init__()

        self.backbone = SwinBackbone(
            pretrained=pretrained
        )

        self.fpn = FPN(
            in_channels=(96, 192, 384, 768),
            out_channels=fpn_channels,
        )

        self.head = FCOSHead(
            in_channels=fpn_channels,
            num_classes=num_classes,
        )

    def forward(self, x):
        # Swin backbone
        backbone_features = self.backbone(x)

        # FPN
        fpn_features = self.fpn(backbone_features)

        # FCOS head
        outputs = self.head(fpn_features)

        return outputs