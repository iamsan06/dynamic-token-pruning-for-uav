import torch

from models.backbone.swin import SwinBackbone
from models.neck.fpn import FPN


def test_fpn_forward():
    backbone = SwinBackbone(pretrained=False)
    fpn = FPN()
    backbone.eval()
    fpn.eval()

    x = torch.randn(1, 3, 640, 640)

    with torch.no_grad():
        features = backbone(x)
        outputs = fpn(features)

    assert len(outputs) == 4

    expected_shapes = [
        (1, 256, 160, 160),
        (1, 256, 80, 80),
        (1, 256, 40, 40),
        (1, 256, 20, 20),
    ]

    for output, expected in zip(outputs, expected_shapes):
        print("FPN output:", output.shape)
        assert output.shape == expected