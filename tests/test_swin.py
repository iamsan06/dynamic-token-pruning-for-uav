import torch

from models.backbone.swin import SwinBackbone


def test_swin_forward():
    model = SwinBackbone(pretrained=False)
    model.eval()

    x = torch.randn(1, 3, 640, 640)

    with torch.no_grad():
        outputs = model(x)

    assert len(outputs) == 4

    for output in outputs:
        print("Output shape:", output.shape)

    assert outputs[0].shape == (1, 160, 160, 96)
    assert outputs[1].shape == (1, 80, 80, 192)
    assert outputs[2].shape == (1, 40, 40, 384)
    assert outputs[3].shape == (1, 20, 20, 768)