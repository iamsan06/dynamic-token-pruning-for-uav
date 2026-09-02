import torch

from models.backbone.swin import SwinBackbone


def test_swin_attention_capture():
    model = SwinBackbone(pretrained=False)
    model.eval()

    x = torch.randn(1, 3, 640, 640)

    with torch.no_grad():
        outputs = model(x)

    assert len(outputs) == 4

    # Find the final attention block.
    last_stage = model.model.layers_3
    last_block = last_stage.blocks[-1]

    attn = last_block.attn.last_attn

    assert attn is not None
    assert attn.ndim == 4

    print("Feature outputs:")
    for output in outputs:
        print(output.shape)

    print("Attention shape:", attn.shape)

    # Swin-T window size = 7x7 = 49 tokens.
    assert attn.shape[-1] == 49
    assert attn.shape[-2] == 49