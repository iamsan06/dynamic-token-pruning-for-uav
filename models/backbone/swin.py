import torch
import torch.nn as nn
import timm


class SwinBackbone(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()

        self.model = timm.create_model(
            "swin_tiny_patch4_window7_224",
            pretrained=pretrained,
            img_size=640,
            features_only=True,
        )

        self.last_attn = None

        # Access the four Swin stages inside FeatureListNet.
        stages = [
            self.model.layers_0,
            self.model.layers_1,
            self.model.layers_2,
            self.model.layers_3,
        ]

        for stage in stages:
            for block in stage.blocks:
                block.attn.fused_attn = False

                original_forward = block.attn.forward

                def forward_with_capture(
                    x,
                    mask=None,
                    _original_forward=original_forward,
                    _attn=block.attn,
                    _backbone=self,
                ):
                    B_, N, C = x.shape

                    qkv = (
                        _attn.qkv(x)
                        .reshape(
                            B_,
                            N,
                            3,
                            _attn.num_heads,
                            -1,
                        )
                        .permute(2, 0, 3, 1, 4)
                    )

                    q, k, v = qkv.unbind(0)

                    q = q * _attn.scale

                    attn = q @ k.transpose(-2, -1)

                    attn = attn + _attn._get_rel_pos_bias()

                    if mask is not None:
                        num_win = mask.shape[0]

                        attn = (
                            attn.view(
                                -1,
                                num_win,
                                _attn.num_heads,
                                N,
                                N,
                            )
                            + mask.unsqueeze(1).unsqueeze(0)
                        )

                        attn = attn.view(
                            -1,
                            _attn.num_heads,
                            N,
                            N,
                        )

                    attn = _attn.softmax(attn)

                    # Save the attention matrix.
                    _attn.last_attn = attn.detach()
                    _backbone.last_attn = attn.detach()

                    attn = _attn.attn_drop(attn)

                    x = attn @ v

                    x = (
                        x.transpose(1, 2)
                        .reshape(B_, N, -1)
                    )

                    x = _attn.proj(x)
                    x = _attn.proj_drop(x)

                    return x

                block.attn.last_attn = None
                block.attn.forward = forward_with_capture

    def forward(self, x):
        return self.model(x)