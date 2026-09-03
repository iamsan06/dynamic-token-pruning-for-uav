import torch.nn as nn


class CenternessLoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        """
        Args:
            logits:  model centerness logits
            targets: centerness targets in [0, 1]

        Returns:
            Scalar BCE loss.
        """

        return self.loss(
            logits,
            targets,
        )