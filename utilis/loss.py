
import torch.nn as nn

class ClassificationLoss(nn.Module):
    """Binary cross-entropy for multi-label classification: one yes/no per event."""

    def __init__(self, pos_weight=None):
        super().__init__()

        self.loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(self, logits, target):

        return self.loss(logits, target.float())
