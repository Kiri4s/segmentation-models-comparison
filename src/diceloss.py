import torch.nn as nn
import torch
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Implements Dice Loss for segmentation tasks.

    The Dice coefficient measures similarity between predicted segmentation and ground truth
    by computing spatial overlap between two binary masks.

    Args:
        smooth (float, optional): Small constant to avoid division by zero. Defaults to 1e-6.
    """

    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        """Computes Dice Loss between predictions and targets.

        Args:
            logits (torch.Tensor): Raw model outputs of shape (N, C, H, W), where:
                N is batch size, C is number of classes, H and W are spatial dimensions
            targets (torch.Tensor): Ground truth labels of shape (N, H, W) containing class indices

        Returns:
            torch.Tensor: Scalar Dice Loss value
        """
        probs = F.softmax(logits, dim=-3)
        targets_one_hot = F.one_hot(targets, num_classes=logits.shape[1]).permute(
            0, 3, 1, 2
        )

        intersection = torch.sum(probs * targets_one_hot, (0, 2, 3))
        cardinality = torch.sum(probs + targets_one_hot, (0, 2, 3))
        dice_per_class = (2.0 * intersection + self.smooth) / (
            cardinality + self.smooth
        )
        dice_loss = 1 - dice_per_class.mean()

        return dice_loss
