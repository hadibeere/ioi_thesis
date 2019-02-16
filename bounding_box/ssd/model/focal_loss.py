import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.25, reduce=False):
        """
            focusing is parameter that can adjust the rate at which easy
            examples are down-weighted.
            alpha may be set by inverse class frequency or treated as a hyper-param
            If you don't want to balance factor, set alpha to 1
            If you don't want to focusing factor, set gamma to 1
            which is same as normal cross entropy loss
        """
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduce = reduce

    def forward(self, confidence, predicted_locations, labels, gt_locations):
        """Compute classification loss and smooth l1 loss.

        Args:
            confidence (batch_size, num_priors, num_classes): class predictions.
            locations (batch_size, num_priors, 4): predicted locations.
            labels (batch_size, num_priors): real labels of all the priors.
            gt_locations (batch_size, num_priors, 4): real boxes corresponding all the priors.
        """
        num_classes = confidence.size(2)

        tmp_conf = confidence.reshape(-1, num_classes)
        tmp_label = labels

        pt_log = F.binary_cross_entropy_with_logits(tmp_conf, tmp_label, reduce=self.reduce)

        pt = torch.exp(-pt_log)
        classification_loss = self.alpha * (1 - pt) ** self.gamma * pt_log
        pos_mask = labels > 0
        predicted_locations = predicted_locations[pos_mask, :].reshape(-1, 4)
        gt_locations = gt_locations[pos_mask, :].reshape(-1, 4)
        smooth_l1_loss = F.smooth_l1_loss(predicted_locations, gt_locations, size_average=False)
        num_pos = gt_locations.size(0)
        return smooth_l1_loss/num_pos, classification_loss/num_pos