# https://github.com/cedrickchee/ssd-yolo-retinanet/blob/master/models/Loss.py

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

    def forward(self, predictions, targets):
        """
            Args:
                predictions (tuple): (conf_preds, loc_preds)
                    conf_preds shape: [batch, n_anchors, num_cls]
                    loc_preds shape: [batch, n_anchors, 4]
                targets (tensor): (conf_targets, loc_targets)
                    conf_targets shape: [batch, n_anchors]
                    loc_targets shape: [batch, n_anchors, 4]
        """
        pt_log = F.binary_cross_entropy_with_logits(predictions[:,:,4], targets[:,:,4], reduce=self.reduce)

        pt = torch.exp(-pt_log)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * pt_log
        print(focal_loss.shape)
        total_loss = focal_loss.sum(dim=1)  # should be normalized by number of assigned anchors per ground truth --> always 1 or 0
        object_box_inds = torch.nonzero(targets[:, :, 4] > 0).view(-1, 2)
        if object_box_inds is None:
            if self.reduce:
                return torch.mean(total_loss)
            else:
                return total_loss

        gt_ob = targets[object_box_inds[:, 0], object_box_inds[:, 1]]
        pred_ob = predictions[object_box_inds[:, 0], object_box_inds[:, 1]]

        # get centre x and centre y
        centre_x_loss = torch.nn.MSELoss(size_average=self.reduce)(pred_ob[:, 0], gt_ob[:, 0])
        centre_y_loss = torch.nn.MSELoss(size_average=self.reduce)(pred_ob[:, 1], gt_ob[:, 1])

        #print("Num_gt:", gt_ob.shape[0])
        #print("Center_x_loss", float(centre_x_loss))
        #print("Center_y_loss", float(centre_y_loss))

        total_loss += centre_x_loss
        total_loss += centre_y_loss

        # get w,h loss
        w_loss = torch.nn.MSELoss(size_average=self.reduce)(pred_ob[:, 2], gt_ob[:, 2])
        h_loss = torch.nn.MSELoss(size_average=self.reduce)(pred_ob[:, 3], gt_ob[:, 3])

        total_loss += w_loss
        total_loss += h_loss

        if self.reduce:
            return torch.mean(total_loss)
        else:
            return total_loss

