import torch
import numpy as np


class YOLOLoss(torch.nn.Module):
    def __init__(self,delta_noobj=0.5, delta_obj=1.0, delta_coord=5., delta_cls=1.0):
        super(YOLOLoss, self).__init__()
        self.delta_noobj = delta_noobj
        self.delta_obj = delta_obj
        self.delta_coord = delta_coord
        self.delta_cls = delta_cls

    def forward(self, prediction, target):
        batch_size = target.shape[0]
        # get the objectness loss for negative samples
        negative_samples = torch.nonzero(target[:, :, 4] < 0.5)
        ng_conf_pred = torch.sigmoid(prediction[negative_samples[:, 0], negative_samples[:, 1], 4])
        ng_conf_gt = target[negative_samples[:, 0], negative_samples[:, 1], 4]

        no_objectness_loss = self.delta_noobj * torch.nn.MSELoss(size_average=False)(ng_conf_pred, ng_conf_gt)
        total_loss = no_objectness_loss

        if negative_samples.shape[1] == target.shape[1]:
            return total_loss

        # loss only for cells with objects
        # get the objectness loss
        positiv_samples = torch.nonzero(target[:, :, 4] > 0.5)
        pt_conf_pred = torch.sigmoid(prediction[positiv_samples[:, 0], positiv_samples[:, 1], 4])
        pt_conf_gt = target[positiv_samples[:, 0], positiv_samples[:, 1], 4]
        objectness_loss = self.delta_obj * torch.nn.MSELoss(size_average=False)(pt_conf_pred, pt_conf_gt)
        total_loss += objectness_loss

        object_box_inds = torch.nonzero(target[:, :, 4] > 0).view(-1, 2)
        gt_ob = target[object_box_inds[:, 0], object_box_inds[:, 1]]
        pred_ob = prediction[object_box_inds[:, 0], object_box_inds[:, 1]]

        # get centre x and centre y
        centre_x_loss = self.delta_coord * torch.nn.MSELoss(size_average=False)(pred_ob[:, 0], gt_ob[:, 0])
        centre_y_loss = self.delta_coord * torch.nn.MSELoss(size_average=False)(pred_ob[:, 1], gt_ob[:, 1])

        coord_loss = centre_x_loss
        coord_loss += centre_y_loss

        # get w,h loss
        w_loss = self.delta_coord * torch.nn.MSELoss(size_average=False)(pred_ob[:, 2], gt_ob[:, 2])
        h_loss = self.delta_coord * torch.nn.MSELoss(size_average=False)(pred_ob[:, 3], gt_ob[:, 3])

        coord_loss += w_loss
        coord_loss += h_loss
        total_loss += coord_loss

        # class_loss
        cls_scores_pred = pred_ob[:, 5]
        cls_scores_target = gt_ob[:, 5]
        cls_loss = self.delta_cls * torch.nn.BCEWithLogitsLoss()(cls_scores_pred, cls_scores_target)
        total_loss += cls_loss

        return total_loss

