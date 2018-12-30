
from ssd.utils import box_utils

import torch
from yolo.util.bbox import bbox_iou


class StatCollector(object):
    def __init__(self, inp_dim, config, device="cpu"):
        self.true_positives = 0
        self.false_positives = 0
        self.false_negatives = 0
        self.inp_dim = inp_dim
        self.conf_threshold = config.conf_threshold
        self.nms_threshold = config.nms_threshold
        self.device = device
        self.iou_threshold = config.iou_threshold
        self.config = config

    def __call__(self, prediction_conf, prediction_loc, target_conf, target_loc):
        with torch.no_grad():
            abs_target = self.convert_to_absolute(target_conf.float()[:,:,None], target_loc).to(self.device)
            abs_prediction = self.convert_to_absolute(torch.softmax(prediction_conf, dim=2),
                                                      prediction_loc).to(self.device)
            batch_size = prediction_conf.shape[0]
            for batch in range(batch_size):
                # get stats for each image in batch
                positiv_samples = torch.nonzero(abs_prediction[batch, :, 5] > self.conf_threshold)  # get all predictions with confidence above threshold
                pred = abs_prediction[batch, positiv_samples[:,0], :]
                if pred.shape[0] > 0:
                    # nms
                    _, sort_idx = torch.sort(pred[:,5], descending=True)  # sort confidence descending
                    pred = pred[sort_idx]
                    for i in range(pred.shape[0]-1):
                        #get_ious, only 1 box should exist per image
                        try:
                            ious = bbox_iou(pred[i,0:4].unsqueeze(0), pred[i+1:,0:4], device=self.device)
                        except ValueError:
                            break
                        except IndexError:
                            break

                        # Zero out all the detections that have IoU > threshold
                        iou_mask = (ious <= self.nms_threshold).float().unsqueeze(1)
                        pred[i + 1:] *= iou_mask

                        # Remove the non-zero entries
                        non_zero_ind = torch.nonzero(pred[:, 5]).squeeze()
                        pred = pred[non_zero_ind].view(-1, 6)

                # gather stats

                positiv_samples = torch.nonzero(abs_target[batch, :, 4] > 0.5)  # get all class gt labels
                gt = abs_target[batch, positiv_samples, :].view(-1, 5)
                if gt.shape[0] == 0:
                    self.false_positives += pred.shape[0]
                elif pred.shape[0] == 0:
                    self.false_negatives += len(gt)
                else:
                    num_pos_matches = 0
                    for gt_box in gt:
                        ious = bbox_iou(gt_box[0:4].unsqueeze(0), pred[:,0:4], device=self.device)
                        pos_matches = torch.nonzero(ious > self.iou_threshold)
                        num_pos_matches += len(pos_matches)
                        if num_pos_matches == 0:
                            self.false_negatives += 1

                    num_false_matches = len(pred) - num_pos_matches
                    self.true_positives += num_pos_matches
                    self.false_positives += num_false_matches

    def convert_to_absolute(self, conf, locations):
        confidence = conf.to(self.device)
        boxes = box_utils.convert_locations_to_boxes(
            locations.to(self.device), self.config.priors.to(self.device), self.config.center_variance, self.config.size_variance
        )
        boxes = box_utils.center_form_to_corner_form(boxes)
        return torch.cat((boxes, confidence), 2)

    def precision(self):
        return self.true_positives / (self.true_positives + self.false_positives)

    def recall(self):
        return self.true_positives / (self.true_positives + self.false_negatives)
