import torch
import numpy as np
from yolo.util.bbox import center_to_corner, bbox_iou


class StatCollectorTrain(object):
    def __init__(self, inp_dim, anchors, det_strides, conf_threshold=0.5, nms_threshold=0.0, iou_threshold=0.5, device="cpu"):
        self.true_positives = 0
        self.false_positives = 0
        self.false_negatives = 0
        self.inp_dim = inp_dim
        self.anchors = anchors
        self.det_strides = det_strides
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.device = device
        self.iou_threshold = iou_threshold

    def __call__(self, prediction, target):
        with torch.no_grad():
            tmp_pred = prediction.clone()
            self.sigmoid(tmp_pred)
            abs_prediction = center_to_corner(self.convert_to_absolute(tmp_pred))
            abs_target = center_to_corner(self.convert_to_absolute(target.clone()))
            batch_size = prediction.shape[0]
            for batch in range(batch_size):
                # get stats for each image in batch
                positiv_samples = torch.nonzero(abs_prediction[batch, :, 4] > self.conf_threshold)  # get all predictions with confidence above threshold
                pred = abs_prediction[batch, positiv_samples[:,0], :]
                if pred.shape[0] > 0:
                    # nms
                    _, sort_idx = torch.sort(pred[:,4], descending=True)  # sort confidence descending
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
                        non_zero_ind = torch.nonzero(pred[:, 4]).squeeze()
                        pred = pred[non_zero_ind].view(-1, 6)

                # gather stats

                positiv_samples = torch.nonzero(abs_target[batch, :, 5] > 0.5)  # get all class gt labels
                gt = abs_target[batch, positiv_samples[:], :]
                if gt.shape[0] == 0:
                    self.false_positives += pred.shape[0]
                elif pred.shape[0] == 0:
                    self.false_negatives += len(gt)
                else:
                    num_pos_matches = 0
                    for gt_box in gt:
                        ious = bbox_iou(gt_box[0:4], pred[:,0:4], device=self.device)
                        pos_matches = torch.nonzero(ious > self.iou_threshold)
                        num_pos_matches += len(pos_matches)
                        if len(pos_matches) == 0:
                            self.false_negatives += 1

                    num_false_matches = len(pred) - num_pos_matches
                    self.true_positives += num_pos_matches
                    self.false_positives += num_false_matches

    def convert_to_absolute(self, prediction):
        anchor_pos = 0
        num_anchors = 3
        pred_pos = 0
        for stride in self.det_strides:
            grid_size = self.inp_dim // stride
            anchors = [(a[0] / stride, a[1] / stride) for a in self.anchors[anchor_pos:anchor_pos+num_anchors]]
            anchor_pos += num_anchors
            # Add the center offsets
            grid_len = np.arange(grid_size)
            a, b = np.meshgrid(grid_len, grid_len)

            x_offset = torch.FloatTensor(a).view(-1, 1).to(self.device)
            y_offset = torch.FloatTensor(b).view(-1, 1).to(self.device)

            x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(1, num_anchors).view(-1, 2).unsqueeze(0)
            num_pred = grid_size * grid_size * num_anchors
            prediction[:, pred_pos:pred_pos+num_pred, :2] += x_y_offset

            # log space transform height and the width
            anchors = torch.FloatTensor(anchors).to(self.device)

            anchors = anchors.repeat(grid_size * grid_size, 1).unsqueeze(0)
            prediction[:, pred_pos:pred_pos+num_pred, 2:4] = torch.exp(prediction[:, pred_pos:pred_pos+num_pred, 2:4]) * anchors

            prediction[:, pred_pos:pred_pos+num_pred, :4] *= stride
            pred_pos += num_pred

        return prediction

    def sigmoid(self, prediction):
        # Sigmoid the  centre_X, centre_Y. and object confidence
        prediction[:, :, 0] = torch.sigmoid(prediction[:, :, 0])
        prediction[:, :, 1] = torch.sigmoid(prediction[:, :, 1])
        prediction[:, :, 4] = torch.sigmoid(prediction[:, :, 4])
        prediction[:, :, 5] = torch.sigmoid((prediction[:, :, 5]))

    def precision(self):
        return self.true_positives / (self.true_positives + self.false_positives)

    def recall(self):
        return self.true_positives / (self.true_positives + self.false_negatives)
