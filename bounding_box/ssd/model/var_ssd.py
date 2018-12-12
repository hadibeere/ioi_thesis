import torch.nn as nn
import torch
from torch.nn import Conv2d, Sequential, ModuleList, ReLU
import numpy as np
from typing import Tuple
import torch.nn.functional as F

from ssd.utils import box_utils, misc
from ssd.model.var_mobilenet import VarMobileNetV1
import logging
logger = logging.getLogger(__name__)

np.set_printoptions(threshold=np.nan)

class VarSSD(nn.Module):
    def __init__(self, num_classes: int, input_channels=3, is_test=False, config=None, device=None):
        """ Create default SSD model.
        """
        super(VarSSD, self).__init__()

        self.num_classes = num_classes
        self.base_net = VarMobileNetV1(config.start_fm*2**5, input_channels=input_channels, start_fm=config.start_fm).model
        self.source_layer_indexes = [
            12,
            14,
        ]
        self.extras = ModuleList([
            Sequential(
                Conv2d(in_channels=config.start_fm*2**5, out_channels=config.start_fm*2**3, kernel_size=1),
                ReLU(),
                Conv2d(in_channels=config.start_fm*2**3, out_channels=config.start_fm*2**4, kernel_size=3, stride=2, padding=1),
                ReLU()
            ),
            Sequential(
                Conv2d(in_channels=config.start_fm*2**4, out_channels=config.start_fm*2**2, kernel_size=1),
                ReLU(),
                Conv2d(in_channels=config.start_fm*2**2, out_channels=config.start_fm*2**3, kernel_size=3, stride=2, padding=1),
                ReLU()
            ),
            Sequential(
                Conv2d(in_channels=config.start_fm*2**3, out_channels=config.start_fm*2**2, kernel_size=1),
                ReLU(),
                Conv2d(in_channels=config.start_fm*2**2, out_channels=config.start_fm*2**3, kernel_size=3, stride=2, padding=1),
                ReLU()
            ),
            Sequential(
                Conv2d(in_channels=config.start_fm*2**3, out_channels=config.start_fm*2**2, kernel_size=1),
                ReLU(),
                Conv2d(in_channels=config.start_fm*2**2, out_channels=config.start_fm*2**3, kernel_size=3, stride=2, padding=1),
                ReLU()
            )
        ])

        assert(len(config.num_priors) == len(config.channels_priors))
        self.regression_headers = ModuleList()
        self.classification_headers = ModuleList()
        for ch, priors in zip(config.channels_priors, config.num_priors):
            self.regression_headers.append(Conv2d(in_channels=ch, out_channels=priors * 4, kernel_size=3, padding=1))
            self.classification_headers.append(Conv2d(in_channels=ch, out_channels=priors * num_classes, kernel_size=3, padding=1))

        self.is_test = is_test
        self.config = config

        # register layers in source_layer_indexes by adding them to a module list
        self.source_layer_add_ons = nn.ModuleList([t[1] for t in self.source_layer_indexes if isinstance(t, tuple)])
        if device:
            self.device = device
        else:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if is_test:
            self.config = config
            self.priors = config.priors.to(self.device)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        confidences = []
        locations = []
        start_layer_index = 0
        header_index = 0
        for end_layer_index in self.source_layer_indexes:
            if isinstance(end_layer_index, tuple):
                added_layer = end_layer_index[1]
                end_layer_index = end_layer_index[0]
            else:
                added_layer = None
            for layer in self.base_net[start_layer_index: end_layer_index]:
                x = layer(x)
            start_layer_index = end_layer_index
            if added_layer:
                y = added_layer(x)
            else:
                y = x
            confidence, location = self.compute_header(header_index, y)
            header_index += 1
            confidences.append(confidence)
            locations.append(location)

        for layer in self.base_net[end_layer_index:]:
            x = layer(x)

        for layer in self.extras:
            x = layer(x)
            confidence, location = self.compute_header(header_index, x)
            header_index += 1
            confidences.append(confidence)
            locations.append(location)

        confidences = torch.cat(confidences, 1)
        locations = torch.cat(locations, 1)

        if self.is_test:
            confidences = F.softmax(confidences, dim=2)
            boxes = box_utils.convert_locations_to_boxes(
                locations, self.priors, self.config.center_variance, self.config.size_variance
            )
            boxes = box_utils.center_form_to_corner_form(boxes)
            return confidences, boxes
        else:
            return confidences, locations

    def compute_header(self, i, x):
        confidence = self.classification_headers[i](x)
        confidence = confidence.permute(0, 2, 3, 1).contiguous()
        confidence = confidence.view(confidence.size(0), -1, self.num_classes)

        location = self.regression_headers[i](x)
        location = location.permute(0, 2, 3, 1).contiguous()
        location = location.view(location.size(0), -1, 4)

        return confidence, location

    def init_from_base_net(self, model):

        pretrained_dict = torch.load(model, map_location=lambda storage, loc: storage)
        model_dict = self.base_net.state_dict()

        # 1. filter out unnecessary keys
        pretrained_dict = {k[9:]: v for k, v in pretrained_dict.items() if k.startswith("base_net")}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        # 3. load the new state dict
        self.base_net.load_state_dict(pretrained_dict, strict=True)
        self.source_layer_add_ons.apply(_xavier_init_)
        self.extras.apply(_xavier_init_)
        self.classification_headers.apply(_xavier_init_)
        self.regression_headers.apply(_xavier_init_)

    def init_from_pretrained_ssd(self, model):
        state_dict = torch.load(model, map_location=lambda storage, loc: storage)
        state_dict = {k: v for k, v in state_dict.items() if
                      not (k.startswith("classification_headers") or k.startswith("regression_headers"))}
        model_dict = self.state_dict()
        model_dict.update(state_dict)
        self.load_state_dict(model_dict)
        self.classification_headers.apply(_xavier_init_)
        self.regression_headers.apply(_xavier_init_)

    def init(self):
        self.base_net.apply(_xavier_init_)
        self.source_layer_add_ons.apply(_xavier_init_)
        self.extras.apply(_xavier_init_)
        self.classification_headers.apply(_xavier_init_)
        self.regression_headers.apply(_xavier_init_)

    def load(self, model):
        state_dict = torch.load(model)
        self.load_state_dict(state_dict)

    def save(self, model_path):
        torch.save(self.state_dict(), model_path)


class MatchPrior(object):
    def __init__(self, center_form_priors, center_variance, size_variance, iou_threshold):
        self.center_form_priors = center_form_priors
        self.corner_form_priors = box_utils.center_form_to_corner_form(center_form_priors)
        self.center_variance = center_variance
        self.size_variance = size_variance
        self.iou_threshold = iou_threshold

    def __call__(self, gt_boxes, gt_labels):
        if type(gt_boxes) is np.ndarray:
            gt_boxes = torch.from_numpy(gt_boxes)
        if type(gt_labels) is np.ndarray:
            gt_labels = torch.from_numpy(gt_labels)

        boxes, labels = box_utils.assign_priors(gt_boxes, gt_labels,
                                                self.corner_form_priors, self.iou_threshold)
        boxes = box_utils.corner_form_to_center_form(boxes)
        locations = box_utils.convert_boxes_to_locations(boxes, self.center_form_priors, self.center_variance,
                                                         self.size_variance)
        return locations, labels


def _xavier_init_(m: nn.Module):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)
