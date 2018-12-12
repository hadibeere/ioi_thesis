import numpy as np
import cv2
from ssd.transforms.transforms import jaccard_numpy, DeNormalize, FromTensorToNumpy, Compose
from yolo.util.bbox import corner_to_center, center_to_corner, bbox_iou
import matplotlib
import os


class ToRelativeBoxWithLabel(object):
    """Convert ndarray of form (N,(xmin,ymin,xmax,ymax)) to (N,(class_id,center_x, center_y, width, height).
    The coordinates are relative to the image size.
    """
    def __call__(self, image, bbox, labels):
        h, w, _ = image.shape
        dh = 1.0 / h
        dw = 1.0 / w
        new_boxes = np.zeros((bbox.shape[0],5))
        for i in range(bbox.shape[0]):
            new_boxes[i,0] = labels[i]
            if labels[i] >= 0:
                new_boxes[i,1] = (bbox[i,0] + bbox[i,2]) / 2.0 * dw
                new_boxes[i,2] = (bbox[i,1] + bbox[i,3]) / 2.0 * dh
                new_boxes[i,3] = abs(bbox[i,2] - bbox[i,0]) * dw
                new_boxes[i,4] = abs(bbox[i,3] - bbox[i,1]) * dh

        return image, new_boxes, labels


class RandomSquareCrop(object):
    """Crop
    Arguments:
        img (Image): the image being input during training
        boxes (Tensor): the original bounding boxes in pt form
        labels (Tensor): the class labels for each bbox
        mode (float tuple): the min and max jaccard overlaps
    Return:
        (img, boxes, classes)
            img (Image): the cropped image
            boxes (Tensor): the adjusted bounding boxes in pt form
            labels (Tensor): the class labels for each bbox
    """
    def __init__(self):
        self.sample_options = (
            # using entire original input image
            #None,
            # sample a patch s.t. MIN jaccard w/ obj in .1,.3,.4,.7,.9
            #(0.1, None),
            #(0.3, None),
            (0.7, None),
            (0.9, None),
            # randomly sample a patch
            (None, None)
        )

    def __call__(self, image, boxes=None, labels=None):
        height, width, _ = image.shape
        while True:
            # randomly choose a mode
            mode = np.random.choice(self.sample_options)
            print("crop mode: " + str(mode))
            if mode is None:
                return image, boxes, labels

            min_iou, max_iou = mode
            if min_iou is None:
                min_iou = float('-inf')
            if max_iou is None:
                max_iou = float('inf')

            # max trails (50)
            for _ in range(50):
                current_image = image

                w = np.random.uniform(0.3 * width, width)
                h = w
                #h = np.random.uniform(0.3 * height, height)

                # aspect ratio constraint b/t .5 & 2
                #if h / w < 0.5 or h / w > 2:
                #    continue

                left = np.random.uniform(width - w)
                top = np.random.uniform(height - h)

                # convert to integer rect x1,y1,x2,y2
                rect = np.array([int(left), int(top), int(left+w), int(top+h)])

                # calculate IoU (jaccard overlap) b/t the cropped and gt boxes
                overlap = jaccard_numpy(boxes, rect)

                # is min and max overlap constraint satisfied? if not try again
                if overlap.min() < min_iou and max_iou < overlap.max():
                    continue

                # cut the crop from the image
                current_image = current_image[rect[1]:rect[3], rect[0]:rect[2],
                                              :]

                # keep overlap with gt box IF center in sampled patch
                centers = (boxes[:, :2] + boxes[:, 2:]) / 2.0

                # mask in all gt boxes that above and to the left of centers
                m1 = (rect[0] < centers[:, 0]) * (rect[1] < centers[:, 1])

                # mask in all gt boxes that under and to the right of centers
                m2 = (rect[2] > centers[:, 0]) * (rect[3] > centers[:, 1])

                # mask in that both m1 and m2 are true
                mask = m1 * m2

                # have any valid boxes? try again if not
                if not mask.any():
                    continue

                # take only matching gt boxes
                current_boxes = boxes[mask, :].copy()

                # take only matching gt labels
                current_labels = labels[mask]

                # should we use the box left and top corner or the crop's
                current_boxes[:, :2] = np.maximum(current_boxes[:, :2],
                                                  rect[:2])
                # adjust to crop (by substracting crop's left,top)
                current_boxes[:, :2] -= rect[:2]

                current_boxes[:, 2:] = np.minimum(current_boxes[:, 2:],
                                                  rect[2:])
                # adjust to crop (by substracting crop's left,top)
                current_boxes[:, 2:] -= rect[:2]

                return current_image, current_boxes, current_labels


class SquarePad(object):
    """Pad image to be square
    """
    def __init__(self, fill=0):
        self.fill = fill

    def __call__(self, image, bbox, labels):
        height, width, _ = image.shape
        desired_size = max(height, width)

        delta_w = desired_size - width
        delta_h = desired_size - height
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)
        new_img = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                    value=self.fill)
        bbox[:, 0] += left
        bbox[:, 1] += top
        bbox[:, 2] += right
        bbox[:, 3] += bottom
        return new_img, bbox, labels


class MatchAnchors(object):
    def __init__(self, anchors, num_anchors=[3,3,3], inp_dim=416, strides=[32,16,8], num_classes=1):
        self.inp_dim = inp_dim
        self.strides = strides
        self.anchor_nums = num_anchors
        self.num_classes = num_classes

        self.anchors = np.array(anchors)[::-1]
        self.num_pred_boxes = self.get_num_pred_boxes()
        self.box_strides = self.get_box_strides()

    def __call__(self, bbox, labels):
        ground_truth_map = np.zeros((sum(self.num_pred_boxes), 6), dtype = np.float)
        if labels[0] > 0:
            ground_truth = np.zeros((bbox.shape[0],5))
            ground_truth[:,:4] = bbox
            ground_truth[:,4] = labels
            ground_truth = corner_to_center(ground_truth[np.newaxis, :, :]).squeeze().reshape(-1, 5)
            label_table = self.get_pred_box_cords(ground_truth_map)
            # Get the bounding boxes to be assigned to the ground truth
            ground_truth_predictors = self.get_ground_truth_predictors(ground_truth, label_table)

            no_obj_cands = self.get_no_obj_candidates(ground_truth, label_table, ground_truth_predictors)

            ground_truth_predictors = ground_truth_predictors.squeeze(1)

            label_table[:, :2] //= self.box_strides
            label_table[:, [2, 3]] /= self.box_strides

            ground_truth_map = self.get_ground_truth_map(ground_truth, label_table, ground_truth_predictors,
                                                         no_obj_cands)

        return ground_truth_map

    def get_box_strides(self):
        box_strides = np.zeros((sum(self.num_pred_boxes), 1))
        offset = 0
        for i, x in enumerate(self.num_pred_boxes):
            box_strides[offset: offset + x] = self.strides[i]
            offset += x
        return box_strides

    def get_num_pred_boxes(self):
        """ Get number of prediction boxes per scale."""
        detection_map_dims = [(self.inp_dim // stride) for stride in self.strides]
        return [self.anchor_nums[i] * detection_map_dims[i] ** 2 for i in range(len(detection_map_dims))]

    def get_pred_box_cords(self, label_map):
        i = 0
        j = 0

        for n, pred_boxes in enumerate(self.num_pred_boxes):
            unit = self.strides[n]
            corners = np.arange(0, self.inp_dim, unit)
            offset = unit // 2
            grid = np.meshgrid(corners, corners)

            grid = np.concatenate((grid[0][:, :, np.newaxis], grid[1][:, :, np.newaxis]), 2).reshape(-1, 2)

            grid += offset
            grid = grid.repeat(self.anchor_nums[n], axis=0)
            label_map[i:i + pred_boxes, [0, 1]] = grid

            scale_anchors = np.array(self.anchors[j: j + self.anchor_nums[n]])

            num_boxes_in_scale = int(pred_boxes / self.anchor_nums[n])
            scale_anchors = scale_anchors.reshape(1, -1).repeat(num_boxes_in_scale, axis=0).reshape(-1, 2)

            label_map[i:i + pred_boxes, [2, 3]] = scale_anchors

            i += pred_boxes
            j += self.anchor_nums[n]
        return label_map

    def get_ground_truth_predictors(self, ground_truth, label_map):
        i = 0  # indexes the anchor boxes
        j = 0

        total_boxes_per_gt = sum(self.anchor_nums)
        total_num_gt_img = ground_truth.shape[0] # gt = ground truth
        inds = np.zeros((total_num_gt_img, total_boxes_per_gt), dtype=np.int)

        # n index the the detection maps
        for n, anchor in enumerate(self.anchor_nums):
            offset = sum(self.num_pred_boxes[:n])
            try:
                center_cells = (ground_truth[:, [0, 1]]) // self.strides[n]
            except:
                print(ground_truth)
                assert False

            a = offset + self.anchor_nums[n] * (self.inp_dim // self.strides[n] * center_cells[:, 1] + center_cells[:, 0])
            inds[:, sum(self.anchor_nums[:n])] = a
            for x in range(1, self.anchor_nums[n]):
                inds[:, sum(self.anchor_nums[:n]) + x] = a + x

            i += anchor
            j += self.num_pred_boxes[n]

        candidate_boxes = label_map[inds][:, :, :4]
        candidate_boxes = center_to_corner(candidate_boxes)
        candidate_boxes = candidate_boxes.transpose(0, 2, 1)

        ground_truth_boxes = center_to_corner(ground_truth.copy()[np.newaxis]).squeeze(0)[:, :4]
        ground_truth_boxes = ground_truth_boxes[:, :, np.newaxis]

        candidate_ious = bbox_iou(candidate_boxes, ground_truth_boxes, lib="numpy")

        prediction_boxes = np.zeros((total_num_gt_img, 1), dtype=np.int)

        for i in range(total_num_gt_img):
            # get the the row and the column of the highest IoU
            max_iou_ind = np.argmax(candidate_ious)
            max_iou_row = max_iou_ind // total_boxes_per_gt
            max_iou_col = max_iou_ind % total_boxes_per_gt

            # get the index (in label map) of the box with maximum IoU
            max_iou_box = inds[max_iou_row, max_iou_col]

            # assign the bounding box to the appropriate gt
            prediction_boxes[max_iou_row] = max_iou_box

            # zero out all the IoUs for this box so it can't be reassigned to any other gt
            box_mask = (inds != max_iou_ind).reshape(-1, 9)
            candidate_ious *= box_mask

            # zero out all the values of the row representing gt that just got assigned so that it
            # doesn't participate in the process again
            candidate_ious[max_iou_row] *= 0

        return prediction_boxes

    def get_no_obj_candidates(self, ground_truth, label_map, ground_truth_predictors):
        total_boxes_per_gt = sum(self.anchor_nums)
        num_ground_truth_in_im = ground_truth.shape[0]

        inds = np.arange(sum(self.num_pred_boxes)).astype(int)
        inds = inds[np.newaxis].repeat(num_ground_truth_in_im, axis=0)

        candidate_boxes = label_map[inds][:, :, :4]
        candidate_boxes = center_to_corner(candidate_boxes)
        candidate_boxes = candidate_boxes.transpose(0, 2, 1)

        ground_truth_boxes = center_to_corner(ground_truth.copy()[np.newaxis]).squeeze(0)[:, :4]
        ground_truth_boxes = ground_truth_boxes[:, :, np.newaxis]

        candidate_ious = bbox_iou(candidate_boxes, ground_truth_boxes, lib="numpy")
        candidate_ious[:, ground_truth_predictors] = 1

        max_ious_per_box = np.max(candidate_ious, 0)

        no_obj_cands = (np.nonzero(max_ious_per_box < 0.5)[0].astype(int))
        return no_obj_cands

    def get_ground_truth_map(self, ground_truth, label_map, ground_truth_predictors, no_obj_cands):
        # Set the objectness confidence of these boxes to 1
        label_map[:, 4] = -1
        predboxes = label_map[ground_truth_predictors]
        predboxes[:, 4] = 1

        label_map[no_obj_cands] = 0

        assert ground_truth_predictors.shape[0] == predboxes.shape[0], print("Shape mismatch get_ground_truth_map")

        ground_truth_strides = self.box_strides[ground_truth_predictors]
        ground_truth[:, :4] /= ground_truth_strides

        try:
            predboxes[:, [0, 1]] = ground_truth[:, [0, 1]] - predboxes[:, [0, 1]]

        except:
            assert False

        if 0 in predboxes[:, [0, 1]]:
            predboxes[:, [0, 1]] += 0.0001 * (predboxes[:, [0, 1]] == 0)

        predboxes[:, [0, 1]] = -1 * np.log(1 / (predboxes[:, [0, 1]]) - 1)

        mask = np.logical_and(ground_truth[:, 2], ground_truth[:, 3])
        mask = mask.reshape(-1, 1)

        ground_truth *= mask

        nz_inds = np.nonzero(ground_truth[:, 0])
        ground_truth = ground_truth[nz_inds]
        predboxes = predboxes[nz_inds]
        ground_truth_predictors = ground_truth_predictors[nz_inds]

        try:
            predboxes[:, [2, 3]] = np.log(ground_truth[:, [2, 3]] / predboxes[:, [2, 3]])

        except:
            assert False

        predboxes[:, 5] = ground_truth[:, 4]
        label_map[ground_truth_predictors] = predboxes
        return label_map


class PlotImg(object):
    def __init__(self, folder):
        self.counter = 0
        self.folder = folder

    def __call__(self, image, boxes=None, labels=None):
        convert = Compose([FromTensorToNumpy(), DeNormalize(2**12-1)])
        img, _, _ = convert(image.clone())
        if isinstance(boxes, np.ndarray):
            box = boxes.copy()[0]
            self.show_frames(img[:,:,0],box)
            self.counter += 1

        return image, boxes, labels

    def lower_form(self, bbox):
        coord = (bbox[0], bbox[1])
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        return coord, width, height

    def show_frames(self, frame, gt_box):
        """Show image with landmarks"""
        max_c = 2 ** 12 - 1
        fig, ax = matplotlib.pyplot.subplots()
        ax.imshow(frame, cmap='gray', vmin=0, vmax=max_c)
        # xy, width, height with xy lower left
        coord, width, height = self.lower_form(gt_box)
        rect = matplotlib.patches.Rectangle(coord, width, height, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        colors = ['blue', 'green', 'cyan', 'yellow']
        i = 0
        #for box in boxes:
        #    coord, width, height = lower_form(to_box(box))
        #    rect = patches.Rectangle(coord, width, height, linewidth=2, edgecolor=colors[i], facecolor='none')
        #    ax.add_patch(rect)
        #    i += 1

        items = ["ground truth"]
        #for j in range(i):
        #    items.append("box top " + str(j + 1))

        ax.legend(items, title="Legend", loc="upper left",
                  bbox_to_anchor=(1, 0, 0.5, 1))

        fig.savefig(os.path.join(self.folder, "frame_" + str(self.counter) + ".png"), dpi=150,
                                  bbox_inches='tight')
        matplotlib.pyplot.clf()
        matplotlib.pyplot.cla()
