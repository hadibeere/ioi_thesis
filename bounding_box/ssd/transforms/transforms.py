# from https://github.com/amdegroot/ssd.pytorch


import torch
from torchvision import transforms
import cv2
import numpy as np
import types
import random

import logging
logger = logging.getLogger(__name__)


def adjust_bbox(box):
    return np.array([min(box[0],box[2]), min(box[1],box[3]), max(box[0],box[2]), max(box[1],box[3])])


def intersect(box_a, box_b):
    max_xy = np.minimum(box_a[:, 2:], box_b[2:])
    min_xy = np.maximum(box_a[:, :2], box_b[:2])
    inter = np.clip((max_xy - min_xy), a_min=0, a_max=np.inf)
    return inter[:, 0] * inter[:, 1]


def jaccard_numpy(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: Multiple bounding boxes, Shape: [num_boxes,4]
        box_b: Single bounding box, Shape: [4]
    Return:
        jaccard overlap: Shape: [box_a.shape[0], box_a.shape[1]]
    """
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1]))  # [A,B]
    area_b = ((box_b[2]-box_b[0]) *
              (box_b[3]-box_b[1]))  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]


class Compose(object):
    """Composes several augmentations together.
    Args:
        transforms (List[Transform]): list of transforms to compose.
    Example:
        >>> augmentations.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, boxes=None, labels=None):
        for t in self.transforms:
            img, boxes, labels = t(img, boxes, labels)
        return img, boxes, labels


class Lambda(object):
    """Applies a lambda as a transform."""

    def __init__(self, lambd):
        assert isinstance(lambd, types.LambdaType)
        self.lambd = lambd

    def __call__(self, img, boxes=None, labels=None):
        return self.lambd(img, boxes, labels)


class ConvertFromInts(object):
    def __call__(self, image, boxes=None, labels=None):
        return image.astype(np.float32), boxes, labels


class SubtractMeans(object):
    def __init__(self, mean):
        self.mean = np.array(mean, dtype=np.float32)

    def __call__(self, image, boxes=None, labels=None):
        image = image.astype(np.float32)
        image -= self.mean
        return image.astype(np.float32), boxes, labels


class ToAbsoluteCoords(object):
    def __call__(self, image, boxes=None, labels=None):
        height, width = image.shape[:2]
        boxes[:, 0] *= width
        boxes[:, 2] *= width
        boxes[:, 1] *= height
        boxes[:, 3] *= height

        return image, boxes, labels


class ToPercentCoords(object):
    def __call__(self, image, boxes=None, labels=None):
        height, width  = image.shape[:2]
        boxes = boxes.astype(np.float32)
        boxes[:, 0] /= width
        boxes[:, 2] /= width
        boxes[:, 1] /= height
        boxes[:, 3] /= height

        return image, boxes, labels


class RandomSampleCrop(object):
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
            None,
            # sample a patch s.t. MIN jaccard w/ obj in .1,.3,.4,.7,.9
            (0.1, None),
            (0.3, None),
            (0.7, None),
            (0.9, None),
            # randomly sample a patch
            (None, None),
        )

    def __call__(self, image, boxes=None, labels=None):
        height, width = image.shape[:2]
        while True:
            # randomly choose a mode
            mode = np.random.choice(self.sample_options)
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
                h = np.random.uniform(0.3 * height, height)

                # aspect ratio constraint b/t .5 & 2
                if h / w < 0.5 or h / w > 2:
                    continue

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
                if current_image.ndim == 2:
                    current_image = current_image[rect[1]:rect[3], rect[0]:rect[2]]
                else:
                    current_image = current_image[rect[1]:rect[3], rect[0]:rect[2],:]

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


class Normalize(object):
    def __init__(self, depth):
        self.depth = depth

    def __call__(self, image, boxes=None, labels=None):
        image = image.astype(np.float32)
        image = np.divide(image, self.depth)
        return image, boxes, labels


class DeNormalize(object):
    def __init__(self, depth):
        self.depth = depth

    def __call__(self, image, boxes=None, labels=None):
        image = image.astype(np.float32)
        image = np.multiply(image, self.depth)
        return image, boxes, labels


class NormalizeMean(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, boxes=None, labels=None):
        submean = SubtractMeans(self.mean)
        image, _, _ = submean(image)
        normalize = Normalize(self.std)
        image, _, _ = normalize(image)
        return image, boxes, labels


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, image, bbox, labels):
        # numpy image: H x W x C
        # torch image: C X H X W
        if image.ndim == 2:
            image = np.expand_dims(image.astype(np.float32), axis=0)  # C x H x W
        else:
            image = image.astype(np.float32).transpose((2, 0, 1))
        return torch.from_numpy(image), bbox, labels


class FromTensorToNumpy(object):
    """Convert tensors to ndarrays"""
    def __call__(self, image, bbox, labels):
        # numpy image: H x W x C
        # torch image: C X H X W
        img = image.numpy().transpose((1, 2, 0))
        box = None
        lab = None
        if bbox:
            box = bbox.numpy()
        if labels:
            lab = labels.numpy()

        return img, box, lab


class Rotate(object):
    """Rotate image and bounding box around given angle.

    Args:
        angle (int): Rotate image by given angle.
        fill_color (int): Fill Background with given color
    """

    def __init__(self, angle, fill_color=0):
        assert isinstance(angle, int)
        self.angle = angle
        assert isinstance(fill_color, int)
        self.fill_color = fill_color

    def __call__(self, image, bboxes, labels):
        """ image: Dimensions must be height, width,(channels)
           bboxes (ndarray): bounding boxes (N,4) with (xmin,ymin,xmax,ymax)
        """
        rows, cols = image.shape[:2]
        new_bboxes = np.copy(bboxes)
        transformation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), self.angle, 1)
        trans_image = cv2.warpAffine(image, transformation_matrix, (cols, rows),borderMode=cv2.BORDER_CONSTANT,
                           borderValue=self.fill_color)
        for i in range(new_bboxes.shape[0]):
            xmin = bboxes[i, 0]
            xmax = bboxes[i, 2]
            ymin = bboxes[i, 1]
            ymax = bboxes[i, 3]
            box_4_corner = np.array([[[xmin, ymin], [xmin, ymax], [xmax, ymin], [xmax, ymax]]])
            trans_marks = cv2.transform(box_4_corner, transformation_matrix)[0]
            new_bboxes[i, :] = self.get_2_corner_form(trans_marks.reshape(4, 2))

        logger.info(
            f"Rotate angle: {self.angle}"
        )

        return trans_image, new_bboxes, labels

    def get_2_corner_form(self, box):
        min_pt = box.min(0)
        max_pt = box.max(0)
        return np.array([min_pt, max_pt]).reshape(-1)


class RandomRotate(object):
    """Rotate image and bounding box around given angle.

    Args:
        range (tuple): Rotate image by random angle in limit range.
        fill_color (int): Fill Background with given color
    """

    def __init__(self, range, fill_color):
        assert isinstance(range, tuple)
        self.range = range
        assert isinstance(fill_color, int)
        self.fill_color = fill_color

    def __call__(self, image, bboxes, labels):
        angle = int(round(random.uniform(self.range[0], self.range[1])))
        rot = Rotate(angle, self.fill_color)
        return rot(image, bboxes, labels)


class Resize(object):
    """Rescale the image in a sample to a given size and adjust bounding box.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, image, bboxes=None, labels=None):
        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        new_image = cv2.resize(image, (new_w, new_h))

        if bboxes is None:
            return new_image, bboxes, labels

        new_bboxes = np.copy(bboxes)
        ratio_w = new_w / w
        ratio_h = new_h / h
        for i in range(new_bboxes.shape[0]):
            new_bboxes[i,0] = int(round(new_bboxes[i,0] * ratio_w))
            new_bboxes[i,2] = int(round(new_bboxes[i,2] * ratio_w))
            new_bboxes[i,1] = int(round(new_bboxes[i,1] * ratio_h))
            new_bboxes[i,3] = int(round(new_bboxes[i,3] * ratio_h))

        return new_image, new_bboxes, labels


class RandomRescale(object):
    """Rescale the image in a sample by random factor and adjust bounding box.

    Args:
        scale_range (tuple): Desired scaling factor range(min,max).
    """

    def __init__(self, scale_range):
        assert isinstance(scale_range, tuple)
        self.scale_range = scale_range

    def __call__(self, image, bboxes, labels):
        h, w = image.shape[:2]
        scale = round(random.uniform(self.scale_range[0], self.scale_range[1]), 1)
        if scale <= 1.0:
            new_image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        else:
            new_image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

        new_h, new_w = new_image.shape[:2]
        new_bboxes = np.copy(bboxes)
        ratio_w = new_w / w
        ratio_h = new_h / h
        for i in range(new_bboxes.shape[0]):
            new_bboxes[i,0] = int(round(new_bboxes[i,0] * ratio_w))
            new_bboxes[i,2] = int(round(new_bboxes[i,2] * ratio_w))
            new_bboxes[i,1] = int(round(new_bboxes[i,1] * ratio_h))
            new_bboxes[i,3] = int(round(new_bboxes[i,3] * ratio_h))

        return new_image, new_bboxes, labels


class RandomShift(object):
    """Shift image and bbox randomly, so that bbox is still inside image."""

    def __init__(self):
        pass

    def __call__(self, image, bboxes, labels):
        h, w = image.shape[:2]
        new_bboxes = bboxes[:]
        shift_x = random.randint(-np.min(bboxes[:,0]), w - np.max(bboxes[:,2]))
        shift_y = random.randint(-np.min(bboxes[:,1]), h - np.max(bboxes[:,3]))
        new_bboxes[:,0] += shift_x
        new_bboxes[:,2] += shift_x
        new_bboxes[:,1] += shift_y
        new_bboxes[:,3] += shift_y

        trafo_matrix = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        new_image = cv2.warpAffine(image, trafo_matrix, (w, h))

        return new_image, new_bboxes, labels


class RandomFlip(object):
    """ Flip image and bounding box randomly horizontally, vertically or both.
    """

    def __call__(self, image, bboxes, labels):
        flip_kind = random.choice([-1, 0, 1])  # -1 = both, 0 = x axis, 1 = y axis
        h, w = image.shape[:2]
        new_image = cv2.flip(image, flip_kind)
        new_bboxes = np.copy(bboxes)
        if flip_kind > 0:
            new_bboxes[:,0] = w - bboxes[:, 0] - 1
            new_bboxes[:,2] = w - bboxes[:, 2] - 1

        elif flip_kind == 0:
            new_bboxes[:,1] = h - bboxes[:, 1] - 1
            new_bboxes[:,3] = h - bboxes[:, 3] - 1
        else:
            new_bboxes[:,0] = w - bboxes[:, 0] - 1
            new_bboxes[:,2] = w - bboxes[:, 2] - 1
            new_bboxes[:,1] = h - bboxes[:, 1] - 1
            new_bboxes[:,3] = h - bboxes[:, 3] - 1

        for i in range(new_bboxes.shape[0]):
            new_bboxes[i,:] = adjust_bbox(new_bboxes[i,:])

        return new_image, new_bboxes, labels


class RandomContrast(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        self.max = 2**16-1
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    def __call__(self, image, boxes, labels):
        alpha = random.uniform(self.lower, self.upper)
        new_image = image.astype(np.float32)
        new_image *= alpha
        new_image[new_image > self.max] = self.max
        return new_image.astype(np.uint16), boxes, labels


class RandomBrightness(object):
    def __init__(self, delta=500, max=255):
        self.min = 0.0
        self.max = max
        assert delta >= self.min
        assert delta <= self.max
        self.delta = delta

    def __call__(self, image, boxes, labels):
        delta = random.randrange(-self.delta, self.delta)
        new_image = image.astype(np.float32)
        new_image += delta
        new_image[new_image < self.min] = self.min
        new_image[new_image > self.max] = self.max
        return new_image.astype(np.uint16), boxes, labels


class Expand(object):
    def __init__(self, mean):
        self.mean = mean

    def __call__(self, image, boxes, labels):
        height, width = image.shape[:2]
        ratio = random.uniform(1, 4)
        left = random.uniform(0, width*ratio - width)
        top = random.uniform(0, height*ratio - height)

        expand_image = np.zeros(
            (int(height*ratio), int(width*ratio), depth),
            dtype=image.dtype)
        expand_image[:, :, :] = self.mean
        expand_image[int(top):int(top + height),
                     int(left):int(left + width)] = image
        image = expand_image

        boxes = boxes.copy()
        boxes[:, :2] += (int(left), int(top))
        boxes[:, 2:] += (int(left), int(top))

        return image, boxes, labels


class SquarePad(object):
    """Pad image to be square
    """

    def __init__(self, fill=0):
        self.fill = fill

    def __call__(self, image, bbox, labels):
        height, width = image.shape[:2]
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

