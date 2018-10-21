import torch
import cv2
import numpy as np
import random
import torchvision


def is_bbox_image_size(bbox, height, width):
    return bbox[0] == 0 and bbox[1] == 0 and bbox[2] == width and bbox[3] == height


class ToTensor(object):
    """Convert ndarrays in sample to Tensors and normalize image."""
    def __init__(self, mean, std):
        self.mean = mean  # list according to number of channels e.g. RGB --> [1,2,3]
        self.std = std  # same as mean bu with values for standard deviation

    def __call__(self, image, bbox):
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.astype(np.float32).transpose((2, 0, 1))
        sample = torch.from_numpy(image)
        normalize = torchvision.transforms.Normalize(self.mean, self.std)
        if isinstance(bbox, np.ndarray):
            new_bbox = torch.from_numpy(bbox)
        else:
            new_bbox = torch.Tensor(bbox)

        return normalize(sample), new_bbox


class Compose(object):
    """Composes several transforms together.
    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.
    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, bbox):
        for t in self.transforms:
            image, bbox = t(image, bbox)
        return image, bbox


class Rotate(object):
    """Rotate image and bounding box around given angle.
        Image will get black borders.

    Args:
        angle (int): Rotate image by given angle.
    """

    def __init__(self, angle):
        assert isinstance(angle, int)
        self.angle = angle

    def __call__(self, image, bbox):
        """ image: Dimensions must be height, width,(channels)
        """
        rows, cols = image.shape[:2]
        new_bbox = bbox[:]
        transformation_matrix = cv2.getRotationMatrix2D((cols/2,rows/2),self.angle,1)
        trans_image = cv2.warpAffine(image, transformation_matrix,(cols,rows))
        if not is_bbox_image_size(bbox,rows,cols):
            trans_marks = cv2.transform(np.array(bbox).reshape(1,2,2), transformation_matrix)[0]
            new_bbox = trans_marks.reshape(-1)

        return trans_image, new_bbox


class RandomRotate(object):
    """Rotate image and bounding box around given angle.
        Image will get black borders.

    Args:
        range (tuple): Rotate image by random angle in limit range.
    """

    def __init__(self, range):
        assert isinstance(range, tuple)
        self.range = range

    def __call__(self, image, bbox):
        angle = int(round(random.uniform(self.range[0], self.range[1])))
        rot = Rotate(angle)
        return rot(image, bbox)


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

    def __call__(self, image, bbox):
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

        new_bbox = [int(round(bbox[0] * new_w / w)), int(round(bbox[1] * new_h / h)),
                    int(round(bbox[2] * new_w / w)), int(round(bbox[3] * new_h / h))]

        return new_image, new_bbox


class Rescale(object):
    """Rescale the image in a sample by given factor and adjust bounding box.

    Args:
        scale (float): Desired scaling factor.
    """

    def __init__(self, scale):
        assert isinstance(scale, float)
        self.scale = scale

    def __call__(self, image, bbox):
        h, w = image.shape[:2]
        if self.scale <= 1.0:
            new_image = cv2.resize(image, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA)
        else:
            new_image = cv2.resize(image, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_LINEAR)

        new_h, new_w = new_image.shape[:2]
        new_bbox = [int(round(bbox[0] * new_w / w)), int(round(bbox[1] * new_h / h)),
                    int(round(bbox[2] * new_w / w)), int(round(bbox[3] * new_h / h))]

        return new_image, new_bbox


class RandomRescale(object):
    """Rescale the image in a sample by random factor and adjust bounding box.

    Args:
        scale_range (tuple): Desired scaling factor range(min,max).
    """

    def __init__(self, scale_range):
        assert isinstance(scale_range, tuple)
        self.scale_range = scale_range

    def __call__(self, image, bbox):
        h, w = image.shape[:2]
        scale = round(random.uniform(self.scale_range[0], self.scale_range[1]), 1)
        if scale <= 1.0:
            new_image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        else:
            new_image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

        new_h, new_w = new_image.shape[:2]
        new_bbox = [int(round(bbox[0] * new_w / w)), int(round(bbox[1] * new_h / h)),
                    int(round(bbox[2] * new_w / w)), int(round(bbox[3] * new_h / h))]

        return new_image, new_bbox


class RandomShift(object):
    """Shift image and bbox randomly, so that bbox is still inside image."""

    def __init__(self):
        pass

    def __call__(self, image, bbox):
        h, w = image.shape[:2]
        new_bbox = bbox[:]
        if is_bbox_image_size(bbox, h, w):
            # max shift 25%
            shift_x = random.randint(int(-w / 4), int(w / 4))
            shift_y = random.randint(int(-h / 4), int(h / 4))
        else:
            shift_x = random.randint(-bbox[0], w - bbox[2])
            shift_y = random.randint(-bbox[1], h - bbox[3])
            new_bbox = [bbox[0] + shift_x, bbox[1] + shift_y, bbox[2] + shift_x, bbox[3] + shift_y]

        trafo_matrix = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        new_image = cv2.warpAffine(image, trafo_matrix, (w, h))

        return new_image, new_bbox


class RandomFlip(object):
    """ Flip image and bounding box randomly horizontally, vertically or both.
    """

    def __call__(self, image, bbox):
        flip_kind = random.choice([-1, 0, 1])  # -1 = both, 0 = x axis, 1 = y axis
        h, w = image.shape[:2]
        new_image = cv2.flip(image, flip_kind)
        new_bbox = bbox[:]
        if not is_bbox_image_size(bbox, h, w):
            if flip_kind > 0:
                new_bbox[0] = w - bbox[0] - 1
                new_bbox[2] = w - bbox[2] - 1

            elif flip_kind == 0:
                new_bbox[1] = h - bbox[1] - 1
                new_bbox[3] = h - bbox[3] - 1
            else:
                new_bbox[0] = w - bbox[0] - 1
                new_bbox[2] = w - bbox[2] - 1
                new_bbox[1] = h - bbox[1] - 1
                new_bbox[3] = h - bbox[3] - 1

        return new_image, new_bbox


class Pad(object):
    def __init__(self, h, w):
        self.h = h
        self.w = w

    def __call__(self, image, bbox):
        h, w = image.shape[:2]
        new_bbox = bbox[:]
        if is_bbox_image_size(new_bbox, h, w):
            new_bbox[2] += self.w
            new_bbox[3] += self.h

        # pad order top, bottom, left, right
        new_image = cv2.copyMakeBorder(image, 0, self.h, 0, self.w, cv2.BORDER_CONSTANT)

        return new_image, new_bbox


class RandomCrop(object):
    """Crop the image randomly, so that bounding box is still inside.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size (height,width). If int, value is used for both height and width.
    """

    def __init__(self, output_size):
        if isinstance(output_size, int):
            self.width = output_size
            self.height = output_size
        else:
            self.height, self.width = output_size

    def __call__(self, image, bbox):
        new_bbox = bbox[:]
        new_image, new_bbox = self.check_bbox_size(image, new_bbox)
        new_image, new_bbox = self.check_padding(new_image, new_bbox)

        # define range for possible crop start point
        h, w = new_image.shape[:2]
        bbox_w = new_bbox[2] - new_bbox[0]
        bbox_h = new_bbox[3] - new_bbox[1]

        x_range = [0, w - self.width]
        y_range = [0, h - self.height]
        if h > bbox_h or w > bbox_w:  # check if bbox is smaller image
            diff_w = self.width - bbox_w
            x_range[0] = new_bbox[0] - diff_w
            if x_range[0] < 0:
                x_range[0] = 0

            limit = w - self.width
            if limit > new_bbox[0]:
                x_range[1] = new_bbox[0]
            else:
                x_range[1] = limit

            diff_h = self.height - bbox_h
            y_range[0] = new_bbox[1] - diff_h
            if y_range[0] < 0:
                y_range[0] = 0

            limit = h - self.height
            if limit > new_bbox[1]:
                y_range[1] = new_bbox[1]
            else:
                y_range[1] = limit

        x_crop = 0
        y_crop = 0
        if x_range[0] < x_range[1]:
            x_crop = random.randrange(x_range[0], x_range[1])

        if y_range[0] < y_range[1]:
            y_crop = random.randrange(y_range[0], y_range[1])

        # gray scale images might have only 2 dim and not 3
        if new_image.ndim == 2:
            new_image = new_image[y_crop:y_crop + self.height, x_crop:x_crop + self.width]
        else:
            new_image = new_image[y_crop:y_crop + self.height, x_crop:x_crop + self.width, :]

        if not is_bbox_image_size(new_bbox, h, w):
            new_bbox[0] -= x_crop
            new_bbox[2] -= x_crop
            new_bbox[1] -= y_crop
            new_bbox[3] -= y_crop
        else:
            new_bbox = [0, 0, self.width, self.height]

        return new_image, new_bbox

    def check_padding(self, image, bbox):
        h, w = image.shape[:2]
        pad_h = self.height - h
        pad_w = self.width - w
        if pad_h < 0:
            pad_h = 0

        if pad_w < 0:
            pad_w = 0

        if pad_h > 0 or pad_w > 0:
            # pad image before crop
            pad = Pad(pad_h, pad_w)
            new_image, bbox = pad(image, bbox)
        else:
            new_image = image

        return new_image, bbox

    def check_bbox_size(self, image, bbox):
        h, w = image.shape[:2]
        box_w = bbox[2] - bbox[0]
        box_h = bbox[3] - bbox[1]
        new_w = w
        new_h = h
        new_image = image
        if box_w > self.width:
            new_w = int((box_w/self.width-0.1) * w)

        if box_h > self.height:
            new_h = int((box_w/self.height - 0.1) * h)

        if new_h != h or new_w != w:
            trans = Resize((new_h, new_w))
            new_image, bbox = trans(image, bbox)

        return new_image, bbox


class RandomContrast(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        self.max = 2**16-1
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    def __call__(self, image, box):
        alpha = random.uniform(self.lower, self.upper)
        new_image = image.astype(np.float32)
        new_image *= alpha
        new_image[new_image > self.max] = self.max
        return new_image.astype(np.uint16), box


class RandomBrightness(object):
    def __init__(self, delta=500):
        self.min = 0.0
        self.max = 2**16-1
        assert delta >= self.min
        assert delta <= self.max
        self.delta = delta

    def __call__(self, image, box):
        delta = random.randrange(-self.delta, self.delta, 25)
        new_image = image.astype(np.float32)
        new_image += delta
        new_image[new_image < self.min] = self.min
        new_image[new_image > self.max] = self.max
        return new_image.astype(np.uint16), box


class ToPercentCoords(object):
    def __call__(self, image, box):
        height, width = image.shape[:2]
        box[0] /= width
        box[2] /= width
        box[1] /= height
        box[3] /= height

        return image, box
