import random
import torch
from torchvision import transforms
import cv2
import numpy as np


class ComposeImgMask(object):
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

    def __call__(self, img, mask=None):
        for t in self.transforms:
            img, mask = t(img, mask)
        return img, mask


class Normalize(object):
    def __init__(self, depth=2 ** 12 - 1):
        self.depth = depth

    def __call__(self, image, mask=None):
        image = image.astype(np.float32)
        sample = image[:,:,:-1]
        mask = np.squeeze(image[:,:,-1])
        sample = np.divide(sample, self.depth)
        return sample, mask


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, image, mask):
        # numpy image: H x W x C
        # torch image: C X H X W
        if image.ndim == 2:
            image = np.expand_dims(image.astype(np.float32), axis=0)  # C x H x W
        else:
            image = image.astype(np.float32).transpose((2, 0, 1))

        return torch.from_numpy(image), torch.from_numpy(mask)


class RandomResizeCrop(object):
    def __init__(self, margin=0.2):
        self.margin = margin

    """Randomly Crop the Image and preserve original ratio, resize the crop to original size.
        Arguments:
            img (Image): the image being input during training, last channel is mask(0,1)
        Return:
            (img)
                img (Image): the cropped image
     """
    def __call__(self, image):
        h,w,c = image.shape
        new_w = int(np.random.uniform(0.3 * w, w))
        new_h = int(h * new_w / float(w))
        crop_w = int(np.random.uniform(0 - self.margin * w, self.margin * w))
        crop_h = int(np.random.uniform(0 - self.margin * h, self.margin * h))
        crop_box = [crop_w, crop_h, crop_w+new_w, crop_h+new_h]
        return cv2.resize(self.crop(image, crop_box), (w, h))

    def crop(self,img, box):
        if box[0] < 0 or box[1] < 0 or box[2] > img.shape[1] or box[3] > img.shape[0]:
            img, box = self.pad_img_to_fit_box(img, box)
        return img[box[1]:box[3], box[0]:box[2], :]

    def pad_img_to_fit_box(self, img, box):
        img = cv2.copyMakeBorder(img, - min(0, box[1]), max(box[3] - img.shape[0], 0),
                -min(0, box[0]), max(box[2] - img.shape[1], 0),cv2.BORDER_CONSTANT)
        box[3] += -min(0, box[1])
        box[1] += -min(0, box[1])
        box[2] += -min(0, box[0])
        box[0] += -min(0, box[0])
        return img, box


class Rotate(object):
    """Rotate image around given angle.

    Args:
        angle (int): Rotate image by given angle.
        fill_color (int): Fill Background with given color
    """

    def __init__(self, angle, fill_color=0):
        assert isinstance(angle, int)
        self.angle = angle
        assert isinstance(fill_color, int)
        self.fill_color = fill_color

    def __call__(self, image):
        """ image: Dimensions must be height, width,(channels)
        """
        rows, cols = image.shape[:2]
        transformation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), self.angle, 1)
        trans_image = cv2.warpAffine(image, transformation_matrix, (cols, rows),borderMode=cv2.BORDER_CONSTANT,
                           borderValue=self.fill_color)

        return trans_image


class RandomRotate(object):
    """Rotate image around given angle.

    Args:
        range (tuple): Rotate image by random angle in limit range.
        fill_color (int): Fill Background with given color
    """

    def __init__(self, range, fill_color):
        assert isinstance(range, tuple)
        self.range = range
        assert isinstance(fill_color, int)
        self.fill_color = fill_color

    def __call__(self, image):
        angle = int(round(random.uniform(self.range[0], self.range[1])))
        rot = Rotate(angle, self.fill_color)
        return rot(image)


class RandomRescale(object):
    """Rescale the image in a sample by random factor and crop or pad to original size.

    Args:
        scale_range (tuple): Desired scaling factor range(min,max).
    """

    def __init__(self, scale_range):
        assert isinstance(scale_range, tuple)
        self.scale_range = scale_range

    def __call__(self, image):
        h,w = image[:2]
        scale = round(random.uniform(self.scale_range[0], self.scale_range[1]), 1)
        if scale <= 1.0:
            new_image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            new_h, new_w = new_image.shape[:2]
            diff_w = w - new_w
            diff_h = h - new_h
            top = int(diff_h/2)
            bottom = diff_h - top
            left = int(diff_w/2)
            right = diff_w - left
            new_image = cv2.copyMakeBorder(new_image, -top, bottom, -left, right, cv2.BORDER_CONSTANT)
        else:
            new_image = CenterCrop((h,w))(cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR))

        return new_image


class RandomFlip(object):
    """ Flip image and bounding box randomly horizontally, vertically or both.
    """

    def __call__(self, image):
        flip_kind = random.choice([-1, 0, 1])  # -1 = both, 0 = x axis, 1 = y axis
        new_image = cv2.flip(image, flip_kind)

        return new_image


class RandomContrast(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        self.max = 2**16-1
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    def __call__(self, image):
        alpha = random.uniform(self.lower, self.upper)
        new_image = image.astype(np.float32)
        new_image[:,:,:-1] *= alpha
        new_image[new_image > self.max] = self.max
        return new_image.astype(np.uint16)


class RandomBrightness(object):
    def __init__(self, delta=500, max=1000):
        self.min = 0.0
        self.max = max
        assert delta >= self.min
        assert delta <= self.max
        self.delta = delta

    def __call__(self, image):
        delta = random.randrange(-self.delta, self.delta)
        new_image = image.astype(np.float32)
        new_image += delta
        new_image[new_image < self.min] = self.min
        new_image[new_image > self.max] = self.max
        return new_image.astype(np.uint16)


class CenterCrop(object):
    def __init__(self, size):
        if isinstance(size,tuple):
            self.width = size(1)
            self.height = size(0)
        else:
            self.width = self.height = size

    def __call__(self, image):
        h, w = image.shape[:2]
        assert h >= self.height
        assert w >= self.width
        x = int(w/2) - self.width/2
        y = int(h/2) - self.height/2
        return image[y:y+self.height, x:x+self.width, :]
