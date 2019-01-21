import random
import unet.transforms.transforms as tr

# default values
mean = 583.6798
std = 70.0333


class TrainAugmentation:
    def __init__(self, background_color=584, p=0.5):
        """
        Args:
            size (int, tuple): the size the of final image
            normalization (transform object): transformations for normalization of the image
            p (float): probability for applying transformations
        """
        self.probability = p

        self.transform = tr.ComposeImgMask([
            tr.Normalize(),
            tr.ToTensor()
        ])
        self.random_aug = [
            tr.RandomResizeCrop(),
            tr.RandomRotate((-359, 359), background_color),
            tr.RandomFlip(),
            tr.RandomRescale((0.8, 2.0)),
            tr.RandomBrightness(2000, max=2 ** 12 - 1),
            tr.RandomContrast(0.6, 1.4)
        ]

    def __call__(self, img):
        """Apply random number of Augmentations per image + default transformations

        Args:
            img: image.
        """
        for aug in self.random_aug:
            if random.random() < self.probability:
                img = aug(img)

        return self.transform(img)


class TestTransform:
    def __init__(self):
        self.transform = tr.ComposeImgMask([
            tr.Normalize(),
            tr.ToTensor()
        ])

    def __call__(self, image):
        return self.transform(image) # returns image, mask
