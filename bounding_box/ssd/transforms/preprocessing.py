import random
import transforms.transforms as tr


class TrainAugmentation:
    def __init__(self, size, p=0.5):
        """
        Args:
            size (int, tuple): the size the of final image
            p (float): probability for applying transformations
        """
        self.size = size
        self.probability = p
        self.transform = tr.Compose([tr.Resize(self.size), tr.RandomCrop(self.size), tr.ToPercentCoords(), tr.ToTensor()])
        self.random_aug = [
            tr.RandomRotate((-359, 359)),
            tr.RandomFlip(),
            tr.RandomRescale((0.7, 1.3)),
            tr.RandomShift(),
            tr.RandomBrightness(1000),
            tr.RandomContrast(0.6, 1.4)
        ]

    def __call__(self, img, box):
        """Apply random number of Augmentations per image + default transformations

        Args:
            img: image.
            box: bounding box in the form of (x1, y1, x2, y2).
        """
        if self.probability < random.random():
            return self.transform(img, box)

        # maximum apply 3 random transformations at once
        max_num_trans = random.randrange(1, 4)
        random.shuffle(self.random_aug)
        random_compose = tr.Compose(self.random_aug[:max_num_trans])
        img, box = random_compose(img, box)
        return self.transform(img, box)


class TestTransform:
    def __init__(self, size):
        self.size = size
        self.transform = tr.Compose([tr.Resize(self.size), tr.RandomCrop(self.size),
                                     tr.ToPercentCoords(), tr.ToTensor()])

    def __call__(self, image, box):
        return self.transform(image, box)
