from ssd.transforms.transforms import *
from yolo.transforms.transforms import *
import random


class TrainAugmentation:
    def __init__(self, size, background_color=584, p=0.5):
        """
        Args:
            size (int, tuple): the size the of final image
            normalization (transform object): transformations for normalization of the image
            p (float): probability for applying transformations
        """
        self.size = size
        self.probability = p
        self.always_transform = Compose([SquarePad(background_color)])#,
                                         #RandomSquareCrop()])
        self.transform = Compose([
            Resize((self.size, self.size)),
            ConvertFromInts(),
            Normalize(2**12-1),
            ToTensor()
        ])
        self.random_aug = [
            RandomRotate((-359, 359), background_color),
            RandomFlip(),
            RandomRescale((0.9, 2.0)),
            #Expand(background_color),
            RandomSquareCrop(),
            RandomBrightness(1000, max=2**12 -1),
            RandomContrast(0.6, 1.4)
        ]

    def __call__(self, img, boxes, labels):
        """Apply random number of Augmentations per image + default transformations

        Args:
            img: image.
            box: bounding box in the form of (x1, y1, x2, y2).
        """
        img, boxes, labels = self.always_transform(img, boxes, labels)
        if self.probability > random.random():
            return self.transform(img, boxes, labels)

        # maximum apply 3 random transformations at once
        #max_num_trans = random.randrange(1, 4)
        #random.shuffle(self.random_aug)
        #random_compose = Compose(self.random_aug[:max_num_trans])
        #img, boxes, labels = random_compose(img, boxes, labels)
        #for aug in self.random_aug:
        #    if random.random() < self.probability:
        #        img, boxes, labels = aug(img, boxes, labels)
        #        print(aug)

        return self.transform(img, boxes, labels)


class TestTransform:
    def __init__(self, size, background_color=584):
        self.transform = Compose([
            SquarePad(background_color),
            Resize((size, size)),
            ConvertFromInts(),
            Normalize(2 ** 12 - 1),
            ToTensor()
        ])

    def __call__(self, image, boxes, labels):
        return self.transform(image, boxes, labels)

