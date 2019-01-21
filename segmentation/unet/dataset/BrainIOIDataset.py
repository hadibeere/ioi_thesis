import os
import pandas as pd
import numpy as np
import cv2
import math

from torch.utils.data import Dataset
from ssd.utils.misc import collect_filenames


class IOIDatasetETips(Dataset):
    """Frames from intrinsic optical imaging data of the human cortex
        with Elktrodetips as Ground Truth.
    """

    def __init__(self, csv_file, root_dir, use_all=True, border=0.5, num_channels=3, transform=None, is_eval=False):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the frames per patient.
            use_all (bool): Should we create a ground truth forboth electrode tipps or for one
            border (int): additional border in percentage to distance between elctrode tips to expand bounding box
            transform (callable, optional): Optional transform to be applied
                on a sample.
            target_transform (callable): Transform bounding boxes from general representation to specific for cnn

        """
        self.num_channels = num_channels
        self.root_dir = root_dir
        self.annotations = pd.read_csv(csv_file)
        self.classes = ["BACKGROUND", "stimulation"]
        self.subfolders = ["stimulation"]
        self.use_all = use_all
        self.is_eval = is_eval
        if self.is_eval:
            self.subfolders = ["stimulation", "before", "after"]
        self.border = border  # percentage of distance between electrode tips
        self.files, self.class_labels = self.create_file_list()
        self.transform = transform

    def __len__(self):
        return int(len(self.class_labels))

    def __getitem__(self, idx):
        """
        Default number of images will be combined per sample

        Return: combined sample, one bbox as list and one label as list
        """
        sample = self.get_image(idx)
        boxes = self.get_annotation(idx)
        mask = self.create_mask(sample,boxes)

        if self.transform:
            sample, mask = self.transform(np.dstack([sample, mask]))
        else:
            print("Error no transform")

        return sample, mask

    def create_file_list(self):
        file_list = []
        class_labels = []
        for i in range(0, len(self.annotations)):
            parent = os.path.join(self.root_dir, self.annotations.iloc[i, 0])
            for child in self.subfolders:
                files = collect_filenames(os.path.join(parent, child), '.tif')
                pick = 0  # int(len(files) / 2)
                file_list.append(files[pick])
                cur_class = 0
                if child == self.subfolders[0]:
                    cur_class = 1

                labels = [cur_class]
                if self.use_all:
                    labels = [cur_class, cur_class]
                class_labels.append(labels)

        return file_list, class_labels

    def get_image(self, idx):
        image = cv2.imread(self.files[idx], cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        if self.num_channels == 1:
            return image

        return np.dstack([image for _ in range(self.num_channels)])

    def get_annotation(self, idx):
        blist = []
        if self.class_labels[idx][0] == 1:
            tips = self.annotations.iloc[idx, 3:].values.reshape(2,
                                                                 2)  # 2 electrode tips for each picture ([posx1, posy1], [posx2, posy2])
            distance = math.sqrt((tips[1][0] - tips[0][0]) ** 2 + (tips[1][1] - tips[0][1]) ** 2) * self.border
            blist.append([int(tips[0][0] - distance), int(tips[0][1] - distance), int(tips[0][0] + distance), int(tips[0][1] + distance)])
            if self.use_all:
                blist.append(
                    [int(tips[1][0] - distance), int(tips[1][1] - distance), int(tips[1][0] + distance), int(tips[1][1] + distance)])

        return blist

    def create_mask(self, image, boxes):
        rows, cols = image.shape[:2]
        mask = np.zeros((rows,cols), dtype=image.dtype)
        for box in boxes:
            mask[box[1]:box[3], box[0]:box[2]] = 1

        return mask

