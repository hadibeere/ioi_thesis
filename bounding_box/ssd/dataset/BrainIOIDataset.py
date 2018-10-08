import os
import pandas as pd
import numpy as np
import cv2
import torch

import transforms.transforms as tr
from torch.utils.data import Dataset
from utils.misc import count_files, collect_filenames


class BrainIOIDataset(Dataset):
    """Frames from intrinsic optical imaging data of the human cortex"""

    def __init__(self, csv_file, root_dir, border=0, transform=None, target_transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the frames per patient.
            border (int): additional border in pixels to expand bounding box
            transform (callable, optional): Optional transform to be applied
                on a sample.
            target_transform (callable): Transform bounding boxes from general representation to specific for cnn

        """
        self.root_dir = root_dir
        self.annotations = pd.read_csv(csv_file)
        self.classes = ["BACKGROUND", "no_stimulation", "stimulation"]
        self.border = border  # pixel
        self.files, self.class_labels = self.create_file_list()
        self.bins = self.create_bins()
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return int(len(self.class_labels) / 2)

    def __getitem__(self, idx):
        """
        2 images will be combined per sample

        Return: combined sample, one bbox as list and one label as list
        """
        loc = idx * 2
        image1 = cv2.imread(self.files[loc], cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        image2 = cv2.imread(self.files[loc + 1], cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        sample = np.dstack((image1, image2))
        height, width, channel = sample.shape
        bbox = [0, 0, width, height] # default box is whole image
        if self.class_labels[loc] == 1:
            pos = np.digitize(loc, self.bins) - 1  # find out which annotation belongs to data
            tips = self.annotations.iloc[pos, 3:].values.reshape(2,2)  # 2 electrode tips for each picture ([posx1, posy1], [posx2, posy2])
            bbox = [self._min_with_border(tips[0][0], tips[1][0]), self._min_with_border(tips[0][1], tips[1][1]),
                    self._max_with_border(tips[0][0], tips[1][0], width),
                    self._max_with_border(tips[0][1], tips[1][1], height)]

        if self.transform:
            sample, bbox = self.transform(sample, bbox)
        else:
            transform = tr.ToTensor()
            sample, bbox = transform(sample, bbox)

        # create bbox and labels as list, if we should have multiple labels per image
        boxes = bbox.unsqueeze(0)
        labels = torch.LongTensor([self.class_labels[loc]])
        if self.target_transform:
            boxes, labels = self.target_transform(boxes, labels)

        return sample, boxes, labels

    def create_bins(self):
        sub_folders = ["before", "stimulation", "after"]
        bins = [0]
        for i in range(0, len(self.annotations)):
            parent = os.path.join(self.root_dir, self.annotations.iloc[i, 0])
            sum_bin = 0
            for child in sub_folders:
                count = count_files(os.path.join(parent, child), '.tif')
                if count % 2 != 0:
                    count -= 1

                sum_bin += count

            bins.append(bins[i] + sum_bin)

        return bins

    def create_file_list(self):
        sub_folders = ["before", "stimulation", "after"]
        file_list = []
        class_labels = []
        for i in range(0, len(self.annotations)):
            parent = os.path.join(self.root_dir, self.annotations.iloc[i, 0])
            for child in sub_folders:
                files = collect_filenames(os.path.join(parent, child), '.tif')
                if len(files) % 2 != 0:
                    files = files[:-1]

                file_list += files
                if child == sub_folders[1]:
                    cur_class = 2
                else:
                    cur_class = 1

                class_labels += [cur_class for _ in range(len(files))]

        return file_list, class_labels

    def _min_with_border(self, x, y):
        tmp_min = min(x, y)
        if tmp_min >= self.border:
            tmp_min -= self.border

        return tmp_min

    def _max_with_border(self, x, y, limit):
        tmp_max = max(x, y) + self.border
        if tmp_max > limit:
            tmp_max = limit

        return tmp_max

