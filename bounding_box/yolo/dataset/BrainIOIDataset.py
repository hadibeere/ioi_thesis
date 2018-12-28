import os
import pandas as pd
import numpy as np
import cv2
import torch

from torch.utils.data import Dataset
from ssd.utils.misc import count_files, collect_filenames


class BrainIOIDataset(Dataset):
    """Frames from intrinsic optical imaging data of the human cortex"""

    def __init__(self, csv_file, root_dir, border=0, num_channels=3, transform=None, target_transform=None, is_eval=False):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the frames per patient.
            border (int): additional border in pixels to expand bounding box
            transform (callable, optional): Optional transform to be applied
                on a sample.
            target_transform (callable): Transform bounding boxes from general representation to specific for cnn

        """
        self.num_channels = num_channels
        self.root_dir = root_dir
        self.annotations = pd.read_csv(csv_file)
        self.classes = ["stimulation"]
        self.subfolders = ["stimulation", "before", "after"]
        self.border = border  # pixel
        self.files, self.class_labels = self.create_file_list()
        self.bins = self.create_bins()
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return int(len(self.class_labels) / self.num_channels)

    def __getitem__(self, idx):
        """
        Default number of images will be combined per sample

        Return: combined sample, one bbox as list and one label as list
        """
        sample = self.get_image(idx)
        boxes, labels = self.get_annotation(idx)

        if self.transform:
            sample, boxes, labels = self.transform(sample, boxes, labels)
        else:
            print("Error no transform")

        if self.target_transform:
            gt = self.target_transform(boxes, labels)

        return sample, torch.Tensor(gt)

    def create_bins(self):
        bins = [0]
        for i in range(0, len(self.annotations)):
            parent = os.path.join(self.root_dir, self.annotations.iloc[i, 0])
            sum_bin = 0
            for child in self.subfolders:
                count = count_files(os.path.join(parent, child), '.tif')
                if count % self.num_channels != 0:
                    count -= 1

                sum_bin += count

            bins.append(bins[i] + sum_bin)

        return bins

    def create_file_list(self):
        file_list = []
        class_labels = []
        for i in range(0, len(self.annotations)):
            parent = os.path.join(self.root_dir, self.annotations.iloc[i, 0])
            for child in self.subfolders:
                files = collect_filenames(os.path.join(parent, child), '.tif')
                if len(files) % self.num_channels != 0:
                    files = files[:-1]

                file_list += files
                cur_class = 0
                if child == self.subfolders[0]:
                    cur_class = 1
                class_labels += [cur_class for _ in range(len(files))]

        return file_list, class_labels

    def _min_with_border(self, x, y):
        tmp_min = min(x, y)
        if tmp_min >= self.border:
            tmp_min -= self.border

        return tmp_min

    def _max_with_border(self, x, y, limit=1000):
        tmp_max = max(x, y) + self.border
        if tmp_max > limit:
            tmp_max = limit

        return tmp_max

    def get_image(self, idx):
        loc = idx * self.num_channels
        images = []
        for i in range(self.num_channels):
            images.append(cv2.imread(self.files[loc + i], cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH))

        sample = np.dstack(images)
        return sample

    def get_annotation(self, idx):
        loc = idx * self.num_channels
        bbox = [0,0,0,0]
        if self.class_labels[loc] == 1:
            pos = np.digitize(loc, self.bins) - 1  # find out which annotation belongs to data
            tips = self.annotations.iloc[pos, 3:].values.reshape(2,2)  # 2 electrode tips for each picture ([posx1, posy1], [posx2, posy2])
            bbox = [self._min_with_border(tips[0][0], tips[1][0]), self._min_with_border(tips[0][1], tips[1][1]),
                    self._max_with_border(tips[0][0], tips[1][0]),
                    self._max_with_border(tips[0][1], tips[1][1])]

        boxes = np.array([bbox])
        labels = np.array([self.class_labels[loc]])
        return boxes, labels