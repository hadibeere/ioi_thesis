import os
import pandas as pd
import numpy as np
import cv2
import math

from torch.utils.data import Dataset
from ssd.utils.misc import count_files, collect_filenames


class BrainIOIDataset2(Dataset):
    """Frames from intrinsic optical imaging data of the human cortex"""

    def __init__(self, csv_file, root_dir, border=0, transform=None, target_transform=None, is_eval=False):
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
        self.classes = ["BACKGROUND", "stimulation"]
        self.subfolders = ["stimulation"]
        if is_eval:
            self.subfolders = ["stimulation", "before", "after"]
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

        boxes = np.array([bbox])
        labels = np.array([self.class_labels[loc]])

        if self.transform:
            sample, boxes, labels = self.transform(sample, boxes, labels)
        else:
            print("Error no transform")
        if self.target_transform:
            boxes, labels = self.target_transform(boxes, labels)

        return sample, boxes, labels

    def create_bins(self):
        bins = [0]
        for i in range(0, len(self.annotations)):
            parent = os.path.join(self.root_dir, self.annotations.iloc[i, 0])
            sum_bin = 0
            for child in self.subfolders:
                count = count_files(os.path.join(parent, child), '.tif')
                if count % 2 != 0:
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
                if len(files) % 2 != 0:
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
        loc = idx * 2
        image1 = cv2.imread(self.files[loc], cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        image2 = cv2.imread(self.files[loc + 1], cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        sample = np.dstack((image1, image2))
        return sample

    def get_annotation(self, idx):
        loc = idx * 2
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
        self.classes = ["BACKGROUND", "stimulation"]
        self.subfolders = ["stimulation"]
        if is_eval:
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
            boxes, labels = self.target_transform(boxes, labels)

        return sample, boxes, labels

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


class IOIDatasetETips(Dataset):
    """Frames from intrinsic optical imaging data of the human cortex
        with Elktrodetips as Ground Truth.
    """

    def __init__(self, csv_file, root_dir, use_all=True, border=0.5, num_channels=3, transform=None,
                 target_transform=None, is_eval=False):
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
        self.target_transform = target_transform

    def __len__(self):
        return int(len(self.class_labels))

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
            boxes, labels = self.target_transform(boxes, labels)

        return sample, boxes, labels

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

        return np.dstack([image for i in range(self.num_channels)])

    def get_annotation(self, idx):
        blist = []
        if self.class_labels[idx][0] == 1:
            tips = self.annotations.iloc[idx, 3:].values.reshape(2,
                                                                 2)  # 2 electrode tips for each picture ([posx1, posy1], [posx2, posy2])
            distance = int(math.sqrt((tips[1][0] - tips[0][0]) ** 2 + (tips[1][1] - tips[0][1]) ** 2) * self.border)
            blist.append([tips[0][0] - distance, tips[0][1] - distance, tips[0][0] + distance, tips[0][1] + distance])
            if self.use_all:
                blist.append(
                    [tips[1][0] - distance, tips[1][1] - distance, tips[1][0] + distance, tips[1][1] + distance])

        boxes = np.array(blist)
        labels = np.array(self.class_labels[idx])
        return boxes, labels