from torch.utils.data import Dataset
import os
from torchvision.io import read_image
import random
from os import listdir
from os.path import join, isfile
import torch as t
import numpy as np
from torchvision.transforms import Resize


class ImgPairDataset(Dataset):

    def __init__(self, path="/home/zdeeno/Documents/Datasets/skoda",
                 dataset_pair=("2021.08.08.16.52.23", "2021.08.08.17.26.02"), shift=0):
        super(ImgPairDataset, self).__init__()
        self.width = 640
        self.height = 480
        self.shift = shift

        path1 = join(path, dataset_pair[0])
        path2 = join(path, dataset_pair[1])
        self.dataset1 = sorted([join(path1, f) for f in listdir(path1) if isfile(join(path1, f))])
        self.dataset2 = sorted([join(path2, f) for f in listdir(path2) if isfile(join(path2, f))])

        self.dataset_len = min(len(self.dataset1), len(self.dataset2))

    def __len__(self):
        return self.dataset_len - self.shift

    def __getitem__(self, idx):
        source_img = read_image(self.dataset1[idx])/255.0
        target_img = read_image(self.dataset2[idx + self.shift])/255.0
        return source_img, target_img


class SingleReprDataset(Dataset):

    def __init__(self, path="/home/zdeeno/Documents/Datasets/skoda",
                 subfolder_list=None, flip_list=None):
        super(SingleReprDataset, self).__init__()
        assert len(flip_list) == len(subfolder_list), "different number of flips"
        self.width = 640
        self.height = 480

        self.flip = []  # 0 no flip, 1 force flip, 2 random flip
        self.dataset = []
        for idx, sub in enumerate(subfolder_list):
            path1 = join(path, sub)
            extend_list = sorted([join(path1, f) for f in listdir(path1) if isfile(join(path1, f)) and f[-3:] == "npy"])
            self.dataset.extend(extend_list)
            self.flip.extend([flip_list[idx] for _ in range(len(extend_list))])
        self.dataset_len = len(self.dataset)

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, idx):
        # print(self.dataset[idx])
        repr_out = np.load(self.dataset[idx], allow_pickle=True).item(0)["representation"]
        # repr_out = repr_out.
        # print(repr_out.shape)
        if self.flip[idx] == 0:
            return repr_out
        elif self.flip[idx] == 1:
            # TODO force flip here
            return repr_out
        elif self.flip[idx] == 2:
            # TODO random flip
            return repr_out


class PairReprDataset(Dataset):
    def __init__(self, dataset1, dataset2):
        super(PairReprDataset, self).__init__()
        self.width = 640
        self.height = 480
        self.d1 = dataset1
        self.d2 = dataset2
        self.product = [(i, j) for i in np.arange(len(self.d1)) for j in np.arange(len(self.d2))]

    def __len__(self):
        return len(self.product)

    def __getitem__(self, idx):
        i1 = self.d1[self.product[idx][0]]
        i2 = self.d2[self.product[idx][1]]
        return i1, i2
