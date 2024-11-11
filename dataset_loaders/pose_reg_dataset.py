"""Abstract class for pose regression dataset."""

from abc import ABC, abstractmethod

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import transforms

from dataset_loaders.utils.color import rgb_to_yuv


class PoseRegDataset(ABC, Dataset):
    """Abstract class for pose regression dataset."""

    def __init__(self, train, hwf, seed=7, return_orig=False, fix_idx=False, ret_hist=False, hist_bin=10):
        super().__init__()
        self.H, self.W, self.fx, self.fy = hwf
        self.transform = transforms.Compose([
            transforms.Resize((self.H, self.W)),
            transforms.ToTensor(),
        ])
        self.transform_orig = transforms.ToTensor()
        np.random.seed(seed)

        self.train = train
        self.return_orig = return_orig
        self.fix_idx = fix_idx
        self.ret_hist = ret_hist
        self.hist_bin = hist_bin  # histogram bin size

    @property
    @abstractmethod
    def poses(self):
        """Return the poses of the dataset."""
        pass

    @poses.setter
    @abstractmethod
    def poses(self, poses):
        pass

    @property
    @abstractmethod
    def image_paths(self):
        """Return the image paths of the dataset."""
        pass

    def __len__(self):
        """Return the number of samples in the dataset."""
        return self.poses.shape[0]

    def __getitem__(self, index):
        """Get the sample and target at the given index.

        Args:
            index (int): Index of the sample.

        Returns:
            tuple: Sample and target.
        """
        img_orig = Image.open(self.image_paths[index])  # chess img.size = (640,480)
        img = self.transform(img_orig)
        pose = self.poses[index]

        if self.return_orig:
            return img, pose, self.transform_orig(img_orig)

        if self.ret_hist:
            yuv = rgb_to_yuv(img)
            y_img = yuv[0]  # extract y channel only
            hist = torch.histc(y_img, bins=self.hist_bin, min=0., max=1.)  # compute intensity histogram
            hist = hist / (hist.sum()) * 100  # convert to histogram density, in terms of percentage per bin
            hist = torch.round(hist)
            return img, pose, hist

        return img, pose
