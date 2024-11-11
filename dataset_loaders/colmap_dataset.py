import numpy as np
import torch
from tqdm import tqdm

from dataset_loaders.dataset_readers import readColmapSceneInfo
from dataset_loaders.pose_reg_dataset import PoseRegDataset


class ColmapDataset(PoseRegDataset):
    @property
    def image_paths(self):
        return self._image_paths

    @property
    def poses(self):
        return self._poses

    @poses.setter
    def poses(self, poses):
        self._poses = torch.tensor(poses, dtype=torch.float)

    def __init__(self, data_path, train, seed=7, trainskip=1, testskip=1,
                 return_orig=False, fix_idx=False, ret_hist=False, hist_bin=10):
        scene_info = readColmapSceneInfo(data_path, "images", True)
        cameras = scene_info.train_cameras if train else scene_info.test_cameras

        self._image_paths = []
        self._poses = []
        i = 0
        for cam in tqdm(cameras, desc='Preparing Dataloader'):
            i += 1
            if train and i % trainskip != 0:
                continue
            if not train and i % testskip != 0:
                continue
            self._image_paths.append(cam.image_path)
            w2c = np.eye(4)
            w2c[:3, :3] = cam.R
            w2c[:3, 3] = cam.T
            pose = np.linalg.inv(w2c)
            self._poses.append(pose[:3].flatten())
        self._poses = torch.tensor(np.array(self._poses), dtype=torch.float)

        self.near = 0.01
        self.far = 100.0
        hwf = (cameras[0].height, cameras[0].width, cameras[0].fx, cameras[0].fy)
        super().__init__(train, hwf, seed, return_orig, fix_idx, ret_hist, hist_bin)
