from fileinput import filename
import os
import sys
import numpy as np
import torch
from torch_geometric.data import Dataset
from torch_geometric.data import Data
import logging
from tqdm import tqdm

class KITTI3D(Dataset):

    def __init__(self,
                 root,
                 split="training",
                 transform=None,
                 dataset_size=None,
                 multiframe_range=None,
                 skip_ratio=1,
                 **kwargs):

        
        super().__init__(root, transform, None)

        self.split = split
        self.n_frames = 1
        self.multiframe_range = multiframe_range

        logging.info(f"KITTI3D - {split}")

        if split=="train":
            filenames_list = "kitti3d_train.txt"
        elif split=="val":
            filenames_list = "kitti3d_val.txt"

        with open(os.path.join("datasets", filenames_list), "r") as f:
            files = f.readlines()

        if split=="val": # for fast validation
            files = files[::20]

        
        self.files = [os.path.join(self.root, "training/velodyne", f.split("\n")[0]+".bin") for f in files]


        logging.info(f"KITTI3D - {len(self.files)} files")

    def _download(self): # override _download to remove makedirs
        pass

    def download(self):
        pass

    def process(self):
        pass

    def _process(self):
        pass

    def len(self):
        return len(self.files)

    def get(self, idx):

        fname_points = self.files[idx]
        frame_points = np.fromfile(fname_points, dtype=np.float32)

        pos = frame_points.reshape((-1, 4))
        intensities = pos[:,3:]
        pos = pos[:,:3]
        
        pos = torch.tensor(pos, dtype=torch.float)
        intensities = torch.tensor(intensities, dtype=torch.float)
        x = torch.ones((pos.shape[0],1), dtype=torch.float)

        return Data(x=x, intensities=intensities, pos=pos, shape_id=idx, )