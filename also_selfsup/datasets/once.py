import logging
from torch_geometric.data import Dataset
from torch_geometric.data import Data
import numpy as np
import os
import pickle
import copy
import torch

class ONCE(Dataset):

    INFO_PATH= {
        'train': "once_infos_train.pkl",
        'val': "once_infos_val.pkl",
        'test': "once_infos_test.pkl",
        'raw_small': "once_infos_raw_small.pkl",
        'raw_medium': "once_infos_raw_medium.pkl",
        'raw_large': "once_infos_raw_large.pkl",
    }

    def __init__(self,
                 root,
                 split="training",
                 transform=None,
                 skip_ratio=1,
                 **kwargs):

        super().__init__(root, transform, None)

        logging.info(f"ONCE - split {split}")

        info_path = os.path.join(self.root, self.INFO_PATH[split])
        with open(info_path, 'rb') as f:
            self.once_infos = pickle.load(f)


        if split in ["val", "valiation"]:
            self.once_infos = self.once_infos[:100]

        logging.info(f"ONCE dataset {len(self.once_infos)}")


    def _download(self): # override _download to remove makedirs
        pass

    def download(self):
        pass

    def process(self):
        pass

    def _process(self):
        pass

    def len(self):
        return len(self.once_infos)

    def get(self, idx):
        """Get item."""

        # get ids
        frame_id = self.once_infos[idx]['frame_id']
        seq_id = self.once_infos[idx]['sequence_id']

        # load lidar data
        bin_path = os.path.join(self.root, 'data', seq_id, 'lidar_roof', '{}.bin'.format(frame_id))
        points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)

        # get intensities and points
        intensities = points[:,3:4] / 255.
        pos = points[:,:3]

        # transform to tensor
        pos = torch.tensor(pos, dtype=torch.float)
        intensities = torch.tensor(intensities, dtype=torch.float)
        x = torch.ones((pos.shape[0],1), dtype=torch.float)

        return Data(x=x, intensities=intensities, pos=pos, shape_id=idx)
