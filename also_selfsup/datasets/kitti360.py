from fileinput import filename
import os
import sys
import numpy as np
import torch
from torch_geometric.data import Dataset
from torch_geometric.data import Data
import logging
from tqdm import tqdm

class KITTI360(Dataset):

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

        logging.info(f"KITTI360 - {split}")

        if split=="train":
            filenames_list = "kitti_360_train_velodynes.txt"
        elif split=="val":
            filenames_list = "kitti_360_val_velodynes.txt"

        with open(os.path.join("datasets", filenames_list), "r") as f:
            files = f.readlines()

        if split=="val": # for fast validation
            files = files[::20]

        self.files = [os.path.join(self.root, f.split("\n")[0]) for f in files]


        logging.info(f"KITTI360 - {len(self.files)} files")

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



if __name__ == "__main__":

    print("Creating the list of training frames")

    with open("2013_05_28_drive_train.txt", "r") as f:
        lines = f.readlines()

    print("writing train files...")
    with open("kitti_360_train_velodynes.txt", "w") as f:

        for line in tqdm(lines, ncols=100):

            line = line.split("\n")[0]

            directory = line.split("/")[2]

            filename = os.path.basename(line)

            first_file = int(os.path.splitext(filename)[0].split("_")[0])
            second_file = int(os.path.splitext(filename)[0].split("_")[1])

            fnames = []
            for i in range(first_file, second_file+1):
                fname = f"{i:010d}.bin"
                fname = os.path.join("data_3d_raw", directory, "velodyne_points/data", fname)
                fname = str(fname)
                fnames.append(fname)

            for item in fnames:
                # write each item on a new line
                f.write("%s\n" % item)
    print('Done')

    print("Creating the list of val frames")

    with open("2013_05_28_drive_val.txt", "r") as f:
        lines = f.readlines()

    print("writing val files...")
    with open("kitti_360_val_velodynes.txt", "w") as f:

        for line in tqdm(lines, ncols=100):

            line = line.split("\n")[0]

            directory = line.split("/")[2]

            filename = os.path.basename(line)

            first_file = int(os.path.splitext(filename)[0].split("_")[0])
            second_file = int(os.path.splitext(filename)[0].split("_")[1])

            fnames = []
            for i in range(first_file, second_file+1):
                fname = f"{i:010d}.bin"
                fname = os.path.join("data_3d_raw", directory, "velodyne_points/data", fname)
                fname = str(fname)
                fnames.append(fname)

            for item in fnames:
                # write each item on a new line
                f.write("%s\n" % item)
    print('Done')