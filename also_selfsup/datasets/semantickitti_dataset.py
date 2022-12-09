import os
import logging
import yaml
from pathlib import Path
import json
import numpy as np
import torch
from torch_geometric.data import Dataset
from torch_geometric.data import Data


class SemanticKITTI(Dataset):

    N_LABELS = 20

    def __init__(self,
                 root,
                 split="training",
                 transform=None,
                 skip_ratio=1,
                 **kwargs):

        super().__init__(root, transform, None)

        self.split = split
        self.n_frames = 1

        logging.info(f"SemanticKITTI - split {split}")

        # get the scenes
        assert(split in ["train", "val", "test"])
        if split == "train":
            self.sequences = ['00', '01', '02', '03', '04', '05', '06', '07', '09', '10']
        elif split == "val":
            self.sequences = ['08']
        elif split == "test":
            raise NotImplementedError
        else:
            raise ValueError('Unknown set for SemanticKitti data: ', split)

        self.points_datapath = []
        self.labels_datapath = []
        if self.split == "train" and skip_ratio > 1:
            with open("datasets/percentiles_split.json", 'r') as p:
                splits = json.load(p)
                skip_to_percent = {2:'0.5', 4:'0.25', 10:'0.1', 100:'0.01', 1000:'0.001', 10000:'0.0001'}
                if skip_ratio not in skip_to_percent:
                    raise ValueError
                percentage = skip_to_percent[skip_ratio]

                for seq in splits[percentage]:
                    self.points_datapath += splits[percentage][seq]['points']
                    self.labels_datapath += splits[percentage][seq]['labels']
            
            for i in range(len(self.points_datapath)):
                self.points_datapath[i] = self.points_datapath[i].replace("Datasets/SemanticKITTI/", "")
                self.points_datapath[i] = os.path.join(self.root,self.points_datapath[i])
                self.labels_datapath[i] = self.labels_datapath[i].replace("Datasets/SemanticKITTI/", "")
                self.labels_datapath[i] = os.path.join(self.root,self.labels_datapath[i])
        else:

            
            for sequence in self.sequences:
                self.points_datapath += [path for path in Path(os.path.join(self.root, "dataset", "sequences", sequence, "velodyne")).rglob('*.bin')]

            for fname in self.points_datapath:
                fname = str(fname).replace("/velodyne/", "/labels/")
                fname = str(fname).replace(".bin", ".label")
                self.labels_datapath.append(fname)


        # Read labels
        config_file = 'datasets/semantic-kitti.yaml'

        with open(config_file, 'r') as stream:
            doc = yaml.safe_load(stream)
            all_labels = doc['labels']
            learning_map_inv = doc['learning_map_inv']
            learning_map = doc['learning_map']
            self.learning_map = np.zeros((np.max([k for k in learning_map.keys()]) + 1), dtype=np.int32)
            for k, v in learning_map.items():
                self.learning_map[k] = v

            self.learning_map_inv = np.zeros((np.max([k for k in learning_map_inv.keys()]) + 1), dtype=np.int32)
            for k, v in learning_map_inv.items():
                self.learning_map_inv[k] = v

        self.class_colors = np.array([
            [0, 0, 0],
            [245, 150, 100],
            [245, 230, 100],
            [150, 60, 30],
            [180, 30, 80],
            [255, 0, 0],
            [30, 30, 255],
            [200, 40, 255],
            [90, 30, 150],
            [255, 0, 255],
            [255, 150, 255],
            [75, 0, 75],
            [75, 0, 175],
            [0, 200, 255],
            [50, 120, 255],
            [0, 175, 0],
            [0, 60, 135],
            [80, 240, 150],
            [150, 240, 255],
            [0, 0, 255],
        ], dtype=np.uint8)


        logging.info(f"SemanticKITTI dataset {len(self.points_datapath)}")

    def get_weights(self):
        weights = torch.ones(self.N_LABELS)
        weights[0] = 0
        return weights

    @staticmethod
    def get_mask_filter_valid_labels(y):
        return (y>0)

    def get_colors(self, labels):
        return self.class_colors[labels]

    def get_filename(self, index):
        fname = self.points_datapath[index]
        fname = str(fname).split("/")
        fname = ("_").join([fname[-3], fname[-1]])
        fname = fname[:-4]
        return fname

    @staticmethod
    def get_ignore_index():
        return 0

    def _download(self): # override _download to remove makedirs
        pass

    def download(self):
        pass

    def process(self):
        pass

    def _process(self):
        pass

    def len(self):
        return len(self.points_datapath)

    def get(self, idx):
        """Get item."""

        fname_points = self.points_datapath[idx]
        frame_points = np.fromfile(fname_points, dtype=np.float32)

        pos = frame_points.reshape((-1, 4))
        intensities = pos[:,3:]
        pos = pos[:,:3]

        # Read labels
        label_file = self.labels_datapath[idx]
        frame_labels = np.fromfile(label_file, dtype=np.int32)
        frame_labels = frame_labels.reshape((-1))
        y = frame_labels & 0xFFFF  # semantic label in lower half

        # get unlabeled data
        y = self.learning_map[y]
        unlabeled = (y == 0)

        # remove unlabeled points
        y = np.delete(y, unlabeled, axis=0)
        pos = np.delete(pos, unlabeled, axis=0)
        intensities = np.delete(intensities, unlabeled, axis=0)

        pos = torch.tensor(pos, dtype=torch.float)
        y = torch.tensor(y, dtype=torch.long)
        intensities = torch.tensor(intensities, dtype=torch.float)
        x = torch.ones((pos.shape[0],1), dtype=torch.float)

        return Data(x=x, intensities=intensities, pos=pos, y=y,
                    shape_id=idx, )