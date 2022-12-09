
import os
import numpy as np
import logging

import torch

from torch_geometric.data import Dataset
from torch_geometric.data import Data

from nuscenes import NuScenes as NuScenes_
from nuscenes.utils.splits import create_splits_scenes
from nuscenes.utils.data_io import load_bin_file
from nuscenes.utils.data_classes import LidarPointCloud

CUSTOM_SPLIT = [
    "scene-0008", "scene-0009", "scene-0019", "scene-0029", "scene-0032", "scene-0042",
    "scene-0045", "scene-0049", "scene-0052", "scene-0054", "scene-0056", "scene-0066",
    "scene-0067", "scene-0073", "scene-0131", "scene-0152", "scene-0166", "scene-0168",
    "scene-0183", "scene-0190", "scene-0194", "scene-0208", "scene-0210", "scene-0211",
    "scene-0241", "scene-0243", "scene-0248", "scene-0259", "scene-0260", "scene-0261",
    "scene-0287", "scene-0292", "scene-0297", "scene-0305", "scene-0306", "scene-0350",
    "scene-0352", "scene-0358", "scene-0361", "scene-0365", "scene-0368", "scene-0377",
    "scene-0388", "scene-0391", "scene-0395", "scene-0413", "scene-0427", "scene-0428",
    "scene-0438", "scene-0444", "scene-0452", "scene-0453", "scene-0459", "scene-0463",
    "scene-0464", "scene-0475", "scene-0513", "scene-0533", "scene-0544", "scene-0575",
    "scene-0587", "scene-0589", "scene-0642", "scene-0652", "scene-0658", "scene-0669",
    "scene-0678", "scene-0687", "scene-0701", "scene-0703", "scene-0706", "scene-0710",
    "scene-0715", "scene-0726", "scene-0735", "scene-0740", "scene-0758", "scene-0786",
    "scene-0790", "scene-0804", "scene-0806", "scene-0847", "scene-0856", "scene-0868",
    "scene-0882", "scene-0897", "scene-0899", "scene-0976", "scene-0996", "scene-1012",
    "scene-1015", "scene-1016", "scene-1018", "scene-1020", "scene-1024", "scene-1044",
    "scene-1058", "scene-1094", "scene-1098", "scene-1107",
]


class NuScenes(Dataset):

    N_LABELS=17

    def __init__(self,
                 root,
                 split="training",
                 transform=None,
                 skip_ratio=1,
                 skip_for_visu=1,
                 **kwargs):

        super().__init__(root, transform, None)

        self.nusc = NuScenes_(version='v1.0-trainval', dataroot=self.root, verbose=True)
        self.split = split

        self.label_to_name = {0: 'noise',
                               1: 'animal',
                               2: 'human.pedestrian.adult',
                               3: 'human.pedestrian.child',
                               4: 'human.pedestrian.construction_worker',
                               5: 'human.pedestrian.personal_mobility',
                               6: 'human.pedestrian.police_officer',
                               7: 'human.pedestrian.stroller',
                               8: 'human.pedestrian.wheelchair',
                               9: 'movable_object.barrier',
                               10: 'movable_object.debris',
                               11: 'movable_object.pushable_pullable',
                               12: 'movable_object.trafficcone',
                               13: 'static_object.bicycle_rack',
                               14: 'vehicle.bicycle',
                               15: 'vehicle.bus.bendy',
                               16: 'vehicle.bus.rigid',
                               17: 'vehicle.car',
                               18: 'vehicle.construction',
                               19: 'vehicle.emergency.ambulance',
                               20: 'vehicle.emergency.police',
                               21: 'vehicle.motorcycle',
                               22: 'vehicle.trailer',
                               23: 'vehicle.truck',
                               24: 'flat.driveable_surface',
                               25: 'flat.other',
                               26: 'flat.sidewalk',
                               27: 'flat.terrain',
                               28: 'static.manmade',
                               29: 'static.other',
                               30: 'static.vegetation',
                               31: 'vehicle.ego'
                               }
        
        self.label_to_name_reduced = {
            0: 'noise',
            1: 'barrier',
            2: 'bicycle',
            3: 'bus',
            4: 'car',
            5: 'construction_vehicle',
            6: 'motorcycle',
            7: 'pedestrian',
            8: 'traffic_cone',
            9: 'trailer',
            10: 'truck',
            11: 'driveable_surface',
            12: 'other_flat',
            13: 'sidewalk',
            14: 'terrain',
            15: 'manmade',
            16: 'vegetation',
        }
        
        self.label_to_reduced = {
            1: 0,
            5: 0,
            7: 0,
            8: 0,
            10: 0,
            11: 0,
            13: 0,
            19: 0,
            20: 0,
            0: 0,
            29: 0,
            31: 0,
            9: 1,
            14: 2,
            15: 3,
            16: 3,
            17: 4,
            18: 5,
            21: 6,
            2: 7,
            3: 7,
            4: 7,
            6: 7,
            12: 8,
            22: 9,
            23: 10,
            24: 11,
            25: 12,
            26: 13,
            27: 14,
            28: 15,
            30: 16
        }

        self.label_to_reduced_np = np.zeros(32, dtype=np.int)
        for i in range(32):
            self.label_to_reduced_np[i] = self.label_to_reduced[i]

        self.reduced_colors = np.array([
            [0, 0, 0],
            [112, 128, 144],  
            [220, 20, 60],  # Crimson
            [255, 127, 80],  # Coral
            [255, 158, 0],  # Orange
            [233, 150, 70],  # Darksalmon
            [255, 61, 99],  # Red
            [0, 0, 230],  # Blue
            [47, 79, 79],  # Darkslategrey
            [255, 140, 0],  # Darkorange
            [255, 99, 71],  # Tomato
            [0, 207, 191],  # nuTonomy green
            [175, 0, 75],
            [75, 0, 75],
            [112, 180, 60],
            [222, 184, 135],  # Burlywood
            [0, 175, 0],  # Green
            ], dtype=np.uint8)


        ##############
        logging.info(f"Nuscenes dataset - creating splits - split {split}")
        # from nuscenes.utils import splits

        # get the scenes
        assert(split in ["train", "val", "test", "verifying", "parametrizing"])
        if split == "verifying":
            phase_scenes = CUSTOM_SPLIT
        elif split == "parametrizing":
            phase_scenes = list( set(create_splits_scenes()["train"]) - set(CUSTOM_SPLIT) )
        else:
            phase_scenes = create_splits_scenes()[split]


        # create a list of camera & lidar scans
        skip_counter = 0
        self.list_keyframes = []
        for scene_idx in range(len(self.nusc.scene)):
            scene = self.nusc.scene[scene_idx]
            if scene["name"] in phase_scenes:

                skip_counter += 1
                if skip_counter % skip_ratio == 0:
                    current_sample_token = scene["first_sample_token"]

                    # Loop to get all successive keyframes
                    list_data = []
                    while current_sample_token != "":
                        current_sample = self.nusc.get("sample", current_sample_token)
                        list_data.append(current_sample)
                        current_sample_token = current_sample["next"]

                        if skip_for_visu > 1:
                            break

                    # Add new scans in the list
                    self.list_keyframes.extend(list_data)

        self.list_keyframes = self.list_keyframes[::skip_for_visu]

        if len(self.list_keyframes)==0:
            # add only one scene
            # scenes with all labels (parametrizing split) "scene-0392", "scene-0517", "scene-0656", "scene-0730", "scene-0738"
            for scene_idx in range(len(self.nusc.scene)):
                scene = self.nusc.scene[scene_idx]
                if scene["name"] in phase_scenes and scene["name"] in ["scene-0392"]:

                    current_sample_token = scene["first_sample_token"]

                    # Loop to get all successive keyframes
                    list_data = []
                    while current_sample_token != "":
                        current_sample = self.nusc.get("sample", current_sample_token)
                        list_data.append(current_sample)
                        current_sample_token = current_sample["next"]

                    # Add new scans in the list
                    self.list_keyframes.extend(list_data)


        logging.info(f"Nuscenes dataset split {split} - {len(self.list_keyframes)} frames")


    def get_weights(self):
        weights = torch.ones(self.N_LABELS)
        weights[0] = 0
        return weights

    @staticmethod
    def get_mask_filter_valid_labels(y):
        return (y>0)

    @staticmethod
    def get_ignore_index():
        return 0

    def get_colors(self, labels):
        return self.reduced_colors[labels]


    def get_filename(self, index):
        
        # get sample
        sample = self.list_keyframes[index]

        # get the lidar token
        lidar_token = sample["data"]["LIDAR_TOP"]

        return str(lidar_token)

    @property
    def raw_file_names(self):
        return []

    def _download(self): # override _download to remove makedirs
        pass

    def download(self):
        pass

    def process(self):
        pass

    def _process(self):
        pass

    def len(self):
        return len(self.list_keyframes)

    def get(self, idx):
        """Get item."""

        # get sample
        sample = self.list_keyframes[idx]

        # get the lidar token
        lidar_token = sample["data"]["LIDAR_TOP"]

        # the lidar record
        lidar_rec = self.nusc.get('sample_data', sample['data']["LIDAR_TOP"])

        # get intensities
        pc = LidarPointCloud.from_file(os.path.join(self.nusc.dataroot, lidar_rec['filename']))
        pos = pc.points.T[:,:3]
        intensities = pc.points.T[:,3:] / 255 # intensities

        # get the labels
        lidarseg_label_filename = os.path.join(self.nusc.dataroot, self.nusc.get('lidarseg', lidar_token)['filename'])
        y_complete_labels = load_bin_file(lidarseg_label_filename)
        y = self.label_to_reduced_np[y_complete_labels]

        # convert to torch
        pos = torch.tensor(pos, dtype=torch.float)
        y = torch.tensor(y, dtype=torch.long)
        intensities = torch.tensor(intensities, dtype=torch.float)
        x = torch.ones((pos.shape[0],1), dtype=torch.float)

        return Data(x=x, intensities=intensities, pos=pos, y=y, shape_id=idx, )