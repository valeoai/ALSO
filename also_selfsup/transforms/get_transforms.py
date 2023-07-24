import torch_geometric.transforms as T
import logging

import importlib

from transforms.create_inputs import CreateInputs
from transforms.create_points import CreatePoints
from transforms.dupplicate import Dupplicate
from transforms.random_rotate import RandomRotate
from transforms.random_flip import RandomFlip
from transforms.scaling import Scaling
from transforms.voxel_decimation import VoxelDecimation


class CleanData(object):

    def __init__(self, prefixes=[], item_list=[]):
        self.prefixes = prefixes
        self.item_list = item_list

    def __call__(self, data):

        for prefix in self.prefixes:
            for key in data.keys:
                if key.startswith(prefix):
                    data[key] = None

        for key in self.item_list:
            if key in data.keys:
                data[key] = None

        return data



class ToDict(object):

    def __call__(self, data):

        d = {}
        for key in data.keys:
            d[key] = data[key]
        return d

    def __repr__(self):
        return '{}'.format(self.__class__.__name__)

def get_input_channels(input_config):

    # compute the input size:
    in_channels = 0
    for key in input_config:
        if key in ["intensities", "x"]:
            in_channels += 1
        elif key in ["pos", "rgb", "normals", "dirs"]:
            in_channels += 3

    return in_channels


def get_transforms(config, network_function=None, train=True, downstream=False, keep_orignal_data=False):

    logging.info(f"Transforms - Train {train} - Downstream {downstream}")

    augmentations = config["transforms"]
    print(augmentations)
    transforms = []

    if keep_orignal_data:
        transforms.append(Dupplicate(["pos", "y"], "original_"))

    if augmentations['voxel_decimation'] is not None:
        transforms.append(VoxelDecimation(augmentations["voxel_decimation"]))
    

    exact_number_of_points = (config["network"]["backbone"] in ["FKAConv", "DGCNN"])
    n_non_manifold_pts = config["non_manifold_points"] if (not downstream) else None
    non_manifold_dist = config["non_manifold_dist"] if "non_manifold_dist" in config else 0.1
    
    transforms.append(CreatePoints(npts=config["manifold_points"], 
        exact_number_of_points=exact_number_of_points, 
        pts_item_list=["x", "pos", "y", "intensities"], 
        n_non_manifold_pts=n_non_manifold_pts, non_manifold_dist=non_manifold_dist))
    
    if augmentations["scaling_intensities"]:
        logging.info("Transforms - Scale intensities")
        transforms.append(Scaling(255., item_list=["intensities", "intensities_non_manifold"]))


    transforms.append(CleanData(prefixes=[], item_list=["pos2", "intensities2", "sensors2"]))

    if train:
        if augmentations["random_rotation_z"]:
            transforms.append(RandomRotate(180, axis=2, item_list=["pos", "pos_non_manifold"]))

        if augmentations["random_flip"]:
            logging.info("Transforms - Flip")
            transforms.append(RandomFlip(["pos", "pos_non_manifold"]))

    transforms.append(CreateInputs(config["inputs"]))

    if config["network"]["framework"] is not None:
        logging.info(f"Transforms - Quantize - {config['network']['framework']}")
        model_module = importlib.import_module("networks.backbone." + config["network"]["framework"])
        transforms.append(model_module.Quantize(**config["network"]["backbone_params"]["quantization_params"]))

    transforms = T.Compose(transforms)

    return transforms
