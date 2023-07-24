import yaml
import logging
import argparse
import warnings
import importlib

from tqdm import tqdm

from scipy.spatial import KDTree

import torch

from torch_geometric.data import DataLoader

from utils.utils import wgreen
from utils.confusion_matrix import ConfusionMatrix
from transforms import get_transforms, get_input_channels

import datasets
import networks
import networks.decoder
from networks.backbone import *


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)

    logging.getLogger().setLevel("INFO")

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--split', type=str, required=True)
    opts = parser.parse_args()

    config = yaml.load(open(opts.config, "r"), yaml.FullLoader)

    logging.info("Dataset")
    DatasetClass = eval("datasets."+config["dataset_name"])
    test_transforms = get_transforms(config, train=False, downstream=True, keep_orignal_data=True)
    test_dataset = DatasetClass(config["dataset_root"],
                split=opts.split,
                transform=test_transforms,
                )

    logging.info("Dataloader")
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config["threads"],
        follow_batch=["voxel_coords"]
    )

    num_classes = config["downstream"]["num_classes"]
    device = torch.device("cuda")

    logging.info("Network")
    if config["network"]["backbone_params"] is None:
        config["network"]["backbone_params"] = {}
    config["network"]["backbone_params"]["in_channels"] = get_input_channels(config["inputs"])
    config["network"]["backbone_params"]["out_channels"] = config["downstream"]["num_classes"]

    backbone_name = "networks.backbone."
    if config["network"]["framework"] is not None:
        backbone_name += config["network"]["framework"]
    importlib.import_module(backbone_name)
    backbone_name += "."+config["network"]["backbone"]
    net = eval(backbone_name)(**config["network"]["backbone_params"])
    net.to(device)
    net.eval()

    logging.info("Loading the weights from pretrained network")
    net.load_state_dict(torch.load(opts.ckpt), strict=True)

    cm = ConfusionMatrix(num_classes, 0)
    with torch.no_grad():
        t = tqdm(test_loader, ncols=100)
        for data in t:

            data = data.to(device)

            # predictions
            predictions = net(data)
            predictions = torch.nn.functional.softmax(predictions[:,1:], dim=1).max(dim=1)[1]
            predictions = predictions.cpu().numpy() + 1

            # interpolate to original point cloud
            original_pos_np = data["original_pos"].cpu().numpy()
            pos_np = data["pos"].cpu().numpy()
            tree = KDTree(pos_np)
            _, indices = tree.query(original_pos_np, k=1)
            predictions = predictions[indices]

            # update the confusion matric
            targets_np = data["original_y"].cpu().numpy()
            cm.update(targets_np, predictions)

            # compute metrics
            iou_per_class = cm.get_per_class_iou()
            miou = cm.get_mean_iou()
            freqweighted_iou = cm.get_freqweighted_iou()
            description = f"Val. | mIoU {miou*100:.2f} - fIoU {freqweighted_iou*100:.2f}"
            t.set_description_str(wgreen(description))

            torch.cuda.empty_cache()

    logging.info(f"MIoU: {miou}")
    logging.info(f"FIoU: {freqweighted_iou}")
    logging.info(f"IoU per class: {iou_per_class}")
