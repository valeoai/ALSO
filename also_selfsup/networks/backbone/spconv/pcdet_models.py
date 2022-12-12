import torch
from pcdet.models import build_network
from pcdet.config import cfg, cfg_from_yaml_file
import numpy as np

class custom_point_feature_encoder:
    def __init__(self, num):
        self.num_point_features = num

class custom_dataset:

    def __init__(self, cfg, class_names) -> None:
        
        self.dataset_cfg = cfg
        
        self.point_cloud_range = np.array(self.dataset_cfg.POINT_CLOUD_RANGE, dtype=np.float32)
        self.point_feature_encoder = custom_point_feature_encoder(len(self.dataset_cfg.POINT_FEATURE_ENCODING["used_feature_list"]))

        self.class_names = class_names

        for cur in self.dataset_cfg.DATA_PROCESSOR:
            print(cur)
            if "VOXEL_SIZE" in cur:
                self.voxel_size = cur["VOXEL_SIZE"]

        grid_size = (self.point_cloud_range[3:6] - self.point_cloud_range[0:3]) / np.array(self.voxel_size)
        self.grid_size = np.round(grid_size).astype(np.int64)


        self.depth_downsample_factor = None

class SECOND(torch.nn.Module):

    def __init__(self, in_channels, out_channels,
                **kwargs):

        super().__init__()

        config_path= kwargs["config"]

        cfg_from_yaml_file(config_path, cfg)

        self.train_set = custom_dataset(cfg.DATA_CONFIG, cfg.CLASS_NAMES)
        self.model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=self.train_set)

        self.classifier = torch.nn.Conv2d(512, out_channels, kernel_size=1)


    def forward(self, data):

        coords = torch.cat([data["voxel_coords_batch"].unsqueeze(1), data["voxel_coords"]], dim=1).int()
        features = data["voxel_x"]
        batch_size = data["voxel_coords_batch"][-1] + 1
        
        outputs = self.model.backbone_3d.forward(
            {"batch_size":batch_size, "voxel_features":features, "voxel_coords":coords})

        outputs = self.model.map_to_bev_module(outputs)

        outputs = self.model.backbone_2d(outputs)

        outputs = outputs["spatial_features_2d"]
        outputs = self.classifier(outputs)

        y = torch.arange(outputs.shape[2])
        x = torch.arange(outputs.shape[3])
        
        grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')
        grid_x = grid_x.float().to(outputs.device)/grid_x.max()*(self.train_set.point_cloud_range[3] - self.train_set.point_cloud_range[0]) + self.train_set.point_cloud_range[0]
        grid_y = grid_y.float().to(outputs.device)/grid_y.max()*(self.train_set.point_cloud_range[4] - self.train_set.point_cloud_range[1]) + self.train_set.point_cloud_range[1]
        grid_z = torch.zeros_like(grid_x, dtype=torch.float, device=outputs.device)
        
        points = torch.stack([grid_x, grid_y, grid_z], dim=-1).reshape(-1, 3)
        points = points.repeat(outputs.shape[0], 1)

        points_batch = torch.arange(batch_size, device=outputs.device, dtype=torch.long).reshape(-1,1,1).expand((outputs.shape[0], outputs.shape[2], outputs.shape[3]))
        points_batch = points_batch.reshape(-1)

        points_outputs = outputs.permute(0,2,3,1).reshape(-1, outputs.shape[1])

        return {
            'latents': points_outputs,
            'latents_pos': points,
            'latents_batch': points_batch
        }