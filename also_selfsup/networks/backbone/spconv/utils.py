
from spconv.pytorch.utils import PointToVoxel
# from spconv.utils import Point2VoxelCPU3d as PointToVoxel
from torch_geometric.nn import global_mean_pool
import torch
import re
import numpy as np

#####################################################################
# transformation between Cartesian coordinates and polar coordinates
# code from https://github.com/xinge008/Cylinder3D/
# please refer to the repo

def cart2polar(input_xyz):
    rho = torch.sqrt(input_xyz[:, 0] ** 2 + input_xyz[:, 1] ** 2)
    phi = torch.atan2(input_xyz[:, 1], input_xyz[:, 0])
    return torch.stack((rho, phi, input_xyz[:, 2]), dim=1)


def polar2cart(input_xyz_polar):
    # print(input_xyz_polar.shape)
    x = input_xyz_polar[0] * torch.cos(input_xyz_polar[1])
    y = input_xyz_polar[0] * torch.sin(input_xyz_polar[1])
    return torch.stack((x, y, input_xyz_polar[2]), dim=0)
##########################################################


class SpatialExtentCrop(object):


    def __init__(self, spatial_extent, item_list=None, croping_ref_field="pos"):
        self.spatial_extent  = spatial_extent
        self.item_list = item_list
        self.croping_ref_field = croping_ref_field

    def __call__(self, data):

        if self.item_list is None:
            num_nodes = data.num_nodes
        else:
            num_nodes = data[self.item_list[0]].shape[0]

        ref_field = data[self.croping_ref_field]

        mask_x = torch.logical_and(ref_field[:,0]>self.spatial_extent[0], ref_field[:,0]<self.spatial_extent[3])
        mask_y = torch.logical_and(ref_field[:,1]>self.spatial_extent[1], ref_field[:,1]<self.spatial_extent[4])
        mask_z = torch.logical_and(ref_field[:,2]>self.spatial_extent[2], ref_field[:,2]<self.spatial_extent[5])
        mask = mask_x & mask_y & mask_z

        # selecting elements
        if self.item_list is None:
            for key, item in data:
                if bool(re.search('edge', key)):
                    continue
                if (torch.is_tensor(item) and item.size(0) == num_nodes
                        and item.size(0) != 1):
                    data[key] = item[mask]
        else:
            for key, item in data:
                if key in self.item_list:
                    if bool(re.search('edge', key)):
                        continue
                    if (torch.is_tensor(item) and item.size(0) != 1):
                        data[key] = item[mask]
        return data

    def __repr__(self):
        return '{}({}, replace={})'.format(self.__class__.__name__, self.num,
                                           self.replace)


class Quantize(object):

    def __init__(self, voxel_size, spatial_extent, **kwargs) -> None:

        print("Quantize - init")
        print(voxel_size)
        print(spatial_extent)

        self.voxel_size = voxel_size
        self.spatial_extent = spatial_extent
        self.cylinder_coords = kwargs["cylinder_coords"] if "cylinder_coords" in kwargs else False
        self.voxel_num = kwargs["voxel_num"] if "voxel_num" in kwargs else None

        self.num_features = kwargs["num_features"] if "num_features" in kwargs else 4

        if self.voxel_num is not None:
            self.voxel_sizes = [
                (spatial_extent[3]-spatial_extent[0])/self.voxel_num[0],
                (spatial_extent[4]-spatial_extent[1])/self.voxel_num[1],
                (spatial_extent[5]-spatial_extent[2])/self.voxel_num[2],
                ]
            self.gen = PointToVoxel(
                vsize_xyz=self.voxel_sizes, 
                coors_range_xyz=[
                        self.spatial_extent[0], 
                        self.spatial_extent[1], 
                        self.spatial_extent[2], 
                        self.spatial_extent[3]+1, 
                        self.spatial_extent[4]+1, 
                        self.spatial_extent[5]+1, 
                    ],
                num_point_features=self.num_features, 
                max_num_voxels=100000, 
                max_num_points_per_voxel=5)

        else:
            if isinstance(self.voxel_size, list):
                self.gen = PointToVoxel(
                    vsize_xyz=self.voxel_size, 
                    coors_range_xyz=self.spatial_extent, 
                    num_point_features=4, 
                    max_num_voxels=100000, 
                    max_num_points_per_voxel=5)
            else:
                self.gen = PointToVoxel(
                    vsize_xyz=[self.voxel_size, self.voxel_size, self.voxel_size], 
                    coors_range_xyz=self.spatial_extent, 
                    num_point_features=4, 
                    max_num_voxels=100000, 
                    max_num_points_per_voxel=5)

        if self.cylinder_coords:
            self.prior_crop = SpatialExtentCrop(spatial_extent, croping_ref_field="pos_pol")
        else:
            self.prior_crop = SpatialExtentCrop(spatial_extent, croping_ref_field="pos")

    def __call__(self, data):

        data = self.prior_crop(data)

        voxels, coords, num_points_per_voxel, pc_voxel_id = self.gen.generate_voxel_with_id(data["x"], empty_mean=True)

        x_pool = voxels[:, :, :].sum(dim=1, keepdim=False)
        normalizer = torch.clamp_min(num_points_per_voxel.view(-1, 1), min=1.0).type_as(x_pool)
        x_pool = x_pool / normalizer
        x_pool = x_pool.contiguous()

        data["voxel_coords"] = coords
        data["voxel_x"] = x_pool # features
        data["voxel_to_pc_id"] = pc_voxel_id # if index in the key, will be incremented automatically by pytorch geometric
        data["voxel_number"] = coords.shape[0] # number of voxels
        
        return data