import torch
import numpy as np
import logging

class VoxelDecimation(object):

    def __init__(self, voxel_size) -> None:
        logging.info(f"Transforms - VoxelDecimation - {voxel_size}")
        self.v_size = voxel_size

    def __call__(self, data):

        
        pos = data["pos"]
        pos = (pos/self.v_size).long()
        num_pts = pos.shape[0]

        # Numpy version
        pos, indices = np.unique(pos.cpu().numpy(), return_index=True, axis=0)
        
        for key in data.keys:
            if isinstance(data[key], torch.Tensor) and ("second" not in key) and data[key].shape[0] == num_pts:
                data[key] = data[key][indices]

        # if second frame --> decimation of the second frame
        if "second_pos" in data.keys:
            pos = data["second_pos"]
            pos = (pos/self.v_size).long()
            num_pts = pos.shape[0]
            pos, indices = np.unique(pos.cpu().numpy(), return_index=True, axis=0)
            
            for key in data.keys:
                if isinstance(data[key], torch.Tensor) and ("second" in key) and data[key].shape[0] == num_pts:
                    data[key] = data[key][indices]

        return data