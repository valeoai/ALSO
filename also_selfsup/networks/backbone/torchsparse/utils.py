from torchsparse.utils.quantize import sparse_quantize
import torch
import numpy as np

class Quantize(object):

    def __init__(self, voxel_size, **kwargs) -> None:
        self.voxel_size = voxel_size

    def __call__(self, data):
        
        pc_ = np.round(data["pos"].numpy() / self.voxel_size).astype(np.int32)

        pc_ -= pc_.min(0, keepdims=1)

        coords, indices, inverse_map = sparse_quantize(pc_,
                                               return_index=True,
                                               return_inverse=True)

        coords = torch.tensor(coords, dtype=torch.int)

        indices = torch.tensor(indices)
        feats = data["x"][indices]

        inverse_map = torch.tensor(inverse_map, dtype=torch.long)
        
        data["voxel_coords"] = coords
        data["voxel_x"] = feats 
        data["voxel_to_pc_id"] = inverse_map
        data["voxel_number"] = coords.shape[0] 

        return data