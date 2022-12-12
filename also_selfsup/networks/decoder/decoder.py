import torch
import torch.nn as nn
import logging

from torch_geometric.nn import radius as search_radius, knn as search_knn, avg_pool_x
import torch.nn.functional as F

from functools import partial

class InterpNet(torch.nn.Module):

    def __init__(self, latent_size, out_channels, K=1, radius=1.0,  spatial_prefix="", 
            intensity_loss=False,
            radius_search=True,
            column_search=False
            ):
        super().__init__()

        self.intensity_loss = intensity_loss
        self.out_channels = out_channels
        self.column_search = column_search

        logging.info(f"InterpNet - radius={radius} - out_channels={self.out_channels}")

        # layers of the decoder
        self.fc_in = torch.nn.Linear(latent_size+3, latent_size)
        mlp_layers = [torch.nn.Linear(latent_size, latent_size) for _ in range(2)]
        self.mlp_layers = nn.ModuleList(mlp_layers)
        self.fc_out = torch.nn.Linear(latent_size, self.out_channels)
        self.activation = torch.nn.ReLU()
        self.spatial_prefix = spatial_prefix

        # search function
        if radius_search:
            self.radius = radius
            self.K = None
            self.search_function = partial(search_radius, r=self.radius)
        else:
            self.K = int(K)
            self.radius = None
            self.search_function = partial(search_knn, k = self.K)

    def forward(self, data):

        # get the data
        if "latents_pos" in data:
            pos_source = data["latents_pos"]
            batch_source = data["latents_batch"]
        else:
            pos_source = data["pos"]
            batch_source = data["batch"]
        
        pos_target = data["pos_non_manifold"]
        batch_target = data["pos_non_manifold_batch"]
        latents = data["latents"]

        # neighborhood search
        if self.column_search:
            row, col = self.search_function(x=pos_source[:,:2], y=pos_target[:,:2], batch_x=batch_source, batch_y=batch_target)
        else:
            row, col = self.search_function(x=pos_source, y=pos_target, batch_x=batch_source, batch_y=batch_target)

        # compute reltive position between query and input point cloud
        # and the corresponding latent vectors
        pos_relative = pos_target[row] - pos_source[col]
        latents_relative = latents[col]


        x = torch.cat([latents_relative, pos_relative], dim=1)

        # Decoder layers
        x = self.fc_in(x.contiguous())
        for i, l in enumerate(self.mlp_layers):
            x = l(self.activation(x))
        x = self.fc_out(x)

        return_data = {"predictions":x[:, 0],}

        if "occupancies" in data:
            occupancies = data["occupancies"][row]
            return_data["occupancies"] = occupancies

            #### Reconstruction loss
            recons_loss = F.binary_cross_entropy_with_logits(x[:,0], occupancies.float())
            return_data["recons_loss"] = recons_loss

        #### intensity_loss
        if self.intensity_loss:
            intensities = data["intensities_non_manifold"][row].squeeze(-1)
            intensity_mask = (intensities >= 0)
            return_data["intensity_loss"] = F.l1_loss(x[:,1][intensity_mask], intensities[intensity_mask])

        return return_data


