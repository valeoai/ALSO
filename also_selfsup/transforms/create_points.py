import torch
import numpy as np
import torch.nn.functional as F
import logging
import re


class CreatePoints(object):

    def __init__(self, npts=None, exact_number_of_points=None, pts_item_list=None, n_non_manifold_pts=None, non_manifold_dist=0.1):

        logging.info(f"Transforms - CreatePoints - npts {npts} - exact_number_of_points {exact_number_of_points} - non_manifold {n_non_manifold_pts} - non_manifold_dist {non_manifold_dist}")
        self.npts = npts
        self.exact_number_of_points = exact_number_of_points
        self.n_non_manifold_pts = n_non_manifold_pts
        self.pts_item_list = pts_item_list
        self.non_manifold_dist = non_manifold_dist

    def __call__(self, data):

        num_nodes = data.num_nodes if (self.pts_item_list is None) else data[self.pts_item_list[0]].shape[0]

        # select the points
        choice = None
        if self.npts < num_nodes:
            choice = torch.randperm(num_nodes)[:self.npts]
        elif self.npts > num_nodes and self.exact_number_of_points:
            choice = np.random.choice(num_nodes, self.npts, replace=True)
            choice = torch.from_numpy(choice).to(torch.long)

        # non manifold points
        if self.n_non_manifold_pts is not None:

            # nmp -> non_manifold points
            if "pos2" in data.keys:
                
                n_nmp2 = self.n_non_manifold_pts // 2
                n_nmp = self.n_non_manifold_pts - n_nmp2

                
                n_nmp2_out = n_nmp2 // 3
                n_nmp2_out_far = n_nmp2 // 3
                n_nmp2_in = n_nmp2 - 2 * (n_nmp2//3)
                nmp2_choice_in = torch.randperm(data["pos2"].shape[0])[:n_nmp2_in]
                nmp2_choice_out = torch.randperm(data["pos2"].shape[0])[:n_nmp2_out]
                nmp2_choice_out_far = torch.randperm(data["pos2"].shape[0])[:n_nmp2_out_far]

            else:
                n_nmp2 = 0
                n_nmp = self.n_non_manifold_pts

            # select the points for the current frame
            n_nmp_out = n_nmp // 3
            n_nmp_out_far = n_nmp // 3
            n_nmp_in = n_nmp - 2 * (n_nmp//3)
            nmp_choice_in = torch.randperm(data["pos"].shape[0])[:n_nmp_in]
            nmp_choice_out = torch.randperm(data["pos"].shape[0])[:n_nmp_out]
            nmp_choice_out_far = torch.randperm(data["pos"].shape[0])[:n_nmp_out_far]

            # center
            center = torch.zeros((1,3), dtype=torch.float)

            # in points
            pos = data["pos"][nmp_choice_in]
            dirs = F.normalize(pos, dim=1)
            pos_in = pos + self.non_manifold_dist * dirs * torch.rand((pos.shape[0],1))
            occ_in = torch.ones(pos_in.shape[0], dtype=torch.long)
            
            # out points
            pos = data["pos"][nmp_choice_out]
            dirs = F.normalize(pos, dim=1)
            pos_out = pos - self.non_manifold_dist * dirs * torch.rand((pos.shape[0],1))
            occ_out = torch.zeros(pos_out.shape[0], dtype=torch.long)

            # out far points
            pos = data["pos"][nmp_choice_out_far]
            dirs = F.normalize(pos, dim=1)
            pos_out_far = (pos - center) * torch.rand((pos.shape[0],1)) + center
            occ_out_far = torch.zeros(pos_out_far.shape[0], dtype=torch.long)


            pos_non_manifold = torch.cat([pos_in, pos_out, pos_out_far], dim=0)
            occupancies = torch.cat([occ_in, occ_out, occ_out_far], dim=0)
            intensities = None
            rgb = None

            if "intensities" in data:
                intensities_in = data["intensities"][nmp_choice_in]
                intensities_out = data["intensities"][nmp_choice_out]
                intensities_out_far = torch.full((pos_out_far.shape[0],1), fill_value=-1)
                intensities = torch.cat([intensities_in, intensities_out, intensities_out_far], dim=0)

            if "rgb" in data:
                rgb_in = data["rgb"][nmp_choice_in]
                rgb_out = data["rgb"][nmp_choice_out]
                rgb_out_far = torch.full((pos_out_far.shape[0],3), fill_value=-1)
                rgb = torch.cat([rgb_in, rgb_out, rgb_out_far], dim=0)


            if n_nmp2 > 0:
                # multiframe setting

                # in points
                pos = data["pos2"][nmp2_choice_in]
                dirs = F.normalize(pos - data["sensors2"][nmp2_choice_in], dim=1)
                pos_in = pos + self.non_manifold_dist * dirs * torch.rand((pos.shape[0],1))
                occ_in = torch.ones(pos_in.shape[0], dtype=torch.long)
                
                # out points
                pos = data["pos2"][nmp2_choice_out]
                dirs = F.normalize(pos - data["sensors2"][nmp2_choice_out], dim=1)
                pos_out = pos - self.non_manifold_dist * dirs * torch.rand((pos.shape[0],1))
                occ_out = torch.zeros(pos_out.shape[0], dtype=torch.long)

                # out far points
                pos = data["pos2"][nmp2_choice_out_far]
                dirs = F.normalize(pos - data["sensors2"][nmp2_choice_out_far], dim=1)
                pos_out_far = (pos - center) * torch.rand((pos.shape[0],1)) + center
                occ_out_far = torch.zeros(pos_out_far.shape[0], dtype=torch.long)


                pos_non_manifold2 = torch.cat([pos_in, pos_out, pos_out_far], dim=0)
                occupancies2 = torch.cat([occ_in, occ_out, occ_out_far], dim=0)
                intensities2 = None

                pos_non_manifold = torch.cat([pos_non_manifold, pos_non_manifold2], dim=0)
                occupancies = torch.cat([occupancies, occupancies2], dim=0)

                if "intensities2" in data:
                    intensities_in = data["intensities2"][nmp2_choice_in]
                    intensities_out = data["intensities2"][nmp2_choice_out]
                    intensities_out_far = torch.full((pos_out_far.shape[0],1), fill_value=-1)
                    intensities2 = torch.cat([intensities_in, intensities_out, intensities_out_far], dim=0)
                    intensities = torch.cat([intensities, intensities2], dim=0)

            data["pos_non_manifold"] = pos_non_manifold
            data["occupancies"] = occupancies
            if intensities is not None:
                data["intensities_non_manifold"] = intensities

            if rgb is not None:
                data["rgb_non_manifold"] = rgb

            


        # replace in data
        if choice is not None:

            # selecting elements
            if self.pts_item_list is None:
                for key, item in data:
                    if bool(re.search('edge', key)):
                        continue
                    if (torch.is_tensor(item) and item.size(0) == num_nodes
                            and item.size(0) != 1):
                        data[key] = item[choice]
            else:
                for key, item in data:
                    if key in self.pts_item_list:
                        if bool(re.search('edge', key)):
                            continue
                        if (torch.is_tensor(item) and item.size(0) != 1):
                            data[key] = item[choice]

        return data