from multiprocessing import context
import torch.nn as nn
import torchsparse
import torchsparse.nn as spnn
import torch
from contextlib import nullcontext
import logging

__all__ = ['TorchSparseMinkUNet']


class BasicConvolutionBlock(nn.Module):

    def __init__(self, inc, outc, ks=3, stride=1, dilation=1):
        super().__init__()
        self.net = nn.Sequential(
            spnn.Conv3d(inc,
                        outc,
                        kernel_size=ks,
                        dilation=dilation,
                        stride=stride),
            spnn.BatchNorm(outc),
            spnn.ReLU(True),
        )

    def forward(self, x):
        out = self.net(x)
        return out


class BasicDeconvolutionBlock(nn.Module):

    def __init__(self, inc, outc, ks=3, stride=1):
        super().__init__()
        self.net = nn.Sequential(
            spnn.Conv3d(inc,
                        outc,
                        kernel_size=ks,
                        stride=stride,
                        transposed=True),
            spnn.BatchNorm(outc),
            spnn.ReLU(True),
        )

    def forward(self, x):
        return self.net(x)


class ResidualBlock(nn.Module):

    def __init__(self, inc, outc, ks=3, stride=1, dilation=1):
        super().__init__()
        self.net = nn.Sequential(
            spnn.Conv3d(inc,
                        outc,
                        kernel_size=ks,
                        dilation=dilation,
                        stride=stride),
            spnn.BatchNorm(outc),
            spnn.ReLU(True),
            spnn.Conv3d(outc, outc, kernel_size=ks, dilation=dilation,
                        stride=1),
            spnn.BatchNorm(outc),
        )

        if inc == outc and stride == 1:
            self.downsample = nn.Sequential()
        else:
            self.downsample = nn.Sequential(
                spnn.Conv3d(inc, outc, kernel_size=1, dilation=1,
                            stride=stride),
                spnn.BatchNorm(outc),
            )

        self.relu = spnn.ReLU(True)

    def forward(self, x):
        out = self.relu(self.net(x) + self.downsample(x))
        return out


class MinkUNetBase(nn.Module):

    INIT_DIM = 32
    PLANES = (32, 64, 128, 256, 256, 256, 256, 256)
    LAYERS = (2, 2, 2, 2, 2, 2, 2, 2)

    def __init__(self, **kwargs):
        super().__init__()

        # cr = kwargs.get('cr', 1.0)
        # cs = [32, 32, 64, 128, 256, 256, 128, 96, 96]
        # cs = [int(cr * x) for x in cs]
        # self.run_up = kwargs.get('run_up', True)

        self.make_layers(**kwargs)    

        self.weight_initialization()
        # self.dropout = nn.Dropout(0.3, True)

        self.linear_probing = False
        self.context_manager = nullcontext()

    def set_linear_probing(self):
        self.linear_probing = True
        self.context_manager = torch.no_grad()

    # modified train function to take into account the linear probing
    def train(self, mode: bool = True):
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        self.training = mode
        if self.linear_probing:
            for module in self.children():
                    module.train(False)
            self.classifier.train(mode)
        else:
            for module in self.children():
                module.train(mode)
        return self
    
    # modified parameters function to take into account the linear probing
    def parameters(self, recurse: bool = True):
        if self.linear_probing:
            for name, param in self.named_parameters(recurse=recurse):
                if "classifier" in name:
                    yield param
        else:
            for name, param in self.named_parameters(recurse=recurse):
                yield param


    def make_layers(self, **kwargs):

        in_channels = kwargs["in_channels"]

        self.inplanes = self.INIT_DIM
        l0 = [
            spnn.Conv3d(in_channels, self.inplanes, kernel_size=3, stride=1),
            spnn.BatchNorm(self.inplanes), 
            spnn.ReLU(True),
        ]
        self.stem = nn.Sequential(*l0)

        l1 = [BasicConvolutionBlock(self.inplanes, self.inplanes, ks=2, stride=2, dilation=1),]
        for _ in range(self.LAYERS[0]):
            l1.append(ResidualBlock(self.inplanes, self.PLANES[0], ks=3, stride=1, dilation=1))
            self.inplanes = self.PLANES[0]
        self.stage1 = nn.Sequential(*l1)

        l2 = [BasicConvolutionBlock(self.inplanes, self.inplanes, ks=2, stride=2, dilation=1),]
        for _ in range(self.LAYERS[1]):
            l2.append(ResidualBlock(self.inplanes, self.PLANES[1] , ks=3, stride=1, dilation=1))
            self.inplanes = self.PLANES[1]
        self.stage2 = nn.Sequential(*l2)

        l3 = [BasicConvolutionBlock(self.inplanes, self.inplanes, ks=2, stride=2, dilation=1),]
        for _ in range(self.LAYERS[2]):
            l3.append(ResidualBlock(self.inplanes, self.PLANES[2], ks=3, stride=1, dilation=1))
            self.inplanes = self.PLANES[2]
        self.stage3 = nn.Sequential(*l3)
            
        l4 = [BasicConvolutionBlock(self.inplanes, self.inplanes, ks=2, stride=2, dilation=1),]
        for _ in range(self.LAYERS[3]):
            l4.append(ResidualBlock(self.inplanes, self.PLANES[3], ks=3, stride=1, dilation=1))
            self.inplanes = self.PLANES[3]
        self.stage4 = nn.Sequential(*l4)

        u10 = BasicDeconvolutionBlock(self.inplanes, self.PLANES[4], ks=2, stride=2)
        self.inplanes = self.PLANES[4] + self.PLANES[2]
        u11 = []
        for _ in range(self.LAYERS[4]):
            u11.append(ResidualBlock(self.inplanes, self.PLANES[4], ks=3, stride=1, dilation=1),)
            self.inplanes = self.PLANES[4]
        self.up1 = nn.ModuleList(
            [u10, nn.Sequential(*u11)]
        )

        u20 = BasicDeconvolutionBlock(self.inplanes, self.PLANES[5], ks=2, stride=2)
        self.inplanes = self.PLANES[5] + self.PLANES[1]
        u21 = []
        for _ in range(self.LAYERS[5]):
            u21.append(ResidualBlock(self.inplanes, self.PLANES[5], ks=3, stride=1, dilation=1),)
            self.inplanes = self.PLANES[5]
        self.up2 = nn.ModuleList(
            [u20, nn.Sequential(*u21)]
        )

        u30 = BasicDeconvolutionBlock(self.inplanes, self.PLANES[6], ks=2, stride=2)
        self.inplanes = self.PLANES[6] + self.PLANES[0]
        u31 = []
        for _ in range(self.LAYERS[6]):
            u31.append(ResidualBlock(self.inplanes, self.PLANES[6], ks=3, stride=1, dilation=1),)
            self.inplanes = self.PLANES[6]
        self.up3 = nn.ModuleList(
            [u30, nn.Sequential(*u31)]
        )

        u40 = BasicDeconvolutionBlock(self.inplanes, self.PLANES[7], ks=2, stride=2)
        self.inplanes = self.PLANES[7] + self.INIT_DIM
        u41 = []
        for _ in range(self.LAYERS[7]):
            u41.append(ResidualBlock(self.inplanes, self.PLANES[7], ks=3, stride=1, dilation=1),)
            self.inplanes = self.PLANES[7]
        self.up4 = nn.ModuleList(
            [u40, nn.Sequential(*u41)]
        )

        # default is we create the classifier
        if kwargs['num_classes'] > 0:
            if 'head' in kwargs and kwargs['head']=="bn_linear":
                logging.info("network - bn linear head")
                self.classifier = nn.Sequential(nn.BatchNorm1d(self.PLANES[7], affine=False), nn.Linear(self.PLANES[7], kwargs['num_classes']))
            else:
                logging.info("network - linear head")
                self.classifier = nn.Sequential(nn.Linear(self.PLANES[7], kwargs['num_classes']))
        else:
            self.classifier = nn.Identity()

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):

        with self.context_manager:
            x0 = self.stem(x)
            x1 = self.stage1(x0)
            x2 = self.stage2(x1)
            x3 = self.stage3(x2)
            x4 = self.stage4(x3)

            y1 = self.up1[0](x4)
            y1 = torchsparse.cat([y1, x3])
            y1 = self.up1[1](y1)

            y2 = self.up2[0](y1)
            y2 = torchsparse.cat([y2, x2])
            y2 = self.up2[1](y2)

            y3 = self.up3[0](y2)
            y3 = torchsparse.cat([y3, x1])
            y3 = self.up3[1](y3)

            y4 = self.up4[0](y3)
            y4 = torchsparse.cat([y4, x0])
            y4 = self.up4[1](y4)

        out = self.classifier(y4.F)

        return out


class MinkUNet(MinkUNetBase):


    def __init__(self, in_channels, out_channels,
                **kwargs
                ):
        
        super().__init__(
            in_channels = in_channels,
            num_classes = out_channels, **kwargs)

    def forward(self, data):

        coords = torch.cat([data["voxel_coords"], data["voxel_coords_batch"].unsqueeze(1)], dim=1).int()
        feats = data["voxel_x"]

        input = torchsparse.SparseTensor(coords=coords, feats=feats)

        # print(input.C.shape, input.F.shape)
        # import numpy as np
        # coord = input.C.clone().cpu().numpy()
        # feats = input.F.clone().cpu().numpy()
        # mask = coord[:,-1]==0
        # coord = coord[mask][:,:3]
        # feats = feats[mask]
        # np.savetxt("/root/no_backup/pts_ours.xyz", np.concatenate([coord, feats], axis=1))
        # exit()


        # forward in the network
        outputs = super().forward(input)

        # interpolate the outputs
        # outputs = outputs[data["sparse_input_invmap"]]

        vox_num = data["voxel_number"]
        increment = torch.cat([vox_num.new_zeros((1,)), vox_num[:-1]], dim=0)
        increment = increment.cumsum(0)
        increment = increment[data["batch"]]
        inv_map = data["voxel_to_pc_id"] + increment
        
        # interpolate the outputs
        outputs = outputs[inv_map]
        

        return outputs 
        
    def get_last_layer_channels(self):
        return self.PLANES[-1]

    
class MinkUNet34(MinkUNet):
    LAYERS = (2, 3, 4, 6, 2, 2, 2, 2)
    PLANES = (32, 64, 128, 256, 256, 128, 96, 96)


class MinkUNet18(MinkUNet):
    LAYERS = (2, 2, 2, 2, 2, 2, 2, 2)
    PLANES = (32, 64, 128, 256, 256, 128, 96, 96)


class MinkUNet18SC(MinkUNet):
    LAYERS = (2, 2, 2, 2, 2, 2, 2, 2)
    PLANES = (32, 64, 128, 256, 256, 128, 96, 96)

    def make_layers(self, **kwargs):

        in_channels = kwargs["in_channels"]

        self.inplanes = self.INIT_DIM
        l0 = [
            spnn.Conv3d(in_channels, self.inplanes, kernel_size=3, stride=1),
            spnn.BatchNorm(self.inplanes), 
            spnn.ReLU(True),
            spnn.Conv3d(self.inplanes, self.inplanes, kernel_size=3, stride=1),
            spnn.BatchNorm(self.inplanes), 
            spnn.ReLU(True),
        ]
        self.stem = nn.Sequential(*l0)

        l1 = [BasicConvolutionBlock(self.inplanes, self.inplanes, ks=2, stride=2, dilation=1),]
        for _ in range(self.LAYERS[0]):
            l1.append(ResidualBlock(self.inplanes, self.PLANES[0], ks=3, stride=1, dilation=1))
            self.inplanes = self.PLANES[0]
        self.stage1 = nn.Sequential(*l1)

        l2 = [BasicConvolutionBlock(self.inplanes, self.inplanes, ks=2, stride=2, dilation=1),]
        for _ in range(self.LAYERS[1]):
            l2.append(ResidualBlock(self.inplanes, self.PLANES[1] , ks=3, stride=1, dilation=1))
            self.inplanes = self.PLANES[1]
        self.stage2 = nn.Sequential(*l2)

        l3 = [BasicConvolutionBlock(self.inplanes, self.inplanes, ks=2, stride=2, dilation=1),]
        for _ in range(self.LAYERS[2]):
            l3.append(ResidualBlock(self.inplanes, self.PLANES[2], ks=3, stride=1, dilation=1))
            self.inplanes = self.PLANES[2]
        self.stage3 = nn.Sequential(*l3)
            
        l4 = [BasicConvolutionBlock(self.inplanes, self.inplanes, ks=2, stride=2, dilation=1),]
        for _ in range(self.LAYERS[3]):
            l4.append(ResidualBlock(self.inplanes, self.PLANES[3], ks=3, stride=1, dilation=1))
            self.inplanes = self.PLANES[3]
        self.stage4 = nn.Sequential(*l4)

        u10 = BasicDeconvolutionBlock(self.inplanes, self.PLANES[4], ks=2, stride=2)
        self.inplanes = self.PLANES[4] + self.PLANES[2]
        u11 = []
        for _ in range(self.LAYERS[4]):
            u11.append(ResidualBlock(self.inplanes, self.PLANES[4], ks=3, stride=1, dilation=1),)
            self.inplanes = self.PLANES[4]
        self.up1 = nn.ModuleList(
            [u10, nn.Sequential(*u11)]
        )

        u20 = BasicDeconvolutionBlock(self.inplanes, self.PLANES[5], ks=2, stride=2)
        self.inplanes = self.PLANES[5] + self.PLANES[1]
        u21 = []
        for _ in range(self.LAYERS[5]):
            u21.append(ResidualBlock(self.inplanes, self.PLANES[5], ks=3, stride=1, dilation=1),)
            self.inplanes = self.PLANES[5]
        self.up2 = nn.ModuleList(
            [u20, nn.Sequential(*u21)]
        )

        u30 = BasicDeconvolutionBlock(self.inplanes, self.PLANES[6], ks=2, stride=2)
        self.inplanes = self.PLANES[6] + self.PLANES[0]
        u31 = []
        for _ in range(self.LAYERS[6]):
            u31.append(ResidualBlock(self.inplanes, self.PLANES[6], ks=3, stride=1, dilation=1),)
            self.inplanes = self.PLANES[6]
        self.up3 = nn.ModuleList(
            [u30, nn.Sequential(*u31)]
        )

        u40 = BasicDeconvolutionBlock(self.inplanes, self.PLANES[7], ks=2, stride=2)
        self.inplanes = self.PLANES[7] + self.INIT_DIM
        u41 = []
        for _ in range(self.LAYERS[7]):
            u41.append(ResidualBlock(self.inplanes, self.PLANES[7], ks=3, stride=1, dilation=1),)
            self.inplanes = self.PLANES[7]
        self.up4 = nn.ModuleList(
            [u40, nn.Sequential(*u41)]
        )

        
        # default is we create the classifier
        if kwargs['num_classes'] > 0:
            if 'head' in kwargs and kwargs['head']=="bn_linear":
                logging.info("network - bn linear head")
                self.classifier = nn.Sequential(nn.BatchNorm1d(self.PLANES[7], affine=False), nn.Linear(self.PLANES[7], kwargs['num_classes']))
            else:
                logging.info("network - linear head")
                self.classifier = nn.Sequential(nn.Linear(self.PLANES[7], kwargs['num_classes']))
        else:
            self.classifier = nn.Identity()
