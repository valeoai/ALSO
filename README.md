<div align='center'>

# ALSO: Automotive Lidar Self-supervision by Occupancy estimation

[Alexandre Boulch](https://boulch.eu/)<sup>1,3</sup>&nbsp;&nbsp;&nbsp;
[Corentin Sautier](https://fr.linkedin.com/in/corentin-sautier-74415917b)<sup>1,2</sup>&nbsp;&nbsp;&nbsp;
[Björn Michele](https://scholar.google.com/citations?user=xQcKnXkAAAAJ&hl=en&oi=ao)<sup>1,4</sup>&nbsp;&nbsp;&nbsp;
[Gilles Puy](https://sites.google.com/site/puygilles/)<sup>1,3</sup>&nbsp;&nbsp;&nbsp;
[Renaud Marlet](http://imagine.enpc.fr/~marletr/)<sup>1,2,3</sup>

<sub>
<sup>1</sup> Valeo.ai, Paris, France
<sup>2</sup> LIGM, Ecole des Ponts, Univ Gustave Eiffel, CNRS, Marne-la-Vallée, France

<sup>3</sup> Inria, Paris, France
<sup>4</sup> CNRS, IRISA, Univ. Bretagne Sud, Vannes, France
</sub>

<br/>

[![Arxiv](http://img.shields.io/badge/paper-arxiv.XXXX.XXXXX-B31B1B.svg)](https://github.com/valeoai/ALSO)

<br/>


![Overview](doc/overview.png)

This is the reference PyTorch implementation for training and testing self-supervision for automotive lidar point cloud using the method described in our paper [**ALSO: Automotive Lidar Self-supervision by Occupancy estimation**](https://github.com/valeoai/ALSO)

</div>

## Overview

- [Citation](#citation)
- [Dependencies](#dependencies)
- [Installation](#installation)
- [Semantic segmentation](#semantic-segmentation)
- [Pre-trained models](#pre-trained-models)
- [Performances](#performances)
- [Acknowledgments](#acknowledgments)

---

## Citation

Please acknowledge our work in your publications:

```
To be added
```

## Dependencies

We give the library version(s), we used for experiments.

- Hydra (1.2.0)
- Pytorch-geometric (2.1.0)
- Pytorch-Lightning (1.7.2)

*Semantic segmentation specific libraries*
- torchsparse (1.4.0)

*Detection specific libraries*
- spconv (2.1.21)
- openpcdet (0.6.0)

## Installation

At root of the folder run:
```
pip install -ve .
```
Then all scripts can be run from `also_selfsup` folder:
```
cd also_selfsup
```

## Datasets

### ONCE

We follow intruction from 
[ONCE dataset](https://github.com/PointsCoder/ONCE_Benchmark/blob/master/docs/GETTING_STARTED.md) to generate the information files of the `raw_small` dataset.


## Semantic segmentation

### Pre-training


```
python train_selfsupervised.py cfg=nuscenes
```

for a pre-training on [nuScenes](https://www.nuscenes.org/nuscenes).

for other datasets, replace `cfg=nuscenes` by the desired dataset
- [nuScenes](https://www.nuscenes.org/nuscenes): `cfg=nuscenes`
- [SemanticKITTI](https://www.nuscenes.org/nuscenes): `cfg=semantickitti`
- [SemanticPOSS](http://www.poss.pku.edu.cn/semanticposs.html): *To be added*
- [LivoxSimuDataset](https://www.livoxtech.com/simu-dataset): *To be added*

### Parameters

We use [Hydra](https://hydra.cc/) for easy configuration of trainings.
The pretraining parameters are set for pre-training with a NVidia-V100 16Gb.
In order to pre-train on other configurations, one can modify parameters directly in the command line, e.g., setting the batch size to 8:
```
python train_selfsupervised.py ... cfg.training.batch_size=8
```
or the backbone to SPVCNN:
```
python train_selfsupervised.py ... cfg.network.backbone=SPVCNN
```

### Downstream

#### From scratch

```bash

# 100%
python train_downstream_semseg.py cfg=nuscenes cfg.downstream.max_epochs=30 cfg.val_interval=5 cfg.downstream.skip_ratio=1

# 50%
python train_downstream_semseg.py cfg=nuscenes cfg.downstream.max_epochs=50 cfg.val_interval=5 cfg.downstream.skip_ratio=2

# 10%
python train_downstream_semseg.py cfg=nuscenes cfg.downstream.max_epochs=100 cfg.val_interval=10 cfg.downstream.skip_ratio=10

# 1%
python train_downstream_semseg.py cfg=nuscenes cfg.downstream.max_epochs=500 cfg.val_interval=50 cfg.downstream.skip_ratio=100

# 0.1%
python train_downstream_semseg.py cfg=nuscenes cfg.downstream.max_epochs=1000 cfg.val_interval=100 cfg.downstream.skip_ratio=1000
```

#### With pretrained models

First, convert the pre-trained model to be loaded
```bash
python convert_models.py --ckpt path_to_pretraining_checkpoint
```

Second, run downstream training
```bash
python train_downstream_semseg.py cfg=nuscenes cfg.downstream.checkpoint_dir='path_to_checkpoint_directory' cfg.downstream.checkpoint_name='pretrained_backbone_XXX.ckpt'
```

#### Evaluation of downstream models

First, convert the trained model to be loaded
```bash
python convert_models.py --downstream --ckpt path_to_downstream_checkpoint
```

Second, run the evaluation script
```bash
python eval.py --split val --config path_to_downstream_model/config.yaml --ckpt path_to_downstream_checkpoint/trained_model_XXX.ckpt
```

## Detection

### Pre-training

Similar to semantic segmentation, pre-training for detection can simply be run using:

```bash
python train_selfsupervised.py cfg=kitti3d_second
python train_selfsupervised.py cfg=nuscenes_second_kitti
```

### Downstream

First, convert the trained model to be loaded
```bash
python convert_models.py --ckpt path_to_pretraining_checkpoint
```

Second, use the pre-trained model with the official OpenPCDet code:
```bash
cd path_to_OpenPCDet/tools/
python train.py --cfg_file cfgs/kitti_models/second.yaml --extra_tag --pretrained_model path_to_pretrained_model/pretrained_backbone_XXX.ckpt
```

## Pre-trained models

Task | Pre-training dataset | Model | Link | Notes |
---|---|---|---|---|
Semantic segmentation | nuScenes | MinkUNet34   | [download](https://github.com/valeoai/ALSO/releases/download/v0.0.0/pretrained_backbone_also_nuscenes_minkunet_epoch_199.ckpt) |
&nbsp; | &nbsp; | SPVCNN     | [download](https://github.com/valeoai/ALSO/releases/download/v0.0.0/pretrained_backbone_also_nuscenes_spvcnn_epoch_199.ckpt) |
&nbsp; | SemanticKITTI | MinkUNet18SC  | [download](https://github.com/valeoai/ALSO/releases/download/v0.0.0/pretrained_backbone_also_semantickitti_minkunetsc_epoch_49.ckpt) |
&nbsp; | &nbsp; | SPVCNN     | [download](https://github.com/valeoai/ALSO/releases/download/v0.0.0/pretrained_backbone_also_semantickitti_spvcnn_epoch_49.ckpt) |
Detection | KITTI3D | SECOND/PVRCNN | [download](https://github.com/valeoai/ALSO/releases/download/v0.0.0/pretrained_backbone_also_kitti3d_second_epoch_499.ckpt) | trained for KITTI3D |
&nbsp; | nuScenes | SECOND/PVRCNN | [download](https://github.com/valeoai/ALSO/releases/download/v0.0.0/pretrained_backbone_also_nuscenes_second_epoch_99.ckpt) | trained for KITTI3D |
&nbsp; | KITTI360 | SECOND/PVRCNN | [download](https://github.com/valeoai/ALSO/releases/download/v0.0.0/pretrained_backbone_also_kitti360_second_epoch_74.ckpt) | trained for KITTI3D |
&nbsp; | ONCE Small | SECOND/PVRCNN | [download](https://github.com/valeoai/ALSO/releases/download/v0.0.0/pretrained_backbone_also_once_second_epoch_39.ckpt) | trained for ONCE |


Here are links to pre-trained models

## Acknowledgments

This project would not have been possible without many community ressources and repositories. Among them:

- [PointContrast](https://github.com/facebookresearch/PointContrast)
- [SLidR](https://github.com/valeoai/SLidR)
- [SegContrast](https://github.com/PRBonn/segcontrast)
- [SPVNAS](https://github.com/mit-han-lab/spvnas)
- [POCO](https://github.com/valeoai/POCO)

Please, consider acknowleding these project.
