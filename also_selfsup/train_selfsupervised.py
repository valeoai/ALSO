import os
import yaml
import logging
import warnings
import importlib

import hydra
from omegaconf import DictConfig, OmegaConf

import torch
import numpy as np

from sklearn.metrics import confusion_matrix


from torch_geometric.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor

import datasets
import networks.decoder

import utils.metrics as metrics
from utils.utils import wblue, wgreen
from utils.callbacks import CustomProgressBar
from transforms import get_transforms, get_input_channels


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def save_config_file(config, filename):
    with open(filename, 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)

def logs_file(filepath, epoch, log_data):

    
    if not os.path.exists(filepath):
        log_str = f"epoch"
        for key, value in log_data.items():
            log_str += f", {key}"
        log_str += "\n"
        with open(filepath, "a+") as logs:
            logs.write(log_str)
            logs.flush()

    # write the logs
    log_str = f"{epoch}"
    for key, value in log_data.items():
        log_str += f", {value}"
    log_str += "\n"
    with open(filepath, "a+") as logs:
        logs.write(log_str)
        logs.flush()


class LightningSelfSupervisedTrainer(pl.LightningModule):

    def __init__(self, config):
        super().__init__()

        self.config = config
        
        if config["network"]["backbone_params"] is None:
            config["network"]["backbone_params"] = {}
        config["network"]["backbone_params"]["in_channels"] = get_input_channels(config["inputs"])
        config["network"]["backbone_params"]["out_channels"] = config["network"]["latent_size"]

        backbone_name = "networks.backbone."
        if config["network"]["framework"] is not None:
            backbone_name += config["network"]["framework"]
        importlib.import_module(backbone_name)
        backbone_name += "."+config["network"]["backbone"]

        logging.info(f"Backbone - {backbone_name}")

        self.backbone = eval(backbone_name)(**config["network"]["backbone_params"])

        logging.info(f"Decoder - {config['network']['decoder']}")

        config["network"]["decoder_params"]["latent_size"] = config["network"]["latent_size"]
        self.decoder = eval("networks.decoder."+config["network"]["decoder"])(**config["network"]["decoder_params"])

        self.train_cm = np.zeros((2,2))
        self.val_cm = np.zeros((2,2))

    def forward(self, data):
        outputs = self.backbone(data)

        if isinstance(outputs, dict):
            for k,v in outputs.items():
                data[k] = v
        else:
            data["latents"] = outputs

        return_data = self.decoder(data)

        return return_data

    
    def configure_optimizers(self):
        optimizer = eval(self.config["optimizer"])(self.parameters(), **self.config["optimizer_params"])
        return optimizer


    def compute_confusion_matrix(self, output_data):
        outputs = output_data["predictions"].squeeze(-1)
        occupancies = output_data["occupancies"].float()
        
        output_np = (torch.sigmoid(outputs).cpu().detach().numpy() > 0.5).astype(int)
        target_np = occupancies.cpu().numpy().astype(int)
        cm = confusion_matrix(
            target_np.ravel(), output_np.ravel(), labels=list(range(2))
        )
        return cm


    def compute_loss(self, output_data, prefix):

        loss = 0
        loss_values = {}
        for key, value in output_data.items():
            if "loss" in key and (self.config["loss"][key+"_lambda"] > 0):
                loss = loss + self.config["loss"][key+"_lambda"] * value
                self.log(prefix+"/"+key, value.item(), on_step=True, on_epoch=False, prog_bar=True, logger=False)
                loss_values[key] = value.item()

        # log also the total loss
        self.log(prefix+"/loss", loss.item(), on_step=True, on_epoch=False, prog_bar=True, logger=False)

        if self.train_cm.sum() > 0:
            self.log(prefix+"/iou", metrics.stats_iou_per_class(self.train_cm)[0], on_step=True, on_epoch=False, prog_bar=True, logger=False)

        return loss, loss_values

    def on_train_epoch_start(self) -> None:
        self.train_cm = np.zeros((2,2))
        return super().on_train_epoch_start()

    def on_validation_epoch_start(self) -> None:
        self.val_cm = np.zeros((2,2))
        return super().on_validation_epoch_start()

    def training_step(self, data, batch_idx):

        if batch_idx % 10 == 0:
            torch.cuda.empty_cache()

        output_data = self.forward(data)
        loss, individual_losses = self.compute_loss(output_data, prefix="train")
        cm = self.compute_confusion_matrix(output_data)
        self.train_cm += cm
        
        individual_losses["loss"] = loss

        return individual_losses

    def validation_step(self, data, batch_idx):
        
        if batch_idx % 10 == 0:
            torch.cuda.empty_cache()
        
        output_data = self.forward(data)
        loss, individual_losses = self.compute_loss(output_data, prefix="val")
        cm = self.compute_confusion_matrix(output_data)
        self.val_cm += cm
        
        individual_losses["loss"] = loss

        return individual_losses
    

    def compute_log_data(self, outputs, cm, prefix):
        
        # compute iou
        iou = metrics.stats_iou_per_class(cm)[0]
        self.log(prefix+"/iou", iou, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        log_data = {}
        keys = outputs[0].keys()
        for key in keys:
            if "loss" not in key:
                continue
            if key == "loss":
                loss = np.mean([d[key].item() for d in outputs])
            else:
                loss = np.mean([d[key] for d in outputs])
            log_data[key] = loss

        log_data["iou"] = iou
        log_data["steps"] = self.global_step

        return log_data

    def get_description_string(self, log_data):
        desc = f"Epoch {self.current_epoch} |"
        for key, value in log_data.items():
            if "iou" in key:
                desc += f"{key}:{value*100:.2f} |"
            elif "steps" in key:
                desc += f"{key}:{value} |"
            else:
                desc += f"{key}:{value:.3e} |"
        return desc


    def training_epoch_end(self, outputs):


        log_data = self.compute_log_data(outputs, self.train_cm, prefix="train")

        os.makedirs(self.logger.log_dir, exist_ok=True)
        logs_file(os.path.join(self.logger.log_dir, "logs_train.csv"), self.current_epoch, log_data)

        if (self.global_step > 0) and (not self.config["interactive_log"]):
            desc = "Train "+ self.get_description_string(log_data)
            print(wblue(desc))


    def validation_epoch_end(self, outputs):
        
        if self.global_step > 0:

            log_data = self.compute_log_data(outputs, self.val_cm, prefix="val")

            os.makedirs(self.logger.log_dir, exist_ok=True)
            logs_file(os.path.join(self.logger.log_dir, "logs_val.csv"), self.current_epoch, log_data)

            if (not self.config["interactive_log"]):
                desc = "Val "+ self.get_description_string(log_data)
                print(wgreen(desc))



def get_savedir_name(config):
    
    savedir = f"{config['network']['backbone']}_{config['network']['decoder']}"
    if config["network"]['framework'] is not None:
        savedir += f"_{config['network']['framework']}"
    savedir += f"_{config['manifold_points']}_{config['non_manifold_points']}"
    savedir += f"_{config['train_split']}Split"
    savedir += f"_radius{config['network']['decoder_params']['radius']}"
    if ("desc" in config) and config["desc"]:
        savedir += f"_{config['desc']}"

    return savedir

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(config : DictConfig):

    config = OmegaConf.to_container(config)["cfg"]

    warnings.filterwarnings("ignore", category=UserWarning) 
    logging.getLogger().setLevel(config["logging"])

    logging.info("Getting the dataset and dataloader")
    DatasetClass = eval("datasets."+config["dataset_name"])
    train_transforms = get_transforms(config, train=True)
    test_transforms = get_transforms(config, train=False)

    # build the dataset
    train_dataset = DatasetClass(config["dataset_root"], 
                split=config["train_split"], 
                transform=train_transforms, 
                )
    val_dataset = DatasetClass(config["dataset_root"],
                split=config["val_split"], 
                transform=test_transforms, 
                )

    # build the data loaders
    train_loader = DataLoader(
            train_dataset,
            batch_size=config["training"]["batch_size"],
            shuffle=True,
            num_workers=config["threads"],
            follow_batch=["pos_non_manifold", "voxel_coords", "voxel_proj_yx"]
        )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["threads"],
        follow_batch=["pos_non_manifold", "voxel_coords", "voxel_proj_yx"]
    )

    logging.info("Creating trainer")

    savedir_root = get_savedir_name(config)
    savedir_root = os.path.join(config["save_dir"], "Pretraining", savedir_root)

    logging.info(f"Savedir_root {savedir_root}")
    
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    tb_logger = pl_loggers.TensorBoardLogger(save_dir=savedir_root)
    trainer = pl.Trainer(
            gpus= config["num_device"],
            check_val_every_n_epoch=config["training"]["val_interval"],
            logger=tb_logger,
            max_epochs=config["training"]["max_epochs"],
            callbacks=[
                CustomProgressBar(refresh_rate=int(config["interactive_log"])),
                lr_monitor
                ]
            )
    
    # save the config file
    logging.info(f"Saving at {trainer.logger.log_dir}")
    os.makedirs(trainer.logger.log_dir, exist_ok=True)
    yaml.dump(config, open(os.path.join(trainer.logger.log_dir, "config.yaml"), "w"), default_flow_style=False)

    model = LightningSelfSupervisedTrainer(config)
    trainer.fit(model, train_loader, val_loader, ckpt_path=config["resume"])


if __name__ == "__main__":
    main()