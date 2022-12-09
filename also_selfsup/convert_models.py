import torch
import os
import argparse
import warnings
import logging

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning) 
    
    parser = argparse.ArgumentParser(description='Self supervised.')
    parser.add_argument('--downstream', action="store_true")
    parser.add_argument('--ckpt', '-c', type=str, required=True)
    opts = parser.parse_args()
    
    logging.getLogger().setLevel("INFO")

    logging.info("Loading the checkpoint")
    state_dict = torch.load(opts.ckpt, map_location="cpu")["state_dict"]

    ckpt_dir = os.path.dirname(opts.ckpt)
    ckpt_name = os.path.basename(opts.ckpt)

    if opts.downstream:
        logging.info("Filtering the state dict")
        # filter the state dict
        trained_dict = {}
        for k,v in state_dict.items():
            if k[:4] != "net.": # keep only the weights of the backbone
                continue
            trained_dict[k[4:]] = v
        
        logging.info("Saving the weights")
        torch.save( trained_dict, os.path.join(ckpt_dir, "trained_model_"+ckpt_name))
    else:

    
        logging.info("Filtering the state dict")
        # filter the state dict
        pretrained_dict = {}
        classifier_dict = {}
        for k,v in state_dict.items():
            if "backbone." not in k: # keep only the weights of the backbone
                continue
            if "classifier." in k: # do not keep the weights of the classifier
                classifier_dict[k.replace("backbone.", "")] = v
                continue
            # print("backbone", k)
            pretrained_dict[k.replace("backbone.", "")] = v

        logging.info("Saving the weights")
        torch.save( pretrained_dict, os.path.join(ckpt_dir, "pretrained_backbone_"+ckpt_name))
        torch.save( classifier_dict, os.path.join(ckpt_dir, "pretrained_classifier_"+ckpt_name))



    logging.info("Done")
