import torch
import logging

class CreateInputs(object):

    def __init__(self, item_list):

        if isinstance(item_list, list):
            self.item_list = item_list
        elif isinstance(item_list, str):
            if item_list[0]=="[":
                item_list = item_list[1:]
            if item_list[-1]=="]":
                item_list = item_list[:-1]
            item_list = item_list.split(",")
            self.item_list = item_list
        logging.info(f"CreateInputs -- {item_list}")
    
    def __call__(self, data):
        
        features = []
        for key in self.item_list:
            features.append(data[key])

        data["x"] = torch.cat(features, dim=1) 
        return data