import torch

class Scaling(object):

    def __init__(self, scale, item_list=["pos"]):
        self.scale = scale
        self.item_list = item_list

    def __call__(self, data):

        for key in self.item_list:
            if key in data.keys:
                data[key] = data[key] * self.scale

        return data