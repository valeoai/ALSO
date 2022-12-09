import torch
import logging
import numbers
import math
import random



class RandomRotate(object):
    def __init__(self, degrees, axis=0, item_list=["pos"]):
        if isinstance(degrees, numbers.Number):
            degrees = (-abs(degrees), abs(degrees))
        assert isinstance(degrees, (tuple, list)) and len(degrees) == 2
        
        logging.info(f"Transforms - Axis {axis} - {item_list}")
        self.degrees = degrees
        self.axis = axis
        self.item_list = item_list

    def __call__(self, data):
        degree = math.pi * random.uniform(*self.degrees) / 180.0
        sin, cos = math.sin(degree), math.cos(degree)

        if data.pos.size(-1) == 2:
            matrix = [[cos, sin], [-sin, cos]]
        else:
            if self.axis == 0:
                matrix = [[1, 0, 0], [0, cos, sin], [0, -sin, cos]]
            elif self.axis == 1:
                matrix = [[cos, 0, -sin], [0, 1, 0], [sin, 0, cos]]
            else:
                matrix = [[cos, sin, 0], [-sin, cos, 0], [0, 0, 1]]

        matrix = torch.tensor(matrix)

        for key, item in data:
            if key in self.item_list:
                if torch.is_tensor(item):
                    data[key] = torch.matmul(item, matrix.to(item.dtype).to(item.device))
                    if ("second_" + key) in data.keys:
                        data["second_" + key] = torch.matmul(data["second_" + key], matrix.to(item.dtype).to(item.device))

        return data
