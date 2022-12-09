import logging


class Dupplicate(object):

    def __init__(self, item_list, prefix) -> None:
        
        logging.info(f"Transforms - Dupplicate {item_list} - {prefix}")
        self.item_list = item_list
        self.prefix = prefix

    def __call__(self, data):

        for item in self.item_list:
            data[self.prefix+item] = data[item].clone()
        
        return data
