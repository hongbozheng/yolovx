'''
Implementation of YOLOv3 Architecture
'''

import logging
import torch.nn as nn
from parse import parse_cfg
from model import create_model

logging.basicConfig(level=logging.DEBUG)

class Darknet(nn.Module):
    def __init__(self,cfg):
        super(Darknet,self).__init__()
        self.blocks = parse_cfg(cfg)
        self.net,self.model,self.cache_module_index = create_model(blocks=self.blocks)

    def get_blocks(self):
        return self.blocks

    def get_net(self):
        return self.net

    def get_model(self):
        return self.model

    def get_cache_module_index(self):
        return self.cache_module_index

    def forward(self,x):
       
        module_cache = {}

        for i in range(len(model)):
            module_type = model[i]['type']

            if module_type == 'convolutional' or module_type == 'upsample':
                x = self.model[i](x)
            
            if i in self.cache_module_index:
                    module_cache[i] = x

YOLOv3 = Darknet('../cfg/yolov3.cfg')

def search(blocks,layer_type):
    print('[Model]: {}'.format(layer_type))
    for i in range(len(blocks)):
        if blocks[i]['type'] == layer_type:
            print(i,blocks[i])

search(YOLOv3.get_blocks()[1:],'route')
