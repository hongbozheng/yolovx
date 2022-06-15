'''
Implementation of YOLOv3 Architecture
'''

import logging
import torch
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

            elif module_type == 'shortcut':
                x = module_cache[i-1]
                try:
                    for layer in block['from']:
                        x += module_cache[layer]
                except:
                    x += module_cache[block['from']]
            
            elif module_type == 'route':
                x = module_cache[block['layers'][0]]
                try:
                    for k in range(len(block['layers'][1:])):
                        x = torch.cat(tuple=(x,block['layers'][i]),dim=0)
                except:
                    pass

            if i in self.cache_module_index:
                    module_cache[i] = x

YOLOv3 = Darknet('../cfg/yolov3.cfg')

def search(blocks,layer_type):
    print('[Model]: {}'.format(layer_type))
    for i in range(len(blocks)):
        if blocks[i]['type'] == layer_type:
            print(i,blocks[i])

search(YOLOv3.get_blocks()[1:],'route')
'''
a = torch.tensor([[[1,1,1],
                   [2,2,2],
                   [3,3,3]],
                  [[4,4,4],
                   [5,5,5],
                   [6,6,6]]])
b = torch.tensor([[[7,7,7],
                   [8,8,8],
                   [9,9,9]],
                  [[1,1,1],
                   [2,2,2],
                   [3,3,3]]])
print('a: {}'.format(a))
print('dim a: {}'.format(a.size()))
print('b: {}'.format(b))
print('dim b: {}'.format(b.size()))
c = torch.cat((a,b),dim=0)
print('c: {}'.format(c))
print('dim c: {}'.format(c.size()))
d = torch.zeros([5,5,3])
print(d)
print(d.size())
'''
