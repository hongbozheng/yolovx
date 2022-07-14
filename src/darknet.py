'''
Implementation of YOLOv3 Architecture
'''

from parse import parse_cfg
from model import create_model
import torch
import torch.nn as nn

class Darknet(nn.Module):
    def __init__(self,cfg,yolo_weights):
        super(Darknet,self).__init__()
        self.configuration = parse_cfg(cfg)
        self.net,self.model,self.cache_module_index = create_model(configuration=self.configuration,yolo_weights=yolo_weights)

    def get_configuration(self):
        return self.configuration

    def get_net(self):
        return self.net

    def get_model(self):
        return self.model

    def get_cache_module_index(self):
        return self.cache_module_index

    def forward(self,x):
        module_cache = {}
        detection = []
        write = False

        for i in range(len(self.model)):
            module_type = self.configuration[i+1]['type']

            if module_type == 'convolutional' or module_type == 'upsample':
                x = self.model[i](x)

            elif module_type == 'shortcut':
                try:
                    for layer in self.configuration[i+1]['from']:
                        x += module_cache[layer]
                except:
                    x += module_cache[self.configuration[i+1]['from']]
            
            elif module_type == 'route':
                try:
                    x = module_cache[self.configuration[i+1]['layers'][0]]
                except:
                    x = module_cache[self.configuration[i+1]['layers']]
                try:
                    for k in range(len(self.configuration[i+1]['layers'][1:])):
                        x = torch.cat(tensors=(x,module_cache[self.configuration[i+1]['layers'][k+1]]),dim=1)
                except:
                    pass

            elif module_type == 'yolo':
                detection.append((i,x))
           
            if i in self.cache_module_index:
                module_cache[i] = x
            
        return detection
