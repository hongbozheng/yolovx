'''
Implementation of YOLOv3 architecture
'''

from parse import *
import torch
import torch.nn as nn

def create_model(blocks):
    model = nn.ModuleList()
    prev_filters = 3
    index = 0
    
    for block in blocks:
        module = nn.Sequential()

        if block['type'] == 'net':
            continue
        if block['type'] == 'convolutional':
            try:
                batch_normalize = block['batch_normalize']
                bias = False
            except:
                batch_normalize = 0
                bias = True
            filters = int(block['filters'])
            kernel_size = int(block['size'])
            stride = int(block['stride'])
            padding = int(block['pad'])
            activation = block['activation']
            
            # information on darknet wiki
            if padding:
                padding = kernel_size//2
            else:
                padding = 0

            # why in_channels = 3 ???
            convolutional_layer = nn.Conv2d(in_channels=prev_filters,out_channels=filters,
                                            kernel_size=kernel_size,stride=stride,
                                            padding=padding,bias=bias)
            module.add_module('Conv2d_{}'.format(index),convolutional_layer)

            if batch_normalize:
                batch_norm = nn.BatchNorm2d(filters)
                module.add_module('BatchNorm2d_{}'.format(index),batch_norm)

            if activation == 'leaky':
                activation_function = nn.LeakyReLU(0.1,inplace=True)
                module.add_module('LeakyReLU_{}'.format(index),activation_function)

        elif block['type'] == 'shortcut':
            from_ = int(block['from'])
            # module.add_module('short_cut{}'.format(index),nn.Module)
            continue
        elif block['type'] == 'yolo':
            continue

        elif block['type'] == 'route':
            continue

        elif block['type'] == 'upsample':
            stride = int(block['stride'])
            upsample = nn.Upsample(scale_factor=2,mode='nearest')
            module.add_module('upsample_{}'.format(index),upsample)
    
        model.append(module)
        prev_filters = filters
        index+=1

    return model

blocks = parse_cfg('../cfg/yolov3.cfg')
print(create_model(blocks=blocks))
