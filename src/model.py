'''
Create model of YOLOv3 architecture
'''

import torch.nn as nn
import numpy as np
import torch

def create_model(configuration, yolo_weights):
    
    net = configuration[0]
    model = nn.ModuleList()
    cache_module_index = []

    prev_filters = 3
    filters = 0
    filters_list = []
    
    file = open(yolo_weights,'rb')
    weights_info = np.fromfile(file=file,dtype=np.int32,count=5)
    weights = np.fromfile(file=file,dtype=np.float32)
    print('[YOLOv3 Weights INFO]: {}'.format(weights_info))
    print('[YOLOv3 Weights]:      {}'.format(weights))
    print('[YOLOv3 Weights LEN]:  {}'.format(len(weights)))

    ptr = 0
    weight_num = 0

    for layer_index,layer_config in enumerate(configuration[1:]):
        module = nn.Sequential()
        
        if layer_config['type'] == 'convolutional':
            try:
                batch_normalize = layer_config['batch_normalize']
                bias = False
            except:
                batch_normalize = 0
                bias = True
            
            filters     = layer_config['filters']
            kernel_size = layer_config['size']
            padding     = layer_config['pad']
            
            # information on darknet wiki
            if padding:
                padding = kernel_size//2
            else:
                padding = 0

            convolutional_layer = nn.Conv2d(in_channels=prev_filters,
                                            out_channels=filters,
                                            kernel_size=kernel_size,
                                            stride=layer_config['stride'],
                                            padding=padding,
                                            bias=bias)

            if batch_normalize:
                batch_norm = nn.BatchNorm2d(num_features=filters)

                weight_num = batch_norm.bias.numel()
                # print('weight_num: {}'.format(weight_num))

                # print('weights: {}'.format(weights[ptr:ptr+weight_num]))
                # or .view_as(bn.bias.data)
                batch_norm_bias = torch.from_numpy(weights[ptr:ptr+weight_num]).view(batch_norm.bias.data.size())
                # print('bn_bias: {}'.format(bn_bias))
                batch_norm.bias.data.copy_(batch_norm_bias)
                ptr += weight_num
                # print('bn.bias: {}'.format(bn.bias))

                batch_norm_weight = torch.from_numpy(weights[ptr:ptr+weight_num]).view(batch_norm.weight.data.size())
                batch_norm.weight.data.copy_(batch_norm_weight)
                ptr += weight_num

                batch_norm_running_mean = torch.from_numpy(weights[ptr:ptr+weight_num]).view(batch_norm.running_mean.data.size())
                batch_norm.running_mean.data.copy_(batch_norm_running_mean)
                ptr += weight_num

                batch_norm_running_var = torch.from_numpy(weights[ptr:ptr+weight_num]).view(batch_norm.running_var.data.size())
                batch_norm.running_var.data.copy_(batch_norm_running_var)
                ptr += weight_num
            
            else:
                weight_num = convolutional_layer.bias.numel()
                convolutional_layer_bias = torch.from_numpy(weights[ptr:ptr+weight_num]).view(convolutional_layer.bias.data.size())
                convolutional_layer.bias.data.copy_(convolutional_layer_bias)
                ptr += weight_num

            weight_num = convolutional_layer.weight.numel()
            convolutional_layer_weight = torch.from_numpy(weights[ptr:ptr+weight_num]).view(convolutional_layer.weight.data.size())
            convolutional_layer.weight.data.copy_(convolutional_layer_weight)
            ptr += weight_num

            module.add_module('conv2d_{}'.format(layer_index),convolutional_layer)
            
            if batch_normalize:
                module.add_module('batchnorm2d_{}'.format(layer_index),batch_norm)
            
            # may need to add more cases for other activation functions in the future
            if layer_config['activation'] == 'leaky':
                activation_function = nn.LeakyReLU(negative_slope=0.1,inplace=True)
                module.add_module('leakyrelu_{}'.format(layer_index),activation_function)

            # convolutional layer before yolo layer, activation function = linear
            # if we have linear layer, conv2d bias=T/F, linear bias=T/F
            # elif block['activation'] == 'linear':
            #     activation_function = nn.Linear(in_features=filters,
            #                                     out_features=filters,
            #                                     bias=bias)
            #     module.add_module('Linear_{}'.format(index),activation_function)

        elif layer_config['type'] == 'shortcut':
            # what is he doing with from_ in Github ???
            # EmptyLayer() class inherit from nn.Module, necessary?
            # cache_module_index.append(index-1)
            try:
                for i in range(len(layer_config['from'])):
                    layer_config['from'][i] += layer_index
                    cache_module_index.append(layer_config['from'][i])
            except:
                layer_config['from'] += layer_index
                cache_module_index.append(layer_config['from'])
            module.add_module('shortcut_{}'.format(layer_index),nn.Module())
        
        elif layer_config['type'] == 'yolo':
            # cache_module_index.append(layer_index)
            module.add_module('yolo_{}'.format(layer_index),nn.Module())

        elif layer_config['type'] == 'route':
            # EmptyLayer() class inherit from nn.Module, necessary?
            filters = 0
            try:
                for i in range(len(layer_config['layers'])):
                    if layer_config['layers'][i] < 0:
                        layer_config['layers'][i] += layer_index
                        filters += filters_list[layer_config['layers'][i]]
                        cache_module_index.append(layer_config['layers'][i])
                    else:
                        filters += filters_list[layer_config['layers'][i]]
                        cache_module_index.append(layer_config['layers'][i])
            except:
                if layer_config['layers'] < 0:
                    layer_config['layers'] += layer_index
                    filters = filters_list[layer_config['layers']]
                    cache_module_index.append(layer_config['layers'])
                else:
                    cache_module_index.append(layer_config['layers'])
                    filters = filters_list(layer_config['layers'])

            module.add_module('route_{}'.format(layer_index),nn.Module())

        elif layer_config['type'] == 'upsample':
            upsample = nn.Upsample(scale_factor=layer_config['stride'],mode='bilinear')
            module.add_module('upsample_{}'.format(layer_index),upsample)

        else:
            print('[ERROR]: Module type NOT FOUND; Check cfg file')
            assert False
    
        model.append(module)
        prev_filters = filters
        filters_list.append(filters)
  
    print(ptr)
    cache_module_index.sort(reverse=False)
    print('[INFO]: Finish creating model')
    print('[Net]:  {}'.format(net))
    # print('[Model]: {}'.format(model))
    print('[Cache Module Index]: {}'.format(cache_module_index))
    return net, model, cache_module_index

# Test
# blocks = parse_cfg('../cfg/yolov3.cfg')
# net,model,cache_module_index = create_model(blocks=blocks)
# print('[Model]: {}'.format(model))
# print('[Cache Module Index]: {}'.format(cache_module_index))
