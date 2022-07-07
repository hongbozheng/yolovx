'''
Create model of YOLOv3 architecture
'''

import torch.nn as nn
# Test
# from parse import parse_cfg

def create_model(configuration):
    
    net = configuration[0]
    model = nn.ModuleList()
    cache_module_index = []

    prev_filters = 3
    filters = 0
    filters_list = []
    index = 0

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
            module.add_module('conv2d_{}'.format(index),convolutional_layer)

            if batch_normalize:
                batch_norm = nn.BatchNorm2d(num_features=filters)
                module.add_module('batchnorm2d_{}'.format(index),batch_norm)

            # may need to add more cases for other activation functions in the future
            if layer_config['activation'] == 'leaky':
                activation_function = nn.LeakyReLU(negative_slope=0.1,inplace=True)
                module.add_module('leakyrelu_{}'.format(index),activation_function)

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
                    layer_config['from'][i] += index
                    cache_module_index.append(layer_config['from'][i])
            except:
                layer_config['from'] += index
                cache_module_index.append(layer_config['from'])
            module.add_module('shortcut_{}'.format(index),nn.Module())
        
        elif layer_config['type'] == 'yolo':
            # cache_module_index.append(layer_index)
            module.add_module('yolo_{}'.format(index),nn.Module())

        elif layer_config['type'] == 'route':
            # EmptyLayer() class inherit from nn.Module, necessary?
            filters = 0
            try:
                for i in range(len(layer_config['layers'])):
                    if layer_config['layers'][i] < 0:
                        layer_config['layers'][i] += index
                        filters += filters_list[layer_config['layers'][i]]
                        cache_module_index.append(layer_config['layers'][i])
                    else:
                        filters += filters_list[layer_config['layers'][i]]
                        cache_module_index.append(layer_config['layers'][i])
            except:
                if layer_config['layers'] < 0:
                    layer_config['layers'] += index
                    filters = filters_list[layer_config['layers']]
                    cache_module_index.append(layer_config['layers'])
                else:
                    cache_module_index.append(layer_config['layers'])
                    filters = filters_list(layer_config['layers'])

            module.add_module('route_{}'.format(index),nn.Module())

        elif layer_config['type'] == 'upsample':
            upsample = nn.Upsample(scale_factor=layer_config['stride'],mode='bilinear')
            module.add_module('upsample_{}'.format(index),upsample)

        else:
            print('[ERROR]: Module type NOT FOUND; Check cfg file')
            assert False
    
        model.append(module)
        prev_filters = filters
        filters_list.append(filters)
        index+=1
    
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
