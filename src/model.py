'''
Create model of YOLOv3 architecture
'''

import torch.nn as nn
# Test
# from parse import parse_cfg

def create_model(blocks):
    
    net = blocks[0]
    model = nn.ModuleList()
    cache_module_index = []

    prev_filters = 3
    filters = 0
    filters_list = []
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

            filters     = block['filters']
            kernel_size = block['size']
            padding     = block['pad']
            
            # information on darknet wiki
            if padding:
                padding = kernel_size//2
            else:
                padding = 0

            convolutional_layer = nn.Conv2d(in_channels=prev_filters,
                                            out_channels=filters,
                                            kernel_size=kernel_size,
                                            stride=block['stride'],
                                            padding=padding,
                                            bias=bias)
            module.add_module('Conv2d_{}'.format(index),convolutional_layer)

            if batch_normalize:
                batch_norm = nn.BatchNorm2d(num_features=filters)
                module.add_module('BatchNorm2d_{}'.format(index),batch_norm)

            # may need to add more cases for other activation functions in the future
            if block['activation'] == 'leaky':
                activation_function = nn.LeakyReLU(negative_slope=0.1,inplace=True)
                module.add_module('LeakyReLU_{}'.format(index),activation_function)

            # convolutional layer before yolo layer, activation function = linear
            # if we have linear layer, conv2d bias=T/F, linear bias=T/F
            # elif block['activation'] == 'linear':
            #     activation_function = nn.Linear(in_features=filters,
            #                                     out_features=filters,
            #                                     bias=bias)
            #     module.add_module('Linear_{}'.format(index),activation_function)

        elif block['type'] == 'shortcut':
            # what is he doing with from_ in Github ???
            # EmptyLayer() class inherit from nn.Module, necessary?
            cache_module_index.append(index-1)
            try:
                for i in range(len(block['from'])):
                    block['from'][i] += index
                    cache_module_index.append(block['from'][i])
            except:
                block['from'] += index
                cache_module_index.append(block['from'])
            module.add_module('short_cut_{}'.format(index),nn.Module())
        
        elif block['type'] == 'yolo':
            module.add_module('yolo_{}'.format(index),nn.Module())

        elif block['type'] == 'route':
            # EmptyLayer() class inherit from nn.Module, necessary?
            filters = 0
            try:
                for i in range(len(block['layers'])):
                    if block['layers'][i] < 0:
                        block['layers'][i] += index
                        filters += filters_list[block['layers'][i]]
                        cache_module_index.append(block['layers'][i])
                    else:
                        filters += filters_list[block['layers'][i]]
                        cache_module_index.append(block['layers'][i])
            except:
                if block['layers'] < 0:
                    block['layers'] += index
                    filters = filters_list[block['layers']]
                    cache_module_index.append(block['layers'])
                else:
                    cache_module_index.append(block['layers'])
                    filters = filters_list(block['layers'])

            module.add_module('route_{}'.format(index),nn.Module())

        elif block['type'] == 'upsample':
            upsample = nn.Upsample(scale_factor=block['stride'],mode='nearest')
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
