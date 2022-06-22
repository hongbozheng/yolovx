'''
Parse YOLO cfg file
'''

import logging
from enum import IntEnum

logging.basicConfig(level=logging.INFO)

class LayerType(IntEnum):
    net           = 0
    convolutional = 1
    shortcut      = 2
    route         = 3
    upsample      = 4
    yolo          = 5

def parse_cfg(cfg):

    def append_block():
        nonlocal block
        if len(block) != 0:
            blocks.append(block)
            block = {}

    def parse_cfg_parameter():
        try:
            block[key] = int(value)
            logging.debug('[try int]  : {}={}'.format(key,block[key]))
            logging.debug('[Var Type] : {}'.format(type(block[key])))
        except:
            try:
                block[key] = float(value)
                logging.debug('[try float]: {}={}'.format(key,block[key]))
                logging.debug('[Var Type] : {}'.format(type(block[key])))
            except:
                block[key] = value.lstrip().split(',')
                try:
                    block[key] = [int(x) for x in block[key]]
                    logging.debug('[try int list] : {}={}'.format(key,block[key]))
                    logging.debug('[Var Type list]: {}'.format(type(block[key][0])))
                except:
                    try:
                        block[key] = [float(x) for x in block[key]]
                        logging.debug('[try float list]: {}={}'.format(key,block[key]))
                        logging.debug('[Var Type list] : {}'.format(type(block[key][0])))
                    except:
                        block[key] = value.lstrip()
                        logging.debug('[try str]  : {}={}'.format(key,block[key]))
                        logging.debug('[Var Type] : {}'.format(type(block[key])))

    file = open(cfg,'r')
    lines = file.read().split('\n')
    lines = [x for x in lines if len(x) > 0]
    lines = [x for x in lines if x[0] != '#']
    
    block = {}
    blocks = []

    logging.debug('[cfg lines]: {}'.format(lines))

    layer_type = -1

    for line in lines:
        if line == '[net]':
            layer_type = LayerType.net
            logging.debug('[LayerType]: ----- net -----')
            block['type'] = 'net'
            continue
        if line == '[convolutional]':
            append_block() 
            layer_type = LayerType.convolutional
            logging.debug('[LayerType]: ----- convolutional -----')
            block['type'] = 'convolutional'
            continue
        if line == '[shortcut]':
            append_block()
            layer_type = LayerType.shortcut
            logging.debug('[LayerType]: ----- shortcut -----')
            block['type'] = 'shortcut'
            continue
        if line == '[route]':
            append_block()
            layer_type = LayerType.route
            logging.debug('[LayerType]: ----- route -----')
            block['type'] = 'route'
            continue
        if line == '[upsample]':
            append_block()
            layer_type = LayerType.upsample
            logging.debug('[LayerType]: ----- upsample -----')
            block['type'] = 'upsample'
            continue
        if line == '[yolo]':
            append_block()
            layer_type = LayerType.yolo
            logging.debug('[LayerType]: ----- yolo -----')
            block['type'] = 'yolo'
            continue

        key,value = line.split('=')
        key = key.rstrip()

        if layer_type == LayerType.net:    
            parse_cfg_parameter()
        elif layer_type == LayerType.convolutional:
            parse_cfg_parameter()
        elif layer_type == LayerType.shortcut:
            parse_cfg_parameter()
        elif layer_type == LayerType.route:
            parse_cfg_parameter()
        elif layer_type == LayerType.upsample:
            parse_cfg_parameter()
        elif layer_type == LayerType.yolo:
            parse_cfg_parameter()
            if key == 'anchors':
                block['anchors'] = [(block['anchors'][i],block['anchors'][i+1]) 
                                    for i in range(0,len(block['anchors']),2)]
                logging.debug('[tuple]    : {}'.format(block['anchors']))
                logging.debug('[Var Type] : {}'.format(type(block['anchors'][0])))
        else:
            logging.error('[ERROR]: INVALID LAYER TYPE!')
            assert False

    append_block()
    logging.debug('[blocks]: {}'.format(blocks))
    print('[INFO]       : Finish parsing cfg file')
    print('[len(blocks)]: [Net]={} + [Layer]={}'.format(1,len(blocks)-1))
    return blocks

# Test
# parse_cfg('../cfg/yolov3.cfg')
