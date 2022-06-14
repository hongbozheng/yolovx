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
            block[key.rstrip()] = int(value)
            logging.debug('[try int]: {}={}'.format(key.rstrip(),block[key.rstrip()]))
            logging.debug('[Variable Type]: {}'.format(type(block[key.rstrip()])))
        except:
            try:
                block[key.rstrip()] = float(value)
                logging.debug('[try float]: {}={}'.format(key.rstrip(),block[key.rstrip()]))
                logging.debug('[Variable Type]: {}'.format(type(block[key.rstrip()])))
            except:
                block[key.rstrip()] = value.lstrip().split(',')
                try:
                    block[key.rstrip()] = [int(x) for x in block[key.rstrip()]]
                    logging.debug('[try int list]: {}={}'.format(key.rstrip(),block[key.rstrip()]))
                    logging.debug('[Variable Type list]: {}'.format(type(block[key.rstrip()][0])))
                except:
                    try:
                        block[key.rstrip()] = [float(x) for x in block[key.rstrip()]]
                        logging.debug('[try float list]: {}={}'.format(key.rstrip(),block[key.rstrip()]))
                        logging.debug('[Variable Type list]: {}'.format(type(block[key.rstrip()][0])))
                    except:
                        block[key.rstrip()] = value.lstrip()
                        logging.debug('[try str]: {}={}'.format(key.rstrip(),block[key.rstrip()]))
                        logging.debug('[Variable Type]: {}'.format(type(block[key.rstrip()])))

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
            logging.debug('[Layer Type]: ----- net -----')
            block['type'] = 'net'
            continue
        if line == '[convolutional]':
            append_block() 
            layer_type = LayerType.convolutional
            logging.debug('[Layer Type]: ----- convolutional -----')
            block['type'] = 'convolutional'
            continue
        if line == '[shortcut]':
            append_block()
            layer_type = LayerType.shortcut
            logging.debug('[Layer Type]: ----- shortcut -----')
            block['type'] = 'shortcut'
            continue
        if line == '[route]':
            append_block()
            layer_type = LayerType.route
            logging.debug('[Layer Type]: ----- route -----')
            block['type'] = 'route'
            continue
        if line == '[upsample]':
            append_block()
            layer_type = LayerType.upsample
            logging.debug('[Layer Type]: ----- upsample -----')
            block['type'] = 'upsample'
            continue
        if line == '[yolo]':
            append_block()
            layer_type = LayerType.yolo
            logging.debug('[Layer Type]: ----- yolo -----')
            block['type'] = 'yolo'
            continue

        key,value = line.split('=')

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
            if key.rstrip() == 'anchors':
                block['anchors'] = [(block['anchors'][i],block['anchors'][i+1]) 
                                    for i in range(0,len(block['anchors']),2)]
                logging.debug('[tuple]: {}'.format(block['anchors']))
                logging.debug('[Varibale Type]: {}'.format(type(block['anchors'][0])))
        else:
            logging.error('[ERROR]: INVALID LAYER TYPE!')
            assert False

    append_block()
    logging.debug('[blocks]: {}'.format(blocks))
    logging.debug('[len(blocks)]: {}'.format(len(blocks)))
    return blocks

parse_cfg('../cfg/yolov3.cfg')
