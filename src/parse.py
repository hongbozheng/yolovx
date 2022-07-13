import logging

logging.basicConfig(level=logging.INFO)

'''
$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ Parse YOLO cfg file $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
'''
def parse_cfg(cfg):
    
    file = open(cfg,'r')
    lines = file.read().split('\n')
    lines = [x for x in lines if len(x) > 0]
    lines = [x for x in lines if x[0] != '#']
    
    layer_config = {}
    configuration = []
    
    logging.debug('[cfg lines]: {}'.format(lines))

    for line in lines:
        if line[0] == '[':
            if len(layer_config) != 0:
                configuration.append(layer_config)
                layer_config = {}
            layer_config['type'] = line[1:-1]
        else:
            key,value = line.split('=')
            key = key.rstrip()
        
            try:
                layer_config[key] = int(value)
                logging.debug('[try int]  : {}={}'.format(key,layer_config[key]))
                logging.debug('[Var Type] : {}'.format(type(layer_config[key])))
            except:
                try:
                    layer_config[key] = float(value)
                    logging.debug('[try float]: {}={}'.format(key,layer_config[key]))
                    logging.debug('[Var Type] : {}'.format(type(layer_config[key])))
                except:
                    layer_config[key] = value.lstrip().split(',')
                    try:
                        layer_config[key] = [int(x) for x in layer_config[key]]
                        logging.debug('[try int list] : {}={}'.format(key,layer_config[key]))
                        logging.debug('[Var Type list]: {}'.format(type(layer_config[key][0])))
                    except:
                        try:
                            layer_config[key] = [float(x) for x in layer_config[key]]
                            logging.debug('[try float list]: {}={}'.format(key,layer_config[key]))
                            logging.debug('[Var Type list] : {}'.format(type(layer_config[key][0])))
                        except:
                            layer_config[key] = value.lstrip()
                            logging.debug('[try str]  : {}={}'.format(key,layer_config[key]))
                            logging.debug('[Var Type] : {}'.format(type(layer_config[key])))
        
            if key == 'anchors':
                layer_config['anchors'] = [(layer_config['anchors'][i],layer_config['anchors'][i+1]) 
                                    for i in range(0,len(layer_config['anchors']),2)]
                logging.debug('[tuple]    : {}'.format(layer_config['anchors']))
                logging.debug('[Var Type] : {}'.format(type(layer_config['anchors'][0])))

    configuration.append(layer_config)
    logging.debug('[blocks]: {}'.format(layer_config))
    print('[INFO]:   Finish parsing cfg file')
    print('[CONFIG]: [Net]={} + [Layer]={}'.format(1,len(configuration)-1))
    return configuration
