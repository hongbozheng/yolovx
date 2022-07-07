'''
Implementation of YOLOv3 Architecture
'''

import logging
import torch
import torch.nn as nn
from utils import parse_cfg
from model import create_model
import numpy as np
# logging.basicConfig(level=logging.DEBUG)

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
                # x = module_cache[i-1]
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
                '''
                input_dimension = self.net['height']
                anchor = self.blocks[i+1]['anchors']
                num_class = self.blocks[i+1]['classes']

                # TODO: Implement yolo layer (detection layer)
                x = prediction_transformation(prediction=x.data,input_dimension=input_dimension,anchor=anchor,
                                              num_class=num_class,CUDA=True)
                
                # don't understand
                if type(x) == int:
                    continue

                if not write:
                    detection = x
                    write = 1
                else:
                    detection = torch.cat(tensors=(detection,x),dim=1)
                '''
            if i in self.cache_module_index:
                    module_cache[i] = x
            
        return detection

'''
def get_test_input():
    img = cv2.imread("../dog-cycle-car.png")
    img = cv2.resize(img, (416,416))          #Resize to the input dimension
    img_ =  img[:,:,::-1].transpose((2,0,1))  # BGR -> RGB | H X W C -> C X H X W 
    img_ = img_[np.newaxis,:,:,:]/255.0       #Add a channel at 0 (for batch) | Normalise
    img_ = torch.from_numpy(img_).float()     #Convert to float
    img_ = Variable(img_)                     # Convert to Variable
    return img_
'''

'''
a = torch.tensor([[1,1,1],
                  [2,2,2],
                  [3,3,3]])
b = torch.tensor([[4,4,4],
                  [5,5,5],
                  [6,6,6]])
a = torch.cat((a,a),1)
print('a: {}'.format(a))
'''

'''
YOLOv3 = Darknet('../cfg/yolov3.cfg')
YOLOv3.load_weights('../weights/yolov3.weights')
inp = get_test_input()
pred = YOLOv3.forward(inp)
# print(pred)
print(pred[0][1].size())
print(pred[1][1].size())
print(pred[2][1].size())
# print(YOLOv3.get_blocks())
'''

'''
a = torch.tensor([[1,1,1],
                  [2,2,2],
                  [3,3,3]])
b = torch.tensor([[4,4,4],
                  [5,5,5],
                  [6,6,6]])
a = torch.cat((a,b),0)
print('a: {}'.format(a))
'''

def search(blocks,layer_type):
    print('[Model]: {}'.format(layer_type))
    for i in range(len(blocks)):
        if blocks[i]['type'] == layer_type:
            print(i,blocks[i])

# search(YOLOv3.get_blocks()[1:],'route')
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
