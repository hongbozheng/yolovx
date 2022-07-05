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
        detection = []
        write = False

        for i in range(len(self.model)):
            
            module_type = self.blocks[i+1]['type']

            if module_type == 'convolutional' or module_type == 'upsample':
                x = self.model[i](x)

            elif module_type == 'shortcut':
                # x = module_cache[i-1]
                try:
                    for layer in self.blocks[i+1]['from']:
                        x += module_cache[layer]
                except:
                    x += module_cache[self.blocks[i+1]['from']]
            
            elif module_type == 'route':
                try:
                    x = module_cache[self.blocks[i+1]['layers'][0]]
                except:
                    x = module_cache[self.blocks[i+1]['layers']]
                try:
                    for k in range(len(self.blocks[i+1]['layers'][1:])):
                        x = torch.cat(tensors=(x,module_cache[self.blocks[i+1]['layers'][k+1]]),dim=1)
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

    def load_weights(self,yolo_weights):
        file = open(yolo_weights,'rb')
        weights_info = np.fromfile(file=file,dtype=np.int32,count=5)
        weights = np.fromfile(file=file,dtype=np.float32)
        print('[weights_info]:       {}'.format(weights_info))
        print('[pretrained weights]: {}'.format(weights))
        print('[len(weights)]: {}'.format(len(weights)))
        
        ptr = 0
        weight_num = 0

        for i in range(0,len(self.model)):
            if self.blocks[i+1]['type'] == 'convolutional':
                try:
                    batch_normalize = self.blocks[i+1]['batch_normalize']
                except:
                    batch_normalize = 0
                conv = self.model[i][0]

                if batch_normalize:
                    bn = self.model[i][1]
                    weight_num = bn.bias.numel()
                    # print('weight_num: {}'.format(weight_num))

                    # print('weights: {}'.format(weights[ptr:ptr+weight_num]))
                    # or .view_as(bn.bias.data)
                    bn_bias = torch.from_numpy(weights[ptr:ptr+weight_num])\
                                   .view(bn.bias.data.size())
                    # print('bn_bias: {}'.format(bn_bias))
                    bn.bias.data.copy_(bn_bias)
                    ptr += weight_num
                    # print('bn.bias: {}'.format(bn.bias))

                    bn_weight = torch.from_numpy(weights[ptr:ptr+weight_num])\
                                     .view(bn.weight.data.size())
                    bn.weight.data.copy_(bn.weight)
                    ptr += weight_num

                    bn_running_mean = torch.from_numpy(weights[ptr:ptr+weight_num])\
                                           .view(bn.running_mean.data.size())
                    bn.running_mean.data.copy_(bn_running_mean)
                    ptr += weight_num

                    bn_running_var = torch.from_numpy(weights[ptr:ptr+weight_num])\
                                          .view(bn.running_var.data.size())
                    bn.running_var.data.copy_(bn_running_var)
                    ptr += weight_num

                else:
                    weight_num = conv.bias.numel()
                    conv_bias = torch.from_numpy(weights[ptr:ptr+weight_num])\
                                     .view(conv.bias.data.size())
                    conv.bias.data.copy_(conv_bias)
                    ptr += weight_num

                weight_num = conv.weight.numel()
                conv_weight = torch.from_numpy(weights[ptr:ptr+weight_num])\
                                   .view(conv.weight.data.size())
                conv.weight.data.copy_(conv_weight)
                ptr += weight_num

        print('[ptr]:          {}'.format(ptr))
        if ptr == len(weights):
            print('[INFO]: {} successfully loaded!'.format(yolo_weights[11:]))
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
