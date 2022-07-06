import logging
from enum import IntEnum
import torch
import numpy as np

epsilon = 1e-6

'''
$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ Parse YOLO cfg file $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
'''
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
    print('[INFO]: Finish parsing cfg file')
    print('[len(blocks)]: [Net]={} + [Layer]={}'.format(1,len(blocks)-1))
    return blocks

'''
$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ detection post processing $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
'''
def detection_postprocessing(detection, batch, input_dimension, anchors, num_class, CUDA=True):
    batch_size = detection.size(dim=0)
    grid_scale = detection.size(dim=2)
    grid_size = input_dimension//grid_scale
    num_anchors = len(anchors)
    bbox_attribute = 5+num_class
   
    # print(detection)
    # print(detection.size())

    detection = detection.view(batch_size,num_anchors*bbox_attribute,grid_scale*grid_scale).transpose(dim0=1,dim1=2).contiguous()
    detection = detection.view(batch_size,grid_scale*grid_scale*num_anchors,bbox_attribute)
   
    y_offset,x_offset = torch.FloatTensor(np.meshgrid(np.arange(grid_scale),np.arange(grid_scale)))
    xy_offset = torch.cat(tensors=(x_offset.view(-1,1),y_offset.view(-1,1)),dim=1).repeat(1,num_anchors).view(-1,2).unsqueeze(dim=0)
    # print(xy_offset.size())
    detection[:,:,:2] = (torch.sigmoid(detection[:,:,:2])+xy_offset)*grid_size
    detection[:,:,4:] = torch.sigmoid(detection[:,:,4:])
    anchors = torch.FloatTensor(anchors).repeat(grid_scale*grid_scale,1).unsqueeze(dim=0)
    # print(anchors)
    detection[:,:,2:4] = torch.exp(detection[:,:,2:4])*anchors
    
    # print(detection)
    # print(detection.size())

    '''
    # print('Original detection: ',detection)
    # print(detection.size())
    # detection[:,:,:,:2] = torch.sigmoid(detection[:,:,:,:2])
    # print(detection)
    detection = torch.transpose(detection,dim0=1,dim1=2)
    detection = torch.transpose(detection,dim0=2,dim1=3)
    # print(detection.size())
    batch_size = detection.size(dim=0)
    grid_scale = detection.size(dim=1)
    grid_size = input_dimension//grid_scale
    bbox_attribute = 5+num_class
    # anchors = [(anchor[0]/grid_size,anchor[1]/grid_size) for anchor in anchors]
    # print(detection)
    detection = detection.view(batch,grid_scale,grid_scale,-1,bbox_attribute)
    detection[:,:,:,:,:2] = torch.sigmoid(detection[:,:,:,:,:2])
    detection[:,:,:,:,4:] = torch.sigmoid(detection[:,:,:,:,4:])
    # print('After sigmoid: ',detection)
    # print(detection.size())
    
    x_offset, y_offset = torch.FloatTensor(np.meshgrid(np.arange(grid_scale),np.arange(grid_scale)))
    # x_offset = x_offset.view(grid_scale*grid_scale,-1)
    # y_offset = y_offset.view(grid_scale*grid_scale,-1)

    # print(x_offset)
    # print(y_offset)

    xy_offset = torch.cat(tensors=(y_offset.view(grid_scale*grid_scale,-1),x_offset.view(grid_scale*grid_scale,-1)),dim=1).view(grid_scale,grid_scale,-1).unsqueeze(dim=0).unsqueeze(dim=3).repeat(1,1,1,3,1)
    # xy_offset = torch.cat(tensors=(xy_offset,xy_offset,xy_offset),dim=3)
    # xy_offset = torch.cat(tensors=(xy_offset = torch.cat(tensors=(y_offset,x_offset),dim=1).view(grid_scale,grid_scale,-1).unsqueeze(dim=0).unsqueeze(dim=3),
    # xy_offset = torch.cat(tensors=(torch.cat(tensors=(y_offset,x_offset),dim=1).view(grid_scale,grid_scale,-1).unsqueeze(dim=0).unsqueeze(dim=3),torch.cat(tensors=(y_offset,x_offset),dim=1).view(grid_scale,grid_scale,-1).unsqueeze(dim=0).unsqueeze(dim=3),torch.cat(tensors=(y_offset,x_offset),dim=1).view(grid_scale,grid_scale,-1).unsqueeze(dim=0).unsqueeze(dim=3)),dim=3)

    # print('base',xy_offset)
    # print('base size',xy_offset.size())

    # xy_offset = xy_offset.repeat(1,1,1,3,1)
    # print('xy_offset',xy_offset)
    # print('size xy',xy_offset.size())


    detection[:,:,:,:,:2] = (detection[:,:,:,:,:2]+xy_offset)*grid_size
    # detection[:,:,:,:,:2] *= grid_size
    # detection[:,:,:,:,:2] *= grid_size
    # print(detection)
    # print(detection.size())
    
    # print(anchors)
    # print(len(anchors))
   
    anchors = torch.FloatTensor(anchors).unsqueeze(dim=0).unsqueeze(dim=0).unsqueeze(dim=0).repeat(1,grid_scale,grid_scale,1,1)
    # print(anchors)
    # print(anchors.size())
    
    detection[:,:,:,:,2:4] = torch.exp(detection[:,:,:,:,2:4])*anchors
    print(detection)
    print(detection.size())
    '''

    '''
    print(detection)
    print(detection.size())
    print(detection[0,0,0,0,4])
    print(detection[...,4:5])
    print(detection[...,4:5].size())
    print('-----',detection[detection[...,4]>=0.5])
    '''

    '''
    prediction = prediction.view(batch_size, grid_size*grid_size*3, bbox_attribute)
    print('pred over: {}'.format(prediction.size()))
    
    # i think [...,0] will also work
    prediction[:,:,0] = torch.sigmoid(prediction[:,:,0])
    prediction[:,:,1] = torch.sigmoid(prediction[:,:,1])
    prediction[:,:,4] = torch.sigmoid(prediction[:,:,4])

    grid_length = np.arange(grid_size)
    a,b = np.meshgrid(grid_length,grid_length)

    x_offset = torch.FloatTensor(a).view(-1,1)
    y_offset = torch.FloatTensor(b).view(-1,1)
    
    x_y_offset = torch.cat(tensors=(x_offset,y_offset),dim=1).repeat(1,3).unsqueeze(dim=0)

    prediction[:,:,:2] += x_y_offset

    anchor = torch.FloatTensor(anchor)
    anchor = anchor.repeat(grid_size*grid_size,1).unsqueeze(0)
    prediction[:,:,2:4] = torch.exp(prediction[:,:,2:4])*anchor

    prediction[:,:,5:5+num_class] = torch.sigmoid(prediction[:,:,5:5+num_class])
    prediction[:,:,:4] *= stride
    '''

    return detection

'''
$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ intersection over union $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
'''
def intersection_over_union(box_1, box_2, box_format='midpoint'):
    # TODO: need to remove ... since box_1 and box_2 are 1-d tensor
    if box_format == 'midpoint':
        # maybe need to change to box_1[:,0] ???
        box1_x1 = box_1[0] - box_1[2]/2
        box1_y1 = box_1[1] - box_1[3]/2
        box1_x2 = box_1[0] + box_1[2]/2
        box1_y2 = box_1[1] + box_1[3]/2
        box2_x1 = box_2[0] - box_2[2]/2
        box2_y1 = box_2[1] - box_2[3]/2
        box2_x2 = box_2[0] + box_2[2]/2
        box2_y2 = box_2[1] + box_2[3]/2

    if box_format == 'corner':
        box1_x1 = box_1[0]
        box1_y1 = box_1[1]
        box1_x2 = box_1[2]
        box1_y2 = box_1[3]
        box2_x1 = box_2[0]
        box2_y1 = box_2[1]
        box2_x2 = box_2[2]
        box2_y2 = box_2[3]

    x1 = torch.max(box1_x1,box2_x1)
    y1 = torch.max(box1_y1,box2_y1)
    x2 = torch.min(box1_x2,box2_x2)
    y2 = torch.min(box1_y2,box2_y2)
    
    # maybe can use cuda over here
    intersection = (x2-x1).clamp(min=0)*(y2-y1).clamp(min=0)

    box1_area = abs((box1_x2-box1_x1)*(box1_y2-box1_y1))
    box2_area = abs((box2_x2-box2_x1)*(box2_y2-box2_y1))

    return intersection/(box1_area+box2_area+epsilon)

'''
$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ non max suppression $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
'''
def non_max_suppression(final_detection, iou_threshold, box_format='midpoint'):
    # bbox [[X,Y,W,H,OS(Highest),CI,CS],
    #                  ...             ,
    #       [X,Y,W,H,OS(Lowest) ,CI,CS]]
    # probably make class_prediction a list of tensor is easier ????? idk
    final_detection = final_detection[torch.sort(input=final_detection[:,:,4],descending=True)[1]]
    print(final_detection.size())

    for i in range(final_detection.size(dim=0)):
        for k in range(final_detection[i+1:].size(dim=0)):
            if intersection_over_union(final_detection[i],final_detection[k]) > iou_threshold:
                final_detection[k]*=0
    
    final_detection[final_detection[:,0] != 0]

    return final_detection 
    # need this function ?????
    # yes indeed

'''
$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ get evaluation box $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
'''
def get_evaluation_box(yolo_detection, obj_score_threshold, num_class, iou_threshold=0.5, box_format='midpoint'):
    yolo_detection *= (yolo_detection[:,:,4] >= obj_score_threshold).float().unsqueeze(dim=2)

    print(yolo_detection)
    print(yolo_detection.size())
    
    if torch.nonzero(yolo_detection).numel() == 0:
        return 0
    
    for i in range(yolo_detection.size(dim=0)):
        image_prediction = yolo_detection[i]
        image_prediction = image_prediction[image_prediction[:,4]!=0]

    print('image_prediction',image_prediction)

    '''
    write = False
    prediction *= (prediction[:,:,4] >= obj_score_threshold).float().unsqueeze(dim=2)
    
    # i think this is better than his way
    # if torch.nonzero(prediction).numel() == 0:
    #    return 0

    for i in range(prediction.size(dim=0)):
        image_prediction = prediction[i]
        # nonzero_index = torch.nonzero(image_prediction[:,4])
        # image_prediction = image_prediction[torch.nonzero(image_prediction[:,4]).squeeze(),:].view(-1,5+num_class)
        # isn't this easier ?????
        image_prediction = image_prediction[image_prediction[:,4] != 0]
        if image_prediction.numel() == 0:
            continue
        highest_class_score, class_index = torch.max(image_prediction[:,5:5+num_class],dim=1)
        # maybe image_prediction[...,:5] also work ?????
        image_prediction = torch.cat(tensors=(image_prediction[:,:5],
                                              # why he used .float() for class index ?????
                                              class_index.float().unsqueeze(dim=1),
                                              highest_class_score.float().unsqueeze(dim=1)),dim=1)

        # i don't understand why needs try except here ?????
        image_class = torch.unique(image_prediction[:,-2])
        for c in image_class:
            # class_prediction = image_prediction*(image_prediction[:,-2]==c).float().unsqueeze(dim=1)
            class_prediction = image_prediction[image_prediction[:,-2] == c]
            # nonzero_index = torch.nonzero(class_prediction[:,-1]).squeeze()
            # class_prediction = class_prediction[torch.nonzero(class_prediction[:,-1]).squeeze()].view(-1,7)
            # isn't this easier ?????
            # class_prediction = class_prediction[class_prediction[:,-1] != 0]
            # obj_score_sort_index = torch.sort(input=class_prediction[:,4],descending=True)[1]
            # class_prediction = class_prediction[torch.sort(input=class_prediction[:,4],descending=True)[1]]

            if class_prediction.size(dim=0) > 1:
                non_max_suppression(class_prediction=class_prediction,iou_threshold=iou_threshold,box_format=box_format)
            
            batch_index = class_prediction.new(class_prediction.size(dim=0),1).fill_(i)

            if not write:
                final_prediction = torch.cat(tensors=(batch_index,class_prediction),dim=1)
                write = True
            else:
                class_prediction = torch.cat(tensors=(batch_index,class_prediction),dim=1)
                final_prediction = torch.cat(tensors=(final_prediction,class_prediction),dim=0)
    '''
    return final_detection

'''
a = torch.tensor([[[[0,0],
                    [1,1]],
                   [[2,2],
                    [3,3]],
                   [[4,4],
                    [5,5]],
                   [[6,6],
                    [7,7]],
                   [[8,8],
                    [9,9]],
                   [[1,1],
                    [0,0]]]])
print('a',a)
print('size a',a.size())

a = torch.transpose(a,1,2)
a = torch.transpose(a,2,3)
print(a)
print(a.size())
a = a.view(1,2,2,-1,3)
print(a)
print(a.size())
'''

'''
a = torch.tensor([[[1,2,3,4,5,6],
                   [1,2,3,4,5,6],
                   [1,2,3,4,5,6]],
                  [[6,7,8,9,0,1],
                   [6,7,8,9,0,1],
                   [6,7,8,9,0,1]],
                  [[1,3,5,7,9,0],
                   [1,3,5,7,9,0],
                   [1,3,5,7,9,0]]])
print('a',a)
print('a size',a.size())
a = a.view(3,3,-1,3)
print(a)
print(a.size())
'''
'''
a = torch.tensor([[[1,1,1,5,1],
                   [1,1,1,1,2],
                   [3,3,5,3,3]],
                  [[4,4,4,4,4],
                   [5,5,5,5,5],
                   [6,6,6,6,6]]],dtype=float)
print('a[i]: {}'.format(a[0]))
print('size a: {}'.format(a.size()))
# a = a[a[:,:,4] == 0]
# print('--: {}'.format(a))
mask = (a[:,:,4] <= 2).float().unsqueeze(2)
print('mask: {}'.format(mask))
print('after mask: {}'.format(a[0]*mask))
print('torch nonzero mask: {}'.format(torch.nonzero(mask)))
print('check empty: {}'.format(torch.nonzero(mask).numel()==0))
a *= (a[:,:,4] <= 2).float().unsqueeze(2)
print('cleaned up prediction: {}'.format(a))

c = a[0]
print('c: {}'.format(c))
nonzero_i = torch.nonzero(c[:,4])
c = c[torch.nonzero(c[:,4]).squeeze(),:].view(-1,5)
print('c: {}'.format(c))
hcs, ci = torch.max(c[:,2:5],dim=1)
print('class_index: {}'.format(ci))
print('highest_class_score: {}'.format(hcs))
c = torch.cat(tensors=(c[:,:2],ci.int().unsqueeze(1),hcs.float().unsqueeze(1)),dim=1)
print('cleaned up image_pred: {}'.format(c))

print('unique: {}'.format(torch.unique(c[:,-2])))

box_a = a.new(a.shape)
print('box_a: {}'.format(box_a))
box_a[0,0,0] = 10
print('a: {}'.format(a))
print('box_a: {}'.format(box_a))

print('-----------------')

b = torch.tensor([[1,2,3,4,5],
                  [6,7,8,9,0]])
max_conf, max_conf_score = torch.max(b[:,1:5],1)
print('b: {}'.format(b))
print('max_conf: {}'.format(max_conf))
print('max_conf_score: {}'.format(max_conf_score.unsqueeze(dim=1)))

d = torch.tensor([[1,4,7,2,3],
                  [1,2,5,2,7],
                  [1,6,3,2,4]])
e = d
f = d
g = d
h = d
sort = torch.sort(d[:,4])[1]
print('sort: {}'.format(sort))
d = d[torch.sort(d[:,4])[1]]
print('sorted: {}'.format(d))
# e = sorted(e, key=lambda x:x[4], reverse=True)
# print('sorted e: {}'.format(e))
e = e[e[:,4] < 6]
print('e: {}'.format(e))
print('size: {}'.format(e.size()))
for i in range(f.size(dim=0)):
    if f[i][2] == 5:
        f[i] *= 0

print('f: {}'.format(f))

batch_ind = g.new(g.size(0),1).fill_(0)
o = torch.cat((batch_ind,g),1)
print('o: {}'.format(o))

print('h: {}'.format(h))
h = h[0]
for i in range(h.size(0)):
    for k in range(h[i+1:].size(0)):
        if h[i] > h[k]:
            print('hello')
print('h un: {}'.format(h[0].unsqueeze(0)))

pred = torch.tensor([[[1,1,1,1,1],
                      [2,2,2,2,2],
                      [3,3,3,3,3],
                      [4,4,4,4,4],
                      [5,5,5,5,5]]])

input_dim = 608
print('dim pred: {}'.format(pred.size()))
print('transpose: {}'.format(torch.transpose(pred,1,2).contiguous()))
bs = pred.size(0)
print('bs: {}'.format(bs))
stride = input_dim//pred.size(2)
print('stride: {}'.format(stride))
grid_size = input_dim//stride
print('grid_size: {}'.format(grid_size))
bbox_attr = 3
num_anchor = 9

pred = pred.view(bs,bbox_attr*9,grid_size**2)
print('pred: {}'.format(pred))
'''
