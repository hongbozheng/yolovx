import torch

epsilon = 1e-6

def intersection_over_union(box_1, box_2, box_format='midpoint'):
    # TODO: need to remove ... since box_1 and box_2 are 1-d tensor
    if box_format == 'midpoint':
        # maybe need to change to box_1[:,0] ???
        box1_x1 = box_1[...,0:1] - box_1[...,2:3]/2
        box1_y1 = box_1[...,1:2] - box_1[...,3:4]/2
        box1_x2 = box_1[...,0:1] + box_1[...,2:3]/2
        box1_y2 = box_1[...,1:2] + box_1[...,3:4]/2
        box2_x1 = box_2[...,0:1] - box_2[...,2:3]/2
        box2_y1 = box_2[...,1:2] - box_2[...,3:4]/2
        box2_x2 = box_2[...,0:1] + box_2[...,2:3]/2
        box2_y2 = box_2[...,1:2] + box_2[...,3:4]/2

    if box_format == 'corner':
        box1_x1 = box_1[...,0:1]
        box1_y1 = box_1[...,1:2]
        box1_x2 = box_1[...,2:3]
        box1_y2 = box_1[...,3:4]
        box2_x1 = box_2[...,0:1]
        box2_y1 = box_2[...,1:2]
        box2_x2 = box_2[...,2:3]
        box2_y2 = box_2[...,3:4]

    x1 = torch.max(box1_x1,box2_x1)
    y1 = torch.max(box1_y1,box2_y1)
    x2 = torch.min(box1_x2,box2_x2)
    y2 = torch.min(box1_y2,box2_y2)
    
    # maybe can use cuda over here
    intersection = (x2-x1).clamp(min=0)*(y2-y1).clamp(min=0)

    box1_area = abs((box1_x2-box1_x1)*(box1_y2-box1_y1))
    box2_area = abs((box2_x2-box2_x1)*(box2_y2-box2_y1))

    return intersection/(box1_area+box2_area+epsilon)

def non_max_suppression(class_prediction, iou_threshold, box_format='midpoint'):
    # bbox [[X,Y,W,H,OS(Highest),CI,CS],
    #                  ...             ,
    #       [X,Y,W,H,OS(Lowest) ,CI,CS]]
    # probably make class_prediction a list of tensor is easier ????? idk
    class_prediction = class_prediction[torch.sort(input=class_prediction[:,4],descending=True)[1]]
    
    for i in range(class_prediction.size(dim=0)):
        if intersection_over_union(class_prediction[i],class_prediction[i+1]) > iou_threshold:
            class_prediction[i+1]*=0

    class_prediction[class_prediction[:,0] != 0]

    return class_prediction 
    # need this function ?????
    # yes indeed

def get_evaluation_box(prediction, obj_score_threshold, num_class, NMS=True, iou_threshold=0.5, box_format='midpoint'):
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

            if class_prediction.size(dim=0) > 1 and NMS:
                non_max_suppression(class_prediction=class_prediction,iou_threshold=iou_threshold,box_format=box_format)
            
            batch_index = class_prediction.new(class_prediction.size(dim=0),1).fill_(i)

            if not write:
                final_prediction = torch.cat(tensors=(batch_index,class_prediction),dim=1)
                write = True
            else:
                class_prediction = torch.cat(tensors=(batch_index,class_prediction),dim=1)
                final_prediction = torch.cat(tensors=(final_prediction,class_prediction),dim=0)

    return final_prediction

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
