import torch

epsilon = 1e-6

def intersection_over_union(box_1, box_2, box_format='midpoint'):
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

def non_max_suppression(bbox, iou_threshold, objectiveness_threshold, box_format='format'):
    assert type(bbox) == list

    bbox = [box for box in bbox if box[4] > objectiveness_threshold]
    bbox = sorted(bbox, key=lambda x:x[4], reverse=True)
    bbox_nms = []
    
    # need this function ?????

def get_evaluation_box(prediction,obj_score_threshold,num_class,nms=True,num_threshold=0.5):
    # maybe prediction[...,4] also work ?????
    obj_score_mask = (prediction[:,:,4] >= obj_score_threshold).float().unsqueeze(dim=2)
    prediction *= obj_score_mask

    for i in range(prediction.size(dim=0)):
        nonzero_index = torch.nonzero(prediction[i][:,4])
        class_index, highest_class_score = torch.max(image_prediction[:,5:5+num_class],dim=1)
        # maybe image_prediction[...,:5] also work ?????
        image_prediction = torch.cat(tuple=(image_prediction[:,:5],
                                            # why he used .float() for class index ?????
                                            class_index.int().unsqueeze(dim=1),
                                            highest_class_score.float().unsqueeze(dim=1)),dim=1)
       # try:


    pass

a = torch.tensor([[[1,1,1,1,1],
                   [2,2,2,2,2],
                   [3,3,3,3,3]]])
print('a: {}'.format(a))
print('size a: {}'.format(a.size()))

mask = (a[:,:,4] >= 4).float().unsqueeze(2)
print('mask: {}'.format(mask))
print('check nonzero: {}'.format(torch.nonzero(mask)))

prediction = a*mask
print('prediction: {}'.format(prediction))

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
