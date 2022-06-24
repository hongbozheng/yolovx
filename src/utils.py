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

    if box_format = 'corner':
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
