import config
import torch
import numpy as np
import cv2

epsilon = 1e-6

'''
$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ load dataset label $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
'''
def load_label(data_label_file):
    fp = open(data_label_file,'r')
    label = fp.read().split('\n')[:-1]
    return label

'''
$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ detection post processing $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
'''
def detection_postprocessing(detection, batch, input_dimension, anchors, num_class, CUDA=False):
    batch_size = detection.size(dim=0)
    grid_scale = detection.size(dim=2)
    grid_size = input_dimension//grid_scale
    num_anchors = len(anchors)
    bbox_attribute = 5+num_class

    detection = detection.view(batch_size,num_anchors*bbox_attribute,grid_scale*grid_scale).transpose(dim0=1,dim1=2).contiguous()
    detection = detection.view(batch_size,grid_scale*grid_scale*num_anchors,bbox_attribute)
   
    x_offset,y_offset = torch.FloatTensor(np.meshgrid(np.arange(grid_scale),np.arange(grid_scale)))
    if CUDA:
        xy_offset = torch.cat(tensors=(x_offset.view(-1,1).cuda(),y_offset.view(-1,1).cuda()),dim=1).repeat(1,num_anchors).view(-1,2).unsqueeze(dim=0)
    else:
        xy_offset = torch.cat(tensors=(x_offset.view(-1,1),y_offset.view(-1,1)),dim=1).repeat(1,num_anchors).view(-1,2).unsqueeze(dim=0)
    
    detection[:,:,:2] = (torch.sigmoid(detection[:,:,:2])+xy_offset)*grid_size
    detection[:,:,4:] = torch.sigmoid(detection[:,:,4:])
    if CUDA:
        anchors = torch.FloatTensor(anchors).repeat(grid_scale*grid_scale,1).unsqueeze(dim=0).cuda()
    else:
        anchors = torch.FloatTensor(anchors).repeat(grid_scale*grid_scale,1).unsqueeze(dim=0)
    detection[:,:,2:4] = torch.exp(detection[:,:,2:4])*anchors

    return detection

'''
$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ intersection over union $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
'''
def intersection_over_union(box_1, box_2, box_format='midpoint'):
    if box_format == 'midpoint':
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

    return intersection/(box1_area+box2_area-intersection+epsilon)

'''
$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ non max suppression $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
'''
def non_max_suppression(class_detection, iou_threshold, box_format='midpoint'):
    class_detection = class_detection[torch.sort(input=class_detection[:,4],descending=True)[1]]
    
    for i in range(class_detection.size(dim=0)):
        for k in range(i+1,class_detection.size(dim=0)):
            if intersection_over_union(class_detection[i],class_detection[k]) > iou_threshold:
                class_detection[k] *= 0

    class_detection = class_detection[class_detection[:,0]!=0]

    return class_detection 

'''
$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ get final detection $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
'''
def get_final_detection(yolo_detection, obj_score_threshold, num_class, iou_threshold=0.5, box_format='midpoint', CUDA=False):
    yolo_detection *= (yolo_detection[:,:,4] >= obj_score_threshold).float().unsqueeze(dim=2)
    
    if CUDA:
        final_detection = torch.FloatTensor().cuda()
    else:
        final_detection = torch.FloatTensor()

    for i in range(yolo_detection.size(dim=0)):
        detection = yolo_detection[i][yolo_detection[i][:,4]!=0]

        if detection.numel() == 0:
            continue

        highest_class_score, class_index = torch.max(detection[:,5:],dim=1)
        detection = torch.cat(tensors=(detection[:,:5],class_index.float().unsqueeze(dim=1),highest_class_score.float().unsqueeze(dim=1)),dim=1)
        detect_class = torch.unique(detection[:,-2])
        
        if CUDA:
            batch_detection = torch.FloatTensor().cuda()
        else:
            batch_detection = torch.FloatTensor()

        for c in detect_class:
            class_detection = detection[detection[:,-2]==c] 
            if class_detection.size(dim=0) > 1:
                class_detection = non_max_suppression(class_detection=class_detection,iou_threshold=iou_threshold,box_format=box_format)
            
            batch_detection = torch.cat(tensors=(batch_detection,class_detection),dim=0)
        final_detection = torch.cat(tensors=(final_detection,batch_detection.unsqueeze(dim=0)),dim=0)
    return final_detection

'''
$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ # of detection in each yolo layer $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
'''
def get_yolo_layer_num_detection(detection,obj_score_threshold,yolo_layer_index):
    detection *= (detection[:,:,4] >= obj_score_threshold).float().unsqueeze(dim=2)

    for i in range(detection.size(dim=0)):
        print('[INFO]: Batch '+str(i)+' YOLO Layer '+str(yolo_layer_index)+' makes '+str(detection[i][detection[i][:,4]!=0].size(dim=0))+' detection(s)')

'''
$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ draw bounding box $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
'''
def draw_bounding_box(class_label,input_dimension,final_detection,images):
    final_image_detection = []
    for (detections,image) in zip(final_detection,images):
        image_height,image_width,_ = image.shape
        for (detection,color) in zip(detections,config.COLOR):
            TL = (int((detection[0]-detection[2]/2)*image_width/input_dimension),int((detection[1]-detection[3]/2)*image_height/input_dimension))
            cv2.rectangle(image,TL,(int((detection[0]+detection[2]/2)*image_width/input_dimension),int((detection[1]+detection[3]/2)*image_height/input_dimension)),color,config.BOUNDING_BOX_THICKNESS)
            label = class_label[int(detection[5])]+' {:.2f}'.format(float(detection[6])*100)+'%'
            label_size = cv2.getTextSize(label,config.LABEL_FONT,config.LABEL_SCALE,1)[0]
            cv2.rectangle(image,(TL[0],TL[1]-label_size[1]),(TL[0]+label_size[0],TL[1]),color,-1)
            cv2.putText(image,label,TL,config.LABEL_FONT,config.LABEL_SCALE,config.BLACK,1,cv2.LINE_AA)

        final_image_detection.append(image)
    return final_image_detection

'''
$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ draw anchor box $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
'''
def draw_anchor_box(input_dimension,configuration,yolo_layer_index,image,mode='together'):
    anchors = [anchor for anchor in configuration[yolo_layer_index[0]]['anchors']]
    image_height,image_width,_ = image.shape
    
    if mode == 'together':
        for (anchor,color) in zip(anchors,config.COLOR):
            cv2.rectangle(image,(int(image_width/2-anchor[0]/2*image_width/input_dimension),int(image_height/2-anchor[1]/2*image_height/input_dimension)),(int(image_width/2+anchor[0]/2*image_width/input_dimension),int(image_height/2+anchor[1]/2*image_height/input_dimension)),color,config.BOUNDING_BOX_THICKNESS)
        cv2.imwrite("../anchor_image/anchor.jpg",image)
        print('[INFO]: Anchor image saved at ../anchor_image/anchor.jpg')
    else:
        for index,(anchor,color) in enumerate(zip(anchors,config.COLOR)):
            image_ = np.copy(image)
            cv2.rectangle(image_,(int(image_width/2-anchor[0]/2*image_width/input_dimension),int(image_height/2-anchor[1]/2*image_height/input_dimension)),(int(image_width/2+anchor[0]/2*image_width/input_dimension),int(image_height/2+anchor[1]/2*image_height/input_dimension)),color,config.BOUNDING_BOX_THICKNESS)
            cv2.imwrite("../anchor_image/anchor_"+str(index)+".jpg",image_)
            print('[INFO]: Anchor '+str(index)+' image saved at ../anchor_image/anchor_'+str(index)+'.jpg')
