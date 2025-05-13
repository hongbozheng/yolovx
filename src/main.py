#! /usr/bin/env python3

import config
import utils
import darknet
import cv2
import numpy as np
from torch.autograd import Variable
import torch
import time


def get_input_image(image_path,input_dimension):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    image = cv2.resize(image,(input_dimension,input_dimension))
    image = image.transpose((2,0,1))
    image = image[np.newaxis,:,:,:]/255.0
    image = torch.from_numpy(image).float()
    image = Variable(image)

    return image


def main():
    class_label = utils.load_label(config.CLASS_LABEL)
    YOLOv3 = darknet.Darknet(config.YOLO_CFG,config.YOLO_WEIGHTS) 
    YOLOv3.eval()
    net = YOLOv3.get_net()
    configuration = YOLOv3.get_configuration()[1:]
    batch = net['batch']
    input_dimension = net['height']
    input_image = get_input_image(config.IMAGE,input_dimension)

    if config.CUDA:
        print('[INFO]: Loading YOLO into CUDA')
        YOLOv3.cuda()
        input_image = input_image.cuda()
        print('[INFO]: YOLO Loaded into CUDA')

    detections = YOLOv3.forward(input_image)

    if config.CUDA:
        yolo_detection = torch.FloatTensor().cuda()
    else:
        yolo_detection = torch.FloatTensor() 
    print('[INFO]: Start post-processing')
    postprocessing_start_time = time.time()
    for (yolo_layer_index,detection) in enumerate(detections):
        try:
            anchors = [anchor for index,anchor in enumerate(configuration[detection[0]]['anchors']) if index in configuration[detection[0]]['mask']]
        except:
            anchors = [configuration[detection[0]]['anchors'][configuration[detection[0]]['mask']]]
        num_class = configuration[detection[0]]['classes']
        detection = utils.detection_postprocessing(detection=detection[1],batch=batch,input_dimension=input_dimension,anchors=anchors,num_class=num_class,CUDA=config.CUDA)
        # # of detection in each Yolo Layer
        if config.YOLO_LAYER_NUM_DETECTION:
            utils.get_yolo_layer_num_detection(detection=detection,obj_score_threshold=config.OBJ_SCORE_THRESHOLD,yolo_layer_index=yolo_layer_index)
        yolo_detection = torch.cat(tensors=(yolo_detection,detection),dim=1)

    # Plot Anchor Box
    image = cv2.imread(config.IMAGE)
    image = np.array(image)
    if config.PLOT_ANCHOR_BOX:
        yolo_layer_index = [detection[0] for detection in detections]
        anchor_image = utils.draw_anchor_box(input_dimension=net['height'],configuration=configuration,yolo_layer_index=yolo_layer_index,image=image,mode='separate')

    final_detection = utils.get_final_detection(yolo_detection=yolo_detection,obj_score_threshold=config.OBJ_SCORE_THRESHOLD,num_class=num_class,iou_threshold=config.IOU_THRESHOLD,box_format='midpoint',CUDA=config.CUDA)

    print('[INFO]: Finish Post-Processing')
    print('[INFO]: Post-Processing took {}ms'.format((time.time()-postprocessing_start_time)*1.0e3))
    print('[INFO]: YOLO made {} detection(s)'.format(final_detection.size(dim=1)))
    # print('[Final Detection]:     {}'.format(final_detection))
    # print('[Final Detection Dim]: {}'.format(final_detection.size()))

    images = []
    images.append(image)

    # only work for 1 image (1 batch) right now
    print('[INFO]: Start to draw bounding box')
    final_image_detection = utils.draw_bounding_box(class_label=class_label,input_dimension=net['height'],final_detection=final_detection,images=images)
    print('[INFO]: Finish drawing bounding box')
    for index,image in enumerate(final_image_detection):
        cv2.imwrite("../detection_"+str(index)+".png",image)


if __name__ == '__main__':
    main() 
