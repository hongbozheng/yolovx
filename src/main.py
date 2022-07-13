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
    image = cv2.resize(image,(input_dimension,input_dimension))
    image = image.transpose((2,0,1))
    image = image[np.newaxis,:,:,:]/255.0
    image = torch.from_numpy(image).float()
    image = Variable(image)

    return image

def main():
    YOLOv3 = darknet.Darknet(config.YOLOv3_CFG,config.YOLOv3_WEIGHTS)
    # if CUDA:
    #     YOLOv3.to(DEVICE)
    #     print('[INFO]: YOLOv3 Model Loaded into CUDA')
    YOLOv3.eval()
    net = YOLOv3.get_net()
    configuration = YOLOv3.get_configuration()[1:]
    batch = net['batch']
    input_dimension = net['height']
    input_image = get_input_image(config.IMAGE,input_dimension)
    
    start = time.time()
    detections = YOLOv3.forward(input_image)
    
    yolo_detection = torch.FloatTensor()

    for (yolo_layer_index,detection) in enumerate(detections):
        anchors = [anchor for index,anchor in enumerate(configuration[detection[0]]['anchors']) if index in configuration[detection[0]]['mask']]
        num_class = configuration[detection[0]]['classes']
        detection = utils.detection_postprocessing(detection=detection[1],batch=batch,input_dimension=input_dimension,anchors=anchors,num_class=num_class,CUDA=True)
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
   
    final_detection = utils.get_final_detection(yolo_detection=yolo_detection,obj_score_threshold=config.OBJ_SCORE_THRESHOLD,num_class=num_class,iou_threshold=config.IOU_THRESHOLD,box_format='midpoint')

    print('[Final Detection]:     {}'.format(final_detection))
    print('[Final Detection Dim]: {}'.format(final_detection.size()))
    
    images = []
    images.append(image)

    # only work for 1 image (1 batch) right now
    final_image_detection = utils.draw_bounding_box(input_dimension=net['height'],final_detection=final_detection,images=images)
    end = time.time()
    print('[INFO]: Inference takes {}'.format((end-start)*1.0e6))

    for index,image in enumerate(final_image_detection):
        cv2.imwrite("../detection_"+str(index)+".png",image)

if __name__ == '__main__':
    main() 
