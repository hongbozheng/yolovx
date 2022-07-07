from config import *
import darknet
from utils import detection_postprocessing
from utils import get_final_detection
import cv2
import numpy as np
from torch.autograd import Variable
import torch

def get_input_image(image_path,input_dimension):
    image = cv2.imread(image_path)
    image = cv2.resize(image,(input_dimension,input_dimension))
    image = image[:,:,::-1].transpose((2,0,1))
    image = image[np.newaxis,:,:,:]/255.0
    image = torch.from_numpy(image).float()
    image = Variable(image)
    return image

def main():
    YOLOv3 = darknet.Darknet(YOLOv3_CFG,YOLOv3_WEIGHTS)
    # if CUDA:
    #     YOLOv3.to(DEVICE)
    #     print('[INFO]: YOLOv3 Model Loaded into CUDA')
    YOLOv3.eval()
    # YOLOv3.load_weights(YOLOv3_WEIGHTS)
    net = YOLOv3.get_net()
    configuration = YOLOv3.get_configuration()[1:]
    batch = net['batch']
    input_dimension = net['height']
    input_image = get_input_image('../dog-cycle-car.png',input_dimension)
    detections = YOLOv3.forward(input_image)
    
    # print(detections[0][1])
    # print(detections[1][1])
    # print(detections[2][1])

    yolo_detection = torch.FloatTensor()

    for detection in detections:
        anchors = [anchor for index,anchor in enumerate(configuration[detection[0]]['anchors']) if index in configuration[detection[0]]['mask']]
        num_class = configuration[detection[0]]['classes']
        detection = detection_postprocessing(detection=detection[1],batch=batch,input_dimension=input_dimension,anchors=anchors,num_class=num_class,CUDA=True)
        yolo_detection = torch.cat(tensors=(yolo_detection,detection),dim=1)
        # print('-----')
    # print(yolo_detection)
    # print(yolo_detection.size())

    final_detection = get_final_detection(yolo_detection=yolo_detection,obj_score_threshold=OBJ_SCORE_THRESHOLD,num_class=num_class,iou_threshold=IOU_THRESHOLD,box_format='midpoint')

if __name__ == '__main__':
    main() 
