import darknet
from utils import detection_postprocessing
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
    YOLOv3 = darknet.Darknet('../cfg/yolov3.cfg')
    YOLOv3.load_weights('../weights/yolov3.weights')
    net = YOLOv3.get_net()
    blocks = YOLOv3.get_blocks()[1:]
    input_dimension = net['height']
    input_image = get_input_image('../dog-cycle-car.png',input_dimension)
    detection = YOLOv3.forward(input_image)
    
    # print(detection[0][1].size())
    # print(detection[1][1].size())
    # print(detection[2][1].size())
    num_class = blocks[detection[0][0]]['classes']
    anchor = [x for index,x in enumerate(blocks[detection[0][0]]['anchors']) if index in blocks[detection[0][0]]['mask']]
    
    print(anchor)
    detection = detection_postprocessing(detection=detection[0][1],input_dimension=input_dimension,anchor=anchor,num_class=num_class,CUDA=True)

if __name__ == '__main__':
    main()
    
