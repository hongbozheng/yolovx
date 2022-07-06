import darknet
from utils import detection_postprocessing
from utils import get_evaluation_box
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
    batch = net['batch']
    input_dimension = net['height']
    input_image = get_input_image('../dog-cycle-car.png',input_dimension)
    detections = YOLOv3.forward(input_image)
    
    # print(detection[0][1].size())
    # print(detection[1][1].size())
    # print(detection[2][1].size())
    
    final_detection = torch.FloatTensor()

    for detection in detections:
        anchors = [anchor for index,anchor in enumerate(blocks[detection[0]]['anchors']) if index in blocks[detection[0]]['mask']]
        num_class = blocks[detection[0]]['classes']
        detection = detection_postprocessing(detection=detection[1],batch=batch,input_dimension=input_dimension,anchors=anchors,num_class=num_class,CUDA=True)
        final_detection = torch.cat(tensors=(final_detection,detection),dim=1)
        # print('-----')
    print(final_detection)
    print(final_detection.size())

    get_evaluation_box(final_detection=final_detection,obj_score_threshold=0.7,num_class=num_class,iou_threshold=0.5, box_format='midpoint')

if __name__ == '__main__':
    main()
    
