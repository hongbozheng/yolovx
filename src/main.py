import config
import utils
import darknet
import cv2
import numpy as np
from torch.autograd import Variable
import torch

def get_input_image(image_path,input_dimension):
    image = cv2.imread(image_path)
    image = cv2.resize(image,(input_dimension,input_dimension))
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
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
    input_image = get_input_image('../dog-cycle-car.png',input_dimension)
    detections = YOLOv3.forward(input_image)
    
    yolo_detection = torch.FloatTensor()

    for detection in detections:
        anchors = [anchor for index,anchor in enumerate(configuration[detection[0]]['anchors']) if index in configuration[detection[0]]['mask']]
        num_class = configuration[detection[0]]['classes']
        detection = utils.detection_postprocessing(detection=detection[1],batch=batch,input_dimension=input_dimension,anchors=anchors,num_class=num_class,CUDA=True)
        yolo_detection = torch.cat(tensors=(yolo_detection,detection),dim=1)
    
    # Plot Anchor Box
    image = cv2.imread('../dog-cycle-car.png')
    image = np.array(image)
    # yolo_layer_index = [detection[0] for detection in detections]
    # anchor_image = utils.draw_anchor_box(net=net,configuration=configuration,yolo_layer_index=yolo_layer_index,image=image,mode='separate')
    
    image_height,image_width,_ = image.shape
    image = image[np.newaxis,:,:,:]

    final_detection = utils.get_final_detection(yolo_detection=yolo_detection,obj_score_threshold=config.OBJ_SCORE_THRESHOLD,num_class=num_class,iou_threshold=config.IOU_THRESHOLD,box_format='midpoint')
    
    print('[Final Detection]:     {}'.format(final_detection))
    print('[Final Detection Dim]: {}'.format(final_detection.size()))
    
    # only work for 1 image (1 batch) right now
    final_image_detection = utils.draw_bounding_box(final_detection=final_detection,images=image,height_ratio=image_height/input_dimension,width_ratio=image_width/input_dimension)
    for image in final_image_detection:
        cv2.imwrite("../dog-cycle-truck.jpg",image)
    

if __name__ == '__main__':
    main() 
