import torch

YOLOv3_CFG = '../cfg/yolov3.cfg'
YOLOv3_WEIGHTS = '../weights/yolov3.weights'
CUDA = torch.cuda.is_available()
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
IOU_THRESHOLD = 0.4
OBJ_SCORE_THRESHOLD = 0.75
