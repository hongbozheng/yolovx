import torch
import cv2

YOLOv3_CFG = '../cfg/yolov4.cfg'
YOLOv3_WEIGHTS = '../weights/yolov3.weights'
CUDA = torch.cuda.is_available()
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
IOU_THRESHOLD = 0.4
OBJ_SCORE_THRESHOLD = 0.1
BBOX_ATTRIBUTE = 85
IMAGE = '../image/COCO_train2014_000000003220.jpeg'

# FLAG
YOLO_LAYER_NUM_DETECTION = False 
PLOT_ANCHOR_BOX = False

# COLOR (B,G,R)
CYAN = (255,255,0)
BRIGHT_ORANGE = (0,165,255)
BRIGHT_YELLOW = (0,255,255)
VIOLET = (255,0,143)
RED = (0,0,255)
NEON_GREEN = (20,255,57)
PINK = (203,192,255)
BLUE = (255,0,0)
MAGENTA = (255,0,255)
CORAL = (80,127,255)
TIFFANY_BLUE = (181,186,10)
WHITE = (255,255,255)
ROSE = (127,0,255)
TURQUOISE = (200,213,200)
SILVER = (192,192,192)
BLACK = (0,0,0)
COLOR = [CYAN,BRIGHT_ORANGE,BRIGHT_YELLOW,VIOLET,RED,NEON_GREEN,PINK,BLUE,MAGENTA,CORAL,TIFFANY_BLUE,WHITE,ROSE,TURQUOISE,SILVER,CYAN,BRIGHT_ORANGE,BRIGHT_YELLOW,VIOLET,RED,NEON_GREEN,PINK,BLUE,MAGENTA,CORAL,TIFFANY_BLUE,WHITE,ROSE,TURQUOISE,SILVER]

BOUNDING_BOX_THICKNESS = 1
LABEL_FONT = cv2.FONT_HERSHEY_SIMPLEX
LABEL_SCALE = 0.35

COCO = ['person',
 'bicycle',
 'car',
 'motorcycle',
 'airplane',
 'bus',
 'train',
 'truck',
 'boat',
 'traffic light',
 'fire hydrant',
 'stop sign',
 'parking meter',
 'bench',
 'bird',
 'cat',
 'dog',
 'horse',
 'sheep',
 'cow',
 'elephant',
 'bear',
 'zebra',
 'giraffe',
 'backpack',
 'umbrella',
 'handbag',
 'tie',
 'suitcase',
 'frisbee',
 'skis',
 'snowboard',
 'sports ball',
 'kite',
 'baseball bat',
 'baseball glove',
 'skateboard',
 'surfboard',
 'tennis racket',
 'bottle',
 'wine glass',
 'cup',
 'fork',
 'knife',
 'spoon',
 'bowl',
 'banana',
 'apple',
 'sandwich',
 'orange',
 'broccoli',
 'carrot',
 'hot dog',
 'pizza',
 'donut',
 'cake',
 'chair',
 'couch',
 'potted plant',
 'bed',
 'dining table',
 'toilet',
 'tv',
 'laptop',
 'mouse',
 'remote',
 'keyboard',
 'cell phone',
 'microwave',
 'oven',
 'toaster',
 'sink',
 'refrigerator',
 'book',
 'clock',
 'vase',
 'scissors',
 'teddy bear',
 'hair drier',
 'toothbrush']
