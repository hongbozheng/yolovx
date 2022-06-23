# YOLOvX

Python Implementation of YOLOv3

## Project Setup

#### 1. Environment Setup [Linux]

##### - Install `conda`

Go to [conda >> User guide >> Installation >> Installing on Linux](https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html)

Under section `Download the Installer`, click on `Anaconda Installer for Linux`

In the terminal, go to `Downloads` folder and run the following command
```
bash Anaconda3-<latest version>-Linux-x86_64.sh
```
`<latest version>` the latest date of the released version of Anaconda3

Follow the instruction and install conda

After installing conda successfully, run the following command in the terminal
```
source ~/.bashrc
```
This will source the bash script

##### - Create a new conda environment

Go to [conda >> User guide >> Tasks >> Managing environments](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-with-commands)

In the terminal, create a new environment with the following command
```
conda create -n <name> python=<python version>
```
`<name>` name of the environment

`<version>` python version (e.g. python=3.9)

Activate the new environment with the following command
```
conda activate <name>
```
`<name>` name of the environment just created

##### - Install Python Library

* numpy
* torch

In the terminal, executing the following commands
```
conda install numpy
conda install -c pytorch pytorch
```

#### 2. Download cfg file
Download the `cfg` file for YOLOv3 from [here](https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg)

#### 3. Download pre-trained weights
```
wget https://pjreddie.com/media/files/yolov3.weights
```
or download [here (237MB)](https://pjreddie.com/media/files/yolov3.weights)

## Implementation of YOLO

#### 1. Parse `cfg` file
[CFG Parameters in the `net` section](https://github.com/AlexeyAB/darknet/wiki/CFG-Parameters-in-the-%5Bnet%5D-section)

[CFG Parameters in the different layers](https://github.com/AlexeyAB/darknet/wiki/CFG-Parameters-in-the-different-layers)

#### 2. Create `model` from the parsed `cfg` file

#### 3. Create `darknet`
* forward pass
* load pre-trained weights

## Test YOLOv3

STILL WORKING ON IT

## Reference

[YOLOv3 Paper](https://pjreddie.com/media/files/papers/YOLOv3.pdf)

[darknet GitHub Repository](https://github.com/pjreddie/darknet)

[darknet yolo](https://pjreddie.com/darknet/yolo/)

[Implement YOLOv3 from scratch in Pytorch](https://blog.paperspace.com/how-to-implement-a-yolo-object-detector-in-pytorch/)

[Implement YOLOv3 from scratch in Pytorch GitHub Repository](https://github.com/ayooshkathuria/pytorch-yolo-v3)

Another `Implement YOLOv3 with Training setup from scratch` tutorial

[Implement YOLOv3 with Training setup from scratch YouTube Video](https://www.youtube.com/watch?v=Grir6TZbc1M)

[Implement YOLOv3 with Training setup from scratch](https://sannaperzon.medium.com/yolov3-implementation-with-training-setup-from-scratch-30ecb9751cb0)

[Implement YOLOv3 with Training setup GitHub Repository](https://github.com/aladdinpersson/Machine-Learning-Collection/tree/master/ML/Pytorch/object_detection/YOLOv3)
