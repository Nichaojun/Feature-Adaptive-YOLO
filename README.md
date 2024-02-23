# Feature Adaptive YOLO for remote sensing detection in adverse weather conditions
##  Accepted by VCIP 2023 [[IEEE]](https://ieeexplore.ieee.org/document/10402716) 
[Chaojun Ni](https://github.com/Nichaojun), [**Wenhui Jiang**](http://sim.jxufe.edu.cn/down/show-31909.aspx?id=98), Chao Cai, Qishou Zhu, [**Yuming Fang**](http://sim.jxufe.edu.cn/down/show-1226.aspx?id=98)

|                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      ![space-1.jpg](https://github.com/Nichaojun/Feature-Adaptive-YOLO/blob/master/picture/1.1.png)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      | 
|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:| 
|                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            *Fig. 1. Algorithm framework diagram of FA-YOLO.*                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| *Target detection in remote sensing has been one of the most challenging tasks in the past few decades. However, the detection performance in adverse weather conditions still needs to be satisfactory, mainly caused by the low-quality image features and the fuzzy boundary information. This work proposes a novel framework called Feature Adaptive YOLO (FA-YOLO). Specifically, we present a Hierarchical Feature Enhancement Module (HFEM), which adaptively performs feature-level enhancement to tackle the adverse impacts of different weather conditions. Then, we propose an Adaptive receptive Field enhancement Module (AFM) that dynamically adjusts the receptive field of the features and thus can enrich the context information for feature augmentation. In addition, we introduce Deformable Gated Head (DG-Head) which reduces the clutter caused by adverse weather. Experimental results on RTTS and two synthetic datasets demonstrate that our proposed FA-YOLO significantly outperforms other state-of-the-art target detection models.* |

## Update

| Date       | Updates                                 | Bug Fixes                                         |
|------------|-----------------------------------------|---------------------------------------------------|
| 2023-03-13 | Reproduced IA-YOLO algorithm, achieving results close to the original paper on RTTS dataset. | Fixed the issue of incorrect image size reading in FA-YOLO, and added a new data augmentation module. |
| 2023-03-21 | Reproduced GDIP-YOLO algorithm.         | Fixed the "HFDIP" parameter, allowing it to control whether to enable the filtering module.         |
| 2023-04-12 | Reproduced Togethernet algorithm.      |                                                   |


## Datasets and Models

| Datasets and Models                            | Links                                              |
|-----------------------------------------------|----------------------------------------------------|
| **Datasets in dark conditions**               | [dark](http://host.robots.ox.ac.uk/pascal/VOC/)   |
|                                               |                                                    |
| **Datasets under dense fog conditions**       | [fog](http://host.robots.ox.ac.uk/pascal/VOC/)    |
|                                               |                                                    |
| **DIOR remote sensing dataset**               | [DIOR](http://host.robots.ox.ac.uk/pascal/VOC/)   |
|                                               |                                                    |
| **DIOR remote sensing dataset with fog**      | [DIOR-FOG](https://github.com/cs-chan/Exclusively-Dark-Image-Dataset/tree/master/Dataset) |


## Result
| Algorithm       | Backbone           | Dior\_Foggy $AP_{50}$ | Dior\_Foggy $AP$ | Dior\_Severe\_Foggy $AP_{50}$ | Dior\_Severe\_Foggy $AP$ |
|-----------------|--------------------|-----------------------|-------------------|------------------------------|--------------------------|
| Faster RCNN     | ResNet-101-FPN     | 56.9                  | 33.5              | 49.7                         | 31.8                     |
| RetinaNet       | ResNet-101-FPN     | 60.3                  | 38.3              | 51.5                         | 33.3                     |
| DETR            | ResNet-101-FPN     | 59.5                  | 37.2              | 53.6                         | 34.3                     |
| CENTERNET       | ResNet-101-FPN     | 57.1                  | 36.1              | 49.0                         | 30.6                     |
| MSBDN           | -                  | 54.7                  | 32.1              | 47.3                         | 28.1                     |
| DSNet           | -                  | 57.8                  | 36.9              | 48.6                         | 29.8                     |
| DAYOLO          | CSP-Darknet53      | 58.3                  | 36.3              | 51.0                         | 32.7                     |
| YOLOV5S         | CSP-Darknet53      | 62.9                  | 40.2              | 53.5                         | 33.7                     |
| AOD-YOLOV5S     | CSP-Darknet53      | 61.4                  | 42.1              | 54.3                         | 32.6                     |
| DehazeNet-YOLOV5S | CSP-Darknet53  | 60.8                  | 41.3              | 53.5                         | 35.2                     |
| FFA-YOLOV5S     | CSP-Darknet53      | 63.2                  | 42.3              | 54.5                         | 34.7                     |
| IA-YOLO         | CSP-Darknet53      | 61.7                  | 39.9              | 55.8                         | 34.5                     |
| Ours            | CSP-Darknet53      | **70.3**              | **47.8**          | **60.8**                     | **40.0**                 |

## Problems

| ![space-1.jpg](https://github.com/Nichaojun/Feature-Adaptive-YOLO/blob/master/picture/0.4.png) |
|:-----------------------------------------------------------------------------------------------|
| *Failure detection examples from the YOLOV5 model. (a) Clean images. (b) The same image under adverse weather conditions. (c) Activations of feature maps from YOLOV5. (d) Detection results, where green boxes indicate correct detections, red boxes indicate false detections, orange boxes indicate missed detections.*                                                                                             |


## Framework

|    ![space-1.jpg](https://github.com/Nichaojun/Feature-Adaptive-YOLO/blob/master/picture/1.1.png)      |
|:-----|
|   *Fig. 1. Algorithm framework diagram of FA-YOLO.*    |


## AFM
| ![space-1.jpg](https://github.com/Nichaojun/Feature-Adaptive-YOLO/blob/master/picture/12.png) |
|:----------------------------------------------------------------------------------------------|
| *The structure of Adaptive Filters*                                                           |                                                                                             |

## DG-Head
| ![space-1.jpg](https://github.com/Nichaojun/Feature-Adaptive-YOLO/blob/master/picture/3.png) |
|:---------------------------------------------------------------------------------------------|
| *The structure of DG-Head.*                                                          |                                                                                             |


## Visualization
| ![space-1.jpg](https://github.com/Nichaojun/Feature-Adaptive-YOLO/blob/master/picture/6.4.png)                                                                                                                                                                                                |
|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| *Visualization of detection results. (a) Clean images. (b) The same image under adverse weather conditions. (c) Results of IA-YOLO. (d) Results of FA-YOLO, where green boxes indicate correct detections, red boxes indicate false detections, and orange boxes indicate missed detections.* 