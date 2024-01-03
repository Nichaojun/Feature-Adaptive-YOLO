# Feature Adaptive YOLO for remote sensing detection in adverse weather conditions
##  Accepted by VCIP 2023 [[Baidu Cloud]](https://pan.baidu.com/s/1636ofSq77uXaqAlRjs4HEQ?pwd=70ts) 
[Chaojun Ni](https://github.com/Nichaojun), [**Wenhui Jiang**](http://sim.jxufe.edu.cn/down/show-31909.aspx?id=98), Chao Cai, Qishou Zhu, [**Yuming Fang**](http://sim.jxufe.edu.cn/down/show-1226.aspx?id=98)

| ![space-1.jpg](https://github.com/Nichaojun/Feature-Adaptive-YOLO/blob/master/picture/1.1.png) | 
|:--:| 
|*Fig. 1. Algorithm framework diagram of FA-YOLO.*|

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

