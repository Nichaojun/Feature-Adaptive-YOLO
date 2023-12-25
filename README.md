# Feature Adaptive YOLO for remote sensing detection in adverse weather conditions
##  Accepted by VCIP 2023 [[Baidu Cloud]](https://pan.baidu.com/s/1636ofSq77uXaqAlRjs4HEQ?pwd=70ts) 
[Chaojun Ni](https://github.com/Nichaojun), Wenhui Jiang, Chao Cai, Qishou Zhu, [**Yuming Fang**](http://sim.jxufe.cn/JDMKL/ymfang.html)

| ![space-1.jpg](https://github.com/Nichaojun/Feature-Adaptive-YOLO/blob/master/picture/1.1.png) | 
|:--:| 
|*Fig. 1. Algorithm framework diagram of FA-YOLO.*|

| ![space-1.jpg](https://github.com/Nichaojun/Feature-Adaptive-YOLO/blob/master/picture/12.png) | 
|:--:| 
|*Fig. 2. The structure of Adaptive Filters.*|

| ![space-1.jpg](https://github.com/Nichaojun/Feature-Adaptive-YOLO/blob/master/picture/3.png) | 
|:--:| 
|*Fig. 3. The structure of DG-Head.*|
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
| Faster RCNN \cite{ref13}    | ResNet-101-FPN     | 56.9                  | 33.5              | 49.7                         | 31.8                     |
| RetinaNet \cite{ref22}       | ResNet-101-FPN     | 60.3                  | 38.3              | 51.5                         | 33.3                     |
| DETR \cite{ref11}            | ResNet-101-FPN     | 59.5                  | 37.2              | 53.6                         | 34.3                     |
| CENTERNET \cite{ref12}       | ResNet-101-FPN     | 57.1                  | 36.1              | 49.0                         | 30.6                     |
| MSBDN \cite{ref16}           | -                  | 54.7                  | 32.1              | 47.3                         | 28.1                     |
| DSNet \cite{ref17}           | -                  | 57.8                  | 36.9              | 48.6                         | 29.8                     |
| DAYOLO \cite{ref25}          | CSP-Darknet53      | 58.3                  | 36.3              | 51.0                         | 32.7                     |
| YOLOV5S \cite{yolo}          | CSP-Darknet53      | 62.9                  | 40.2              | 53.5                         | 33.7                     |
| AOD-YOLOV5S \cite{Aod}       | CSP-Darknet53      | 61.4                  | 42.1              | 54.3                         | 32.6                     |
| DehazeNet-YOLOV5S \cite{dehaze} | CSP-Darknet53  | 60.8                  | 41.3              | 53.5                         | 35.2                     |
| FFA-YOLOV5S \cite{ffa}       | CSP-Darknet53      | 63.2                  | 42.3              | 54.5                         | 34.7                     |
| IA-YOLO \cite{ref15}         | CSP-Darknet53      | 61.7                  | 39.9              | 55.8                         | 34.5                     |
| Ours                          | CSP-Darknet53      | **70.3**              | **47.8**          | **60.8**                     | **40.0**                 |




```bash  
# put checkpoint model in the corresponding directory 
# change the data and model paths in core/config.py
$ python evaluate.py 
```
# Installation
```bash
$ git clone https://github.com/wenyyu/Image-Adaptive-YOLO.git  
$ cd Image-Adaptive-YOLO  
# Require python3 and tensorflow
$ pip install -r ./docs/requirements.txt

![image](https://user-images.githubusercontent.com/24246792/146735760-4fcf7be9-fdd2-4694-8d91-d254144c52eb.png)

# Train and Evaluate on the datasets
1. Download VOC PASCAL trainval and test data
```bashrc
$ wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
$ wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
$ wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
```
Extract all of these tars into one directory and rename them, which should have the following basic structure.
```bashrc

VOC           # path:  /home/lwy/work/code/tensorflow-yolov3/data/VOC
├── test
|    └──VOCdevkit
|        └──VOC2007 (from VOCtest_06-Nov-2007.tar)
└── train
     └──VOCdevkit
         └──VOC2007 (from VOCtrainval_06-Nov-2007.tar)
         └──VOC2012 (from VOCtrainval_11-May-2012.tar)
                     
$ python scripts/voc_annotation.py
```
2. Generate Voc_foggy_train and Voc_foggy_val dataset offline
```bash  
# generate ten levels' foggy training images and val images, respectively
$ python ./core/data_make.py 
```

3. Edit core/config.py to configure  
```bashrc
--vocfog_traindata_dir'  = '/data/vdd/liuwenyu/data_vocfog/train/JPEGImages/'
--vocfog_valdata_dir'    = '/data/vdd/liuwenyu/data_vocfog/val/JPEGImages/'
--train_path             = './data/dataset_fog/voc_norm_train.txt'
--test_path              = './data/dataset_fog/voc_norm_test.txt'
--class_name             = './data/classes/vocfog.names'
```
4. Train and Evaluate
```bash  
$ python train.py # we trained our model from scratch.  
$ python evaluate.py   
$ cd ./experiments/.../mAP & python main.py 
``` 
5. More details of Preparing dataset or Train with your own dataset  
   reference the implementation [tensorflow-yolov3](https://github.com/YunYang1994/tensorflow-yolov3).
   
# Train and Evaluate on low_light images
The overall process is the same as above, run the *_lowlight.py to train or evaluate.

# Acknowledgments

The code is based on [tensorflow-yolov3](https://github.com/YunYang1994/tensorflow-yolov3), [exposure](https://github.com/yuanming-hu/exposure).

# Citation

```shell
@inproceedings{liu2022imageadaptive,
  title={Image-Adaptive YOLO for Object Detection in Adverse Weather Conditions},
  author={Liu, Wenyu and Ren, Gaofeng and Yu, Runsheng and Guo, Shi and Zhu, Jianke and Zhang, Lei},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2022}
}

@article{liu2022improving,
  title={Improving Nighttime Driving-Scene Segmentation via Dual Image-adaptive Learnable Filters},
  author={Liu, Wenyu and Li, Wentong and Zhu, Jianke and Cui, Miaomiao and Xie, Xuansong and Zhang, Lei},
  journal={arXiv e-prints},
  pages={arXiv--2207},
  year={2022}
}
```
