# Feature Adaptive YOLO for remote sensing detection in adverse weather conditions
##  Accepted by VCIP 2023 [[Baidu Cloud]](https://pan.baidu.com/s/1636ofSq77uXaqAlRjs4HEQ?pwd=70ts) 
[Chaojun Ni](https://github.com/Nichaojun), Wenhui Jiang, Chao Cai, Qishou Zhu, [**Yuming Fang**](http://sim.jxufe.cn/JDMKL/ymfang.html)

| ![space-1.jpg](https://github.com/Nichaojun/Feature-Adaptive-YOLO/blob/master/picture/1.1.png) | 
|:--:| 
|*Fig. 1. Algorithm framework diagram of FA-YOLO.*|

## Update
### Date: 2023-03-13

#### Updates:

Reproduced IA-YOLO algorithm, achieving results close to the original paper on RTTS dataset.

#### Bug Fixes:

Fixed the issue of incorrect image size reading in FA-YOLO, and added a new data augmentation module.

### Date: 2023-03-21

#### Updates:

Reproduced GDIP-YOLO algorithm.

#### Bug Fixes:

Fixed the "HFDIP" parameter, allowing it to control whether to enable the filtering module.

### Date: 2023-04-12

#### Updates:

Reproduced Togethernet algorithm.

## Datasets and Models
### Datasets in dark conditions
[dark](http://host.robots.ox.ac.uk/pascal/VOC/) 

### Datasets under dense fog conditions
[fog](http://host.robots.ox.ac.uk/pascal/VOC/) 

### DIOR remote sensing dataset
[DIOR](http://host.robots.ox.ac.uk/pascal/VOC/)

### DIOR remote sensing dataset with fog
[DIOR-FOG](https://github.com/cs-chan/Exclusively-Dark-Image-Dataset/tree/master/Dataset)  

| Datasets and Models                            | Links                                              |
|-----------------------------------------------|----------------------------------------------------|
| **Datasets in dark conditions**               | [dark](http://host.robots.ox.ac.uk/pascal/VOC/)   |
|                                               |                                                    |
| **Datasets under dense fog conditions**       | [fog](http://host.robots.ox.ac.uk/pascal/VOC/)    |
|                                               |                                                    |
| **DIOR remote sensing dataset**               | [DIOR](http://host.robots.ox.ac.uk/pascal/VOC/)   |
|                                               |                                                    |
| **DIOR remote sensing dataset with fog**      | [DIOR-FOG](https://github.com/cs-chan/Exclusively-Dark-Image-Dataset/tree/master/Dataset) |


# Quick test
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
