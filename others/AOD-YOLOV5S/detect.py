import argparse
import os
import platform
import shutil
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import numpy as np
from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords,
    xyxy2xywh, plot_one_box, strip_optimizer, set_logging)
from utils.torch_utils import select_device, load_classifier, time_synchronized

class Option:
  def __init__(self,weight_path='', output_path='',source='',view_img=False,save_txt=False,img_size=640,device='',conf_thres=0.4,iou_thres=0.5,classes=None,agnostic_nms=False,augment=False):
    self.output=output_path
    self.source=source
    self.weights = weight_path
    self.view_img=view_img
    self.save_txt=save_txt
    self.img_size=img_size
    self.device=device
    self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms = conf_thres, iou_thres, classes, agnostic_nms
    self.augment = augment

# Load model
class detector:
  def __init__(self,opt):
    out, source, weights, view_img, save_txt, imgsz = \
          opt.output, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    webcam = False
    set_logging()
    if not os.path.exists(out):
        os.makedirs(out)  # make new output folder
    self.device = select_device(opt.device)
    self.model = attempt_load(weights, map_location=self.device)  # load FP32 model
    self.imgsz = check_img_size(imgsz, s=self.model.stride.max())  # check img_size
    self.half = self.device.type != 'cpu'  # half precision only supported on CUDA
    if self.half:
        self.model.half()  # to FP16
    

    # Set Dataloader
    # vid_path, vid_writer = None, None
    # if webcam:
    #     view_img = True
    #     cudnn.benchmark = True  # set True to speed up constant image size inference
    #     self.dataset = LoadStreams(source, img_size=self.imgsz)
    # else:
    #     save_img = True
        


  def detect(self,opt,save_img=False):
    img = letterbox(opt.source, new_shape=self.imgsz)[0]
      # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)

    dataset = [(img,opt.source)]

    img = torch.zeros((1, 3, self.imgsz, self.imgsz), device=self.device)  # init img
    result = []
    for img, im0s in dataset:
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        pred = self.model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        for i, det in enumerate(pred):  # detections per image
          if det is not None and len(det):
              # Rescale boxes from img_size to im0 size
              det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0s.shape).round()
              # Write results
              for *xyxy, conf, cls in reversed(det):
                  # label = '%s %.2f' % (names[int(cls)], conf)
                  c1,c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
                  result.append(((c1,c2),int(cls),conf))   
    return result
