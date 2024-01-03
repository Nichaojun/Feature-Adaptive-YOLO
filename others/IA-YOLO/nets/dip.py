import torch
import torch.nn as nn
import math
import numpy as np
from easydict import EasyDict as edict
from nets.filters import DefogFilter,ImprovedWhiteBalanceFilter,GammaFilter,\
                            ToneFilter, ContrastFilter, UsmFilter
from numba import jit
import random


class Dip_Cnn(nn.Module):
    def __init__(self):
        super(Dip_Cnn, self).__init__()
        channels = 16
        self.cnnnet = nn.Sequential(
                        nn.Conv2d(3, channels, kernel_size=3, stride=2, padding=1, bias=True),
                        nn.LeakyReLU(negative_slope=0.1),
                        
                        nn.Conv2d(channels, channels*2, kernel_size=3, stride=2, padding=1, bias=True),
                        nn.LeakyReLU(negative_slope=0.1),
                        
                        nn.Conv2d(channels*2, channels*2, kernel_size=3, stride=2, padding=1, bias=True),
                        nn.LeakyReLU(negative_slope=0.1),
                        
                        nn.Conv2d(channels*2, channels*2, kernel_size=3, stride=2, padding=1, bias=True),
                        nn.LeakyReLU(negative_slope=0.1),
                        
                        nn.Conv2d(channels*2, channels*2, kernel_size=3, stride=2, padding=1, bias=True),
                        nn.LeakyReLU(negative_slope=0.1),
        )
        self.full_layers1 = nn.Sequential(
                        nn.Linear(2048, 64),
                        nn.Linear(64, 15),
        )
    def forward(self, x):
        out = self.cnnnet(x)
        out = out.view(out.size(0), -1)
        out = self.full_layers1(out)
        return out

def fog_image(image):
    @jit
    def AddHaz_loop(img_f, center, size, beta, A):
        (row, col, chs) = img_f.shape
        for j in range(row):
            for l in range(col):
                d = -0.04 * math.sqrt((j - center[0]) ** 2 + (l - center[1]) ** 2) + size
                td = math.exp(-beta * d)
                img_f[j][l][:] = img_f[j][l][:] * td + A * (1 - td)
        return img_f
    img_f = image/255
    (row, col, chs) = image.shape
    A = 0.5  
    # beta = 0.08 
    beta = random.randint(0, 9) 
    beta = 0.01 * beta + 0.05
    size = math.sqrt(max(row, col)) 
    center = (row // 2, col // 2)  
    foggy_image = AddHaz_loop(img_f, center, size, beta, A)
    img_f = np.clip(foggy_image*255, 0, 255)
    img_f = img_f.astype(np.uint8)

    return img_f
#----------------
#过滤器参数
#---------------
cfg = edict() 
cfg.filters = [
    DefogFilter, ImprovedWhiteBalanceFilter,  GammaFilter,
    ToneFilter, ContrastFilter, UsmFilter
]
cfg.num_filter_parameters = 15

cfg.defog_begin_param = 0

cfg.wb_begin_param = 1
cfg.gamma_begin_param = 4
cfg.tone_begin_param = 5
cfg.contrast_begin_param = 13
cfg.usm_begin_param = 14


cfg.curve_steps = 8
cfg.gamma_range = 3
cfg.exposure_range = 3.5
cfg.wb_range = 1.1
cfg.color_curve_range = (0.90, 1.10)
cfg.lab_curve_range = (0.90, 1.10)
cfg.tone_curve_range = (0.5, 2)
cfg.defog_range = (0.1, 1.0)
cfg.usm_range = (0.0, 5)        
         
def Dip_filters(features, cfg, img):
    filtered_image_batch = img
    B, C, W, H = img.shape
    #----------------
    #构建defog的过滤器参数
    #----------------
    dark = torch.zeros([B, W, H],dtype=torch.float32).to(img.device)   
    defog_A = torch.zeros([B, C],dtype=torch.float32).to(img.device)   
    IcA = torch.zeros([B, W, H],dtype=torch.float32).to(img.device)   
    for i in range(B):
        dark_i = DarkChannel(img[i])
        defog_A_i = AtmLight(img[i], dark_i)
        IcA_i = DarkIcA(img[i], defog_A_i)
        dark[i, ...] = dark_i
        defog_A[i, ...] = defog_A_i
        IcA[i, ...] = IcA_i
    IcA = IcA.unsqueeze(-1)
    #需要经过的6个过滤器
    filters = cfg.filters
    filters = [x(filtered_image_batch, cfg) for x in filters]
    filter_features = features
    for j, filter in enumerate(filters):

        filtered_image_batch, filter_parameter = filter.apply(
            filtered_image_batch, filter_features, defog_A, IcA)
    
    return filtered_image_batch

def DarkChannel(im):
    R = im[0, :, :]
    G = im[1, :, :]
    B = im[2, :, :]
    dc = torch.min(torch.min(R, G), B)
    return dc

def AtmLight(im, dark):
    c, h, w = im.shape
    imsz = h * w
    numpx = int(max(torch.floor(torch.tensor(imsz) / 1000), torch.tensor(1)))
    darkvec = dark.reshape(imsz, 1)
    imvec = im.reshape(3, imsz)

    indices = torch.argsort(darkvec)
    indices = indices[(imsz - numpx):imsz]

    atmsum = torch.zeros([3, 1]).to(imvec.device)
    for ind in range(1, numpx):
        atmsum = atmsum + imvec[:, indices[ind]]

    A = atmsum / numpx
    return A.reshape(1, 3)

def DarkIcA(im, A):
    c, h, w = im.shape
    im3 = torch.zeros([c,h,w]).to(im.device)
    for ind in range(0, 3):
        im3[ind, :, :] = im[ind, :, :] / A[0, ind]
    return DarkChannel(im3)