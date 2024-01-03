        
from nets.dip import Dip_Cnn, cfg,Dip_filters   
import cv2
import torch
import numpy as np
import torchvision.transforms as transforms
if __name__ == "__main__":
    device          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Dip_Cnn()
    #-------权重读入的部分----------#
    model_path = "/home/xujintao/yolo3-pytorch-master/logs/ep030-loss0.073-val_loss0.056.pth"
    
    
    model_dict      = model.state_dict()
    pretrained_dict = torch.load(model_path, map_location = device)
    load_key, no_load_key, temp_dict = [], [], {}
    for k, v in pretrained_dict.items():
       
        k = ".".join(k.split(".")[1:])
        if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
            temp_dict[k] = v
            load_key.append(k)
        else:
            no_load_key.append(k)
    model_dict.update(temp_dict)
    model.load_state_dict(model_dict)
    model = model.to(device)
    #--------选择图片-------------#
    x = cv2.imread("/home/xujintao/yolo3-pytorch-master/random2.jpg")
    x = x/255.0
    x = torch.tensor(x).to(device).to(torch.float32)
    x = x.permute(2,0,1)
    resize_image = transforms.Resize((256,256))(x.clone())
    resize_image = resize_image.unsqueeze(0)
    features = model(resize_image)
    x = x.unsqueeze(0)
    filtered_img = Dip_filters(features, cfg, x)
    filtered_img = filtered_img.permute(0,2,3,1)
    filtered_img = filtered_img.to("cpu").detach().numpy()
    filtered_img = np.clip(filtered_img*255, 0, 255)
    #-------------保存图片----------#
    cv2.imwrite("random_defog_30.jpg", filtered_img[0])