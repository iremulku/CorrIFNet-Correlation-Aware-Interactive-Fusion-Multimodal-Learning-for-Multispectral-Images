from __future__ import print_function
import os
import torch 
import torch.nn as nn
from torch.utils.data import DataLoader
#import numpy as np
import datetime
import time
from thop import profile
from thop import clever_format
import torchsummary
from torchsummary import summary
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR
from F1_UNET_V1_1 import UNetV1
from F3_DATASET import satellitedata
from F4_TRAIN import train_model
from F6_CROSSVAL import CrossVal
from F7_TEST2 import test_model
from F8_IMAGES4 import get_images4
from F9_UNET_V2_3 import UNetV2
from mmmvit2 import MMVit2
from mmvit1 import MMVit1
from mmformer import mmformer
from mmvit5 import MMVit5
from mmvit4 import MMVit4
from RFNet import RFNet
from RobustSeg import RobustMseg
from MultiSenseSeg import MultiSenseSeg
import timm
from lora import LoRA_ViT
from base_vit import ViT
from seg_vit import SegWrapForViT
import ssl
ssl._create_default_https_context = ssl._create_unverified_context


class Config(object):
    NAME= "dfaNet"

    #set the output every STEP_PER_EPOCH iteration
    STEP_PER_EPOCH = 100
    ENCODER_CHANNEL_CFG=ch_cfg=[[8,48,96],
                                [240,144,288],
                                [240,144,288]]

##############################################################################   
if __name__ == '__main__':

    if (torch.cuda.is_available()):
        print(torch.cuda.get_device_name(0))
    

     
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #dev = torch.device("cpu")
    device = torch.device(dev)
        

    modeltype='MultiSenseSeg'
    chindex='all20Ch'
    trainSetSize=5985
    fno=2
    fsiz=5
    miniBatchSize=8
    tsind,trind,vlind = CrossVal(trainSetSize,fno,fsiz)
    input_images, target_masks, trMeanR, trMeanG, trMeanB = get_images4(trainSetSize, fno, fsiz, tsind, trind, vlind, chindex)
             
    params = {'batch_size': miniBatchSize, 'shuffle': False}     
         
        
    test_set = satellitedata(input_images[tsind], target_masks[tsind])
    test_generator = DataLoader(test_set, **params)
        
    
    if modeltype=='UNetV2':
        net = UNetV2(classes=1).to(device)
    elif modeltype=='MultiSenseSeg':              
        net = MultiSenseSeg(n_classes=1, in_chans=(3, 3, 3),  n_branch=3).to(device)            
    elif modeltype=='MMVit4':              
        net = MMVit4().to(device)         
    elif modeltype=='MMVit2':              
        net = MMVit2().to(device)  
    elif modeltype=='MMVit1':              
        net = MMVit1().to(device)  
    elif modeltype=='MMVit5':              
        net = MMVit5().to(device)  
    elif modeltype=='RFNet':              
        net = RFNet().to(device) 
    elif modeltype=='RobustMseg':              
        net = RobustMseg().to(device)         
    elif modeltype=='mmformer':              
        net = mmformer().to(device)           
    elif modeltype == 'LoRA_ViT':
        model1 = ViT('B_16_imagenet1k')
    #model1.load_state_dict(torch.load('B_16_imagenet1k.pth'))           
        #model1.load_state_dict(torch.load('https://github.com/lukemelas/PyTorch-Pretrained-ViT/releases/download/0.0.2/B_16_imagenet1k.pth'))
        lora_model = LoRA_ViT(model1, r=4).to(device)
        net = SegWrapForViT(vit_model=lora_model, image_size=224,
                                    patches=16, dim=768, n_classes=1).to(device)
    elif modeltype == 'LoRA_ViT2':
        model1 = ViT('B_16_imagenet1k')
    #model1.load_state_dict(torch.load('B_16_imagenet1k.pth'))
        #model1.load_state_dict(torch.load('https://github.com/lukemelas/PyTorch-Pretrained-ViT/releases/download/0.0.2/B_16_imagenet1k.pth'))            
        # model = LoRA_ViT(model1, r=4).to(device)
        net = SegWrapForViT(vit_model=model1, image_size=224,
                                    patches=16, dim=768, n_classes=1).to(device)            
    elif modeltype == 'LoRA_ViT3':
        model1 = ViT('L_16_imagenet1k')
    #model1.load_state_dict(torch.load('B_16_imagenet1k.pth'))           
        #model1.load_state_dict(torch.load('https://github.com/lukemelas/PyTorch-Pretrained-ViT/releases/download/0.0.2/B_16_imagenet1k.pth'))
        lora_model = LoRA_ViT(model1, r=4).to(device)
        net = SegWrapForViT(vit_model=lora_model, image_size=224,
                                    patches=16, dim=1024, n_classes=1).to(device)
    elif modeltype == 'LoRA_ViT4':
        model1 = ViT('L_16_imagenet1k')
    #model1.load_state_dict(torch.load('B_16_imagenet1k.pth'))
        #model1.load_state_dict(torch.load('https://github.com/lukemelas/PyTorch-Pretrained-ViT/releases/download/0.0.2/B_16_imagenet1k.pth'))            
        # model = LoRA_ViT(model1, r=4).to(device)
        net = SegWrapForViT(vit_model=model1, image_size=224,
                                    patches=16, dim=1024, n_classes=1).to(device) 
    elif modeltype == 'LoRA_ViT5':
        model1 = ViT('B_16')
        lora_model = LoRA_ViT(model1, r=4).to(device)
        net = SegWrapForViT(vit_model=lora_model, image_size=224,
                                    patches=16, dim=768, n_classes=1).to(device)            
  
    
    first_image_batch = None
    
    # Iterate through the DataLoader to get the first batch
    for testim, testmas in test_generator:
        
        images=testim.to(device)
        masks=testmas.to(device)
        first_image_batch_for_input = images, masks
        masks = masks[:, 0, ...]  
        images = images[:, 0, ...]         
        first_image_batch = images, masks
        break  # Break the loop after obtaining the first batch
    
    # Extract the first image from the batch
    first_image = first_image_batch[0]
    first_image_for_input = first_image_batch_for_input[0]
       
    pathm = os.path.join("../../experiments/irem")
            
    net.load_state_dict(torch.load(os.path.join(pathm, "Finaliremmodel0.pt")))

    net.eval()
    
    flops, params = profile(net, inputs=(first_image_for_input,))
    flops, params = clever_format([flops, params], "%.2f")
    flops_value = float(flops[:-1])
    
    print(f"FLOPs: {flops_value} GLOPs")
    