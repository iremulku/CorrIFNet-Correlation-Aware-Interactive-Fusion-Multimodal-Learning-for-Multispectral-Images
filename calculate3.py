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
from F3_DATASET import satellitedata
from F4_TRAIN import train_model
from F6_CROSSVAL import CrossVal
from F7_TEST2 import test_model
from F8_IMAGES4 import get_images4
from F9_UNET_V2_3 import UNetV2
from mmmvit2 import MMVit2
from mmvit1 import MMVit1
from mmvit5 import MMVit5
from mmformer import mmformer
from RFNet import RFNet
from mmvit4 import MMVit4
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
        model = UNetV2(classes=1).to(device)
    if modeltype=='MultiSenseSeg':              
        model = MultiSenseSeg(n_classes=1, in_chans=(3, 3, 3),  n_branch=3 ).to(device)            
    if modeltype=='MMVit4':              
        model = MMVit4().to(device)         
    if modeltype=='MMVit2':              
        model = MMVit2().to(device)  
    if modeltype=='MMVit1':              
        model = MMVit1().to(device)  
    if modeltype=='MMVit5':              
        model = MMVit5().to(device)  
    if modeltype=='RFNet':              
        model = RFNet().to(device)  
    if modeltype=='RobustMseg':              
        model = RobustMseg().to(device)          
    if modeltype=='mmformer':              
        model = mmformer().to(device)          
    if modeltype == 'SegWrapForViT1':
        model1 = ViT('B_16_imagenet1k')
        lora_model = LoRA_ViT(model1, r=4).to(device)
        model = SegWrapForViT(vit_model=lora_model, image_size=224,
                                    patches=16, dim=768, n_classes=1).to(device)                
    if modeltype == 'SegWrapForViT2':
        model1 = ViT('B_16_imagenet1k')
        model = SegWrapForViT(vit_model=model1, image_size=224,
                                    patches=16, dim=768, n_classes=1).to(device)
    if modeltype == 'SegWrapForViT3':
        model1 = ViT('L_16_imagenet1k')
        # LoRA_ViT3
        lora_model = LoRA_ViT(model1, r=4).to(device)
        model = SegWrapForViT(vit_model=lora_model, image_size=224,
                                    patches=16, dim=1024, n_classes=1).to(device)
    if modeltype == 'SegWrapForViT4':
        model1 = ViT('L_16_imagenet1k')
        # LoRA_ViT4
        model = SegWrapForViT(vit_model=model1, image_size=224,
                                    patches=16, dim=1024, n_classes=1).to(device)       
    if modeltype == 'SegWrapForViT5':
        model1 = ViT('B_16')
        lora_model = LoRA_ViT(model1, r=4).to(device)
        model = SegWrapForViT(vit_model=lora_model, image_size=224,
                                    patches=16, dim=768, n_classes=1).to(device) 
    if modeltype == 'SegWrapForViT6':
        model1 = ViT('B_32_imagenet1k')
    #model1.load_state_dict(torch.load('B_16_imagenet1k.pth'))           
        #model1.load_state_dict(torch.load('https://github.com/lukemelas/PyTorch-Pretrained-ViT/releases/download/0.0.2/B_16_imagenet1k.pth'))
        lora_model = LoRA_ViT(model1, r=4).to(device)
        model = SegWrapForViT(vit_model=lora_model, image_size=224,
                                    patches=32, dim=3072, n_classes=1).to(device)
    if modeltype == 'SegWrapForViT7':
        model1 = ViT('B_32_imagenet1k')
    #model1.load_state_dict(torch.load('B_16_imagenet1k.pth'))
        #model1.load_state_dict(torch.load('https://github.com/lukemelas/PyTorch-Pretrained-ViT/releases/download/0.0.2/B_16_imagenet1k.pth'))            
        # model = LoRA_ViT(model1, r=4).to(device)
        model = SegWrapForViT(vit_model=model1, image_size=224,
                                    patches=32, dim=3072, n_classes=1).to(device)        
    if modeltype == 'SegWrapForViT':
        model1 = ViT('L_32_imagenet1k')
    #model1.load_state_dict(torch.load('B_16_imagenet1k.pth'))           
        #model1.load_state_dict(torch.load('https://github.com/lukemelas/PyTorch-Pretrained-ViT/releases/download/0.0.2/B_16_imagenet1k.pth'))
        lora_model = LoRA_ViT(model1, r=4).to(device)
        model = SegWrapForViT(vit_model=lora_model, image_size=224,
                                    patches=32, dim=1024, n_classes=1).to(device)
    if modeltype == 'SegWrapForViT9':
        model1 = ViT('L_32_imagenet1k')
    #model1.load_state_dict(torch.load('B_16_imagenet1k.pth'))
        #model1.load_state_dict(torch.load('https://github.com/lukemelas/PyTorch-Pretrained-ViT/releases/download/0.0.2/B_16_imagenet1k.pth'))            
        # model = LoRA_ViT(model1, r=4).to(device)
        model = SegWrapForViT(vit_model=model1, image_size=224,
                                    patches=32, dim=1024, n_classes=1).to(device)          
                              
  

    
    first_image_batch = None
    

    for testim, testmas in test_generator:    
        images=testim.to(device)
        masks=testmas.to(device)
        masks = masks[:, 0, ...]  
        images = images[:, 0, ...] 
        first_image_batch = images, masks
        break  
    

    first_image = first_image_batch[0]
       
    pathm = os.path.join("../../experiments/irem")            
    model.load_state_dict(torch.load(os.path.join(pathm, "Finaliremmodel0.pt")))

    # Calculate the total number of trainable parameters in the loaded model
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    total_params_million = total_params / 1_000_000

    print(f"Total trainable parameters: {total_params_million:.2f} M")
    #print(f"Total trainable parameters: {total_params}")

    


    


    
    
    

    
    
    
    

