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
#from F7_TEST import test_model
from F7_TEST2 import test_model
from F8_IMAGES import get_images
#from F9_UNET_V2_4 import UNetV2
from F9_UNET_V2_3 import UNetV2
from F10_SEGNET_V1 import SegNet
from F12_DLINKNET_V3 import DinkNet101 
from F20_DILATEDUNET import CamDUNet
from F15_DFANET import DFANet 
from F21_GENERAL_UNET import R2U_Net, AttU_Net, R2AttU_Net
from F22_NESTEDUNET import NestedUNet
from F23_DULANORM_UNET import DualNorm_Unet
from F24_INCEPTION_UNET import InceptionUNet
from F25_SCAG_UNET import AttU_Net_with_scAG
from F26_FSFNet import FSFNet
from F27_LMFFNet import LMFFNet
from LMFFNet_IREM import LMFFNet2
from LMFFNet_IREM3 import LMFFNet3
from F28_FASSDNet import FASSDNet
from F29_ENet import ENet
from F30_ELANet import ELANet
from F16_UNETFORMER2 import UNetFormer
#from HiFormer import HiFormer
import timm
from lora import LoRA_ViT
from base_vit import ViT
from seg_vit import SegWrapForViT
from F14_DEEPLABV3PLUS_V4_xception import DeepLabv3_plus
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
        

    modeltype='UNetFormer'
    chindex='RGBs'
    trainSetSize=5985
    fno=4
    fsiz=5
    miniBatchSize=8
    tsind,trind,vlind = CrossVal(trainSetSize,fno,fsiz)
    input_images, target_masks, trMeanR, trMeanG, trMeanB = get_images(trainSetSize, fno, fsiz, tsind, trind, vlind, chindex)
             
    params = {'batch_size': miniBatchSize, 'shuffle': False}     
         
        
    test_set = satellitedata(input_images[tsind], target_masks[tsind])
    test_generator = DataLoader(test_set, **params)
        
    
    if modeltype=='UNetV1':
        net = UNetV1(classes=1).to(device)
    elif modeltype=='UNetV2':
        net = UNetV2(classes=1).to(device) 
    elif modeltype=='SegNet':
        net = SegNet(classes=1).to(device) 
    elif modeltype=='DinkNet101':
        net =  DinkNet101(num_classes=1).to(device)        
    elif modeltype=='DeepLabv3_plus':
        net = DeepLabv3_plus(num_classes=1, small=True, pretrained=True).to(device) 
    elif modeltype=='CamDUNet':
       net = CamDUNet().to(device)
    elif modeltype=='DFANet':
       cfg=Config()
       net =  DFANet(cfg.ENCODER_CHANNEL_CFG,decoder_channel=64,num_classes=1).to(device)       
    elif modeltype=='R2U_Net':
       net = R2U_Net(img_ch=3,output_ch=1).to(device)        
    elif modeltype=='AttU_Net':
       net = AttU_Net(img_ch=3,output_ch=1).to(device)                
    elif modeltype=='R2AttU_Net':
       net = R2AttU_Net(img_ch=3,output_ch=1).to(device)   
    elif modeltype=='NestedUNet':
       net = NestedUNet(in_ch=3, out_ch=1).to(device)         
    elif modeltype=='DualNorm_Unet':
       net = DualNorm_Unet(n_channels=3, n_classes=1).to(device)       
    elif modeltype=='InceptionUNet':
       net = InceptionUNet(n_channels=3, n_classes=1, bilinear=True).to(device)       
    elif modeltype=='AttU_Net_with_scAG':
       net = AttU_Net_with_scAG(img_ch=3, output_ch=1,ratio=16).to(device)       
    elif modeltype=='FSFNet':
       net = FSFNet(num_classes=1).to(device)             
    elif modeltype=='LMFFNet':
       net = LMFFNet(classes=1, block_1=3, block_2=8) .to(device)
    elif modeltype=='LMFFNet2':
       net = LMFFNet2(classes=1, block_1=3, block_2=8) .to(device)
    elif modeltype=='LMFFNet3':
       net = LMFFNet3(classes=1, block_1=3, block_2=8) .to(device)       
    elif modeltype=='FASSDNet':
       net = FASSDNet(n_classes=1, alpha=2).to(device) 
    elif modeltype=='ENet':
       net = ENet(classes=1).to(device)  
    elif modeltype=='ELANet':  
        net = ELANet().to(device)   
    elif modeltype == 'UNetFormer':
        net = UNetFormer(decode_channels=64, #64
             dropout=0.1,
             backbone_name='swsl_resnet18', #resnet18
             pretrained=False, # was true
             window_size=4, #8 
             num_classes=1).to(device)
    # elif modeltype == 'HiFormer':
    #     model = HiFormer().to(device)
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
        first_image_batch = images, masks
        break  # Break the loop after obtaining the first batch
    
    # Extract the first image from the batch
    first_image = first_image_batch[0]
       
    pathm = os.path.join("../../experiments/irem")
            
    net.load_state_dict(torch.load(os.path.join(pathm, "Finaliremmodel0.pt")))

    net.eval()

    
    
    total_time = 0.0
    num_iterations = 1000
    for _ in range(num_iterations):

        start_time = time.time()        
        with torch.no_grad():
            _ = net(first_image)

        end_time = time.time()
        inference_time = end_time - start_time
        total_time += inference_time


    fps = num_iterations / total_time    
    print(f"FPS: {fps:.2f}")

    


    
    
    

    
    
    
    

