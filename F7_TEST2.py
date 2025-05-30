from __future__ import print_function
import torch 
import os
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from F5_JACCARD2 import Jaccard2
from F9_UNET_V2_3 import UNetV2
from mmmvit2 import MMVit2
from mmvit1 import MMVit1
from mmvit5 import MMVit5
from mmvit4 import MMVit4
from mmformer import mmformer
from RFNet import RFNet
from F11_SEGPLOT import segplot
from segformer import Segformer
from MultiSenseSeg import MultiSenseSeg
import timm
from lora import LoRA_ViT
from base_vit import ViT
from seg_vit import SegWrapForViT
import time
from F14_DEEPLABV3PLUS_V4_xception import DeepLabv3_plus

class Config(object):
    NAME= "dfaNet"

    #set the output every STEP_PER_EPOCH iteration
    STEP_PER_EPOCH = 100
    ENCODER_CHANNEL_CFG=ch_cfg=[[8,48,96],
                                [240,144,288],
                                [240,144,288]]


dev = "cuda:0"  
device = torch.device(dev) 

def test_model(test_generator, lim, testFile, testaccFile, i, modeltype, pathm, trMeanR, trMeanG, trMeanB):
    
    
    if modeltype=='UNetV2':
        net = UNetV2(classes=1).to(device)   
    elif modeltype=='MultiSenseSeg':              
        net = MultiSenseSeg(n_classes=1, in_chans=(3, 3, 3),  n_branch=3 ).to(device)             
    elif modeltype=='Segformer':              
        net = Segformer(num_classes=1).to(device)          
    elif modeltype=='MMVit2':              
        net = MMVit2().to(device)   
    elif modeltype=='MMVit1':              
        net = MMVit1().to(device) 
    elif modeltype=='MMVit5':              
        net = MMVit5().to(device)  
    elif modeltype=='MMVit4':              
        net = MMVit4().to(device)         
    elif modeltype=='mmformer':              
        net = mmformer().to(device) 
    elif modeltype=='RFNet':              
        net = RFNet().to(device)           
    elif modeltype=='DeepLabv3_plus':
        net = DeepLabv3_plus(num_classes=1, small=True, pretrained=True).to(device)          
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
    elif modeltype == 'LoRA_ViT6':
        model1 = ViT('B_32_imagenet1k')
    #model1.load_state_dict(torch.load('B_16_imagenet1k.pth'))           
        #model1.load_state_dict(torch.load('https://github.com/lukemelas/PyTorch-Pretrained-ViT/releases/download/0.0.2/B_16_imagenet1k.pth'))
        lora_model = LoRA_ViT(model1, r=4).to(device)
        net = SegWrapForViT(vit_model=lora_model, image_size=224,
                                    patches=32, dim=768, n_classes=1).to(device)
    elif modeltype == 'LoRA_ViT7':
        model1 = ViT('B_32_imagenet1k')
    #model1.load_state_dict(torch.load('B_16_imagenet1k.pth'))
        #model1.load_state_dict(torch.load('https://github.com/lukemelas/PyTorch-Pretrained-ViT/releases/download/0.0.2/B_16_imagenet1k.pth'))            
        # model = LoRA_ViT(model1, r=4).to(device)
        net = SegWrapForViT(vit_model=model1, image_size=224,
                                    patches=32, dim=768, n_classes=1).to(device)        
        
    elif modeltype == 'LoRA_ViT8':
        model1 = ViT('L_32_imagenet1k')
    #model1.load_state_dict(torch.load('B_16_imagenet1k.pth'))           
        #model1.load_state_dict(torch.load('https://github.com/lukemelas/PyTorch-Pretrained-ViT/releases/download/0.0.2/B_16_imagenet1k.pth'))
        lora_model = LoRA_ViT(model1, r=4).to(device)
        net = SegWrapForViT(vit_model=lora_model, image_size=224,
                                    patches=32, dim=1024, n_classes=1).to(device)
    elif modeltype == 'LoRA_ViT9':
        model1 = ViT('L_32_imagenet1k')
    #model1.load_state_dict(torch.load('B_16_imagenet1k.pth'))
        #model1.load_state_dict(torch.load('https://github.com/lukemelas/PyTorch-Pretrained-ViT/releases/download/0.0.2/B_16_imagenet1k.pth'))            
        # model = LoRA_ViT(model1, r=4).to(device)
        net = SegWrapForViT(vit_model=model1, image_size=224,
                                    patches=32, dim=1024, n_classes=1).to(device)     
       
 
        
    net.load_state_dict(torch.load(os.path.join(pathm, "Finaliremmodel{}.pt".format(i))))

    jI = 0
    totalBatches = 0
    test_losses = []
    net.eval()
    with torch.no_grad():
        t_losses = []
        t=0
        start_time = time.time()
        for testim, testmas in test_generator:
            images=testim.to(device)
            masks=testmas.to(device)
            outputs = net(images)
            if t==0:
                fig=plt.figure()
                axes=[]
                images2 = images[:, 0, ...]
                fimage=images2[0].permute(1, 2, 0)
                fimage[:,:,0]=(images2[0][0,:,:])
                fimage[:,:,1]=(images2[0][1,:,:])
                fimage[:,:,2]=(images2[0][2,:,:])
                fimage=fimage.cpu().numpy()
                axes.append(fig.add_subplot(1, 2, 1))
                outputs2 = outputs[:, 0, ...] 
                foutput=outputs2[0].permute(1, 2, 0)
                foutput=foutput.cpu().numpy()
                plt.imshow(np.squeeze(foutput, axis=2),  cmap='gray')
                subplot_title=("Test Predicted Mask")
                axes[-1].set_title(subplot_title)
                axes.append(fig.add_subplot(1, 2, 2))
                masks2 = masks[:, 0, ...]
                fmask=masks2[0].permute(1, 2, 0)
                fmask=fmask.cpu().numpy()
                plt.imshow(np.squeeze(fmask, axis=2),  cmap='gray')
                subplot_title=("Ground Truth Mask")
                axes[-1].set_title(subplot_title)
                n_curve = 'mask_comparison.png'
                plt.savefig(os.path.join(pathm, n_curve))
                plt.show()
                segplot(pathm, lim, fimage, foutput, fmask,  trMeanR, trMeanG, trMeanB)
            losst=nn.BCEWithLogitsLoss()
            output = losst(outputs, masks)
            t_losses.append(output.item())
            batchLoad = len(masks)*lim*lim
            totalBatches = totalBatches + batchLoad
            masks = masks[:, 0, ...]   # Remove extra channel
            outputs = outputs[:, 0, ...] 
            thisJac = Jaccard2(torch.reshape(masks,(batchLoad,1)),torch.reshape(outputs,(batchLoad,1)))*batchLoad
            jI = jI+thisJac.data[0]
            t+=1
   
    dn=jI/totalBatches
    dni=dn.item()
    test_loss = np.mean(t_losses)
    test_losses.append(test_loss)
    testFile.write(str(test_losses[0])+"\n")
    testaccFile.write(str(dni)+"\n")
    print("Test Jaccard:",dni)

