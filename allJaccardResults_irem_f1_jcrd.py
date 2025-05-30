import os
import torch 
import numpy as np
from F6_CROSSVAL import CrossVal
from F8_IMAGES import get_images
from F8_IMAGES4 import get_images4
from F3_DATASET import satellitedata
from torch.utils.data import DataLoader
from F9_UNET_V2_3 import UNetV2 
from mmmvit2 import MMVit2
from mmvit1 import MMVit1
from mmvit5 import MMVit5
from mmvit4 import MMVit4
from RFNet import RFNet
from MultiSenseSeg import MultiSenseSeg
import timm
from lora import LoRA_ViT
from base_vit import ViT
from seg_vit import SegWrapForViT
from F5_JACCARD2 import Jaccard2, Jaccard, JaccardAndF1
import matplotlib.pyplot as plt
from F11_SEGPLOT2 import segplot

import warnings


class Config(object):
    NAME= "dfaNet"

    #set the output every STEP_PER_EPOCH iteration
    STEP_PER_EPOCH = 100
    ENCODER_CHANNEL_CFG=ch_cfg=[[8,48,96],
                                [240,144,288],
                                [240,144,288]]


warnings.filterwarnings("ignore")
createFigures = False
dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device(dev) 

#main
fout = open("iremf1.txt", "w")

with open('irem-input-dstl.txt') as f:
    contents = f.readlines()
prevline = ""
prevInputType = ""
for line in contents:        
    if line[0]=='C':
        # the folder
        modelPath = line.replace("\n","")
        liste = os.listdir(modelPath)
        # print(liste)
        
        # the model name (does it exist)
        FinalExists = False;
        for afile in liste:                
            if (afile[-3:-1] + afile[-1]) == ".pt":
                if afile[0:5] == "Final":
                    FinalExists = True;
                    break
        # if a final***.pt model exists get the model number in the end
        if FinalExists:
            if line[-3]!='l':
                modelName = "Finaliremmodel" + line[-3:-1] + ".pt" 
            else:
                modelName = "Finaliremmodel" + line[-2] + ".pt"
                
        else:
            modelName = prevline + ".pt"
        print(modelName)
        
        # find the log file
        logfile = ""
        for afile in liste:                
            if (afile[-3:-1] + afile[-1]) == "txt":
                logfile = afile 
                break
        with open(modelPath + "/" + logfile) as log:
            logs = log.readlines()
            # fold number 
            foldNo = int(logs[4][-2:-1])
            # the input modality
            inputType = logs[18][14:-1]
            # the model
            modelType = logs[21][0:-2]
            # print( modelType)
            # if logs[22][3:6] ==  "xce":
            #     modelType = "DeepLabv3_plusX"
            # if logs[22][3:6] ==  "res":
            #     if logs[216][0:6] == "Epoch:":
            #         modelType = "DeepLabv3_plus34"
            #     else:
            #         modelType = "DeepLabv3_plus101"
           
            #if ((inputType=='all20Ch') | (modelType=='SegNet') | (inputType=='NDVI_NDMI_NDWI_WRI_ARVI_SAVI')):
            # if (inputType=='RGBs'):
            #     continue
            
           
            
            figuresPath = "results/" + modelType + '_' + inputType + '_' + str(foldNo)
            if createFigures & ~os.path.isdir(figuresPath) :
                os.mkdir(figuresPath)
           
            ### now run the Jaccard ###                
            #first construct the model

            if modelType=='UNetV2':
                model = UNetV2(classes=1).to(device)
            if modelType=='MultiSenseSeg':              
                model = MultiSenseSeg(n_classes=1, in_chans=(3, 3, 3),  n_branch=3 ).to(device)                 
            if modelType=='MMVit2':              
                model = MMVit2().to(device)  
            if modelType=='MMVit1':              
                model = MMVit1().to(device)     
            if modelType=='MMVit5':              
                model = MMVit5().to(device) 
            if modelType=='MMVit4':              
                model = MMVit4().to(device)                 
            if modelType=='RFNet':              
                model = RFNet().to(device)                  
            if modelType == 'SegWrapForViT1':
                model1 = ViT('B_16_imagenet1k')
            #model1.load_state_dict(torch.load('B_16_imagenet1k.pth'))           
                #model1.load_state_dict(torch.load('https://github.com/lukemelas/PyTorch-Pretrained-ViT/releases/download/0.0.2/B_16_imagenet1k.pth'))
                lora_model = LoRA_ViT(model1, r=4).to(device)
                model = SegWrapForViT(vit_model=lora_model, image_size=224,
                                            patches=16, dim=768, n_classes=1).to(device)
            if modelType == 'SegWrapForViT2':
                model1 = ViT('B_16_imagenet1k')
            #model1.load_state_dict(torch.load('B_16_imagenet1k.pth'))
                #model1.load_state_dict(torch.load('https://github.com/lukemelas/PyTorch-Pretrained-ViT/releases/download/0.0.2/B_16_imagenet1k.pth'))            
                # model = LoRA_ViT(model1, r=4).to(device)
                model = SegWrapForViT(vit_model=model1, image_size=224,
                                            patches=16, dim=768, n_classes=1).to(device)            
            if modelType == 'SegWrapForViT':
                model1 = ViT('L_16_imagenet1k')
            #model1.load_state_dict(torch.load('B_16_imagenet1k.pth'))           
                #model1.load_state_dict(torch.load('https://github.com/lukemelas/PyTorch-Pretrained-ViT/releases/download/0.0.2/B_16_imagenet1k.pth'))
                lora_model = LoRA_ViT(model1, r=4).to(device)
                model = SegWrapForViT(vit_model=lora_model, image_size=224,
                                            patches=16, dim=1024, n_classes=1).to(device)
            if modelType == 'SegWrapForViT4':
                model1 = ViT('L_16_imagenet1k')
            #model1.load_state_dict(torch.load('B_16_imagenet1k.pth'))
                #model1.load_state_dict(torch.load('https://github.com/lukemelas/PyTorch-Pretrained-ViT/releases/download/0.0.2/B_16_imagenet1k.pth'))            
                # model = LoRA_ViT(model1, r=4).to(device)
                model = SegWrapForViT(vit_model=model1, image_size=224,
                                            patches=16, dim=1024, n_classes=1).to(device) 
            if modelType == 'SegWrapForViT5':
                model1 = ViT('B_16')
                lora_model = LoRA_ViT(model1, r=4).to(device)
                model = SegWrapForViT(vit_model=lora_model, image_size=224,
                                            patches=16, dim=768, n_classes=1).to(device)      
            if modelType == 'SegWrapForViT6':
                model1 = ViT('B_32_imagenet1k')
            #model1.load_state_dict(torch.load('B_16_imagenet1k.pth'))           
                #model1.load_state_dict(torch.load('https://github.com/lukemelas/PyTorch-Pretrained-ViT/releases/download/0.0.2/B_16_imagenet1k.pth'))
                lora_model = LoRA_ViT(model1, r=4).to(device)
                model = SegWrapForViT(vit_model=lora_model, image_size=224,
                                            patches=32, dim=768, n_classes=1).to(device)
            if modelType == 'SegWrapForViT7':
                model1 = ViT('B_32_imagenet1k')
            #model1.load_state_dict(torch.load('B_16_imagenet1k.pth'))
                #model1.load_state_dict(torch.load('https://github.com/lukemelas/PyTorch-Pretrained-ViT/releases/download/0.0.2/B_16_imagenet1k.pth'))            
                # model = LoRA_ViT(model1, r=4).to(device)
                model = SegWrapForViT(vit_model=model1, image_size=224,
                                            patches=32, dim=768, n_classes=1).to(device)        
                
            if modelType == 'SegWrapForViT8':
                model1 = ViT('L_32_imagenet1k')
            #model1.load_state_dict(torch.load('B_16_imagenet1k.pth'))           
                #model1.load_state_dict(torch.load('https://github.com/lukemelas/PyTorch-Pretrained-ViT/releases/download/0.0.2/B_16_imagenet1k.pth'))
                lora_model = LoRA_ViT(model1, r=4).to(device)
                model = SegWrapForViT(vit_model=lora_model, image_size=224,
                                            patches=32, dim=1024, n_classes=1).to(device)
            if modelType == 'SegWrapForViT9':
                model1 = ViT('L_32_imagenet1k')
            #model1.load_state_dict(torch.load('B_16_imagenet1k.pth'))
                #model1.load_state_dict(torch.load('https://github.com/lukemelas/PyTorch-Pretrained-ViT/releases/download/0.0.2/B_16_imagenet1k.pth'))            
                # model = LoRA_ViT(model1, r=4).to(device)
                model = SegWrapForViT(vit_model=model1, image_size=224,
                                            patches=32, dim=1024, n_classes=1).to(device)            
                

        
            # and load the model 
            #print(modelPath + "/" + modelName)               
            model.load_state_dict(torch.load(modelPath + "/" + modelName))
            #print("Keys expected by model:", model.state_dict().keys())
            model.eval()
            
            # load input (for DSTL and RIT18) 
            
            tsind,trind,vlind = CrossVal(5985,foldNo,5);
            input_images, target_masks, trMeanR, trMeanG, trMeanB = get_images4(5985, foldNo, 5, tsind, trind, vlind, inputType)


            params = {'batch_size': 1, 'shuffle': False}    
            test_set = satellitedata(input_images[tsind], target_masks[tsind])
            test_generator = DataLoader(test_set, **params)
            
            f1All = np.empty(test_generator.dataset.images.shape[0],dtype='float')
            jcrdsAll = np.empty(test_generator.dataset.images.shape[0],dtype='float')
            
            with torch.no_grad():
                ts = 0;
                for testim, testmas in test_generator:
                    # the model
                    images=testim.to(device)
                    masks=testmas.to(device)
                    outputs = model(images) 

                    masks = masks[:, 0, ...]   # Remove extra channel
                    outputs = outputs[:, 0, ...]                     

                    f1 = JaccardAndF1(torch.reshape(masks,(224*224,1)),torch.reshape(outputs,(224*224,1)))                                    
                    jcrd = Jaccard2(torch.reshape(masks,(224*224,1)),torch.reshape(outputs,(224*224,1)))
                    jcrdsAll[ts] = jcrd.to('cpu').numpy()[0]
                    f1All[ts] = f1.to('cpu').numpy()[0]
                    
                    if createFigures:
                        images2 = images[:, 0, ...]
                        fimage=images2[0].permute(1, 2, 0)
                        fimage[:,:,0]=(images2[0][0,:,:])
                        fimage[:,:,1]=(images2[0][1,:,:])
                        fimage[:,:,2]=(images2[0][2,:,:])
                        fimage=fimage.cpu().numpy()
                        foutput=outputs[0].permute(1, 2, 0)
                        foutput=foutput.cpu().numpy()
                        fmask=masks[0].permute(1, 2, 0)
                        fmask=fmask.cpu().numpy()
                        segplot(figuresPath, 224, fimage, foutput, fmask,  trMeanR, trMeanG, trMeanB, ts)
                   
                    ts = ts+1;      
            
            print(modelType + ", " + inputType + ", f1: ", f1All.mean() , "±" , f1All.std())
            print(modelType + ", " + inputType + ", Jaccard: ", jcrdsAll.mean() , "±" , jcrdsAll.std())
                        
    else:
        prevline = line[0:-1]
fout.close()
