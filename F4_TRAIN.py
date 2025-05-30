from __future__ import print_function
import os
import torch 
import torch.nn as nn
import numpy as np
from F5_JACCARD2 import Jaccard2
from F9_UNET_V2_3 import UNetV2
from mmmvit2 import MMVit2
from mmvit1 import MMVit1
from mmvit5 import MMVit5
from mmvit4 import MMVit4
from mmformer import mmformer
from RFNet import RFNet
from segformer import Segformer
from MultiSenseSeg import MultiSenseSeg
import timm
from lora import LoRA_ViT
from base_vit import ViT
from seg_vit import SegWrapForViT
from F14_DEEPLABV3PLUS_V4_xception import DeepLabv3_plus



class Config(object):
    NAME= "dfaNet"

    #set the output every STEP_PER_EPOCH iteration
    STEP_PER_EPOCH = 100
    ENCODER_CHANNEL_CFG=ch_cfg=[[8,48,96],
                                [240,144,288],
                                [240,144,288]]


dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#dev = torch.device("cpu")
device = torch.device(dev) 


def train_model(n_epochs, trainloss, validationloss, accuracy, model, scheduler, lrFile, training_generator, optim, lim, trainFile, trainaccFile, trainepochFile, validation_generator, valFile, valaccFile, pathm, i, modeltype):
    training_losses = []
    for epoch in range(n_epochs):
        model.train()
        batch_losses = []
        jI = 0
        totalBatches = 0
        scheduler.step()
        print('Epoch:', epoch,'LR:', scheduler.get_lr())
        lrFile.write('Epoch:'+' '+str(epoch)+' '+'LR:'+' '+str(scheduler.get_lr())+"\n")
        lrFile.write(str(scheduler.state_dict())+"\n")

        mb=0
        for trainim, trainmas in training_generator:
            mb+=1
            optim.zero_grad()
            images=trainim.to(device)
            masks=trainmas.to(device)
            outputs=model(images)
            if trainloss =='BCEWithLogitsLoss':
                loss=nn.BCEWithLogitsLoss()
                output = loss(outputs, masks)            
            output.backward()
            optim.step()
                        
            batch_losses.append(output.item())
            batchLoad = len(masks)*lim*lim
            totalBatches = totalBatches + batchLoad
            if accuracy == 'Jaccard':
                masks = masks[:, 0, ...]   # Remove extra channel
                outputs = outputs[:, 0, ...] 
                thisJac = Jaccard2(torch.reshape(masks,(batchLoad,1)),torch.reshape(outputs,(batchLoad,1)))*batchLoad
                jI = jI+thisJac.data[0]
         
            
        training_loss = np.mean(batch_losses)
        training_losses.append(training_loss)
        trainFile.write(str(training_losses[epoch])+"\n")
        trainaccFile.write(str((jI/totalBatches).item())+"\n")
        trainepochFile.write(str(epoch)+"\n")
        print("Training Jaccard:",(jI/totalBatches).item()," (epoch:",epoch,")")
        lrFile.write("Training loss:"+str(training_losses[epoch])+"\n")
        lrFile.write("Training accuracy:"+str((jI/totalBatches).item())+"\n")
        
        
        torch.save(model.state_dict(), os.path.join(pathm, "iremmodel{}.pt".format(i)))
        validate(validationloss, accuracy, validation_generator, valFile, valaccFile, lim, lrFile, pathm, i, modeltype)
    torch.save(model.state_dict(), os.path.join(pathm, "Finaliremmodel{}.pt".format(i)))        
        
                
        
def validate(validationloss, accuracy, validation_generator, valFile, valaccFile, lim, lrFile, pathm, i, modeltype):
    jI = 0
    totalBatches = 0
    validation_losses = []
    
    
    if modeltype=='UNetV2':
        model = UNetV2(classes=1).to(device)     
    elif modeltype=='MultiSenseSeg':              
        model = MultiSenseSeg(n_classes=1, in_chans=(3, 3, 3),  n_branch=3 ).to(device)             
    elif modeltype=='Segformer':              
        model = Segformer(num_classes=1).to(device)          
    elif modeltype=='MMVit2':              
        model = MMVit2().to(device)   
    elif modeltype=='MMVit1':              
        model = MMVit1().to(device) 
    elif modeltype=='MMVit5':              
        model = MMVit5().to(device)   
    elif modeltype=='MMVit4':              
        model = MMVit4().to(device)         
    elif modeltype=='mmformer':              
        model = mmformer().to(device) 
    elif modeltype=='RFNet':              
        model = RFNet().to(device)           
    elif modeltype=='DeepLabv3_plus':
        model = DeepLabv3_plus(num_classes=1, small=True, pretrained=True).to(device)          
    elif modeltype == 'LoRA_ViT':
        model1 = ViT('B_16_imagenet1k')
    #model1.load_state_dict(torch.load('B_16_imagenet1k.pth'))           
        #model1.load_state_dict(torch.load('https://github.com/lukemelas/PyTorch-Pretrained-ViT/releases/download/0.0.2/B_16_imagenet1k.pth'))
        lora_model = LoRA_ViT(model1, r=4).to(device)
        model = SegWrapForViT(vit_model=lora_model, image_size=224,
                                    patches=16, dim=768, n_classes=1).to(device)
    elif modeltype == 'LoRA_ViT2':
        model1 = ViT('B_16_imagenet1k')
    #model1.load_state_dict(torch.load('B_16_imagenet1k.pth'))
        #model1.load_state_dict(torch.load('https://github.com/lukemelas/PyTorch-Pretrained-ViT/releases/download/0.0.2/B_16_imagenet1k.pth'))            
        # model = LoRA_ViT(model1, r=4).to(device)
        model = SegWrapForViT(vit_model=model1, image_size=224,
                                    patches=16, dim=768, n_classes=1).to(device)            
    elif modeltype == 'LoRA_ViT3':
        model1 = ViT('L_16_imagenet1k')
    #model1.load_state_dict(torch.load('B_16_imagenet1k.pth'))           
        #model1.load_state_dict(torch.load('https://github.com/lukemelas/PyTorch-Pretrained-ViT/releases/download/0.0.2/B_16_imagenet1k.pth'))
        lora_model = LoRA_ViT(model1, r=4).to(device)
        model = SegWrapForViT(vit_model=lora_model, image_size=224,
                                    patches=16, dim=1024, n_classes=1).to(device)
    elif modeltype == 'LoRA_ViT4':
        model1 = ViT('L_16_imagenet1k')
    #model1.load_state_dict(torch.load('B_16_imagenet1k.pth'))
        #model1.load_state_dict(torch.load('https://github.com/lukemelas/PyTorch-Pretrained-ViT/releases/download/0.0.2/B_16_imagenet1k.pth'))            
        # model = LoRA_ViT(model1, r=4).to(device)
        model = SegWrapForViT(vit_model=model1, image_size=224,
                                    patches=16, dim=1024, n_classes=1).to(device) 
    elif modeltype == 'LoRA_ViT5':
        model1 = ViT('B_16')
        lora_model = LoRA_ViT(model1, r=4).to(device)
        model = SegWrapForViT(vit_model=lora_model, image_size=224,
                                    patches=16, dim=768, n_classes=1).to(device)      
    elif modeltype == 'LoRA_ViT6':
        model1 = ViT('B_32_imagenet1k')
    #model1.load_state_dict(torch.load('B_16_imagenet1k.pth'))           
        #model1.load_state_dict(torch.load('https://github.com/lukemelas/PyTorch-Pretrained-ViT/releases/download/0.0.2/B_16_imagenet1k.pth'))
        lora_model = LoRA_ViT(model1, r=4).to(device)
        model = SegWrapForViT(vit_model=lora_model, image_size=224,
                                    patches=32, dim=768, n_classes=1).to(device)
    elif modeltype == 'LoRA_ViT7':
        model1 = ViT('B_32_imagenet1k')
    #model1.load_state_dict(torch.load('B_16_imagenet1k.pth'))
        #model1.load_state_dict(torch.load('https://github.com/lukemelas/PyTorch-Pretrained-ViT/releases/download/0.0.2/B_16_imagenet1k.pth'))            
        # model = LoRA_ViT(model1, r=4).to(device)
        model = SegWrapForViT(vit_model=model1, image_size=224,
                                    patches=32, dim=768, n_classes=1).to(device)        
        
    elif modeltype == 'LoRA_ViT8':
        model1 = ViT('L_32_imagenet1k')
    #model1.load_state_dict(torch.load('B_16_imagenet1k.pth'))           
        #model1.load_state_dict(torch.load('https://github.com/lukemelas/PyTorch-Pretrained-ViT/releases/download/0.0.2/B_16_imagenet1k.pth'))
        lora_model = LoRA_ViT(model1, r=4).to(device)
        model = SegWrapForViT(vit_model=lora_model, image_size=224,
                                    patches=32, dim=1024, n_classes=1).to(device)
    elif modeltype == 'LoRA_ViT9':
        model1 = ViT('L_32_imagenet1k')
    #model1.load_state_dict(torch.load('B_16_imagenet1k.pth'))
        #model1.load_state_dict(torch.load('https://github.com/lukemelas/PyTorch-Pretrained-ViT/releases/download/0.0.2/B_16_imagenet1k.pth'))            
        # model = LoRA_ViT(model1, r=4).to(device)
        model = SegWrapForViT(vit_model=model1, image_size=224,
                                    patches=32, dim=1024, n_classes=1).to(device)      


    model.load_state_dict(torch.load(os.path.join(pathm, "iremmodel{}.pt".format(i))))
    model.eval()
    with torch.no_grad():
        val_losses = []
        for valim, valmas in validation_generator:
            #model.eval()
            images=valim.to(device)
            masks=valmas.to(device)
            outputs=model(images)
            if validationloss == 'BCEWithLogitsLoss':
                loss=nn.BCEWithLogitsLoss()
                output = loss(outputs, masks)
            val_losses.append(output.item())
            batchLoad = len(masks)*lim*lim
            totalBatches = totalBatches + batchLoad
            if accuracy == 'Jaccard':
                masks = masks[:, 0, ...]   # Remove extra channel
                outputs = outputs[:, 0, ...] 
                thisJac = Jaccard2(torch.reshape(masks,(batchLoad,1)),torch.reshape(outputs,(batchLoad,1)))*batchLoad
                jI = jI+thisJac.data[0] 
    dn=jI/totalBatches
    dni=dn.item()
    validation_loss = np.mean(val_losses)
    validation_losses.append(validation_loss)
    valFile.write(str(validation_losses[0])+"\n")
    valaccFile.write(str(dni)+"\n")
    print("Validation Jaccard:",dni)
    lrFile.write("Validation loss:"+str(validation_losses[0])+"\n")
    lrFile.write("Validation accuracy:"+str(dni)+"\n")
