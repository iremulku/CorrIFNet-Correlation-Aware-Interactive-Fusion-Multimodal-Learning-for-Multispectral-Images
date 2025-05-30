import torch.nn as nn
import torch.nn.functional as F
import torch
import math


def normalization(planes, norm='bn'):
    if norm == 'bn':
        m = nn.BatchNorm3d(planes)
    elif norm == 'gn':
        m = nn.GroupNorm(4, planes)
    elif norm == 'in':
        m = nn.InstanceNorm3d(planes)
    else:
        raise ValueError('normalization type {} is not supported'.format(norm))
    return m

class general_conv3d(nn.Module):
    def __init__(self, in_ch, out_ch, k_size=3, stride=1, padding=1, pad_type='replicate', norm='in', is_training=True, act_type='lrelu', relufactor=0.2):
        super(general_conv3d, self).__init__()
        self.conv = nn.Conv3d(in_channels=in_ch, out_channels=out_ch, kernel_size=k_size, stride=stride, padding=padding, padding_mode=pad_type, bias=True)

        self.norm = normalization(out_ch, norm=norm)
        if act_type == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif act_type == 'lrelu':
            self.activation = nn.LeakyReLU(negative_slope=relufactor, inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x
        
class prm_generator_laststage(nn.Module):
    def __init__(self, in_channel=16, norm='in', num_cls=1):
        super(prm_generator_laststage, self).__init__()

        self.embedding_layer = nn.Sequential(
                            general_conv3d(in_channel * 3, int(in_channel//4), k_size=1, padding=0, stride=1),
                            general_conv3d(int(in_channel//4), int(in_channel//4), k_size=3, padding=1, stride=1),
                            general_conv3d(int(in_channel//4), in_channel, k_size=1, padding=0, stride=1))
        
         

        self.prm_layer = nn.Sequential(
                            general_conv3d(in_channel, 16, k_size=1, stride=1, padding=0),
                            nn.Conv3d(16, num_cls, kernel_size=1, padding=0, stride=1, bias=True),
                            nn.Softmax(dim=1))

    def forward(self, x):
        B, K, C, H, W, Z = x.size()
        y = x.view(B, -1, H, W, Z)
        z=self.embedding_layer(y)
        seg = self.prm_layer(z)
        return seg        

class prm_generator(nn.Module):
    def __init__(self, in_channel=64, norm='in', num_cls=1):
        super(prm_generator, self).__init__()

        self.embedding_layer = nn.Sequential(
                            general_conv3d(in_channel*3, int(in_channel//4), k_size=1, padding=0, stride=1),
                            general_conv3d(int(in_channel//4), int(in_channel//4), k_size=3, padding=1, stride=1),
                            general_conv3d(int(in_channel//4), in_channel, k_size=1, padding=0, stride=1))


        self.prm_layer = nn.Sequential(
                            general_conv3d(in_channel*2, 16, k_size=1, stride=1, padding=0),
                            nn.Conv3d(16, num_cls, kernel_size=1, padding=0, stride=1, bias=True),
                            nn.Softmax(dim=1))

    def forward(self, x1, x2):
        B, K, C, H, W, Z = x2.size()
        y = x2.view(B, -1, H, W, Z)
        emb = self.embedding_layer(y) 
        emb_up = F.interpolate(emb, size=x1.shape[2:], mode='trilinear', align_corners=True)
        seg = self.prm_layer(torch.cat((x1, emb_up), dim=1))
        return seg     
        
class modal_fusion(nn.Module):
    def __init__(self, in_channel=64, num_modalities=3):
        super(modal_fusion, self).__init__()
        # feat_avg: [B, C*K, 1,1,1] + 1 kanȧl → toplam C*K+1 girdi kanalı
        self.weight_layer = nn.Sequential(
            nn.Conv3d(in_channel * num_modalities + 1, 128, kernel_size=1, padding=0, bias=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            # çıktı olarak K adet ağırlık
            nn.Conv3d(128, num_modalities, kernel_size=1, padding=0, bias=True)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, prm, region_name):
        B, K, C, H, W, Z = x.size()

        prm_avg = torch.mean(prm, dim=(3,4,5), keepdim=False) + 1e-7
        feat_avg = torch.mean(x, dim=(3,4,5), keepdim=False) / prm_avg

        feat_avg = feat_avg.view(B, K*C, 1, 1, 1)
        feat_avg = torch.cat((feat_avg, prm_avg[:, 0, 0, ...].view(B, 1, 1, 1, 1)), dim=1)
        weight = torch.reshape(self.weight_layer(feat_avg), (B, K, 1))
        weight = self.sigmoid(weight).view(B, K, 1, 1, 1, 1)

        ###we find directly using weighted sum still achieve competing performance
        region_feat = torch.sum(x * weight, dim=1)
        return region_feat

###fuse region feature
class region_fusion(nn.Module):
    def __init__(self, in_channel=32, num_cls=1):
        super(region_fusion, self).__init__()
        self.fusion_layer = nn.Sequential(
                        general_conv3d(in_channel*num_cls, in_channel, k_size=1, padding=0, stride=1),
                        general_conv3d(in_channel, in_channel, k_size=3, padding=1, stride=1),
                        general_conv3d(in_channel, in_channel//2, k_size=1, padding=0, stride=1))

    def forward(self, x):
        B, C_in, H, W, Z = x.size()
        x = torch.reshape(x, (B, -1, H, W, Z))
        return self.fusion_layer(x)        

class region_aware_modal_fusion(nn.Module):
    def __init__(self, in_channel=64, norm='in', num_cls=1):
        super(region_aware_modal_fusion, self).__init__()
        self.num_cls = num_cls

        self.modal_fusion = modal_fusion(in_channel=in_channel)  # Directly define modal_fusion
        self.region_fusion = region_fusion(in_channel=in_channel, num_cls=num_cls)
        self.short_cut = nn.Sequential(
            general_conv3d(in_channel*3, in_channel, k_size=1, padding=0, stride=1),
            general_conv3d(in_channel, in_channel, k_size=3, padding=1, stride=1),
            general_conv3d(in_channel, in_channel//2, k_size=1, padding=0, stride=1)
        )

    def forward(self, x, prm):
        B, K, C, H, W, Z = x.size()

        prm = torch.unsqueeze(prm, 2).repeat(1, 1, C, 1, 1, 1)
        
        # Split the modal features
        RGB = x[:, 0:1, :, :, :] * prm
        NIR = x[:, 1:2, :, :, :] * prm
        SWIR = x[:, 2:3, :, :, :] * prm
        
        # Combine modal features
        modal_feat = torch.cat((RGB, NIR, SWIR), dim=1)

        # Perform modal fusion (no need for region splitting)
        region_fused_feat = self.modal_fusion(modal_feat, prm[:, 0:1, ...], 'RGB')

        # Final feature combination
        final_feat = torch.cat((self.region_fusion(region_fused_feat), self.short_cut(x.view(B, -1, H, W, Z))), dim=1)

        return final_feat   

basic_dims = 8
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.e1_c1 = general_conv3d(1, basic_dims, pad_type='replicate')
        self.e1_c2 = general_conv3d(basic_dims, basic_dims, pad_type='replicate')
        self.e1_c3 = general_conv3d(basic_dims, basic_dims, pad_type='replicate')

        self.e2_c1 = general_conv3d(basic_dims, basic_dims*2, stride=2, pad_type='replicate')
        self.e2_c2 = general_conv3d(basic_dims*2, basic_dims*2, pad_type='replicate')
        self.e2_c3 = general_conv3d(basic_dims*2, basic_dims*2, pad_type='replicate')

        self.e3_c1 = general_conv3d(basic_dims*2, basic_dims*4, stride=2, pad_type='replicate')
        self.e3_c2 = general_conv3d(basic_dims*4, basic_dims*4, pad_type='replicate')
        self.e3_c3 = general_conv3d(basic_dims*4, basic_dims*4, pad_type='replicate')

        self.e4_c1 = general_conv3d(basic_dims*4, basic_dims*8, stride=2, pad_type='replicate')
        self.e4_c2 = general_conv3d(basic_dims*8, basic_dims*8, pad_type='replicate')
        self.e4_c3 = general_conv3d(basic_dims*8, basic_dims*8, pad_type='replicate')

    def forward(self, x):
        x1 = self.e1_c1(x)
        x1 = x1 + self.e1_c3(self.e1_c2(x1))

        x2 = self.e2_c1(x1)
        x2 = x2 + self.e2_c3(self.e2_c2(x2))

        x3 = self.e3_c1(x2)
        x3 = x3 + self.e3_c3(self.e3_c2(x3))

        x4 = self.e4_c1(x3)
        x4 = x4 + self.e4_c3(self.e4_c2(x4))

        return x1, x2, x3, x4


class sigmoidOut(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(sigmoidOut, self).__init__()
        self.outsig = nn.Sigmoid()
    def forward(self, x):
        return self.outsig(x)        

class Decoder_fuse(nn.Module):
    def __init__(self, num_cls=1):
        super(Decoder_fuse, self).__init__()

        self.d3_c1 = general_conv3d(basic_dims*8, basic_dims*4, pad_type='replicate')
        self.d3_c2 = general_conv3d(basic_dims*8, basic_dims*4, pad_type='replicate')
        self.d3_out = general_conv3d(basic_dims*4, basic_dims*4, k_size=1, padding=0, pad_type='replicate')

        self.d2_c1 = general_conv3d(basic_dims*4, basic_dims*2, pad_type='replicate')
        self.d2_c2 = general_conv3d(basic_dims*4, basic_dims*2, pad_type='replicate')
        self.d2_out = general_conv3d(basic_dims*2, basic_dims*2, k_size=1, padding=0, pad_type='replicate')

        self.d1_c1 = general_conv3d(basic_dims*2, basic_dims, pad_type='replicate')
        self.d1_c2 = general_conv3d(basic_dims*2, basic_dims, pad_type='replicate')
        self.d1_out = general_conv3d(basic_dims, basic_dims, k_size=1, padding=0, pad_type='replicate')

        self.seg_layer = nn.Conv3d(in_channels=basic_dims, out_channels=num_cls, kernel_size=1, stride=1, padding=0, bias=True)

        self.up2 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.up4 = nn.Upsample(scale_factor=4, mode='trilinear', align_corners=True)
        self.up8 = nn.Upsample(scale_factor=8, mode='trilinear', align_corners=True)

        self.RFM4 = region_aware_modal_fusion(in_channel=basic_dims*8, num_cls=num_cls)
        self.RFM3 = region_aware_modal_fusion(in_channel=basic_dims*4, num_cls=num_cls)
        self.RFM2 = region_aware_modal_fusion(in_channel=basic_dims*2, num_cls=num_cls)
        self.RFM1 = region_aware_modal_fusion(in_channel=basic_dims*1, num_cls=num_cls)

        self.prm_generator4 = prm_generator_laststage(in_channel=basic_dims*8, num_cls=num_cls)
        self.prm_generator3 = prm_generator(in_channel=basic_dims*4, num_cls=num_cls)
        self.prm_generator2 = prm_generator(in_channel=basic_dims*2, num_cls=num_cls)
        self.prm_generator1 = prm_generator(in_channel=basic_dims*1, num_cls=num_cls)
        self.outsig = sigmoidOut(1, 1)

    def forward(self, x1, x2, x3, x4):


        prm_pred4 = self.prm_generator4(x4)
        de_x4 = self.RFM4(x4, prm_pred4.detach())
        de_x4 = F.interpolate(de_x4, (16, 16, 16))
        de_x4 = self.d3_c1(self.up2(de_x4))

        prm_pred3 = self.prm_generator3(de_x4, x3)
        prm = prm_pred3.detach()
        prm = F.interpolate(prm, size=x3.shape[3:],    
                    mode='trilinear',
                    align_corners=True)
        de_x3 = self.RFM3(x3, prm)
        de_x3 = F.interpolate(de_x3, (32, 32, 32))
        de_x3 = torch.cat((de_x3, de_x4), dim=1)
        de_x3 = self.d3_out(self.d3_c2(de_x3))
        de_x3 = self.d2_c1(self.up2(de_x3))

        prm_pred2 = self.prm_generator2(de_x3, x2)
        prm2 = prm_pred2.detach()
        prm2 = F.interpolate(prm2, 
                             size=x2.shape[3:],    
                             mode='trilinear', 
                             align_corners=True)    

        de_x2 = self.RFM2(x2, prm2)
        de_x2 = F.interpolate(de_x2, (64, 64, 64))
        de_x2 = torch.cat((de_x2, de_x3), dim=1)
        de_x2 = self.d2_out(self.d2_c2(de_x2))
        de_x2 = self.d1_c1(self.up2(de_x2))

        prm_pred1 = self.prm_generator1(de_x2, x1)
        prm1 = prm_pred1.detach()
        prm1 = F.interpolate(prm1, 
                             size=x1.shape[3:],   
                             mode='trilinear', 
                             align_corners=True)
        de_x1 = self.RFM1(x1, prm1)
        de_x1 = F.interpolate(de_x1, (128, 128, 128))
        de_x1 = torch.cat((de_x1, de_x2), dim=1)
        de_x1 = self.d1_out(self.d1_c2(de_x1))

        logits = self.seg_layer(de_x1)
        logits = F.interpolate(logits, 
                            size=(1, 224, 224),   
                            mode='trilinear', 
                            align_corners=True)
        pred = self.outsig(logits)
        pred = pred.repeat(1, 3, 1, 1, 1)

        return pred


class RFNet(nn.Module):
    def __init__(self, num_cls=1):
        super(RFNet, self).__init__()
        self.RGB_encoder = Encoder()
        self.NIR_encoder = Encoder()
        self.SWIR_encoder = Encoder()

        self.decoder_fuse = Decoder_fuse(num_cls=num_cls)

        self.is_training = False

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                torch.nn.init.kaiming_normal_(m.weight) #

    def forward(self, x):
        #extract feature from different layers        
        RGB_x1, RGB_x2, RGB_x3, RGB_x4 = self.RGB_encoder(x[:, 0:1, :, :, :])
        NIR_x1, NIR_x2, NIR_x3, NIR_x4 = self.NIR_encoder(x[:, 1:2, :, :, :])
        SWIR_x1, SWIR_x2, SWIR_x3, SWIR_x4 = self.SWIR_encoder(x[:, 2:3, :, :, :])
        
        
        x1 = torch.stack((RGB_x1, NIR_x1, SWIR_x1), dim=1)
        x2 = torch.stack((RGB_x2, NIR_x2, SWIR_x2), dim=1)
        x3 = torch.stack((RGB_x3, NIR_x3, SWIR_x3), dim=1)     
        x4 = torch.stack((RGB_x4, NIR_x4, SWIR_x4), dim=1)   
         

        fuse_pred = self.decoder_fuse(x1, x2, x3, x4)

        return fuse_pred