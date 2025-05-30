import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, constant_
import torch
import math
import numpy as np

basic_dims = 8
transformer_basic_dims = 512
mlp_dim = 512
num_heads = 8
depth = 1
num_modals = 3
patch_size = 8

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
    
class general_conv3d_prenorm(nn.Module):
    def __init__(self, in_ch, out_ch, k_size=3, stride=1, padding=1, pad_type='zeros', norm='in', act_type='relu', relufactor=0.2):
        super(general_conv3d_prenorm, self).__init__()
        self.conv = nn.Conv3d(in_channels=in_ch, out_channels=out_ch, kernel_size=k_size, stride=stride, padding=padding, padding_mode=pad_type, bias=True)

        self.norm = normalization(out_ch, norm=norm)
        if act_type == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif act_type == 'lrelu':
            self.activation = nn.LeakyReLU(negative_slope=relufactor, inplace=True)


    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        x = self.norm(x)
        return x   

class fusion_prenorm(nn.Module):
    def __init__(self, in_channel=64, num_cls=1):
        super(fusion_prenorm, self).__init__()
        self.fusion_layer = nn.Sequential(
                        general_conv3d_prenorm(in_channel*num_cls, in_channel, k_size=1, padding=0, stride=1),
                        general_conv3d_prenorm(in_channel, in_channel, k_size=3, padding=1, stride=1),
                        general_conv3d_prenorm(in_channel, in_channel, k_size=1, padding=0, stride=1))

    def forward(self, x):
        return self.fusion_layer(x)


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.e1_c1 = nn.Conv3d(in_channels=1, out_channels=basic_dims, kernel_size=3, stride=1, padding=1, padding_mode='replicate', bias=True)
        self.e1_c2 = general_conv3d_prenorm(basic_dims, basic_dims, pad_type='replicate')
        self.e1_c3 = general_conv3d_prenorm(basic_dims, basic_dims, pad_type='replicate')

        self.e2_c1 = general_conv3d_prenorm(basic_dims, basic_dims*2, stride=2, pad_type='replicate')
        self.e2_c2 = general_conv3d_prenorm(basic_dims*2, basic_dims*2, pad_type='replicate')
        self.e2_c3 = general_conv3d_prenorm(basic_dims*2, basic_dims*2, pad_type='replicate')

        self.e3_c1 = general_conv3d_prenorm(basic_dims*2, basic_dims*4, stride=2,  pad_type='replicate')
        self.e3_c2 = general_conv3d_prenorm(basic_dims*4, basic_dims*4, pad_type='replicate')
        self.e3_c3 = general_conv3d_prenorm(basic_dims*4, basic_dims*4, pad_type='replicate')

        self.e4_c1 = general_conv3d_prenorm(basic_dims*4, basic_dims*8, stride=2, pad_type='replicate')
        self.e4_c2 = general_conv3d_prenorm(basic_dims*8, basic_dims*8, pad_type='replicate')
        self.e4_c3 = general_conv3d_prenorm(basic_dims*8, basic_dims*8, pad_type='replicate')

        self.e5_c1 = general_conv3d_prenorm(basic_dims*8, basic_dims*8, stride=2,  pad_type='replicate')
        self.e5_c2 = general_conv3d_prenorm(basic_dims*8, basic_dims*8, pad_type='replicate')
        self.e5_c3 = general_conv3d_prenorm(basic_dims*8, basic_dims*8, pad_type='replicate')
        self.conv = nn.Conv3d(in_channels=basic_dims*23, out_channels=basic_dims*8, kernel_size=1, stride=1, padding=0, padding_mode='replicate', bias=True)

    def forward(self, x):
        x1 = self.e1_c1(x)
        x1 = x1 + self.e1_c3(self.e1_c2(x1))

        x2 = self.e2_c1(x1)
        x2 = x2 + self.e2_c3(self.e2_c2(x2))

        x3 = self.e3_c1(x2)
        x3 = x3 + self.e3_c3(self.e3_c2(x3))

        x4 = self.e4_c1(x3)
        x4 = x4 + self.e4_c3(self.e4_c2(x4))

        x5 = self.e5_c1(x4)
        x5 = x5 + self.e5_c3(self.e5_c2(x5))
        x1_ = F.interpolate(x1, (8, 8, 8))
        x2_ = F.interpolate(x2, (8, 8, 8))
        x3_ = F.interpolate(x3, (8, 8, 8))
        x4_ = F.interpolate(x4, (8, 8, 8))
        x5_ = F.interpolate(x5, (8, 8, 8))
        x6 = torch.cat([x1_, x2_, x3_, x4_,x5_],dim=1)
        x6 = self.conv(x6)
        return x1, x2, x3, x4, x5, x6


class sigmoidOut(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(sigmoidOut, self).__init__()
        self.outsig= nn.Sigmoid()

    def forward(self, x):
        x = self.outsig(x)
        return x  




class Decoder_fuse(nn.Module):
    def __init__(self, num_cls=1):
        super(Decoder_fuse, self).__init__()

        self.d4_c1 = general_conv3d_prenorm(basic_dims*24, basic_dims*16, pad_type='replicate')
        self.d4_c2 = general_conv3d_prenorm(basic_dims*40, basic_dims*8, pad_type='replicate')
        self.d4_out = general_conv3d_prenorm(basic_dims*8, basic_dims*8, k_size=1, padding=0, pad_type='replicate')

        self.d3_c1 = general_conv3d_prenorm(basic_dims*8, basic_dims*4, pad_type='replicate')
        self.d3_c2 = general_conv3d_prenorm(basic_dims*16, basic_dims*4, pad_type='replicate')
        self.d3_out = general_conv3d_prenorm(basic_dims*4, basic_dims*4, k_size=1, padding=0, pad_type='replicate')

        self.d2_c1 = general_conv3d_prenorm(basic_dims*4, basic_dims*2, pad_type='replicate')
        self.d2_c2 = general_conv3d_prenorm(basic_dims*8, basic_dims*2, pad_type='replicate')
        self.d2_out = general_conv3d_prenorm(basic_dims*2, basic_dims*2, k_size=1, padding=0, pad_type='replicate')

        self.d1_c1 = general_conv3d_prenorm(basic_dims*2, basic_dims, pad_type='replicate')
        self.d1_c2 = general_conv3d_prenorm(basic_dims*4, basic_dims, pad_type='replicate')
        self.d1_out = general_conv3d_prenorm(basic_dims, basic_dims, k_size=1, padding=0, pad_type='replicate')

        self.seg_d4 = nn.Conv3d(in_channels=basic_dims*8, out_channels=num_cls, kernel_size=1, stride=1, padding=0, bias=True)
        self.seg_d3 = nn.Conv3d(in_channels=basic_dims*8, out_channels=num_cls, kernel_size=1, stride=1, padding=0, bias=True)
        self.seg_d2 = nn.Conv3d(in_channels=basic_dims*4, out_channels=num_cls, kernel_size=1, stride=1, padding=0, bias=True)
        self.seg_d1 = nn.Conv3d(in_channels=basic_dims*2, out_channels=num_cls, kernel_size=1, stride=1, padding=0, bias=True)
        self.seg_layer = nn.Conv3d(in_channels=basic_dims, out_channels=num_cls, kernel_size=1, stride=1, padding=0, bias=True)
        self.outsig = sigmoidOut(1,1)

        self.up2 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.up4 = nn.Upsample(scale_factor=4, mode='trilinear', align_corners=True)
        self.up8 = nn.Upsample(scale_factor=8, mode='trilinear', align_corners=True)
        self.up16 = nn.Upsample(scale_factor=16, mode='trilinear', align_corners=True)
        
        self.RFM5 = fusion_prenorm(in_channel=192, num_cls=num_cls)
        self.RFM4 = fusion_prenorm(in_channel=192, num_cls=num_cls)
        self.RFM3 = fusion_prenorm(in_channel=96, num_cls=num_cls)
        self.RFM2 = fusion_prenorm(in_channel=48, num_cls=num_cls)
        self.RFM1 = fusion_prenorm(in_channel=24, num_cls=num_cls)

        
        #bunlar new
        self.up_to_224 = nn.Upsample(size=(1, 224, 224), mode='trilinear', align_corners=True)
        self.final_conv = nn.Conv3d(in_channels=8, out_channels=3, kernel_size=1, stride=1, padding=0, bias=True)



    def forward(self, x1, x2, x3, x4, x5):
        #print("x1, x2, x3, x4, x5 shapes", x1.shape, x2.shape, x3.shape, x4.shape, x5.shape)
        de_x5 = self.RFM5(x5)
        #print("de_x5 shape", de_x5.shape)
        #print("self.up2(de_x5) shape", self.up2(de_x5).shape)
        de_x5 = self.d4_c1(self.up2(de_x5))
        #print("final de_x5 shape", de_x5.shape)
        
        de_x4 = self.RFM4(x4)
        de_x4 = F.interpolate(de_x4, (16, 16, 16))
        #print("de_x4 shape", de_x4.shape)
        de_x4 = torch.cat((de_x4, de_x5), dim=1)
        #print("concat de_x4 shape", de_x4.shape)
        #print("self.d4_c2(de_x4) shape", self.d4_c2(de_x4).shape)
        de_x4 = self.d4_out(self.d4_c2(de_x4))
        #print("self.d4_c2(de_x4) shape", self.d4_c2(de_x4).shape)
        de_x4 = self.d3_c1(self.up2(de_x4))
        #print("final de_x4 shape", de_x4.shape)

        de_x3 = self.RFM3(x3)
        #print("de_x3 shape", de_x3.shape)
        de_x3 = F.interpolate(de_x3, (32, 32, 32))
        de_x3 = torch.cat((de_x3, de_x4), dim=1)
        #print("concat de_x3 shape", de_x3.shape)
        de_x3 = self.d3_out(self.d3_c2(de_x3))
        de_x3 = self.d2_c1(self.up2(de_x3))
        #print("final de_x3 shape", de_x3.shape)

        de_x2 = self.RFM2(x2)
        #print("de_x2 shape", de_x2.shape)
        de_x2 = F.interpolate(de_x2, (64, 64, 64))
        de_x2 = torch.cat((de_x2, de_x3), dim=1)
        #print("concat de_x2 shape", de_x2.shape)
        de_x2 = self.d2_out(self.d2_c2(de_x2))
        de_x2 = self.d1_c1(self.up2(de_x2))
        #print("final de_x2 shape", de_x2.shape)

        de_x1 = self.RFM1(x1)
        #print("de_x1 shape", de_x1.shape)
        de_x1 = F.interpolate(de_x1, (128, 128, 128))
        de_x1 = torch.cat((de_x1, de_x2), dim=1)
        #print("concat de_x1 shape", de_x1.shape)
        de_x1 = self.d1_out(self.d1_c2(de_x1))
        #print("final de_x1 shape", de_x1.shape)
        
        
        #bunlar new
        de_x1_up = self.up_to_224(de_x1)
        #print("de_x1_up shape", de_x1_up.shape)
        logits = self.final_conv(de_x1_up)
        #print("logits shape", logits.shape)


        #logits = self.seg_layer(de_x1)
        pred = self.outsig(logits)
        #print("pred shape", pred.shape)


        return pred


class SelfAttention(nn.Module):
    def __init__(
        self, dim, heads=8, qkv_bias=False, qk_scale=None, dropout_rate=0.0
    ):
        super().__init__()
        self.num_heads = heads
        head_dim = dim // heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(dropout_rate)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout_rate)

    def forward(self, x):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x):
        return self.fn(self.norm(x))


class PreNormDrop(nn.Module):
    def __init__(self, dim, dropout_rate, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fn = fn

    def forward(self, x):
        return self.dropout(self.fn(self.norm(x)))


class GELU(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return F.gelu(x)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout_rate):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            GELU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(p=dropout_rate),
        )

    def forward(self, x):
        return self.net(x)


class Transformer(nn.Module):
    def __init__(self, embedding_dim, depth, heads, mlp_dim, dropout_rate=0.1, n_levels=1, n_points=4):
        super(Transformer, self).__init__()
        self.cross_attention_list = []
        self.cross_ffn_list = []
        self.depth = depth
        for j in range(self.depth):
            self.cross_attention_list.append(
                Residual(
                    PreNormDrop(
                        embedding_dim,
                        dropout_rate,
                        SelfAttention(embedding_dim, heads=heads, dropout_rate=dropout_rate),
                    )
                )
            )
            self.cross_ffn_list.append(
                Residual(
                    PreNorm(embedding_dim, FeedForward(embedding_dim, mlp_dim, dropout_rate))
                )
            )

        self.cross_attention_list = nn.ModuleList(self.cross_attention_list)
        self.cross_ffn_list = nn.ModuleList(self.cross_ffn_list)


    def forward(self, x, pos):
        for j in range(self.depth):
            x = x + pos
            x = self.cross_attention_list[j](x)
            x = self.cross_ffn_list[j](x)
        return x



class mmformer(nn.Module):
    def __init__(self, num_cls=1):
        super(mmformer, self).__init__()
        self.RGB_encoder = Encoder()
        self.NIR_encoder = Encoder()
        self.SWIR_encoder = Encoder()

        ########### IntraFormer
        self.RGB_encode_conv = nn.Conv3d(basic_dims*8, transformer_basic_dims, kernel_size=1, stride=1, padding=0)
        self.NIR_encode_conv = nn.Conv3d(basic_dims*8, transformer_basic_dims, kernel_size=1, stride=1, padding=0)
        self.SWIR_encode_conv = nn.Conv3d(basic_dims*8, transformer_basic_dims, kernel_size=1, stride=1, padding=0)

        self.RGB_decode_conv = nn.Conv3d(transformer_basic_dims, basic_dims*8, kernel_size=1, stride=1, padding=0)
        self.NIR_decode_conv = nn.Conv3d(transformer_basic_dims, basic_dims*8, kernel_size=1, stride=1, padding=0)
        self.SWIR_decode_conv = nn.Conv3d(transformer_basic_dims, basic_dims*8, kernel_size=1, stride=1, padding=0)

        self.RGB_pos = nn.Parameter(torch.zeros(1, patch_size**3, transformer_basic_dims))
        self.NIR_pos = nn.Parameter(torch.zeros(1, patch_size**3, transformer_basic_dims))
        self.SWIR_pos = nn.Parameter(torch.zeros(1, patch_size**3, transformer_basic_dims))

        self.RGB_transformer = Transformer(embedding_dim=transformer_basic_dims, depth=depth, heads=num_heads, mlp_dim=mlp_dim)
        self.NIR_transformer = Transformer(embedding_dim=transformer_basic_dims, depth=depth, heads=num_heads, mlp_dim=mlp_dim)
        self.SWIR_transformer = Transformer(embedding_dim=transformer_basic_dims, depth=depth, heads=num_heads, mlp_dim=mlp_dim)
        ########### IntraFormer

        ########### InterFormer
        self.multimodal_transformer = Transformer(embedding_dim=transformer_basic_dims, depth=depth, heads=num_heads, mlp_dim=mlp_dim, n_levels=num_modals)
        self.multimodal_decode_conv = nn.Conv3d(transformer_basic_dims*num_modals, basic_dims*8*num_modals, kernel_size=1, padding=0)
        ########### InterFormer

        self.decoder_fuse = Decoder_fuse(num_cls=num_cls)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                torch.nn.init.kaiming_normal_(m.weight) #

    def forward(self, x):
        #extract feature from different layers
        RGB_x1, RGB_x2, RGB_x3, RGB_x4, RGB_x5, RGB_x6 = self.RGB_encoder(x[:, 0:1, :, :, :])
        NIR_x1, NIR_x2, NIR_x3, NIR_x4, NIR_x5, NIR_x6 = self.NIR_encoder(x[:, 1:2, :, :, :])
        SWIR_x1, SWIR_x2, SWIR_x3, SWIR_x4, SWIR_x5, SWIR_x6 = self.SWIR_encoder(x[:, 2:3, :, :, :])

        ########### IntraFormer
        RGB_token_x6 = self.RGB_encode_conv(RGB_x6).permute(0, 2, 3, 4, 1).contiguous().view(x.size(0), -1, transformer_basic_dims)
        NIR_token_x6 = self.NIR_encode_conv(NIR_x6).permute(0, 2, 3, 4, 1).contiguous().view(x.size(0), -1, transformer_basic_dims)
        SWIR_token_x6 = self.SWIR_encode_conv(SWIR_x6).permute(0, 2, 3, 4, 1).contiguous().view(x.size(0), -1, transformer_basic_dims)

        RGB_intra_token_x6 = self.RGB_transformer(RGB_token_x6, self.RGB_pos)
        NIR_intra_token_x6 = self.NIR_transformer(NIR_token_x6, self.NIR_pos)
        SWIR_intra_token_x6 = self.SWIR_transformer(SWIR_token_x6, self.SWIR_pos)

        RGB_intra_x6 = RGB_intra_token_x6.view(x.size(0), patch_size, patch_size, patch_size, transformer_basic_dims).permute(0, 4, 1, 2, 3).contiguous()
        NIR_intra_x6 = NIR_intra_token_x6.view(x.size(0), patch_size, patch_size, patch_size, transformer_basic_dims).permute(0, 4, 1, 2, 3).contiguous()
        SWIR_intra_x6 = SWIR_intra_token_x6.view(x.size(0), patch_size, patch_size, patch_size, transformer_basic_dims).permute(0, 4, 1, 2, 3).contiguous()

        ########### IntraFormer
        x1 = torch.stack((RGB_x1, NIR_x1, SWIR_x1), dim=1)
        B, K, C, H, W, Z = x1.size()
        x1 = x1.view(B, -1, H, W, Z)
        x2 = torch.stack((RGB_x2, NIR_x2, SWIR_x2), dim=1)
        B, K, C, H, W, Z = x2.size()
        x2 = x2.view(B, -1, H, W, Z)
        x3 = torch.stack((RGB_x3, NIR_x3, SWIR_x3), dim=1)
        B, K, C, H, W, Z = x3.size()
        x3 = x3.view(B, -1, H, W, Z)       
        x4 = torch.stack((RGB_x4, NIR_x4, SWIR_x4), dim=1)
        B, K, C, H, W, Z = x4.size()
        x4 = x4.view(B, -1, H, W, Z)        
        x6_intra = torch.stack((RGB_intra_x6, NIR_intra_x6, SWIR_intra_x6), dim=1)
        B, K, C, H, W, Z = x6_intra.size()
        x6_intra = x6_intra.view(B, -1, H, W, Z)         

        ########### InterFormer
        RGB_intra_x6, NIR_intra_x6, SWIR_intra_x6 = torch.chunk(x6_intra, num_modals, dim=1)
               
        multimodal_token_x6 = torch.cat((RGB_intra_x6.permute(0, 2, 3, 4, 1).contiguous().view(x.size(0), -1, transformer_basic_dims),
                                         NIR_intra_x6.permute(0, 2, 3, 4, 1).contiguous().view(x.size(0), -1, transformer_basic_dims), 
                                         SWIR_intra_x6.permute(0, 2, 3, 4, 1).contiguous().view(x.size(0), -1, transformer_basic_dims)), dim=1)                               
                                         
        multimodal_pos = torch.cat((self.RGB_pos, self.NIR_pos, self.SWIR_pos), dim=1)
        
        multimodal_inter_token_x6 = self.multimodal_transformer(multimodal_token_x6, multimodal_pos)                                 

        multimodal_inter_x6 = self.multimodal_decode_conv(multimodal_inter_token_x6.view(multimodal_inter_token_x6.size(0), patch_size, patch_size, patch_size, transformer_basic_dims*num_modals).permute(0, 4, 1, 2, 3).contiguous())
        x6_inter = multimodal_inter_x6
        fuse_pred= self.decoder_fuse(x1, x2, x3, x4, x6_inter)
        return fuse_pred

