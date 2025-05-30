import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Function

# 2D-only implementation for input: [B, 2, 3, H, W] and target: [B, 2, 1, H, W]

class RobustMseg(nn.Module):
    def __init__(self, n_base_filters=16, final_sigmoid=True):
        super(RobustMseg, self).__init__()
        self.final_sigmoid = final_sigmoid

        in_channels_per_mod = 3
        num_modalities = 3

        # Style & content encoders (2D)
        self.style_enc_list = nn.ModuleList([
            StyleEncoder2d(in_channels=in_channels_per_mod, base_ch=32)
            for _ in range(num_modalities)
        ])
        self.content_enc_list = nn.ModuleList([
            ContentEncoder2d(in_channels=in_channels_per_mod, base_filters=n_base_filters)
            for _ in range(num_modalities)
        ])

        # Content attention & share conv
        self.content_attn = nn.ModuleList()
        self.content_share = nn.ModuleList()
        ch = n_base_filters
        for _ in range(4):  # 4 levels
            self.content_attn.append(
                BasicConv2d(ch * num_modalities, num_modalities, kernel_size=3, stride=1, padding=1, relu=False, norm=True)
            )
            self.content_share.append(
                BasicConv2d(ch * num_modalities, ch, kernel_size=1, stride=1, padding=0, relu=True, norm=True)
            )
            ch *= 2

        # Reconstruction decoders
        content_ch = ch // 2
        self.recon_decoders = nn.ModuleList([
            ImageDecoder2d(style_ch=128, content_ch=content_ch, mlp_ch=128, out_ch=in_channels_per_mod)
            for _ in range(num_modalities)
        ])

        # Segmentation decoder
        self.seg_decoder = MaskDecoder2d(in_ch=content_ch, num_classes=1)

    def forward(self, x, drop=None, valid=False):
        # x: [B, M, C, H, W]
        B, M, C, H, W = x.shape
        if drop is None:
            drop = torch.sum(x.view(B, M, -1), dim=2) == 0

        # 1) Encode
        style_feats = []
        content_feats = [[] for _ in range(4)]
        for m in range(M):
            xm = x[:, m]  # [B,C,H,W]
            s = self.style_enc_list[m](xm)  # [B,128,1,1]
            if valid:
                s = s.data.new(s.size()).normal_()
            style_feats.append(s)
            cont_list = self.content_enc_list[m](xm)
            for lvl, feat in enumerate(cont_list):
                content_feats[lvl].append(ZeroLayer.apply(feat, drop[:, m]))

        # 2) Content attention & fusion
        shared = []
        for lvl, feats in enumerate(content_feats):
            cat = torch.cat(feats, dim=1)
            attn = torch.sigmoid(self.content_attn[lvl](cat))  # [B,2,H,W]
            weighted = torch.cat([feats[i] * attn[:, i:i+1] for i in range(M)], dim=1)
            shared.append(self.content_share[lvl](weighted))

        # 3) Reconstruction
        recon_outs = []
        mu_list, sigma_list = [], []
        top_shared = shared[-1]
        for m in range(M):
            recon, mu, sigma = self.recon_decoders[m](style_feats[m], top_shared, valid)
            recon_outs.append(recon)
            mu_list.append(mu)
            sigma_list.append(sigma)
        recon_out = torch.cat(recon_outs, dim=1)  # [B, M*C, H, W]

        # 4) Segmentation
        segs = []
        for _ in range(M):
            mask = self.seg_decoder(shared)
            mask = torch.sigmoid(mask) if self.final_sigmoid else F.softmax(mask, dim=1)
            segs.append(mask)
        seg_out = torch.stack(segs, dim=1)  # [B, M, 1, H, W]

        return seg_out


class StyleEncoder2d(nn.Module):
    def __init__(self, in_channels=3, base_ch=32):
        super().__init__()
        layers = [
            BasicConv2d(in_channels, base_ch, 7, stride=1, padding=3, relu=True, norm=False),
            BasicConv2d(base_ch, base_ch*2, 4, stride=2, padding=1, relu=True, norm=False),
            BasicConv2d(base_ch*2, base_ch*4, 4, stride=2, padding=1, relu=True, norm=False),
            BasicConv2d(base_ch*4, base_ch*4, 4, stride=2, padding=1, relu=True, norm=False),
            BasicConv2d(base_ch*4, base_ch*4, 4, stride=2, padding=1, relu=True, norm=False),
        ]
        self.encoder = nn.Sequential(*layers)
        self.final = BasicConv2d(base_ch*4, base_ch*4, 1, stride=1, padding=0, relu=False, norm=False)

    def forward(self, x):
        x = self.encoder(x)
        x = x.mean(dim=[2,3], keepdim=True)  # [B, C, 1,1]
        x = self.final(x)
        return x  # [B,128,1,1]


class ContentEncoder2d(nn.Module):
    def __init__(self, in_channels=3, base_filters=16, drop_rate=0.3):
        super().__init__()
        def cb(in_ch, out_ch, down=False):
            stride = 2 if down else 1
            return [
                BasicConv2d(in_ch, out_ch, 3, stride=stride, padding=1, relu=True, norm=False),
                BasicConv2d(out_ch, out_ch, 3, stride=1, padding=1, relu=True, norm=False, drop_rate=drop_rate),
                BasicConv2d(out_ch, out_ch, 3, stride=1, padding=1, relu=True, norm=False)
            ]
        layers = []
        layers += cb(in_channels, base_filters, down=False)
        layers += cb(base_filters, base_filters*2, down=True)
        layers += cb(base_filters*2, base_filters*4, down=True)
        layers += cb(base_filters*4, base_filters*8, down=True)
        self.e1c1, self.e1c2, self.e1c3, self.e2c1, self.e2c2, self.e2c3, \
        self.e3c1, self.e3c2, self.e3c3, self.e4c1, self.e4c2, self.e4c3 = layers

    def forward(self, x):
        x1 = self.e1c3(self.e1c2(self.e1c1(x)))
        out1 = x1 + self.e1c1(x)
        x2 = self.e2c3(self.e2c2(self.e2c1(out1)))
        out2 = x2 + self.e2c1(out1)
        x3 = self.e3c3(self.e3c2(self.e3c1(out2)))
        out3 = x3 + self.e3c1(out2)
        x4 = self.e4c3(self.e4c2(self.e4c1(out3)))
        out4 = x4 + self.e4c1(out3)
        return [out1, out2, out3, out4]


class ImageDecoder2d(nn.Module):
    def __init__(self, style_ch=128, content_ch=128, mlp_ch=128, out_ch=3):
        super().__init__()
        self.mlp = MLP2d(style_ch, mlp_ch)
        self.res_blocks = nn.ModuleList([AdaptiveRes2d((content_ch if i==0 else mlp_ch), mlp_ch) for i in range(4)])
        self.up_blocks = nn.ModuleList()
        ch = mlp_ch
        for _ in range(3):
            self.up_blocks.append(
                nn.Sequential(
                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                    BasicConv2d(ch, ch//2, 5, stride=1, padding=2, relu=False, norm=False)
                )
            )
            ch //= 2
        self.final = BasicConv2d(ch, out_ch, 7, stride=1, padding=3, relu=False, norm=False)

    def forward(self, style, content, valid=False):
        mu, sigma = self.mlp(style)
        x = content
        for rb in self.res_blocks:
            x = rb(x, mu, sigma)
        for up in self.up_blocks:
            x = up(x)
            x = F.layer_norm(x, x.shape[1:])
            x = F.relu(x, inplace=True)
        x = self.final(x)
        return x, mu, sigma


class MaskDecoder2d(nn.Module):
    def __init__(self, in_ch=128, num_classes=1):
        super().__init__()
        # Stage 3: process deepest feature
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.c3_1 = BasicConv2d(in_ch, in_ch//2, 3, stride=1, padding=1, relu=True, norm=True)
        self.c3_2 = BasicConv2d(in_ch//2, in_ch//2, 3, stride=1, padding=1, relu=True, norm=True)
        self.c3_3 = BasicConv2d(in_ch//2, in_ch//2, 1, stride=1, padding=0, relu=True, norm=True)

        # Stage 2: fuse with next skip connection
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.c2_1 = BasicConv2d(in_ch, in_ch//4, 3, stride=1, padding=1, relu=True, norm=True)
        self.c2_2 = BasicConv2d(in_ch//4, in_ch//4, 3, stride=1, padding=1, relu=True, norm=True)
        self.c2_3 = BasicConv2d(in_ch//4, in_ch//4, 1, stride=1, padding=0, relu=True, norm=True)

        # Stage 1: fuse with shallow skip
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.c1_1 = BasicConv2d(in_ch//2, in_ch//8, 3, stride=1, padding=1, relu=True, norm=True)
        self.c1_2 = BasicConv2d(in_ch//8, in_ch//8, 3, stride=1, padding=1, relu=True, norm=True)
        self.c1_3 = BasicConv2d(in_ch//8, in_ch//8, 1, stride=1, padding=0, relu=True, norm=True)

        # Final: fuse with first skip and output mask
        self.final = BasicConv2d(in_ch//4, num_classes, 1, stride=1, padding=0, relu=False, norm=False)

    def forward(self, feats):
        # feats: [lvl1, lvl2, lvl3, lvl4]
        # Stage 3
        x = self.up3(feats[3])                 # [B,128,H4,W4] → [B,128,2*H4,2*W4]
        x = self.c3_1(x)                       # → [B,64,...]
        x = self.c3_2(x)                       # → [B,64,...]
        x = self.c3_3(x)                       # → [B,64,...]
        x = torch.cat([x, feats[2]], dim=1)    # → [B,64+64=128,...]

        # Stage 2
        x = self.up2(x)                       # → [B,128,...]
        x = self.c2_1(x)                      # → [B,32,...]
        x = self.c2_2(x)                      # → [B,32,...]
        x = self.c2_3(x)                      # → [B,32,...]
        x = torch.cat([x, feats[1]], dim=1)   # → [B,32+32=64,...]

        # Stage 1
        x = self.up1(x)                       # → [B,64,...]
        x = self.c1_1(x)                      # → [B,16,...]
        x = self.c1_2(x)                      # → [B,16,...]
        x = self.c1_3(x)                      # → [B,16,...]
        x = torch.cat([x, feats[0]], dim=1)   # → [B,16+16=32,...]

        # Final mask
        return self.final(x)                  # → [B,1,...]


class MLP2d(nn.Module):
    def __init__(self, in_ch=128, mlp_ch=128):
        super().__init__()
        self.l1 = nn.Linear(in_ch, mlp_ch)
        self.relu1 = nn.ReLU(inplace=True)
        self.l2 = nn.Linear(mlp_ch, mlp_ch)
        self.relu2 = nn.ReLU(inplace=True)
        self.l_mu = nn.Linear(mlp_ch, mlp_ch)
        self.l_sigma = nn.Linear(mlp_ch, mlp_ch)
        self.ch = mlp_ch

    def forward(self, style):
        x = style.view(style.size(0), -1)
        x = self.relu1(self.l1(x))
        x = self.relu2(self.l2(x))
        mu = self.l_mu(x).view(-1, self.ch, 1, 1)
        sigma = self.l_sigma(x).view(-1, self.ch, 1, 1)
        return mu, sigma


class AdaptiveRes2d(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = BasicConv2d(in_ch, out_ch, 3, stride=1, padding=1, relu=False, norm=False)
        self.norm1 = AdaptiveInstanceNorm2d()
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = BasicConv2d(in_ch, out_ch, 3, stride=1, padding=1, relu=False, norm=False)
        self.norm2 = AdaptiveInstanceNorm2d()

    def forward(self, x, gamma, beta):
        y = self.conv1(x)
        y = self.norm1(y, gamma, beta)
        y = self.relu(y)
        y = self.conv2(x)
        y = self.norm2(y, gamma, beta)
        return x + y


class AdaptiveInstanceNorm2d(nn.Module):
    def forward(self, content, gamma, beta, eps=1e-5):
        mean = content.mean(dim=[2,3], keepdim=True)
        std = content.std(dim=[2,3], keepdim=True)
        return gamma * ((content - mean) / (std + eps)) + beta


class BasicConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, relu=True, norm=True, bias=False, drop_rate=0.0):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, dilation, groups, bias)
        self.drop = nn.Dropout2d(drop_rate) if drop_rate>0 else None
        self.norm = nn.InstanceNorm2d(out_ch) if norm else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.drop: x = self.drop(x)
        if self.norm: x = self.norm(x)
        if self.relu: x = self.relu(x)
        return x


class ZeroLayer(Function):
    @staticmethod
    def forward(ctx, x, mask):
        ctx.mask = mask
        y = x.clone()
        y[mask] = 0
        return y

    @staticmethod
    def backward(ctx, grad_output):
        grad = grad_output.clone()
        grad[ctx.mask] = 0
        return grad, None


