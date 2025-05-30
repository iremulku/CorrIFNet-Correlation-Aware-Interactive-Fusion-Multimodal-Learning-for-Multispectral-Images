import torch
import torch.nn.functional as F
from einops import repeat
from timm.models.layers import to_2tuple
from torch import nn
from einops.layers.torch import Rearrange
import numpy as np
import torch.utils.checkpoint as checkpoint
from einops import rearrange
from einops.layers.torch import Rearrange
from timm.models.layers import to_2tuple, trunc_normal_, DropPath



def creat_norm_layer(norm_layer, channel, is_token=False):
    if not is_token:
        if norm_layer == 'LN':
            norm = nn.Sequential(
                Rearrange('b c h w -> b h w c'),
                nn.LayerNorm(channel),
                Rearrange('b h w c -> b c h w')
            )
        elif norm_layer == 'BN':
            norm = nn.BatchNorm2d(channel)
        else:
            raise NotImplementedError(f"norm layer type does not exist, please check the 'norm_layer' arg!")
    else:
        if norm_layer == 'LN':
            norm = nn.LayerNorm(channel)
        elif norm_layer == 'BN':
            norm = nn.Sequential(
                Rearrange('b d n -> b n d'),
                nn.BatchNorm1d(channel)
            )
        else:
            raise NotImplementedError(f"norm layer type does not exist, please check the 'norm_layer' arg!")

    return norm


class Spatial_attention(nn.Module):
    def __init__(self, encoder_chans, decoder_chans, attn_chans=None, act_layer=nn.ReLU):
        super().__init__()
        attn_chans = attn_chans or decoder_chans
        self.conv1 = nn.Sequential(
            nn.MaxPool2d(4),
            nn.Conv2d(encoder_chans, attn_chans, kernel_size=1),
            nn.BatchNorm2d(attn_chans)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(decoder_chans, attn_chans, kernel_size=1),
            nn.BatchNorm2d(attn_chans)
        )
        self.attn = nn.Sequential(
            act_layer(inplace=True) if act_layer != nn.GELU else act_layer(),
            nn.Conv2d(attn_chans, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

    def forward(self, x_en, x_de):
        x_en = self.conv1(x_en)
        x_de = self.conv2(x_de)

        return x_de * self.attn(x_en + x_de)


class Dw_spatial_attention(nn.Module):
    def __init__(self, in_chans):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Conv2d(in_chans, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

    def forward(self, _, x):
        return x * self.attn(x)


class SE_channel_attention(nn.Module):
    def __init__(self, in_chans, ratio=4, act_layer=nn.ReLU6):
        super().__init__()
        self.attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_chans, in_chans // ratio, kernel_size=1, bias=False),
            act_layer(inplace=True) if act_layer != nn.GELU else act_layer(),
            nn.Conv2d(in_chans // ratio, in_chans, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.attn(x)


class CBAM_channel_attention(nn.Module):
    def __init__(self, in_chans, ratio=4, act_layer=nn.ReLU6):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.conv1 = nn.Conv2d(in_chans, in_chans // ratio, kernel_size=1, bias=False)
        self.act_layer = act_layer(inplace=True) if act_layer != nn.GELU else act_layer()
        self.conv2 = nn.Conv2d(in_chans // ratio, in_chans, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.conv2(self.act_layer(self.conv1(self.avg_pool(x))))
        max_out = self.conv2(self.act_layer(self.conv1(self.max_pool(x))))
        weight = self.sigmoid(avg_out + max_out)
        return x * weight

class Build_decode_gate(nn.Module):
    def __init__(self, in_chans, n_classes, norm_layer, act_layer, head_chans=None,
                 chan_ratio=16, chan_attn_type='SE', dw_spac_attn=False, en_chans=None):
        super().__init__()
        head_chans = head_chans or in_chans // 2

        self.conv = nn.Sequential(
            nn.Conv2d(in_chans, head_chans, kernel_size=3, padding=1, bias=False),
            creat_norm_layer(norm_layer, head_chans)
        )

        self.spat_attn = Spatial_attention(encoder_chans=en_chans,
                                           decoder_chans=head_chans,
                                           attn_chans=None,
                                           act_layer=act_layer
                                           ) if not dw_spac_attn else Dw_spatial_attention(head_chans)

        self.dwconv = nn.Sequential(
            nn.Conv2d(head_chans, head_chans, kernel_size=3, padding=1, groups=head_chans),
            creat_norm_layer(norm_layer, head_chans),
            nn.Conv2d(head_chans, in_chans, kernel_size=1, bias=False)
        )

        self.out = nn.Sequential(
            act_layer(inplace=True) if act_layer != nn.GELU else act_layer(),
            nn.Conv2d(in_chans, n_classes, kernel_size=1)
        )

        if chan_attn_type == 'CBAM':
            self.chan_attn = CBAM_channel_attention(in_chans=head_chans, ratio=chan_ratio)
        elif chan_attn_type == 'SE':
            self.chan_attn = SE_channel_attention(in_chans=head_chans, ratio=chan_ratio)
        else:
            raise NotImplementedError(f"Build channel attention does not support {chan_attn_type}")


    def forward(self, x, x1):
        short_cut = x1
        x1 = self.conv(x1)

        spat_x = self.spat_attn(x, x1)
        chan_x = self.chan_attn(x1)
        fuse_attn_x = self.dwconv(spat_x + chan_x)

        x = short_cut + fuse_attn_x
        x = self.out(x)
        x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=True)

        return x

def creat_norm_layer(norm_layer, channel):
    if norm_layer == 'LN':
        norm = nn.Sequential(
            Rearrange('b c h w -> b h w c'),
            nn.LayerNorm(channel),
            Rearrange('b h w c -> b c h w')
        )
    elif norm_layer == 'BN':
        norm = nn.BatchNorm2d(channel)
    else:
        raise NotImplementedError(f"norm layer type does not exist, please check the 'norm_layer' arg!")
    return norm


class PPM(nn.Module):
    def __init__(self, ppm_in_chans, out_chans=512, pool_sizes=(1, 2, 3, 6), norm_layer='BN', act_layer=nn.ReLU):
        super().__init__()
        self.pool_projs = nn.ModuleList(
            nn.Sequential(
                nn.AdaptiveMaxPool2d(pool_size),
                nn.Conv2d(ppm_in_chans, out_chans, kernel_size=1, bias=False),
                # creat_norm_layer(norm_layer, out_chans),
                act_layer(inplace=True) if act_layer != nn.GELU else act_layer()
            )for pool_size in pool_sizes)

        self.bottom = nn.Sequential(
            nn.Conv2d(ppm_in_chans + len(pool_sizes) * out_chans, out_chans, kernel_size=3, padding=1, bias=False),
            creat_norm_layer(norm_layer, out_chans),
            act_layer(inplace=True) if act_layer != nn.GELU else act_layer()
        )

    def forward(self, x):
        xs = [x]
        for pool_proj in self.pool_projs:
            pool_x = F.interpolate(pool_proj(x), size=(x.size(2), x.size(3)), mode='bilinear', align_corners=True)
            xs.append(pool_x)

        x = torch.cat(xs, dim=1)
        x = self.bottom(x)

        return x


class FPN_neck(nn.Module):
    def __init__(self, in_chans, depth, out_chans=512, norm_layer='BN', act_layer=nn.ReLU):
        super().__init__()
        self.depth = depth
        stage = [i for i in range(depth)]

        self.conv_ = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(in_chans * 2 ** stage[::-1][i + 1], out_chans, kernel_size=1, bias=False),
                creat_norm_layer(norm_layer, out_chans),
                act_layer(inplace=True) if act_layer != nn.GELU else act_layer()
            )for i in range(depth - 1))

        self.fpn_conv = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1, bias=False),
                creat_norm_layer(norm_layer, out_chans),
                act_layer(inplace=True) if act_layer != nn.GELU else act_layer()
            )for _ in range(depth - 1))

        self.out = nn.Sequential(
            nn.Conv2d(out_chans * depth, out_chans, kernel_size=3, padding=1, bias=False),
            creat_norm_layer(norm_layer, out_chans),
            act_layer(inplace=True) if act_layer != nn.GELU else act_layer()
        )

    def forward(self, x):
        fpn_x = x[0]
        out = [fpn_x]
        for i in range(self.depth - 1):
            fpn_x = F.interpolate(x[i], scale_factor=2, mode='bilinear', align_corners=True)
            fpn_x = self.fpn_conv[i](fpn_x) + self.conv_[i](x[i + 1])
            x[i + 1] = fpn_x
            out.append(fpn_x)
        out = out[::-1]
        _, _, H, W = out[0].shape
        for i in range(1, len(out)):
            out[i] = F.interpolate(out[i], size=(H, W), mode='bilinear', align_corners=True)
        x = torch.cat(out, dim=1)

        return self.out(x)

class Build_neck(nn.Module):
    def __init__(self, in_chans, out_chans, depth, pool_sizes=(1, 2, 3, 6), norm_layer='BN', act_layer=nn.ReLU):
        super().__init__()
        self.ppm_head = PPM(ppm_in_chans=in_chans * 2 ** (depth - 1),
                            out_chans=out_chans,
                            pool_sizes=pool_sizes,
                            norm_layer=norm_layer,
                            act_layer=act_layer)
        self.fpn_neck = FPN_neck(in_chans=in_chans,
                                 out_chans=out_chans,
                                 depth=depth,
                                 norm_layer=norm_layer,
                                 act_layer=act_layer)

    def forward(self, x):
        x = list(x)[::-1]
        x[0] = self.ppm_head(x[0])
        x = self.fpn_neck(x)
        return x



def creat_norm_layer(norm_layer, channel, is_token=False):
    if not is_token:
        if norm_layer == 'LN':
            norm = nn.Sequential(
                Rearrange('b c h w -> b h w c'),
                nn.LayerNorm(channel),
                Rearrange('b h w c -> b c h w')
            )
        elif norm_layer == 'BN':
            norm = nn.BatchNorm2d(channel)
        else:
            raise NotImplementedError(f"norm layer type does not exist, please check the 'norm_layer' arg!")
    else:
        if norm_layer == 'LN':
            norm = nn.LayerNorm(channel)
        elif norm_layer == 'BN':
            norm = nn.Sequential(
                Rearrange('b n d -> b d n'),
                nn.BatchNorm1d(channel),
                Rearrange('b d n -> b n d'),
            )
        else:
            raise NotImplementedError(f"norm layer type does not exist, please check the 'norm_layer' arg!")

    return norm


def sa_window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def sa_window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class MLP(nn.Module):
    def __init__(self, d=96, hidden_dim=None, out_dim=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_dim = out_dim or d
        hidden_dim = hidden_dim or d
        self.fc1 = nn.Linear(d, hidden_dim)
        self.act_layer = act_layer(inplace=True) if act_layer != nn.GELU else act_layer()
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x, *_):
        x = self.fc1(x)
        x = self.act_layer(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        return x


class CNNMlp(nn.Module):
    def __init__(self, in_chans, hidden_chans, group_dim, drop=0., norm_layer='BN', act_layer=nn.ReLU):
        super().__init__()
        if group_dim is None:
            n_group = 1
        else:
            assert in_chans % group_dim == 0, f"The total number of channels is {in_chans}, while the group dimension is {group_dim}."
            n_group = in_chans // group_dim

        self.convup = nn.Sequential(
            nn.Conv2d(in_chans, hidden_chans, kernel_size=1, groups=n_group),
            act_layer(inplace=True) if act_layer != nn.GELU else act_layer()
        )
        self.dw_conv = nn.Sequential(
            nn.Conv2d(hidden_chans, hidden_chans, kernel_size=3, padding=1, bias=False, groups=hidden_chans),
            creat_norm_layer(norm_layer, hidden_chans),
            act_layer(inplace=True) if act_layer != nn.GELU else act_layer(),
        )
        self.convdown = nn.Conv2d(hidden_chans, in_chans, kernel_size=1)
        self.drop = nn.Dropout(drop)

    def forward(self, x, C, H, W):
        x = x.transpose(1, 2).contiguous().view(-1, C, H, W)
        short_cut = x
        x = self.convup(x)
        x = self.drop(x)
        x = self.dw_conv(x)
        x = self.drop(x)
        x = self.convdown(x)
        x = self.drop(x)
        x = short_cut + x

        return x.flatten(2).transpose(1, 2)


class FC_window_self_attention(nn.Module):
    def __init__(self, d, window_size, n_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0., ratio=None):
        super().__init__()
        self.dim = d
        self.window_size = window_size
        self.n_heads = n_heads
        head_dim = d // n_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), n_heads))
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        qkv_dim = d * 3 if ratio is None else d + 2 * (d // ratio // n_heads) * n_heads
        self.qkv = nn.Linear(d, int(qkv_dim), bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(d, d)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        B_, n, d = x.shape
        q_k_v = self.qkv(x)
        q, k = q_k_v[..., :-d].chunk(2, dim=-1)
        v = q_k_v[..., -d:]
        q = q.reshape(B_, n, self.n_heads, -1).permute(0, 2, 1, 3)
        k = k.reshape(B_, n, self.n_heads, -1).permute(0, 2, 1, 3)
        v = v.reshape(B_, n, self.n_heads, -1).permute(0, 2, 1, 3)

        qk = (q @ k.transpose(-2, -1)) * self.scale

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        qk = qk + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            qk = qk.view(B_ // nW, nW, self.n_heads, n, n) + mask.unsqueeze(1).unsqueeze(0)
            qk = qk.view(-1, self.n_heads, n, n)
            qk = self.softmax(qk)
        else:
            qk = self.softmax(qk)

        qk = self.attn_drop(qk)

        x = (qk @ v).transpose(1, 2).reshape(B_, n, d)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CNN_window_self_attention(nn.Module):
    def __init__(self, d, window_size, n_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0., ratio=None):
        super().__init__()
        self.dim = d
        self.window_size = window_size
        self.Wh, self.Ww = window_size
        self.n_heads = n_heads
        head_dim = d // n_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), n_heads))
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        qkv_dim = d * 3 if ratio is None else d + 2 * (d // ratio // n_heads) * n_heads
        self.qkv = nn.Conv2d(d, int(qkv_dim), kernel_size=1, bias=qkv_bias)  # change kernel_size and padding to extract local context
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Conv2d(d, d, kernel_size=1, bias=False)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        x = rearrange(x, 'B (Wh Ww) d -> B d Wh Ww', Wh=self.Wh, Ww=self.Ww, d=self.dim)
        q_k_v = self.qkv(x)
        q, k = q_k_v[:, :-self.dim, ...].flatten(2).permute(0, 2, 1).chunk(2, dim=-1)
        v = q_k_v[:, -self.dim:, ...].flatten(2).permute(0, 2, 1)
        B_, n, d = v.shape
        q = q.reshape(B_, n, self.n_heads, -1).permute(0, 2, 1, 3)
        k = k.reshape(B_, n, self.n_heads, -1).permute(0, 2, 1, 3)
        v = v.reshape(B_, n, self.n_heads, -1).permute(0, 2, 1, 3)

        qk = (q @ k.transpose(-2, -1)) * self.scale

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        qk = qk + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            qk = qk.view(B_ // nW, nW, self.n_heads, n, n) + mask.unsqueeze(1).unsqueeze(0)
            qk = qk.view(-1, self.n_heads, n, n)
            qk = self.softmax(qk)
        else:
            qk = self.softmax(qk)

        qk = self.attn_drop(qk)

        x = (qk @ v).transpose(1, 2).reshape(B_, n, d)
        x = rearrange(x, 'B (Wh Ww) d -> B d Wh Ww', Wh=self.Wh, Ww=self.Ww, d=self.dim)
        x = self.proj(x)
        x = rearrange(x, 'B d Wh Ww -> B (Wh Ww) d', Wh=self.Wh, Ww=self.Ww, d=self.dim)
        x = self.proj_drop(x)
        return x


class PatchEmbed_block(nn.Module):
    def __init__(self, patch_size=4, in_chans=3, out_chans=96):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size

        self.embed = nn.Conv2d(in_chans, out_chans, kernel_size=patch_size, stride=patch_size)
        self.ln = nn.LayerNorm(out_chans)

    def forward(self, x):
        _, _, H, W = x.shape
        if W % self.patch_size[1] != 0:
            x = F.pad(x, (0, self.patch_size[1] - W % self.patch_size[1]))
        if H % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[0] - H % self.patch_size[0]))

        x = self.embed(x)
        _, c, H, W = x.size()
        x = x.flatten(2).transpose(1, 2)
        x = self.ln(x)
        x = x.transpose(1, 2).view(-1, c, H, W)

        return x


class Downsampling_block(nn.Module):
    def __init__(self, in_chans, out_chans):
        super().__init__()
        dim = in_chans * 4
        self.reduction = nn.Linear(dim, out_chans, bias=False)
        self.ln = nn.LayerNorm(dim)

    def forward(self, x, H, W):
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)

        # padding
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.ln(x)
        x = self.reduction(x)

        return x


class Basic_block(nn.Module):
    def __init__(self, d, n_heads, window_size=8, shift_size=0, mlp_ratio=4., qk_ratio=None,
                 qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0., qkv_type='CNN',
                 ffn_type='CNN', norm_layer='LN', act_layer=nn.GELU, group_dim=16, idx2group=None):
        super().__init__()
        self.dim = d
        self.num_heads = n_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.ffn_type = ffn_type
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"
        assert qkv_type in {'CNN', 'FC'}, "qkv type must be either 'CNN' (cnn feed forward) or 'FC' (mlp feed forward)"
        assert ffn_type in {'CNN', 'FC'}, "ffd type must be either 'CNN' (cnn feed forward) or 'FC' (mlp feed forward)"
        self.H, self.W = None, None
        group_dims = [group_dim, None]

        if qkv_type == 'CNN':
            self.attn = CNN_window_self_attention(
                d, window_size=to_2tuple(self.window_size), n_heads=n_heads, qkv_bias=qkv_bias,
                qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, ratio=qk_ratio)
        elif qkv_type == 'FC':
            self.attn = FC_window_self_attention(
                d, window_size=to_2tuple(self.window_size), n_heads=n_heads, qkv_bias=qkv_bias,
                qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, ratio=qk_ratio)

        self.norm1 = nn.LayerNorm(d)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(d * mlp_ratio)

        if ffn_type == 'CNN':
            self.mlp = CNNMlp(in_chans=d, hidden_chans=mlp_hidden_dim, group_dim=group_dims[idx2group], drop=drop, act_layer=act_layer)
        elif ffn_type == 'FC':
            self.mlp = MLP(d=d, hidden_dim=mlp_hidden_dim, drop=drop, act_layer=nn.GELU)

        self.norm2 = creat_norm_layer(norm_layer, d, True)

    def forward(self, x, mask_matrix):
        B, L, C = x.shape
        H, W = self.H, self.W
        assert L == H * W, "Input feature has wrong size"
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape

        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None

        x_windows = sa_window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
        attn_windows = self.attn(x_windows, mask=attn_mask)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = sa_window_reverse(attn_windows, self.window_size, Hp, Wp)

        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(x)

        x = self.drop_path(self.mlp(self.norm2(x), C, H, W)) + x

        return x


class BasicLayer(nn.Module):
    def __init__(self,
                 d,
                 group_dim,
                 depth,
                 n_heads,
                 window_size=8,
                 mlp_ratio=4.,
                 qk_ratio=3,
                 down_ratio=2,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 downsample=None,
                 use_checkpoint=False,
                 qkv_type='CNN',
                 ffn_type='CNN',
                 norm_layer='LN',
                 act_layer=nn.GELU):
        super().__init__()
        self.window_size = window_size
        self.shift_size = window_size // 2
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        self.long_blocks = nn.ModuleList([
            Basic_block(
                d=d,
                n_heads=n_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qk_ratio=qk_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                qkv_type=qkv_type,
                ffn_type=ffn_type,
                norm_layer=norm_layer,
                act_layer=act_layer,
                group_dim=group_dim,
                idx2group=i % 2)
            for i in range(depth)])

        if downsample is not None:
            self.downsample = downsample(in_chans=d, out_chans=d * down_ratio)
        else:
            self.downsample = None

    def forward(self, x, H, W):
        Hp = int(np.ceil(H / self.window_size)) * self.window_size
        Wp = int(np.ceil(W / self.window_size)) * self.window_size
        img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = sa_window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        for i, blk in enumerate(self.long_blocks):
            blk.H, blk.W = H, W
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, attn_mask)
            else:
                x = blk(x, attn_mask)

        if self.downsample is not None:
            x_down = self.downsample(x, H, W)
            Wh, Ww = (H + 1) // 2, (W + 1) // 2
            return x, H, W, x_down, Wh, Ww
        else:
            return x, H, W, x, H, W


class Build_backbone(nn.Module):

    def __init__(self,
                 patch_size=4,
                 in_chans=3,
                 embed_dim=96,
                 group_dim=16,
                 depths=(2, 2, 6, 2),
                 num_heads=(3, 6, 12, 24),
                 window_size=8,
                 mlp_ratio=4.,
                 qk_ratio=3,
                 down_ratio=2,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.2,
                 norm_layer='BN',
                 act_layer=nn.ReLU,
                 patch_norm=True,
                 out_indices=(0, 1, 2, 3),
                 frozen_stages=-1,
                 use_checkpoint=False,
                 qkv_type='FC',
                 ffn_type='CNN'):
        super().__init__()
        for i, stage_depth in enumerate(depths):
            assert stage_depth % 2 == 0, f"Stage{i}'s depth must be even, but stage{i}_depth = {stage_depth} !!"
        self.patch_size = patch_size
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages

        if patch_size is not None:
            patch_size = to_2tuple(patch_size)
            self.patch_embed = PatchEmbed_block(patch_size=patch_size, in_chans=in_chans, out_chans=embed_dim)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build MSEs
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                d=int(embed_dim * 2 ** i_layer),
                group_dim=group_dim,
                depth=depths[i_layer],
                n_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qk_ratio=qk_ratio,
                down_ratio=down_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                downsample=Downsampling_block if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint,
                qkv_type=qkv_type,
                ffn_type=ffn_type,
                norm_layer=norm_layer,
                act_layer=act_layer)
            self.layers.append(layer)

        num_features = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]
        self.num_features = num_features

        # add a norm layer for each output
        for i_layer in out_indices:
            layer = creat_norm_layer('LN', num_features[i_layer], True)
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)

        self._freeze_stages()

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False

        if self.frozen_stages >= 2:
            self.pos_drop.eval()
            for i in range(0, self.frozen_stages - 1):
                m = self.layers[i]
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def forward(self, x):
        if self.patch_size is not None:
            x = self.patch_embed(x)

        Wh, Ww = x.size(2), x.size(3)
        x = x.flatten(2).transpose(1, 2)
        x = self.pos_drop(x)

        outs = []
        for i in range(self.num_layers):
            layer = self.layers[i]
            x_out, H, W, x, Wh, Ww = layer(x, Wh, Ww)

            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                x_out = norm_layer(x_out)

                out = x_out.view(-1, H, W, self.num_features[i]).permute(0, 3, 1, 2).contiguous()
                outs.append(out)

        return tuple(outs)

    def train(self, mode=True):
        """Convert the model into training mode while keep MSEs freezed."""
        super(Build_backbone, self).train(mode)
        self._freeze_stages()


class CNN_Block(nn.Module):
    expansion = 1

    def __init__(self, in_chans, planes, stride=1):
        super(CNN_Block, self).__init__()
        self.conv1 = nn.Conv2d(in_chans, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_chans != planes * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_chans, planes * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * self.expansion)
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out


class CNN_backbone(nn.Module):
    def __init__(self, chans):
        super(CNN_backbone, self).__init__()
        self.in_planes = chans
        self.layer1 = self._make_layer(chans, 3, stride=1)
        self.layer2 = self._make_layer(chans*2, 4, stride=2)
        self.layer3 = self._make_layer(chans*4, 6, stride=2)
        self.layer4 = self._make_layer(chans*8, 3, stride=2)

    def _make_layer(self, planes, blocks, stride):
        strides = [stride] + [1] * (blocks - 1)
        layers = []
        for stride in strides:
            layers.append(CNN_Block(self.in_planes, planes, stride))
            self.in_planes = planes * CNN_Block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        feat1 = self.layer1(x)
        feat2 = self.layer2(feat1)
        feat3 = self.layer3(feat2)
        feat4 = self.layer4(feat3)
        return feat1, feat2, feat3, feat4

def creat_norm_layer(norm_layer, channel, is_token=False):
    if not is_token:
        if norm_layer == 'LN':
            norm = nn.Sequential(
                Rearrange('b c h w -> b h w c'),
                nn.LayerNorm(channel),
                Rearrange('b h w c -> b c h w')
            )
        elif norm_layer == 'BN':
            norm = nn.BatchNorm2d(channel)
        else:
            raise NotImplementedError(f"norm layer type does not exist, please check the 'norm_layer' arg!")
    else:
        if norm_layer == 'LN':
            norm = nn.LayerNorm(channel)
        elif norm_layer == 'BN':
            norm = nn.Sequential(
                Rearrange('b d n -> b n d'),
                nn.BatchNorm1d(channel)
            )
        else:
            raise NotImplementedError(f"norm layer type does not exist, please check the 'norm_layer' arg!")

    return norm


class MSE(nn.Module):
    def __init__(self, in_chans, out_chans, n_group=4, use_pos=True, channel_attn_type='SE', ratio=16):
        super().__init__()
        self.use_pos = use_pos

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_chans),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Conv2d(out_chans, out_chans // 2, kernel_size=1, bias=False)
        self.conv3 = nn.Sequential(
            nn.Conv2d(out_chans // 2, out_chans // 2, kernel_size=3, padding=1, groups=n_group),
            nn.BatchNorm2d(out_chans // 2),
            nn.Conv2d(out_chans // 2, out_chans, kernel_size=1),
            nn.ReLU(inplace=True)
        )
        if channel_attn_type == 'SE':
            self.attn = SE_channel_attention(out_chans, ratio)
        else:
            self.attn = CBAM_channel_attention(out_chans, ratio)

    def forward(self, x, pos):
        x = self.conv1(x)
        short_cut = x
        x = self.conv2(x)
        if self.use_pos:
            b, c, H, W = x.shape
            pos = repeat(pos, '1 -> b c H W', b=b, c=c, H=H, W=W)
            x = x + pos
        x = self.conv3(x)
        x = x + short_cut
        x = self.attn(x)

        return x


class AMM(nn.Module):
    def __init__(self, in_chans,
                 out_chans,
                 n_branch,
                 offset_scale=16,
                 patch_size=4,
                 n_heads=4,
                 fuse_drop=0.,
                 qkv_bias=True):
        super().__init__()
        self.n_heads = n_heads
        self.patch_size = to_2tuple(patch_size)

        self.short_cut_conv = nn.Sequential(nn.Conv2d(in_chans, out_chans, kernel_size=patch_size, stride=patch_size),
                                            creat_norm_layer('LN', out_chans))

        self.q = nn.Conv2d(in_chans, in_chans, kernel_size=1, bias=qkv_bias, groups=n_branch)
        self.k = nn.Conv2d(in_chans, in_chans, kernel_size=1, bias=qkv_bias, groups=n_branch)
        self.v = nn.Conv2d(in_chans, in_chans, kernel_size=1, bias=qkv_bias, groups=n_branch)
        self.q_proj = nn.Sequential(nn.MaxPool2d(offset_scale, stride=offset_scale),
                                    nn.Conv2d(in_chans, in_chans, kernel_size=3, stride=1, groups=in_chans))
        self.k_proj = nn.Sequential(nn.MaxPool2d(offset_scale, stride=offset_scale),
                                    nn.Conv2d(in_chans, in_chans, kernel_size=3, stride=1, groups=in_chans))
        self.v_proj = nn.Conv2d(in_chans, in_chans, kernel_size=patch_size, stride=patch_size, groups=in_chans)
        self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((n_heads, 1, 1))), requires_grad=True)

        self.cpb_mlp = nn.Sequential(nn.Linear(1, 16 * n_branch, bias=True),
                                     nn.ReLU(inplace=True),
                                     nn.Linear(16 * n_branch, n_heads, bias=False))

        coords = torch.zeros([in_chans, in_chans], dtype=torch.int64)
        for idx in range(in_chans):
            coords[idx] = torch.arange(in_chans) - idx
        relative_position_bias = coords / coords.max()
        relative_position_bias *= 8  # normalize to -8, 8
        relative_position_bias = torch.sign(relative_position_bias) * torch.log2(torch.abs(relative_position_bias) + 1.0) / np.log2(8)
        self.register_buffer("relative_position_bias", relative_position_bias.unsqueeze(-1))

        self.dropout = nn.Dropout(fuse_drop)
        self.norm = creat_norm_layer('LN', out_chans)
        self.softmax = nn.Softmax(dim=-1)
        self.softmax1 = nn.Softmax(dim=-1)
        self.proj = nn.Sequential(nn.Conv2d(in_chans, in_chans, kernel_size=1),
                                  nn.GELU(),
                                  nn.Conv2d(in_chans, out_chans, kernel_size=1))

    def forward(self, x):
        short_cut = x
        b, c, H, W = x.shape
        q, k, v = self.q(x), self.k(x), self.v(x)  # b, c, h, w
        q, k, v = self.q_proj(q).flatten(2), self.k_proj(k).flatten(2), self.v_proj(v).flatten(2)  # b, c, h*w
        q = q.reshape(b, c, self.n_heads, -1).permute(0, 2, 1, 3)  # b, n, c, h*w//n
        k = k.reshape(b, c, self.n_heads, -1).permute(0, 2, 1, 3)
        v = v.reshape(b, c, self.n_heads, -1).permute(0, 2, 1, 3)

        # cosine attention
        sim = (F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1))
        logit_scale = torch.clamp(self.logit_scale, max=torch.log(torch.tensor(1. / 0.01).to(sim.device))).exp()
        sim = sim * logit_scale

        relative_position_bias = self.cpb_mlp(self.relative_position_bias).view(-1, self.n_heads).view(c, c, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        relative_position_bias = torch.sigmoid(relative_position_bias)
        sim = sim + relative_position_bias.unsqueeze(0)

        sim = self.softmax1(1 - self.softmax(sim))
        sim = self.dropout(sim)
        x = (sim @ v).transpose(1, 2).reshape(b, c, -1)  # b, c, h*w
        x = x.view(b, -1, H // self.patch_size[0], W // self.patch_size[1])  # b, c, h, w
        x = self.proj(x)
        x = self.dropout(x)
        x = self.norm(x) + self.short_cut_conv(short_cut)

        return x, short_cut


class SE_channel_attention(nn.Module):
    def __init__(self, in_chans, ratio=4, act_layer=nn.ReLU6):
        super().__init__()
        self.attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_chans, in_chans // ratio, kernel_size=1, bias=False),
            act_layer(inplace=True) if act_layer != nn.GELU else act_layer(),
            nn.Conv2d(in_chans // ratio, in_chans, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.attn(x)


class CBAM_channel_attention(nn.Module):

    def __init__(self, in_chans, ratio=4, act_layer=nn.ReLU6):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.conv1 = nn.Conv2d(in_chans, in_chans // ratio, kernel_size=1, bias=False)
        self.act_layer = act_layer(inplace=True) if act_layer != nn.GELU else act_layer()
        self.conv2 = nn.Conv2d(in_chans // ratio, in_chans, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.conv2(self.act_layer(self.conv1(self.avg_pool(x))))
        max_out = self.conv2(self.act_layer(self.conv1(self.max_pool(x))))
        weight = self.sigmoid(avg_out + max_out)
        return x * weight


class Build_multimodal_fuse_head(nn.Module):

    def __init__(self,
                 n_branch,
                 in_chans=(3, 3, 3, 3),
                 out_chans=36,
                 n_group=3,
                 use_pos=True,
                 patch_size=4,
                 attn_drop=0.1,
                 qkv_bias=False,
                 offset_scale=8,
                 chan_ratio=16,
                 chan_attn_type='SE',
                 n_heads=2,
                 fuse_type=None,
                 embed_dim=None):
        super().__init__()
        in_chans = in_chans if isinstance(in_chans, tuple) else tuple([in_chans for _ in range(n_branch)])
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.fuse_type = fuse_type
        self.use_pos = use_pos

        self.MSEs = nn.ModuleList([
            MSE(in_chans=in_chans[i],
                out_chans=out_chans,
                n_group=n_group,
                use_pos=use_pos,
                channel_attn_type=chan_attn_type,
                ratio=chan_ratio)
            for i in range(n_branch)])

        if use_pos:
            ang_table = [ang for ang in range(0, 136, 135 // n_branch)]
            self.pos = [nn.Parameter(torch.tensor([np.cos(ang_table[i] * np.pi / 180)], dtype=torch.float32))
                        for i in range(n_branch)]

        smooth_chans = int(out_chans * n_branch)
        self.smooth = nn.Sequential(
            nn.Conv2d(smooth_chans, out_chans, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_chans),
            nn.ReLU()
        )

        if self.fuse_type is None:
            self.fuse_proj = AMM(in_chans=smooth_chans,
                                 out_chans=embed_dim,
                                 n_branch=n_branch,
                                 n_heads=n_heads,
                                 offset_scale=offset_scale,
                                 patch_size=patch_size,
                                 fuse_drop=attn_drop,
                                 qkv_bias=qkv_bias)
        else:
            self.fuse_proj = nn.Identity()

    def forward(self, x):
        x = x if isinstance(x, tuple and list) else tuple([x])
        fuse = []
        for i, layer in enumerate(self.MSEs):
            x_branch = layer(x[i], self.pos[i].to(device=self.device) if self.use_pos else None)
            fuse.append(x_branch)
        x = self.fuse_proj(torch.cat(fuse, dim=1))

        if self.fuse_type is not None:
            x = self.smooth(x)
            return x
        else:
            de_x = self.smooth(x[1])
            return x[0], de_x

class MultiSenseSeg(nn.Module):
 
    def __init__(self,
                 n_classes,
                 n_branch=None,
                 decoder_chans=512,
                 patch_size=4,
                 in_chans=(3,3),
                 head_out_chans=32,
                 group_dim=8,
                 use_pos=True,
                 embed_dim=96,
                 offset_scale=8,
                 fuse_proj=None,
                 depths=(2, 2, 8, 2),
                 num_heads=(3, 6, 12, 24),
                 window_size=8,
                 mlp_ratio=4.,
                 qk_ratio=1.5,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.1,
                 attn_drop_rate=0.1,
                 drop_path_rate=0.1,
                 norm_layer='BN',
                 act_layer=nn.GELU,
                 fpn_norm_layer='BN',
                 fpn_act_layer=nn.ReLU,
                 patch_norm=True,
                 out_indices=(0, 1, 2, 3),
                 frozen_stages=-1,
                 use_checkpoint=False,
                 qkv_type='FC',
                 ffn_type='CNN',
                 pool_sizes=(1, 2, 3, 6),
                 chan_attn_num_head=4,
                 chan_ratio=8,
                 chan_attn_type='SE',
                 aux=False,
                 use_faster=False):
        super().__init__()
        #self.n_branch = n_branch if n_branch is not None else (len(in_chans) if isinstance(in_chans, (list,tuple)) else 1)
        self.n_branch = n_branch if n_branch is not None else (len(in_chans) if isinstance(in_chans, (tuple,list)) else 1)
        use_pos = False if n_branch == 1 else use_pos
        self.n_classes = n_classes
        self.depths = depths
        patch_size = to_2tuple(patch_size)
        self.num_layers = len(depths)
        embed_dim = 64 if use_faster else embed_dim
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.aux = aux

        self.build_MSEs_AMM = Build_multimodal_fuse_head(
            n_branch=n_branch,
            in_chans=in_chans,
            out_chans=head_out_chans,
            n_group=head_out_chans // 2 // group_dim,
            use_pos=use_pos,
            patch_size=patch_size,
            offset_scale=offset_scale,
            attn_drop=attn_drop_rate,
            qkv_bias=qkv_bias,
            chan_ratio=chan_ratio,
            n_heads=chan_attn_num_head,
            chan_attn_type=chan_attn_type,
            fuse_type=fuse_proj,
            embed_dim=embed_dim
        )

        self.build_pipeline = Build_backbone(
            patch_size=None if fuse_proj is None else patch_size,
            in_chans=head_out_chans,
            embed_dim=embed_dim,
            group_dim=group_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qk_ratio=qk_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            act_layer=act_layer,
            patch_norm=patch_norm,
            out_indices=out_indices,
            frozen_stages=frozen_stages,
            use_checkpoint=use_checkpoint,
            qkv_type=qkv_type,
            ffn_type=ffn_type) if not use_faster else CNN_backbone(embed_dim)

        self.build_neck = Build_neck(
            in_chans=embed_dim,
            out_chans=decoder_chans,
            depth=len(depths),
            pool_sizes=pool_sizes,
            norm_layer=fpn_norm_layer,
            act_layer=fpn_act_layer)

        self.build_decode_head = Build_decode_gate(
            in_chans=decoder_chans,
            head_chans=None,
            n_classes=n_classes,
            norm_layer=fpn_norm_layer,
            act_layer=fpn_act_layer,
            chan_ratio=chan_ratio,
            chan_attn_type=chan_attn_type,
            en_chans=head_out_chans)

        if self.aux:
            self.aux_out = nn.Sequential(
                nn.Conv2d(embed_dim * 2 ** (len(depths) - 2), decoder_chans // 2, kernel_size=3, padding=1, bias=False),
                creat_norm_layer(fpn_norm_layer, decoder_chans // 2),
                fpn_act_layer(inplace=True) if fpn_act_layer != nn.GELU else fpn_act_layer(),
                nn.Conv2d(decoder_chans // 2, n_classes, kernel_size=1)
            )
        # self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.bias, 0.0)
                nn.init.constant_(m.weight, 1.0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.bias, 0.0)
                nn.init.constant_(m.weight, 1.0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0.0)
                nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
       
        # --- A) 5D stack check & unpack ---
        if isinstance(x, torch.Tensor) and x.dim() == 5:
            x = list(x.unbind(dim=1))
        elif isinstance(x, (list, tuple)):
            x = list(x)
        else:
            raise ValueError(f"Beklenmeyen input: {type(x)}, dim={getattr(x,'dim',None)}")
       
       
       
        x = self.build_MSEs_AMM(x)
        x = x if isinstance(x, tuple) else (x, x)
        x, back_bone_input = x
        x = self.build_pipeline(x)
        #aux_x = x[-2] if self.aux else None
        x = self.build_neck(x)
        x = self.build_decode_head(back_bone_input, x)
        x = x.unsqueeze(1).repeat(1, self.n_branch, 1, 1, 1)
 
        x = torch.sigmoid(x)
        return x