# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

import pdb
import numpy as np
import matplotlib.pyplot as plt
from einops import rearrange
import math
import torch.nn.functional as F

'''
각각에 따른 shape 변화(window_size: 8 & input shape: [B, 3, 224, 224] & 임베딩: [B, 3136(56X56), 96(임베딩 차원)] 가정
depth: [2, 2, 18, 2], window_size를 [7, 7, 7, 7]이라 가정 시

Stage 3:
view : [B, 56, 56, 96] -> [4, 8, 8, 7, 7, 96] by 56/window=7 -> [256, 49, 96] by [4x8x8, 49, 96] 윈도우 하나당 49개의 값

Stage 4:
qkv 진행 시, [256, 49, 96x3] -> [256, 49, 3, 3, 32] by [.., qkv, num_head, embed/num_head] -> [3, 256, 3, 49, 32]
qk : [256, 3, 49, 49] and qktv : [256, 7, 7, 96] by qkv.transpose(1,2).reshape(B,N,C)
window_reverse : [4, 56/7, 56/7, 7, 7, 96] -> [4, 56, 56, 96] -> [4, 56x56, 96] 이 최종 output

이를 depth[i]만큼 진행, 각 i별 output은 merge 기준
[B, 784(28x28), 192] / [B, 196(14x14), 384] / [B, 49(7x7), 768]이 되며, 최종적으로 이 [B, 49, 738]에 mean을 적용한 [B, 768]

관건 -
우선, dual-former을 따르지 않고, 오직 1/2로만 진행
MLP와 concat gate를 만들어야 함.  --> hybrid에서는 ACM, MBFFN
'''

try:
    import os, sys

    kernel_path = os.path.abspath(os.path.join('..'))
    sys.path.append(kernel_path)
    from kernels.window_process.window_process import WindowProcess, WindowProcessReverse

except:
    WindowProcess = None
    WindowProcessReverse = None
    print("[Warning] Fused window process have not been installed. Please refer to get_started.md for installation.")


##############################################################################
## MLP module
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class GDFN(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(GDFN, self).__init__()
        hidden_features = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class GCFN(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(GCFN, self).__init__()
        pass

    def forward(self, x):
        pass

##########################################################################
## window reshape
def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()  # [B, H/ws, W/ws, ws, ws, C]
    windows = x.view(-1, window_size, window_size, C)  # [B x H/ws x W/ws, ws, ws, C]
    return windows


def window_partition_v2(x, window_size, H, W):
    """
    Args:
        x: (B, H, W, C)
        window_size (int H, int W): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)

    x is already divided into Head(=H)
    """
    QKV, B, N, Head, C = x.shape
    x = x.view(QKV, B, H, W, Head, C)
    x = x.view(QKV, B, H // window_size[0], window_size[1], W // window_size[0], window_size[1], Head, C)
    x = x.permute(0, 1, 2, 4, 3, 5, 6, 7).contiguous()  # [QKV, B, H/ws, W/ws, ws, ws, Head, C]
    windows = x.reshape(QKV, -1, Head, window_size[0] * window_size[1], C)  # [QKV, B x H/ws x W/ws, Head, ws x ws, C]
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    B = Batch

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


##########################################################################################
## Channel and Spaital Mixing
class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=False, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


# 2)
class spatial_attn_layer(nn.Module):
    def __init__(self, kernel_size=5):
        super(spatial_attn_layer, self).__init__()
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False)
        print('2) init : ', self.spatial)

    def forward(self, x):
        print('2) input : ', x.shape)
        x_compress = self.compress(x)
        print('2) after GAP, GMP: ', x_compress.shape)

        x_out = self.spatial(x_compress)
        print('2) after spatial: ', x_out.shape)

        scale = torch.sigmoid(x_out)  # broadcasting
        return x * scale

class ca_layer(nn.Module):
    def __init__(self, channel, reduction=8, bias=True):
        super(ca_layer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
            nn.Sigmoid()
        )
        print('3) init : ', self.conv_du)

    def forward(self, x):
        print('3) input : ', x.shape)
        y = self.avg_pool(x)
        print('3) after pool : ', y.shape)
        y = self.conv_du(y)
        print('3) after du : ', y.shape)
        return x * y

class DAU(nn.Module):
    '''
    1) Conv - GELU - Conv
    2) 1)의 GAP, GMP 이후 concat - Conv - Sigmoid, 1)과 Element-wise
    3) 1)의 GAP, Conv, Gelu, Conv, Sigmoid, 1)과 Element-wise
    4) 2) 3)을 Concat, Conv, 이후 원본과 element-sum
    '''
    def __init__(self, dim, kernel_size=3, reduction=8, bias=False, act=nn.ReLU()):
        super(DAU, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size, bias),
            act,
            nn.Conv2d(dim // 2, dim, kernel_size, bias)
        )

        ## Spatial Attention
        self.SA = spatial_attn_layer()

        ## Channel Attention
        self.CA = ca_layer(dim, reduction, bias=bias)

        self.conv1x1 = nn.Conv2d(dim * 2, dim, kernel_size=1, bias=bias)
        print('4) init : ', self.conv1x1)


    def forward(self, x):
        print('0) x input : ', x.shape)
        res = self.body(x)
        print('1) x after CGC : ', x.shape)

        sa_branch = self.SA(res)
        print('2) x after SA : ', x.shape)

        ca_branch = self.CA(res)
        print('3) x after CA : ', x.shape)

        res = torch.cat([sa_branch, ca_branch], dim=1)
        res = self.conv1x1(res)
        print('4) x final : ', x.shape)
        res += x

        import pdb;pdb.set_trace()
        return res


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

##########################################################################################
## Fourth Arch : Calculate Attn
class HybridAttention(nn.Module):
    r"""
    Channel-wise Self Attention.
    Some heads calculate CWSA and be concat with the other general MHSA.

    Modified code is being '--added--
    """

    def __init__(self, dim, window_size, num_heads, chan_heads=1, qkv_bias=True, qk_scale=None, attn_drop=0.,
                 proj_drop=0., bias=False, depth=0, layer_idx=0):
        super().__init__()
        
        # dim = input으로부터 1/2로 나눠짐
        self.dim = dim // 2
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads

        # --added-- number of heads that calculate CWSA
        self.chan_heads = chan_heads

        # hybrid : 다시 1/2로 쪼개서 cw, sw로 나눠야됨.
        head_dim = (dim//2) // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table_sw = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH
        self.relative_position_bias_table_cw = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), chan_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        # Spatial QKV
        self.qkv = nn.Linear(self.dim, self.dim * 3, bias=qkv_bias)

        #  --added-- depth-wise convolution, no bias
        self.qkv_dwconv = nn.Conv2d(self.dim * 3, self.dim * 3, kernel_size=3, stride=1, padding=1,
                                    groups=self.dim * 3, bias=bias)
        self.project_out = nn.Conv2d(self.dim, self.dim, kernel_size=1, bias=bias)
        self.temperature = nn.Parameter(torch.ones(chan_heads, 1, 1))

        # FFN
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

        self.depth = depth
        self.layer_idx = layer_idx

        # Spatial and Channel Mixing
        act = SimpleGate()
        self.mix = DAU(dim, kernel_size=3, reduction=4, bias=bias, act=act)



    def forward(self, x, mask=None):
        """
        Args:
            x1: input features with shape of (num_windows*B, N, C)
            x2 : B H W C
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        x1, x2 = x[0], x[1]

        # --------------------------- Spatial Attn ------------------------- #
        B1, N1, C1 = x1.shape

        qkv = self.qkv(x1).reshape(B1, N1, self.num_heads, C1//self.num_heads).permute(2,0,3,1,4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B1 // nW, nW, self.num_heads, N1, N1)
            attn = attn + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N1, N1)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn_s = self.attn_drop(attn)
        attn_s = (attn_s @ v)
        x1 = attn_s.transpose(1, 2)
        x1 = x1.reshape(B1, N1, C1)
        # --------------------------- Spatial Over ------------------------- #

        # --------------------------- Channel Attn ------------------------- #
        B2, H2, W2, C2 = x2.shape

        qkv2 = self.qkv_dwconv(self.qkv(x))
        qc, kc, vc = qkv2[0], qkv[1], qkv[2]

        qc = rearrange(qc, 'b (head c) h w -> b head c (h w)', head=self.chan_heads)
        kc = rearrange(kc, 'b (head c) h w -> b head c (h w)', head=self.chan_heads)
        vc = rearrange(vc, 'b (head c) h w -> b head c (h w)', head=self.chan_heads)

        qc = torch.nn.functional.normalize(qc, dim=-1)
        kc = torch.nn.functional.normalize(kc, dim=-1)

        attn = (qc @ kc.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        x2 = (attn @ vc)
        x2 = rearrange(x2, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=H2, w=W2)
        # --------------------------- Spatial Over ------------------------- #

        # --------------------------- Concat Mixing ------------------------ #


        # spatial :
        x1 = self.proj(x1)
        x1 = self.proj_drop(x1)
        
        # channel :
        x2 = self.project_out(x2)
        
        # 새로운 concat 및 게이트를 만들어야 함.
        out = torch.cat([x1, x2])



        return out  # [256, 49, 96] / [64, 49, 192]

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops

    def vis_reshape(self, vis_all_head, vis_each_head, H_fig, W_fig):
        vis_all = rearrange(vis_all_head,
                            '(hw hfig) (ww wfig) -> hw hfig ww wfig', hw=H_fig, ww=W_fig)
        vis_all = vis_all.permute(0, 2, 1, 3).reshape(-1, H_fig, W_fig)
        vis_all = vis_all[:, 0, :].reshape(-1, 7, 7)
        first_shape, _, _ = vis_all.shape
        vis_all = vis_all.reshape(int(math.sqrt(first_shape)), int(math.sqrt(first_shape)), 7, 7)
        vis_all = vis_all.permute(0, 2, 1, 3)
        vis_all = rearrange(vis_all, 'hw h ww w -> (hw h) (ww w)').detach().cpu().numpy()

        vis_each = rearrange(vis_each_head,
                             '(hw hfig) (ww wfig) -> hw hfig ww wfig', hw=H_fig, ww=W_fig)
        vis_each = vis_each.permute(0, 2, 1, 3).reshape(-1, H_fig, W_fig)
        vis_each = vis_each[:, 0, :].reshape(-1, 7, 7)
        first_shape, _, _ = vis_each.shape
        vis_each = vis_each.reshape(int(math.sqrt(first_shape)), int(math.sqrt(first_shape)), 7, 7)
        vis_each = vis_each.permute(0, 2, 1, 3)
        vis_each = rearrange(vis_each, 'hw h ww w -> (hw h) (ww w)').detach().cpu().numpy()

        return vis_all, vis_each


##########################################################################################
## Third Arch : Define Depth per Stage
class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
        fused_window_process (bool, optional): If True, use one kernel to fused window shift & window partition for acceleration, similar for the reversed part. Default: False
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 fused_window_process=False, chan_heads=1, depth=1, layer_idx=1):
        super().__init__()

        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads

        # window processing
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        # attn 전 func
        self.norm1 = norm_layer(dim)

        # Attention
        self.attn = HybridAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads, chan_heads=chan_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,
            depth=depth, layer_idx=layer_idx)

        # attn 후 func
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        # mask 위한 slice 처리
        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
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

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)
        self.fused_window_process = fused_window_process

    def forward(self, x):
        '''
        input --> LN -> SWSA --> LN --> MLP Gate
          |                  |
          ----> LN -> CWSA ---
        '''
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x).view(B, H, W, C)

        # cyclic shift and C방향으로 1/2씩 쪼갬, 다음으로 x1만 win*win으로 분할
        if self.shift_size > 0:
            # default : False
            if not self.fused_window_process:
                shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
                x1, x2 = shifted_x[:,:,:,:C//2], shifted_x[:,:,:,C//2:]
                # partition windows
                x_windows = window_partition(x1, self.window_size)  # nW*B, window_size, window_size, C
            # not executed
            else:
                x_windows = WindowProcess.apply(x, B, H, W, C, -self.shift_size, self.window_size)
        else:
            shifted_x = x
            x1, x2 = shifted_x[:, :, :, :C // 2], shifted_x[:, :, :, C // 2:]
            # partition windows
            x_windows = window_partition(x1, self.window_size)  # nW*B, window_size, window_size, C

        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn([x_windows, x2], mask=self.attn_mask)  # [256, 49, 96]이 되어야 함












        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)

        # reverse cyclic shift
        if self.shift_size > 0:
            if not self.fused_window_process:
                shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C
                x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
            else:
                x = WindowProcessReverse.apply(attn_windows, B, H, W, C, self.shift_size, self.window_size)
        else:
            shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C
            x = shifted_x
        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(x)

        # FFN
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops


##########################################################################
## Resizing modules
class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.dim
        flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        return flops


##########################################################################
## Second Arch : Define Stage
class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        fused_window_process (bool, optional): If True, use one kernel to fused window shift & window partition for acceleration, similar for the reversed part. Default: False
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 fused_window_process=False,
                 chan_heads=1, layer_idx=1):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer,
                                 fused_window_process=fused_window_process,
                                 chan_heads=chan_heads,
                                 depth=i,
                                 layer_idx=layer_idx)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops


##########################################################################
## Patch Embedding
class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops


##########################################################################
## SwinTransformer First Arch
class SwinTransformer(nn.Module):
    r""" Swin Transformer
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030

    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
        fused_window_process (bool, optional): If True, use one kernel to fused window shift & window partition for acceleration, similar for the reversed part. Default: False
    
    1) 전체에 pos_embed 더하기
    2) spatial에만 pos_embed 더하기
    3) 아예 안 더하기
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, fused_window_process=False, chan_heads=[1, 1, 1, 1], **kwargs):
        super().__init__()

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # CWSA
        self.chan_heads = chan_heads

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                               use_checkpoint=use_checkpoint,
                               fused_window_process=fused_window_process,
                               chan_heads=chan_heads[i_layer],
                               layer_idx=i_layer)

            self.layers.append(layer)

        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_features(self, x):
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        for i, layer in enumerate(self.layers):
            x = layer(x)

        x = self.norm(x)  # B L C
        x = self.avgpool(x.transpose(1, 2))  # B C 1
        x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += self.num_features * self.patches_resolution[0] * self.patches_resolution[1] // (2 ** self.num_layers)
        flops += self.num_features * self.num_classes
        return flops