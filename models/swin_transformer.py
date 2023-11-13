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

import matplotlib.pyplot as plt
from einops import rearrange
import math

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


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    # print('window_partition & window_size ---- ', x.shape, window_size)
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    # print('window partition permute :', x.shape, ' by batch * H//wndw * W//wndw, wndw, wndw, chan')
    windows = x.view(-1, window_size, window_size, C)
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
    # print('window reverse input :', windows.shape, window_size, H, W, B)
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    # print('window reverse x :', x.shape)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    # print('window reverse output :', x.shape)
    return x

class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.,
                depth=0, layer_idx=0):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

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

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

        self.depth = depth
        self.layer_idx = layer_idx

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None

        1) qkv : [256, 49, 288]
        2) qkv reshape and permute : [3, 256, 3, 49, 32] / q, k, v : [256, 3, 49, 32]
        3) qkt : [256, 3, 49, 49] (=attn)
        4) attn @ v : [256, 3, 49, 32]
        5) x after reshape(B_ N C) : [256, 49, 96]
        """
        # print('----------------------------- 4) ATTN Block ------------------------')
        B_, N, C = x.shape

        qkv = self.qkv(x)
        # print('qkv : ', qkv.shape)
        qkv = qkv.reshape(B_, N, 3, self.num_heads, C // self.num_heads)
        # print('qkv reshape : ', qkv.shape)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        # print('qkv after permute : ', qkv.shape)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        # print('attn first : ', attn.shape)

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)
        # print('attn second : ', attn.shape)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N)
            attn = attn + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)
        attn_fig = attn.clone()

        # --------------------------- To make Heatmap Start -------------------- #
        # def savefig(attn_mh):
        #     attention_matrix = attn_mh.clone()
        #     H, W = int(math.sqrt(attention_matrix.shape[0] // 2)), int(math.sqrt(attention_matrix.shape[0] // 2))
        #     _, M_fig, H_fig, W_fig = attention_matrix.shape  # H/W_fig = 49
        #     attention_matrix = attention_matrix.reshape(2, H, W,
        #                                                 -1, H_fig, W_fig)
        #     attention_matrix = attention_matrix.permute(0, 3, 1, 2, 4, 5)
        #     attention_matrix = attention_matrix.permute(0, 1, 2, 4, 3, 5)
        #     attention_matrix = rearrange(attention_matrix, 'b mh h hw w ww -> b mh (h hw) (w ww)')  # B, Head, 49h, 49w
        #
        #     for batch_idx in range(2):
        #         '''
        #         head 별로 24번째 row of attention을 visualization
        #         batch, head, 49xh, 49xw
        #         이를 [n, 49, 7, 7]로 바꾸는 과정. 따라서 총 n개의 윈도우들에 대한 7x7이 만들어짐.
        #         '''
        #
        #         fig = plt.figure(figsize=(16, 8))
        #         filename = f"Vis_{batch_idx}_{self.layer_idx}_{self.depth}.png"
        #         filepath = os.path.join('./experiment', 'plt_swin', filename)
        #         fig.suptitle(filename, fontsize=24)
        #
        #         rows = 2
        #         cols = (self.num_heads) // 2 + 1
        #
        #         attn_matrix_clone = attention_matrix.clone()
        #         vis_all_head = torch.zeros(attn_matrix_clone[0, 0].shape)  # 모든 head의 같은 위치의 것을 합하기(패치 수만큼 존재)
        #         vis_each_head = torch.zeros(attn_matrix_clone[0, 0].shape)  # 하나의 head의 모든 걸 합하기(헤드 수만큼 존재)
        #         vis_all_head, vis_each_head = self.vis_reshape(vis_all_head, vis_each_head, H_fig, W_fig)
        #
        #         vis_shape = vis_all_head.shape
        #         vis_all_arr = []
        #
        #         for head_idx in range(self.num_heads):  # visualize the 48th rows of attention matrices in the 0-last heads
        #             # attn_heatmap = attention_matrix[batch_idx, head_idx].reshape((7,7)).detach().cpu().numpy()
        #             attn_heatmap = rearrange(attn_matrix_clone[batch_idx, head_idx],
        #                                      '(hw hfig) (ww wfig) -> hw hfig ww wfig', hfig=H_fig, wfig=W_fig)
        #             attn_heatmap = attn_heatmap.permute(0, 2, 1, 3).reshape(-1, H_fig, W_fig)
        #             attn_heatmap = attn_heatmap[:, 24, :].reshape(-1, 7, 7)
        #             first_shape, _, _ = attn_heatmap.shape
        #             attn_heatmap = attn_heatmap.reshape(int(math.sqrt(first_shape)), int(math.sqrt(first_shape)), 7, 7)
        #             attn_heatmap = rearrange(attn_heatmap, 'hw ww h w -> (hw h) (ww w)').detach().cpu().numpy()
        #             vis_all_head += attn_heatmap
        #             ax = fig.add_subplot(rows, cols, head_idx + 1)
        #             ax.imshow(attn_heatmap, cmap='jet')
        #
        #         ax = fig.add_subplot(rows, cols, head_idx + 2)  # +2 to plot in the next subplot
        #         ax.imshow(vis_all_head, cmap='jet')
        #
        #         plt.savefig(filepath)
        #         plt.close(fig)
        #
        #         # 7x7만 따로 224x224로 upsample
        #         if self.layer_idx == 3:
        #             fig = plt.figure(figsize=(16, 8))
        #             filename = f"VisUp_{batch_idx}_{self.layer_idx}_{self.depth}.png"
        #             filepath = os.path.join('./experiment', 'plt_swin', filename)
        #             fig.suptitle(filename, fontsize=24)
        #
        #             rows = 2
        #             cols = (self.num_heads) // 2 + 1
        #
        #             vis_each_head = torch.zeros(224, 224)
        #
        #             for head_idx in range(self.num_heads):
        #                 # visualize the 48th rows of attention matrices in the 0-last heads
        #                 attn_heatmap = rearrange(attn_matrix_clone[batch_idx, head_idx],
        #                                          '(hw hfig) (ww wfig) -> hw hfig ww wfig', hw=H_fig, ww=W_fig)
        #                 attn_heatmap = attn_heatmap.permute(0, 2, 1, 3).reshape(-1, H_fig, W_fig)
        #                 attn_heatmap = attn_heatmap[:, 24, :].reshape(-1, 7, 7)
        #                 first_shape, _, _ = attn_heatmap.shape
        #                 attn_heatmap = attn_heatmap.reshape(int(math.sqrt(first_shape)),
        #                                                     int(math.sqrt(first_shape)), 7, 7)
        #                 attn_heatmap = attn_heatmap.permute(0, 2, 1, 3)
        #                 attn_heatmap = rearrange(attn_heatmap, 'hw h ww w -> (hw h) (ww w)')
        #
        #                 # Create an upsampling layer and apply it
        #                 upsampler = nn.Upsample(scale_factor=32, mode='bilinear',
        #                                         align_corners=True)  # align_corners=True might be needed for 'bilinear' mode
        #                 upscaled_attn_heatmap = upsampler(
        #                     attn_heatmap.unsqueeze(0).unsqueeze(0))  # Add batch and channel dimensions
        #
        #                 upscaled_attn_heatmap = upscaled_attn_heatmap.squeeze().detach().cpu().numpy()
        #
        #                 vis_each_head += upscaled_attn_heatmap
        #
        #                 ax = fig.add_subplot(rows, cols, head_idx + 1)
        #                 ax.imshow(upscaled_attn_heatmap, cmap='jet')
        #
        #             plt.savefig(filepath)
        #             plt.close(fig)
        #
        #             # --------- 4th layer upscale sum 따로 저장
        #             fig = plt.figure(figsize=(16, 8))
        #             filename = f"VisUpSum_{batch_idx}_{self.layer_idx}_{self.depth}.png"
        #             filepath = os.path.join('./experiment', 'plt_swin', filename)
        #             fig.suptitle(filename, fontsize=24)
        #
        #             rows = 1
        #             cols = 1
        #
        #             ax = fig.add_subplot(rows, cols, 1)
        #             ax.imshow(vis_each_head, cmap='jet')
        #
        #             plt.savefig(filepath)
        #             plt.close(fig)

        # --------------------------- To make Heatmap End ---------------------  #

        # savefig(attn_fig)

        # print('attn third : ', attn.shape)

        x = (attn @ v)
        # print('x after attn v ', x.shape)
        x = x.transpose(1, 2)
        x = x.reshape(B_, N, C)
        # print('x after reshape ', x.shape)

        x = self.proj(x)
        x = self.proj_drop(x)
        # print('attn output ', x.shape)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

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
                 fused_window_process=False, depth=1, layer_idx=1):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,
            depth=depth, layer_idx=layer_idx)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

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
        1) input : [B, sq of 56/28/14/7(56), 96/12/256/512(96)
        2) after view : [B, 56, 56, 96]
        3-1) window_partition view & permute : [4, 8, 8, 7, 7, 96]
        3-2) window_partition output : [256, 7, 7, 96]
        4) x_windows : [256, 49, 96] by [-1, wn x wn, embed]
        5) attn_windows : [256, 49, 96]
        6) attn_windows after view : [256, 7, 7, 96] by [-1, wn, wn, embed]
        7-1) window_reverse : [4, 8, 8, 7, 7, 96]
        7-2) window_reverse output : [4, 56, 56, 96]
        8) x after roll and view : [4, 3136, 96] = output
        '''
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        # print('----------------------------- 3) SwinT Block ------------------------')
        # print('SwinTBlock : ', x.shape)
        # print('shift size : ', self.shift_size)

        shortcut = x
        x = self.norm1(x)
        # print('x after norm : ', x.shape)
        x = x.view(B, H, W, C)
        # print('x after view : ', x.shape)

        # cyclic shift
        if self.shift_size > 0:
            if not self.fused_window_process:
                shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
                # partition windows
                x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
            else:
                x_windows = WindowProcess.apply(x, B, H, W, C, -self.shift_size, self.window_size)
        else:
            shifted_x = x
            # partition windows
            x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C

        # print('x_win 1 : ', x_windows.shape, ' by [-1, wn, wn, chan]')
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C
        # print('x_win 2 : ', x_windows.shape, ' by [-1, wn * wn, chan]')

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C
        # print('attn window shape : ', attn_windows.shape)

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        # print('attn window shape : ', attn_windows.shape, ' by [-1, wndw, wndw, chan]')

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
        # print('SwinTBlock shifted_x : ', x.shape, ' by view [B, H * W, C]')
        x = shortcut + self.drop_path(x)
        # print('SwinTBlock output : ', x.shape)
        # FFN
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        # print('SwinT output : ', x.shape)
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
                 fused_window_process=False, layer_idx=1):

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
                                 depth=i,
                                 layer_idx=layer_idx)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        # print('----------------------------- 2) BasicLayer ----------------------------------')
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                # print('Basic input shape : ', x.shape)
                # print('Basic SwinBlock param : ',self.dim, self.input_resolution)
                x = blk(x)
                # print('Basic Output : ', x.shape)
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
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, fused_window_process=False, **kwargs):
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
        # print('----------------------------- 1) SwinTransformer-----------------------------')
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for i, layer in enumerate(self.layers):
            # print('forward features :', x.shape)
            # print('Layer parameter : ', i, 'embed_dim * 2 ** i - ', int(self.embed_dim * 2 ** i),
            #       'patch_reso // (2**i) - ',
            #       (self.patches_resolution[0] // (2 ** i), self.patches_resolution[1] // (2 ** i)))
            x = layer(x)
            # print('------------------------------------------------------------------------')

        # print('swinT layer over : ', x.shape)
        x = self.norm(x)  # B L C
        # print('after norm : ', x.shape)
        x = self.avgpool(x.transpose(1, 2))  # B C 1
        x = torch.flatten(x, 1)
        # print('layer forward output : ', x.shape)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        # print('final output :', x.shape)
        # print('-----------------------SwinT over----------------------')
        return x

    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += self.num_features * self.patches_resolution[0] * self.patches_resolution[1] // (2 ** self.num_layers)
        flops += self.num_features * self.num_classes
        return flops
