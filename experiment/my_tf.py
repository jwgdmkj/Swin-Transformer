import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import einsum

'''
lepe 더하는 곳에서 차원이 안 맞음. q에 lepe를 적용한다.
drop_path, mlp 등 다 갖고 와서 여기에서도 실험해볼 것.

아니면 v에서 q에 해당하는 부분만 잘라서 get_lepe 하는 건 어떄?
'''

def drop_path_f(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path_f(x, self.drop_prob, self.training)

class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

class myAttention(nn.Module):
    def __init__(self, c1, c2, num_heads, window_size, norm_layer=nn.LayerNorm, bias=False,
                 drop_path=0., mlp_ratio=4., act_layer=nn.GELU, drop=0.):
        super().__init__()

        # init parameter
        self.dim = c2
        self.num_heads = num_heads
        self.window_size = window_size

        # kernel size and scale
        self.kernelX = (window_size, window_size * 3)
        self.kernelY = (window_size * 3, window_size)

        # head variance, head_dim split by 2 because x divided into x1 and x2
        head_dim = (c2/2) // num_heads
        self.scale = head_dim ** -0.5

        # linear to make QKV
        self.norm1 = norm_layer(c2)
        self.qkv = nn.Linear(c2, c2*3, bias)
        self.get_v = nn.Conv2d(c2//2, c2//2, kernel_size=3, stride=1, padding=1, groups=c2//2)

        # Position Embedding
        # self.rel_pos_emb = RelPosEmb(
        #     block_size=window_size,
        #     rel_size=window_size * 3,
        #     dim_head=head_dim
        # )

        # Function after calculate QKV
        mlp_hidden_dim = int(c2 * mlp_ratio)
        self.proj = nn.Linear(c2, c2)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.mlp = Mlp(in_features=c2, hidden_features=mlp_hidden_dim, out_features=c2, act_layer=act_layer,
                       drop=drop)
        self.norm2 = norm_layer(c2)

    def get_lepe(self, x, func, flag):
        '''
        input : [Bhw, H/hw * W/ws, c]
        input must be [B, C, H, W] at func
        1) after transpose & contiguous : [Bhw c H/hw W/ws]
        2) so, after func, x : [Bhw c H/hw W/ws]
        3) lepe must be same size with qktv : [Bhw, H/ws * W/ws,

        이를 chan을 head로 나누고 그 head를 배치로 넣어야 함. 또한 HW 한꺼번에 만들어야 함.
        '''
        # print('5) get_lepe input : ', x.shape)

        B_, N_, C_ = x.shape

        if flag == 'x_axis':
            x = x.transpose(-2, -1).contiguous().view(B_, C_, self.window_size, self.window_size * 3)
            lepe = x[:, :, :, self.window_size : 2 * self.window_size]
        else :
            x = x.transpose(-2, -1).contiguous().view(B_, C_, self.window_size * 3, self.window_size)
            lepe = x[:, :, self.window_size : 2 * self.window_size, :]
        lepe = func(lepe)
        # print('6) lepe and x after func : ', x.shape, lepe.shape)

        # y : height, x : width
        x, lepe = map(lambda t: rearrange(t, 'b (h c) y x -> (b h) (y x) c', h = self.num_heads), (x, lepe))
        # print('7) get_lepe output x lepe ', x.shape, lepe.shape)
        # lepe를 reshape 해야 함.
        return x, lepe

    def Attn(self, x, flag, H, W):
        '''
        Input x : [3, B, HW, C/2(=c)]
        1) q_, k_, v_ = [B, HW, c]
        2-1) q_ : [Bhw, H/ws * W/ws, c] such as [16, 4, 2]
        2-2) k_/v_ : [Bhw, H/ws * W/ws, c] (단, padding값도 H/ws, W/ws에 포함) such as [Bx4x4, 12(2x6), 2(chan)]
        3) q, k, v after map(lambda t ... ) : [Bhw * head, H/ws * W/ws, c]
        '''
        # print('3) Attn input - ', x.shape)
        B_, N_, C_ = x.shape[1], x.shape[2], x.shape[3]

        # Implement qkv, here, divide into self.dim//2 because original input x is divided into x1/x2
        q_, k_, v_ = x[0], x[1], x[2]
        q_ = rearrange(q_, 'b (h w) c -> b h w c', h=H, w=W)
        q_ = rearrange(q_, 'b (h p1) (w p2) c -> (b h w) (p1 p2) c', p1=self.window_size, p2=self.window_size)

        k_ = rearrange(k_, 'b (h w) c -> b h w c', h=H, w=W).contiguous().permute(0, 3, 1, 2)
        v_ = rearrange(v_, 'b (h w) c -> b h w c', h=H, w=W).contiguous().permute(0, 3, 1, 2)

        if flag == 'x_axis':
            # print('-----------x axis-------------')
            k_ = F.pad(k_, (self.window_size, self.window_size, 0, 0))
            k_ = F.unfold(k_, kernel_size = (self.window_size, self.window_size * 3), stride = self.window_size)
            v_ = F.pad(v_, (self.window_size, self.window_size, 0, 0))
            v_ = F.unfold(v_, kernel_size=(self.window_size, self.window_size * 3), stride=self.window_size)
        else :
            # print('----------y axis--------------')
            k_ = F.pad(k_, (0, 0, self.window_size, self.window_size))
            k_ = F.unfold(k_, kernel_size=(self.window_size * 3, self.window_size), stride=self.window_size)
            v_ = F.pad(v_, (0, 0, self.window_size, self.window_size))
            v_ = F.unfold(v_, kernel_size=(self.window_size * 3, self.window_size), stride=self.window_size)
        k_ = rearrange(k_, 'b (c j) i -> (b i) j c', c=self.dim // 2)
        v_ = rearrange(v_, 'b (c j) i -> (b i) j c', c=self.dim // 2)
        # print('q_ k_ v_ : ', q_.shape, k_.shape, v_.shape, self.num_heads)

        # Divide Embedding into Head
        q, k = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = self.num_heads), (q_, k_))
        q *= self.scale
        # print('4) q k v_ - ', q.shape, k.shape, v_.shape)

        # v에 get_lepe 또는 q에 rel_pos_emb 더하기
        v, lepe = self.get_lepe(v_, self.get_v, flag)

        # Attn 구하기
        sim = einsum('b i d, b j d -> b i j', q, k)

        # ------------ Halo Attn Mask & Position Embedding Start -------- #
        # sim += self.rel_pos_emb(q)
        #
        # # mask out padding (in the paper, they claim to not need masks, but what about padding?)
        # device = x.device
        # mask = torch.ones(1, 1, H, W, device=device)
        # mask = F.unfold(mask, kernel_size=self.window_size * 3, stride=self.window_size, padding=self.window_size)
        # # print('mask unfold and block halo ', mask.shape, block, halo)
        # mask = repeat(mask, '() j i -> (b i h) () j', b=B_, h=self.num_heads)
        # mask = mask.bool()
        #
        # max_neg_value = -torch.finfo(sim.dtype).max
        #
        # # https://thought-process-ing.tistory.com/79
        # # sim의 바꾸고자 하는 값(mask)를 max_neg_value로 변경
        # sim.masked_fill_(mask, max_neg_value)
        # ------------ Halo Attn Mask & Position Embedding End----------- #

        attn = sim.softmax(dim=-1)
        out = einsum('b i j, b j d -> b i d', attn, v)
        # print('8) qktv - ', out.shape)
        out = out + lepe

        # merge and combine heads
        out = rearrange(out, '(b h) n d -> b n (h d)', h=self.num_heads)

        # merge blocks back to original feature map
        out = rearrange(out, '(b h w) (p1 p2) c -> b (h p1) (w p2) c', b=B_, h=(H//self.window_size),
                        w=(W//self.window_size), p1=self.window_size, p2=self.window_size)
        out = out.reshape(B_, -1, C_)

        # print('9) Attn output : ', out.shape)
        # print('------------------------')
        return out


    def forward(self, x):
        '''
        input x : [B, C, H, W]
        1) permute and view : [B, HW, C] (=[B, -1, C] 이후 norm)
        2) qkv : [3, B, HW, C]
        2-1) Attn input : [3, B, HW, C//2]
        3) Attend_x : cat(x1, x2) : [B, HW, C]
        '''
        B_, C_, H_, W_ = x.shape
        assert H_ >= self.window_size, 'window should be less than feature map size'
        assert W_ >= self.window_size, 'window should be less than feature map size'

        # H, W,가 window_size의 배수가 아닐 경우, 패딩
        Padding = False
        if min(H_, W_) < self.window_size or H_ % self.window_size != 0 or W_ % self.window_size != 0:
            # print('padding condition : ', H_, W_, self.split_size)
            Padding = True
            # print(f'img_size {min(H_, W_)} is less than (or not divided by) split_size {self.split_size}, Padding.')
            pad_r = (self.window_size - W_ % self.window_size) % self.window_size
            pad_b = (self.window_size - H_ % self.window_size) % self.window_size
            x = F.pad(x, (0, pad_r, 0, pad_b))
        # print('X after padding : ', x.shape)

        B, C, H, W = x.shape

        x = x.permute(0, 2, 3, 1).contiguous().view(B_, H * W, C)
        # print('1) x [B HW C] - ', x.shape)

        # 우선 qkv를 만든 다음, channel을 1/2로 나눠, 하나는 가로방향, 하나는 세로방향으로 진행해야 됨.
        img = self.norm1(x)
        qkv = self.qkv(img).reshape(B_, H * W, 3, C).permute(2, 0, 1, 3) # [3, 1, 64, 4]
        # print('2) qkv ', qkv.shape)

        x1 = self.Attn(qkv[:, :, :, :C//2], 'x_axis', H, W)   # x-axis such as (2, 6). [3, 1, 64, 2]
        x2 = self.Attn(qkv[:, :, :, C//2:], 'y_axis', H, W)   # y-axis such as (6, 2). [3, 1, 64, 2]

        attened_x = torch.cat([x1, x2], dim=2)
        attened_x = self.proj(attened_x)
        x = x + self.drop_path(attened_x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        # change shape into 4 size, [batch, embed, height, width]
        x = x.permute(0, 2, 1).contiguous()

        # print('x shape after permute : ', x.shape)
        x = x.view(-1, C, H, W)  # b c h w

        # print(Padding)
        # print(x.shape)
        # reverse padding
        if Padding:
            x = x[:, :, :H_, :W_]

        # print('10) Final output : ', x.shape)
        return x

####################################### vanila start #################################

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)

class VanilaAttention(nn.Module):
    def __init__(self, c1, c2, num_heads, patch_size, num_classes, dim, depth, heads, mlp_dim,
                 pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()

        # variance
        self.dim = c2
        self.num_heads = num_heads
        self.patch_size = patch_size

        patch_dim = channels * patch_size * patch_size
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2 =patch_size),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Linear(dim, num_classes)

    def forward(self, x):
        B_, C_, H_, W_ = x.shape

        # H, W,가 window_size의 배수가 아닐 경우, 패딩
        Padding = False
        if min(H_, W_) < self.patch_size or H_ % self.patch_size != 0 or W_ % self.patch_size != 0:
            # print('padding condition : ', H_, W_, self.split_size)
            Padding = True
            # print(f'img_size {min(H_, W_)} is less than (or not divided by) split_size {self.split_size}, Padding.')
            pad_r = (self.patch_size - W_ % self.patch_size) % self.patch_size
            pad_b = (self.patch_size - H_ % self.patch_size) % self.patch_size
            x = F.pad(x, (0, pad_r, 0, pad_b))

        B, C, H, W = x.shape

        x = self.to_patch_embedding(x)
        b, n, _ = x.shape

        x += self.pos_embedding[:, :n]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)

        # reverse padding
        if Padding:
            x = x[:, :, :H_, :W_]

        return x

####################################### vanila end #################################

if __name__ == "__main__":
    # x = torch.arange(96).float()
    x = torch.randn(1, 32, 21, 12)
    x = x

    B, C, H, W = x.shape

    # attn = myAttention(
    #     c1 = 2,
    #     c2= 32,  # dimension of feature map
    #     num_heads = 4,    # window_size to split
    #     window_size=4  # number of attention heads
    # )

    attn = VanilaAttention(
        c1 = 2,
        c2= 32,  # dimension of feature map
        num_heads = 4,    # window_size to split
        window_size=4  # number of attention heads
    )

    output = attn(x)  # (1, 512, 32, 32)
    print(output.shape)

