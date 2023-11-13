import torch
import torch.nn as nn

class myAttention(nn.Module):
    def __init__(self, c1 ,c2, num_heads, window_size):
        super().__init__()
        self.layer_norm = nn.LayerNorm([5, 10, 10])

    def forward(self, input):
        output = self.layer_norm(input)
        return output

if __name__ == "__main__":
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

    N, C, H, W = 20, 5, 10, 10
    input = torch.randn(N, C, H, W)
    output = attn(input)  # (1, 512, 32, 32)
    print(output.shape)



