import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange, repeat

def organize_tensor(input_tensor, window_size):
    batch_size, channels, height, width = input_tensor.shape    # [1, 512, 8, 8]
    print('input - ', input_tensor.shape)

    # Calculate the number of splits for height and width
    height_splits = height // window_size[0]
    width_splits = width // window_size[1]

    # Reshape the tensor according to window splits
    reshaped_tensor = input_tensor.reshape(batch_size, channels, height_splits, window_size[0], width_splits,
                                           window_size[1])  # [1, 512, 4, 2, 4, 2]
    print('after split - ', reshaped_tensor.shape)
    print(reshaped_tensor)

    # Reshape and transpose to match the desired format
    reshaped_tensor = reshaped_tensor.permute(0, 2, 4, 3, 5, 1)   # [1, 4, 4, 2, 2, 512]
    print('after transpose - ',reshaped_tensor.shape)
    # reshaped_tensor = reshaped_tensor.reshape(batch_size * height_splits * width_splits, window_split, window_size[1],
    #                                           channels)
    # print('after reshpae - ', reshaped_tensor.shape)
    print(reshaped_tensor)

    return reshaped_tensor


def pad_and_store_windows(input_array, window_size, H, W):
    padded_windows = []
    padded_idx = H // window_size[0]
    print('padded idx - ', padded_idx)
    print('input array ', input_array.shape)    # [1, 4, 4, 2, 2, 2]

    for i in range(input_array.shape[1]):
        for j in range(input_array.shape[2]):
        # window = input_array[i]
        # padded_window = np.pad(window, ((0, 0), (0, 0), (1, 1), (window_size[0], window_size[0]), (window_size[1], window_size[1])), mode='constant')
        # padded_windows.append(padded_window)
            print(i * padded_idx + j)
            if (i * padded_idx + j) % padded_idx == 0:
                print('0 - ', i * padded_idx + j % padded_idx)
                query = input_array[:, i, j:j + 2, :, :, :]
                print(query.shape)
                # print(query)
                continue
            elif (i * padded_idx + j) % padded_idx == padded_idx - 1:
                print('-1 - ', i * padded_idx + j % padded_idx)
                query = input_array[:, i, j - 1:j + 1, :, :, :]
                print(query.shape)
                # print(query)
                continue
            else :
                query = input_array[:, i, j - 1:j + 2, :, :, :]
                print(query.shape)
                # print(query)

    return np.concatenate(padded_windows, axis=0)


# Input tensor [1, 512, 4, 4]
x = torch.arange(96).float()
x = x.reshape(3, 2, 4, 4)
B_, C_, H_, W_ = x.shape
print(x)
# Desired window size and split
window_size = (2, 2)

# x = F.pad(x, (window_size[0], window_size[0], 0, 0))
# print(x)

kv_inp = F.unfold(x, kernel_size = 4, stride = 2, padding = 1)
print(kv_inp)
print(kv_inp.shape)

kv_inp = rearrange(kv_inp, 'b (c j) i -> (b i) j c', c = 2)
print(kv_inp)
print(kv_inp.shape)

# k, v = self.to_kv(kv_inp).chunk(2, dim = -1)
# print('k v shape ', k.shape, v.shape)
#
# # split heads
# q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = heads), (q, k, v))
# print('qkv after map ', q.shape, k.shape, v.shape)
