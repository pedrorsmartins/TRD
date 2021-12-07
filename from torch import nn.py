from torch import nn
import torch
# # With square kernels and equal stride
# m = nn.Conv2d(16, 33, 3, stride=2)
# # non-square kernels and unequal stride and with padding
# m = nn.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2))
# non-square kernels and unequal stride and with padding and dilation
m = nn.Conv2d(1, 100, (5, 16), padding=(2, 0))
input = torch.randn(128, 1, 168, 16)
output = m(input)
print(output.shape)