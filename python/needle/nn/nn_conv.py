"""The module.
"""
from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np
from .nn_basic import Parameter, Module


class Conv(Module):
    """
    Multi-channel 2D convolutional layer
    IMPORTANT: Accepts inputs in NCHW format, outputs also in NCHW format
    Only supports padding=same
    No grouped convolution or dilation
    Only supports square kernels
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        if isinstance(kernel_size, tuple):
            kernel_size = kernel_size[0]
        if isinstance(stride, tuple):
            stride = stride[0]
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

        # Initialize weights
        weight = init.kaiming_uniform(0, 0, shape=(self.kernel_size, self.kernel_size, in_channels, out_channels), device=device, dtype=dtype)
        self.weight = Parameter(weight, device=device, dtype=dtype)

        # Initialize bias
        if bias:
            a = 1 / np.sqrt(in_channels * self.kernel_size * self.kernel_size)
            bias = init.rand(out_channels, low=-a, high=a, device=device, dtype=dtype)
            self.bias = Parameter(bias, device=device, dtype=dtype)

        self.padding = self.kernel_size // 2

    def forward(self, x: Tensor) -> Tensor:
        x_permute = x.transpose((1, 2)).transpose((2, 3))
        out = ops.conv(x_permute, self.weight, stride=self.stride, padding=self.padding)
        out_p = out.transpose((2, 3)).transpose((1, 2))

        if hasattr(self, "bias"):
            out_p += self.bias.reshape((1, self.out_channels, 1, 1)).broadcast_to(out_p.shape)

        return out_p
    