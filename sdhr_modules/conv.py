# Ultralytics YOLO ??, AGPL-3.0 license
"""
Convolution modules
"""
from typing import Tuple
from torch import Tensor
from einops import rearrange
import math
import torch.nn.functional as F
import numpy as np
import torch
import torch.nn as nn
from .wavelet import (
    DWT_Haar_HF, DWT_Haar_LL, DWT_Haar,
    WaveletSpatialGate, WaveletChannelGate,WaveletCoupledGate,
    CBAMSpatialGate, CBAMChannelGate,WaveletCoupledGate_HC
)



__all__ = ('Conv', 'LightConv', 'DWConv', 'DWConvTranspose2d', 'ConvTranspose', 'Focus', 'GhostConv',
           'ChannelAttention', 'SpatialAttention', 'CBAM', 'Concat', 'RepConv',
           'FCM', 'Pzconv',  'FCM_3',
           'FCM_2', 'FCM_1', 'Down')


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))


class Conv2(Conv):
    """Simplified RepConv module with Conv fusing."""

    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__(c1, c2, k, s, p, g=g, d=d, act=act)
        self.cv2 = nn.Conv2d(c1, c2, 1, s, autopad(1, p, d), groups=g, dilation=d, bias=False)  # add 1x1 conv

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x) + self.cv2(x)))

    def forward_fuse(self, x):
        """Apply fused convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def fuse_convs(self):
        """Fuse parallel convolutions."""
        w = torch.zeros_like(self.conv.weight.data)
        i = [x // 2 for x in w.shape[2:]]
        w[:, :, i[0]: i[0] + 1, i[1]: i[1] + 1] = self.cv2.weight.data.clone()
        self.conv.weight.data += w
        self.__delattr__("cv2")
        self.forward = self.forward_fuse


class LightConv(nn.Module):
    """
    Light convolution with args(ch_in, ch_out, kernel).

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, c2, k=1, act=nn.ReLU()):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv1 = Conv(c1, c2, 1, act=False)
        self.conv2 = DWConv(c2, c2, k, act=act)

    def forward(self, x):
        """Apply 2 convolutions to input tensor."""
        return self.conv2(self.conv1(x))


class DWConv(Conv):
    """Depth-wise convolution."""

    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):  # ch_in, ch_out, kernel, stride, dilation, activation
        """Initialize Depth-wise convolution with given parameters."""
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)


class DWConvTranspose2d(nn.ConvTranspose2d):
    """Depth-wise transpose convolution."""

    def __init__(self, c1, c2, k=1, s=1, p1=0, p2=0):  # ch_in, ch_out, kernel, stride, padding, padding_out
        """Initialize DWConvTranspose2d class with given parameters."""
        super().__init__(c1, c2, k, s, p1, p2, groups=math.gcd(c1, c2))


class ConvTranspose(nn.Module):
    """Convolution transpose 2d layer."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=2, s=2, p=0, bn=True, act=True):
        """Initialize ConvTranspose2d layer with batch normalization and activation function."""
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(c1, c2, k, s, p, bias=not bn)
        self.bn = nn.BatchNorm2d(c2) if bn else nn.Identity()
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Applies transposed convolutions, batch normalization and activation to input."""
        return self.act(self.bn(self.conv_transpose(x)))

    def forward_fuse(self, x):
        """Applies activation and convolution transpose operation to input."""
        return self.act(self.conv_transpose(x))


class Focus(nn.Module):
    """Focus wh information into c-space."""

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        """Initializes Focus object with user defined channel, convolution, padding, group and activation values."""
        super().__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act=act)
        # self.contract = Contract(gain=2)

    def forward(self, x):
        """
        Applies convolution to concatenated tensor and returns the output.

        Input shape is (b,c,w,h) and output shape is (b,4c,w/2,h/2).
        """
        return self.conv(torch.cat((x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]), 1))
        # return self.conv(self.contract(x))


class GhostConv(nn.Module):
    """Ghost Convolution https://github.com/huawei-noah/ghostnet."""

    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):
        """Initializes the GhostConv object with input channels, output channels, kernel size, stride, groups and
        activation.
        """
        super().__init__()
        c_ = c2 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, k, s, None, g, act=act)
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act=act)

    def forward(self, x):
        """Forward propagation through a Ghost Bottleneck layer with skip connection."""
        y = self.cv1(x)
        return torch.cat((y, self.cv2(y)), 1)


class RepConv(nn.Module):
    """
    RepConv is a basic rep-style block, including training and deploy status.

    This module is used in RT-DETR.
    Based on https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py
    """

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=3, s=1, p=1, g=1, d=1, act=True, bn=False, deploy=False):
        """Initializes Light Convolution layer with inputs, outputs & optional activation function."""
        super().__init__()
        assert k == 3 and p == 1
        self.g = g
        self.c1 = c1
        self.c2 = c2
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

        self.bn = nn.BatchNorm2d(num_features=c1) if bn and c2 == c1 and s == 1 else None
        self.conv1 = Conv(c1, c2, k, s, p=p, g=g, act=False)
        self.conv2 = Conv(c1, c2, 1, s, p=(p - k // 2), g=g, act=False)

    def forward_fuse(self, x):
        """Forward process."""
        return self.act(self.conv(x))

    def forward(self, x):
        """Forward process."""
        id_out = 0 if self.bn is None else self.bn(x)
        return self.act(self.conv1(x) + self.conv2(x) + id_out)

    def get_equivalent_kernel_bias(self):
        """Returns equivalent kernel and bias by adding 3x3 kernel, 1x1 kernel and identity kernel with their biases."""
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)
        kernelid, biasid = self._fuse_bn_tensor(self.bn)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        """Pads a 1x1 tensor to a 3x3 tensor."""
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        """Generates appropriate kernels and biases for convolution by fusing branches of the neural network."""
        if branch is None:
            return 0, 0
        if isinstance(branch, Conv):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        elif isinstance(branch, nn.BatchNorm2d):
            if not hasattr(self, "id_tensor"):
                input_dim = self.c1 // self.g
                kernel_value = np.zeros((self.c1, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.c1):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def fuse_convs(self):
        """Combines two convolution layers into a single layer and removes unused attributes from the class."""
        if hasattr(self, "conv"):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv = nn.Conv2d(
            in_channels=self.conv1.conv.in_channels,
            out_channels=self.conv1.conv.out_channels,
            kernel_size=self.conv1.conv.kernel_size,
            stride=self.conv1.conv.stride,
            padding=self.conv1.conv.padding,
            dilation=self.conv1.conv.dilation,
            groups=self.conv1.conv.groups,
            bias=True,
        ).requires_grad_(False)
        self.conv.weight.data = kernel
        self.conv.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__("conv1")
        self.__delattr__("conv2")
        if hasattr(self, "nm"):
            self.__delattr__("nm")
        if hasattr(self, "bn"):
            self.__delattr__("bn")
        if hasattr(self, "id_tensor"):
            self.__delattr__("id_tensor")


class ChannelAttention(nn.Module):
    """Channel-attention module https://github.com/open-mmlab/mmdetection/tree/v3.0.0rc1/configs/rtmdet."""

    def __init__(self, channels: int) -> None:
        """Initializes the class and sets the basic configurations and instance variables required."""
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies forward pass using activation on convolutions of the input, optionally using batch normalization."""
        return x * self.act(self.fc(self.pool(x)))


class SpatialAttention(nn.Module):
    """Spatial-attention module."""

    def __init__(self, kernel_size=7):
        """Initialize Spatial-attention module with kernel size argument."""
        super().__init__()
        assert kernel_size in {3, 7}, "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        self.cv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x):
        """Apply channel and spatial attention on input for feature recalibration."""
        return x * self.act(self.cv1(torch.cat([torch.mean(x, 1, keepdim=True), torch.max(x, 1, keepdim=True)[0]], 1)))


class CBAM(nn.Module):
    """Convolutional Block Attention Module."""

    def __init__(self, c1, kernel_size=7):
        """Initialize CBAM with given input channel (c1) and kernel size."""
        super().__init__()
        self.channel_attention = ChannelAttention(c1)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        """Applies the forward pass through C1 module."""
        return self.spatial_attention(self.channel_attention(x))


class SE(nn.Module):
    def __init__(self, c, r=16):
        super().__init__()
        mid = max(c // r, 1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(c, mid, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, c, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.fc(self.pool(x))
    
class WaveletChannelGate(nn.Module):
    """
    输出 (B,C,1,1) 的通道门控，频域引导版本
    mode: 'hf' | 'll' | 'full'
    """
    def __init__(self, c, mode='hf'):
        super().__init__()
        assert mode in ('hf', 'll', 'full')
        self.mode = mode
        self.dwt_hf = DWT_Haar_HF()
        self.dwt_ll = DWT_Haar_LL()
        self.dwt_full = DWT_Haar()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.sigmoid = nn.Sigmoid()
        self.c = c

    def forward(self, x):
        # x: (B,C,H,W)
        if self.mode == 'hf':
            y = self.dwt_hf(x)  # (B,3C,H/2,W/2)
            # 聚合 3 个高频子带 -> (B,C,H/2,W/2)
            y = y.view(x.size(0), 3, self.c, y.size(2), y.size(3)).mean(dim=1)
        elif self.mode == 'll':
            y = self.dwt_ll(x)  # (B,C,H/2,W/2)
        else:
            y = self.dwt_full(x)  # (B,4C,H/2,W/2)
            # 聚合 4 个子带 -> (B,C,H/2,W/2)
            y = y.view(x.size(0), 4, self.c, y.size(2), y.size(3)).mean(dim=1)

        # 频域能量 -> 通道权重
        y = self.pool(y.abs())              # (B,C,1,1)
        w = self.sigmoid(y)
        return w

class WaveletSpatialGate(nn.Module):
    """
    输出 (B,1,H,W) 的空间门控，频域引导版本
    mode: 'hf' | 'll' | 'full'
    """
    def __init__(self, c, mode='hf'):
        super().__init__()
        assert mode in ('hf', 'll', 'full')
        self.mode = mode
        self.dwt_hf = DWT_Haar_HF()
        self.dwt_ll = DWT_Haar_LL()
        self.dwt_full = DWT_Haar()
        self.sigmoid = nn.Sigmoid()
        self.c = c

    def forward(self, x):
        B, C, H, W = x.shape

        if self.mode == 'hf':
            y = self.dwt_hf(x)  # (B,3C,H/2,W/2)
            y = y.view(B, 3, C, y.size(2), y.size(3)).mean(dim=1)  # (B,C,H/2,W/2)
        elif self.mode == 'll':
            y = self.dwt_ll(x)  # (B,C,H/2,W/2)
        else:
            y = self.dwt_full(x)  # (B,4C,H/2,W/2)
            y = y.view(B, 4, C, y.size(2), y.size(3)).mean(dim=1)  # (B,C,H/2,W/2)

        # 回到原分辨率，形成空间响应图
        y = F.interpolate(y.abs(), size=(H, W), mode="nearest")   # (B,C,H,W)
        s = y.mean(dim=1, keepdim=True)                           # (B,1,H,W)
        return self.sigmoid(s)


class ECA(nn.Module):
    def __init__(self, c, k=3):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv1d = nn.Conv1d(1, 1, kernel_size=k, padding=(k-1)//2, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x):
        y = self.pool(x)                      # (B,C,1,1)
        y = y.squeeze(-1).transpose(1, 2)     # (B,1,C)
        y = self.conv1d(y)
        y = self.act(y).transpose(1, 2).unsqueeze(-1)  # (B,C,1,1)
        return x * y
    
class ECA_Gate(nn.Module):
    def __init__(self, c, k=3):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv1d = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.pool(x)                      # (B,C,1,1)
        y = y.squeeze(-1).transpose(1, 2)     # (B,1,C)
        y = self.conv1d(y)
        y = self.sigmoid(y).transpose(1, 2).unsqueeze(-1)  # (B,C,1,1)
        return y

class SpatialGate_CBAM(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        assert kernel_size in (3, 7)
        padding = 3 if kernel_size == 7 else 1
        self.cv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = torch.cat([x.mean(1, keepdim=True), x.max(1, keepdim=True)[0]], 1)
        return self.sigmoid(self.cv1(y))  # (B,1,H,W)

class CoordAtt(nn.Module):
    def __init__(self, c, r=16):
        super().__init__()
        mid = max(8, c // r)
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        self.conv1 = nn.Conv2d(c, mid, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid)
        self.act = nn.SiLU()

        self.conv_h = nn.Conv2d(mid, c, 1, bias=False)
        self.conv_w = nn.Conv2d(mid, c, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()
        x_h = self.pool_h(x)                  # (B,C,H,1)
        x_w = self.pool_w(x).permute(0,1,3,2) # (B,C,W,1)

        y = torch.cat([x_h, x_w], dim=2)      # (B,C,H+W,1)
        y = self.act(self.bn1(self.conv1(y)))

        y_h, y_w = torch.split(y, [h, w], dim=2)
        y_w = y_w.permute(0,1,3,2)

        a_h = self.sigmoid(self.conv_h(y_h))
        a_w = self.sigmoid(self.conv_w(y_w))
        return x * a_h * a_w

class Concat(nn.Module):
    """Concatenate a list of tensors along dimension."""

    def __init__(self, dimension=1):
        """Concatenates a list of tensors along a specified dimension."""
        super().__init__()
        self.d = dimension

    def forward(self, x):
        """Forward pass for the YOLOv8 mask Proto module."""
        return torch.cat(x, self.d)




class DWConv(nn.Module):
    """Depthwise Conv + Conv"""

    def __init__(self, in_channels):
        super().__init__()
        self.dconv = nn.Conv2d(
            in_channels, in_channels, 3,
            1, 1, groups=in_channels
        )

    def forward(self, x):
        x = self.dconv(x)
        return x


class Channel(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv = self.dconv = nn.Conv2d(
            dim, dim, 3,
            1, 1, groups=dim
        )
        self.Apt = nn.AdaptiveAvgPool2d(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x2 = self.dwconv(x)
        x5 = self.Apt(x2)
        x6 = self.sigmoid(x5)

        return x6


class Spatial(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, 1, 1, 1)
        self.bn = nn.BatchNorm2d(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.conv1(x)
        x5 = self.bn(x1)
        x6 = self.sigmoid(x5)

        return x6


""" class FCM_3(nn.Module):
    def __init__(self, dim,dim_out):
        super().__init__()
        self.one = dim - dim // 4
        self.two = dim // 4
        self.conv1 = Conv(dim - dim // 4, dim - dim // 4, 3, 1, 1)
        self.conv12 = Conv(dim - dim // 4, dim - dim // 4, 3, 1, 1)
        self.conv123 = Conv(dim - dim // 4, dim, 1, 1)
        self.conv2 = Conv(dim // 4, dim, 1, 1)
        self.spatial = Spatial(dim)
        self.channel = Channel(dim)

    def forward(self, x):
        x1, x2 = torch.split(x, [self.one, self.two], dim=1)
        x3 = self.conv1(x1)
        x3 = self.conv12(x3)
        x3 = self.conv123(x3)
        x4 = self.conv2(x2)
        x33 = self.spatial(x4) * x3
        x44 = self.channel(x3) * x4
        x5 = x33 + x44
        return x5 """
""" class FCM_3(nn.Module):
    def __init__(self, dim, dim_out, wg_mode="base", wg_band="hf"):
        super().__init__()
        assert wg_mode in ("base", "wg")
        assert wg_band in ("hf", "ll", "full")

        self.one = dim - dim // 4
        self.two = dim // 4

        self.conv1 = Conv(dim - dim // 4, dim - dim // 4, 3, 1, 1)
        self.conv12 = Conv(dim - dim // 4, dim - dim // 4, 3, 1, 1)
        self.conv123 = Conv(dim - dim // 4, dim, 1, 1)

        self.conv2 = Conv(dim // 4, dim, 1, 1)

        if wg_mode == "base":
            self.spatial = Spatial(dim)
            self.channel = Channel(dim)
        else:
            from .wavelet import WaveletSpatialGate, WaveletChannelGate
            self.spatial = WaveletSpatialGate(dim, mode=wg_band)
            self.channel = WaveletChannelGate(dim, mode=wg_band)

    def forward(self, x):
        x1, x2 = torch.split(x, [self.one, self.two], dim=1)

        x3 = self.conv1(x1)
        x3 = self.conv12(x3)
        x3 = self.conv123(x3)

        x4 = self.conv2(x2)

        x33 = self.spatial(x4) * x3
        x44 = self.channel(x3) * x4
        x5 = x33 + x44
        return x5 """



""" class FCM_2(nn.Module):
    def __init__(self, dim,dim_out):
        super().__init__()
        self.one = dim - dim // 4
        self.two = dim // 4
        self.conv1 = Conv(dim - dim // 4, dim - dim // 4, 3, 1, 1)
        self.conv12 = Conv(dim - dim // 4, dim - dim // 4, 3, 1, 1)
        self.conv123 = Conv(dim - dim // 4, dim, 1, 1)

        self.conv2 = Conv(dim // 4, dim, 1, 1)
        self.spatial = Spatial(dim)
        self.channel = Channel(dim)

    def forward(self, x):
        x1, x2 = torch.split(x, [self.one, self.two], dim=1)
        x3 = self.conv1(x1)
        x3 = self.conv12(x3)
        x3 = self.conv123(x3)
        x4 = self.conv2(x2)
        x33 = self.spatial(x4) * x3
        x44 = self.channel(x3) * x4
        x5 = x33 + x44

        return x5 """
""" class FCM_2(nn.Module):
    def __init__(self, dim, dim_out, wg_mode="base", wg_band="hf"):
        super().__init__()
        assert wg_mode in ("base", "wg")
        assert wg_band in ("hf", "ll", "full")

        self.one = dim - dim // 4
        self.two = dim // 4

        self.conv1 = Conv(dim - dim // 4, dim - dim // 4, 3, 1, 1)
        self.conv12 = Conv(dim - dim // 4, dim - dim // 4, 3, 1, 1)
        self.conv123 = Conv(dim - dim // 4, dim, 1, 1)

        self.conv2 = Conv(dim // 4, dim, 1, 1)

        if wg_mode == "base":
            self.spatial = Spatial(dim)
            self.channel = Channel(dim)
        else:
            from .wavelet import WaveletSpatialGate, WaveletChannelGate
            self.spatial = WaveletSpatialGate(dim, mode=wg_band)
            self.channel = WaveletChannelGate(dim, mode=wg_band)

    def forward(self, x):
        x1, x2 = torch.split(x, [self.one, self.two], dim=1)

        x3 = self.conv1(x1)
        x3 = self.conv12(x3)
        x3 = self.conv123(x3)

        x4 = self.conv2(x2)

        x33 = self.spatial(x4) * x3
        x44 = self.channel(x3) * x4
        x5 = x33 + x44
        return x5 """



""" class FCM_1(nn.Module):
    def __init__(self, dim,dim_out):
        super().__init__()

        self.one = dim // 4
        self.two = dim - dim // 4
        self.conv1 = Conv(dim // 4, dim // 4, 3, 1, 1)
        self.conv12 = Conv(dim // 4, dim // 4, 3, 1, 1)
        self.conv123 = Conv(dim // 4, dim, 1, 1)
        self.conv2 = Conv(dim - dim // 4, dim, 1, 1)
        self.spatial = Spatial(dim)
        self.channel = Channel(dim)

    def forward(self, x):
        x1, x2 = torch.split(x, [self.one, self.two], dim=1)
        x3 = self.conv1(x1)
        x3 = self.conv12(x3)
        x3 = self.conv123(x3)
        x4 = self.conv2(x2)
        x33 = self.spatial(x4) * x3
        x44 = self.channel(x3) * x4
        x5 = x33 + x44

        return x5 """
""" class FCM_1(nn.Module):
    def __init__(self, dim, dim_out, wg_mode="base", wg_band="hf"):
        super().__init__()
        assert wg_mode in ("base", "wg")
        assert wg_band in ("hf", "ll", "full")

        self.one = dim // 4
        self.two = dim - dim // 4

        self.conv1 = Conv(dim // 4, dim // 4, 3, 1, 1)
        self.conv12 = Conv(dim // 4, dim // 4, 3, 1, 1)
        self.conv123 = Conv(dim // 4, dim, 1, 1)

        self.conv2 = Conv(dim - dim // 4, dim, 1, 1)

        if wg_mode == "base":
            self.spatial = Spatial(dim)
            self.channel = Channel(dim)
        else:
            from .wavelet import WaveletSpatialGate, WaveletChannelGate
            self.spatial = WaveletSpatialGate(dim, mode=wg_band)
            self.channel = WaveletChannelGate(dim, mode=wg_band)

    def forward(self, x):
        x1, x2 = torch.split(x, [self.one, self.two], dim=1)

        x3 = self.conv1(x1)
        x3 = self.conv12(x3)
        x3 = self.conv123(x3)

        x4 = self.conv2(x2)

        x33 = self.spatial(x4) * x3
        x44 = self.channel(x3) * x4
        x5 = x33 + x44
        return x5 """



""" class FCM(nn.Module):
    def __init__(self, dim,dim_out, wg_mode="base", wg_band="hf"):
        super().__init__()
        self.one = dim // 4
        self.two = dim - dim // 4
        self.conv1 = Conv(dim // 4, dim // 4, 3, 1, 1)
        self.conv12 = Conv(dim // 4, dim // 4, 3, 1, 1)
        self.conv123 = Conv(dim // 4, dim, 1, 1)

        self.conv2 = Conv(dim - dim // 4, dim, 1, 1)
        self.conv3 = Conv(dim, dim, 1, 1)
        self.spatial = Spatial(dim)
        self.channel = Channel(dim)
        
    def forward(self, x):
        x1, x2 = torch.split(x, [self.one, self.two], dim=1)
        x3 = self.conv1(x1)
        x3 = self.conv12(x3)
        x3 = self.conv123(x3)
        x4 = self.conv2(x2)
        x33 = self.spatial(x4) * x3
        x44 = self.channel(x3) * x4
        x5 = x33 + x44
        x5 = self.conv3(x5)
        return x5 """
""" class FCM(nn.Module):
    def __init__(self, dim, dim_out, wg_mode="base", wg_band="hf"):
        super().__init__()
        assert wg_mode in ("base", "wg")
        assert wg_band in ("hf", "ll", "full")

        self.one = dim // 4
        self.two = dim - dim // 4

        self.conv1 = Conv(dim // 4, dim // 4, 3, 1, 1)
        self.conv12 = Conv(dim // 4, dim // 4, 3, 1, 1)
        self.conv123 = Conv(dim // 4, dim, 1, 1)

        self.conv2 = Conv(dim - dim // 4, dim, 1, 1)
        self.conv3 = Conv(dim, dim, 1, 1)

        if wg_mode == "base":
            self.spatial = Spatial(dim)
            self.channel = Channel(dim)
        else:
            # ✅ lazy import 防循环引用
            from .wavelet import WaveletSpatialGate, WaveletChannelGate
            self.spatial = WaveletSpatialGate(dim, mode=wg_band)
            self.channel = WaveletChannelGate(dim, mode=wg_band)

    def forward(self, x):
        x1, x2 = torch.split(x, [self.one, self.two], dim=1)
        x3 = self.conv123(self.conv12(self.conv1(x1)))
        x4 = self.conv2(x2)

        x33 = self.spatial(x4) * x3
        x44 = self.channel(x3) * x4

        x5 = self.conv3(x33 + x44)
        return x5 """

""" class FCM_3(nn.Module):
    def __init__(self, dim, dim_out, wg_mode="base", wg_band="hf"):
        super().__init__()
        self.one = dim - dim // 4
        self.two = dim // 4

        self.conv1 = Conv(dim - dim // 4, dim - dim // 4, 3, 1, 1)
        self.conv12 = Conv(dim - dim // 4, dim - dim // 4, 3, 1, 1)
        self.conv123 = Conv(dim - dim // 4, dim, 1, 1)

        self.conv2 = Conv(dim // 4, dim, 1, 1)

        # ---- gates ----
        if wg_mode == "base":
            self.spatial = Spatial(dim)
            self.channel = Channel(dim)

        elif wg_mode == "cbam":
            # ✅ 真 CBAM：spatial 用 kernel_size；channel 用 reduction
            self.spatial = CBAMSpatialGate(kernel_size=7)
            self.channel = CBAMChannelGate(dim, reduction=16)

        elif wg_mode == "wg":
            # ✅ Wavelet-guided：由你的 WaveletSpatialGate/WaveletChannelGate 决定 hf/ll/full
            self.spatial = WaveletSpatialGate(dim, mode=wg_band)               # 不传 kernel_size
            self.channel = WaveletChannelGate(dim, mode=wg_band)               # 不传 reduction（你当前实现里也没有）

        else:
            raise ValueError(wg_mode)

    def forward(self, x):
        x1, x2 = torch.split(x, [self.one, self.two], dim=1)
        x3 = self.conv1(x1)
        x3 = self.conv12(x3)
        x3 = self.conv123(x3)
        x4 = self.conv2(x2)
        x33 = self.spatial(x4) * x3
        x44 = self.channel(x3) * x4
        x5 = x33 + x44
        return x5


class FCM_2(nn.Module):
    def __init__(self, dim, dim_out, wg_mode="base", wg_band="hf"):
        super().__init__()
        self.one = dim - dim // 4
        self.two = dim // 4

        self.conv1 = Conv(dim - dim // 4, dim - dim // 4, 3, 1, 1)
        self.conv12 = Conv(dim - dim // 4, dim - dim // 4, 3, 1, 1)
        self.conv123 = Conv(dim - dim // 4, dim, 1, 1)

        self.conv2 = Conv(dim // 4, dim, 1, 1)

        # ---- gates ----
        if wg_mode == "base":
            self.spatial = Spatial(dim)
            self.channel = Channel(dim)

        elif wg_mode == "cbam":
            self.spatial = CBAMSpatialGate(kernel_size=7)
            self.channel = CBAMChannelGate(dim, reduction=16)

        elif wg_mode == "wg":
            self.spatial = WaveletSpatialGate(dim, mode=wg_band)
            self.channel = WaveletChannelGate(dim, mode=wg_band)

        else:
            raise ValueError(wg_mode)

    def forward(self, x):
        x1, x2 = torch.split(x, [self.one, self.two], dim=1)
        x3 = self.conv1(x1)
        x3 = self.conv12(x3)
        x3 = self.conv123(x3)
        x4 = self.conv2(x2)
        x33 = self.spatial(x4) * x3
        x44 = self.channel(x3) * x4
        x5 = x33 + x44
        return x5


class FCM_1(nn.Module):
    def __init__(self, dim, dim_out, wg_mode="base", wg_band="hf"):
        super().__init__()
        self.one = dim // 4
        self.two = dim - dim // 4

        self.conv1 = Conv(dim // 4, dim // 4, 3, 1, 1)
        self.conv12 = Conv(dim // 4, dim // 4, 3, 1, 1)
        self.conv123 = Conv(dim // 4, dim, 1, 1)

        self.conv2 = Conv(dim - dim // 4, dim, 1, 1)

        # ---- gates ----
        if wg_mode == "base":
            self.spatial = Spatial(dim)
            self.channel = Channel(dim)

        elif wg_mode == "cbam":
            self.spatial = CBAMSpatialGate(kernel_size=7)
            self.channel = CBAMChannelGate(dim, reduction=16)

        elif wg_mode == "wg":
            self.spatial = WaveletSpatialGate(dim, mode=wg_band)
            self.channel = WaveletChannelGate(dim, mode=wg_band)

        else:
            raise ValueError(wg_mode)

    def forward(self, x):
        x1, x2 = torch.split(x, [self.one, self.two], dim=1)
        x3 = self.conv1(x1)
        x3 = self.conv12(x3)
        x3 = self.conv123(x3)
        x4 = self.conv2(x2)
        x33 = self.spatial(x4) * x3
        x44 = self.channel(x3) * x4
        x5 = x33 + x44
        return x5


class FCM(nn.Module):
    def __init__(self, dim, dim_out, wg_mode="base", wg_band="hf"):
        super().__init__()
        self.one = dim // 4
        self.two = dim - dim // 4

        self.conv1 = Conv(dim // 4, dim // 4, 3, 1, 1)
        self.conv12 = Conv(dim // 4, dim // 4, 3, 1, 1)
        self.conv123 = Conv(dim // 4, dim, 1, 1)

        self.conv2 = Conv(dim - dim // 4, dim, 1, 1)
        self.conv3 = Conv(dim, dim, 1, 1)

        # ---- gates ----
        if wg_mode == "base":
            self.spatial = Spatial(dim)
            self.channel = Channel(dim)

        elif wg_mode == "cbam":
            self.spatial = CBAMSpatialGate(kernel_size=7)
            self.channel = CBAMChannelGate(dim, reduction=16)

        elif wg_mode == "wg":
            self.spatial = WaveletSpatialGate(dim, mode=wg_band)
            self.channel = WaveletChannelGate(dim, mode=wg_band)

        else:
            raise ValueError(wg_mode)

    def forward(self, x):
        x1, x2 = torch.split(x, [self.one, self.two], dim=1)
        x3 = self.conv1(x1)
        x3 = self.conv12(x3)
        x3 = self.conv123(x3)
        x4 = self.conv2(x2)

        x33 = self.spatial(x4) * x3
        x44 = self.channel(x3) * x4

        x5 = x33 + x44
        x5 = self.conv3(x5)
        return x5 """

class FCM_3(nn.Module):
    def __init__(self, dim, dim_out, wg_mode="base", wg_band="hf"):
        super().__init__()
        self.one = dim - dim // 4
        self.two = dim // 4

        self.conv1 = Conv(self.one, self.one, 3, 1, 1)
        self.conv12 = Conv(self.one, self.one, 3, 1, 1)
        self.conv123 = Conv(self.one, dim, 1, 1)
        self.conv2 = Conv(self.two, dim, 1, 1)

        # ---- gates ----
        self.use_couple = False
        if wg_mode == "base":
            self.spatial = Spatial(dim)
            self.channel = Channel(dim)

        elif wg_mode == "cbam":
            # 你的 CBAM 版（注意：WaveletSpatialGate/ChannelGate 要支持 mode="none"）
            self.spatial = WaveletSpatialGate(dim, mode="none", kernel_size=7)
            self.channel = WaveletChannelGate(dim, mode="none", reduction=16)

        elif wg_mode == "wg":
            # 你原来的 wavelet-guided（单向权重）
            self.spatial = WaveletSpatialGate(dim, mode=wg_band, kernel_size=7)
            self.channel = WaveletChannelGate(dim, mode=wg_band, reduction=16)

        elif wg_mode == "couple":
            # ✅ 真正耦合：LL ↔ HF 双向门控
            self.use_couple = True
            self.couple_gate = WaveletCoupledGate(dim, reduction=16)

        else:
            raise ValueError(wg_mode)

    def forward(self, x):
        x1, x2 = torch.split(x, [self.one, self.two], dim=1)

        x3 = self.conv1(x1)
        x3 = self.conv12(x3)
        x3 = self.conv123(x3)

        x4 = self.conv2(x2)

        if getattr(self, "use_couple", False):
            ll_gate, hf_gate = self.couple_gate(x)   # (B,C,1,1), (B,C,1,1)
            x33 = hf_gate * x3                        # HF 门控 x3
            x44 = ll_gate * x4                        # LL 门控 x4
        else:
            x33 = self.spatial(x4) * x3
            x44 = self.channel(x3) * x4

        x5 = x33 + x44
        return x5


class FCM_2(nn.Module):
    def __init__(self, dim, dim_out, wg_mode="base", wg_band="hf"):
        super().__init__()
        self.one = dim - dim // 4
        self.two = dim // 4

        self.conv1 = Conv(self.one, self.one, 3, 1, 1)
        self.conv12 = Conv(self.one, self.one, 3, 1, 1)
        self.conv123 = Conv(self.one, dim, 1, 1)
        self.conv2 = Conv(self.two, dim, 1, 1)

        # ---- gates ----
        self.use_couple = False
        if wg_mode == "base":
            self.spatial = Spatial(dim)
            self.channel = Channel(dim)

        elif wg_mode == "cbam":
            self.spatial = WaveletSpatialGate(dim, mode="none", kernel_size=7)
            self.channel = WaveletChannelGate(dim, mode="none", reduction=16)

        elif wg_mode == "wg":
            self.spatial = WaveletSpatialGate(dim, mode=wg_band, kernel_size=7)
            self.channel = WaveletChannelGate(dim, mode=wg_band, reduction=16)

        elif wg_mode == "couple":
            self.use_couple = True
            self.couple_gate = WaveletCoupledGate(dim, reduction=16)

        else:
            raise ValueError(wg_mode)

    def forward(self, x):
        x1, x2 = torch.split(x, [self.one, self.two], dim=1)

        x3 = self.conv1(x1)
        x3 = self.conv12(x3)
        x3 = self.conv123(x3)

        x4 = self.conv2(x2)

        if getattr(self, "use_couple", False):
            ll_gate, hf_gate = self.couple_gate(x)
            x33 = hf_gate * x3
            x44 = ll_gate * x4
        else:
            x33 = self.spatial(x4) * x3
            x44 = self.channel(x3) * x4

        x5 = x33 + x44
        return x5


class FCM_1(nn.Module):
    def __init__(self, dim, dim_out, wg_mode="base", wg_band="hf"):
        super().__init__()
        self.one = dim // 4
        self.two = dim - dim // 4

        self.conv1 = Conv(self.one, self.one, 3, 1, 1)
        self.conv12 = Conv(self.one, self.one, 3, 1, 1)
        self.conv123 = Conv(self.one, dim, 1, 1)

        self.conv2 = Conv(self.two, dim, 1, 1)

        # ---- gates ----
        self.use_couple = False
        if wg_mode == "base":
            self.spatial = Spatial(dim)
            self.channel = Channel(dim)

        elif wg_mode == "cbam":
            self.spatial = WaveletSpatialGate(dim, mode="none", kernel_size=7)
            self.channel = WaveletChannelGate(dim, mode="none", reduction=16)

        elif wg_mode == "wg":
            self.spatial = WaveletSpatialGate(dim, mode=wg_band, kernel_size=7)
            self.channel = WaveletChannelGate(dim, mode=wg_band, reduction=16)

        elif wg_mode == "couple":
            self.use_couple = True
            self.couple_gate = WaveletCoupledGate(dim, reduction=16)

        else:
            raise ValueError(wg_mode)

    def forward(self, x):
        x1, x2 = torch.split(x, [self.one, self.two], dim=1)

        x3 = self.conv1(x1)
        x3 = self.conv12(x3)
        x3 = self.conv123(x3)

        x4 = self.conv2(x2)

        if getattr(self, "use_couple", False):
            ll_gate, hf_gate = self.couple_gate(x)
            x33 = hf_gate * x3
            x44 = ll_gate * x4
        else:
            x33 = self.spatial(x4) * x3
            x44 = self.channel(x3) * x4

        x5 = x33 + x44
        return x5


class FCM(nn.Module):
    def __init__(self, dim, dim_out, wg_mode="base", wg_band="hf"):
        super().__init__()
        self.one = dim // 4
        self.two = dim - dim // 4

        self.conv1 = Conv(self.one, self.one, 3, 1, 1)
        self.conv12 = Conv(self.one, self.one, 3, 1, 1)
        self.conv123 = Conv(self.one, dim, 1, 1)

        self.conv2 = Conv(self.two, dim, 1, 1)
        self.conv3 = Conv(dim, dim, 1, 1)

        # ---- gates ----
        self.use_couple = False
        if wg_mode == "base":
            self.spatial = Spatial(dim)
            self.channel = Channel(dim)

        elif wg_mode == "cbam":
            self.spatial = WaveletSpatialGate(dim, mode="none", kernel_size=7)
            self.channel = WaveletChannelGate(dim, mode="none", reduction=16)

        elif wg_mode == "wg":
            self.spatial = WaveletSpatialGate(dim, mode=wg_band, kernel_size=7)
            self.channel = WaveletChannelGate(dim, mode=wg_band, reduction=16)

        elif wg_mode == "couple":
            self.use_couple = True
            self.couple_gate = WaveletCoupledGate(dim, reduction=16)

        else:
            raise ValueError(wg_mode)

    def forward(self, x):
        x1, x2 = torch.split(x, [self.one, self.two], dim=1)

        x3 = self.conv1(x1)
        x3 = self.conv12(x3)
        x3 = self.conv123(x3)

        x4 = self.conv2(x2)

        if getattr(self, "use_couple", False):
            ll_gate, hf_gate = self.couple_gate(x)
            x33 = hf_gate * x3
            x44 = ll_gate * x4
        else:
            x33 = self.spatial(x4) * x3
            x44 = self.channel(x3) * x4

        x5 = self.conv3(x33 + x44)
        return x5


class Pzconv(nn.Module):
    def __init__(self, dim, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv1 = nn.Conv2d(
            dim, dim, 3,
            1, 1, groups=dim
        )
        self.conv2 = Conv(dim, dim, k=1, s=1, )
        self.conv3 = nn.Conv2d(
            dim, dim, 5,
            1, 2, groups=dim
        )
        self.conv4 = Conv(dim, dim, 1, 1)
        self.conv5 = nn.Conv2d(
            dim, dim, 7,
            1, 3, groups=dim
        )

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        x6 = x5 + x
        return x6


class Down(nn.Module):
    def __init__(self, dim, dim_out):
        super().__init__()
        self.conv2 = Conv(dim, dim, 3, 2, 1, g=dim // 2, act=False)
        self.conv4 = Conv(dim, dim_out, 1, 1)

    def forward(self, x):
        x2 = self.conv2(x)
        x2 = self.conv4(x2)
        return x2

class FCM_3_Res(nn.Module):
    def __init__(self, dim, dim_out, wg_mode="base", wg_band="hf"):
        super().__init__()
        self.one = dim - dim // 4
        self.two = dim // 4

        self.conv1 = Conv(self.one, self.one, 3, 1, 1)
        self.conv12 = Conv(self.one, self.one, 3, 1, 1)
        self.conv123 = Conv(self.one, dim, 1, 1)
        self.conv2 = Conv(self.two, dim, 1, 1)

        self.use_couple = False
        if wg_mode == "base":
            self.spatial = Spatial(dim)
            self.channel = Channel(dim)
        elif wg_mode == "cbam":
            self.spatial = WaveletSpatialGate(dim, mode="none", kernel_size=7)
            self.channel = WaveletChannelGate(dim, mode="none", reduction=16)
        elif wg_mode == "wg":
            self.spatial = WaveletSpatialGate(dim, mode=wg_band, kernel_size=7)
            self.channel = WaveletChannelGate(dim, mode=wg_band, reduction=16)
        elif wg_mode == "couple":
            self.use_couple = True
            self.couple_gate = WaveletCoupledGate(dim, reduction=16)
            self.alpha = nn.Parameter(torch.tensor(0.1))
        else:
            raise ValueError(wg_mode)

    def forward(self, x):
        x1, x2 = torch.split(x, [self.one, self.two], dim=1)

        x3 = self.conv1(x1)
        x3 = self.conv12(x3)
        x3 = self.conv123(x3)
        x4 = self.conv2(x2)

        if getattr(self, "use_couple", False):
            ll_gate, hf_gate = self.couple_gate(x)
            x33 = hf_gate * x3
            x44 = ll_gate * x4

            base = x3 + x4
            enhanced = x33 + x44
            return base + self.alpha * (enhanced - base)
        else:
            x33 = self.spatial(x4) * x3
            x44 = self.channel(x3) * x4
            return x33 + x44


class FCM_2_Res(nn.Module):
    def __init__(self, dim, dim_out, wg_mode="base", wg_band="hf"):
        super().__init__()
        self.one = dim - dim // 4
        self.two = dim // 4

        self.conv1 = Conv(self.one, self.one, 3, 1, 1)
        self.conv12 = Conv(self.one, self.one, 3, 1, 1)
        self.conv123 = Conv(self.one, dim, 1, 1)
        self.conv2 = Conv(self.two, dim, 1, 1)

        self.use_couple = False
        if wg_mode == "base":
            self.spatial = Spatial(dim)
            self.channel = Channel(dim)
        elif wg_mode == "cbam":
            self.spatial = WaveletSpatialGate(dim, mode="none", kernel_size=7)
            self.channel = WaveletChannelGate(dim, mode="none", reduction=16)
        elif wg_mode == "wg":
            self.spatial = WaveletSpatialGate(dim, mode=wg_band, kernel_size=7)
            self.channel = WaveletChannelGate(dim, mode=wg_band, reduction=16)
        elif wg_mode == "couple":
            self.use_couple = True
            self.couple_gate = WaveletCoupledGate(dim, reduction=16)
            self.alpha = nn.Parameter(torch.tensor(0.1))
        else:
            raise ValueError(wg_mode)

    def forward(self, x):
        x1, x2 = torch.split(x, [self.one, self.two], dim=1)

        x3 = self.conv1(x1)
        x3 = self.conv12(x3)
        x3 = self.conv123(x3)
        x4 = self.conv2(x2)

        if getattr(self, "use_couple", False):
            ll_gate, hf_gate = self.couple_gate(x)
            x33 = hf_gate * x3
            x44 = ll_gate * x4

            base = x3 + x4
            enhanced = x33 + x44
            return base + self.alpha * (enhanced - base)
        else:
            x33 = self.spatial(x4) * x3
            x44 = self.channel(x3) * x4
            return x33 + x44


class FCM_1_Res(nn.Module):
    def __init__(self, dim, dim_out, wg_mode="base", wg_band="hf"):
        super().__init__()
        self.one = dim // 4
        self.two = dim - dim // 4

        self.conv1 = Conv(self.one, self.one, 3, 1, 1)
        self.conv12 = Conv(self.one, self.one, 3, 1, 1)
        self.conv123 = Conv(self.one, dim, 1, 1)
        self.conv2 = Conv(self.two, dim, 1, 1)

        self.use_couple = False
        if wg_mode == "base":
            self.spatial = Spatial(dim)
            self.channel = Channel(dim)
        elif wg_mode == "cbam":
            self.spatial = WaveletSpatialGate(dim, mode="none", kernel_size=7)
            self.channel = WaveletChannelGate(dim, mode="none", reduction=16)
        elif wg_mode == "wg":
            self.spatial = WaveletSpatialGate(dim, mode=wg_band, kernel_size=7)
            self.channel = WaveletChannelGate(dim, mode=wg_band, reduction=16)
        elif wg_mode == "couple":
            self.use_couple = True
            self.couple_gate = WaveletCoupledGate(dim, reduction=16)
            self.alpha = nn.Parameter(torch.tensor(0.1))
        else:
            raise ValueError(wg_mode)

    def forward(self, x):
        x1, x2 = torch.split(x, [self.one, self.two], dim=1)

        x3 = self.conv1(x1)
        x3 = self.conv12(x3)
        x3 = self.conv123(x3)
        x4 = self.conv2(x2)

        if getattr(self, "use_couple", False):
            ll_gate, hf_gate = self.couple_gate(x)
            x33 = hf_gate * x3
            x44 = ll_gate * x4

            base = x3 + x4
            enhanced = x33 + x44
            return base + self.alpha * (enhanced - base)
        else:
            x33 = self.spatial(x4) * x3
            x44 = self.channel(x3) * x4
            return x33 + x44


class FCM_Res(nn.Module):
    def __init__(self, dim, dim_out, wg_mode="base", wg_band="hf"):
        super().__init__()
        self.one = dim // 4
        self.two = dim - dim // 4

        self.conv1 = Conv(self.one, self.one, 3, 1, 1)
        self.conv12 = Conv(self.one, self.one, 3, 1, 1)
        self.conv123 = Conv(self.one, dim, 1, 1)
        self.conv2 = Conv(self.two, dim, 1, 1)
        self.conv3 = Conv(dim, dim, 1, 1)

        self.use_couple = False
        if wg_mode == "base":
            self.spatial = Spatial(dim)
            self.channel = Channel(dim)
        elif wg_mode == "cbam":
            self.spatial = WaveletSpatialGate(dim, mode="none", kernel_size=7)
            self.channel = WaveletChannelGate(dim, mode="none", reduction=16)
        elif wg_mode == "wg":
            self.spatial = WaveletSpatialGate(dim, mode=wg_band, reduction=16)
            self.channel = WaveletChannelGate(dim, mode=wg_band, reduction=16)
        elif wg_mode == "couple":
            self.use_couple = True
            self.couple_gate = WaveletCoupledGate(dim, reduction=16)
            self.alpha = nn.Parameter(torch.tensor(0.1))
        else:
            raise ValueError(wg_mode)

    def forward(self, x):
        x1, x2 = torch.split(x, [self.one, self.two], dim=1)

        x3 = self.conv1(x1)
        x3 = self.conv12(x3)
        x3 = self.conv123(x3)
        x4 = self.conv2(x2)

        if getattr(self, "use_couple", False):
            ll_gate, hf_gate = self.couple_gate(x)
            x33 = hf_gate * x3
            x44 = ll_gate * x4

            base = x3 + x4
            enhanced = x33 + x44
            x5 = base + self.alpha * (enhanced - base)
        else:
            x33 = self.spatial(x4) * x3
            x44 = self.channel(x3) * x4
            x5 = x33 + x44

        return self.conv3(x5)

class FCM_3_HC(nn.Module):
    def __init__(self, dim, dim_out, wg_mode="base", wg_band="hf"):
        super().__init__()
        self.one = dim - dim // 4
        self.two = dim // 4

        self.conv1 = Conv(self.one, self.one, 3, 1, 1)
        self.conv12 = Conv(self.one, self.one, 3, 1, 1)
        self.conv123 = Conv(self.one, dim, 1, 1)
        self.conv2 = Conv(self.two, dim, 1, 1)

        self.use_couple = False
        if wg_mode == "couple":
            self.use_couple = True
            self.couple_gate = WaveletCoupledGate_HC(dim, reduction=16)
        elif wg_mode == "base":
            self.spatial = Spatial(dim)
            self.channel = Channel(dim)
        elif wg_mode == "cbam":
            self.spatial = WaveletSpatialGate(dim, mode="none", kernel_size=7)
            self.channel = WaveletChannelGate(dim, mode="none", reduction=16)
        elif wg_mode == "wg":
            self.spatial = WaveletSpatialGate(dim, mode=wg_band, kernel_size=7)
            self.channel = WaveletChannelGate(dim, mode=wg_band, reduction=16)
        else:
            raise ValueError(wg_mode)

    def forward(self, x):
        x1, x2 = torch.split(x, [self.one, self.two], dim=1)
        x3 = self.conv1(x1)
        x3 = self.conv12(x3)
        x3 = self.conv123(x3)
        x4 = self.conv2(x2)

        if getattr(self, "use_couple", False):
            ll_gate, hf_gate = self.couple_gate(x)
            x33 = hf_gate * x3
            x44 = ll_gate * x4
        else:
            x33 = self.spatial(x4) * x3
            x44 = self.channel(x3) * x4

        return x33 + x44


class FCM_2_HC(nn.Module):
    def __init__(self, dim, dim_out, wg_mode="base", wg_band="hf"):
        super().__init__()
        self.one = dim - dim // 4
        self.two = dim // 4

        self.conv1 = Conv(self.one, self.one, 3, 1, 1)
        self.conv12 = Conv(self.one, self.one, 3, 1, 1)
        self.conv123 = Conv(self.one, dim, 1, 1)
        self.conv2 = Conv(self.two, dim, 1, 1)

        self.use_couple = False
        if wg_mode == "couple":
            self.use_couple = True
            self.couple_gate = WaveletCoupledGate_HC(dim, reduction=16)
        elif wg_mode == "base":
            self.spatial = Spatial(dim)
            self.channel = Channel(dim)
        elif wg_mode == "cbam":
            self.spatial = WaveletSpatialGate(dim, mode="none", kernel_size=7)
            self.channel = WaveletChannelGate(dim, mode="none", reduction=16)
        elif wg_mode == "wg":
            self.spatial = WaveletSpatialGate(dim, mode=wg_band, kernel_size=7)
            self.channel = WaveletChannelGate(dim, mode=wg_band, reduction=16)
        else:
            raise ValueError(wg_mode)

    def forward(self, x):
        x1, x2 = torch.split(x, [self.one, self.two], dim=1)
        x3 = self.conv1(x1)
        x3 = self.conv12(x3)
        x3 = self.conv123(x3)
        x4 = self.conv2(x2)

        if getattr(self, "use_couple", False):
            ll_gate, hf_gate = self.couple_gate(x)
            x33 = hf_gate * x3
            x44 = ll_gate * x4
        else:
            x33 = self.spatial(x4) * x3
            x44 = self.channel(x3) * x4

        return x33 + x44


class FCM_1_HC(nn.Module):
    def __init__(self, dim, dim_out, wg_mode="base", wg_band="hf"):
        super().__init__()
        self.one = dim // 4
        self.two = dim - dim // 4

        self.conv1 = Conv(self.one, self.one, 3, 1, 1)
        self.conv12 = Conv(self.one, self.one, 3, 1, 1)
        self.conv123 = Conv(self.one, dim, 1, 1)
        self.conv2 = Conv(self.two, dim, 1, 1)

        self.use_couple = False
        if wg_mode == "couple":
            self.use_couple = True
            self.couple_gate = WaveletCoupledGate_HC(dim, reduction=16)
        elif wg_mode == "base":
            self.spatial = Spatial(dim)
            self.channel = Channel(dim)
        elif wg_mode == "cbam":
            self.spatial = WaveletSpatialGate(dim, mode="none", kernel_size=7)
            self.channel = WaveletChannelGate(dim, mode="none", reduction=16)
        elif wg_mode == "wg":
            self.spatial = WaveletSpatialGate(dim, mode=wg_band, kernel_size=7)
            self.channel = WaveletChannelGate(dim, mode=wg_band, reduction=16)
        else:
            raise ValueError(wg_mode)

    def forward(self, x):
        x1, x2 = torch.split(x, [self.one, self.two], dim=1)
        x3 = self.conv1(x1)
        x3 = self.conv12(x3)
        x3 = self.conv123(x3)
        x4 = self.conv2(x2)

        if getattr(self, "use_couple", False):
            ll_gate, hf_gate = self.couple_gate(x)
            x33 = hf_gate * x3
            x44 = ll_gate * x4
        else:
            x33 = self.spatial(x4) * x3
            x44 = self.channel(x3) * x4

        return x33 + x44


class FCM_HC(nn.Module):
    def __init__(self, dim, dim_out, wg_mode="base", wg_band="hf"):
        super().__init__()
        self.one = dim // 4
        self.two = dim - dim // 4

        self.conv1 = Conv(self.one, self.one, 3, 1, 1)
        self.conv12 = Conv(self.one, self.one, 3, 1, 1)
        self.conv123 = Conv(self.one, dim, 1, 1)
        self.conv2 = Conv(self.two, dim, 1, 1)
        self.conv3 = Conv(dim, dim, 1, 1)

        self.use_couple = False
        if wg_mode == "couple":
            self.use_couple = True
            self.couple_gate = WaveletCoupledGate_HC(dim, reduction=16)
        elif wg_mode == "base":
            self.spatial = Spatial(dim)
            self.channel = Channel(dim)
        elif wg_mode == "cbam":
            self.spatial = WaveletSpatialGate(dim, mode="none", kernel_size=7)
            self.channel = WaveletChannelGate(dim, mode="none", reduction=16)
        elif wg_mode == "wg":
            self.spatial = WaveletSpatialGate(dim, mode=wg_band, kernel_size=7)
            self.channel = WaveletChannelGate(dim, mode=wg_band, reduction=16)
        else:
            raise ValueError(wg_mode)

    def forward(self, x):
        x1, x2 = torch.split(x, [self.one, self.two], dim=1)
        x3 = self.conv1(x1)
        x3 = self.conv12(x3)
        x3 = self.conv123(x3)
        x4 = self.conv2(x2)

        if getattr(self, "use_couple", False):
            ll_gate, hf_gate = self.couple_gate(x)
            x33 = hf_gate * x3
            x44 = ll_gate * x4
        else:
            x33 = self.spatial(x4) * x3
            x44 = self.channel(x3) * x4

        return self.conv3(x33 + x44)


class FCM_3_MS(nn.Module):
    """
    Multi-scale FCM_3.
    This module introduces a lightweight multi-scale branch to enhance
    local detail and context representation for small UAV objects.
    """
    def __init__(self, dim, dim_out, wg_mode="base", wg_band="hf"):
        super().__init__()
        self.one = dim - dim // 4
        self.two = dim // 4

        # Main branch
        self.conv1 = Conv(self.one, self.one, 3, 1, 1)

        # Multi-scale lightweight branches
        self.dw3 = nn.Conv2d(self.one, self.one, kernel_size=3, stride=1, padding=1,
                             groups=self.one, bias=False)
        self.dw5 = nn.Conv2d(self.one, self.one, kernel_size=5, stride=1, padding=2,
                             groups=self.one, bias=False)
        self.ms_bn = nn.BatchNorm2d(self.one)
        self.ms_act = nn.SiLU()

        # Fuse multi-scale features
        self.ms_fuse = Conv(self.one * 2, self.one, 1, 1)

        self.conv123 = Conv(self.one, dim, 1, 1)
        self.conv2 = Conv(self.two, dim, 1, 1)

        self.use_couple = False
        if wg_mode == "base":
            self.spatial = Spatial(dim)
            self.channel = Channel(dim)

        elif wg_mode == "cbam":
            self.spatial = WaveletSpatialGate(dim, mode="none", kernel_size=7)
            self.channel = WaveletChannelGate(dim, mode="none", reduction=16)

        elif wg_mode == "wg":
            self.spatial = WaveletSpatialGate(dim, mode=wg_band, kernel_size=7)
            self.channel = WaveletChannelGate(dim, mode=wg_band, reduction=16)

        elif wg_mode == "couple":
            self.use_couple = True
            self.couple_gate = WaveletCoupledGate(dim, reduction=16)

        else:
            raise ValueError(wg_mode)

    def forward(self, x):
        x1, x2 = torch.split(x, [self.one, self.two], dim=1)

        x3 = self.conv1(x1)

        ms3 = self.ms_act(self.ms_bn(self.dw3(x3)))
        ms5 = self.ms_act(self.ms_bn(self.dw5(x3)))
        x3 = self.ms_fuse(torch.cat([ms3, ms5], dim=1))

        x3 = self.conv123(x3)
        x4 = self.conv2(x2)

        if getattr(self, "use_couple", False):
            ll_gate, hf_gate = self.couple_gate(x)
            x33 = hf_gate * x3
            x44 = ll_gate * x4
        else:
            x33 = self.spatial(x4) * x3
            x44 = self.channel(x3) * x4

        return x33 + x44


class FCM_2_MS(nn.Module):
    """
    Multi-scale FCM_2.
    """
    def __init__(self, dim, dim_out, wg_mode="base", wg_band="hf"):
        super().__init__()
        self.one = dim - dim // 4
        self.two = dim // 4

        self.conv1 = Conv(self.one, self.one, 3, 1, 1)

        self.dw3 = nn.Conv2d(self.one, self.one, kernel_size=3, stride=1, padding=1,
                             groups=self.one, bias=False)
        self.dw5 = nn.Conv2d(self.one, self.one, kernel_size=5, stride=1, padding=2,
                             groups=self.one, bias=False)
        self.ms_bn = nn.BatchNorm2d(self.one)
        self.ms_act = nn.SiLU()
        self.ms_fuse = Conv(self.one * 2, self.one, 1, 1)

        self.conv123 = Conv(self.one, dim, 1, 1)
        self.conv2 = Conv(self.two, dim, 1, 1)

        self.use_couple = False
        if wg_mode == "base":
            self.spatial = Spatial(dim)
            self.channel = Channel(dim)

        elif wg_mode == "cbam":
            self.spatial = WaveletSpatialGate(dim, mode="none", kernel_size=7)
            self.channel = WaveletChannelGate(dim, mode="none", reduction=16)

        elif wg_mode == "wg":
            self.spatial = WaveletSpatialGate(dim, mode=wg_band, kernel_size=7)
            self.channel = WaveletChannelGate(dim, mode=wg_band, reduction=16)

        elif wg_mode == "couple":
            self.use_couple = True
            self.couple_gate = WaveletCoupledGate(dim, reduction=16)

        else:
            raise ValueError(wg_mode)

    def forward(self, x):
        x1, x2 = torch.split(x, [self.one, self.two], dim=1)

        x3 = self.conv1(x1)

        ms3 = self.ms_act(self.ms_bn(self.dw3(x3)))
        ms5 = self.ms_act(self.ms_bn(self.dw5(x3)))
        x3 = self.ms_fuse(torch.cat([ms3, ms5], dim=1))

        x3 = self.conv123(x3)
        x4 = self.conv2(x2)

        if getattr(self, "use_couple", False):
            ll_gate, hf_gate = self.couple_gate(x)
            x33 = hf_gate * x3
            x44 = ll_gate * x4
        else:
            x33 = self.spatial(x4) * x3
            x44 = self.channel(x3) * x4

        return x33 + x44


class FCM_1_MS(nn.Module):
    """
    Multi-scale FCM_1.
    """
    def __init__(self, dim, dim_out, wg_mode="base", wg_band="hf"):
        super().__init__()
        self.one = dim // 4
        self.two = dim - dim // 4

        self.conv1 = Conv(self.one, self.one, 3, 1, 1)

        self.dw3 = nn.Conv2d(self.one, self.one, kernel_size=3, stride=1, padding=1,
                             groups=self.one, bias=False)
        self.dw5 = nn.Conv2d(self.one, self.one, kernel_size=5, stride=1, padding=2,
                             groups=self.one, bias=False)
        self.ms_bn = nn.BatchNorm2d(self.one)
        self.ms_act = nn.SiLU()
        self.ms_fuse = Conv(self.one * 2, self.one, 1, 1)

        self.conv123 = Conv(self.one, dim, 1, 1)
        self.conv2 = Conv(self.two, dim, 1, 1)

        self.use_couple = False
        if wg_mode == "base":
            self.spatial = Spatial(dim)
            self.channel = Channel(dim)

        elif wg_mode == "cbam":
            self.spatial = WaveletSpatialGate(dim, mode="none", kernel_size=7)
            self.channel = WaveletChannelGate(dim, mode="none", reduction=16)

        elif wg_mode == "wg":
            self.spatial = WaveletSpatialGate(dim, mode=wg_band, kernel_size=7)
            self.channel = WaveletChannelGate(dim, mode=wg_band, reduction=16)

        elif wg_mode == "couple":
            self.use_couple = True
            self.couple_gate = WaveletCoupledGate(dim, reduction=16)

        else:
            raise ValueError(wg_mode)

    def forward(self, x):
        x1, x2 = torch.split(x, [self.one, self.two], dim=1)

        x3 = self.conv1(x1)

        ms3 = self.ms_act(self.ms_bn(self.dw3(x3)))
        ms5 = self.ms_act(self.ms_bn(self.dw5(x3)))
        x3 = self.ms_fuse(torch.cat([ms3, ms5], dim=1))

        x3 = self.conv123(x3)
        x4 = self.conv2(x2)

        if getattr(self, "use_couple", False):
            ll_gate, hf_gate = self.couple_gate(x)
            x33 = hf_gate * x3
            x44 = ll_gate * x4
        else:
            x33 = self.spatial(x4) * x3
            x44 = self.channel(x3) * x4

        return x33 + x44


class FCM_MS(nn.Module):
    """
    Multi-scale FCM.
    """
    def __init__(self, dim, dim_out, wg_mode="base", wg_band="hf"):
        super().__init__()
        self.one = dim // 4
        self.two = dim - dim // 4

        self.conv1 = Conv(self.one, self.one, 3, 1, 1)

        self.dw3 = nn.Conv2d(self.one, self.one, kernel_size=3, stride=1, padding=1,
                             groups=self.one, bias=False)
        self.dw5 = nn.Conv2d(self.one, self.one, kernel_size=5, stride=1, padding=2,
                             groups=self.one, bias=False)
        self.ms_bn = nn.BatchNorm2d(self.one)
        self.ms_act = nn.SiLU()
        self.ms_fuse = Conv(self.one * 2, self.one, 1, 1)

        self.conv123 = Conv(self.one, dim, 1, 1)
        self.conv2 = Conv(self.two, dim, 1, 1)
        self.conv3 = Conv(dim, dim, 1, 1)

        self.use_couple = False
        if wg_mode == "base":
            self.spatial = Spatial(dim)
            self.channel = Channel(dim)

        elif wg_mode == "cbam":
            self.spatial = WaveletSpatialGate(dim, mode="none", kernel_size=7)
            self.channel = WaveletChannelGate(dim, mode="none", reduction=16)

        elif wg_mode == "wg":
            self.spatial = WaveletSpatialGate(dim, mode=wg_band, kernel_size=7)
            self.channel = WaveletChannelGate(dim, mode=wg_band, reduction=16)

        elif wg_mode == "couple":
            self.use_couple = True
            self.couple_gate = WaveletCoupledGate(dim, reduction=16)

        else:
            raise ValueError(wg_mode)

    def forward(self, x):
        x1, x2 = torch.split(x, [self.one, self.two], dim=1)

        x3 = self.conv1(x1)

        ms3 = self.ms_act(self.ms_bn(self.dw3(x3)))
        ms5 = self.ms_act(self.ms_bn(self.dw5(x3)))
        x3 = self.ms_fuse(torch.cat([ms3, ms5], dim=1))

        x3 = self.conv123(x3)
        x4 = self.conv2(x2)

        if getattr(self, "use_couple", False):
            ll_gate, hf_gate = self.couple_gate(x)
            x33 = hf_gate * x3
            x44 = ll_gate * x4
        else:
            x33 = self.spatial(x4) * x3
            x44 = self.channel(x3) * x4

        x5 = self.conv3(x33 + x44)
        return x5

class FCM_3_SD(nn.Module):
    """
    Small-object Detail-enhanced FCM_3.

    The original FCM branch is preserved, and an additional lightweight
    local detail branch is introduced to enhance small-object edge and
    texture representation.
    """
    def __init__(self, dim, dim_out, wg_mode="base", wg_band="hf"):
        super().__init__()
        self.one = dim - dim // 4
        self.two = dim // 4

        # Original FCM main branch
        self.conv1 = Conv(self.one, self.one, 3, 1, 1)
        self.conv12 = Conv(self.one, self.one, 3, 1, 1)
        self.conv123 = Conv(self.one, dim, 1, 1)

        # Original FCM complementary branch
        self.conv2 = Conv(self.two, dim, 1, 1)

        # Lightweight small-object detail branch
        self.detail_dw = nn.Conv2d(
            self.one, self.one,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=self.one,
            bias=False
        )
        self.detail_bn = nn.BatchNorm2d(self.one)
        self.detail_act = nn.SiLU()
        self.detail_pw = Conv(self.one, dim, 1, 1)

        # Learnable detail scaling coefficient
        self.beta = nn.Parameter(torch.tensor(0.1))

        # Gates
                # Gates
        self.use_couple = False
        self.use_weak = False

        if wg_mode == "base":
            self.spatial = Spatial(dim)
            self.channel = Channel(dim)

        elif wg_mode == "cbam":
            self.spatial = WaveletSpatialGate(dim, mode="none", kernel_size=7)
            self.channel = WaveletChannelGate(dim, mode="none", reduction=16)

        elif wg_mode == "wg":
            self.spatial = WaveletSpatialGate(dim, mode=wg_band, kernel_size=7)
            self.channel = WaveletChannelGate(dim, mode=wg_band, reduction=16)

        elif wg_mode == "couple":
            # Strong frequency-guided coupled modulation
            self.use_couple = True
            self.use_weak = False
            self.couple_gate = WaveletCoupledGate(dim, reduction=16)

        elif wg_mode == "weakcouple":
            # Weak frequency-guided coupled modulation
            # x -> x * (1 + gamma * gate)
            self.use_couple = True
            self.use_weak = True
            self.couple_gate = WaveletCoupledGate(dim, reduction=16)
            self.gamma = nn.Parameter(torch.tensor(0.1))

        else:
            raise ValueError(wg_mode)

    def forward(self, x):
        x1, x2 = torch.split(x, [self.one, self.two], dim=1)

        # Original FCM main branch
        x3 = self.conv1(x1)
        x3 = self.conv12(x3)
        x3_main = self.conv123(x3)

        # Detail branch from shallow split feature
        detail = self.detail_dw(x1)
        detail = self.detail_bn(detail)
        detail = self.detail_act(detail)
        detail = self.detail_pw(detail)

        # Detail-enhanced main feature
        x3 = x3_main + self.beta * detail

        # Complementary branch
        x4 = self.conv2(x2)

        if getattr(self, "use_couple", False):
            ll_gate, hf_gate = self.couple_gate(x)

            if getattr(self, "use_weak", False):
                gamma = torch.clamp(self.gamma, 0.0, 0.5)
                x33 = x3 * (1.0 + gamma * hf_gate)
                x44 = x4 * (1.0 + gamma * ll_gate)
            else:
                x33 = hf_gate * x3
                x44 = ll_gate * x4
        else:
            x33 = self.spatial(x4) * x3
            x44 = self.channel(x3) * x4

        return x33 + x44


class FCM_2_SD(nn.Module):
    """
    Small-object Detail-enhanced FCM_2.
    """
    def __init__(self, dim, dim_out, wg_mode="base", wg_band="hf"):
        super().__init__()
        self.one = dim - dim // 4
        self.two = dim // 4

        self.conv1 = Conv(self.one, self.one, 3, 1, 1)
        self.conv12 = Conv(self.one, self.one, 3, 1, 1)
        self.conv123 = Conv(self.one, dim, 1, 1)

        self.conv2 = Conv(self.two, dim, 1, 1)

        self.detail_dw = nn.Conv2d(
            self.one, self.one,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=self.one,
            bias=False
        )
        self.detail_bn = nn.BatchNorm2d(self.one)
        self.detail_act = nn.SiLU()
        self.detail_pw = Conv(self.one, dim, 1, 1)

        self.beta = nn.Parameter(torch.tensor(0.1))

        self.use_couple = False
        if wg_mode == "base":
            self.spatial = Spatial(dim)
            self.channel = Channel(dim)

        elif wg_mode == "cbam":
            self.spatial = WaveletSpatialGate(dim, mode="none", kernel_size=7)
            self.channel = WaveletChannelGate(dim, mode="none", reduction=16)

        elif wg_mode == "wg":
            self.spatial = WaveletSpatialGate(dim, mode=wg_band, kernel_size=7)
            self.channel = WaveletChannelGate(dim, mode=wg_band, reduction=16)

        elif wg_mode == "couple":
            self.use_couple = True
            self.couple_gate = WaveletCoupledGate(dim, reduction=16)

        else:
            raise ValueError(wg_mode)

    def forward(self, x):
        x1, x2 = torch.split(x, [self.one, self.two], dim=1)

        x3 = self.conv1(x1)
        x3 = self.conv12(x3)
        x3_main = self.conv123(x3)

        detail = self.detail_dw(x1)
        detail = self.detail_bn(detail)
        detail = self.detail_act(detail)
        detail = self.detail_pw(detail)

        x3 = x3_main + self.beta * detail

        x4 = self.conv2(x2)

        if getattr(self, "use_couple", False):
            ll_gate, hf_gate = self.couple_gate(x)
            x33 = hf_gate * x3
            x44 = ll_gate * x4
        else:
            x33 = self.spatial(x4) * x3
            x44 = self.channel(x3) * x4

        return x33 + x44


class FCM_1_SD(nn.Module):
    """
    Small-object Detail-enhanced FCM_1.
    """
    def __init__(self, dim, dim_out, wg_mode="base", wg_band="hf"):
        super().__init__()
        self.one = dim // 4
        self.two = dim - dim // 4

        self.conv1 = Conv(self.one, self.one, 3, 1, 1)
        self.conv12 = Conv(self.one, self.one, 3, 1, 1)
        self.conv123 = Conv(self.one, dim, 1, 1)

        self.conv2 = Conv(self.two, dim, 1, 1)

        self.detail_dw = nn.Conv2d(
            self.one, self.one,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=self.one,
            bias=False
        )
        self.detail_bn = nn.BatchNorm2d(self.one)
        self.detail_act = nn.SiLU()
        self.detail_pw = Conv(self.one, dim, 1, 1)

        self.beta = nn.Parameter(torch.tensor(0.1))

        self.use_couple = False
        if wg_mode == "base":
            self.spatial = Spatial(dim)
            self.channel = Channel(dim)

        elif wg_mode == "cbam":
            self.spatial = WaveletSpatialGate(dim, mode="none", kernel_size=7)
            self.channel = WaveletChannelGate(dim, mode="none", reduction=16)

        elif wg_mode == "wg":
            self.spatial = WaveletSpatialGate(dim, mode=wg_band, kernel_size=7)
            self.channel = WaveletChannelGate(dim, mode=wg_band, reduction=16)

        elif wg_mode == "couple":
            self.use_couple = True
            self.couple_gate = WaveletCoupledGate(dim, reduction=16)

        else:
            raise ValueError(wg_mode)

    def forward(self, x):
        x1, x2 = torch.split(x, [self.one, self.two], dim=1)

        x3 = self.conv1(x1)
        x3 = self.conv12(x3)
        x3_main = self.conv123(x3)

        detail = self.detail_dw(x1)
        detail = self.detail_bn(detail)
        detail = self.detail_act(detail)
        detail = self.detail_pw(detail)

        x3 = x3_main + self.beta * detail

        x4 = self.conv2(x2)

        if getattr(self, "use_couple", False):
            ll_gate, hf_gate = self.couple_gate(x)
            x33 = hf_gate * x3
            x44 = ll_gate * x4
        else:
            x33 = self.spatial(x4) * x3
            x44 = self.channel(x3) * x4

        return x33 + x44


class FCM_SD(nn.Module):
    """
    Small-object Detail-enhanced FCM.
    """
    def __init__(self, dim, dim_out, wg_mode="base", wg_band="hf"):
        super().__init__()
        self.one = dim // 4
        self.two = dim - dim // 4

        self.conv1 = Conv(self.one, self.one, 3, 1, 1)
        self.conv12 = Conv(self.one, self.one, 3, 1, 1)
        self.conv123 = Conv(self.one, dim, 1, 1)

        self.conv2 = Conv(self.two, dim, 1, 1)
        self.conv3 = Conv(dim, dim, 1, 1)

        self.detail_dw = nn.Conv2d(
            self.one, self.one,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=self.one,
            bias=False
        )
        self.detail_bn = nn.BatchNorm2d(self.one)
        self.detail_act = nn.SiLU()
        self.detail_pw = Conv(self.one, dim, 1, 1)

        self.beta = nn.Parameter(torch.tensor(0.1))

        self.use_couple = False
        if wg_mode == "base":
            self.spatial = Spatial(dim)
            self.channel = Channel(dim)

        elif wg_mode == "cbam":
            self.spatial = WaveletSpatialGate(dim, mode="none", kernel_size=7)
            self.channel = WaveletChannelGate(dim, mode="none", reduction=16)

        elif wg_mode == "wg":
            self.spatial = WaveletSpatialGate(dim, mode=wg_band, kernel_size=7)
            self.channel = WaveletChannelGate(dim, mode=wg_band, reduction=16)

        elif wg_mode == "couple":
            self.use_couple = True
            self.couple_gate = WaveletCoupledGate(dim, reduction=16)

        else:
            raise ValueError(wg_mode)

    def forward(self, x):
        x1, x2 = torch.split(x, [self.one, self.two], dim=1)

        x3 = self.conv1(x1)
        x3 = self.conv12(x3)
        x3_main = self.conv123(x3)

        detail = self.detail_dw(x1)
        detail = self.detail_bn(detail)
        detail = self.detail_act(detail)
        detail = self.detail_pw(detail)

        x3 = x3_main + self.beta * detail

        x4 = self.conv2(x2)

        if getattr(self, "use_couple", False):
            ll_gate, hf_gate = self.couple_gate(x)
            x33 = hf_gate * x3
            x44 = ll_gate * x4
        else:
            x33 = self.spatial(x4) * x3
            x44 = self.channel(x3) * x4

        x5 = self.conv3(x33 + x44)
        return x5

class FCM_3_SD_HF(nn.Module):
    """
    Shallow Detail-enhanced FCM with High-Frequency Residual Enhancement.

    This module preserves the original FCM_3 mapping path, adds a lightweight
    spatial detail branch, and further injects DWT-based high-frequency residual
    features. The frequency branch is used as a residual enhancement rather than
    a multiplicative gate, avoiding suppression of the original features.

    Input:  (B, C, H, W)
    Output: (B, C, H, W)
    """
    def __init__(self, dim, dim_out, wg_mode="base", wg_band="hf"):
        super().__init__()
        self.one = dim - dim // 4
        self.two = dim // 4

        # Original FCM main branch
        self.conv1 = Conv(self.one, self.one, 3, 1, 1)
        self.conv12 = Conv(self.one, self.one, 3, 1, 1)
        self.conv123 = Conv(self.one, dim, 1, 1)

        # Original FCM complementary branch
        self.conv2 = Conv(self.two, dim, 1, 1)

        # Spatial detail branch
        self.detail_dw = nn.Conv2d(
            self.one, self.one,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=self.one,
            bias=False
        )
        self.detail_bn = nn.BatchNorm2d(self.one)
        self.detail_act = nn.SiLU()
        self.detail_pw = Conv(self.one, dim, 1, 1)

        # Learnable spatial detail scale
        self.beta = nn.Parameter(torch.tensor(0.1))

        # Haar high-frequency residual branch
        # DWT_Haar_HF outputs 3 * self.one channels with H/2, W/2
        self.hf_dwt = DWT_Haar_HF()
        self.hf_reduce = nn.Sequential(
            nn.Conv2d(self.one * 3, self.one, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.one),
            nn.SiLU()
        )
        self.hf_pw = Conv(self.one, dim, 1, 1)

        # Learnable high-frequency residual scale
        self.alpha = nn.Parameter(torch.tensor(0.05))

        # Original FCM gates
        self.spatial = Spatial(dim)
        self.channel = Channel(dim)

    def forward(self, x):
        x1, x2 = torch.split(x, [self.one, self.two], dim=1)

        # Original FCM main branch
        x3 = self.conv1(x1)
        x3 = self.conv12(x3)
        x3_main = self.conv123(x3)

        # Spatial detail residual branch
        detail = self.detail_dw(x1)
        detail = self.detail_bn(detail)
        detail = self.detail_act(detail)
        detail = self.detail_pw(detail)

        # Frequency high-frequency residual branch
        hf = self.hf_dwt(x1)                         # (B, 3*self.one, H/2, W/2)
        hf = self.hf_reduce(hf)                      # (B, self.one, H/2, W/2)
        hf = F.interpolate(hf, size=x1.shape[-2:], mode="nearest")
        hf = self.hf_pw(hf)                          # (B, dim, H, W)

        # Residual enhancement
        beta = torch.clamp(self.beta, 0.0, 0.5)
        alpha = torch.clamp(self.alpha, 0.0, 0.5)
        x3 = x3_main + beta * detail + alpha * hf

        # Complementary branch
        x4 = self.conv2(x2)

        # Original FCM spatial/channel interaction
        x33 = self.spatial(x4) * x3
        x44 = self.channel(x3) * x4

        return x33 + x44

class FCM_3_HF(nn.Module):
    """
    FCM_3 with High-Frequency Residual Enhancement only.

    This module preserves the original FCM_3 mapping path and injects
    DWT-based high-frequency residual features into the main branch.
    It does not use the SD-P3 spatial detail branch.
    """
    def __init__(self, dim, dim_out, wg_mode="base", wg_band="hf"):
        super().__init__()
        self.one = dim - dim // 4
        self.two = dim // 4

        # Original FCM main branch
        self.conv1 = Conv(self.one, self.one, 3, 1, 1)
        self.conv12 = Conv(self.one, self.one, 3, 1, 1)
        self.conv123 = Conv(self.one, dim, 1, 1)

        # Original FCM complementary branch
        self.conv2 = Conv(self.two, dim, 1, 1)

        # Haar high-frequency residual branch
        # DWT_Haar_HF outputs 3 * self.one channels with H/2, W/2
        self.hf_dwt = DWT_Haar_HF()
        self.hf_reduce = nn.Sequential(
            nn.Conv2d(self.one * 3, self.one, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.one),
            nn.SiLU()
        )
        self.hf_pw = Conv(self.one, dim, 1, 1)

        # Learnable high-frequency residual scale
        self.alpha = nn.Parameter(torch.tensor(0.05))

        # Original FCM gates
        self.spatial = Spatial(dim)
        self.channel = Channel(dim)

    def forward(self, x):
        x1, x2 = torch.split(x, [self.one, self.two], dim=1)

        # Original FCM main branch
        x3 = self.conv1(x1)
        x3 = self.conv12(x3)
        x3_main = self.conv123(x3)

        # High-frequency residual branch
        hf = self.hf_dwt(x1)                         # (B, 3*self.one, H/2, W/2)
        hf = self.hf_reduce(hf)                      # (B, self.one, H/2, W/2)
        hf = F.interpolate(hf, size=x1.shape[-2:], mode="nearest")
        hf = self.hf_pw(hf)                          # (B, dim, H, W)

        alpha = torch.clamp(self.alpha, 0.0, 0.5)
        x3 = x3_main + alpha * hf

        # Complementary branch
        x4 = self.conv2(x2)

        # Original FCM spatial/channel interaction
        x33 = self.spatial(x4) * x3
        x44 = self.channel(x3) * x4

        return x33 + x44