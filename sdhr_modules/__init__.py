# Ultralytics YOLO ??, AGPL-3.0 license
"""
Ultralytics modules. Visualize with:

from ultralytics.nn.modules import *
import torch
import os

x = torch.ones(1, 128, 40, 40)
m = Conv(128, 128)
f = f'{m._get_name()}.onnx'
torch.onnx.export(m, x, f)
os.system(f'onnxsim {f} {f} && open {f}')
"""

from .block import (C1, C2, C3, C3TR, DFL, SPP, SPPF, Bottleneck, BottleneckCSP, C2f, C3Ghost, C3x, GhostBottleneck,
                    HGBlock, HGStem, Proto, RepC3,SEBlock,ECABlock)
from .conv import (CBAM, ChannelAttention, Concat, Conv, Conv2, ConvTranspose, DWConv, DWConvTranspose2d, Focus,
                   GhostConv, LightConv, RepConv, SpatialAttention,Pzconv,FCM_3,FCM_2,FCM_1,FCM,Down,FCM_3_Res, FCM_2_Res, FCM_1_Res, FCM_Res,FCM_3_HC, FCM_2_HC, FCM_1_HC, FCM_HC,FCM_3_MS, FCM_2_MS, FCM_1_MS, FCM_MS,FCM_3_SD, FCM_2_SD, FCM_1_SD, FCM_SD,FCM_3_SD_HF,FCM_3_HF)
from .head import Classify, Detect, Pose, RTDETRDecoder, Segment
from .transformer import (AIFI, MLP, DeformableTransformerDecoder, DeformableTransformerDecoderLayer, LayerNorm2d,
                          MLPBlock, MSDeformAttn, TransformerBlock, TransformerEncoderLayer, TransformerLayer)
from .wavelet import DWT_Haar, IDWT_Haar,DWT_Enhance,DWT_Haar_HF,DWT_Haar_LL,DWT_Haar_AW,DWT_Haar_Couple,DWT_Haar_GateV3,DWT_Haar_GateV2,DWT_Haar_GateV3,DWT_Haar_GateV3_Res,DWT_Haar_Couple_Res    



__all__ = ('Conv', 'Conv2', 'LightConv', 'RepConv', 'DWConv', 'DWConvTranspose2d', 'ConvTranspose', 'Focus',
           'GhostConv', 'ChannelAttention', 'SpatialAttention', 'CBAM', 'Concat', 'TransformerLayer',
           'TransformerBlock', 'MLPBlock', 'LayerNorm2d', 'DFL', 'HGBlock', 'HGStem', 'SPP', 'SPPF', 'C1', 'C2', 'C3',
           'C2f', 'C3x', 'C3TR', 'C3Ghost', 'GhostBottleneck', 'Bottleneck', 'BottleneckCSP', 'Proto', 'Detect',
           'Segment', 'Pose', 'Classify', 'TransformerEncoderLayer', 'RepC3', 'RTDETRDecoder', 'AIFI',
           'DeformableTransformerDecoder', 'DeformableTransformerDecoderLayer', 'MSDeformAttn', 'MLP','Pzconv','FCM', 'Pzconv',  'FCM_3', 'FCM_2','Down',
           'DWT_Haar', 'IDWT_Haar','DWT_Enhance','DWT_Haar_HF','DWT_Haar_LL','DWT_Haar_AW','DWT_Haar_Couple','DWT_Haar_GateV3','DWT_Haar_GateV2','DWT_Haar_Couple_Res')
