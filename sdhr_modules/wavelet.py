# ultralytics/nn/modules/wavelet.py
import torch
import torch.nn as nn
import torch.nn.functional as F



class DWT_Haar(nn.Module):
    """
    超轻量 Haar DWT:
    输入:  (B, C, H, W)
    输出:  (B, 4C, H/2, W/2)  -> [LL, LH, HL, HH]
    无可训练参数，只是固定 depthwise conv。
    """
    def __init__(self):
        super().__init__()
        ll = torch.tensor([[0.5, 0.5],
                           [0.5, 0.5]])
        lh = torch.tensor([[-0.5, -0.5],
                           [ 0.5,  0.5]])
        hl = torch.tensor([[-0.5, 0.5],
                           [-0.5, 0.5]])
        hh = torch.tensor([[0.5, -0.5],
                           [-0.5, 0.5]])

        weight = torch.stack([ll, lh, hl, hh], dim=0)   # (4,2,2)
        self.register_buffer("weight", weight.unsqueeze(1))  # (4,1,2,2)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H % 2 == 0 and W % 2 == 0, "H,W must be even"

        # 给每个输入通道复制4个Haar核 -> (4C,1,2,2)
        w = self.weight.to(dtype=x.dtype, device=x.device).repeat(C, 1, 1, 1)

        # depthwise over channels only
        out = F.conv2d(x, w, stride=2, groups=C)  # (B,4C,H/2,W/2)
        return out

class DWT_Haar_HF(nn.Module):
    """
    Haar-frequency partial-channel transform used in the reported experiments.

    A fixed Haar grouped convolution first produces 4C transformed responses.
    The flattened output-channel representation is then divided into four
    equal channel blocks. The first block is omitted and the remaining three
    blocks are retained.

    Because grouped-convolution outputs are arranged per input channel, these
    blocks should not be interpreted as isolated LL/LH/HL/HH subbands.
    The retained representation therefore contains mixed Haar-frequency
    responses and is used as a compact frequency-domain residual feature.

    Input:
        (B, C, H, W)

    Output:
        (B, 3C, H/2, W/2)
    """
    def __init__(self):
        super().__init__()
        ll = torch.tensor([[0.5, 0.5],
                           [0.5, 0.5]])
        lh = torch.tensor([[-0.5, -0.5],
                           [ 0.5,  0.5]])
        hl = torch.tensor([[-0.5,  0.5],
                           [-0.5,  0.5]])
        hh = torch.tensor([[ 0.5, -0.5],
                           [-0.5,  0.5]])
        weight = torch.stack([ll, lh, hl, hh], dim=0)   # (4,2,2)
        self.register_buffer("weight", weight.unsqueeze(1))  # (4,1,2,2)

    def forward(self, x):
        B, C, H, W = x.shape
        w = self.weight.repeat(C, 1, 1, 1)             # (4C,1,2,2)

        out = F.conv2d(x, w, stride=2, groups=C)       # (B,4C,H/2,W/2)

        # Divide the flattened grouped-convolution output into four
        # equal channel blocks. These blocks are mixed Haar responses,
        # rather than isolated LL/LH/HL/HH subbands.
        ll, lh, hl, hh = torch.chunk(out, 4, dim=1)

        # Omit the first channel block and retain the remaining three.
        out_hf = torch.cat([lh, hl, hh], dim=1)        # (B,3C,H/2,W/2)
        return out_hf

class DWT_Haar_LL(nn.Module):
    """
    Haar DWT 只保留 LL (低频)
    输入:  (B, C, H, W)
    输出:  (B, C, H/2, W/2)
    """
    def __init__(self):
        super().__init__()
        ll = torch.tensor([[0.5, 0.5],
                           [0.5, 0.5]], dtype=torch.float32)
        self.register_buffer("weight", ll.view(1, 1, 2, 2))  # (1,1,2,2)

    def forward(self, x):
        B, C, H, W = x.shape

        # 每个通道用同一个 LL 核
        w = self.weight.repeat(C, 1, 1, 1)  # (C,1,2,2)

        # depthwise conv: groups=C
        out = F.conv2d(x, w, stride=2, groups=C)  # (B,C,H/2,W/2)
        return out

class DWT_Haar_Couple(nn.Module):
    """
    通用频域耦合模块（支持三种策略）：
    mode = 'concat' : 4 个子带直接拼接（等价于 full DWT，作为对照）
    mode = 'aw'     : 频带自适应加权（类似你现在的 DWT_Haar_AW）
    mode = 'gate'   : HF <-> LL 互相“门控”，根据能量动态调整权重

    输入:  (B, C, H, W)
    输出:  (B, 4C, H/2, W/2)   —— 保持 4C，方便直接接你现在的 head 结构
    """
    def __init__(self, mode='concat', reduction=4):
        super().__init__()
        assert mode in ('concat', 'aw', 'gate'), f"Unknown mode {mode}"
        self.mode = mode
        self.reduction = reduction

        # ------ 固定 Haar 核 -------
        ll = torch.tensor([[0.5, 0.5],
                           [0.5, 0.5]])
        lh = torch.tensor([[-0.5, -0.5],
                           [0.5,  0.5]])
        hl = torch.tensor([[-0.5, 0.5],
                           [-0.5, 0.5]])
        hh = torch.tensor([[0.5, -0.5],
                           [-0.5, 0.5]])

        weight = torch.stack([ll, lh, hl, hh], dim=0)  # (4, 2, 2)
        self.register_buffer("weight", weight.unsqueeze(1))  # (4, 1, 2, 2)

        # ------ 只有 aw 模式才需要的频带 MLP ------
        if self.mode == 'aw':
            hidden = max(4 // reduction, 1)
            self.band_mlp = nn.Sequential(
                nn.Linear(4, hidden, bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(hidden, 4, bias=False),
                nn.Sigmoid()
            )

    def forward(self, x):
        B, C, H, W = x.shape
        H2, W2 = H // 2, W // 2

        # ===== 1) depthwise DWT（groups=C）=====
        # w: (4C, 1, 2, 2)
        w = self.weight.repeat(C, 1, 1, 1)

        # 直接对 (B,C,H,W) 做 depthwise conv
        # groups=C => 每个通道一组，共 C 组；每组有 4 个滤波器 => 输出 4C 通道
        out = F.conv2d(x, w, stride=2, groups=C)  # (B, 4C, H/2, W/2)

        # reshape => (B,4,C,H/2,W/2)
        out_band = out.view(B, 4, C, H2, W2)

        # 拆 LL / HF
        LL = out_band[:, 0:1, ...]      # (B,1,C,H2,W2)
        HF3 = out_band[:, 1:4, ...]     # (B,3,C,H2,W2)

        # ===== 2) 耦合策略 =====
        if self.mode == 'concat':
            out_fused = out_band

        elif self.mode == 'aw':
            band_stat = out_band.mean(dim=(2, 3, 4))   # (B,4)
            band_w = self.band_mlp(band_stat).view(B, 4, 1, 1, 1)
            out_fused = out_band * band_w

        elif self.mode == 'gate':
            band_stat = out_band.mean(dim=(2, 3, 4))   # (B,4)
            ll_e = band_stat[:, 0:1]                   # (B,1)
            hf_e = band_stat[:, 1:].mean(dim=1, keepdim=True)  # (B,1)

            alpha = torch.sigmoid(hf_e - ll_e)         # (B,1)
            w_ll = 1.0 + alpha
            w_hf = 1.0 - alpha

            band_w = torch.cat([w_ll, w_hf.repeat(1, 3)], dim=1)  # (B,4)
            band_w = band_w.view(B, 4, 1, 1, 1)

            out_fused = out_band * band_w

        # ===== 3) 展平回 (B,4C,H/2,W/2) =====
        out = out_fused.view(B, 4 * C, H2, W2)
        return out


class DWT_Haar_GateV2(nn.Module):
    """
    Haar DWT + Learnable Gate (gate-v2)

    输入:  x (B, C, H, W)
    输出:  (B, 4C, H/2, W/2)
    """
    def __init__(self, reduction=4):
        super().__init__()

        # Haar 4 个 2x2 核
        ll = torch.tensor([[0.5, 0.5],
                           [0.5, 0.5]])
        lh = torch.tensor([[-0.5, -0.5],
                           [0.5,  0.5]])
        hl = torch.tensor([[-0.5, 0.5],
                           [-0.5, 0.5]])
        hh = torch.tensor([[0.5, -0.5],
                           [-0.5, 0.5]])
        weight = torch.stack([ll, lh, hl, hh], dim=0)  # (4,2,2)
        self.register_buffer("weight", weight.unsqueeze(1))  # (4,1,2,2)

        # 学 4 个频段权重
        hidden = max(4 // reduction, 1)
        self.mlp = nn.Sequential(
            nn.Linear(4, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 4, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        B, C, H, W = x.shape

        # ===== 1) depthwise DWT =====
        w = self.weight.repeat(C, 1, 1, 1)          # (4C,1,2,2)
        out = F.conv2d(x, w, stride=2, groups=C)    # (B,4C,H/2,W/2)
        H2, W2 = H // 2, W // 2

        out_band = out.view(B, 4, C, H2, W2)        # (B,4,C,H2,W2)

        # ===== 2) 频段能量统计 → 学习门控权重 =====
        band_stat = out_band.mean(dim=(2, 3, 4))    # (B,4)
        band_w = self.mlp(band_stat).view(B, 4, 1, 1, 1)

        # ===== 3) 加权融合 =====
        out_band = out_band * band_w

        return out_band.view(B, 4 * C, H2, W2)

class DWT_Haar_GateV3(nn.Module):
    """
    GateV3: 支持 guide 输入的频域门控
    - 兼容 YOLO 的 list-input / dummy-forward
    - 输出仍为 4C, H/2, W/2
    """
    def __init__(self, reduction=4):
        super().__init__()

        # ===== 固定 Haar 4 个 2x2 核 =====
        ll = torch.tensor([[0.5, 0.5],
                           [0.5, 0.5]])
        lh = torch.tensor([[-0.5, -0.5],
                           [0.5,  0.5]])
        hl = torch.tensor([[-0.5, 0.5],
                           [-0.5, 0.5]])
        hh = torch.tensor([[0.5, -0.5],
                           [-0.5, 0.5]])

        weight = torch.stack([ll, lh, hl, hh], dim=0)  # (4,2,2)
        self.register_buffer("weight", weight.unsqueeze(1))  # (4,1,2,2)

        # ===== 例：一个轻量 gate MLP（你有自己的就替换）=====
        hidden = max(4 // reduction, 1)
        self.band_mlp = nn.Sequential(
            nn.Linear(4, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 4, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x, guide=None):
        # --- 情况2：YOLO 多输入会传 list ---
        if isinstance(x, (list, tuple)):
            if len(x) == 2:
                x, guide = x
            else:
                x = x[0]
                guide = None

        # --- 情况3：dummy stride forward 没 guide ---
        if guide is None:
            guide = x

        B, C, H, W = x.shape

        # ===== 1) Haar DWT =====
        w = self.weight.repeat(C, 1, 1, 1)          # (4C,1,2,2)
        out = F.conv2d(x, w, stride=2, groups=C)    # ✅ groups=C
        H2, W2 = H // 2, W // 2
        out_band = out.view(B, 4, C, H2, W2)


        # ===== 2) GateV3（示例：用 guide 的频带能量来调 x 的频带）=====
        # guide 的 DWT 能量
        gw = self.weight.repeat(C, 1, 1, 1)
        gout = F.conv2d(guide, gw, stride=2, groups=C)
        g_band = gout.view(B, 4, C, H2, W2)

        # 计算 guide 每个频带能量 (B,4)
        g_stat = g_band.mean(dim=(2,3,4))
        g_w = self.band_mlp(g_stat).view(B, 4, 1, 1, 1)

        # 用 guide 权重去加权 x 的 4 个子带
        out_band = out_band * g_w

        # ===== 3) reshape 回 4C =====
        out = out_band.view(B, 4*C, H2, W2)
        return out


class DWT_Haar_AW(nn.Module):
    """
    Haar DWT + Band-Adaptive Weighting
    输入:  (B, C, H, W)
    输出:  (B, 4C, H/2, W/2)
    """
    def __init__(self, reduction=4):
        super().__init__()

        ll = torch.tensor([[0.5, 0.5],
                           [0.5, 0.5]])
        lh = torch.tensor([[-0.5, -0.5],
                           [0.5,  0.5]])
        hl = torch.tensor([[-0.5, 0.5],
                           [-0.5, 0.5]])
        hh = torch.tensor([[0.5, -0.5],
                           [-0.5, 0.5]])

        weight = torch.stack([ll, lh, hl, hh], dim=0)  # (4,2,2)
        self.register_buffer("weight", weight.unsqueeze(1))  # (4,1,2,2)

        hidden = max(4 // reduction, 1)
        self.band_mlp = nn.Sequential(
            nn.Linear(4, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 4, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        B, C, H, W = x.shape

        # ✅ depthwise DWT: groups 只按 C，不按 B
        w = self.weight.repeat(C, 1, 1, 1)          # (4C, 1, 2, 2)
        out = F.conv2d(x, w, stride=2, groups=C)    # (B, 4C, H/2, W/2)

        # reshape → (B, 4, C, H/2, W/2)
        out_band = out.view(B, 4, C, H // 2, W // 2)

        # band-wise pooling → (B, 4)
        band_stat = out_band.mean(dim=(2, 3, 4))

        # adaptive weights → (B, 4, 1, 1, 1)
        band_w = self.band_mlp(band_stat).view(B, 4, 1, 1, 1)

        # weighted bands
        out_band = out_band * band_w

        # back to (B, 4C, H/2, W/2)
        out = out_band.view(B, 4 * C, H // 2, W // 2)
        return out


class IDWT_Haar(nn.Module):
    def __init__(self):
        super().__init__()
        ll = torch.tensor([[0.5, 0.5],
                           [0.5, 0.5]])
        lh = torch.tensor([[-0.5, -0.5],
                           [0.5,  0.5]])
        hl = torch.tensor([[-0.5, 0.5],
                           [-0.5, 0.5]])
        hh = torch.tensor([[0.5, -0.5],
                           [-0.5, 0.5]])
        weight = torch.stack([ll, lh, hl, hh], dim=0)          # (4,2,2)
        self.register_buffer("weight", weight.unsqueeze(1))    # (4,1,2,2)

    def forward(self, x):
        B, C4, H, W = x.shape
        assert C4 % 4 == 0, "Input channels must be divisible by 4"
        C = C4 // 4

        w = self.weight.to(dtype=x.dtype, device=x.device).repeat(C, 1, 1, 1)  # (4C,1,2,2)
        out = F.conv_transpose2d(x, w, stride=2, groups=C)  # (B,C,2H,2W)
        return out

    
class DWT_Enhance(nn.Module):
    def __init__(self, c1):
        super().__init__()
        self.dwt = DWT_Haar()
        self.reduce = nn.Sequential(
            nn.Conv2d(c1 * 4, c1, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU()
        )

    def forward(self, x):
        y = self.dwt(x)  # (B, 4C, H/2, W/2)
        y = F.interpolate(y, scale_factor=2, mode="nearest")  # (B, 4C, H, W)
        y = self.reduce(y)  # (B, C, H, W)
        return y


class WaveletChannelGate(nn.Module):
    """
    Wavelet-guided Channel Gate
    输入:  x (B, C, H, W)
    输出:  w (B, C, 1, 1)  —— 通道权重，用于 x * w

    mode:
      - 'hf'   : 用 LH/HL/HH 三个高频子带引导（更贴小目标边缘/纹理）
      - 'll'   : 用 LL 低频引导
      - 'full' : 用 4 个子带均值引导（LL+LH+HL+HH）
    """
    def __init__(self, c: int, mode: str = "hf"):
        super().__init__()
        assert mode in ("hf", "ll", "full"), f"Unknown mode={mode}"
        self.mode = mode
        self.c = c

        self.dwt_hf = DWT_Haar_HF()
        self.dwt_ll = DWT_Haar_LL()
        self.dwt_full = DWT_Haar()

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        assert C == self.c, f"Channel mismatch: got {C}, expected {self.c}"
        assert H % 2 == 0 and W % 2 == 0, "H,W must be even for Haar DWT"

        if self.mode == "hf":
            y = self.dwt_hf(x)  # (B, 3C, H/2, W/2)
            # 聚合 3 个高频子带 -> (B, C, H/2, W/2)
            y = y.view(B, 3, C, y.size(2), y.size(3)).mean(dim=1)
        elif self.mode == "ll":
            y = self.dwt_ll(x)  # (B, C, H/2, W/2)
        else:  # "full"
            y = self.dwt_full(x)  # (B, 4C, H/2, W/2)
            # 聚合 4 个子带 -> (B, C, H/2, W/2)
            y = y.view(B, 4, C, y.size(2), y.size(3)).mean(dim=1)

        # 频域响应强度 -> 通道权重
        w = self.pool(y.abs())  # (B, C, 1, 1)
        return self.sigmoid(w)

class WaveletSpatialGate(nn.Module):
    """
    Wavelet-guided Spatial Gate
    输入:  x (B, C, H, W)
    输出:  s (B, 1, H, W)  —— 空间权重，用于 x * s

    mode:
      - 'hf'   : 用 LH/HL/HH 高频引导
      - 'll'   : 用 LL 低频引导
      - 'full' : 用 4 个子带均值引导
    fuse:
      - 'mean' : 通道聚合用 mean
      - 'meanmax' : mean + max（更敏感，常对小目标更友好，开销仍很小）
    """
    def __init__(self, c: int, mode: str = "hf", fuse: str = "mean"):
        super().__init__()
        assert mode in ("hf", "ll", "full"), f"Unknown mode={mode}"
        assert fuse in ("mean", "meanmax"), f"Unknown fuse={fuse}"
        self.mode = mode
        self.fuse = fuse
        self.c = c

        self.dwt_hf = DWT_Haar_HF()
        self.dwt_ll = DWT_Haar_LL()
        self.dwt_full = DWT_Haar()

        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        assert C == self.c, f"Channel mismatch: got {C}, expected {self.c}"
        assert H % 2 == 0 and W % 2 == 0, "H,W must be even for Haar DWT"

        if self.mode == "hf":
            y = self.dwt_hf(x)  # (B, 3C, H/2, W/2)
            y = y.view(B, 3, C, y.size(2), y.size(3)).mean(dim=1)  # (B,C,H/2,W/2)
        elif self.mode == "ll":
            y = self.dwt_ll(x)  # (B, C, H/2, W/2)
        else:  # "full"
            y = self.dwt_full(x)  # (B, 4C, H/2, W/2)
            y = y.view(B, 4, C, y.size(2), y.size(3)).mean(dim=1)  # (B,C,H/2,W/2)

        # 上采样回原分辨率，做空间响应图
        y = F.interpolate(y.abs(), size=(H, W), mode="nearest")  # (B,C,H,W)

        if self.fuse == "mean":
            s = y.mean(dim=1, keepdim=True)  # (B,1,H,W)
        else:  # meanmax
            s = y.mean(dim=1, keepdim=True) + y.max(dim=1, keepdim=True)[0]  # (B,1,H,W)

        return self.sigmoid(s)

class CBAMChannelGate(nn.Module):
    """CBAM Channel Attention: avg+max pooling -> shared MLP (1x1 conv) -> sigmoid."""
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(channels // reduction, 1)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)  # (B,C,1,1)


class CBAMSpatialGate(nn.Module):
    """CBAM Spatial Attention: channel avg+max -> conv(k) -> sigmoid."""
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        assert kernel_size in (3, 5, 7), "CBAM typically uses kernel_size=7 (or 3/5 for lighter)."
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)          # (B,1,H,W)
        max_out, _ = torch.max(x, dim=1, keepdim=True)        # (B,1,H,W)
        attn = torch.cat([avg_out, max_out], dim=1)           # (B,2,H,W)
        return self.sigmoid(self.conv(attn))                  # (B,1,H,W)

class WaveletCoupledGate(nn.Module):
    """
    True Frequency-domain Coupled Gating
    LL <-> HF 双向门控
    """
    def __init__(self, c, reduction=16):
        super().__init__()
        self.c = c
        self.dwt = DWT_Haar()

        hidden = max(c // reduction, 4)

        # LL -> HF gate
        self.ll_to_hf = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, c, 1, bias=False),
            nn.Sigmoid()
        )

        # HF -> LL gate
        self.hf_to_ll = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, c, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        x: (B, C, H, W)
        return:
            ll_gate: (B, C, 1, 1)
            hf_gate: (B, C, 1, 1)
        """
        B, C, H, W = x.shape
        assert C == self.c

        # Haar DWT
        y = self.dwt(x)                     # (B, 4C, H/2, W/2)
        y = y.view(B, 4, C, H // 2, W // 2)

        LL = y[:, 0]                        # (B, C, H/2, W/2)
        HF = y[:, 1:].mean(dim=1)           # (B, C, H/2, W/2)

        # Coupled gating
        hf_gate = self.ll_to_hf(LL)         # LL -> HF
        ll_gate = self.hf_to_ll(HF)         # HF -> LL

        return ll_gate, hf_gate

class DWT_Haar_GateV3_Res(nn.Module):
    """
    Residual-scaled GateV3 Haar DWT module.

    Input:
        x: (B, C, H, W)
    Output:
        (B, 4C, H/2, W/2)

    The output is formulated as:
        out = raw_dwt + alpha * (gated_dwt - raw_dwt)

    This keeps the module close to the original Haar DWT at the beginning
    of training and gradually learns frequency-guided refinement.
    """
    def __init__(self, reduction=4):
        super().__init__()

        ll = torch.tensor([[0.5, 0.5],
                           [0.5, 0.5]])
        lh = torch.tensor([[-0.5, -0.5],
                           [0.5,  0.5]])
        hl = torch.tensor([[-0.5, 0.5],
                           [-0.5, 0.5]])
        hh = torch.tensor([[0.5, -0.5],
                           [-0.5, 0.5]])

        weight = torch.stack([ll, lh, hl, hh], dim=0)
        self.register_buffer("weight", weight.unsqueeze(1))

        hidden = max(4 // reduction, 1)
        self.band_mlp = nn.Sequential(
            nn.Linear(4, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 4, bias=False),
            nn.Sigmoid()
        )

        # Learnable residual scaling coefficient
        self.alpha = nn.Parameter(torch.tensor(0.1))

    def forward(self, x, guide=None):
        if isinstance(x, (list, tuple)):
            if len(x) == 2:
                x, guide = x
            else:
                x = x[0]
                guide = None

        if guide is None:
            guide = x

        B, C, H, W = x.shape

        # Raw Haar DWT
        w = self.weight.to(dtype=x.dtype, device=x.device).repeat(C, 1, 1, 1)
        raw_out = F.conv2d(x, w, stride=2, groups=C)

        H2, W2 = H // 2, W // 2
        out_band = raw_out.view(B, 4, C, H2, W2)

        # Guide branch
        gw = self.weight.to(dtype=guide.dtype, device=guide.device).repeat(C, 1, 1, 1)
        guide_out = F.conv2d(guide, gw, stride=2, groups=C)
        guide_band = guide_out.view(B, 4, C, H2, W2)

        guide_stat = guide_band.mean(dim=(2, 3, 4))
        guide_weight = self.band_mlp(guide_stat).view(B, 4, 1, 1, 1)

        # Gated frequency features
        gated_band = out_band * guide_weight
        gated_out = gated_band.view(B, 4 * C, H2, W2)

        # Residual-scaled frequency refinement
        out = raw_out + self.alpha * (gated_out - raw_out)
        return out

class DWT_Haar_Couple_Res(nn.Module):
    """
    Residual-scaled frequency-coupled Haar DWT module.

    Input:
        x: (B, C, H, W)
    Output:
        (B, 4C, H/2, W/2)

    The output is formulated as:
        out = raw_dwt + alpha * (coupled_dwt - raw_dwt)

    This keeps the module close to the original Haar DWT at the beginning
    of training and gradually learns frequency-guided refinement.
    """
    def __init__(self, mode='concat', reduction=4):
        super().__init__()
        assert mode in ('concat', 'aw', 'gate'), f"Unknown mode {mode}"
        self.mode = mode
        self.reduction = reduction

        ll = torch.tensor([[0.5, 0.5],
                           [0.5, 0.5]])
        lh = torch.tensor([[-0.5, -0.5],
                           [0.5,  0.5]])
        hl = torch.tensor([[-0.5, 0.5],
                           [-0.5, 0.5]])
        hh = torch.tensor([[0.5, -0.5],
                           [-0.5, 0.5]])

        weight = torch.stack([ll, lh, hl, hh], dim=0)
        self.register_buffer("weight", weight.unsqueeze(1))

        if self.mode == 'aw':
            hidden = max(4 // reduction, 1)
            self.band_mlp = nn.Sequential(
                nn.Linear(4, hidden, bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(hidden, 4, bias=False),
                nn.Sigmoid()
            )

        # Learnable residual scaling coefficient
        self.alpha = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        B, C, H, W = x.shape
        H2, W2 = H // 2, W // 2

        w = self.weight.to(dtype=x.dtype, device=x.device).repeat(C, 1, 1, 1)

        # Raw Haar DWT feature
        raw_out = F.conv2d(x, w, stride=2, groups=C)
        out_band = raw_out.view(B, 4, C, H2, W2)

        if self.mode == 'concat':
            coupled_band = out_band

        elif self.mode == 'aw':
            band_stat = out_band.mean(dim=(2, 3, 4))
            band_w = self.band_mlp(band_stat).view(B, 4, 1, 1, 1)
            coupled_band = out_band * band_w

        elif self.mode == 'gate':
            band_stat = out_band.mean(dim=(2, 3, 4))
            ll_e = band_stat[:, 0:1]
            hf_e = band_stat[:, 1:].mean(dim=1, keepdim=True)

            alpha_gate = torch.sigmoid(hf_e - ll_e)
            w_ll = 1.0 + alpha_gate
            w_hf = 1.0 - alpha_gate

            band_w = torch.cat([w_ll, w_hf.repeat(1, 3)], dim=1)
            band_w = band_w.view(B, 4, 1, 1, 1)

            coupled_band = out_band * band_w

        coupled_out = coupled_band.view(B, 4 * C, H2, W2)

        # Residual-scaled refinement in frequency space
        out = raw_out + self.alpha * (coupled_out - raw_out)
        return out

class WaveletCoupledGate_HC(nn.Module):
    """
    High-frequency calibrated Wavelet Coupled Gate.

    Compared with WaveletCoupledGate, this module introduces a lightweight
    ECA-style channel calibration operation into the high-frequency branch.
    It aims to emphasize informative high-frequency edge and texture responses
    while suppressing redundant background noise.
    """
    def __init__(self, c, reduction=16, k_size=3):
        super().__init__()
        self.c = c
        self.dwt = DWT_Haar()

        hidden = max(c // reduction, 4)

        # LL -> HF gate
        self.ll_to_hf = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, c, 1, bias=False),
            nn.Sigmoid()
        )

        # HF -> LL gate
        self.hf_to_ll = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, c, 1, bias=False),
            nn.Sigmoid()
        )

        # ECA-style high-frequency channel calibration
        self.hf_pool = nn.AdaptiveAvgPool2d(1)
        self.hf_eca = nn.Conv1d(
            in_channels=1,
            out_channels=1,
            kernel_size=k_size,
            padding=(k_size - 1) // 2,
            bias=False
        )
        self.sigmoid = nn.Sigmoid()

    def high_frequency_calibration(self, hf):
        # hf: (B, C, H/2, W/2)
        w = self.hf_pool(hf)                    # (B, C, 1, 1)
        w = w.squeeze(-1).transpose(-1, -2)     # (B, 1, C)
        w = self.hf_eca(w)                      # (B, 1, C)
        w = self.sigmoid(w)
        w = w.transpose(-1, -2).unsqueeze(-1)   # (B, C, 1, 1)
        return hf * w

    def forward(self, x):
        """
        x: (B, C, H, W)
        return:
            ll_gate: (B, C, 1, 1)
            hf_gate: (B, C, 1, 1)
        """
        B, C, H, W = x.shape
        assert C == self.c

        y = self.dwt(x)                         # (B, 4C, H/2, W/2)
        y = y.view(B, 4, C, H // 2, W // 2)

        LL = y[:, 0]                            # (B, C, H/2, W/2)
        HF = y[:, 1:].mean(dim=1)               # (B, C, H/2, W/2)

        # High-frequency channel calibration
        HF = self.high_frequency_calibration(HF)

        hf_gate = self.ll_to_hf(LL)             # LL -> HF
        ll_gate = self.hf_to_ll(HF)             # HF -> LL

        return ll_gate, hf_gate