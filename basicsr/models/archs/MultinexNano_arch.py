# Copyright (c) 2026 Alexandru Brateanu
# Multinex is licensed for non-commercial research and educational use only.
# Commercial use requires prior written permission.
# See LICENSE for details.

import torch.nn as nn
import torch
import torch.nn.functional as F
import math
import warnings
from torch.nn.init import _calculate_fan_in_and_fan_out
import math
from typing import Dict, Optional, Sequence, Tuple

import os
from pathlib import Path
import numpy as np
from PIL import Image
import torch



def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)
    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def variance_scaling_(tensor, scale=1.0, mode='fan_in', distribution='normal'):
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    if mode == 'fan_in':
        denom = fan_in
    elif mode == 'fan_out':
        denom = fan_out
    elif mode == 'fan_avg':
        denom = (fan_in + fan_out) / 2
    variance = scale / denom
    if distribution == "truncated_normal":
        trunc_normal_(tensor, std=math.sqrt(variance) / .87962566103423978)
    elif distribution == "normal":
        tensor.normal_(std=math.sqrt(variance))
    elif distribution == "uniform":
        bound = math.sqrt(3 * variance)
        tensor.uniform_(-bound, bound)
    else:
        raise ValueError(f"invalid distribution {distribution}")


def lecun_normal_(tensor):
    variance_scaling_(tensor, mode='fan_in', distribution='truncated_normal')


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)


class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)


def conv(in_channels, out_channels, kernel_size, bias=False, padding=1, stride=1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias, stride=stride)


# input [bs,28,256,310]  output [bs, 28, 256, 256]
def shift_back(inputs, step=2):
    [bs, nC, row, col] = inputs.shape
    down_sample = 256 // row
    step = float(step) / float(down_sample * down_sample)
    out_col = row
    for i in range(nC):
        inputs[:, i, :, :out_col] = \
            inputs[:, i, :, int(step * i):int(step * i) + out_col]
    return inputs[:, :, :, :out_col]


# ---------- tiny utils ----------
def make_act(act):
    import torch.nn as nn
    if act is None:
        return nn.Identity()
    if isinstance(act, nn.Module):
        return act
    if isinstance(act, str):
        name = act.lower()
        if name in ('silu', 'swish'):
            return nn.SiLU()
        if name in ('relu',):
            return nn.ReLU(inplace=True)
        if name in ('gelu',):
            return nn.GELU()
        if name in ('lrelu', 'leakyrelu'):
            return nn.LeakyReLU(0.1, inplace=True)
        if name in ('prelu',):
            return nn.PReLU()
        return nn.SiLU()  # default
    if isinstance(act, type):
        return act()  # a class like nn.SiLU
    return nn.SiLU()

def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)

class DWSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1, act='SiLU', bn=True):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, k, s, p, groups=in_ch, bias=False)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, 1, 0, bias=False)
        self.bn = nn.BatchNorm2d(out_ch) if bn else nn.Identity()
        self.act = make_act(act)                     # <-- change here
    def forward(self, x):
        return self.act(self.bn(self.pw(self.dw(x))))

class ConvBNAct(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1, act='SiLU', bn=True):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(out_ch) if bn else nn.Identity()
        self.act = make_act(act)                     # <-- change here
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

# ---------- main net ----------

class SEBlock(nn.Module):
    def __init__(self, input_channels, reduction_ratio=16):
        super(SEBlock, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(input_channels, input_channels // reduction_ratio)
        self.fc2 = nn.Linear(input_channels // reduction_ratio, input_channels)

    def forward(self, x):
        batch_size, num_channels, _, _ = x.size()
        y = self.pool(x).reshape(batch_size, num_channels)
        y = F.relu(self.fc1(y))
        y = torch.tanh(self.fc2(y))
        y = y.reshape(batch_size, num_channels, 1, 1)
        return x * y


# ---------- MSEF -------------

class MSEFBlock(nn.Module):
    def __init__(self, ch, reduction_ratio=16):
        super(MSEFBlock, self).__init__()
        self.depthwise_conv = nn.Conv2d(ch, ch, kernel_size=3, padding=1, groups=ch)
        self.se_attn = SEBlock(ch, reduction_ratio)

    def forward(self, x):
        x1 = self.depthwise_conv(x)
        x2 = self.se_attn(x)
        x_fused = x1 * x2
        x_out = x_fused + x
        return x_out

# ---------- MSEF -------------

# ---- MDTA ----

import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers

from einops import rearrange



##########################################################################
## Layer Norm

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


# --------


# ---------- illumination stack (single-op maps) ----------

class IlluminationExtractor(nn.Module):
    """
    Returns K single-op illumination maps stacked along channel dim.
    Toggle which maps to use via 'use_flags'.
    """
    def __init__(self,
                 use_flags: Dict[str, bool] = None,
                 eps: float = 1e-6):
        super().__init__()
        # default: enable all six
        default = dict(mean=True, rec709=True, vmax=True, lightness=True, ycgco=True, l2norm=True)
        self.use = default if use_flags is None else {**default, **use_flags}
        self.eps = eps
        # fixed weights for Rec.709
        self.register_buffer("w_rec709", torch.tensor([0.2126, 0.7152, 0.0722]).view(1,3,1,1))
        # fixed weights for YCgCo Y = 1/4 R + 1/2 G + 1/4 B
        self.register_buffer("w_ycgco", torch.tensor([0.25, 0.5, 0.25]).view(1,3,1,1))

        # list order to keep channel order deterministic
        self.order = [k for k, v in self.use.items() if v]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        assert C == 3, "Input must be RGB (B,3,H,W)"
        maps = []
        R, G, Bc = x[:,0:1], x[:,1:2], x[:,2:3]

        if self.use.get("mean", False):
            maps.append((R + G + Bc) / 3.0)
        if self.use.get("rec709", False):
            maps.append((x * self.w_rec709).sum(dim=1, keepdim=True))
        if self.use.get("vmax", False):
            maps.append(torch.maximum(R, torch.maximum(G, Bc)))
        if self.use.get("lightness", False):
            mx = torch.maximum(R, torch.maximum(G, Bc))
            mn = torch.minimum(R, torch.minimum(G, Bc))
            maps.append((mx + mn) / 2.0)
        if self.use.get("ycgco", False):
            maps.append((x * self.w_ycgco).sum(dim=1, keepdim=True))
        if self.use.get("l2norm", False):
            maps.append(torch.sqrt(R*R + G*G + Bc*Bc + self.eps))

        return torch.cat(maps, dim=1)  # (B, K, H, W)

class ChrominanceExtractor(nn.Module):
    """
    Returns K_C chrominance maps stacked along channel dim.
    Toggle which maps to use via 'use_flags'.
    All maps are formula-based & fast.
    """
    def __init__(self, use_flags: Dict[str, bool] = None, eps: float = 1e-6):
        super().__init__()
        default = dict(
            yuv_uv=True,       # U, V (BT.601)
            ycbcr_cbcr=True,   # Cb, Cr (BT.601)
            opponent=True,     # O1, O2
            chroma_rg=True,    # r, g (chromaticity)
            hsv_s=True         # saturation S
        )
        self.use = default if use_flags is None else {**default, **use_flags}
        self.eps = eps

        # Fixed matrices (BT.601)
        # YUV (Y = 0.299R + 0.587G + 0.114B), we only use U,V
        self.register_buffer("yuv_U", torch.tensor([-0.14713, -0.28886, 0.436]).view(1,3,1,1))
        self.register_buffer("yuv_V", torch.tensor([ 0.61500, -0.51499, -0.10001]).view(1,3,1,1))

        # YCbCr luma coeffs for Y (reuse from Illum if needed); for Cb, Cr we can use direct linear forms:
        self.register_buffer("ycbcr_Cb", torch.tensor([-0.168736, -0.331264, 0.5]).view(1,3,1,1))
        self.register_buffer("ycbcr_Cr", torch.tensor([ 0.5,      -0.418688, -0.081312]).view(1,3,1,1))

        # Opponent color space
        self.register_buffer("opp_O1", torch.tensor([ 1.0, -1.0,  0.0]).view(1,3,1,1) / (2**0.5))
        self.register_buffer("opp_O2", torch.tensor([ 1.0,  1.0, -2.0]).view(1,3,1,1) / (6**0.5))

        # Keep deterministic channel order
        self.order = []
        if self.use.get('yuv_uv', False):
            self.order += ['U','V']
        if self.use.get('ycbcr_cbcr', False):
            self.order += ['Cb','Cr']
        if self.use.get('opponent', False):
            self.order += ['O1','O2']
        if self.use.get('chroma_rg', False):
            self.order += ['r','g']
        if self.use.get('hsv_s', False):
            self.order += ['S']

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        assert C == 3, "Input must be RGB (B,3,H,W)"
        R, G, Bc = x[:,0:1], x[:,1:2], x[:,2:3]
        maps = []

        # YUV U,V (no Y)
        if self.use.get('yuv_uv', False):
            maps.append((x * self.yuv_U).sum(1, keepdim=True))  # U
            maps.append((x * self.yuv_V).sum(1, keepdim=True))  # V

        # YCbCr Cb, Cr (linear forms)
        if self.use.get('ycbcr_cbcr', False):
            maps.append((x * self.ycbcr_Cb).sum(1, keepdim=True))  # Cb
            maps.append((x * self.ycbcr_Cr).sum(1, keepdim=True))  # Cr

        # Opponent O1,O2
        if self.use.get('opponent', False):
            maps.append((x * self.opp_O1).sum(1, keepdim=True))    # O1
            maps.append((x * self.opp_O2).sum(1, keepdim=True))    # O2

        # Chromaticity r,g
        if self.use.get('chroma_rg', False):
            denom = (R + G + Bc).clamp_min(self.eps)
            maps.append(R / denom)  # r
            maps.append(G / denom)  # g

        # HSV Saturation S = 1 - min/max
        if self.use.get('hsv_s', False):
            mx = torch.maximum(R, torch.maximum(G, Bc))
            mn = torch.minimum(R, torch.minimum(G, Bc))
            S = (mx - mn) / (mx.clamp_min(self.eps))  # 1 - mn/mx
            maps.append(S)

        return torch.cat(maps, dim=1) if maps else torch.zeros(B,0,H,W, device=x.device, dtype=x.dtype)








class HaarDWT2D(nn.Module):
    """Fixed one-level 2-D orthonormal Haar analysis transform."""

    def __init__(self, in_channels: int = 4, padding_mode: str = "reflect"):
        super().__init__()
        if in_channels < 1:
            raise ValueError("in_channels must be at least 1")
        if padding_mode not in {"reflect", "replicate"}:
            raise ValueError("padding_mode must be 'reflect' or 'replicate'")

        self.in_channels = int(in_channels)
        self.groups = int(in_channels)
        self.padding_mode = padding_mode

        kernels = torch.tensor(
            [
                [[[0.5, 0.5], [0.5, 0.5]]],       # LL
                [[[0.5, -0.5], [0.5, -0.5]]],     # HL
                [[[0.5, 0.5], [-0.5, -0.5]]],     # LH
                [[[0.5, -0.5], [-0.5, 0.5]]],     # HH
            ],
            dtype=torch.float32,
        )
        kernels = kernels.repeat(self.in_channels, 1, 1, 1)
        self.register_buffer("kernel", kernels, persistent=True)

    def _pad_to_even(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        pad_h = h % 2
        pad_w = w % 2
        if not (pad_h or pad_w):
            return x

        mode = self.padding_mode
        if mode == "reflect" and (h < 2 or w < 2):
            mode = "replicate"
        return F.pad(x, (0, pad_w, 0, pad_h), mode=mode)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if x.ndim != 4:
            raise ValueError(f"Expected a 4-D BCHW tensor, got shape {tuple(x.shape)}")
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} channels, got {x.shape[1]}"
            )

        x_even = self._pad_to_even(x)
        bands = F.conv2d(
            x_even,
            self.kernel.to(dtype=x.dtype),
            stride=2,
            groups=self.groups,
        )
        b, _, h2, w2 = bands.shape
        bands = bands.reshape(b, self.in_channels, 4, h2, w2)
        return bands[:, :, 0], bands[:, :, 1], bands[:, :, 2], bands[:, :, 3]


class HaarIDWT2D(nn.Module):
    """Fixed inverse matching :class:`HaarDWT2D` exactly."""

    def __init__(self, out_channels: int = 4):
        super().__init__()
        if out_channels < 1:
            raise ValueError("out_channels must be at least 1")

        self.out_channels = int(out_channels)
        kernels = torch.tensor(
            [
                [[[0.5, 0.5], [0.5, 0.5]]],       # LL
                [[[0.5, -0.5], [0.5, -0.5]]],     # HL
                [[[0.5, 0.5], [-0.5, -0.5]]],     # LH
                [[[0.5, -0.5], [-0.5, 0.5]]],     # HH
            ],
            dtype=torch.float32,
        )
        kernels = kernels.repeat(self.out_channels, 1, 1, 1)
        self.register_buffer("kernel", kernels, persistent=True)

    def forward(
        self,
        ll: torch.Tensor,
        hl: torch.Tensor,
        lh: torch.Tensor,
        hh: torch.Tensor,
        output_size: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        shapes = {tuple(t.shape) for t in (ll, hl, lh, hh)}
        if len(shapes) != 1:
            raise ValueError("LL, HL, LH and HH must have identical shapes")
        if ll.ndim != 4 or ll.shape[1] != self.out_channels:
            raise ValueError(
                f"Expected BCHW bands with C={self.out_channels}, got {tuple(ll.shape)}"
            )

        b, c, h, w = ll.shape
        packed = torch.stack((ll, hl, lh, hh), dim=2).reshape(b, c * 4, h, w)
        reconstructed = F.conv_transpose2d(
            packed,
            self.kernel.to(dtype=ll.dtype),
            stride=2,
            groups=self.out_channels,
        )

        if output_size is not None:
            out_h, out_w = output_size
            if out_h < 1 or out_w < 1:
                raise ValueError("output_size values must be positive")
            reconstructed = reconstructed[..., :out_h, :out_w]
        return reconstructed


class PositiveDepthwiseSmoothing(nn.Module):
    """Trainable depthwise smoothing with positive, sum-to-one kernels."""

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        padding_mode: str = "reflect",
    ):
        super().__init__()
        if channels < 1:
            raise ValueError("channels must be at least 1")
        if kernel_size % 2 == 0 or kernel_size < 1:
            raise ValueError("kernel_size must be a positive odd number")
        if padding_mode not in {"reflect", "replicate"}:
            raise ValueError("padding_mode must be 'reflect' or 'replicate'")

        self.channels = int(channels)
        self.kernel_size = int(kernel_size)
        self.padding = kernel_size // 2
        self.padding_mode = padding_mode

        self.logits = nn.Parameter(
            torch.zeros(self.channels, 1, kernel_size, kernel_size)
        )

    def normalized_kernel(self) -> torch.Tensor:
        flat = self.logits.flatten(start_dim=2)
        return torch.softmax(flat, dim=-1).view_as(self.logits)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != self.channels:
            raise ValueError(
                f"Expected BCHW tensor with C={self.channels}, got {tuple(x.shape)}"
            )

        mode = self.padding_mode
        if mode == "reflect" and (
            x.shape[-2] <= self.padding or x.shape[-1] <= self.padding
        ):
            mode = "replicate"
        x_pad = F.pad(
            x,
            (self.padding, self.padding, self.padding, self.padding),
            mode=mode,
        )
        return F.conv2d(
            x_pad,
            self.normalized_kernel().to(dtype=x.dtype),
            groups=self.channels,
        )


class HaarEdgeIllumination(nn.Module):
    """Refined Haar illumination-correction module with spatial context restoration."""

    _MAD_NORMALIZER = 0.6744897501960817

    def __init__(
        self,
        in_channels: int = 4,
        eps: float = 1e-6,
        threshold_init: Sequence[float] = (1.0, 1.0, 1.0),
        edge_rho: float = 1.0,
        delta_max: float = 1.0,
        detach_noise_estimate: bool = True,
        padding_mode: str = "reflect",
    ):
        super().__init__()
        if in_channels < 1:
            raise ValueError("in_channels must be at least 1")
        if eps <= 0:
            raise ValueError("eps must be positive")
        if edge_rho <= 0:
            raise ValueError("edge_rho must be positive")
        if delta_max <= 0:
            raise ValueError("delta_max must be positive")

        threshold_tensor = torch.as_tensor(threshold_init, dtype=torch.float32)
        if threshold_tensor.numel() == 1:
            threshold_tensor = threshold_tensor.repeat(3)
        if threshold_tensor.numel() != 3 or torch.any(threshold_tensor <= 0):
            raise ValueError("threshold_init must contain three positive values")

        self.in_channels = int(in_channels)
        self.eps = float(eps)
        self.edge_rho = float(edge_rho)
        self.delta_max = float(delta_max)
        self.detach_noise_estimate = bool(detach_noise_estimate)

        self.dwt = HaarDWT2D(in_channels, padding_mode=padding_mode)
        self.idwt = HaarIDWT2D(in_channels)

        raw_init = torch.log(torch.expm1(threshold_tensor.clamp_min(1e-4)))
        self.raw_threshold_scales = nn.Parameter(raw_init)

        self.smoother = PositiveDepthwiseSmoothing(
            in_channels,
            kernel_size=3,
            padding_mode=padding_mode,
        )

        # اصلاح: استفاده از یک بلوک کانولوشن محلی $3\times3$ برای بازگرداندن درک فضایی
        # بجای کانولوشن ساده $1\times1$ که باعث کوری شبکه می‌شد.
        self.spatial_context_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=True),
            nn.SiLU(),
            nn.Conv2d(in_channels, 1, kernel_size=1, bias=True)
        )
        
        nn.init.constant_(self.spatial_context_conv[-1].weight, 1e-3 / in_channels)
        nn.init.zeros_(self.spatial_context_conv[-1].bias)

    @property
    def threshold_scales(self) -> torch.Tensor:
        return F.softplus(self.raw_threshold_scales) + self.eps

    def _estimate_noise_mad(self, hh: torch.Tensor) -> torch.Tensor:
        flat = hh.flatten(start_dim=2)
        center = flat.median(dim=2, keepdim=True).values
        mad = (flat - center).abs().median(dim=2, keepdim=True).values
        sigma = (mad / self._MAD_NORMALIZER).view(
            hh.shape[0], hh.shape[1], 1, 1
        )
        if self.detach_noise_estimate:
            sigma = sigma.detach()
        return sigma

    @staticmethod
    def _soft_threshold(x: torch.Tensor, threshold: torch.Tensor) -> torch.Tensor:
        return torch.sign(x) * F.relu(x.abs() - threshold)

    def forward(
        self,
        l_stack: torch.Tensor,
        return_aux: bool = False,
    ):
        if l_stack.ndim != 4:
            raise ValueError(
                f"Expected a 4-D BCHW tensor, got shape {tuple(l_stack.shape)}"
            )
        if l_stack.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} channels, got {l_stack.shape[1]}"
            )

        original_size = l_stack.shape[-2:]

        ll, hl, lh, hh = self.dwt(l_stack)
        sigma = self._estimate_noise_mad(hh)

        lambdas = self.threshold_scales
        tau_hl = lambdas[0].view(1, 1, 1, 1) * sigma
        tau_lh = lambdas[1].view(1, 1, 1, 1) * sigma
        tau_hh = lambdas[2].view(1, 1, 1, 1) * sigma

        hl_d = self._soft_threshold(hl, tau_hl)
        lh_d = self._soft_threshold(lh, tau_lh)
        hh_d = self._soft_threshold(hh, tau_hh)

        detail_power = hl_d.square() + lh_d.square() + hh_d.square()
        detail_magnitude = (
            torch.sqrt(detail_power + self.eps) - self.eps ** 0.5
        ).clamp_min(0.0)
        edge_per_channel = torch.tanh(
            detail_magnitude / (self.edge_rho * sigma + self.eps)
        )
        edge_half = edge_per_channel.mean(dim=1, keepdim=True)

        x_d = self.idwt(ll, hl_d, lh_d, hh_d, output_size=original_size)
        x_s = self.smoother(x_d)

        edge_full = F.interpolate(
            edge_half,
            size=original_size,
            mode="bilinear",
            align_corners=False,
        ).clamp_(0.0, 1.0)
        z = edge_full * x_d + (1.0 - edge_full) * x_s

        # اعمال فیوژن فضایی جدید برای حفظ بافت و کنتراست نوری
        q = self.spatial_context_conv(z)
        delta_l = self.delta_max * torch.tanh(q)

        if not return_aux:
            return delta_l

        aux = {
            "sigma": sigma,
            "threshold_scales": lambdas,
            "thresholds": (tau_hl, tau_lh, tau_hh),
            "denoised_bands": (hl_d, lh_d, hh_d),
            "edge_confidence_half": edge_half,
            "edge_confidence": edge_full,
            "x_denoised": x_d,
            "x_smoothed": x_s,
            "fused_features": z,
        }
        return delta_l, aux

    def trainable_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class MultinexNano(nn.Module):
    """
    RetinexMSEFMultiLum with direct (non-gating) attention application and 3-stage block layout:
      Branch = pre_blocks -> [optional attention] -> mid_blocks -> post_blocks
      Illum:  maps -> stem(C) -> pre -> (attn?) -> mid -> post -> head(1)
      Chroma: maps -> stem(C) -> pre -> (attn?) -> mid -> post -> head(3)
      Fusion: out = chroma3 * delta_luma1 (+ optional residual)

    When use_haar_edge_illum=True, the illumination branch is replaced by the
    refined fixed-DWT / adaptive-threshold / edge-aware-smoothing pipeline.

    All previous knobs preserved; new knobs add maximum flexibility.
    """
    def __init__(self,
             in_ch: int = 3,
             out_ch: int = 3,
             base_channels: int = 32,
             use_depthwise: bool = True,
             per_illum_proj: bool = True,
             reduction_ratio: int = 16,
             width_mult: float = 1.0,
             target_params: Optional[int] = None,
             illum_flags: Optional[Dict[str, bool]] = None,
             act=nn.SiLU,

             chroma_flags: Optional[Dict[str, bool]] = None,
             per_chroma_proj: bool = True,

             use_illum_attn: bool = True,
             use_chroma_attn: bool = True,

             illum_mid: int = 1,
             chroma_mid: int = 1,

             # Heads & fusion
             luma_head_act: Optional[str] = 'sigmoid',
             chroma_head_act: Optional[str] = 'tanh',
             retinex_residual: bool = True,
             eps: float = 1e-6,

             use_haar_edge_illum: bool = False,
             haar_threshold_init: Sequence[float] = (1.0, 1.0, 1.0),
             haar_edge_rho: float = 1.0,
             haar_delta_max: float = 1.0,
             haar_detach_noise_estimate: bool = True,
             ):
        super().__init__()
        self.use_depthwise = use_depthwise
        self.per_illum_proj = per_illum_proj
        self.per_chroma_proj = per_chroma_proj
        self.reduction_ratio = reduction_ratio
        self.base_channels = base_channels
        self.width_mult = width_mult
        self.act = act
        self.eps = eps
        self.luma_head_act = luma_head_act
        self.chroma_head_act = chroma_head_act
        self.retinex_residual = retinex_residual

        self.use_illum_attn = use_illum_attn
        self.use_chroma_attn = use_chroma_attn
        self.use_haar_edge_illum = use_haar_edge_illum

        # ---- extractors
        self.illum_extractor = IlluminationExtractor(illum_flags)
        self.chroma_extractor = ChrominanceExtractor(chroma_flags)

        # determine K_L and K_C
        with torch.no_grad():
            dummy = torch.zeros(1,3,8,8)
            K_L = self.illum_extractor(dummy).shape[1]
            K_C = self.chroma_extractor(dummy).shape[1]
        self.K_L, self.K_C = K_L, K_C

        # ---- channel base
        C = max(3, int(round(self.base_channels * self.width_mult)))
        self.C = C

        # ---- stems
        self.chroma_stem = nn.Conv2d(max(1, K_C), C, kernel_size=1, bias=True)

        if self.use_haar_edge_illum:
            # K_L is normally 4. max(1, K_L) also supports an empty extractor.
            self.haar_illum = HaarEdgeIllumination(
                in_channels=max(1, self.K_L),
                eps=eps,
                threshold_init=haar_threshold_init,
                edge_rho=haar_edge_rho,
                delta_max=haar_delta_max,
                detach_noise_estimate=haar_detach_noise_estimate,
            )
        else:
            self.illum_stem  = nn.Conv2d(max(1, K_L), C, kernel_size=1, bias=True)

            # ---- attention modules per branch (produce (B,K, H, W))
            self.illum_att = nn.Conv2d(max(1, K_L), max(1, K_L), kernel_size=7, stride=1, padding=3, groups=max(1, K_L), bias=True)

            # ---- attention projection K->C per branch
            if self.per_illum_proj:
                self.illum_att_proj = nn.ModuleList([nn.Conv2d(1, C, 1, 1, 0, bias=True) for _ in range(max(1,K_L))])
            else:
                self.illum_att_proj = nn.Conv2d(max(1,K_L), C, 1, 1, 0, bias=True)

            self.illum_mid_seq = self._make_stack_blocks(C, C, depth=illum_mid)

        self.chroma_att = nn.Conv2d(max(1, K_C), max(1, K_C), kernel_size=7, stride=1, padding=3, groups=max(1, K_C), bias=True)

        if self.per_chroma_proj:
            self.chroma_att_proj = nn.ModuleList([nn.Conv2d(1, C, 1, 1, 0, bias=True) for _ in range(max(1,K_C))])
        else:
            self.chroma_att_proj = nn.Conv2d(max(1,K_C), C, 1, 1, 0, bias=True)

        self.chroma_mid_seq = self._make_stack_blocks(C, C, depth=chroma_mid)

        # ---- heads
        if not self.use_haar_edge_illum:
            self.head_luma = nn.Conv2d(C, 1, kernel_size=1, bias=True)
        self.head_chroma = nn.Conv2d(C, out_ch, kernel_size=1, bias=True)

        # ---- optional param budget auto-shrink
        if target_params is not None:
            self._fit_param_budget(target_params, in_ch, out_ch)

    # ---- helpers (unchanged except we don’t build BN/act gates now)
    def _make_block(self, c_in, c_out):
        if self.use_depthwise:
            return nn.Sequential(
                DWSeparableConv(c_in, c_out, k=3, s=1, p=1, act=self.act),
                MSEFBlock(c_out, self.reduction_ratio),
            )
        else:
            return nn.Sequential(
                ConvBNAct(c_in, c_out, k=3, s=1, p=1, act=self.act),
                MSEFBlock(c_out, self.reduction_ratio),
            )

    def _make_stack_blocks(self, c_in, c_out, depth=1):
        return nn.Sequential(*[self._make_block(c_in, c_out) for _ in range(depth)]) if depth > 0 else nn.Identity()

    def _att_project_to_C(self, att, proj):
        # att: (B,K,H,W) ; proj: ModuleList or Conv2d
        if isinstance(proj, nn.ModuleList):
            out = 0.0
            for i in range(att.shape[1]):
                out = out + proj[i](att[:, i:i+1])
            return out
        else:
            return proj(att)

    def _apply_head_act(self, x, kind: Optional[str]):
        if kind is None:
            return x
        k = kind.lower() if isinstance(kind, str) else None
        if k == 'sigmoid': return torch.sigmoid(x)
        if k == 'tanh':    return torch.tanh(x)
        if k == 'relu':    return F.relu(x, inplace=False)
        return x

    def _fit_param_budget(self, target_params: int, in_ch: int, out_ch: int):
        lo, hi = 0.25, self.width_mult
        best = self.width_mult
        for _ in range(10):
            mid = (lo + hi) / 2
            self.width_mult = mid
            C = max(3, int(round(self.base_channels * self.width_mult)))
            self.C = C

            self.chroma_stem = nn.Conv2d(max(1,self.K_C), C, 1, 1, 0, bias=True)

            if not self.use_haar_edge_illum:
                self.illum_stem  = nn.Conv2d(max(1,self.K_L), C, 1, 1, 0, bias=True)
                self.illum_mid_seq = self._make_stack_blocks(
                    C, C, depth=self.illum_mid_seq.__len__() if isinstance(self.illum_mid_seq, nn.Sequential) else 0)
                if self.per_illum_proj:
                    self.illum_att_proj = nn.ModuleList([nn.Conv2d(1, C, 1, 1, 0, bias=True) for _ in range(max(1,self.K_L))])
                else:
                    self.illum_att_proj = nn.Conv2d(max(1,self.K_L), C, 1, 1, 0, bias=True)
                self.head_luma = nn.Conv2d(C, 1, kernel_size=1, bias=True)

            self.chroma_mid_seq = self._make_stack_blocks(
                C, C, depth=self.chroma_mid_seq.__len__() if isinstance(self.chroma_mid_seq, nn.Sequential) else 0)
            self.head_chroma = nn.Conv2d(C, out_ch, kernel_size=1, bias=True)

            if self.per_chroma_proj:
                self.chroma_att_proj = nn.ModuleList([nn.Conv2d(1, C, 1, 1, 0, bias=True) for _ in range(max(1,self.K_C))])
            else:
                self.chroma_att_proj = nn.Conv2d(max(1,self.K_C), C, 1, 1, 0, bias=True)

            p = count_params(self)
            if p <= target_params:
                best = mid
                lo = mid
            else:
                hi = mid
        self.width_mult = best

    # ---- forward (with direct attention application) ----
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, H_init, W_init = x.shape
        rgb_in = x
        B, Cin, H, W = x.shape

        C_stack = self.chroma_extractor(x)
        if C_stack.shape[1] == 0:
            C_stack = x.new_zeros(B,1,H,W)

        fC = self.chroma_stem(C_stack)

        if self.use_chroma_attn:
            C_att = self.chroma_att(C_stack)
            C_mask = torch.sigmoid(self._att_project_to_C(C_att, self.chroma_att_proj))
            fC = fC * C_mask
        fC = self.chroma_mid_seq(fC)

        if self.use_haar_edge_illum:
            L_stack = self.illum_extractor(x)
            if L_stack.shape[1] == 0:
                L_stack = x.new_zeros(B, 1, H, W)
            L_hat = self.haar_illum(L_stack) 
        else:
            L_stack = self.illum_extractor(x)
            if L_stack.shape[1] == 0:
                L_stack = x.new_zeros(B,1,H,W)
            fL = self.illum_stem(L_stack)
            if self.use_illum_attn:
                L_att = self.illum_att(L_stack)
                L_mask = torch.sigmoid(self._att_project_to_C(L_att, self.illum_att_proj))
                fL = fL * L_mask
            fL = self.illum_mid_seq(fL)
            L_hat = self.head_luma(fL)
            L_hat = self._apply_head_act(L_hat, self.luma_head_act)

        C_hat = self.head_chroma(fC)
        C_hat = self._apply_head_act(C_hat, self.chroma_head_act)

        out = C_hat * L_hat
        if self.retinex_residual:
            out = out + rgb_in
        return out
    def param_count(self) -> int:
        return count_params(self)
