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
    """Fixed one-level orthonormal 2D Haar transform."""

    def __init__(self, in_channels: int = 1) -> None:
        super().__init__()
        if in_channels < 1:
            raise ValueError("in_channels must be positive")

        self.in_channels = in_channels

        # 2D outer products of [1, 1] / sqrt(2) and [1, -1] / sqrt(2).
        # With this normalization, a constant 2x2 block with value c has LL=2c.
        kernels = torch.tensor(
            [
                [[[0.5, 0.5], [0.5, 0.5]]],       # LL
                [[[0.5, -0.5], [0.5, -0.5]]],     # HL
                [[[0.5, 0.5], [-0.5, -0.5]]],     # LH
                [[[0.5, -0.5], [-0.5, 0.5]]],     # HH
            ],
            dtype=torch.float32,
        )
        kernels = kernels.repeat(in_channels, 1, 1, 1)
        self.register_buffer("kernel", kernels, persistent=True)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if x.ndim != 4:
            raise ValueError(f"Expected BCHW input, got shape {tuple(x.shape)}")
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} channels, got {x.shape[1]}"
            )

        batch, channels, height, width = x.shape
        pad_h, pad_w = height % 2, width % 2
        if pad_h or pad_w:
            # Reflection is preferred, but it is invalid when a padded dimension is 1.
            pad_mode = "reflect" if height > 1 and width > 1 else "replicate"
            x = F.pad(x, (0, pad_w, 0, pad_h), mode=pad_mode)

        coeffs = F.conv2d(
            x,
            self.kernel,
            stride=2,
            groups=self.in_channels,
        )
        coeffs = coeffs.reshape(
            batch,
            channels,
            4,
            coeffs.shape[-2],
            coeffs.shape[-1],
        )
        return (
            coeffs[:, :, 0],  # LL
            coeffs[:, :, 1],  # HL
            coeffs[:, :, 2],  # LH
            coeffs[:, :, 3],  # HH
        )


class HaarReliabilityEstimator(nn.Module):
    """
    Estimate a full-resolution reliability map from RGB input.

    Haar is used only for reliability estimation:
        1. LL/2 estimates local signal strength.
        2. HH supplies a robust per-image noise scale through MAD.
        3. Detail coefficients are clipped to the estimated noise range.
        4. C = mu / (mu + nu + tau).

    The module has either zero learned parameters or one learned scalar (tau).
    """

    def __init__(
        self,
        noise_threshold: float = 2.5,
        tau_init: float = 1e-3,
        learnable_tau: bool = True,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if noise_threshold <= 0:
            raise ValueError("noise_threshold must be positive")
        if tau_init <= 0:
            raise ValueError("tau_init must be positive")
        if eps <= 0:
            raise ValueError("eps must be positive")

        self.noise_threshold = float(noise_threshold)
        self.learnable_tau = bool(learnable_tau)
        self.eps = float(eps)
        self.dwt = HaarDWT2D(in_channels=1)

        if self.learnable_tau:
            # Initialize softplus(raw_tau) exactly at tau_init.
            raw_tau = math.log(math.expm1(tau_init))
            self.raw_tau = nn.Parameter(torch.tensor(raw_tau, dtype=torch.float32))
        else:
            self.register_buffer(
                "fixed_tau",
                torch.tensor(tau_init, dtype=torch.float32),
                persistent=True,
            )

        self.register_buffer(
            "rgb_to_luma",
            torch.tensor([0.2126, 0.7152, 0.0722], dtype=torch.float32).view(
                1, 3, 1, 1
            ),
            persistent=True,
        )

    def _tau(self, reference: torch.Tensor) -> torch.Tensor:
        if self.learnable_tau:
            tau = F.softplus(self.raw_tau)
        else:
            tau = self.fixed_tau
        return tau.to(device=reference.device, dtype=reference.dtype)

    @staticmethod
    def _spatial_median(x: torch.Tensor) -> torch.Tensor:
        """Median over HxW, returned as BxCx1x1."""
        batch, channels = x.shape[:2]
        return x.flatten(2).median(dim=2).values.reshape(batch, channels, 1, 1)

    @staticmethod
    def _symmetric_clip(x: torch.Tensor, bound: torch.Tensor) -> torch.Tensor:
        """Elementwise clip to [-bound, bound], where bound may vary by image."""
        return torch.maximum(torch.minimum(x, bound), -bound)

    def forward(
        self,
        rgb: torch.Tensor,
        return_statistics: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        if rgb.ndim != 4 or rgb.shape[1] != 3:
            raise ValueError(
                f"Expected RGB tensor with shape Bx3xHxW, got {tuple(rgb.shape)}"
            )

        height, width = rgb.shape[-2:]
        luma_weights = self.rgb_to_luma.to(device=rgb.device, dtype=rgb.dtype)
        y = (rgb * luma_weights).sum(dim=1, keepdim=True)

        ll, hl, lh, hh = self.dwt(y)
        mu = (ll / 2.0).clamp_min(0.0)

        # Robust, per-image MAD estimate from HH. Stop-gradient follows the proposal.
        hh_center = self._spatial_median(hh)
        mad = self._spatial_median((hh - hh_center).abs())
        sigma = (mad / 0.6745).detach()

        bound = self.noise_threshold * sigma
        n_hl = self._symmetric_clip(hl, bound)
        n_lh = self._symmetric_clip(lh, bound)
        n_hh = self._symmetric_clip(hh, bound)

        nu = torch.sqrt(
            (n_hl.square() + n_lh.square() + n_hh.square()) / 3.0
            + self.eps
        )
        tau = self._tau(mu)
        confidence_half = mu / (mu + nu + tau)
        confidence = F.interpolate(
            confidence_half,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        ).clamp_(0.0, 1.0)

        if not return_statistics:
            return confidence

        stats = {
            "luminance": y,
            "signal_half": mu,
            "uncertainty_half": nu,
            "noise_sigma": sigma,
            "confidence_half": confidence_half,
            "tau": tau.reshape(1),
        }
        return confidence, stats


class MultinexNano(nn.Module):
    """
    Multinex-Nano with Haar-guided reliability-adaptive chromatic correction.

    Both original Multinex branches are preserved:
        L_hat = luminance correction, shape Bx1xHxW
        C_hat = RGB reflectance correction, shape Bx3xHxW

    The final reflectance correction is
        C_ach + reliability * C_chr,
    where C_ach is the RGB channel mean and C_chr is the zero-mean
    color-changing component.
    """

    def __init__(
        self,
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
        luma_head_act: Optional[str] = "sigmoid",
        chroma_head_act: Optional[str] = "tanh",
        retinex_residual: bool = True,
        eps: float = 1e-6,
        # New reliability settings
        use_haar_reliability: bool = True,
        haar_noise_threshold: float = 2.5,
        haar_tau_init: float = 1e-3,
        learnable_haar_tau: bool = True,
        # Backward-compatible alias for old configs. Its old illumination-replacement
        # behavior is intentionally removed; True now enables the new reliability path.
        use_haar_edge_illum: Optional[bool] = None,
    ) -> None:
        super().__init__()
        if in_ch != 3 or out_ch != 3:
            raise ValueError(
                "Reliability-adaptive chromatic correction is defined for RGB "
                "input/output; set in_ch=out_ch=3."
            )
        if illum_mid < 0 or chroma_mid < 0:
            raise ValueError("illum_mid and chroma_mid must be non-negative")

        if use_haar_edge_illum is not None:
            # This preserves configuration compatibility without preserving the obsolete
            # Haar illumination branch. The flag now selects the proposed method.
            use_haar_reliability = bool(use_haar_edge_illum)

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
        self.use_haar_reliability = use_haar_reliability
        self.illum_mid = illum_mid
        self.chroma_mid = chroma_mid
        self.out_ch = out_ch

        self.illum_extractor = IlluminationExtractor(illum_flags)
        self.chroma_extractor = ChrominanceExtractor(chroma_flags)

        with torch.no_grad():
            dummy = torch.zeros(1, 3, 8, 8)
            self.K_L = self.illum_extractor(dummy).shape[1]
            self.K_C = self.chroma_extractor(dummy).shape[1]

        if self.use_haar_reliability:
            self.haar_reliability = HaarReliabilityEstimator(
                noise_threshold=haar_noise_threshold,
                tau_init=haar_tau_init,
                learnable_tau=learnable_haar_tau,
                eps=eps,
            )
        else:
            self.haar_reliability = None

        channels = max(3, int(round(self.base_channels * self.width_mult)))
        self._build_width_dependent_modules(channels)

        if target_params is not None:
            self._fit_param_budget(target_params)

    def _make_block(self, c_in: int, c_out: int) -> nn.Module:
        if self.use_depthwise:
            return nn.Sequential(
                DWSeparableConv(c_in, c_out, k=3, s=1, p=1, act=self.act),
                MSEFBlock(c_out, self.reduction_ratio),
            )
        return nn.Sequential(
            ConvBNAct(c_in, c_out, k=3, s=1, p=1, act=self.act),
            MSEFBlock(c_out, self.reduction_ratio),
        )

    def _make_stack_blocks(
        self, c_in: int, c_out: int, depth: int = 1
    ) -> nn.Module:
        if depth <= 0:
            return nn.Identity()
        return nn.Sequential(
            *[self._make_block(c_in, c_out) for _ in range(depth)]
        )

    def _build_width_dependent_modules(self, channels: int) -> None:
        """Build all modules whose shape depends on the hidden width."""
        self.C = channels

        self.illum_stem = nn.Conv2d(max(1, self.K_L), channels, 1, bias=True)
        self.chroma_stem = nn.Conv2d(max(1, self.K_C), channels, 1, bias=True)

        self.illum_att = nn.Conv2d(
            max(1, self.K_L),
            max(1, self.K_L),
            kernel_size=7,
            stride=1,
            padding=3,
            groups=max(1, self.K_L),
            bias=True,
        )
        self.chroma_att = nn.Conv2d(
            max(1, self.K_C),
            max(1, self.K_C),
            kernel_size=7,
            stride=1,
            padding=3,
            groups=max(1, self.K_C),
            bias=True,
        )

        if self.per_illum_proj:
            self.illum_att_proj = nn.ModuleList(
                [nn.Conv2d(1, channels, 1, bias=True) for _ in range(max(1, self.K_L))]
            )
        else:
            self.illum_att_proj = nn.Conv2d(
                max(1, self.K_L), channels, 1, bias=True
            )

        if self.per_chroma_proj:
            self.chroma_att_proj = nn.ModuleList(
                [nn.Conv2d(1, channels, 1, bias=True) for _ in range(max(1, self.K_C))]
            )
        else:
            self.chroma_att_proj = nn.Conv2d(
                max(1, self.K_C), channels, 1, bias=True
            )

        self.illum_mid_seq = self._make_stack_blocks(
            channels, channels, depth=self.illum_mid
        )
        self.chroma_mid_seq = self._make_stack_blocks(
            channels, channels, depth=self.chroma_mid
        )
        self.head_luma = nn.Conv2d(channels, 1, kernel_size=1, bias=True)
        self.head_chroma = nn.Conv2d(
            channels, self.out_ch, kernel_size=1, bias=True
        )

    def _att_project_to_C(
        self, att: torch.Tensor, proj: Union[nn.ModuleList, nn.Conv2d]
    ) -> torch.Tensor:
        if isinstance(proj, nn.ModuleList):
            projected = proj[0](att[:, 0:1])
            for index in range(1, att.shape[1]):
                projected = projected + proj[index](att[:, index : index + 1])
            return projected
        return proj(att)

    @staticmethod
    def _apply_head_act(x: torch.Tensor, kind: Optional[str]) -> torch.Tensor:
        if kind is None:
            return x
        key = kind.lower() if isinstance(kind, str) else ""
        if key == "sigmoid":
            return torch.sigmoid(x)
        if key == "tanh":
            return torch.tanh(x)
        if key == "relu":
            return F.relu(x, inplace=False)
        if key in {"none", "identity", "linear"}:
            return x
        raise ValueError(f"Unsupported head activation: {kind}")

    def _fit_param_budget(self, target_params: int) -> None:
        if target_params <= 0:
            raise ValueError("target_params must be positive")

        lo, hi = 0.25, self.width_mult
        best = lo
        for _ in range(10):
            mid = (lo + hi) / 2.0
            channels = max(3, int(round(self.base_channels * mid)))
            self._build_width_dependent_modules(channels)
            if count_params(self) <= target_params:
                best = mid
                lo = mid
            else:
                hi = mid

        self.width_mult = best
        best_channels = max(3, int(round(self.base_channels * best)))
        self._build_width_dependent_modules(best_channels)

        if count_params(self) > target_params:
            raise ValueError(
                "target_params is below the minimum achievable size with the "
                "current fixed modules and minimum hidden width."
            )

    def _extract_stacks(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch, _, height, width = x.shape

        l_stack = self.illum_extractor(x)
        if l_stack.shape[1] == 0:
            l_stack = x.new_zeros(batch, 1, height, width)

        c_stack = self.chroma_extractor(x)
        if c_stack.shape[1] == 0:
            c_stack = x.new_zeros(batch, 1, height, width)

        return l_stack, c_stack

    def _forward_branch(
        self,
        stack: torch.Tensor,
        stem: nn.Conv2d,
        attention: nn.Conv2d,
        projection: Union[nn.ModuleList, nn.Conv2d],
        blocks: nn.Module,
        use_attention: bool,
    ) -> torch.Tensor:
        features = stem(stack)
        if use_attention:
            attention_features = attention(stack)
            mask = torch.sigmoid(
                self._att_project_to_C(attention_features, projection)
            )
            features = features * mask
        return blocks(features)

    @staticmethod
    def _reliability_adaptive_chroma(
        chroma: torch.Tensor, reliability: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        achromatic = chroma.mean(dim=1, keepdim=True)
        chromatic = chroma - achromatic
        adjusted = achromatic + reliability * chromatic
        return adjusted, achromatic, chromatic

    def forward(
        self,
        x: torch.Tensor,
        return_aux: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        if x.ndim != 4 or x.shape[1] != 3:
            raise ValueError(f"Expected Bx3xHxW input, got {tuple(x.shape)}")

        rgb_in = x
        l_stack, c_stack = self._extract_stacks(x)

        f_l = self._forward_branch(
            l_stack,
            self.illum_stem,
            self.illum_att,
            self.illum_att_proj,
            self.illum_mid_seq,
            self.use_illum_attn,
        )
        l_hat = self._apply_head_act(self.head_luma(f_l), self.luma_head_act)

        f_c = self._forward_branch(
            c_stack,
            self.chroma_stem,
            self.chroma_att,
            self.chroma_att_proj,
            self.chroma_mid_seq,
            self.use_chroma_attn,
        )
        c_hat = self._apply_head_act(
            self.head_chroma(f_c), self.chroma_head_act
        )

        if self.use_haar_reliability:
            if return_aux:
                reliability, reliability_stats = self.haar_reliability(
                    x, return_statistics=True
                )
            else:
                reliability = self.haar_reliability(x)
                reliability_stats = None

            c_adjusted, c_ach, c_chr = self._reliability_adaptive_chroma(
                c_hat, reliability
            )
        else:
            reliability = x.new_ones(x.shape[0], 1, x.shape[2], x.shape[3])
            reliability_stats = None
            c_adjusted = c_hat
            c_ach = c_hat.mean(dim=1, keepdim=True)
            c_chr = c_hat - c_ach

        correction = c_adjusted * l_hat
        output = rgb_in + correction if self.retinex_residual else correction

        if not return_aux:
            return output

        aux = {
            "reliability": reliability,
            "luma_correction": l_hat,
            "chroma_raw": c_hat,
            "chroma_achromatic": c_ach,
            "chroma_chromatic": c_chr,
            "chroma_adjusted": c_adjusted,
            "image_correction": correction,
        }
        if reliability_stats is not None:
            aux.update({f"haar_{key}": value for key, value in reliability_stats.items()})
        return output, aux

    def param_count(self) -> int:
        return count_params(self)
