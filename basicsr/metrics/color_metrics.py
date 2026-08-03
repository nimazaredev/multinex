# basicsr/metrics/color_metrics.py
"""
Color-accuracy and reliability-interval metrics for low-light enhancement.
All functions assume:
    img1, img2: HWC numpy arrays, RGB order, float in [0,1] (or uint8 [0,255])
"""
import numpy as np
import torch
from skimage.color import rgb2lab, deltaE_ciede2000

from basicsr.metrics.metric_util import reorder_image


def _to_float01(img):
    img = img.astype(np.float64)
    if img.max() > 1.0:
        img = img / 255.0
    return img


# ----------------------------------------------------------------------
# 1) CIEDE2000
# ----------------------------------------------------------------------
def calculate_ciede2000(img1, img2, crop_border=0, input_order='HWC'):
    """Mean CIEDE2000 perceptual color-difference over the image.

    Returns:
        float: mean delta-E (lower is better).
    """
    img1 = reorder_image(np.asarray(img1), input_order=input_order)
    img2 = reorder_image(np.asarray(img2), input_order=input_order)
    img1 = _to_float01(img1)
    img2 = _to_float01(img2)

    if crop_border != 0:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

    lab1 = rgb2lab(img1)
    lab2 = rgb2lab(img2)
    delta_e = deltaE_ciede2000(lab1, lab2)  # HxW map
    return float(np.mean(delta_e))


def calculate_ciede2000_map(img1, img2, crop_border=0, input_order='HWC'):
    """Same as above but returns the full per-pixel delta-E map,
    needed for interval binning."""
    img1 = reorder_image(np.asarray(img1), input_order=input_order)
    img2 = reorder_image(np.asarray(img2), input_order=input_order)
    img1 = _to_float01(img1)
    img2 = _to_float01(img2)

    if crop_border != 0:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

    lab1 = rgb2lab(img1)
    lab2 = rgb2lab(img2)
    return deltaE_ciede2000(lab1, lab2)  # HxW


# ----------------------------------------------------------------------
# 2) Angular Chromaticity Error  (a.k.a. RGB angular color error)
#    angle between the two RGB vectors at each pixel
# ----------------------------------------------------------------------
def calculate_angular_error(img1, img2, crop_border=0, input_order='HWC', eps=1e-8):
    """Mean angular error (in degrees) between RGB vectors of pred/gt.

    ang = arccos( (p . g) / (|p||g|) )
    """
    img1 = reorder_image(np.asarray(img1), input_order=input_order)
    img2 = reorder_image(np.asarray(img2), input_order=input_order)
    img1 = _to_float01(img1)
    img2 = _to_float01(img2)

    if crop_border != 0:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

    dot = np.sum(img1 * img2, axis=-1)
    norm1 = np.linalg.norm(img1, axis=-1)
    norm2 = np.linalg.norm(img2, axis=-1)
    cos_theta = dot / (norm1 * norm2 + eps)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    ang_map = np.degrees(np.arccos(cos_theta))
    return float(np.mean(ang_map))


def calculate_angular_error_map(img1, img2, crop_border=0, input_order='HWC', eps=1e-8):
    img1 = reorder_image(np.asarray(img1), input_order=input_order)
    img2 = reorder_image(np.asarray(img2), input_order=input_order)
    img1 = _to_float01(img1)
    img2 = _to_float01(img2)
    if crop_border != 0:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]
    dot = np.sum(img1 * img2, axis=-1)
    norm1 = np.linalg.norm(img1, axis=-1)
    norm2 = np.linalg.norm(img2, axis=-1)
    cos_theta = np.clip(dot / (norm1 * norm2 + eps), -1.0, 1.0)
    return np.degrees(np.arccos(cos_theta))


# ----------------------------------------------------------------------
# 3) Saturation Error (HSV-S difference)
# ----------------------------------------------------------------------
def _rgb_to_saturation(img):
    """img: HWC float [0,1] -> HxW saturation map S = 1 - min/max."""
    mx = np.max(img, axis=-1)
    mn = np.min(img, axis=-1)
    return (mx - mn) / (mx + 1e-8)


def calculate_saturation_error(img1, img2, crop_border=0, input_order='HWC'):
    """Mean absolute saturation error between pred and gt."""
    img1 = reorder_image(np.asarray(img1), input_order=input_order)
    img2 = reorder_image(np.asarray(img2), input_order=input_order)
    img1 = _to_float01(img1)
    img2 = _to_float01(img2)
    if crop_border != 0:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

    s1 = _rgb_to_saturation(img1)
    s2 = _rgb_to_saturation(img2)
    return float(np.mean(np.abs(s1 - s2)))


def calculate_saturation_error_map(img1, img2, crop_border=0, input_order='HWC'):
    img1 = reorder_image(np.asarray(img1), input_order=input_order)
    img2 = reorder_image(np.asarray(img2), input_order=input_order)
    img1 = _to_float01(img1)
    img2 = _to_float01(img2)
    if crop_border != 0:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]
    s1 = _rgb_to_saturation(img1)
    s2 = _rgb_to_saturation(img2)
    return np.abs(s1 - s2)


# ----------------------------------------------------------------------
# 4) Reliability / Brightness map (used for binning)
# ----------------------------------------------------------------------
def compute_confidence_map(gt_img, mode='luminance'):
    """Builds the C / Y map used to bin pixels.

    mode='luminance' -> Rec.709 luma of GT, in [0,1]
    You may swap this for the model's own illumination map if you log it.
    """
    gt = _to_float01(np.asarray(gt_img))
    w = np.array([0.2126, 0.7152, 0.0722])
    y = np.tensordot(gt, w, axes=([-1], [0]))
    return y  # HxW, range ~[0,1]


# ----------------------------------------------------------------------
# 5) Interval binning driver
# ----------------------------------------------------------------------
DEFAULT_BINS = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]


def _masked_psnr(pred, gt, mask, max_value=1.0):
    if mask.sum() < 1:
        return None
    diff2 = (pred - gt) ** 2
    mse = diff2[mask].mean() if diff2.ndim == 2 else diff2[..., :][np.broadcast_to(mask[..., None], diff2.shape)].mean()
    if mse == 0:
        return float('inf')
    return 20. * np.log10(max_value / np.sqrt(mse))


def calculate_binned_metrics(pred_img, gt_img, bins=None, crop_border=0,
                             conf_map=None, dark_flat_thresh=0.1,
                             dark_flat_std_thresh=0.02):
    """Computes PSNR / SSIM(proxy) / CIEDE2000 / saturation-error per
    brightness bin, plus a dedicated 'dark & flat region' error.

    Args:
        pred_img, gt_img: HWC, [0,1] or [0,255]
        bins: list of (lo, hi) tuples; default = 5 bins of width 0.2
        conf_map: optional externally supplied HxW confidence/illumination
                  map in [0,1]. If None, computed from GT luminance.
        dark_flat_thresh: luminance below which a region is "dark".
        dark_flat_std_thresh: local std below which a region is "flat"
                              (uses a simple 3x3 local std as a cheap proxy).

    Returns:
        dict: {
          'bins': {'[0.0-0.2)': {'psnr':.., 'ciede2000':.., 'sat_err':.., 'count':..}, ...},
          'dark_flat': {'ciede2000':.., 'sat_err':.., 'count':..}
        }
    """
    from skimage.color import rgb2lab
    from skimage.metrics import structural_similarity as sk_ssim

    bins = bins or DEFAULT_BINS
    pred = _to_float01(np.asarray(pred_img))
    gt = _to_float01(np.asarray(gt_img))

    if crop_border != 0:
        pred = pred[crop_border:-crop_border, crop_border:-crop_border, ...]
        gt = gt[crop_border:-crop_border, crop_border:-crop_border, ...]

    if conf_map is None:
        conf_map = compute_confidence_map(gt)
    else:
        conf_map = conf_map[crop_border:-crop_border, crop_border:-crop_border] \
            if crop_border != 0 else conf_map

    # per-pixel error maps computed once, then masked per bin (fast)
    de_map = deltaE_ciede2000(rgb2lab(pred), rgb2lab(gt))
    sat_map = calculate_saturation_error_map(pred, gt)  # already 0..1 imgs
    diff2 = np.mean((pred - gt) ** 2, axis=-1)  # per-pixel MSE across channels

    results = {'bins': {}}
    for lo, hi in bins:
        key = f'[{lo:.1f}-{hi:.1f})' if hi < 1.0 else f'[{lo:.1f}-{hi:.1f}]'
        mask = (conf_map >= lo) & (conf_map < hi if hi < 1.0 else conf_map <= hi)
        count = int(mask.sum())
        if count == 0:
            results['bins'][key] = {'psnr': None, 'ciede2000': None,
                                    'sat_err': None, 'count': 0}
            continue

        mse = diff2[mask].mean()
        psnr = float('inf') if mse == 0 else 20. * np.log10(1.0 / np.sqrt(mse))
        ciede = float(de_map[mask].mean())
        sat_err = float(sat_map[mask].mean())

        results['bins'][key] = {
            'psnr': psnr, 'ciede2000': ciede, 'sat_err': sat_err, 'count': count
        }

    # ---- dark & flat region metric ----
    gray = np.mean(gt, axis=-1)
    # cheap local-std proxy via box filter (no extra deps)
    from scipy.ndimage import uniform_filter
    local_mean = uniform_filter(gray, size=3)
    local_sqmean = uniform_filter(gray ** 2, size=3)
    local_std = np.sqrt(np.clip(local_sqmean - local_mean ** 2, 0, None))

    dark_flat_mask = (gray < dark_flat_thresh) & (local_std < dark_flat_std_thresh)
    dcount = int(dark_flat_mask.sum())
    if dcount > 0:
        results['dark_flat'] = {
            'ciede2000': float(de_map[dark_flat_mask].mean()),
            'sat_err': float(sat_map[dark_flat_mask].mean()),
            'count': dcount
        }
    else:
        results['dark_flat'] = {'ciede2000': None, 'sat_err': None, 'count': 0}

    return results