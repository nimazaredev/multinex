# basicsr/data/mixed_degradation_dataset.py
# Copyright (c) 2026 Alexandru Brateanu
# Multinex is licensed for non-commercial research and educational use only.
# Commercial use requires prior written permission.
# See LICENSE for details.

import random
import torch
from torchvision.transforms.functional import normalize

from basicsr.data.paired_image_dataset import Dataset_PairedImage
from basicsr.utils import FileClient, imfrombytes, img2tensor, padding
from basicsr.data.transforms import paired_random_crop, random_augmentation


class Dataset_PairedImage_MixedDegradation(Dataset_PairedImage):
    """Paired image dataset for 'Ours', identical to Dataset_PairedImage but
    with an extra Mixed-Degradation augmentation applied ONLY during training.

    With probability p in [0.3, 0.5], the LQ patch's illumination/intensity
    is scaled down by a random factor in [0.1, 0.4] to simulate extreme
    low-light / high-noise conditions. The GT patch is left untouched.

    This class does not affect validation/testing (opt['phase'] != 'train'),
    and it is only wired into "Ours" training configs, never into baseline
    configs.
    """

    def __init__(self, opt):
        super(Dataset_PairedImage_MixedDegradation, self).__init__(opt)
        # Degradation application probability, sampled once per dataset
        # instance from [0.3, 0.5]; can also be re-sampled per-item if you
        # prefer per-sample randomness (see __getitem__ below).
        self.mixed_deg_p_range = opt.get('mixed_deg_p_range', [0.5, 0.8])
        self.mixed_deg_scale_range = opt.get('mixed_deg_scale_range', [0.02, 0.15])
        self.mixed_deg_add_noise = opt.get('mixed_deg_add_noise', True)
        self.mixed_deg_noise_std_range = opt.get('mixed_deg_noise_std_range', [0.0, 0.02])

    def _apply_mixed_degradation(self, img_lq: torch.Tensor) -> torch.Tensor:
        """Randomly darken the LQ tensor in-place-safe manner.

        Args:
            img_lq (Tensor): LQ tensor, shape (C, H, W), assumed range [0, 1].

        Returns:
            Tensor: possibly-degraded LQ tensor, clipped to [0, 1].
        """
        p_apply = random.uniform(*self.mixed_deg_p_range)
        if random.random() < p_apply:
            scale = random.uniform(*self.mixed_deg_scale_range)
            img_lq = img_lq * scale
            img_lq = torch.clamp(img_lq, 0.0, 1.0)
        return img_lq

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']
        index = index % len(self.paths)

        gt_path = self.paths[index]['gt_path']
        img_bytes = self.file_client.get(gt_path, 'gt')
        try:
            img_gt = imfrombytes(img_bytes, float32=True)
        except Exception:
            raise Exception("gt path {} not working".format(gt_path))

        lq_path = self.paths[index]['lq_path']
        img_bytes = self.file_client.get(lq_path, 'lq')
        try:
            img_lq = imfrombytes(img_bytes, float32=True)
        except Exception:
            raise Exception("lq path {} not working".format(lq_path))

        # augmentation for training (identical to base class)
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            img_gt, img_lq = padding(img_gt, img_lq, gt_size)
            img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale, gt_path)
            if self.geometric_augs:
                img_gt, img_lq = random_augmentation(img_gt, img_lq)

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=True, float32=True)

        # ---- Mixed-Degradation Augmentation: TRAIN-ONLY, "Ours"-ONLY ----
        if self.opt['phase'] == 'train':
            img_lq = self._apply_mixed_degradation(img_lq)
        # ------------------------------------------------------------------

        # normalize (unchanged from base class)
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)

        return {
            'lq': img_lq,
            'gt': img_gt,
            'lq_path': lq_path,
            'gt_path': gt_path
        }