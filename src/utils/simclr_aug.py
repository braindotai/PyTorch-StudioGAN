# Training GANs with Stronger Augmentations via Contrastive Discriminator
# Jongheon Jeong, Jinwoo Shin

import math
import numbers
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import affine_grid, grid_sample
from torch.autograd import Function



class SimCLRAugment(nn.Module):
    def __init__(self, s):
        super(SimCLRAugment, self).__init__()
        self.RandomResizeCrop = RandomResizeCropLayer(scale=(0.5, 1.0), ratio=(3./4., 4./3.))
        self.HorizontalFlip = HorizontalFlipLayer()
        self.ColorJitter = ColorJitterLayer(0.8*s, 0.8*s, 0.8*s, 0.*s)

    def forward(self, x):
        x = self.RandomResizeCrop(x)
        x = self.HorizontalFlip(x)
        x = self.ColorJitter(x)
        return x


class HorizontalFlipLayer(nn.Module):
    def __init__(self):
        super(HorizontalFlipLayer, self).__init__()

        _eye = torch.eye(2, 3)
        self.register_buffer('_eye', _eye)

    def forward(self, inputs):
        _device = inputs.device

        N = inputs.size(0)
        _theta = self._eye.repeat(N, 1, 1)
        r_sign = torch.bernoulli(torch.ones(N, device=_device) * 0.5) * 2 - 1
        _theta[:, 0, 0] = r_sign
        grid = affine_grid(_theta, inputs.size(), align_corners=False)
        output = grid_sample(inputs, grid, padding_mode='reflection', align_corners=False)
        return output


class RandomResizeCropLayer(nn.Module):
    def __init__(self, scale, ratio=(3./4., 4./3.)):
        '''
            Inception Crop
            scale (tuple): range of size of the origin size cropped
            ratio (tuple): range of aspect ratio of the origin aspect ratio cropped
        '''
        super(RandomResizeCropLayer, self).__init__()

        _eye = torch.eye(2, 3)
        self.register_buffer('_eye', _eye)
        self.scale = scale
        self.ratio = ratio

    def forward(self, inputs):
        _device = inputs.device
        N, _, width, height = inputs.shape

        _theta = self._eye.repeat(N, 1, 1)

        # N * 10 trial
        area = height * width
        target_area = np.random.uniform(*self.scale, N * 10) * area
        log_ratio = (math.log(self.ratio[0]), math.log(self.ratio[1]))
        aspect_ratio = np.exp(np.random.uniform(*log_ratio, N * 10))

        # If doesn't satisfy ratio condition, then do central crop
        w = np.round(np.sqrt(target_area * aspect_ratio))
        h = np.round(np.sqrt(target_area / aspect_ratio))
        cond = (0 < w) * (w <= width) * (0 < h) * (h <= height)
        w = w[cond]
        h = h[cond]
        if len(w) > N:
            inds = np.random.choice(len(w), N, replace=False)
            w = w[inds]
            h = h[inds]
        transform_len = len(w)

        r_w_bias = np.random.randint(w - width, width - w + 1) / width
        r_h_bias = np.random.randint(h - height, height - h + 1) / height
        w = w / width
        h = h / height

        _theta[:transform_len, 0, 0] = torch.tensor(w, device=_device)
        _theta[:transform_len, 1, 1] = torch.tensor(h, device=_device)
        _theta[:transform_len, 0, 2] = torch.tensor(r_w_bias, device=_device)
        _theta[:transform_len, 1, 2] = torch.tensor(r_h_bias, device=_device)

        grid = affine_grid(_theta, inputs.size(), align_corners=False)
        output = grid_sample(inputs, grid, padding_mode='reflection', align_corners=False)

        return output


class ColorJitterLayer(nn.Module):
    def __init__(self, brightness, contrast, saturation, hue):
        super(ColorJitterLayer, self).__init__()
        self.brightness = self._check_input(brightness, 'brightness')
        self.contrast = self._check_input(contrast, 'contrast')
        self.saturation = self._check_input(saturation, 'saturation')
        self.hue = self._check_input(hue, 'hue', center=0, bound=(-0.5, 0.5),
                                     clip_first_on_zero=False)


    def _check_input(self, value, name, center=1, bound=(0, float('inf')), clip_first_on_zero=True):
        if isinstance(value, numbers.Number):
            if value < 0:
                raise ValueError("If {} is a single number, it must be non negative.".format(name))
            value = [center - value, center + value]
            if clip_first_on_zero:
                value[0] = max(value[0], 0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                raise ValueError("{} values should be between {}".format(name, bound))
        else:
            raise TypeError("{} should be a single number or a list/tuple with lenght 2.".format(name))

        # if value is 0 or (1., 1.) for brightness/contrast/saturation
        # or (0., 0.) for hue, do nothing
        if value[0] == value[1] == center:
            value = None
        return value

    def adjust_contrast(self, x):
        if self.contrast:
            factor = x.new_empty(x.size(0), 1, 1, 1).uniform_(*self.contrast)
            means = torch.mean(x, dim=[2, 3], keepdim=True)
            x = (x - means) * factor + means
        return torch.clamp(x, 0, 1)

    def adjust_hsv(self, x):
        f_h = x.new_zeros(x.size(0), 1, 1)
        f_s = x.new_ones(x.size(0), 1, 1)
        f_v = x.new_ones(x.size(0), 1, 1)

        if self.hue:
            f_h.uniform_(*self.hue)
        if self.saturation:
            f_s = f_s.uniform_(*self.saturation)
        if self.brightness:
            f_v = f_v.uniform_(*self.brightness)

        return RandomHSVFunction.apply(x, f_h, f_s, f_v)

    def transform(self, inputs):
        # Shuffle transform
        if np.random.rand() > 0.5:
            transforms = [self.adjust_contrast, self.adjust_hsv]
        else:
            transforms = [self.adjust_hsv, self.adjust_contrast]

        for t in transforms:
            inputs = t(inputs)

        return inputs

    def forward(self, inputs):
        return self.transform(inputs)


class ContrastJitterLayer(nn.Module):
    def __init__(self, contrast):
        super(ContrastJitterLayer, self).__init__()
        self.contrast = self._check_input(contrast, 'contrast')

    def _check_input(self, value, name, center=1, bound=(0, float('inf')), clip_first_on_zero=True):
        if isinstance(value, numbers.Number):
            if value < 0:
                raise ValueError("If {} is a single number, it must be non negative.".format(name))
            value = [center - value, center + value]
            if clip_first_on_zero:
                value[0] = max(value[0], 0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                raise ValueError("{} values should be between {}".format(name, bound))
        else:
            raise TypeError("{} should be a single number or a list/tuple with lenght 2.".format(name))

        # if value is 0 or (1., 1.) for brightness/contrast/saturation
        # or (0., 0.) for hue, do nothing
        if value[0] == value[1] == center:
            value = None
        return value

    def adjust_contrast(self, x):
        if self.contrast:
            factor = x.new_empty(x.size(0), 1, 1, 1).uniform_(*self.contrast)
            means = torch.mean(x, dim=[2, 3], keepdim=True)
            x = (x - means) * factor + means
        return torch.clamp(x, 0, 1)

    def forward(self, inputs):
        return self.adjust_contrast(inputs)



class RandomHSVFunction(Function):
    @staticmethod
    def forward(ctx, x, f_h, f_s, f_v):
        # ctx is a context object that can be used to stash information
        # for backward computation
        x = rgb2hsv(x)
        h = x[:, 0, :, :]
        h += (f_h * 255. / 360.)
        h = (h % 1)
        x[:, 0, :, :] = h
        x[:, 1, :, :] = x[:, 1, :, :] * f_s
        x[:, 2, :, :] = x[:, 2, :, :] * f_v
        x = torch.clamp(x, 0, 1)
        x = hsv2rgb(x)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        # We return as many input gradients as there were arguments.
        # Gradients of non-Tensor arguments to forward must be None.
        grad_input = None
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.clone()
        return grad_input, None, None, None


def rgb2hsv(rgb):
    """Convert a 4-d RGB tensor to the HSV counterpart.
    Here, we compute hue using atan2() based on the definition in [1],
    instead of using the common lookup table approach as in [2, 3].
    Those values agree when the angle is a multiple of 30°,
    otherwise they may differ at most ~1.2°.
    >>> %timeit rgb2hsv(rgb)
    1.07 ms ± 2.96 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)
    >>> %timeit rgb2hsv_fast(rgb)
    380 µs ± 555 ns per loop (mean ± std. dev. of 7 runs, 1000 loops each)
    >>> (rgb2hsv(rgb) - rgb2hsv_fast(rgb)).abs().max()
    tensor(0.0031, device='cuda:0')
    References
    [1] https://en.wikipedia.org/wiki/Hue
    [2] https://www.rapidtables.com/convert/color/rgb-to-hsv.html
    [3] https://github.com/scikit-image/scikit-image/blob/master/skimage/color/colorconv.py#L212
    """

    r, g, b = rgb[:, 0, :, :], rgb[:, 1, :, :], rgb[:, 2, :, :]

    Cmax = rgb.max(1)[0]
    Cmin = rgb.min(1)[0]

    hue = torch.atan2(math.sqrt(3) * (g - b), 2 * r - g - b)
    hue = (hue % (2 * math.pi)) / (2 * math.pi)
    saturate = 1 - Cmin / (Cmax + 1e-8)
    value = Cmax
    hsv = torch.stack([hue, saturate, value], dim=1)
    hsv[~torch.isfinite(hsv)] = 0.
    return hsv


def hsv2rgb(hsv):
    """Convert a 4-d HSV tensor to the RGB counterpart.

    >>> %timeit hsv2rgb(hsv)
    2.37 ms ± 13.4 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
    >>> %timeit rgb2hsv_fast(rgb)
    298 µs ± 542 ns per loop (mean ± std. dev. of 7 runs, 1000 loops each)
    >>> torch.allclose(hsv2rgb(hsv), hsv2rgb_fast(hsv), atol=1e-6)
    True

    References
    [1] https://en.wikipedia.org/wiki/HSL_and_HSV#HSV_to_RGB_alternative
    """

    h, s, v = hsv[:, [0]], hsv[:, [1]], hsv[:, [2]]
    c = v * s

    n = hsv.new_tensor([5, 3, 1]).view(3, 1, 1)
    k = (n + h * 6) % 6
    t = torch.min(k, 4.-k)
    t = torch.clamp(t, 0, 1)

    return v - c * t
