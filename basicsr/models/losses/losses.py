import torch
from torch import nn as nn
from torch.nn import functional as F
import numpy as np
from basicsr.models.losses.ssim import SSIM
from basicsr.models.losses.vgg import VGG

from basicsr.models.losses.loss_util import weighted_loss

_reduction_modes = ['none', 'mean', 'sum']


@weighted_loss
def l1_loss(pred, target):
    return F.l1_loss(pred, target, reduction='none')


@weighted_loss
def mse_loss(pred, target):
    return F.mse_loss(pred, target, reduction='none')

# @weighted_loss
# def charbonnier_loss(pred, target, eps=1e-12):
#     return torch.sqrt((pred - target)**2 + eps)


class L1Loss(nn.Module):
    """L1 (mean absolute error, MAE) loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(L1Loss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * l1_loss(
            pred, target, weight, reduction=self.reduction)

class MSELoss(nn.Module):
    """MSE (L2) loss.

    Args:
        loss_weight (float): Loss weight for MSE loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(MSELoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * mse_loss(
            pred, target, weight, reduction=self.reduction)

class PSNRLoss(nn.Module):

    def __init__(self, loss_weight=1.0, reduction='mean', toY=False):
        super(PSNRLoss, self).__init__()
        assert reduction == 'mean'
        self.loss_weight = loss_weight
        self.scale = 10 / np.log(10)
        self.toY = toY
        self.coef = torch.tensor([65.481, 128.553, 24.966]).reshape(1, 3, 1, 1)
        self.first = True

    def forward(self, pred, target):
        assert len(pred.size()) == 4
        if self.toY:
            if self.first:
                self.coef = self.coef.to(pred.device)
                self.first = False

            pred = (pred * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.
            target = (target * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.

            pred, target = pred / 255., target / 255.
            pass
        assert len(pred.size()) == 4

        return self.loss_weight * self.scale * torch.log(((pred - target) ** 2).mean(dim=(1, 2, 3)) + 1e-8).mean()

class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, loss_weight=1.0, reduction='mean', eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        # loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps*self.eps)))
        return loss

class SSIMLoss(nn.Module):
    """
    SSIM 기반 손실: 1 - SSIM(pred, target)
    - SSIM은 스칼라(또는 배치 스칼라)를 내므로 @weighted_loss를 쓰지 않습니다.
    - 입력 범위는 보통 [0,1] 가정.
    """
    def __init__(self, loss_weight=1.0, reduction='mean', **kwargs):
        super().__init__()
        if reduction not in _reduction_modes:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported: {_reduction_modes}')
        self.loss_weight = loss_weight
        self.reduction = reduction
        self.ssim = SSIM(**kwargs)  # 네가 만든 SSIM 구현의 인자에 맞게 전달

    def forward(self, pred, target, weight=None, **kwargs):
        ssim_val = self.ssim(pred, target)  # scalar 또는 [N]
        loss = 1.0 - ssim_val
        if isinstance(loss, torch.Tensor) and loss.ndim > 0:
            if self.reduction == 'mean':
                loss = loss.mean()
            elif self.reduction == 'sum':
                loss = loss.sum()
        return self.loss_weight * loss


class LabColorBalanceLoss(nn.Module):
    """Match local and global chroma to a paired ground-truth image.

    RGB inputs are converted to normalized CIE Lab.  The pixel term preserves
    spatially varying object colours, while the mean term directly penalizes
    an image-wide red/green or blue/yellow cast.  Luminance is deliberately
    excluded because pixel and SSIM losses already supervise it.
    """

    def __init__(self, loss_weight=0.05, mean_weight=0.5,
                 reduction='mean'):
        super().__init__()
        if reduction != 'mean':
            raise ValueError('LabColorBalanceLoss only supports mean reduction.')
        self.loss_weight = float(loss_weight)
        self.mean_weight = float(mean_weight)

    @staticmethod
    def _rgb_to_normalized_ab(rgb):
        # Training tensors are sRGB in [0, 1].  MSE remains responsible for
        # pulling residual outputs that temporarily leave this range back in.
        rgb = rgb.clamp(0.0, 1.0)
        linear = torch.where(
            rgb > 0.04045,
            ((rgb + 0.055) / 1.055).pow(2.4),
            rgb / 12.92)
        red, green, blue = linear.unbind(dim=1)

        # sRGB D65 -> XYZ, normalized by the D65 reference white.
        x = (0.4124564 * red + 0.3575761 * green + 0.1804375 * blue) / 0.95047
        y = 0.2126729 * red + 0.7151522 * green + 0.0721750 * blue
        z = (0.0193339 * red + 0.1191920 * green + 0.9503041 * blue) / 1.08883

        delta = 6.0 / 29.0
        threshold = delta ** 3

        def lab_f(value):
            # torch.where evaluates both branches during autograd.  Taking a
            # cube root exactly at zero has an infinite derivative, which can
            # contaminate the otherwise inactive branch with NaN gradients.
            # Clamp the cube-root branch at its switching threshold so both
            # branches always have finite derivatives.
            cube_root = value.clamp_min(threshold).pow(1.0 / 3.0)
            return torch.where(
                value > threshold,
                cube_root,
                value / (3.0 * delta ** 2) + 4.0 / 29.0)

        fx, fy, fz = lab_f(x), lab_f(y), lab_f(z)
        a = 500.0 * (fx - fy) / 128.0
        b = 200.0 * (fy - fz) / 128.0
        return torch.stack((a, b), dim=1)

    def forward(self, pred, target):
        pred_ab = self._rgb_to_normalized_ab(pred)
        target_ab = self._rgb_to_normalized_ab(target)

        pixel_loss = F.smooth_l1_loss(pred_ab, target_ab)
        pred_mean = pred_ab.mean(dim=(2, 3))
        target_mean = target_ab.mean(dim=(2, 3))
        mean_loss = F.smooth_l1_loss(pred_mean, target_mean)
        return self.loss_weight * (
            pixel_loss + self.mean_weight * mean_loss)

class VGGLoss(nn.Module):
    """
    VGG Perceptual Loss 래퍼.
    - 보통 입력 [0,1], 3채널(RGB) 가정.
    - 네가 만든 VGG 구현의 인자(layer_weights 등)는 YAML에서 opt로 넘어오게 하면 편함.
    """
    def __init__(self, loss_weight=1.0, reduction='mean', **kwargs):
        super().__init__()
        self.loss_weight = loss_weight
        self.vgg = VGG(**kwargs)
        # 특징 추출용 파라미터는 고정하는 것이 일반적
        for p in self.vgg.parameters():
            p.requires_grad = False

    def forward(self, pred, target, **kwargs):
        loss = self.vgg(pred, target)  # 스칼라 또는 [N]
        if isinstance(loss, torch.Tensor) and loss.ndim > 0:
            loss = loss.mean()
        return self.loss_weight * loss
