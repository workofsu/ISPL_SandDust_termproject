import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class MeanShift(nn.Conv2d):
    def __init__(self, rgb_range, rgb_mean, rgb_std, sign=-1):
        super().__init__(3, 3, kernel_size=1)
        std = torch.tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.weight.data.div_(std.view(3, 1, 1, 1))
        self.bias.data = sign * rgb_range * torch.tensor(rgb_mean)
        self.bias.data.div_(std)
        # Freeze parameters properly
        for p in self.parameters():
            p.requires_grad = False

class VGG(nn.Module):
    """
    conv_index:
      '22' -> conv2_2까지
      '54' -> conv5_4까지
    필요하면 SLICE 맵을 조정하세요(버전별 features 인덱스 차이 가능).
    """
    _SLICE = {
        '22': 9,   # 일부 버전에선 8일 수 있음
        '54': 36   # 일부 버전에선 35일 수 있음
    }

    def __init__(self, conv_index='22', rgb_range=1, pretrained=True):
        super().__init__()
        conv_index = str(conv_index)
        if conv_index not in self._SLICE:
            raise ValueError(f"Unsupported conv_index={conv_index}. Choose from {list(self._SLICE.keys())}.")

        # 권장 API (torchvision>=0.13)
        weights = models.VGG19_Weights.IMAGENET1K_V1 if pretrained else None
        vgg_features = models.vgg19(weights=weights).features

        cut = self._SLICE[conv_index]
        self.vgg = nn.Sequential(*list(vgg_features)[:cut])

        vgg_mean = (0.485, 0.456, 0.406)
        vgg_std  = (0.229 * rgb_range, 0.224 * rgb_range, 0.225 * rgb_range)
        self.sub_mean = MeanShift(rgb_range, vgg_mean, vgg_std)

        # 고정 + 평가 모드
        self.vgg.eval()
        self.vgg.requires_grad_(False)

    def forward(self, sr, hr):
        # 입력은 [0,1] 범위 3채널(RGB)라고 가정
        def _forward(x):
            x = self.sub_mean(x)
            return self.vgg(x)

        vgg_sr = _forward(sr)
        with torch.no_grad():
            vgg_hr = _forward(hr)

        return F.mse_loss(vgg_sr, vgg_hr)
