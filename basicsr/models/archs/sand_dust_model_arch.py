"""
    Sand-dust removal model based on the dual-branch network.
"""

import math
import numbers

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class YB(nn.Module):
    """학습 가능한 휘도 강화 블록.

    처리 과정:
        RGB -> 학습 가능한 RGB-to-Y 변환
            -> min-max 정규화
            -> Gaussian 초기화 smoothing
            -> unsharp masking

    입력:
        x: [B, 3, H, W]

    출력:
        out: [B, 1, H, W]
    """

    def __init__(self, sharp_strength=0.7, eps=1e-6, learnable_blur=True):
        super().__init__()
        self.eps = eps

        # 학습 가능한 RGB -> Y 변환
        self.rgb_to_y = nn.Conv2d(
            in_channels=3,
            out_channels=1,
            kernel_size=1,
            bias=False
        )
        with torch.no_grad():
            weight = torch.tensor(
                [0.299, 0.587, 0.114],
                dtype=torch.float32
            ).view(1, 3, 1, 1)
            self.rgb_to_y.weight.copy_(weight)

        # Gaussian kernel로 초기화한 smoothing layer
        self.blur = nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=5,
            padding=2,
            bias=False,
            padding_mode="reflect"
        )
        kernel = torch.tensor(
            [
                [1, 4, 6, 4, 1],
                [4, 16, 24, 16, 4],
                [6, 24, 36, 24, 6],
                [4, 16, 24, 16, 4],
                [1, 4, 6, 4, 1],
            ],
            dtype=torch.float32
        )
        kernel = (kernel / kernel.sum()).view(1, 1, 5, 5)
        with torch.no_grad():
            self.blur.weight.copy_(kernel)

        if not learnable_blur:
            for parameter in self.blur.parameters():
                parameter.requires_grad = False

        # alpha 범위는 (0, 2)이며 초기값은 sharp_strength와 일치한다.
        sharp_strength = float(sharp_strength)
        sharp_strength = max(1e-4, min(sharp_strength, 2.0 - 1e-4))
        init_logit = math.log(sharp_strength / (2.0 - sharp_strength))
        self.sharp_logit = nn.Parameter(
            torch.tensor(init_logit, dtype=torch.float32)
        )

    def forward(self, x):
        y = self.rgb_to_y(x)

        y_min = y.amin(dim=(2, 3), keepdim=True)
        y_max = y.amax(dim=(2, 3), keepdim=True)
        y_norm = (y - y_min) / (y_max - y_min + self.eps)

        blur = self.blur(y_norm)
        detail = y_norm - blur
        alpha = 2.0 * torch.sigmoid(self.sharp_logit)
        y_sharp = y_norm + alpha * detail

        # 강화된 1채널 휘도 맵을 반환한다.
        return torch.clamp(y_sharp, 0.0, 1.0)


class SortUnsortB(nn.Module):
    def __init__(self, ref_channel=1, method="arithmetic"):
        super().__init__()
        self.method = method
        self.ref_channel = ref_channel
        self.eps = 1e-6

    def forward(self, feat):
        B, C, H, W = feat.shape
        feat_out = torch.zeros_like(feat)

        g = feat[:, self.ref_channel, :, :].reshape(B, -1)
        g_sorted, _ = torch.sort(g, dim=1)

        for c in range(C):
            if c == self.ref_channel:
                feat_out[:, c, :, :] = feat[:, c, :, :]
                continue

            x = feat[:, c, :, :].reshape(B, -1)
            x_sorted, idx = torch.sort(x, dim=1)

            if self.method == "arithmetic":
                aligned = (x_sorted + g_sorted) / 2
            else:
                aligned = 2 / (1 / (x_sorted + self.eps) + 1 / (g_sorted + self.eps))

            unsort_idx = torch.argsort(idx, dim=1)
            x_restored = torch.gather(
                aligned,
                dim=1,
                index=unsort_idx
            ).reshape(B, H, W)

            feat_out[:, c, :, :] = x_restored

        return torch.clamp(feat_out, 0, 1)


##########################################################################
## Utility Functions
##########################################################################
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


##########################################################################
## Layer Normalization
##########################################################################
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
        return x / torch.sqrt(sigma + 1e-5) * self.weight


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
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


##########################################################################
## Multi-Scale CNN Block (MSCB)
##   Parallel 3x3, 5x5, 7x7 convolutions + GELU -> Concat -> 3x3 fusion + GELU
##########################################################################
class MSCB(nn.Module):
    """Multi-kernel feature fusion; residual addition is handled by callers.

    The bias argument is retained for caller compatibility. Branch convolutions
    use no bias and fusion uses bias, matching the supplied multi-kernel block.
    """

    def __init__(self, dim, bias=False):
        super(MSCB, self).__init__()

        self.conv3 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GELU()
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=5, stride=1, padding=2, bias=False),
            nn.GELU()
        )
        self.conv7 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=7, stride=1, padding=3, bias=False),
            nn.GELU()
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(dim * 3, dim, kernel_size=3, padding=1),
            nn.GELU()
        )

    def forward(self, x):
        f3 = self.conv3(x)
        f5 = self.conv5(x)
        f7 = self.conv7(x)
        return self.fuse(torch.cat([f3, f5, f7], dim=1))


##########################################################################
## Average-pooling token mixer
##   Preserves spatial dimensions for the encoder/decoder residual paths
##########################################################################
class Attention(nn.Module):
    """Pooling-only replacement; constructor kept compatible with callers."""

    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.pool = nn.AvgPool2d(
            kernel_size=3, stride=1, padding=1, count_include_pad=False)

    def forward(self, x):
        return self.pool(x)


##########################################################################
## Full Spatial Self-Attention (FSSA)
##   Standard spatial multi-head self-attention (HW x HW attention map)
##   Designed for the bottleneck where spatial resolution is small (16x16)
##   Each spatial position attends to all other spatial positions
##########################################################################
class SpatialAttention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(SpatialAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1,
                                    groups=dim * 3, bias=bias)

        # Learnable relative position bias for spatial structure awareness
        # Registered as buffer-generating parameter; actual bias computed in forward
        self.pos_bias = nn.Parameter(torch.zeros(num_heads, 1, 1))

        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        # Spatial attention: tokens are spatial positions, channels are features
        # [B, heads, HW, head_dim]
        q = rearrange(q, 'b (head d) h w -> b head (h w) d', head=self.num_heads)
        k = rearrange(k, 'b (head d) h w -> b head (h w) d', head=self.num_heads)
        v = rearrange(v, 'b (head d) h w -> b head (h w) d', head=self.num_heads)

        # Scaled dot-product attention: [B, heads, HW, HW]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn + self.pos_bias
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head (h w) d -> b (head d) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


##########################################################################
## Standard Feed-Forward Network
##########################################################################
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3,
                                stride=1, padding=1, groups=hidden_features * 2, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


##########################################################################
## Selective Fusion Module (SFM)
##   SKA (channel-wise selective exchange) + Spatial Gate
##   Bidirectional: each branch adaptively absorbs from the other
##   Spatial gate handles spatially-varying weather degradation
##########################################################################
class SelectiveFusion(nn.Module):
    def __init__(self, dim, bias=False):
        super(SelectiveFusion, self).__init__()

        # Channel-wise selective exchange (SKA)
        self.fc_main = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        self.fc_guide = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        # Spatial gate: decides where to blend
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(dim * 2, dim, kernel_size=1, bias=bias),
            nn.Conv2d(dim, 1, kernel_size=7, stride=1, padding=3, bias=bias),
        )

    def forward(self, x_main, x_guide):
        # Channel-wise selective weights
        s = F.adaptive_avg_pool2d(x_main + x_guide, 1)        # [B, dim, 1, 1]
        w_main = torch.sigmoid(self.fc_main(s))                # [B, dim, 1, 1]
        w_guide = torch.sigmoid(self.fc_guide(s))              # [B, dim, 1, 1]

        # Spatial gate
        gate = torch.sigmoid(self.spatial_gate(
            torch.cat([x_main, x_guide], dim=1)))              # [B, 1, H, W]

        # Bidirectional fusion
        out_main = x_main + w_guide * x_guide * gate
        out_guide = x_guide + w_main * x_main * (1 - gate)

        return out_main, out_guide


##########################################################################
## Dual-Stream Aggregation Block  - Encoder (DSAB)
##   Main branch (CNN):  MSCB (independent)
##   Guide branch:  AvgPool -> FFN (independent)
##   Fusion: SelectiveFusion (lightweight inter-branch exchange)
##########################################################################
class DSB_Encoder(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(DSB_Encoder, self).__init__()

        ## Main branch: CNN path
        self.norm_main = LayerNorm(dim, LayerNorm_type)
        self.mscb = MSCB(dim, bias)

        ## Guide branch: pooling path (Contextual Feature Extraction Block(CFEB)
        self.norm_guide1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm_guide2 = LayerNorm(dim, LayerNorm_type)
        self.ffn_guide = FeedForward(dim, ffn_expansion_factor, bias)

        ## Selective Fusion
        self.fusion = SelectiveFusion(dim, bias)

    def forward(self, x_main, x_guide):
        ## Main branch: MSCB
        x_main = x_main + self.mscb(self.norm_main(x_main))

        ## Guide branch: AvgPool -> FFN (Contextual Feature Extraction Block(CFEB)
        x_guide = x_guide + self.attn(self.norm_guide1(x_guide))
        x_guide = x_guide + self.ffn_guide(self.norm_guide2(x_guide))

        ## Selective Fusion (bidirectional exchange)
        out_main, out_guide = self.fusion(x_main, x_guide)

        return out_main, out_guide


##########################################################################
## Dual-Stream Recovery Block - Decoder (DSRB)
##   Roles swapped from encoder:
##   Guide branch (CNN):  MSCB (independent)
##   Main branch:    AvgPool -> FFN (independent)
##   Fusion: SelectiveFusion (lightweight inter-branch exchange)
##########################################################################
class DSB_Decoder(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(DSB_Decoder, self).__init__()

        ## Guide branch: CNN path (swapped)
        self.norm_guide = LayerNorm(dim, LayerNorm_type)
        self.mscb = MSCB(dim, bias)

        ## Main branch: pooling path (swapped) (Contextual Feature Extraction Block(CFEB)
        self.norm_main1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm_main2 = LayerNorm(dim, LayerNorm_type)
        self.ffn_main = FeedForward(dim, ffn_expansion_factor, bias)

        ## Selective Fusion
        self.fusion = SelectiveFusion(dim, bias)

    def forward(self, x_main, x_guide):
        ## Guide branch (CNN): MSCB
        x_guide = x_guide + self.mscb(self.norm_guide(x_guide))

        ## Main branch: AvgPool -> FFN (Contextual Feature Extraction Block(CFEB)
        x_main = x_main + self.attn(self.norm_main1(x_main))
        x_main = x_main + self.ffn_main(self.norm_main2(x_main))

        ## Selective Fusion (bidirectional exchange)
        out_main, out_guide = self.fusion(x_main, x_guide)

        return out_main, out_guide


##########################################################################
## Joint Fusion Block
##   Concatenates main and guide branches along channel dimension
##   Applies full spatial self-attention on the joint representation
##   The attention naturally captures both intra-branch and inter-branch
##   spatial relationships in a single unified operation
##   At 16x16 resolution (256 tokens), full spatial attention is efficient
##   cat/chunk preserves clean residual path for training stability
##########################################################################
class LatentBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(LatentBlock, self).__init__()

        joint_dim = dim * 2   # concatenated main + guide channels

        ## Full Spatial Self-Attention on joint features
        self.norm_attn = LayerNorm(joint_dim, LayerNorm_type)
        self.spatial_attn = SpatialAttention(joint_dim, num_heads, bias)

        ## FFN on joint features
        self.norm_ffn = LayerNorm(joint_dim, LayerNorm_type)
        self.ffn = FeedForward(joint_dim, ffn_expansion_factor, bias)

    def forward(self, x_main, x_guide):
        ## Concatenate main and guide branches (no learned projection)
        x_joint = torch.cat([x_main, x_guide], dim=1)      # [B, dim*2, H, W]

        ## Spatial Self-Attention (HW x HW)
        x_joint = x_joint + self.spatial_attn(self.norm_attn(x_joint))

        ## FFN
        x_joint = x_joint + self.ffn(self.norm_ffn(x_joint))

        ## Split back into two branches (no learned projection)
        x_main, x_guide = x_joint.chunk(2, dim=1)           # each [B, dim, H, W]

        return x_main, x_guide


##########################################################################
## Overlapped image patch embedding with 3x3 Conv
##########################################################################
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)
        return x


##########################################################################
## Resizing modules
##########################################################################
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelUnshuffle(2)
        )

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2)
        )

    def forward(self, x):
        return self.body(x)


class SandDustModel(nn.Module):
    """Keep swt_new9's dual branches and use model_152-style inputs.

    Branch inputs:
        main:  RGB -> YB -> 1-channel enhanced luminance
        guide: RGB -> SortUnsortB twice -> 3-channel guide image

    Spatial dimensions are padded to a multiple of eight for the three
    downsampling stages, then cropped back to the original size.
    """

    def __init__(self,
                 inp_channels=3,
                 out_channels=3,
                 dim=48,
                 num_blocks=(4, 4, 6, 3),
                 heads=(1, 2, 4, 8),
                 ffn_expansion_factor=2.66,
                 latent_ffn_expansion_factor=1.,
                 bias=False,
                 LayerNorm_type='WithBias',
                 sharp_strength=0.7,
                 learnable_blur=True,
                 sort_ref_channel=1,
                 sort_method='arithmetic',
                 use_residual=True):
        super().__init__()

        if inp_channels != 3:
            raise ValueError('YB and SortUnsortB require inp_channels=3.')
        if not 0 <= sort_ref_channel < inp_channels:
            raise ValueError('sort_ref_channel must select an input channel.')
        if use_residual and out_channels != inp_channels:
            raise ValueError(
                'out_channels must equal inp_channels when use_residual=True.')
        if len(num_blocks) != 4 or len(heads) != 4:
            raise ValueError('num_blocks and heads must each contain four values.')

        self.use_residual = use_residual

        # model_152 input construction
        self.ymap = YB(
            sharp_strength=sharp_strength,
            learnable_blur=learnable_blur)
        self.sortunsort = SortUnsortB(
            ref_channel=sort_ref_channel,
            method=sort_method)

        # YB produces one channel; SortUnsortB preserves the RGB channels.
        self.patch_embed_main = OverlapPatchEmbed(1, dim, bias=bias)
        self.patch_embed_guide = OverlapPatchEmbed(inp_channels, dim, bias=bias)

        self.encoder1 = nn.ModuleList([
            DSB_Encoder(dim, heads[0], ffn_expansion_factor, bias,
                        LayerNorm_type)
            for _ in range(num_blocks[0])
        ])
        self.down_main1 = Downsample(dim)
        self.down_guide1 = Downsample(dim)

        self.encoder2 = nn.ModuleList([
            DSB_Encoder(dim * 2, heads[1], ffn_expansion_factor, bias,
                        LayerNorm_type)
            for _ in range(num_blocks[1])
        ])
        self.down_main2 = Downsample(dim * 2)
        self.down_guide2 = Downsample(dim * 2)

        self.encoder3 = nn.ModuleList([
            DSB_Encoder(dim * 4, heads[2], ffn_expansion_factor, bias,
                        LayerNorm_type)
            for _ in range(num_blocks[2])
        ])
        self.down_main3 = Downsample(dim * 4)
        self.down_guide3 = Downsample(dim * 4)

        latent_dim = dim * 8
        self.latent = nn.ModuleList([
            LatentBlock(latent_dim, heads[3],
                        latent_ffn_expansion_factor, bias, LayerNorm_type)
            for _ in range(num_blocks[3])
        ])

        self.up_main3 = Upsample(latent_dim)
        self.up_guide3 = Upsample(latent_dim)
        self.reduce_main3 = nn.Conv2d(dim * 8, dim * 4, 1, bias=bias)
        self.reduce_guide3 = nn.Conv2d(dim * 8, dim * 4, 1, bias=bias)
        self.decoder3 = nn.ModuleList([
            DSB_Decoder(dim * 4, heads[2], ffn_expansion_factor, bias,
                        LayerNorm_type)
            for _ in range(num_blocks[2])
        ])

        self.up_main2 = Upsample(dim * 4)
        self.up_guide2 = Upsample(dim * 4)
        self.reduce_main2 = nn.Conv2d(dim * 4, dim * 2, 1, bias=bias)
        self.reduce_guide2 = nn.Conv2d(dim * 4, dim * 2, 1, bias=bias)
        self.decoder2 = nn.ModuleList([
            DSB_Decoder(dim * 2, heads[1], ffn_expansion_factor, bias,
                        LayerNorm_type)
            for _ in range(num_blocks[1])
        ])

        self.up_main1 = Upsample(dim * 2)
        self.up_guide1 = Upsample(dim * 2)
        self.reduce_main1 = nn.Conv2d(dim * 2, dim, 1, bias=bias)
        self.reduce_guide1 = nn.Conv2d(dim * 2, dim, 1, bias=bias)
        self.decoder1 = nn.ModuleList([
            DSB_Decoder(dim, heads[0], ffn_expansion_factor, bias,
                        LayerNorm_type)
            for _ in range(num_blocks[0])
        ])

        # These tensors are not wavelet coefficients, so direct RGB fusion is
        # used in place of swt_new9's output_main/output_guide + ISWT path.
        self.output = nn.Sequential(
            nn.Conv2d(dim * 2, dim, kernel_size=1, bias=bias),
            nn.Conv2d(dim, out_channels, kernel_size=3, padding=1, bias=bias),
        )

    @staticmethod
    def _pad_to_multiple_of_eight(x):
        height, width = x.shape[-2:]
        pad_h = (8 - height % 8) % 8
        pad_w = (8 - width % 8) % 8
        if pad_h or pad_w:
            # Reflect padding needs every padding amount to be smaller than its
            # corresponding input dimension. Replicate also supports tiny inputs.
            mode = 'reflect' if pad_h < height and pad_w < width else 'replicate'
            x = F.pad(x, (0, pad_w, 0, pad_h), mode=mode)
        return x, height, width

    def forward(self, inp_img):
        x, height, width = self._pad_to_multiple_of_eight(inp_img)

        main = self.ymap(x)
        guide = self.sortunsort(self.sortunsort(x))

        x_main = self.patch_embed_main(main)
        x_guide = self.patch_embed_guide(guide)

        for block in self.encoder1:
            x_main, x_guide = block(x_main, x_guide)
        skip_main1, skip_guide1 = x_main, x_guide
        x_main = self.down_main1(x_main)
        x_guide = self.down_guide1(x_guide)

        for block in self.encoder2:
            x_main, x_guide = block(x_main, x_guide)
        skip_main2, skip_guide2 = x_main, x_guide
        x_main = self.down_main2(x_main)
        x_guide = self.down_guide2(x_guide)

        for block in self.encoder3:
            x_main, x_guide = block(x_main, x_guide)
        skip_main3, skip_guide3 = x_main, x_guide
        x_main = self.down_main3(x_main)
        x_guide = self.down_guide3(x_guide)

        for block in self.latent:
            x_main, x_guide = block(x_main, x_guide)

        x_main = self.reduce_main3(torch.cat(
            [self.up_main3(x_main), skip_main3], dim=1))
        x_guide = self.reduce_guide3(torch.cat(
            [self.up_guide3(x_guide), skip_guide3], dim=1))
        for block in self.decoder3:
            x_main, x_guide = block(x_main, x_guide)

        x_main = self.reduce_main2(torch.cat(
            [self.up_main2(x_main), skip_main2], dim=1))
        x_guide = self.reduce_guide2(torch.cat(
            [self.up_guide2(x_guide), skip_guide2], dim=1))
        for block in self.decoder2:
            x_main, x_guide = block(x_main, x_guide)

        x_main = self.reduce_main1(torch.cat(
            [self.up_main1(x_main), skip_main1], dim=1))
        x_guide = self.reduce_guide1(torch.cat(
            [self.up_guide1(x_guide), skip_guide1], dim=1))
        for block in self.decoder1:
            x_main, x_guide = block(x_main, x_guide)

        out = self.output(torch.cat([x_main, x_guide], dim=1))
        if self.use_residual:
            out = out + x

        return out[:, :, :height, :width]


if __name__ == '__main__':
    model = SandDustModel(
        dim=16, num_blocks=(1, 1, 1, 1), heads=(1, 2, 4, 8))
    test_input = torch.randn(1, 3, 65, 67)
    test_output = model(test_input)
    print(test_input.shape, test_output.shape)
