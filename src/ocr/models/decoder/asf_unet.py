'''
DBNet++ Decoder with Adaptive Scale Fusion (ASF)

참고 논문: https://arxiv.org/abs/2202.10304
참고 레포: https://github.com/MhLiao/DB
'''

from itertools import accumulate
import torch
import torch.nn as nn


class ASFModule(nn.Module):
    '''
    Adaptive Scale Fusion
    UNet이 만든 out_features (각 64ch, 모두 같은 H×W)를 받아서
    ① 스케일 어텐션: 4개 피처 중 어느 스케일이 중요한지 가중치 계산
    ② 공간 어텐션:  각 피처에 픽셀별 중요도 마스크 적용
    DBHead가 기대하는 list 형태 그대로 반환 → DBHead 수정 불필요
    '''
    def __init__(self, out_channels, num_scales=4):
        super().__init__()
        self.num_scales = num_scales

        # ① 스케일 어텐션
        # 4개 피처를 concat → GlobalAvgPool → FC → 스케일별 가중치 (합=1)
        self.scale_pool = nn.AdaptiveAvgPool2d(1)
        self.scale_fc = nn.Sequential(
            nn.Linear(out_channels * num_scales, num_scales),
            nn.Softmax(dim=1)
        )

        # ② 공간 어텐션
        # 가중합 피처 → Conv → 스케일별 공간 마스크 (num_scales개 채널)
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(out_channels, out_channels // 4, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 4, num_scales, kernel_size=1),
            nn.Sigmoid()
        )

        self.scale_fc.apply(self._init_weights)
        self.spatial_attn.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight.data)
            if m.bias is not None:
                nn.init.constant_(m.bias, 1.0)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight.data)

    def forward(self, features):
        '''
        features: list of (B, C, H, W) — UNet out_features, 모두 같은 크기
        returns:  list of (B, C, H, W) — 어텐션 적용된 피처, 형태 동일
        '''
        B, C, H, W = features[0].shape

        # ① 스케일 어텐션: 어느 스케일이 중요한가?
        concat = torch.cat(features, dim=1)              # (B, C*N, H, W)
        pooled = self.scale_pool(concat).view(B, -1)     # (B, C*N)
        scale_w = self.scale_fc(pooled)                  # (B, N)  합=1.0

        # 스케일 가중합 → 공간 어텐션의 입력
        weighted_sum = sum(
            features[i] * scale_w[:, i].view(B, 1, 1, 1)
            for i in range(self.num_scales)
        )                                                # (B, C, H, W)

        # ② 공간 어텐션: 어느 위치가 중요한가?  (스케일별로 따로)
        spatial_maps = self.spatial_attn(weighted_sum)   # (B, N, H, W)

        # 각 피처에 자신의 공간 마스크 곱하기
        result = [
            features[i] * spatial_maps[:, i:i+1, :, :]
            for i in range(self.num_scales)
        ]                                                # list of (B, C, H, W)
        return result


class ASFUNet(nn.Module):
    '''
    UNet Decoder + ASF
    UNet과 완전히 동일한 구조, forward 마지막에 ASFModule만 추가
    in_channels, strides, inner_channels, output_channels → unet.yaml과 동일하게 설정
    '''
    def __init__(self,
                 in_channels=[96, 192, 384, 768],
                 strides=[4, 8, 16, 32],
                 inner_channels=256,
                 output_channels=64,
                 bias=False):
        super().__init__()

        assert len(strides) == len(in_channels)
        num_scales = len(in_channels)

        # ── UNet 구조 (unet.py와 동일) ──────────────────────────
        upscale_factors = [strides[i] // strides[i - 1] for i in range(1, len(strides))]
        outscale_factors = list(accumulate(upscale_factors, lambda x, y: x * y))

        self.upsamples = nn.ModuleList([
            nn.Upsample(scale_factor=s, mode='nearest') for s in upscale_factors
        ])
        self.inners = nn.ModuleList([
            nn.Conv2d(c, inner_channels, kernel_size=1, bias=bias) for c in in_channels
        ])
        self.outers = nn.ModuleList()
        for outscale in reversed(outscale_factors):
            self.outers.append(nn.Sequential(
                nn.Conv2d(inner_channels, output_channels, kernel_size=3, padding=1, bias=bias),
                nn.Upsample(scale_factor=outscale, mode='nearest')
            ))
        self.outers.append(
            nn.Conv2d(inner_channels, output_channels, kernel_size=3, padding=1, bias=bias)
        )
        # ──────────────────────────────────────────────────────────

        # ── ASF 추가 ──────────────────────────────────────────────
        self.asf = ASFModule(out_channels=output_channels, num_scales=num_scales)
        # ──────────────────────────────────────────────────────────

        self.upsamples.apply(self.weights_init)
        self.inners.apply(self.weights_init)
        self.outers.apply(self.weights_init)

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight.data)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1.)
            m.bias.data.fill_(1e-4)

    def forward(self, features):
        # UNet forward (unet.py와 동일)
        in_features = [inner(feat) for feat, inner in zip(features, self.inners)]

        up_features = []
        up = in_features[-1]
        for i in range(len(in_features) - 1, 0, -1):
            up = self.upsamples[i - 1](up) + in_features[i - 1]
            up_features.append(up)

        out_features = [self.outers[0](in_features[-1])]
        out_features += [outer(feat) for feat, outer in zip(up_features, self.outers[1:])]

        # ASF: out_features를 어텐션으로 재가중 후 반환
        # DBHead는 이 list를 받아 torch.cat → 형태 변화 없음
        out_features = self.asf(out_features)

        return out_features