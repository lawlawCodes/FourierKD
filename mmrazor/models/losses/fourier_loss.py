# Copyright (c) OpenMMLab. All rights reserved.
from typing import Tuple, Union
import torch
import torch.nn as nn
from mmrazor.registry import MODELS
import torch.fft
from mmengine.logging import MessageHub
import torch.nn.functional as F
import numpy as np


class ATFM(nn.Module):
    def __init__(self, in_channels = 256):
        super().__init__()
        self.sigma_tensor = nn.Parameter(torch.zeros(in_channels, dtype=torch.float32).cuda(), requires_grad=True)
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(
                in_channels, in_channels , kernel_size=1))
        self.conv1x1_b = nn.Sequential(
            nn.Conv2d(
                in_channels, in_channels , kernel_size=1))

        self.message_hub = MessageHub.get_current_instance()


    def fourier_feature(self,x):
        x = torch.fft.fftshift(torch.fft.fft2(x, dim=(2, 3), norm='ortho'))
        return torch.abs(x), torch.angle(x)


    def fusion_adaptive_newmap(self, stu_feature_abs, tea_feature_abs):
        n1 = self.conv1x1(stu_feature_abs)
        n2 = self.conv1x1_b(tea_feature_abs)
        N, C, H, W = stu_feature_abs.shape
        param_expanded = self.sigma_tensor.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand(N, C, H, W)
        alpha = torch.sigmoid(param_expanded)
        fusion_map = alpha * n1 + (1 - alpha) * n2
        self.message_hub.update_scalar(f'train/{"sigma"}', float(torch.mean(alpha)))
        return fusion_map


    def forward(self, stu_feature, tea_feature):
        stu_feature_abs, stu_feature_angle = self.fourier_feature(stu_feature)
        tea_feature_abs, tea_feature_angle = self.fourier_feature(tea_feature)
        N,C,H,W = stu_feature_abs.shape
        stu_feature_abs = self.fusion_adaptive_newmap(stu_feature_abs, tea_feature_abs)
        stu_feature_abs = stu_feature_abs * torch.rand(N,C,H,W, device = stu_feature_abs.device)
        stack_complex_tensor_stu = stu_feature_abs * np.e ** (1j * stu_feature_angle)
        stack_complex_tensor_tea = tea_feature_abs * np.e ** (1j * tea_feature_angle)
        return stack_complex_tensor_stu, stack_complex_tensor_tea


@MODELS.register_module()
class FourierLoss(nn.Module):
    def __init__(self,
                 in_channels=256,
                 loss_weight=1.0,
                 resize_stu=True,
                 ) -> None:
        super(FourierLoss,self).__init__()
        self.loss_weight = loss_weight
        self.resize_stu = resize_stu
        self.in_channels = in_channels
        self.atfm = ATFM(in_channels=in_channels)
        self.predictor = nn.Conv2d(self.in_channels,
                                 self.in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0,
                                 )


    def norm(self, feat: torch.Tensor) -> torch.Tensor:
        N, C, H, W = feat.shape
        batch_norm = nn.BatchNorm2d(C, affine=False).cuda()
        return batch_norm(feat)


    def ToRiemannCoo(self, tenReal, tenImag):
        deno = (tenReal * tenReal + tenImag * tenImag + 1).clamp(min = 1e-9)
        xRie = (2 * tenReal) / deno
        yRie = (2 * tenImag) / deno
        zRie = (tenReal * tenReal + tenImag * tenImag - 1) / deno
        return [xRie, yRie, zRie]


    def forward(self, preds_S: Union[torch.Tensor, Tuple],
                preds_T: Union[torch.Tensor, Tuple]) -> torch.Tensor:
        """Forward computation.

        Args:
            preds_S (torch.Tensor | Tuple[torch.Tensor]): The student model
                prediction. If tuple, it should be several tensors with shape
                (N, C, H, W).
            preds_T (torch.Tensor | Tuple[torch.Tensor]): The teacher model
                prediction. If tuple, it should be several tensors with shape
                (N, C, H, W).

        Return:
            torch.Tensor: The calculated loss value.
        """
        if isinstance(preds_S, torch.Tensor):
            preds_S, preds_T = (preds_S, ), (preds_T, )


        for pred_S, pred_T in zip(preds_S, preds_T):
            size_S, size_T = pred_S.shape[2:], pred_T.shape[2:]
            if size_S[0] != size_T[0]:
                if self.resize_stu:
                    pred_S = F.interpolate(pred_S, size_T, mode='bilinear')
                else:
                    pred_T = F.interpolate(pred_T, size_S, mode='bilinear')
            assert pred_S.shape == pred_T.shape

        loss_frequency = 0.
        N, C, H, W = pred_S.shape
        similarity = (1 - F.cosine_similarity(pred_S.view(N, C, -1), pred_T.view(N, C, -1), dim=2))

        pred_S = self.predictor(pred_S)
        freq_stu, freq_tea = self.atfm(pred_S, pred_T)

        riemannStu = self.ToRiemannCoo(freq_stu.real, freq_stu.imag)
        riemannTea = self.ToRiemannCoo(freq_tea.real, freq_tea.imag)
        loss_frequency += torch.mean(
            F.mse_loss(self.norm(riemannStu[0]), self.norm(riemannTea[0]), reduce=False) * similarity.view(N, C, 1, 1)) / 2
        loss_frequency += torch.mean(
            F.mse_loss(self.norm(riemannStu[1]), self.norm(riemannTea[1]), reduce=False) * similarity.view(N, C, 1, 1)) / 2
        loss_frequency += torch.mean(
            F.mse_loss(self.norm(riemannStu[2]), self.norm(riemannTea[2]), reduce=False) * similarity.view(N, C, 1, 1)) / 2
        return loss_frequency * self.loss_weight