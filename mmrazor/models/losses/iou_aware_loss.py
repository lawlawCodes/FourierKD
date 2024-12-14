# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from mmdet.models import weight_reduce_loss
from mmdet.registry import MODELS
from mmdet.structures.bbox import bbox_overlaps
from mmrazor.registry import MODELS
from mmdet.models.losses.varifocal_loss import VarifocalLoss
from mmdet.models.task_modules.coders.delta_xywh_bbox_coder import DeltaXYWHBBoxCoder


def varifocal_loss(pred: Tensor,
                   target: Tensor,
                   weight: Optional[Tensor] = None,
                   alpha: float = 0.75,
                   gamma: float = 2.0,
                   iou_weighted: bool = True,
                   reduction: str = 'mean',
                   avg_factor: Optional[int] = None) -> Tensor:
    """`Varifocal Loss <https://arxiv.org/abs/2008.13367>`_

    Args:
        pred (Tensor): The prediction with shape (N, C), C is the
            number of classes.
        target (Tensor): The learning target of the iou-aware
            classification score with shape (N, C), C is the number of classes.
        weight (Tensor, optional): The weight of loss for each
            prediction. Defaults to None.
        alpha (float, optional): A balance factor for the negative part of
            Varifocal Loss, which is different from the alpha of Focal Loss.
            Defaults to 0.75.
        gamma (float, optional): The gamma for calculating the modulating
            factor. Defaults to 2.0.
        iou_weighted (bool, optional): Whether to weight the loss of the
            positive example with the iou target. Defaults to True.
        reduction (str, optional): The method used to reduce the loss into
            a scalar. Defaults to 'mean'. Options are "none", "mean" and
            "sum".
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.

    Returns:
        Tensor: Loss tensor.
    """
    # pred and target should be of the same size
    assert pred.size() == target.size()
    pred_sigmoid = pred.sigmoid()
    target = target.type_as(pred)
    if iou_weighted:
        focal_weight = target * (target > 0.0).float() + \
            alpha * (pred_sigmoid - target).abs().pow(gamma) * \
            (target <= 0.0).float()
    else:
        focal_weight = (target > 0.0).float() + \
            alpha * (pred_sigmoid - target).abs().pow(gamma) * \
            (target <= 0.0).float()
    loss = F.binary_cross_entropy_with_logits(
        pred, target, reduction='none') * focal_weight
    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss

@MODELS.register_module()
class VarifocalLoss(nn.Module):

    def __init__(self,
                 use_sigmoid: bool = True,
                 alpha: float = 0.75,
                 gamma: float = 2.0,
                 iou_weighted: bool = True,
                 reduction: str = 'mean',
                 loss_weight: float = 1.0) -> None:
        """`Varifocal Loss <https://arxiv.org/abs/2008.13367>`_

        Args:
            use_sigmoid (bool, optional): Whether the prediction is
                used for sigmoid or softmax. Defaults to True.
            alpha (float, optional): A balance factor for the negative part of
                Varifocal Loss, which is different from the alpha of Focal
                Loss. Defaults to 0.75.
            gamma (float, optional): The gamma for calculating the modulating
                factor. Defaults to 2.0.
            iou_weighted (bool, optional): Whether to weight the loss of the
                positive examples with the iou target. Defaults to True.
            reduction (str, optional): The method used to reduce the loss into
                a scalar. Defaults to 'mean'. Options are "none", "mean" and
                "sum".
            loss_weight (float, optional): Weight of loss. Defaults to 1.0.
        """
        super().__init__()
        assert use_sigmoid is True, \
            'Only sigmoid varifocal loss supported now.'
        assert alpha >= 0.0
        self.use_sigmoid = use_sigmoid
        self.alpha = alpha
        self.gamma = gamma
        self.iou_weighted = iou_weighted
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred: Tensor,
                target: Tensor,
                weight: Optional[Tensor] = None,
                avg_factor: Optional[int] = None,
                reduction_override: Optional[str] = None) -> Tensor:
        """Forward function.

        Args:
            pred (Tensor): The prediction with shape (N, C), C is the
                number of classes.
            target (Tensor): The learning target of the iou-aware
                classification score with shape (N, C), C is
                the number of classes.
            weight (Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Options are "none", "mean" and "sum".

        Returns:
            Tensor: The calculated loss
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if self.use_sigmoid:
            loss_cls = self.loss_weight * varifocal_loss(
                pred,
                target,
                weight,
                alpha=self.alpha,
                gamma=self.gamma,
                iou_weighted=self.iou_weighted,
                reduction=reduction,
                avg_factor=avg_factor)
        else:
            raise NotImplementedError
        return loss_cls


@MODELS.register_module()
class IoUAwareLoss(nn.Module):
    def __init__(
        self,
        loss_weight: float = 1.0,
        cls_out_channels: int = 80,
        num_points : int = 9,
        method : str = 'AnchorBased'
    ) -> None:
        super().__init__()
        self.loss_weight = loss_weight
        self.cls_out_channels = cls_out_channels
        self.method = method
        if self.method == 'AnchorBased':
            self.bbox_coder = DeltaXYWHBBoxCoder()
        elif self.method == 'AnchorFree':
            self.num_points = num_points
            self.points2bbox = MODELS.build(dict(_scope_='mmdet', type='RepPointsHead',
                                                 num_classes=cls_out_channels,
                                                 in_channels=cls_out_channels)).points2bbox
        self.vlfloss = VarifocalLoss()


    def bbox_process(self, cls_score, bbox_pred, tea_cls_target, tea_bbox_target, labels, label_weights, all_anchor):
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)
        tea_cls_target = tea_cls_target.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels).contiguous()
        cls_score = cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels).contiguous()

        if self.method == 'AnchorBased':
            anchors = all_anchor.reshape(-1, 4)
            bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, self.bbox_coder.encode_size)
            tea_bbox_target = tea_bbox_target.permute(0, 2, 3, 1).reshape(-1, self.bbox_coder.encode_size)
            bbox_pred_decoded = self.bbox_coder.decode(anchors, bbox_pred.detach())
            tea_bbox_targets_decoded = self.bbox_coder.decode(anchors, tea_bbox_target.detach())

        elif self.method =='AnchorFree':
        #Currently, only RepPoints is supported
            bbox_pred_decoded = self.points2bbox(
                bbox_pred.reshape(-1, 2 * self.num_points), y_first=False)
            tea_bbox_targets_decoded = self.points2bbox(
                tea_bbox_target.reshape(-1, 2 * self.num_points), y_first=False)

        ious = bbox_overlaps(
            bbox_pred_decoded.detach(),
            tea_bbox_targets_decoded.detach(),
            is_aligned=True)

        pos_inds = ((labels >= 0)
                    & (labels < self.cls_out_channels)).nonzero().reshape(-1)
        pos_ious = ious[pos_inds]
        cls_iou_targets = torch.zeros_like(cls_score)
        cls_iou_targets[pos_inds, :] = pos_ious.unsqueeze(1)
        cls_iou_targets = cls_iou_targets * torch.sigmoid(tea_cls_target)
        return cls_score, cls_iou_targets, label_weights.unsqueeze(1)


    def forward(
        self,
        s_feature_list: torch.Tensor,
        t_feature_list: torch.Tensor,
    ) -> torch.Tensor:

        stu_cls_scores, stu_bbox_preds, labels_list, label_weights_list, avg_factor = s_feature_list[0:5]
        tea_cls_scores, tea_bbox_preds = t_feature_list[0:2]
        loss = 0.
        all_anachor_list = [0] * len(stu_cls_scores)

        if self.method == "AnchorBased":
            all_anachor_list = s_feature_list[5]

        for i in range(0, len(stu_cls_scores)):
            cls_score, cls_iou_targets, label_weight = self.bbox_process(stu_cls_scores[i], stu_bbox_preds[i], tea_cls_scores[i],
                                                                         tea_bbox_preds[i], labels_list[i], label_weights_list[i], all_anachor_list[i])
            loss += self.vlfloss(cls_score, cls_iou_targets, label_weight, avg_factor = avg_factor)
        return self.loss_weight * loss

