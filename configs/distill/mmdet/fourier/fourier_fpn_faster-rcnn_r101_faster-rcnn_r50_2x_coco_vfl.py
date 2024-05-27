_base_ = [
    'mmdet::_base_/datasets/coco_detection.py',
    'mmdet::_base_/schedules/schedule_2x.py',
    'mmdet::_base_/default_runtime.py'
]

teacher_ckpt = 'resources/faster_rcnn_r101_fpn_2x_coco_bbox_mAP-0.398_20200504_210455-1d2dac9c.pth'  # noqa: E501

model = dict(
    _scope_='mmrazor',
    type='SingleTeacherDistill',
    architecture=dict(
        cfg_path='mmdet::faster_rcnn/faster-rcnn_r50_fpn_2x_coco_st.py',
        pretrained=False),
    teacher=dict(
        cfg_path='mmdet::faster_rcnn/faster-rcnn_r101_fpn_2x_coco.py',
        pretrained=False),
    teacher_ckpt=teacher_ckpt,
    distiller=dict(
        type='ConfigurableDistiller',
        student_recorders=dict(fpn=dict(type='ModuleOutputs', source='neck'),
                               vfl_info=dict(type='MethodOutputs',
                                             source='mmdet.models.dense_heads.anchor_head.AnchorHead.get_vlf_distillation')),
        teacher_recorders=dict(fpn=dict(type='ModuleOutputs', source='neck'),
                               vfl_info=dict(type='MethodOutputs',
                                             source='mmdet.models.dense_heads.anchor_head.AnchorHead.get_vlf_distillation')),
        distill_losses=dict(
            loss_fourier_fpn0=dict(type='FourierLoss', loss_weight=7),
            loss_fourier_fpn1=dict(type='FourierLoss', loss_weight=7),
            loss_fourier_fpn2=dict(type='FourierLoss', loss_weight=7),
            loss_fourier_fpn3=dict(type='FourierLoss', loss_weight=7),
            loss_fourier_fpn4=dict(type='FourierLoss', loss_weight=7),
            loss_vfl=dict(type='VlfFasterLoss', loss_weight=1.5)
        ),
        loss_forward_mappings=dict(
            loss_fourier_fpn0=dict(
                preds_S=dict(from_student=True, recorder='fpn', data_idx=0),
                preds_T=dict(from_student=False, recorder='fpn', data_idx=0)),
            loss_fourier_fpn1=dict(
                preds_S=dict(from_student=True, recorder='fpn', data_idx=1),
                preds_T=dict(from_student=False, recorder='fpn', data_idx=1)),
            loss_fourier_fpn2=dict(
                preds_S=dict(from_student=True, recorder='fpn', data_idx=2),
                preds_T=dict(from_student=False, recorder='fpn', data_idx=2)),
            loss_fourier_fpn3=dict(
                preds_S=dict(from_student=True, recorder='fpn', data_idx=3),
                preds_T=dict(from_student=False, recorder='fpn',
                             data_idx=3)),
            loss_fourier_fpn4=dict(
                preds_S=dict(from_student=True, recorder='fpn', data_idx=4),
                preds_T=dict(from_student=False, recorder='fpn',
                             data_idx=4)),
            loss_vfl=dict(
                s_bbox_list=dict(from_student=True, recorder='vfl_info'),
                t_bbox_list=dict(from_student=False, recorder='vfl_info'),
            )
        )))

find_unused_parameters = True

val_cfg = dict(_delete_=True, type='mmrazor.SingleTeacherDistillValLoop')
optim_wrapper = dict(optimizer=dict(lr=0.02),clip_grad=dict(type ='norm'))