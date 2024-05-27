_base_ = [
    'mmdet::_base_/datasets/coco_detection.py',
    'mmdet::_base_/schedules/schedule_2x.py',
    'mmdet::_base_/default_runtime.py'
]
teacher_ckpt = 'resources/retinanet_pvtv2-b2_fpn_1x_coco_20210901_174843-529f0b9a.pth'  # noqa: E501

model = dict(
    _scope_='mmrazor',
    type='SingleTeacherDistill',
    architecture=dict(
        cfg_path='mmdet::retinanet/retinanet_r50_fpn_2x_coco_stvptb2.py'),
    teacher=dict(
        cfg_path='mmdet::pvt/retinanet_pvtv2-b2_fpn_1x_coco.py'),
    teacher_ckpt=teacher_ckpt,
    distiller=dict(
        type='ConfigurableDistiller',
        student_recorders=dict(
            fpn=dict(type='ModuleOutputs', source='neck'),
            pred_vfl=dict(type='MethodOutputs',
                          source='mmdet.models.dense_heads.retina_head.RetinaHead.get_vlf_distillation')
        ),
        teacher_recorders=dict(
            fpn=dict(type='ModuleOutputs', source='neck'),
            pred_vfl=dict(type='MethodOutputs',
                          source='mmdet.models.dense_heads.retina_head.RetinaHead.get_vlf_distillation')
        ),
        distill_losses=dict(
            loss_fourier_fpn0=dict(type='FourierLoss', loss_weight=7),
            loss_fourier_fpn1=dict(type='FourierLoss', loss_weight=7),
            loss_fourier_fpn2=dict(type='FourierLoss', loss_weight=7),
            loss_fourier_fpn3=dict(type='FourierLoss', loss_weight=7),
            loss_fourier_fpn4=dict(type='FourierLoss', loss_weight=7),
            loss_vfl=dict(type='VlfDisLoss', loss_weight=1.5)
        ),
        loss_forward_mappings=dict(
            loss_fourier_fpn0=dict(
                preds_S=dict(from_student=True, recorder='fpn', data_idx=0),
                preds_T=dict(from_student=False, recorder='fpn', data_idx=0),
            ),
            loss_fourier_fpn1=dict(
                preds_S=dict(from_student=True, recorder='fpn', data_idx=1),
                preds_T=dict(from_student=False, recorder='fpn', data_idx=1),
            ),
            loss_fourier_fpn2=dict(
                preds_S=dict(from_student=True, recorder='fpn', data_idx=2),
                preds_T=dict(from_student=False, recorder='fpn', data_idx=2),

            ),
            loss_fourier_fpn3=dict(
                preds_S=dict(from_student=True, recorder='fpn', data_idx=3),
                preds_T=dict(from_student=False, recorder='fpn',data_idx=3),
        ),
            loss_fourier_fpn4=dict(
                preds_S=dict(from_student=True, recorder='fpn', data_idx=4),
                preds_T=dict(from_student=False, recorder='fpn', data_idx=4),
            ),
            loss_vfl=dict(
                s_feature_list=dict(from_student=True, recorder='pred_vfl'),
                t_feature_list=dict(from_student=False, recorder='pred_vfl'))

        )
    ))

optim_wrapper = dict(optimizer=dict(lr=0.01),clip_grad=dict(type ='norm'))
find_unused_parameters = True
val_cfg = dict(_delete_=True, type='mmrazor.SingleTeacherDistillValLoop')