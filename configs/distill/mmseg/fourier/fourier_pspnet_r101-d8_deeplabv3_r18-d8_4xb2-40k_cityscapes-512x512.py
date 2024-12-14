_base_ = [
    'mmseg::_base_/datasets/cityscapes_512x512.py',
    'mmseg::_base_/schedules/schedule_40k.py',
    'mmseg::_base_/default_runtime.py'
]

teacher_ckpt = 'resources/pspnet_r101-d8_512x1024_40k_cityscapes_20200604_232751-467e7cf4.pth'  # noqa: E501
teacher_cfg_path = 'mmseg::pspnet/pspnet_r101-d8_4xb2-40k_cityscapes-512x1024.py'  # noqa: E501
student_cfg_path = 'mmseg::deeplabv3/deeplabv3_r18-d8_4xb2-40k_cityscapes-512x512.py'  # noqa: E501
model = dict(
    _scope_='mmrazor',
    type='SingleTeacherDistill',
    architecture=dict(cfg_path=student_cfg_path, pretrained=False),
    teacher=dict(cfg_path=teacher_cfg_path, pretrained=False),
    teacher_ckpt=teacher_ckpt,
    distiller=dict(
        type='ConfigurableDistiller',
        connectors=dict(
            channels_connector=dict(
                type='ConvModuleConnector',
                in_channel=512,
                out_channel=2048),
        ),
        distill_losses=dict(
            loss_fourier=dict(type='FourierLoss', loss_weight=41, in_channels=2048),
        ),
        student_recorders=dict(
            backbone_feature=dict(type='ModuleOutputs', source='backbone.layer4.1')),
        teacher_recorders=dict(
            backbone_feature=dict(type='ModuleOutputs', source='backbone.layer4.2')),
        loss_forward_mappings=dict(
            loss_fourier=dict(
                preds_S=dict(from_student=True, recorder='backbone_feature', connector = 'channels_connector'),
                preds_T=dict(from_student=False, recorder='backbone_feature'))
        )
    )
)
optim_wrapper = dict(optimizer=dict(lr=0.02),clip_grad=dict(type ='norm'))
find_unused_parameters = True

val_cfg = dict(_delete_=True, type='mmrazor.SingleTeacherDistillValLoop')
