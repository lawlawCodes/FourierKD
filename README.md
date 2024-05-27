## Knowledge Distillation in Fourier Frequency Domain for Dense Prediction

### Introduction

This repository is the official implementation of: Knowledge Distillation in Fourier Frequency Domain for Dense Prediction.

[√]Publish configs and framework codes(based on MMRazor1.0.0)

[ ]Publish core codes (once our paper is accepted)



### Installation

1.Install MMRazor v1.0.0 (reference: [this](https://mmrazor.readthedocs.io/en/latest/get_started/installation.html).)

2.Install MMEngine(reference: [this](https://mmengine.readthedocs.io/en/latest/get_started/installation.html).)

3.Install MMDetection3.x (if you want to use KD on detection task, reference: [this](https://mmdetection.readthedocs.io/en/latest/get_started.html).)

4.Install MMSegmentationv1.0.0 (if you want to use KD on segmentation task, reference: [this](https://mmsegmentation.readthedocs.io/en/latest/get_started.html).)

Please install them from source e.g.:

```
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
pip install -v -e .
# "-v" means verbose, or more output
# "-e" means installing a project in editable mode,
# thus any local modifications made to the code will take effect without reinstallation.
```



### Train

```
python tools/train.py ${CONFIG_FILE} [optional arguments]
```

e.g.:

For Single GPU

```
python tools/train.py configs/distill/mmdet/fourier/fourier_fpn_reppoints_x101_reppoints_r50_2x_coco_fourier_vfl.py
```

For Multi GPUs

```
python -m torch.distributed.launch --nproc_per_node=4 tools/train.py configs/distill/mmdet/fourier/fourier_fpn_reppoints_x101_reppoints_r50_2x_coco_fourier_vfl.py --launcher pytorch
```



### Convert KD ckpt to student-only ckpt

If you want to use the trained checkpoint to isolate the parameters of the teacher network to further deploy the student network on edge devices, you can:

```
python tools/model_converters/convert_kd_ckpt_to_student.py ${checkpoint} --out-path ${out-path}
```