# MAN++: Scaling Momentum Auxiliary Network

This repository contains the cleaned MAN++ release for the paper **MAN++: Scaling Momentum Auxiliary Network for Supervised Local Learning in Vision Tasks**, accepted by **IEEE TPAMI 2026**.

Paper links: [IEEE TPAMI](https://ieeexplore.ieee.org/abstract/document/11458619) | [arXiv](https://arxiv.org/pdf/2507.16279)

The public code is limited to the experiments below:

- CIFAR-10, SVHN, and STL-10: DGL+MAN++ and InfoPro+MAN++ on ResNet-32/110.
- ImageNet: InfoPro+MAN++ on ResNet-101, ResNet-152, and ResNeXt-101 32x8d with `K=2` or `K=4`.
- ImageNet ViT: MAN++ with `K=4` on ViT-Tiny, ViT-Small, and ViT-Base.
- COCO: RetinaNet + MAN++ with `K=4` on ResNet-50/101/152 backbones.

`MAN++` is exposed as `MANPP` in Python module and class names because `+` is not a valid Python identifier. Legacy `*_MAN` import aliases are kept for old checkpoints and commands, but new commands should use `*_MANPP`.

## Requirements

Install PyTorch, torchvision, numpy, tqdm, and dataset-specific packages:

```bash
pip install torch torchvision numpy tqdm scipy pillow pycocotools
```

For ImageNet and COCO, place datasets in the standard layouts used by torchvision:

```text
ImageNet/
  train/<class>/*.JPEG
  val/<class>/*.JPEG

COCO/
  train2017/*.jpg
  val2017/*.jpg
  annotations/instances_train2017.json
  annotations/instances_val2017.json
```

## CIFAR-10, SVHN, STL-10

InfoPro+MAN++:

```bash
cd Exp_on_CIFAR-SVHN-STL

CUDA_VISIBLE_DEVICES=0 python train.py \
  --dataset cifar10 --model resnet --layers 32 \
  --local_module_num 16 --arch resnetInfoPro_MANPP \
  --local_loss_mode cross_entropy --aux_net_widen 1 \
  --aux_net_feature_dim 128 --ixx_1 5 --ixy_1 0.5 \
  --ixx_2 0 --ixy_2 0 --momentum 0.995 --cos_lr
```

DGL+MAN++:

```bash
cd Exp_on_CIFAR-SVHN-STL

CUDA_VISIBLE_DEVICES=0 python train_DGL.py \
  --dataset svhn --model resnet --layers 110 \
  --local_module_num 55 --arch resnetDGL_MANPP \
  --aux_net_feature_dim 128 --momentum 0.995 --cos_lr
```

Supported local-module settings are ResNet-32 with `K=8/16` and ResNet-110 with `K=32/55`. `K=1` is retained for end-to-end sanity checks.

## ImageNet ResNet InfoPro

Single-node DDP example:

```bash
cd Exp_on_ImageNet

torchrun --nproc_per_node=8 imagenet_DDP.py /path/to/imagenet \
  --arch resnetInfoPro_MANPP_ddp --net resnet152 \
  --local_module_num 4 --batch-size 1024 --lr 0.4 \
  --epochs 90 --workers 24 --ixx_r 5 --ixy_r 0.75 \
  --momentum_MANPP 0.995 --dist-url env://
```

Supported `--net` values are `resnet101`, `resnet152`, and `resnext101_32x8d`; supported `--local_module_num` values are `2` and `4`.

## ImageNet ViT

MAN++ K=4:

```bash
cd Exp_on_ImageNet

torchrun --nproc_per_node=8 imagenet_DDP_vit.py /path/to/imagenet \
  --net vit_tiny --local_module_num 4 \
  --batch-size 1024 --lr 0.001 --epochs 100 --workers 16
```

BP baseline:

```bash
cd Exp_on_ImageNet

torchrun --nproc_per_node=8 imagenet_DDP_vit_bp.py /path/to/imagenet \
  --net vit_tiny \
  --batch-size 1024 --lr 0.001 --epochs 100 --workers 16
```

Supported `--net` values are `vit_tiny`, `vit_small`, and `vit_base`. This release supports ViT `K=4` only.

## COCO RetinaNet

```bash
cd Exp_on_COCO

torchrun --nproc_per_node=8 train_coco.py /path/to/coco \
  --backbone resnet50 --epochs 100 --batch-size 8 \
  --lr 4e-4 --momentum-manpp 0.995 --img-size 640
```

Supported COCO backbones are `resnet50`, `resnet101`, and `resnet152`. MAN++ uses `K=4` for all of them.

## Smoke Tests

These commands check syntax and the lightweight model paths without launching full training:

```bash
PYTHONPYCACHEPREFIX=/tmp/manpp_pycache python -m py_compile \
  $(find Exp_on_CIFAR-SVHN-STL Exp_on_ImageNet Exp_on_COCO \
  -path '*/data/*' -prune -o -name '*.py' -print | sort)

cd Exp_on_CIFAR-SVHN-STL
CUDA_VISIBLE_DEVICES=0 python train.py --dataset cifar10 --layers 32 \
  --local_module_num 16 --arch resnetInfoPro_MANPP \
  --local_loss_mode cross_entropy --aux_net_feature_dim 128 \
  --momentum 0.995 --dry-run-model

cd ../Exp_on_ImageNet
CUDA_VISIBLE_DEVICES= python imagenet_DDP.py --smoke-test
```

## Citation

```bibtex
@article{su2026man++,
  title={MAN++: Scaling Momentum Auxiliary Network for Supervised Local Learning in Vision Tasks},
  author={Su, Junhao and Zhu, Feiyu and Shi, Hengyu and Han, Tianyang and Qiu, Yurui and Luo, Junfeng and Wei, Xiaoming and Gao, Jialin},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2026},
  publisher={IEEE}
}
```
