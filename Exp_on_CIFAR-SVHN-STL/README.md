# CIFAR/SVHN/STL MAN++

This directory contains the open-source CIFAR-10, SVHN, and STL-10 MAN++ code
paths for InfoPro and DGL training.

Use the MANPP Python module names in new commands:

```bash
python train.py --dataset cifar10 --layers 32 --local_module_num 16 --arch resnetInfoPro_MANPP
python train.py --dataset stl10 --layers 110 --local_module_num 55 --arch resnetInfoPro_MANPP
python train_DGL.py --dataset cifar10 --layers 32 --local_module_num 16 --arch resnetDGL_MANPP
```

To verify the model-construction path without loading or downloading datasets,
add `--dry-run-model` to either training command.

The old `resnetInfoPro_MAN` and `resnetDGL_MAN` import names are compatibility
aliases only. New code and documentation should use `resnetInfoPro_MANPP` and
`resnetDGL_MANPP`.

Paper MAN++ local-module settings:

```text
ResNet-32:  K=8, K=16
ResNet-110: K=32, K=55
```
