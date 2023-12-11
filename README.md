## resnet-simple

[![Python](https://img.shields.io/pypi/pyversions/resnet-simple.svg?style=plastic)](https://badge.fury.io/py/mcts-simple) [![Version](https://img.shields.io/pypi/v/resnet-simple.svg?logo=pypi)](https://badge.fury.io/py/mcts-simple) [![GitHub license](https://img.shields.io/github/license/denselance/resnet-simple.svg)](https://github.com/DenseLance/mcts-simple/blob/main/LICENSE) [![PyPI downloads](https://img.shields.io/pypi/dm/resnet-simple.svg)](https://pypistats.org/packages/resnet-simple) [![Open All Collab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DenseLance/resnet-simple/blob/main/examples/ResNet50_(CIFAR10,_Single_Label_Classification).ipynb)

*resnet-simple* is a Python3 library that provides a well-documented and easy to use implementation of ResNet (and ResNetv1.5), together with its most basic use case of image classification.

### ResNet

Residual Network, more commonly known as ResNet, is a deep learning model that is most commonly used for image recogition and classification. It was first introduced by He et al. in the paper titled "Deep Residual Learning for Image Recognition". Each ResNet has multiple residual blocks that takes an input $x$ and pass it through stacked non-linear layers which are fit to the residual mapping $F(x) := H(x) - x$, where $H(x)$ is the original unreferenced mapping which can be recast back via $H(x) = F(x) + x$. Intuitively speaking, ResNets should achieve higher accuracy than models that purely use convolutional networks (CNNs) such as VGGNet and AlexNet, as technically it should be easier to optimize and learn residual mapping than an unreferenced one.

This module allows you to choose between the original ResNet (v1) and the <a src = "https://catalog.ngc.nvidia.com/orgs/nvidia/resources/resnet_50_v1_5_for_pytorch">modified ResNet (v1.5)</a>, in which the latter enjoys a slight increase in accuracy at the expense of a slower performance. To do so, set `downsample_3x3 = True` under the `BottleNeck`/`ResNet` class to use ResNetv1.5, and `downsample_3x3 = False` for ResNetv1. This feature only works for ResNets that use BottleNeck blocks such as ResNet50, ResNet101 and ResNet152.

### How to use resnet-simple

*resnet-simple* only supports python 3.7 and above. If you want to use *resnet-simple* for other versions of Python, do note that it may not function as intended.

Module is mainly tested on Google Colab (GPU) and Windows (CPU). Issues when setting up in other environments, or when using multiple GPU cores, are still relatively unknown.

#### Dependencies

The following libraries are compulsory in order to use *resnet-simple*, which is automatically installed together with it unless otherwise specified:

* tqdm
* torch
* safetensors

When pip installing *resnet-simple*, the following libraries will also be installed as they are used in our examples:

* matplotlib
* torchvision
* tensorboard
* scikit-learn (for Python 3.7, use version 1.0.2)

#### User installation

In command prompt on Windows,

```cmd
pip install resnet-simple
```

In your python file,

```python
from resnet_simple import *
```

### Contributions

I appreciate if you are able to contribute to this project as I am the only contributor for now. If you think that something is wrong with my code, you can open an issue and I'll try my best to resolve it!

### Citation

The code from this module is re-implemented based on what was described in the following paper.

```tex
@inproceedings{inproceedings,
author = {He, Kaiming and Zhang, Xiangyu and Ren, Shaoqing and Sun, Jian},
year = {2016},
month = {06},
pages = {770-778},
title = {Deep Residual Learning for Image Recognition},
doi = {10.1109/CVPR.2016.90}
}@misc{he2015deep,
      title={Deep Residual Learning for Image Recognition}, 
      author={Kaiming He and Xiangyu Zhang and Shaoqing Ren and Jian Sun},
      year={2015},
      eprint={1512.03385},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

