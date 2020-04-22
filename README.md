# efficientnet_tf2pytorch
## Installation

### Requirements

- Linux OS: Ubuntu 16.04
- Python 3.6.10
- PyTorch 1.2.0
- Tensorflow 1.15.0
- Torchvision 0.4.0

### Install efficientnet_tf2pytorch

Clone the efficientnet_tf2pytorch repository.
```shell
git clone https://github.com/ChenYingpeng/efficientnet_tf2pytorch
cd efficientnet_tf2pytorch
```

### Load tf pretrained weights
look `pretrained_tf/` dir.

### Convert tf to pytorch
```shell
python convert_params_tf2pytorch.py --model_name efficient-b0 --tf_checkpoint pretrained_tf/efficient-b0/ --output_file pretrained_pytorch/efficient-b0.pth
```

### Test pytorh model
```shell
python test.py --model_name efficient-b0 --checkpoint pretrained_pytorch/efficient-b0.pth
```

### References
Appreciate the great work from the following repositories:
- [official/efficientnet](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet)
- [google/automl](https://github.com/google/automl)
- [lukemelas/EfficientNet-PyTorch](https://github.com/lukemelas/EfficientNet-PyTorch)
- [zylo117/Yet-Another-EfficientDet-Pytorch](https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch)
