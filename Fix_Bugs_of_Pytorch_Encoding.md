## Pytorch-Encoding

After installing prerequisites [as this instruction](Install_Prerequisite_for_Pytorch_Encoding.md), we clone Pytorch-Encoding
```bash
# Tested with commit ce461da
git clone https://github.com/zhanghang1989/PyTorch-Encoding.git
```

Then, according to [this issue](https://github.com/zhanghang1989/PyTorch-Encoding/issues/161), replace `#include <torch/extension.h>` with `#include <torch/serialize/tensor.h>` in all `encoding/lib/*/*.cpp` and `encoding/lib/*/*.cu` files. Also add `#include <torch/serialize/tensor.h>` to `encoding/lib/gpu/operator.h`.

The Pytorch-Encoding doc requires `python setup.py install`. However, it's not necessary. You can just add the package path to `$PYTHONPATH` or `sys.path`.

Run the examples
```bash
export CUDA_HOME=/mnt/data-1/data/houjing.huang/Software/cuda-9.0
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

python scripts/prepare_pcontext.py
python test.py --dataset PContext --model-zoo Encnet_ResNet50_PContext --eval
```

## Misc

- The [default ResNet](https://github.com/zhanghang1989/PyTorch-Encoding/blob/ce461dae8d088253dcd9818d2999d4049bce3493/encoding/models/resnet.py) in Pytorch-Encoding is different than pytorch [torchvision](https://github.com/pytorch/vision/blob/98ca260bc834ec94a8143e4b5cfe9516b0b951a2/torchvision/models/resnet.py). The former provides `deep_base`, `dilation`, `multi_grid` options.