We need
- pytorch
- cuda
- ninja

# CUDA 9.0

You can install [CUDA9.0](https://developer.nvidia.com/cuda-90-download-archive), [cuDNN](https://developer.nvidia.com/cudnn), [NCCL](https://developer.nvidia.com/nccl) to local directory. 

First install CUDA by running the downloaded executable.

Then download and extract cuDNN.

```
cudnn-9.0-linux-x64-v7.4.2.24
|-- include
|   `-- cudnn.h
|-- lib64
|   |-- libcudnn.so -> libcudnn.so.7
|   |-- libcudnn.so.7 -> libcudnn.so.7.4.2
|   |-- libcudnn.so.7.4.2
|   `-- libcudnn_static.a
`-- NVIDIA_SLA_cuDNN_Support.txt
```

Move cuDNN files to CUDA directory by
```bash
cp cudnn-9.0-linux-x64-v7.4.2.24/include/* cuda-9.0/include/
cp cudnn-9.0-linux-x64-v7.4.2.24/lib64/* cuda-9.0/lib64/
```

Then download and extract NCCL.

```
nccl_2.3.7-1+cuda9.0_x86_64
|-- include
|   `-- nccl.h
|-- lib
|   |-- libnccl.so -> libnccl.so.2
|   |-- libnccl.so.2 -> libnccl.so.2.3.7
|   |-- libnccl.so.2.3.7
|   `-- libnccl_static.a
`-- LICENSE.txt
```

Move NCCL files to CUDA directory by
```bash
cp nccl_2.3.7-1+cuda9.0_x86_64/include/* cuda-9.0/include/
cp nccl_2.3.7-1+cuda9.0_x86_64/lib/* cuda-9.0/lib64/
```

# Pytorch 1.0

It's required to install pytorch from source.

Setup your python environment before installing pytorch. Anaconda is required in the following example. Change `WORKING_DIR`, `CUDA_HOME` to your paths and run the following commands.

```bash
WORKING_DIR=<your-directory-to-save-intermediate-results-of-installing-pytorch>
TORCH_DIR_NAME=pytorch_v1.0.0
mkdir -p ${WORKING_DIR}

pip uninstall -y torch

# Install basic dependencies
conda install --yes numpy pyyaml mkl mkl-include setuptools cmake cffi typing
conda install --yes -c mingfeima mkldnn
# Add LAPACK support for the GPU
conda install --yes -c pytorch magma-cuda90

cd ${WORKING_DIR}
rm -rf ${TORCH_DIR_NAME}
# Tested with commit db5d313
git clone --recursive --single-branch --branch v1.0.0  https://github.com/pytorch/pytorch.git ${TORCH_DIR_NAME}
cd ${TORCH_DIR_NAME}

rm -rf build
rm -rf torch.egg-info
export CUDA_HOME=<your-cuda-directory>
export USE_SYSTEM_NCCL=1
export NCCL_LIB_DIR=${CUDA_HOME}/lib64  # For CUDA version > 8.0, you have to download NCCL lib independently
export NCCL_INCLUDE_DIR=${CUDA_HOME}/include
export CMAKE_PREFIX_PATH="$(dirname $(which conda))/../" # [anaconda root directory]

python setup.py install 2>&1 | tee ${WORKING_DIR}/install_pytorch.log
cd ${WORKING_DIR}
python test_data_parallel.py 2>&1 | tee test_pytorch_data_parallel.log
```

The contents of `test_data_parallel.py` in above commands is

```python
import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel

model = nn.Linear(10, 20).cuda()
x = torch.ones(100, 10).float().cuda()
model_w = DataParallel(model, device_ids=[0,1,2,3])
x = model_w(x)
# x = model(x)
print(x.size())  # It should be (100, 20)
```

# ninja

Download and extract ninja 1.8.2 from <https://github.com/ninja-build/ninja/releases/download/v1.8.2/ninja-linux.zip>. Add ninja to environment variable `PATH`.

# Environment Variables

After installing pytorch, cuda and ninja, modify and add following lines to your `.bashrc` file.

```bash
export anaconda_home=<your-anaconda-directory>
export PATH=${anaconda_home}/bin:${PATH}
export LD_LIBRARY_PATH=${anaconda_home}/lib:${LD_LIBRARY_PATH}

export CUDA_HOME=<your-cuda-directory>
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

export PATH=<your-ninja-directory>:${PATH}
```
