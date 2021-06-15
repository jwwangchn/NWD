## Installation

### Requirements

- Linux
- Python 3.7 (Python 2 is not supported)
- PyTorch **1.5** or higher
- CUDA 10.1 or higher
- NCCL 2
- GCC(G++) **5.4** or higher
- [mmcv-full](https://github.com/open-mmlab/mmcv)==**1.3.5**

We have tested the following versions of OS and softwares:

- OS:  Ubuntu 16.04
- GPU: TITAN X
- CUDA: 10.1
- GCC(G++): 5.4.1
- PyTorch: 1.5.0+cu101
- TorchVision: 0.6.0+cu101
- MMCV: 1.3.5
- MMDetection: 2.13.0

### Install

a. Create a conda virtual environment and activate it.

```shell
conda create -n nwd python=3.7 -y
conda activate nwd
```

b. Install PyTorch stable or nightly and torchvision following the [official instructions](https://pytorch.org/), e.g.,

```shell
pip install torch==1.5.0+cu101 torchvision==0.6.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
```

c. Clone the NWD repository.

```shell
git clone https://github.com/NWDMetric/NWD.git
cd NWD
```

d. Install MMCV-NWD

```
cd  ${NWD}/packages/mmcv
MMCV_WITH_OPS=1 pip install -e .  # package mmcv-full will be installed after this step
cd ../..
```

e. Install COCOAPI-AITOD for Evaluating on AI-TOD dataset
```
pip install "git+https://github.com/jwwangchn/cocoapi-aitod.git#subdirectory=aitodpycocotools"
```

f. Install MMDetection-NWD

```shell
# optional
pip install -r requirements.txt

python setup.py develop
# or "pip install -v -e ."
```

## Prepare datasets

Please refer to [AI-TOD](https://github.com/jwwangchn/AI-TOD) for AI-TOD dataset.

It is recommended to symlink the dataset root to `$MMDETECTION/data`.
If your folder structure is different, you may need to change the corresponding paths in config files (configs/_base_/datasets/aitod_detection.py).

```
mmdetection
├── mmdet
├── tools
├── configs
├── data
│   ├── AI-TOD
│   │   ├── annotations
│   │   │    │─── aitod_training_v1.json
│   │   │    │─── aitod_validation_v1.json
│   │   ├── trainval
│   │   │    │─── ***.png
│   │   │    │─── ***.png
│   │   ├── test
│   │   │    │─── ***.png
│   │   │    │─── ***.png
```