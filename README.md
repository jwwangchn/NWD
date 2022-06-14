# A Normalized Gaussian Wasserstein Distance for Tiny Object Detection

This is the official code for the [NWD](https://arxiv.org/abs/2110.13389). The expanded method is accepted by the [ISPRS J P & RS](https://www.sciencedirect.com/science/article/pii/S0924271622001599?dgcid=author) in 2022.

## Installation

### Requirements

- Linux
- Python 3.7 (Python 2 is not supported)
- PyTorch **1.5** or higher
- CUDA 10.1 or higher
- NCCL 2
- GCC(G++) **5.4** or higher
- [mmcv-nwd](git@github.com:jwwangchn/mmcv-nwd.git)==**1.3.5**
- [cocoapi-aitod](https://github.com/jwwangchn/cocoapi-aitod)==**12.0.3**

We have tested the following versions of OS and softwares:

- OS:  Ubuntu 16.04
- GPU: TITAN X
- CUDA: 10.1
- GCC(G++): 5.5.0
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

c. Install MMCV-NWD

```
git clone https://github.com/jwwangchn/mmcv-nwd.git
cd mmcv-nwd
MMCV_WITH_OPS=1 pip install -e .  # package mmcv-full will be installed after this step
cd ../
```

d. Install COCOAPI-AITOD for Evaluating on AI-TOD dataset
```
pip install "git+https://github.com/jwwangchn/cocoapi-aitod.git#subdirectory=aitodpycocotools"
```

e. Install NWD

```shell
git clone https://github.com/jwwangchn/NWD.git
# optional
pip install -r requirements.txt

python setup.py develop
# or "pip install -v -e ."
```

## Prepare datasets

Please refer to [AI-TOD](https://github.com/jwwangchn/AI-TOD) for AI-TOD dataset.

It is recommended to symlink the dataset root to `$NWD/data`.
If your folder structure is different, you may need to change the corresponding paths in config files (configs/_base_/datasets/aitod_detection.py).

```
NWD
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

## Run

The NWD's config files are in [configs/nwd](https://github.com/jwwangchn/NWD/tree/main/configs/nwd).

Please see MMDetection full tutorials [with existing dataset](docs/1_exist_data_model.md) for beginners.

### Training on a single GPU

The basic usage is as follows (e.g. train Faster R-CNN with NWD). Note that the `lr=0.01` in config file needs to be `lr=0.01/4` for training on single GPU.

```shell
python tools/train.py configs/nwd/faster_rcnn_r50_aitod_rpn_nwd.py
```

### Training on multiple GPUs

The basic usage is as follows (e.g. train Faster R-CNN with NWD).

```shell
bash ./tools/dist_train.sh configs/nwd/faster_rcnn_r50_aitod_rpn_nwd.py 4
```

## Inference


## Benchmark

The benchmark and trained models will be publicly available soon.

## Citation
```
@article{NWD_2021_arXiv,
  title={A Normalized Gaussian Wasserstein Distance for Tiny Object Detection},
  author={Wang, Jinwang and Xu, Chang and Yang, Wen and Yu, Lei},
  journal={arXiv preprint arXiv:2110.13389},
  year={2021}
}
```
```
@article{NWD_RKA_2022_ISPRSJ,
  title={Detecting Tiny Objects in Aerial Images: A Normalized Wasserstein Distance and A New Benchmark},
  author={Xu, Chang and Wang, Jinwang and and Yang, Wen and Yu, Huai and Yu, Lei and Xia, Gui-Song},
  journal={ISPRS Journal of Photogrammetry and Remote Sensing (ISPRS J P & RS)},
  year={2022}
}
