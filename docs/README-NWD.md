# A Normalized Gaussian Wasserstein Distance for Tiny Object Detection

The repo is based on [mmdetection](https://github.com/open-mmlab/mmdetection).


### Introduction
Detecting tiny objects is a very challenging problem since a tiny object only contains a few pixels in size. We demonstrate that state-of-the-art detectors do not produce satisfactory results on tiny objects due to the lack of appearance information. Our key observation is that Intersection over Union (IoU) based metrics such as IoU itself and its extensions are very sensitive to the location deviation of the tiny objects, and drastically deteriorate the detection performance when used in anchor-based detectors. To alleviate this, we propose a new evaluation metric using Wasserstein distance for tiny object detection. Specifically, we first model the bounding boxes as 2D Gaussian distributions and then propose a new metric dubbed Normalized Wasserstein Distance (NWD) to compute the similarity between them by their corresponding Gaussian distributions. The proposed NWD metric can be easily embedded into the assignment, non-maximum suppression, and loss function of any anchor-based detector to replace the commonly used IoU metric. We evaluate our metric on a new dataset for tiny object detection (AI-TOD) in which the average object size is much smaller than existing object detection datasets. Extensive experiments show that, when equipped with NWD metric, our approach yields performance that is 6.7 AP points higher than a standard fine-tuning baseline, and 6.0 AP points higher than state-of-the-art competitors.


## Installation

Please refer to [install.md](INSTALL.md) for installation and dataset preparation.


## Getting Started

Please see [getting_started.md](get_started.md) for the basic usage of MMDetection.