#ifndef WASSERSTEIN_NMS_CUDA_KERNEL_CUH
#define WASSERSTEIN_NMS_CUDA_KERNEL_CUH

#include <float.h>
#ifdef MMCV_WITH_TRT
#include "common_cuda_helper.hpp"
#else  // MMCV_WITH_TRT
#ifdef MMCV_USE_PARROTS
#include "parrots_cuda_helper.hpp"
#else  // MMCV_USE_PARROTS
#include "pytorch_cuda_helper.hpp"
#endif  // MMCV_USE_PARROTS
#endif  // MMCV_WITH_TRT

#define DIVUP(m, n) ((m) / (n) + ((m) % (n) > 0))
int const threadsPerBlock = sizeof(unsigned long long int) * 8;

__device__ inline bool devWassersteinDistance(float const *const a, float const *const b,
                              const int offset, const float threshold, const float constant) {
  float a_center_x = (a[0] + a[2]) * 0.5, a_center_y = (a[1] + a[3]) * 0.5;
  float b_center_x = (b[0] + b[2]) * 0.5, b_center_y = (b[1] + b[3]) * 0.5;
  float center_distance = (b_center_x - a_center_x) * (b_center_x - a_center_x) + (b_center_y - a_center_y) * (b_center_y - a_center_y);

  float w1 = a[2] - a[0] + offset, h1 = a[3] - a[1] + offset;
  float w2 = b[2] - b[0] + offset, h2 = b[3] - b[1] + offset;
  float wh_distance = ((w1 - w2) * (w1 - w2) + (h1 - h2) * (h1 - h2)) * 0.25;
  
  float wassersteins = sqrt(center_distance + wh_distance);
  float normalized_wasserstein = expf(-wassersteins/constant);


  return normalized_wasserstein > threshold;
}

__global__ void wasserstein_nms_cuda(const int n_boxes, const float iou_threshold,
                         const int offset, const float constant, const float *dev_boxes,
                         unsigned long long *dev_mask) {
  const int row_start = blockIdx.y;
  const int col_start = blockIdx.x;
  const int tid = threadIdx.x;

  if (row_start > col_start) return;

  const int row_size =
      fminf(n_boxes - row_start * threadsPerBlock, threadsPerBlock);
  const int col_size =
      fminf(n_boxes - col_start * threadsPerBlock, threadsPerBlock);

  __shared__ float block_boxes[threadsPerBlock * 4];
  if (tid < col_size) {
    block_boxes[tid * 4 + 0] =
        dev_boxes[(threadsPerBlock * col_start + tid) * 4 + 0];
    block_boxes[tid * 4 + 1] =
        dev_boxes[(threadsPerBlock * col_start + tid) * 4 + 1];
    block_boxes[tid * 4 + 2] =
        dev_boxes[(threadsPerBlock * col_start + tid) * 4 + 2];
    block_boxes[tid * 4 + 3] =
        dev_boxes[(threadsPerBlock * col_start + tid) * 4 + 3];
  }
  __syncthreads();

  if (tid < row_size) {
    const int cur_box_idx = threadsPerBlock * row_start + tid;
    const float *cur_box = dev_boxes + cur_box_idx * 4;
    int i = 0;
    unsigned long long int t = 0;
    int start = 0;
    if (row_start == col_start) {
      start = tid + 1;
    }
    for (i = start; i < col_size; i++) {
      if (devWassersteinDistance(cur_box, block_boxes + i * 4, offset, iou_threshold, constant)) {
        t |= 1ULL << i;
      }
    }
    dev_mask[cur_box_idx * gridDim.y + col_start] = t;
  }
}
#endif  // WASSERSTEIN_NMS_CUDA_KERNEL_CUH
