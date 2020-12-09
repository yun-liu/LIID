#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include "iou_kernel.h"
#define CUDA_CHECK(condition) \
  /* Code block avoids redefinition of cudaError_t error */ \
  do { \
    cudaError_t error = condition; \
    if (error != cudaSuccess) { \
      std::cout << cudaGetErrorString(error) << std::endl; \
    } \
} while (0)

#define CUDA_KERNEL_LOOP(i,n) for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i +=blockDim.x * gridDim.x)

const int CUDA_NUM_THREADS = 512;
inline int GET_BLOCKS(const int N){
    return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
}
__global__ void iou_kernel(const float* bbxes, float* iou, int n){
    CUDA_KERNEL_LOOP(idx, n*n){
        int i = idx / n, j = idx % n;
        if(i == j)iou[i * n + j] = 1.f;
        if(i > j)continue;
        float left = max(bbxes[4*i], bbxes[4*j]), right = min(bbxes[4*i + 2], bbxes[4*j + 2]);
        float top = max(bbxes[4*i + 1], bbxes[4*j + 1]), bottom = min(bbxes[4*i + 3], bbxes[4*j + 3]);
        float w = max(0.f, right - left + 1.f), h = max(0.f, bottom - top + 1.f);
        float Si = (bbxes[4*i + 2] - bbxes[4*i] + 1.f) * (bbxes[4*i + 3] - bbxes[4*i + 1] + 1.f);
        float Sj = (bbxes[4*j + 2] - bbxes[4*j] + 1.f) * (bbxes[4*j + 3] - bbxes[4*j + 1] + 1.f);
        iou[i * n + j] = iou[j * n + i] = w*h / (Si + Sj - w*h);
    }
}
void iou_cuda_compute(const float* bbxes_gpu, float* iou_gpu, int n){
    iou_kernel<<<GET_BLOCKS(n*n),CUDA_NUM_THREADS>>>
        (bbxes_gpu, iou_gpu, n);
}
