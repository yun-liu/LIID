#include "cython_gpu_dist.hpp"
#include <iostream>

#define CUDA_CHECK(condition) \
  /* Code block avoids redefinition of cudaError_t error */ \
  do { \
    cudaError_t error = condition; \
    if (error != cudaSuccess) { \
      std::cout << cudaGetErrorString(error) << std::endl; \
    } \
  } while (0)

#define DIVUP(m,n) ((m) / (n) + ((m) % (n) > 0))
int const threadsPerBlock = sizeof(unsigned long long) * 8;


__device__ inline float _consine(float const * const a, float const * const b,
    const int ndim) {
  float dot = 0; float mod1 = 0; float mod2 = 0;
  for (int i = 0; i < ndim; i++) {
    dot += a[i] * b[i];
    mod1 += a[i] * a[i];
    mod2 += b[i] * b[i];
  }
  return dot / (sqrt(mod1) * sqrt(mod2)) + 1;
}


__global__ void _dist_kernel(float* consin_dev, const float* feat_dev,
    const int feat_num, const int batch, const int batch_total, const int ndim) {
  const int row_start = blockIdx.y;
  const int col_start = blockIdx.x;
  const int global_row_start = feat_num / batch_total * batch;
  const int global_row_end = (batch + 1 == batch_total) ? feat_num : (feat_num / batch_total * (batch + 1));
  const int global_row_size = global_row_end - global_row_start;

  const int row_size =
        min(global_row_size - row_start * threadsPerBlock, threadsPerBlock);
  const int col_size =
        min(feat_num - col_start * threadsPerBlock, threadsPerBlock);

  if (threadIdx.x < row_size) {
    const int cur_base_idx = threadsPerBlock * row_start + threadIdx.x;
    const int cur_feat_idx = global_row_start + cur_base_idx;
    const float* cur_feat = feat_dev + cur_feat_idx * ndim;
    for (int i = 0; i < col_size; i++) {
      const float* comp_feat = feat_dev + (threadsPerBlock * col_start + i) * ndim;
      const int cur_dist_idx = cur_base_idx*feat_num + threadsPerBlock * col_start + i;
      consin_dev[cur_dist_idx] = _consine(cur_feat, comp_feat, ndim);
    }
  }
}


__global__ void _find_max_dist(float* dist_dev, int* node_dev,
    float* consine_dev, const int feat_num, const int batch,
    const int batch_total, const int top) {
  const int global_row_start = feat_num / batch_total * batch;
  const int global_row_end = (batch + 1 == batch_total) ? feat_num : (feat_num / batch_total * (batch + 1));
  const int global_row_size = global_row_end - global_row_start;

  const int row_start = blockIdx.x;
  const int row_size =
        min(global_row_size - row_start * threadsPerBlock, threadsPerBlock);

  if (threadIdx.x < row_size) {
    const int cur_base_idx = threadsPerBlock * row_start + threadIdx.x;
    const int cur_feat_idx = global_row_start + cur_base_idx;
    float* cur_consine = consine_dev + cur_base_idx*feat_num;
    for (int i = 0; i < top; i++) {
      float max_cons = -1;
      int max_idx = -1;
      for (int j = 0; j < feat_num; j++) {
        if (cur_consine[j] > max_cons) {
          max_cons = cur_consine[j];
          max_idx = j;
        }
      }
      dist_dev[cur_feat_idx*top + i] = max_cons;
      node_dev[cur_feat_idx*top + i] = max_idx;
      cur_consine[max_idx] = -100;
    }
  }
}


void _set_device(int device_id) {
  int current_device;
  CUDA_CHECK(cudaGetDevice(&current_device));
  if (current_device == device_id) {
    return;
  }
  // The call to cudaSetDevice must come before any calls to Get, which
  // may perform initialization using the GPU.
  CUDA_CHECK(cudaSetDevice(device_id));
}


void _dist(int* node_host, float* dist_host, float* feat_host, int feat_num,
    int ndim, int top, int batch, int device_id) {
  _set_device(device_id);

  float* feat_dev = NULL;
  float* dist_dev = NULL;
  int* node_dev = NULL;
  float* consine_dev = NULL;

  CUDA_CHECK(cudaMalloc(&feat_dev,
                        feat_num * ndim * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(feat_dev,
                        feat_host,
                        feat_num * ndim * sizeof(float),
                        cudaMemcpyHostToDevice));

  CUDA_CHECK(cudaMalloc(&dist_dev,
                        feat_num * top * sizeof(float)));
  CUDA_CHECK(cudaMemset(dist_dev, 0, feat_num * top * sizeof(float)));

  CUDA_CHECK(cudaMalloc(&node_dev,
                        feat_num * top * sizeof(int)));
  CUDA_CHECK(cudaMemset(node_dev, 0, feat_num * top * sizeof(float)));

  const int batch_size = feat_num / batch;
  const int max_batch = (batch_size + feat_num % batch);
  CUDA_CHECK(cudaMalloc(&consine_dev,
                        feat_num * max_batch * sizeof(float)));

  for (int i = 0; i < batch; i++) {
    const int row_size = (i + 1 < batch) ? batch_size : max_batch;
    dim3 blocks(DIVUP(feat_num, threadsPerBlock),
                DIVUP(row_size, threadsPerBlock));
    dim3 threads(threadsPerBlock);
    _dist_kernel<<<blocks, threads>>>(consine_dev, feat_dev, feat_num, i,
        batch, ndim);

    dim3 blocks2(DIVUP(row_size, threadsPerBlock));
    _find_max_dist<<<blocks2, threads>>>(dist_dev, node_dev, consine_dev,
        feat_num, i, batch, top);
  }

  CUDA_CHECK(cudaMemcpy(&node_host[0], node_dev, feat_num * top * sizeof(int), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(&dist_host[0], dist_dev, feat_num * top * sizeof(float), cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(feat_dev));
  CUDA_CHECK(cudaFree(dist_dev));
  CUDA_CHECK(cudaFree(node_dev));
  CUDA_CHECK(cudaFree(consine_dev));
}
