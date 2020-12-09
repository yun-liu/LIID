#ifdef __cplusplus
extern "C" {
#endif

void iou_cuda_compute(const float* bbxes_gpu, float* iou_gpu, int n);

#ifdef __cplusplus
}
#endif
