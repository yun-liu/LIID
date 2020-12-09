#include <THC/THC.h>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "iou_kernel.h"

extern THCState *state;

int iou_cuda(THCudaTensor *bbxes, THCudaTensor *iou){
    iou_cuda_compute(THCudaTensor_data(state, bbxes),
                     THCudaTensor_data(state, iou),
                     THCudaTensor_size(state, bbxes, 0));
    return 1;
}
