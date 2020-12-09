cimport cython
import numpy as np
cimport numpy as np

assert sizeof(int) == sizeof(np.int32_t)

cdef extern from "cython_gpu_dist.hpp":
  void _dist(np.int32_t*, np.float32_t*, np.float32_t*, int, int, int, int, int)


def gpu_dist(np.ndarray[np.float32_t, ndim=2] feats, np.int32_t top=10, \
    np.int32_t batch=2, np.int32_t device_id=0):
  cdef int feat_num = feats.shape[0]
  cdef int feat_dim = feats.shape[1]
  cdef np.ndarray[np.int32_t, ndim=2] \
        node = np.zeros((feat_num, top), dtype=np.int32)
  cdef np.ndarray[np.float32_t, ndim=2] \
        dist = np.zeros((feat_num, top), dtype=np.float32)
  _dist(&node[0, 0], &dist[0, 0], &feats[0, 0], feat_num, feat_dim, top, batch, \
      device_id)
  return node, dist
