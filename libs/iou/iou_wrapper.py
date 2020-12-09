import torch
# from core.config import cfg
# from model.nms.nms_gpu import nms_gpu
from iou.iou_gpu import iou_gpu

def iou(bbxes, force_cpu=False):
    if bbxes.shape[0] == 0:
        return []
    # ---pytorch version---
    return iou_gpu(bbxes)
