from __future__ import absolute_import
import torch
import numpy as np
from ._ext import iou
import pdb


def iou_gpu(bbxes):
	iou_out = bbxes.new(bbxes.size(0), bbxes.size(0)).zero_()
	iou.iou_cuda(bbxes, iou_out)
	return iou_out
