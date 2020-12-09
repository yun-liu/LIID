import numpy as np
import torch, torchvision
from torch.autograd import Variable
import os, sys, cv2, json
import copy
from PIL import Image
import xml.etree.ElementTree as ET
import scipy.io as sio
import pickle
from time import time
from copy import deepcopy
from tqdm import tqdm

class VOCDataset(torch.utils.data.Dataset):
    def __init__(self, devkit_path, resized=448):
        
        self.devkit_path = devkit_path
        self.img_path = os.path.join(devkit_path, 'JPEGImages/')
        self.proposal_path = os.path.join(devkit_path, 'proposals/')
        self.img_extension = ".jpg"
        self.num_classes = 20 + 1
        self.classes = ['__background__',  # always index 0
                        'aeroplane', 'bicycle', 'bird', 'boat',
                        'bottle', 'bus', 'car', 'cat', 'chair',
                        'cow', 'diningtable', 'dog', 'horse',
                        'motorbike', 'person', 'pottedplant',
                        'sheep', 'sofa', 'train', 'tvmonitor']
        self.resized = resized
        self.normalize = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])
        self.img_list = open("img_list/trainval_minus_segval.txt", 'r').read().splitlines()
        self.num_imgs = len(self.img_list)

    @staticmethod
    def extract_bbxes(mask_png=None, ):
        num_masks = mask_png.max()
        boxes = np.zeros([num_masks, 4], dtype=np.float32)
        for i in range(num_masks):
            horizontal_indicies = np.where(np.any(mask_png == i + 1, axis=0))[0]
            vertical_indicies = np.where(np.any(mask_png == i + 1, axis=1))[0]
            if horizontal_indicies.shape[0]:
                x1, x2 = horizontal_indicies[[0, -1]]
                y1, y2 = vertical_indicies[[0, -1]]
                x2 += 1
                y2 += 1
            else:
                x1, x2, y1, y2 = 0, 15, 0, 15
            boxes[i] = np.array([x1, y1, x2, y2])
        return boxes

    def bbxes(self, idx):
        try:
            with open(self.proposal_path + self.img_list[idx] + '.pkl', 'rb') as f:
                res = pickle.load(f)
                return res['bbxes']
        except FileNotFoundError:
            res = cv2.imread(self.proposal_path + self.img_list[idx] + '.png', cv2.IMREAD_GRAYSCALE)
            return self.extract_bbxes(res)

    def masks(self, idx):
        try:
            with open(self.proposal_path + self.img_list[idx] + '.pkl', 'rb') as f:
                res = pickle.load(f)
                return res['segs']
        except FileNotFoundError:
            masks = cv2.imread(self.proposal_path + self.img_list[idx] + '.png', cv2.IMREAD_GRAYSCALE)
            num_masks = masks.max()
            masks_ = np.zeros((num_masks, masks.shape[0], masks.shape[1]), dtype=bool)
            for i in range(num_masks):
                masks_[i, :, :] = masks == i+1
            return masks_

    def load_gt_label(self, index):
        filename = os.path.join(self.devkit_path, 'Annotations', self.img_list[index] + '.xml')
        tree = ET.parse(filename)

        objs = tree.findall('object')
        num_objs = len(objs)

        gt_classes = np.zeros(self.num_classes - 1, dtype=np.float32)

        class_to_index = dict(zip(self.classes, range(self.num_classes)))
        for idx, obj in enumerate(objs):
            cls = class_to_index[obj.find('name').text.lower().strip()]
            gt_classes[cls - 1] = 1.
        gt_classes = np.concatenate((np.array([1.]), gt_classes))
        return gt_classes

    def read_img(self, idx):
        return cv2.imread(os.path.join(self.img_path, self.img_list[idx] + self.img_extension))
    
    def load_proposals(self, idxes):
        all_bbxes = np.array([])
        flag = 0
        scale_size = float(self.resized)
        idxes = [idxes]
        for i, idx in enumerate(idxes):
            bbxes = np.array(self.bbxes(idx))
            img = self.read_img(idxes[0])
            scale_x = scale_size / img.shape[1]
            scale_y = scale_size / img.shape[0]
            if bbxes.shape[0] == 0:
                bbxes = np.array([i, 0, 0, img.shape[1] * scale_x, img.shape[0] * scale_y], dtype=np.float32).reshape(1, -1)
                if flag == 0:
                    all_bbxes = bbxes
                else:
                    all_bbxes = np.concatenate((all_bbxes, bbxes), axis=0)
            else:
                bbxes_idx = np.zeros((bbxes.shape[0], 1), dtype=np.int32)
                bbxes_idx[:, 0] = i
                bbxes[:, 0] *= scale_x
                bbxes[:, 1] *= scale_y
                bbxes[:, 2] *= scale_x
                bbxes[:, 3] *= scale_y
                bbxes = np.concatenate((bbxes_idx, bbxes[:, :4]), axis=1)
                if flag == 0:
                    all_bbxes = bbxes
                else:
                    all_bbxes = np.concatenate((all_bbxes, bbxes), axis=0)
            flag = 1
        all_bbxes = Variable(torch.from_numpy(all_bbxes).float())
        return all_bbxes

    def gen_proposals(self, new_labels, save='new_proposals/'):
        os.makedirs(save, exist_ok=True)
        new_labels = pickle.load(open(new_labels, 'rb'))
        old_anno_idx = 0
        for idx in tqdm(range(len(self.img_list))):
            img_name = self.img_list[idx]
            gt_classes = self.load_gt_label(idx)
            masks = self.masks(idx)
            if len(masks) == 0:
                old_anno_idx += 1
            class_ids = []
            for temp in masks:
                class_id = new_labels[old_anno_idx]
                if gt_classes[class_id] == 0:
                    class_id = 0
                class_ids.append(class_id)
                old_anno_idx += 1
            with open(save + img_name + '.pkl', 'wb') as f:
                res = {'segs': masks, 'labels': class_ids}
                pickle.dump(res, f)

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.img_path, self.img_list[idx] + self.img_extension))
        img_preprocess = torchvision.transforms.Compose(
            [torchvision.transforms.Resize([self.resized, self.resized]),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                                [0.229, 0.224, 0.225])])
        img_ = img_preprocess(copy.deepcopy(img))
        bbxes = self.load_proposals(idx).float()
        gt_label = torch.from_numpy(self.load_gt_label(idx))
        return img_, bbxes,gt_label

    def read_img_size(self, idx):
        img = cv2.imread(os.path.join(self.img_path, self.img_list[idx] + self.img_extension))
        return img.shape[:2]

    def __len__(self):
        return self.num_imgs
