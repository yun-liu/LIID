import torch
import torch.nn as nn
import torch.nn.init as init
import matplotlib.pyplot as plt
import os, json, pickle
import argparse
import copy, torch, torchvision
from resnet_roi import resnet50
from PIL import Image
from tqdm import tqdm
import numpy as np
import scipy.io as sio
from copy import deepcopy
from voc import VOCDataset



def inference(dataloader, data, model, gtcls_filter=True, start_idx=0):
    softmax = nn.Softmax(dim=1)
    all_scores = []
    all_vecs = []
    all_gt_classes = []
    for idx, (img, bbxes, classes) in enumerate(tqdm(dataloader)):
        model = model.cuda()
        img = img.cuda()
        bbxes = bbxes.cuda().squeeze(dim=0)
        classes = classes[0].cuda()
        outputs = model(img, bbxes)
        vec = outputs[1].data.cpu().numpy()
        outputs = softmax(outputs[0])
        idx_vec = np.zeros((len(vec), 1)) + idx + start_idx
        idx_vec = idx_vec.astype(np.float32)
        vec = np.concatenate((idx_vec, vec), axis=1)
        if gtcls_filter:
            outputs = outputs * classes.float().unsqueeze(0)
        scores = outputs.data.cpu().numpy()
        gt_classes = classes.data.cpu().unsqueeze(0).numpy()
        scores = np.concatenate((idx_vec, scores), axis=1)
        all_scores.append(scores)
        all_vecs.append(vec)
        all_gt_classes.append(gt_classes)
    all_scores = np.concatenate(all_scores, axis=0)
    all_vecs = np.concatenate(all_vecs, axis=0)
    all_gt_classes = np.concatenate(all_gt_classes, axis=0)

    return all_scores, all_vecs, all_gt_classes

def do_cut(mcut_path="./cut/", data=None, n_rois=None, save=None):
    instance_rois_pwd = os.getcwd()
    multiway_cut_pwd = mcut_path
    os.chdir(multiway_cut_pwd)
    print("doing the multiway cut!!!!")
    print('running sh run.sh {}'.format(n_rois))
    os.system('sh run.sh {}'.format(n_rois))
    os.chdir(instance_rois_pwd)
    print("multiway cut complete. start to generate labeled proposals to the following path:\n{}".format(save))

    data.gen_proposals(new_labels='labels.pkl', save=save)

def main():
    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--weights', type=str,
                        default='R50-pami.pth')
    parser.add_argument('--resized', type=int, default='448')  # adjust the input img's size
    parser.add_argument('--voc-root', type=str, default='/mnt/4Tvolume/wyh/dataset/VOCdevkit/VOC2012/')
    parser.add_argument('--mat-save', type=str, default='feats_labels.mat')

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    num_classes = 21
    print("loading pretrained model: {}!".format(args.weights))
    model = resnet50(pretrained=False, num_classes=num_classes)
    model.load_state_dict(torch.load(args.weights), strict=False)
    print("loading complete")

    dataset = VOCDataset(args.voc_root, args.resized)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1,     
          shuffle=False, num_workers=8, pin_memory=True)
    print("generating scores and labels")
    scores, vecs, gts = inference(data_loader, dataset, model, gtcls_filter=True, start_idx=0) # pred
    
    sio.savemat(args.mat_save, {'vecs': vecs, 'scores': scores, 'gts': gts})
    num_instances = len(vecs)
    print("generate scores and labels completed")

    do_cut(n_rois=num_instances, data=dataset, save='proposals/')

if __name__ == '__main__':
    main()

