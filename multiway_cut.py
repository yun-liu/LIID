import scipy.io as sio
import numpy as np
from numpy.linalg import norm
import pickle as pkl
import os
from libs.distance.gpu_dist import gpu_dist
import sys

is_train = 0
is_test = int(sys.argv[1]);
bias = 0
n_rois = total_rois = int(sys.argv[2])
threshold = 0.9
k = 3
cut_path = 'cut/'
save_edge_path = cut_path + 'multiway_cut/edges/'
        
def get_res():
    res_path = save_edge_path + '/'
    i = 0
    all_labels = np.zeros((total_rois))
    while True:
        try:
            with open(res_path + 'ids_' + str(i)) as f:
                 ids = list(map(int, f.read().splitlines()))
            with open(res_path + 'res_' + str(i)) as f:
                 labels = list(map(int, f.read().splitlines()))
        except:
            break
        assert len(ids) == len(labels)
        for j in range(len(ids)):
            all_labels[ids[j]] = labels[j]
        i += 1
    return all_labels.astype(np.int).tolist()

def get_dist(vecs, gt_labels):
    print(len(vecs))
    node, dist = gpu_dist(vecs[:, 1:].astype(np.float32), k, 2, device_id=0)
    dist = dist - dist.min()
    all_edges = []
    print('gpu_dist done')
    n_rois = len(vecs)
    print("get_dist len(vecs): ", len(vecs))
    for i in range(n_rois):
        for j in range(k):
            v = node[i, j]
            # if v == i:continue
            if v <= i:
                continue
            if np.sum(gt_labels[int(vecs[i, 0]), 1:]*gt_labels[int(vecs[v, 0]), 1:]) == 0:
                continue
            if len(np.where(node[v] == i)) != 0 and dist[i,j] > 1e-4:
                all_edges.append([i, v, dist[i, j]])
    return all_edges

def main():
    res = sio.loadmat('feats_labels.mat')
    vecs = res['vecs']
    scores = res['scores']
    gt_labels = res['gts']
    n_rois = total_rois = len(vecs)

    assert len(vecs) == len(scores)

    if is_test:
        all_labels = get_res()
        with open('labels.pkl', 'wb') as f:
            pkl.dump(all_labels, f, 2)
    else:
        all_edges = get_dist(vecs, gt_labels)
        scores = scores - scores.min()
        for i in range(n_rois):
            weights = np.zeros((n_rois))
            keep = np.argsort(-scores[i, 1:])[:4]
            if scores[i, 1 + keep[0]] >= threshold:
                all_edges.append([i, n_rois + keep[0], scores[i, 1 + keep[0]]])
            else:
                for j in keep:
                    if scores[i, 1 + j] > 1e-4:
                        all_edges.append([i, n_rois + j, scores[i, 1 + j]])
        if not os.path.exists(save_edge_path):
            os.makedirs(save_edge_path)
        print(save_edge_path)
        with open(save_edge_path + '/all_edges', 'w') as f:
            for e in all_edges:
                f.write('%d %d %f\n'%(int(e[0]), int(e[1]), e[2]))

if __name__ == '__main__':
    main()
