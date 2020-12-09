## LIID

[Leveraging Instance-, Image- and Dataset-Level Information for Weakly Supervised Instance Segmentation](http://mftp.mmcheng.net/Papers/21PAMI_InsImgDatasetWSIS.pdf)

This paper has been accepted and early accessed in IEEE TPAMI 2020.

### Introduction


Weakly supervised semantic instance segmentation with only image-level supervision, instead of relying on expensive pixel-wise masks or bounding box annotations, is an important problem to alleviate the data-hungry nature of deep learning. In this paper, we tackle this challenging problem by aggregating the image-level information of all training images into a large knowledge graph and exploiting semantic relationships from this graph. Specifically, our effort starts with some generic segment-based object proposals (SOP) without category priors. We propose a multiple instance learning (MIL) framework, which can be trained in an end-to-end manner using training images with image-level labels. For each proposal, this MIL framework can simultaneously compute probability distributions and category-aware semantic features, with which we can formulate a large undirected graph. The category of background is also included in this graph to remove the massive noisy object proposals. An optimal multi-way cut of this graph can thus assign a reliable category label to each proposal. The denoised SOP with assigned category labels can be viewed as pseudo instance segmentation of training images, which are used to train fully supervised models. The proposed approach achieves state-of-the-art performance for both weakly supervised instance segmentation and semantic segmentation. 


### Citations

If you are using the code/model/data provided here in a publication, please consider citing:

    @ARTICLE{21PAMI_InsImgDatasetWSIS,
    author={Yun Liu and Yu-Huan Wu and Peisong Wen and Yujun Shi and Yu Qiu and Ming-Ming Cheng},
    journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
    title={Leveraging Instance-, Image- and Dataset-Level Information for Weakly Supervised Instance Segmentation}, 
    year={2021},
    volume={},
    number={},
    pages={1-1},
    doi={10.1109/TPAMI.2020.3023152},
    }

### Requirements

* Python 3.5, PyTorch 0.4.1, Torchvision 0.2.2.post3, CUDA 9.0
* Validated on Ubuntu 16.04, NVIDIA TITAN Xp

### Testing LIID

1. Clone the LIID repository
    ```
    git clone https://github.com/yun-liu/LIID.git
    ```

2. [Download the pretrained model of MIL framework](https://drive.google.com/file/d/1KjoBn3ngzZw5aJPAiBaE3ZKh8xAn59kd/view?usp=sharing), and put them into `$ROOT_DIR` folder.

3. [Download Pascal VOC2012 dataset](http://host.robots.ox.ac.uk/pascal/VOC/). Extract the dataset files into `$VOC2012_ROOT` folder.

4. [Download the segment-based object proposals](https://drive.google.com/file/d/1qFIlbkc8S9ejmy1mKVGqEzj5m9FDs2wa/view?usp=sharing), and extract the data to `$VOC2012_ROOT/proposals/` folder.

5. [Download the compiled binary file](https://drive.google.com/file/d/1DMlSwQ1BuZWU2Kp2tUi4Wd5yRtycyEyF/view?usp=sharing), and put the binary files into `$ROOT_DIR/cut/multiway_cut/`.

6. Change the path of `cut/run.sh` to your own project root.

7. run `./make.sh` to build CUDA dependences.

8. Run `python3 gen_proposals.py`. Remember to change the `voc-root` to your own `$VOC2012_ROOT`. The proposals with labels will be generated in the `$ROOT_DIR/proposals` folder.

### Pretrained Models and data

The pretrained model of MIL framework can be downloaded [here](https://drive.google.com/file/d/1KjoBn3ngzZw5aJPAiBaE3ZKh8xAn59kd/view?usp=sharing).

Pascal VOC 2012 dataset files can be downloaded [here](http://host.robots.ox.ac.uk/pascal/VOC/) or other mirror websites.

S$^4$Net proposals used for testing can be downloaded [here](https://drive.google.com/file/d/1qFIlbkc8S9ejmy1mKVGqEzj5m9FDs2wa/view?usp=sharing).

MCG proposals can be downloaded [here](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/mcg/#datasets).

### Training with Pseudo Labels

For instance segmentation, you can use official or popular public Mask R-CNN projects like [mmdetecion](https://github.com/open-mmlab/mmdetection), [Detectron2](https://github.com/facebookresearch/detectron2), [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark), or other popular open-source projects.

For semantic segmentation, you can use official Caffe implementation of [deeplab](http://liangchiehchen.com/projects/DeepLab.html),  third party PyTorch implementation [here](https://github.com/kazuto1011/deeplab-pytorch), or third party Tensorflow Implementation [here](https://github.com/DrSleep/tensorflow-deeplab-resnet).

### Precomputed Results

Results of instance segmentation on Pascal VOC2012 *segmentation val* split can be downloaded [here](https://drive.google.com/file/d/10s5hVEknVgyWu1A63GO5gBA1Sb1wzynl/view?usp=sharing).

Results of semantic segmentation trained with 10K images, 10K images+24K simple imagenet images, 10K images (Res2Net-101) on Pascal VOC2012 *segmentation val* split can be downloaded [here](https://drive.google.com/file/d/1ysV06qPWnhaMKN7EHkaXzaxoyiukvApg/view?usp=sharing).

### Other Notes

Since IBM cplex is hard to install and the configuration of it is somewhat difficult. By default and for convenience, we provide the compiled binary file which can directly run. If you are desired to get the complete source code of solving the multi-way cut and ensures no commercial use of it, please contact Yu-Huan Wu (wuyuhuan (at) mail.nankai.edu(dot)cn).