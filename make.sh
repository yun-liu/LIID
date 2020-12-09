#!/usr/bin/env bash

xxxxx=$pwd
basepath=$(cd `dirname $0`; pwd)
echo $basepath
cd $basepath
cd libs/iou
./make.sh
cd $basepath
cd libs/nms
./make.sh
cd $basepath
cd libs/roi_pooling
./make.sh
cd $basepath
cd libs/distance
make
cd $xxxxx
