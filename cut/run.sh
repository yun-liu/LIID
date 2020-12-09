#!/usr/bin/env bash

ROOT=/mnt/4Tvolume/wyh/LIID_release_public
cd $ROOT/cut/multiway_cut/edges
rm -rf edges_* ids_* res_*
cd $ROOT
CUDA_VISIBLE_DEVICES=2 python3 multiway_cut.py 0  $1
cd $ROOT/cut/multiway_cut/
./block $1 ./
echo start multi-way cut!
./multiway_cut
cd $ROOT
python3 multiway_cut.py 1 $1
