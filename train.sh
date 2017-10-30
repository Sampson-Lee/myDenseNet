#!/usr/bin/env sh
LOG=./snap/log-`date +%Y-%m-%d-%H:%M:%S`.log
caffe train --solver=/data/lixinpeng/myDenseNet/densenet_solver.prototxt --gpu=0,4,6,8,9 2>&1 | tee $LOG