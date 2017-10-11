#!/usr/bin/env sh
LOG=./snap/log-data+%Y-%m-%d-%H-%S.log
caffe train --solver=/data/lixinpeng/myDenseNet/densenet_solver.prototxt --gpu=0,4,6,8,9 2>&1 | tee $LOG