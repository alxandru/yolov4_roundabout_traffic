#/bin/bash

MAKEFILE=${HOME}/tutorial/darknet/Makefile

sed -i'' -e 's/OPENCV=0/OPENCV=1/g' $MAKEFILE
sed -i'' -e 's/GPU=0/GPU=1/g' $MAKEFILE
sed -i'' -e 's/CUDNN=0/CUDNN=1/g' $MAKEFILE
sed -i'' -e 's/CUDNN_HALF=0/CUDNN_HALF=1/g' $MAKEFILE

sed -i'$ a ARCH= -gencode arch=compute_53,code=sm_53 \
                 -gencode arch=compute_53,code=[sm_53,compute_53]' $MAKEFILE
