#! /bin/bash
USER=$(whoami)
CWD=$(dirname $0)

echo $USER:~$CWD$ CUDA_VISIBLE_DEVICES=0 python main.py -ini 10 -eva 150 -dat cifar -cls 100
CUDA_VISIBLE_DEVICES=0 python main.py -ini 10 -eva 150 -dat cifar -cls 100
