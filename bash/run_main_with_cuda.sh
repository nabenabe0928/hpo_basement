#! /bin/bash
USER=$(whoami)
CWD=$(dirname $0)

echo $USER:~$CWD$ singularity exec --nv ubuntu16.04-cuda9.2-anaconda3.img byobu
singularity exec --nv ubuntu16.04-cuda9.2-anaconda3.img byobu

echo $USER:~$CWD$ CUDA_VISIBLE_DEVICES=0 python main.py -ini 10 -eva 150 -dat cifar -cls 100
CUDA_VISIBLE_DEVICES=0 python main.py -ini 10 -eva 150 -dat cifar -cls 100