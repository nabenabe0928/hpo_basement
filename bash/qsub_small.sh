#!/usr/bin/bash
#$-cwd
#$ -l h_rt=168:00:00

echo $(hostname)
source /etc/profile.d/modules.sh
module load singularity/2.6.1
module load cuda/9.2
export SINGULARITY_BINDPATH="/groups1/gaa50073/,/fs1/groups1/gaa50073/"
export CUDA_PATH=/usr/local/cuda-9.2
export LD_LIBRARY_PATH=$CUDA_PATH/lib64:$LD_LIBRARY_PATH

echo $USER:~$CWD$ CUDA_VISIBLE_DEVICES=0 python main.py -ini 10 -eva 150 -dat cifar -cls 100
echo "y" |  singularity exec --nv ubuntu16.04-cuda9.2-anaconda3.img python main.py -ini 10 -eva 150 -dat cifar -cls 100
