#! /bin/bash
USER=$(whoami)
CWD=$(dirname $0)

echo $USER:~$CWD$ singularity exec --nv ubuntu16.04-cuda9.2-anaconda3.img byobu
singularity exec --nv ubuntu16.04-cuda9.2-anaconda3.img byobu
