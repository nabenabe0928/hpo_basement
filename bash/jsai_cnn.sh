#!/usr/bin/bash
#$-cwd
#$ -l h_rt=24:00:00

echo $(hostname)
source /etc/profile.d/modules.sh
module load singularity/2.6.1
module load cuda/9.2
export SINGULARITY_BINDPATH="/groups1/gaa50073/,/fs1/groups1/gaa50073/"
export CUDA_PATH=/usr/local/cuda-9.2
export LD_LIBRARY_PATH=$CUDA_PATH/lib64:$LD_LIBRARY_PATH

while getopts ":a:" o; do
    case "${o}" in
        a) a=${OPTARG};;
    esac
done

sing_exec="singularity exec --nv ubuntu16.04-cuda9.2-anaconda3.img "
py_script="python lhs_main.py "
cond="-fuc cnn -ini 1 -eva 100 -dat cifar -cls 100 -eexp jsai"

for num in `seq 0 3`; do
    $sing_exec$py_script$cond -cuda $num -exp $(($num+$a)) -res $rs &
    pids[${num}]=$!
done

for pid in ${pids[*]}; do
    wait $pid
done
