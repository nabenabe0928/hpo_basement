#! /bin/bash
USER=$(whoami)
CWD=$(dirname $0)

echo $USER:~$CWD$ qsub -g gaa50073 -l rt_G.small=1 bash/qsub_small.sh
qsub -g gaa50073 -l rt_G.small=1 bash/qsub_small.sh
