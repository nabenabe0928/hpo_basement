#! /bin/bash
USER=$(whoami)
CWD=$(dirname $0)

a=0

for i in `seq 0 99`; do
    file="bash/jsai_mlp.sh"
    echo $USER:~$CWD$ qsub -g gcb50329 -l rt_F=1 $file -a $a
    qsub -g gcb50329 -l rt_F=1 $file -a $a
    a=`echo $(($a+4))`
done

a=0

for i in `seq 0 99`; do
    file="bash/jsai_cnn.sh"
    echo $USER:~$CWD$ qsub -g gcb50329 -l rt_F=1 $file -a $a
    qsub -g gcb50329 -l rt_F=1 $file -a $a
    a=`echo $(($a+4))`
done
