#! /bin/bash
USER=$(whoami)
CWD=$(dirname $0)

while getopts ":p:r:a:i:R:b:e:" o; do
    case "${o}" in
        r) restart=${OPTARG};;
        a) a=${OPTARG};;
        i) nini=${OPTARG};;
        b) barrier=${OPTARG};;
        e) addex=${OPTARG};;
    esac
done

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
