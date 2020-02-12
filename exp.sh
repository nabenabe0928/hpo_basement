#!/bin/bash

cond=" -dim 10 -ini 10 -eva 500 -che 0 -fre 100 -exp "
tpe="python tpe_main.py"$cond
mvtpe="python mvtpe_main.py"$cond
lhs="python lhs_main.py"$cond
nm="python nm_main.py"$cond
cma="python cma_main.py"$cond
gp="python gp_main.py"$cond
f=" -fuc "

for fuc in griewank k_tablet michalewicz schwefel sphere styblinski weighted_sphere; do
    echo $fuc
    for num in `seq 0 9`; do
        # echo $tpe$num$f$fuc -re 1
        # $tpe$num$f$fuc -re 1
        # echo $mvtpe$num$f$fuc -re 1
        # $mvtpe$num$f$fuc -re 1
        # echo $lhs$num$f$fuc -re 1
        # $lhs$num$f$fuc -re 1
        # echo $nm$num$f$fuc -re 1
        # $nm$num$f$fuc -re 1
        # echo $cma$num$f$fuc -re 1
        # $cma$num$f$fuc -bar 0 -re 1
        echo $gp$num$f$fuc
        $gp$num$f$fuc
    done
done
