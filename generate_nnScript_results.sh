#! /bin/bash

declare -a hidden_nodes=('4' '8' '12' '16' '20' '50')

for l in `seq 0 5 60`; do
    for n in ${hidden_nodes[@]}; do
        python3 nnScript.py -f=results/nnScript_${n}_${l}.txt -n=${n} -l=${l}
    done
done
