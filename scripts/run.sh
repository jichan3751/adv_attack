#!/bin/bash

set -e

if [ "$#" -ne 1 ]; then
    echo "Provide single arg"
    exit -1
fi

export CUDA_VISIBLE_DEVICES=$1
echo 'Using GPU' $CUDA_VISIBLE_DEVICES

source activate tensorflow_p36

# python -u src/trainer.py --fc-id $1
ipython  --pdb src/trainer.py -- --fc-id $1


#loop
declare -a arr=("1.0" "2.0" "3.0" "4.0")

# ## now loop through the above array
# for i in "${arr[@]}"
# do
#    echo "$i"
#    # python -u src/trainer.py $1 $i
#    # python -u src/trainer.py $1 $i | tee -i out_cnn_nat_eps_$i_rank$1_2.txt
#    # or do whatever with individual element of the array
# done

# python -u src/trainer.py $1 0.3 nat
# python -u src/trainer.py $1 0.3 adv
# ipython src/trainer.py $1 0.3 --pdb

# python -u src/trainer_linf.py $1 0.2 | tee -i out_linf_cnn_adv_eps_0.2_rank$1_2.txt



