#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Provide single arg"
    exit -1
fi

export CUDA_VISIBLE_DEVICES=$1
echo 'Using GPU' $CUDA_VISIBLE_DEVICES

# python -u src/trainer.py $1 | tee -i out_fcn_eps_10_$1.txt

python -u src/trainer.py $1 5.0  | tee -i out_cnn_eps_5_rank$1.txt
# ipython src/trainer.py 3 --pdb


#loop
# declare -a arr=("1.0" "2.0" "3.0" "4.0")

# ## now loop through the above array
# for i in "${arr[@]}"
# do
#    # echo "$i"
#    python -u src/trainer.py $1 $i | tee -i out_cnn_eps_$i_rank$1.txt
#    # or do whatever with individual element of the array
# done

