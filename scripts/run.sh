#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Provide single arg"
    exit -1
fi

export CUDA_VISIBLE_DEVICES=$1
echo 'Using GPU' $CUDA_VISIBLE_DEVICES

# python -u src/trainer.py $1 | tee -i out_fcn_eps_10_$1.txt

python -u src/trainer.py $1 5.0  | tee -i out_cnn_eps_5_$1.txt
python -u src/trainer.py $1 10.0 | tee -i out_cnn_eps_10_$1.txt
python -u src/trainer.py $1 15.0 | tee -i out_cnn_eps_15_$1.txt
python -u src/trainer.py $1 20.0 | tee -i out_cnn_eps_20_$1.txt
python -u src/trainer.py $1 25.0 | tee -i out_cnn_eps_25_$1.txt

# ipython src/trainer.py 3 --pdb
