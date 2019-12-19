#!/bin/bash
fpath=models/tutorials/image/cifar10_estimator

python $fpath/cifar10_main.py --data-dir=$fpath/cifar-10-data --job-dir=/pnfs --num-gpus=0 --train-steps=9000 --sync
#--sync
