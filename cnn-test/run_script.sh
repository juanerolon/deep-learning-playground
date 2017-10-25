#!/bin/bash

##activate desired conda environment
source activate tflow_gpu_opt

##cd into directory containing the python script you wish to run
cd /mnt/linuxdata2/Dropbox/_machine_learning/ai/deep-learning-playground/cnn-test
##execute python script
python cnn-test.py
