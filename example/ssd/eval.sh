#! /usr/bin/bash

# simple eval sh

# ssd_vgg16_reduce  tupu face data
date; python evaluate.py --gpus 0,3 --batch-size 128 --epoch 0 --rec-path ./data_face/val.rec --network vgg16_reduced  --prefix './model/new_train_model_tupu_face/vgg16_reduced/ssd_' --class-names '0,1' --num-class 2 --data-shape 300; date

# ssd_resnet50
date; python evaluate.py --gpus 0,3 --batch-size 128 --epoch 0 --network resnet50 --data-shape 512; date

# ssd_inceptionv3
#date; python evaluate.py --gpus 0,3 --batch-size 128 --epoch 0 --network inceptionv3 --data-shape 512; date
