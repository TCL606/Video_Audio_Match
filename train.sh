#!/bin/bash

cd /root/bqqi/changli/STD2022

model_config=configs/va_model.yaml
train_config=configs/trainva_config.yaml

python3 debug.py --train \
                --model_config $model_config \
                --train_config $train_config \
                --gpu