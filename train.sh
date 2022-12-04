#!/bin/bash

cd /mnt/e/清华/大三秋/视听信息系统导论/大作业/STD2022

model_config=configs/va_model.yaml
train_config=configs/trainva_config.yaml

python3 debug.py --train \
                --model_config $model_config \
                --train_config $train_config