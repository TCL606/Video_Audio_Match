#!/bin/bash

cd /root/bqqi/changli/STD2022

python3 debug.py  --train \
                --test_dir /root/bqqi/changli/Test/Clean \
                --valid_npy /root/bqqi/changli/STD2022/data/valid_334.npy \
                --gpu