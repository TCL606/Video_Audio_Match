#!/bin/bash
python3 main.py --train --gpu
# --train_dir specifies train data directory
# --valid_dir specifies valid data directory (often the same as --train_dir)
# --train_npy specifies the npy file containing ids of training data (usually don't have to change)
# --valid_npy specifies the npy file containing ids of validating data (usually don't have to change)
# all paths if not specified will be assigned to the default value in configs