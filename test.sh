#!/bin/bash
python3 main.py --test --gpu --test_dir /root/Test/Noise -k 5
# --test_dir must be assigned
# -k means to test in the sense of top k accuracy
# --ckpt_path specifies the path to the checkpoint, default is "./output/debug/KMeans5e-4_state_epoch_last.pth"