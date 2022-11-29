import os
import time
import torch
import numpy as np

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n # val*n: how many samples predicted correctly among the n samples
        self.count += n     # totoal samples has been through
        self.avg = self.sum / self.count

def save_checkpoint(model, output_dir, output_name):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    torch.save(model, os.path.join(output_dir, output_name))   
    print("Checkpoint saved to {}".format(os.path.join(output_dir, output_name)))
