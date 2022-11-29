import torch.utils.data as data
import numpy as np
import os

class VideoFeatDataset(data.Dataset):
    def __init__(self, root):
        self.root = root
        self.apath = os.path.join(root, 'afeat')
        self.vpath = os.path.join(root, 'vfeat')

    def __getitem__(self, index):
        vfeat = np.load(os.path.join(self.vpath, '%04d.npy'%(index))).astype('float32')
        afeat = np.load(os.path.join(self.apath, '%04d.npy'%(index))).astype('float32')
        return vfeat, afeat

    def __len__(self):
        return len(os.listdir(self.apath))

    def loader(self, filepath):
        return np.load(filepath)