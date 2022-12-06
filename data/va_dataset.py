from typing import Dict, List
import torch.utils.data as data
import torchaudio
import torch
import numpy as np
import os
from torch.nn.utils.rnn import pad_sequence

class VADataset(data.Dataset):
    def __init__(self, root, npy, split='train'):
        self.root = root
        self.vpath = os.path.join(root, 'vfeat')
        self.apath = os.path.join(root, 'afeat')
        self.idx = np.load(npy)

    def __getitem__(self, i):
        index = self.idx[i]
        afeat = np.load(os.path.join(self.apath, '%04d.npy'%(index))).astype(np.float32)
        vfeat = np.load(os.path.join(self.vpath, '%04d.npy'%(index))).astype(np.float32)
        return i, afeat, vfeat

    def __len__(self):
        return len(self.idx)

class VADataloader(data.DataLoader):
    def __init__(self, dataset: VADataset, batch_size, shuffle=False, num_workers=0):
        self.dataset = dataset
        self.available_sample_num = len(dataset)
        def data_collate(features: List) -> Dict[str, torch.Tensor]:
            index, afeat, vfeat = zip(*features)
            afeat = torch.tensor(np.array(afeat))
            vfeat = torch.tensor(np.array(vfeat))
            return {'index': index, 'afeat': afeat, 'vfeat': vfeat}
        super().__init__(self.dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=data_collate)

    # def get_neg_samples(self, pos_samples: List, neg_sample_num=-1, available_sample_num=-1):
    #     if available_sample_num == -1:
    #         available_sample_num = self.available_sample_num
    #     if neg_sample_num == -1:
    #         neg_sample_num = self.neg_sample_num

    #     samples = np.ones(available_sample_num, dtype=np.int32)
    #     samples[list(pos_samples)] = 0
    #     neg_samples = np.random.choice(np.where(samples == 1)[0], neg_sample_num, replace=False)
    #     return self.dataset.idx[neg_samples]

    def load_neg_afeat(self, pos_samples: List, neg_sample_num, available_sample_num=-1):
        if available_sample_num == -1:
            available_sample_num = self.available_sample_num
        # neg_sample_num = len(pos_samples) * neg_sample_times
        samples = np.ones(available_sample_num, dtype=np.int32)
        samples[list(pos_samples)] = 0
        neg_samples = np.random.choice(np.where(samples == 1)[0], neg_sample_num, replace=False)
        neg_samples = self.dataset.idx[neg_samples]
        afeat_list = []
        for i in neg_samples:
            afeat = np.load(os.path.join(self.dataset.apath, '%04d.npy'%(i)))
            afeat_list.append(afeat)
        return torch.tensor(np.array(afeat_list))
