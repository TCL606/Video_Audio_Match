from typing import Dict, List
import torch.utils.data as data
import torchaudio
import torch
import numpy as np
import os
from torch.nn.utils.rnn import pad_sequence

class VAKMDataset(data.Dataset):
    def __init__(self, root, npy, kmeans_npy):
        self.root = root
        self.vpath = os.path.join(root, 'vfeat')
        self.apath = os.path.join(root, 'afeat')
        self.idx = np.load(npy)
        self.kmeans_label = np.load(kmeans_npy)

    def __getitem__(self, i):
        index = self.idx[i]
        afeat = np.load(os.path.join(self.apath, '%04d.npy'%(index))).astype(np.float32)
        vfeat = np.load(os.path.join(self.vpath, '%04d.npy'%(index))).astype(np.float32)
        kmeans_id = self.kmeans_label[i]
        return i, afeat, vfeat, kmeans_id

    def __len__(self):
        return len(self.idx)

class VAKMDataloader(data.DataLoader):
    def __init__(self, dataset: VAKMDataset, batch_size, shuffle=False, num_workers=0):
        self.dataset = dataset
        self.available_sample_num = len(dataset)
        def data_collate(features: List) -> Dict[str, torch.Tensor]:
            index, afeat, vfeat, kmeans_id = zip(*features)
            afeat = torch.tensor(np.array(afeat))
            vfeat = torch.tensor(np.array(vfeat))
            kmeans_id = torch.tensor(kmeans_id, dtype=torch.int64)
            return {'index': index, 'afeat': afeat, 'vfeat': vfeat, 'kmeans_id': kmeans_id}
        super().__init__(self.dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=data_collate)

    def load_neg_afeat(self, pos_samples: List, neg_sample_num, available_sample_num=-1):
        if available_sample_num == -1:
            available_sample_num = self.available_sample_num
        # neg_sample_num = len(pos_samples) * neg_sample_times
        samples = np.ones(available_sample_num, dtype=np.int32)
        samples[list(pos_samples)] = 0
        neg_idx = np.random.choice(np.where(samples == 1)[0], neg_sample_num, replace=False)
        kmeans_id = self.dataset.kmeans_label[neg_idx]
        neg_samples = self.dataset.idx[neg_idx]
        afeat_list = []
        for i in neg_samples:
            afeat = np.load(os.path.join(self.dataset.apath, '%04d.npy'%(i)))
            afeat_list.append(afeat)
        return torch.tensor(np.array(afeat_list)), torch.tensor(kmeans_id)
