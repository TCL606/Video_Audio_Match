from typing import Dict, List
import torch.utils.data as data
import torchaudio
import torch
import numpy as np
import os
from torch.nn.utils.rnn import pad_sequence

class VADataset(data.Dataset):
    def __init__(self, root):
        self.root = root
        self.vpath = os.path.join(root, 'vfeat')
        self.apath = os.path.join(root, 'afeat')

    def __getitem__(self, index):
        # wav, _ = torchaudio.load(os.path.join(self.apath, '%04d.wav'%(index)))
        # wav = (wav[0] + wav[1]) / 2
        afeat = np.load(os.path.join(self.apath, '%04d.npy'%(index))).astype(np.float32)
        vfeat = np.load(os.path.join(self.vpath, '%04d.npy'%(index))).astype(np.float32)
        return index, afeat, vfeat

    def __len__(self):
        return len(os.listdir(self.apath))

class VADataloader(data.DataLoader):
    def __init__(self, dataset: VADataset, batch_size=1, shuffle=False, num_workers=0):
        self.dataset = dataset
        def data_collate(features: List) -> Dict[str, torch.Tensor]:
            index, afeat, vfeat = zip(*features)
            # wav_len = [x.shape[0] for x in wav]
            # wav_len = torch.LongTensor(wav_len)
            # wav = pad_sequence(wav, batch_first=True)
            afeat = torch.tensor(afeat)
            vfeat = torch.tensor(vfeat)
            return {'index': index, 'afeat': afeat, 'vfeat': vfeat}
        super().__init__(self.dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=data_collate)
