from typing import List
import numpy as np
import torch.nn as nn
import torch
import torchaudio
import os

class VAKMModel(nn.Module):
    def __init__(self, cfg: dict) -> None:
        super().__init__()
        self.video_shape = cfg['video_feat_shape']
        self.audio_shape = cfg['audio_feat_shape']
        self.kmeans_num = cfg['kmeans_num']
        self.audio_extractor = nn.Sequential(
            nn.Conv1d(self.audio_shape[0], 32, 3, 1, 1),
            nn.Conv1d(32, 64, 3, 1, 1),
        )
        self.aemb_extractor = nn.Sequential(
            nn.Conv1d(64, 32, 3, 1, 1),
            nn.Conv1d(32, 1, 3, 1, 1)
        )
        self.aaux_extractor = nn.Sequential(
            nn.Conv1d(64, 1, 3, 1, 1),
            nn.Linear(self.audio_shape[1], self.kmeans_num)
        )

        self.video_extractor = nn.Sequential(
            nn.Conv1d(self.video_shape[0], 32, 3, 1, 1),
            nn.Conv1d(32, 64, 3, 1, 1),
            nn.Conv1d(64, 64, 3, 1, 1),
        )
        self.vemb_extractor = nn.Sequential(
            nn.Conv1d(64, 32, 3, 1, 1),
            nn.Conv1d(32, 1, 3, 1, 1),
            nn.Linear(self.video_shape[1], self.audio_shape[1])
        )

    def ahidden_extract(self, afeat):
        return self.audio_extractor(afeat)

    def vhidden_extract(self, vfeat):
        return self.video_extractor(vfeat)

    def aemb_extract(self, ahidden):
        return self.aemb_extractor(ahidden)

    def vemb_extract(self, vhidden):
        return self.vemb_extractor(vhidden)

    def aaux_extract(self, ahidden):
        pred = self.aaux_extractor(ahidden).squeeze(1)
        return pred

    def forward(self, afeat, vfeat, neg_afeat, pos_ki=None, neg_ki=None):
        device = afeat.device
        ah = self.ahidden_extract(afeat)
        aemb = self.aemb_extract(ah).squeeze(1)
        vh = self.vhidden_extract(vfeat)
        vemb = self.vemb_extract(vh).squeeze(1)
        if neg_afeat is None:
            return aemb, vemb
        elif neg_ki is None or pos_ki is None:
            bz = afeat.shape[0]
            neg_ah = self.ahidden_extract(neg_afeat)
            neg_aemb = self.aemb_extract(neg_ah).squeeze(1)
            total_aemb = [torch.cat((aemb[i: i+1], neg_aemb), dim=0) for i in range(bz)]
            cos_sim = torch.stack([torch.cosine_similarity(vemb[i], total_aemb[i], dim=1) for i in range(bz)])
            labels = torch.zeros(cos_sim.shape[0], dtype=torch.long).to(device)
            return cos_sim, labels
        else:
            bz = afeat.shape[0]
            
            neg_ah = self.ahidden_extract(neg_afeat)
            neg_aemb = self.aemb_extract(neg_ah).squeeze(1)
            total_aemb = [torch.cat((aemb[i: i+1], neg_aemb), dim=0) for i in range(bz)]
            
            # vemb: key; ameb: query
            cos_sim = torch.stack([torch.cosine_similarity(vemb[i], total_aemb[i], dim=1) for i in range(bz)])
            # probs = torch.softmax(cos_sim, dim=1)
            labels = torch.zeros(cos_sim.shape[0], dtype=torch.long).to(device)

            total_ah = torch.cat((ah, neg_ah), dim=0)
            k_preds = self.aaux_extract(total_ah)
            k_labels = torch.cat((pos_ki, neg_ki), dim=0).to(device)
            return cos_sim, labels, k_preds, k_labels
