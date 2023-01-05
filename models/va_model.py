from typing import List
import numpy as np
import torch.nn as nn
import torch
import torchaudio
import os

class VAModel(nn.Module):
    def __init__(self, cfg: dict) -> None:
        super().__init__()
        self.audio_extractor = nn.Sequential(
            nn.Conv1d(10, 32, 3, 1, 1),
            nn.Conv1d(32, 64, 3, 1, 1),
            nn.Conv1d(64, 32, 3, 1, 1),
            nn.Conv1d(32, 1, 3, 1, 1)
        )
        self.video_extractor = nn.Sequential(
            nn.Conv1d(10, 32, 3, 1, 1),
            nn.Conv1d(32, 64, 3, 1, 1),
            nn.Conv1d(64, 64, 3, 1, 1),
            nn.Conv1d(64, 32, 3, 1, 1),
            nn.Conv1d(32, 1, 3, 1, 1),
            nn.Linear(512, 128)
        )

    def forward(self, afeat, vfeat, neg_afeat):
        device = afeat.device
        aemb = self.audio_extractor(afeat).squeeze(1)
        vemb = self.video_extractor(vfeat).squeeze(1)
        if neg_afeat is None:
            return aemb, vemb
        else:
            bz = afeat.shape[0]
            
            neg_aemb = self.audio_extractor(neg_afeat).squeeze(1)
            total_aemb = [torch.cat((aemb[i: i+1], neg_aemb), dim=0) for i in range(bz)]
            
            # vemb: key; ameb: query
            cos_sim = torch.stack([torch.cosine_similarity(vemb[i], total_aemb[i], dim=1) for i in range(bz)])
            # probs = torch.softmax(cos_sim, dim=1)
            labels = torch.zeros(cos_sim.shape[0], dtype=torch.long).to(device)
            return cos_sim, labels
