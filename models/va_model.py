from .audio_extractor import AudioExtractor
from .video_extractor import VideoExtractor
from .transfomer import AudioTransformerConfig, VideoTransformerConfig
from typing import List
import numpy as np
import torch.nn as nn
import torch
import torchaudio
import os

class VAModel(nn.Module):
    def __init__(self, cfg: dict) -> None:
        super().__init__()
        self.data_path = cfg['dataset']['train_dir']
        self.available_sample_num = cfg['va_model']['available_samples']
        self.neg_sample_num = cfg['va_model']['negative_samples']
        self.audio_transformer_cfg = AudioTransformerConfig(cfg)
        self.video_transformer_cfg = VideoTransformerConfig(cfg)
        self.audio_extractor = AudioExtractor(self.audio_transformer_cfg)
        self.video_extractor = VideoExtractor(self.video_transformer_cfg)
        self.linear = nn.Parameter(torch.randn(self.video_transformer_cfg.embed_dim, self.audio_transformer_cfg.embed_dim), requires_grad=True)

    def get_neg_samples(self, pos_samples: List, available_sample_num=-1, neg_sample_num=-1):
        if available_sample_num == -1:
            available_sample_num = self.available_sample_num
        if neg_sample_num == -1:
            neg_sample_num = self.neg_sample_num

        samples = np.ones(available_sample_num, dtype=np.int32)
        samples[list(pos_samples)] = 0
        neg_samples = np.random.choice(np.where(samples == 1)[0], neg_sample_num, replace=False)
        return neg_samples

    def load_neg_samples(self, neg_samples):
        afeat_list = []
        for i in neg_samples:
            afeat = np.load(os.path.join(self.data_path, 'afeat', '%04d.npy'%(i)))
            afeat_list.append(afeat)
        return torch.tensor(afeat_list)


    def extract_audio_feature(self, afeat):
        return self.audio_extractor(afeat)

    def extract_video_feature(self, video):
        return self.video_extractor(video)

    def get_probs(self, aemb, vemb):
        '''
        aemb: (batch_size, sample_num, audio_embed_dim)
        vemb: (batch_size, video_embed_dim)
        '''
        aemb_temp = aemb.transpose(1, 2) 
        logits = nn.functional.cosine_similarity(aemb_temp, torch.matmul(vemb, self.linear).unsqueeze(2))
        probs = nn.functional.softmax(logits, dim=1)
        return probs

    def forward(self, afeat, vfeat):
        aemb = self.extract_audio_feature(afeat)
        vemb = self.extract_video_feature(vfeat)
        return aemb, vemb

