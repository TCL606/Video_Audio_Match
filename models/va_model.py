from .audio_extractor import AudioExtractor
from .video_extractor import VideoExtractor
from typing import List
import numpy as np
import torch.nn as nn
import torch
import torchaudio
import os
from .transfomer import *

class VAModel(nn.Module):
    def __init__(self, cfg: dict) -> None:
        super().__init__()
        self.data_path = cfg['dataset']['train_dir']
        self.available_sample_num = cfg['va_model']['available_samples']
        self.neg_sample_num = cfg['va_model']['negative_samples']
        self.transformer_cfg = AudioTransformerConfig(cfg)
        # self.video_transformer_cfg = VideoTransformerConfig(cfg)
        self.audio_extractor = nn.Sequential(
            nn.Conv1d(10, 32, 3, 1, 1),
            nn.Conv1d(32, 64, 3, 1, 1),
        )
        self.video_extractor = nn.Sequential(
            nn.Conv1d(10, 32, 3, 1, 1),
            nn.Conv1d(32, 64, 3, 1, 1),
            nn.Conv1d(64, 64, 3, 1, 1),
        )

        self.transformer_key = TransformerEncoder(self.transformer_cfg)
        self.joint_extractor_key = nn.Sequential(     # pos
            nn.Conv1d(64, 32, 3, 1, 1),
            nn.Conv1d(32, 1, 3, 1, 1)
        )
        
        self.transformer_query = TransformerEncoder(self.transformer_cfg)
        self.joint_extractor_query = nn.Sequential(     # neg
            nn.Conv1d(64, 32, 3, 1, 1),
            nn.Conv1d(32, 1, 3, 1, 1)
        )

    def forward(self, afeat, vfeat, neg_afeat):
        device = afeat.device
        aemb = self.audio_extractor(afeat).squeeze(1)
        vemb = self.video_extractor(vfeat).squeeze(1)

        joint_emb_key = self.joint_extract_key(aemb, vemb)
        if neg_afeat is None:
            return aemb, vemb, joint_emb_key
        else:
            bz = afeat.shape[0]
            neg_times = int(neg_afeat.shape[0] / bz)
            
            neg_aemb = self.audio_extractor(neg_afeat).squeeze(1)
            total_aemb = torch.cat((aemb, neg_aemb), dim=0)
            total_vemb = vemb.repeat(neg_times + 1, 1, 1)

            joint_emb_query = self.joint_extract_query(total_aemb, total_vemb)
            joint_emb_query = joint_emb_query.view(neg_times + 1, bz, -1)

            cos_sim = torch.stack([torch.cosine_similarity(joint_emb_key[i], joint_emb_query[:, i, :], dim=1) for i in range(bz)])
            probs = torch.softmax(cos_sim, dim=1)
            labels = torch.zeros(probs.shape[0], dtype=torch.long).to(device)
            return probs, labels

    def joint_extract_key(self, aemb, vemb):
        emb_1 = self.transformer_key(torch.cat((vemb, aemb), dim=2).transpose(0, 1)).transpose(0, 1)
        joint_emb_key = self.joint_extractor_key(emb_1).squeeze(1)
        return joint_emb_key
    
    def joint_extract_query(self, total_aemb, total_vemb):
        emb_2 = self.transformer_query(torch.cat((total_vemb, total_aemb), dim=2).transpose(0, 1)).transpose(0, 1)
        joint_emb_query = self.joint_extractor_query(emb_2).squeeze(1)
        return joint_emb_query

    # def get_neg_samples(self, pos_samples: List, neg_sample_num=-1, available_sample_num=-1):
    #     if available_sample_num == -1:
    #         available_sample_num = self.available_sample_num
    #     if neg_sample_num == -1:
    #         neg_sample_num = self.neg_sample_num

    #     samples = np.ones(available_sample_num, dtype=np.int32)
    #     samples[list(pos_samples)] = 0
    #     neg_samples = np.random.choice(np.where(samples == 1)[0], neg_sample_num, replace=False)
    #     return neg_samples

    # def load_neg_samples(self, neg_samples):
    #     afeat_list = []
    #     for i in neg_samples:
    #         afeat = np.load(os.path.join(self.data_path, 'afeat', '%04d.npy'%(i)))
    #         afeat_list.append(afeat)
    #     return torch.tensor(np.array(afeat_list))


    # # def extract_audio_feature(self, afeat):
    # #     return self.audio_extractor(afeat)

    # # def extract_video_feature(self, video):
    # #     return self.video_extractor(video)

    # def get_probs(self, aemb, vemb):
    #     '''
    #     aemb: (batch_size, sample_num, audio_embed_dim)
    #     vemb: (batch_size, video_embed_dim)
    #     '''
    #     aemb_temp = aemb.transpose(1, 2) 
    #     logits = nn.functional.cosine_similarity(aemb_temp, torch.matmul(vemb, self.linear).unsqueeze(2))
    #     probs = torch.softmax(logits, dim=1)
    #     return probs

    # def forward(self, afeat, vfeat):
    #     aemb = self.extract_audio_feature(afeat)
    #     vemb = self.extract_video_feature(vfeat)
    #     return aemb, vemb
