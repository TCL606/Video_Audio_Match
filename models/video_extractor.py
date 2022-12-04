import torch
import torch.nn as nn
import numpy as np
from fairseq.data.audio.audio_utils import _get_torchaudio_fbank, _get_kaldi_fbank
from .subsampler import Conv1dSubsampler
from .fbank_extractor import FbankExtractor
from argparse import Namespace
from .transfomer import VideoTransformerConfig, TransformerEncoder


class VideoExtractor(nn.Module):
    def __init__(self, transformer_cfg: VideoTransformerConfig) -> None:
        super().__init__()
        self.transformer_cfg = transformer_cfg
        # self.feature_extractor = TransformerEncoder(self.transformer_cfg)
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(10, 5, 3, 1, 1),
            nn.Conv1d(5, 3, 3, 1, 1),
            nn.Conv1d(3, 1, 3, 1, 1),
        )
        # self.proj = nn.Linear(10, 1)
        self.layernorm = nn.LayerNorm(self.transformer_cfg.embed_dim)
        

    def forward(self, vfeat: torch.Tensor) -> torch.Tensor:
        feat = self.feature_extractor(vfeat).transpose(1, 2)
        x = feat.squeeze(2)
        # x = self.proj(feat).squeeze(2)
        x = self.layernorm(x)
        return x

    