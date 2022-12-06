from typing import List
import torch
import torch.nn as nn
import numpy as np
# from fairseq.data.audio.audio_utils import _get_torchaudio_fbank, _get_kaldi_fbank
from .subsampler import Conv1dSubsampler
from .fbank_extractor import FbankExtractor
from argparse import Namespace
from .transfomer import AudioTransformerConfig, TransformerEncoder


class AudioExtractor(nn.Module):
    def __init__(self, transformer_cfg: AudioTransformerConfig) -> None:
        super().__init__()
        self.transformer_cfg = transformer_cfg
        # self.feature_extractor = TransformerEncoder(self.transformer_cfg)
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(10, 5, 3, 1, 1),
            nn.Conv1d(5, 3, 2, 1, 1),
            nn.Conv1d(3, 1, 2, 1, 1),
        )
        # self.proj = nn.Linear(10, 1)
        self.layernorm = nn.LayerNorm(self.transformer_cfg.embed_dim)

    # def forward(self, wav: torch.Tensor, wav_len: List) -> torch.Tensor:
    #     fbank, enc_out_len = self.fbank_extractor(wav, wav_len)     # T * B * C 
    #     feat = self.feature_extractor(fbank)
    #     feat = feat.permute(1, 2, 0)
    #     x = self.proj(feat).squeeze(2)
    #     x = self.layernorm(x)
    #     return x

    def forward(self, afeat: torch.Tensor) -> torch.Tensor:
        feat = self.feature_extractor(afeat).transpose(1, 2)
        x = feat.squeeze(2)
        x = self.layernorm(x)
        return x

    