import torch
import torch.nn as nn
import numpy as np
from fairseq.data.audio.audio_utils import _get_torchaudio_fbank, _get_kaldi_fbank
from .subsampler import Conv1dSubsampler

class FbankExtractor(nn.Module):
    def __init__(self, conv_channels, encoder_embed_dim, conv_kernel_sizes, dropout, specaug, cuda=False) -> None:
        super().__init__()
        # self.encoder = encoder
        self.specaug = specaug
        self.gpu = cuda
        self.subsample = Conv1dSubsampler(
            80,
            conv_channels,
            encoder_embed_dim,
            [int(k) for k in conv_kernel_sizes.split(",")] 
        )
        self.linear = torch.nn.Linear(encoder_embed_dim, encoder_embed_dim)
        nn.init.xavier_normal_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
        self.dropout = torch.nn.Dropout(dropout)
        
        # global cmvn
        stats_npz_path = "E:\\清华\\大三秋\\视听信息系统导论\\大作业\\STD2022\\utils\\global_cmvn.npy"
        stats = np.load(stats_npz_path, allow_pickle=True).tolist()
        self.mean, self.std = stats["mean"], stats["std"]
        
        # specaug
        # if specaug:
        #     specaug_config = {"freq_mask_F": 30, "freq_mask_N": 2, "time_mask_N": 2, "time_mask_T": 40, "time_mask_p": 1.0, "time_wrap_W": 0}
        #     from fairseq.data.audio.feature_transforms.specaugment import SpecAugmentTransform
        #     self.specaug_transform = SpecAugmentTransform.from_config_dict(specaug_config)
        #     logger.info(f"Train with specaug")
        # else:
        #     logger.info(f"Train without specaug")
        
    def forward(self, source, src_lengths):
        return self.extract_fbank_features(source, src_lengths)

    def extract_fbank_features(self, source, src_lengths):
        sample_rate = 44100
        n_mel_bins = 80

        fbank_lengths = []
        fbank_features = []
        data_dtype = source.dtype
        with torch.no_grad():
            source = source.float()
            for batch_idx in range(source.size(0)):
                _waveform = source[batch_idx][:src_lengths[batch_idx]]
                _waveform = _waveform * (2 ** 15)
                _waveform = _waveform.float().cpu().unsqueeze(0).numpy()
                features = _get_kaldi_fbank(_waveform, sample_rate, n_mel_bins)
                if features is None:
                    features = _get_torchaudio_fbank(_waveform, sample_rate, n_mel_bins)

                features = torch.from_numpy(features)
                features = np.subtract(features, self.mean)
                features = np.divide(features, self.std)
                if self.gpu:
                    features = features.cuda()

                feat_len = features.size(0)
                if batch_idx == 0:
                    max_len  = feat_len
                else:
                    if feat_len != max_len:
                        pad_len = max_len - feat_len
                        features_padding = features.new(pad_len, n_mel_bins).fill_(0)
                        features = torch.cat([features, features_padding], dim=0)
                        features = features.type(source.dtype)
                # only apply specaug during Training
                # if apply_specaug is True and self.specaug:
                #     features = self.specaug_transform(features)

                fbank_features.append(features)
                fbank_lengths.append(feat_len)

            fbank_features = torch.stack(fbank_features, dim=0).contiguous().type(data_dtype)
            fbank_lengths = torch.Tensor(fbank_lengths).int()
            if self.gpu:
                fbank_lengths = fbank_lengths.cuda()

        fbank_features, encoder_out_lengths = self.subsample(fbank_features, src_lengths=fbank_lengths)
        fbank_features = self.linear(fbank_features)
        fbank_features = self.dropout(fbank_features)
        return fbank_features, encoder_out_lengths