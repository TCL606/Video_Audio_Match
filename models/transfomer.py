from argparse import Namespace
import torch.nn as nn
import torch
# from fairseq.modules.transformer_layer import TransformerEncoderLayer

class TransformerConfig:
    def __init__(self, config):
        ## single transformer encoder layer
        self.dropout = float(config.get('dropout', 0.1))
        self.embed_dim = int(config.get('embed_dim', 512))
        self.ffn_embed_dim = int(config.get('ffn_embed_dim', 1024))

        ## transformer encoder
        self.num_layers = int(config.get('num_layers', 2))

class AudioTransformerConfig(TransformerConfig):
    def __init__(self, cfg):
        assert 'audio_transformer' in cfg
        config = cfg['audio_transformer']
        super().__init__(config)

class VideoTransformerConfig(TransformerConfig):
    def __init__(self, cfg):
        assert 'video_transformer' in cfg
        config = cfg['video_transformer']
        super().__init__(config)

class TransformerEncoder(nn.Module):
    def __init__(self, cfg: TransformerConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.layers = nn.ModuleList(
            [TransformerEncoderLayer(Namespace(**(cfg.__dict__))) for _ in range(cfg.num_layers)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = layer(x, None)
        return x