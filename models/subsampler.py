import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple

class Pooling1DSubsampler(nn.Module):
    def __init__(self,
                 kernel_size: int = 2,
                 stride: int = 2,
                 padding: int = 0,
                 poolingtype: str="average"):
        super(Pooling1DSubsampler, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.poolingtype = poolingtype
        if poolingtype == "average":
            self.poolinglayer = nn.AvgPool1d(kernel_size, stride, padding)
        elif poolingtype == "max":
            self.poolinglayer = nn.MaxPool1d(kernel_size, stride, padding)


    def forward(self, x, x_lengths):
        # encoder output dim: T x B x C
        x = x.transpose(0, 2)
        out = self.poolinglayer(x)
        out = out.transpose(0, 2)
        out_lengths = self.get_out_seq_lens_tensor(x_lengths)
        return out, out_lengths

    def get_out_seq_lens_tensor(self, in_seq_lens_tensor):
        out = in_seq_lens_tensor.clone()
        out = ((out.float() + 2*self.padding - self.kernel_size) / self.stride + 1).floor().long()
        return out


class SuperFrame(nn.Module):
    def __init__(self, odim):
        super(SuperFrame, self).__init__()
        # superframe: concatenates 8 succeeding frames, dim: 8 * 80 = 640
        # frame shift: 30ms
        idim = 640
        self.proj = torch.nn.Linear(idim, odim)
        nn.init.xavier_normal_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, src_tokens, src_lengths):
        n_frames = (src_tokens.size(1) // 3) * 3
        out_lengths = src_lengths // 3 - 2

        x = src_tokens[:, :n_frames, :]
        _1 = x[:, ::3, :]
        _2 = x[:, 1::3, :]
        _3 = x[:, 2::3, :]
        x = torch.cat((_1, _2, _3), dim=-1)

        _1 = x[:, 0:-2, :]
        _2 = x[:, 1:-1, :]
        _3 = x[:, 2:, :]
        x = torch.cat((_1, _2, _3), dim=-1)
        x = x[:, :, 0:x.size(-1)-80]
        x = x.transpose(0, 1)
        x_out = self.proj(x)
        x_out = nn.functional.glu(x_out, dim=1)
        return x_out, out_lengths


class Conv1dSubsampler(nn.Module):
    """Convolutional subsampler: a stack of 1D convolution (along temporal
    dimension) followed by non-linear activation via gated linear units
    (https://arxiv.org/abs/1911.08460)

    Args:
        in_channels (int): the number of input channels
        mid_channels (int): the number of intermediate channels
        out_channels (int): the number of output channels
        kernel_sizes (List[int]): the kernel size for each convolutional layer
    """

    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        out_channels: int,
        kernel_sizes: List[int] = (3, 3),
    ):
        super(Conv1dSubsampler, self).__init__()
        self.n_layers = len(kernel_sizes)
        self.conv_layers = nn.ModuleList(
            nn.Conv1d(
                in_channels if i == 0 else mid_channels // 2,
                mid_channels if i < self.n_layers - 1 else out_channels * 2,
                k,
                stride=2,
                padding=k // 2,
            )
            for i, k in enumerate(kernel_sizes)
        )

    def get_out_seq_lens_tensor(self, in_seq_lens_tensor):
        out = in_seq_lens_tensor.clone()
        for _ in range(self.n_layers):
            out = ((out.float() - 1) / 2 + 1).floor().long()
        return out

    def forward(self, src_tokens, src_lengths):
        bsz, in_seq_len, _ = src_tokens.size()  # B x T x (C x D)
        x = src_tokens.transpose(1, 2).contiguous()  # -> B x (C x D) x T
        for conv in self.conv_layers:
            x = conv(x)
            x = nn.functional.glu(x, dim=1)
        _, _, out_seq_len = x.size()
        x = x.transpose(1, 2).transpose(0, 1).contiguous()  # -> T x B x (C x D)
        return x, self.get_out_seq_lens_tensor(src_lengths)