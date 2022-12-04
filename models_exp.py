import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


class Match(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(Match, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.fc1 = nn.Linear(self.input_size, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.output_size)
        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal(m.weight)
                nn.init.constant(m.bias, 0)

    def forward(self, feat):
        feat = F.relu(self.fc1(feat))
        feat = F.relu(self.fc2(feat))
        return feat


class Prob(nn.Module):

    def __init__(self, input_size, hidden_size):
        super(Prob, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.fc1 = nn.Linear(self.input_size, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, 2)
        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal(m.weight)
                nn.init.constant(m.bias, 0)

    def forward(self, feat):
        feat = F.relu(self.fc1(feat))
        feat = self.fc2(feat)
        return feat


class FrameByFrame(nn.Module):

    def __init__(self,
                 Vinput_size=512,
                 Ainput_size=128,
                 output_size=64,
                 layers_num=3):
        super(FrameByFrame, self).__init__()
        self.Vinput_size = Vinput_size
        self.Ainput_size = Ainput_size
        self.layers_num = layers_num
        self.output_size = output_size
        self.AFeatRNN = nn.LSTM(self.Ainput_size, self.output_size,
                                self.layers_num)
        self.Amatching = Match(self.output_size, self.output_size,
                               self.output_size)
        self.Vmatching = Match(self.Vinput_size, self.output_size,
                               self.output_size)
        self.Prob = Prob(2 * self.output_size, self.output_size)

    def forward(self, Vfeat, Afeat):
        h_0 = Variable(torch.zeros(self.layers_num, Afeat.size(0),
                                   self.output_size),
                       requires_grad=False)
        c_0 = Variable(torch.zeros(self.layers_num, Afeat.size(0),
                                   self.output_size),
                       requires_grad=False)
        if Vfeat.is_cuda:
            h_0 = h_0.cuda()
            c_0 = c_0.cuda()
        outAfeat, _ = self.AFeatRNN((Afeat / 255.0).permute(2, 0, 1),
                                    (h_0, c_0))
        for i in range(10):
            Afeats = self.Amatching(outAfeat[i, :, :])
            Vfeats = self.Vmatching(Vfeat[:, :, i])
            feat = torch.cat((Afeats, Vfeats), dim=1)
            if i == 0:
                prob = self.Prob(feat)
            else:
                prob = prob + self.Prob(feat)
        prob = prob / 10
        return prob
