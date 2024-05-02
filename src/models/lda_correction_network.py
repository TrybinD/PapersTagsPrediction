from torch import nn

import torch

from deepxml.cornet import CorNet


class LDACorrectionNet(nn.Module):
    def __init__(self, num_topics, num_labels, emb_size, init_embs=None):
        super().__init__()
        self.linear1 = nn.Linear(num_topics, emb_size, dtype=float, bias=False)
        self.act = torch.sigmoid
        self.linear2 = nn.Linear(emb_size, num_labels, dtype=float)

        if init_embs is not None:
            with torch.no_grad():
                self.linear1.weight.copy_(torch.tensor(init_embs).T)


    def forward(self, input):
        x = self.linear1(input)
        x = self.act(x)
        x = self.linear2(x)

        return x
    

class CorNetLDACorrectionNet(nn.Module):
    def __init__(self, num_topics, num_labels, emb_size):
        super().__init__()
        self.lda_net = LDACorrectionNet(num_topics, num_labels, emb_size)
        self.cor_net = CorNet(num_topics, cornet_dim=100)

    def forward(self, input):
        x = self.lda_net(input)
        x = self.cor_net(x.float())

        return x
    

class LDACorrectionNetLarge(nn.Module):
    def __init__(self, num_topics, num_labels, emb_size, hidden_states, init_embs=None):
        super().__init__()
        self.embs = nn.Sequential(nn.Linear(num_topics, emb_size, bias=False, dtype=float),
                                  nn.LayerNorm(emb_size, dtype=float),
                                  nn.Linear(emb_size, hidden_states[0], dtype=float))
        self.act = nn.ELU()
        self.dropout = nn.Dropout(0.5)
        self.main = nn.ModuleList(
            [nn.Linear(in_size, out_size, dtype=float) for in_size, out_size in zip(hidden_states, hidden_states[1:])]
        )
        self.output = nn.Linear(hidden_states[-1], num_labels, dtype=float)

        if init_embs is not None:
            with torch.no_grad():
                self.embs.get_submodule("0").weight.copy_(torch.tensor(init_embs).T)

    def forward(self, input):
        x = self.embs(input)
        for l in self.main:
            x = self.dropout(x)
            x = self.act(l(x))

        x = self.output(x)
        return x
    

class CorNetLDACorrectionNetLarge(nn.Module):
    def __init__(self, num_topics, num_labels, emb_size, hidden_states):
        super().__init__()
        self.lda_net = LDACorrectionNetLarge(num_topics, num_labels, emb_size, hidden_states)
        self.cor_net = CorNet(num_topics, cornet_dim=300)

    def forward(self, input):
        x = self.lda_net(input)
        x = self.cor_net(x.float())

        return x