from typing import List, Optional

from transformers import AutoModel
from torch import nn


class RuBERTXML(nn.Module):
    def __init__(self, labels_num: int, hidden_layers: Optional[List], bert_model: str) -> None:
        super().__init__()
        self.bert = AutoModel.from_pretrained(bert_model)
        self.act = nn.ReLU()
        if hidden_layers is None:
            hidden_layers = [312]
        else:
            hidden_layers = [312] + hidden_layers

        self.main = nn.ModuleList(
            [nn.Linear(in_size, out_size, dtype=float) for in_size, out_size in zip(hidden_layers, hidden_layers[1:])]
        )

        self.output = nn.Linear(hidden_layers[-1], labels_num, dtype=float)

    def forward(self, input):
        input_ids = input["ids"]
        attention_mask = input["mask"]

        x = self.bert(input_ids, attention_mask)
        x = x.last_hidden_state[:, 0, :]

        for l in self.main:
            x = self.act(l(x.double()))

        x = self.output(x)
        return x


class CorNetRuBERTXML(nn.Module):
    def __init__(self, labels_num: int, hidden_layers: Optional[List], bert_model: str) -> None:
        super().__init__()
        self.rubert = RuBERTXML(labels_num, hidden_layers, bert_model)
        self.cor_net = self.cor_net = CorNet(labels_num, cornet_dim=100)

    def forward(self, input):
        x = self.rubert(input)
        x = self.cor_net(x.float())

        return x
