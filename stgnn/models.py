"""Three models with the same depth and width: 2 hidden layers + linear read-out.

* MLP:  expression of the spot only.
* SAGE: GraphSAGE (mean aggregation); each layer mixes a spot with its neighbours.
* GCN:  graph convolution with symmetric normalisation and self-loops.

Run with an empty edge set, SAGE and GCN see only the spot itself, which gives
an "architecture-matched, no-neighbour" control.
"""
from __future__ import annotations

import torch
from torch import nn
from torch_geometric.nn import GCNConv, SAGEConv


class MLP(nn.Module):
    def __init__(self, d_in: int, n_classes: int, hidden: int = 128, dropout: float = 0.3):
        super().__init__()
        self.l1, self.l2 = nn.Linear(d_in, hidden), nn.Linear(hidden, hidden)
        self.out = nn.Linear(hidden, n_classes)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, edge_index=None):
        x = self.drop(torch.relu(self.l1(x)))
        x = self.drop(torch.relu(self.l2(x)))
        return self.out(x)


class _GraphNet(nn.Module):
    conv_cls = None

    def __init__(self, d_in: int, n_classes: int, hidden: int = 128, dropout: float = 0.3):
        super().__init__()
        self.c1, self.c2 = self.conv_cls(d_in, hidden), self.conv_cls(hidden, hidden)
        self.out = nn.Linear(hidden, n_classes)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        x = self.drop(torch.relu(self.c1(x, edge_index)))
        x = self.drop(torch.relu(self.c2(x, edge_index)))
        return self.out(x)


class SAGE(_GraphNet):
    conv_cls = SAGEConv


class GCN(_GraphNet):
    conv_cls = GCNConv


MODELS = {"mlp": MLP, "sage": SAGE, "gcn": GCN}


def build_model(arch: str, d_in: int, n_classes: int, **kw) -> nn.Module:
    return MODELS[arch](d_in, n_classes, **kw)
