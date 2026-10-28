"""GCN and GraphSAGE in plain PyTorch (sparse ops) - no torch-geometric dependency."""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F


def _to_torch(m: sp.spmatrix, device) -> torch.Tensor:
    m = m.tocoo()
    idx = torch.tensor(np.vstack([m.row, m.col]), dtype=torch.long)
    return torch.sparse_coo_tensor(idx, torch.tensor(m.data, dtype=torch.float32), m.shape).coalesce().to(device)


def gcn_norm(A_dir: sp.spmatrix) -> sp.csr_matrix:
    """D^-1/2 (A + A^T + I) D^-1/2 (Kipf & Welling)."""
    A = ((A_dir + A_dir.T) > 0).astype("float32") + sp.eye(A_dir.shape[0], dtype="float32")
    d = np.asarray(A.sum(1)).ravel() ** -0.5
    return sp.diags(d) @ A @ sp.diags(d)


def mean_norm(A_dir: sp.spmatrix) -> sp.csr_matrix:
    """Row-normalised undirected adjacency (mean aggregator of GraphSAGE)."""
    A = ((A_dir + A_dir.T) > 0).astype("float32")
    deg = np.asarray(A.sum(1)).ravel()
    inv = np.divide(1.0, deg, out=np.zeros_like(deg), where=deg > 0)
    return sp.diags(inv) @ A


class GCN(nn.Module):
    def __init__(self, in_dim, hidden, layers=2, dropout=0.3):
        super().__init__()
        dims = [in_dim] + [hidden] * (layers - 1) + [hidden]
        self.lin = nn.ModuleList(nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:]))
        self.out = nn.Linear(hidden, 2)
        self.dropout = dropout

    def forward(self, x, adj):
        for lin in self.lin:
            x = F.dropout(x, self.dropout, self.training)
            x = F.relu(torch.sparse.mm(adj, lin(x)))
        return self.out(F.dropout(x, self.dropout, self.training))

    def embed(self, x, adj):
        with torch.no_grad():
            for lin in self.lin:
                x = F.relu(torch.sparse.mm(adj, lin(x)))
        return x


class GraphSAGE(nn.Module):
    """h_v' = ReLU(W [h_v || mean_{u in N(v)} h_u]) with L2-normalised output (Hamilton et al., 2017)."""

    def __init__(self, in_dim, hidden, layers=2, dropout=0.3):
        super().__init__()
        dims = [in_dim] + [hidden] * layers
        self.lin = nn.ModuleList(nn.Linear(2 * a, b) for a, b in zip(dims[:-1], dims[1:]))
        self.out = nn.Linear(hidden, 2)
        self.dropout = dropout

    def _layer(self, lin, x, adj):
        h = lin(torch.cat([x, torch.sparse.mm(adj, x)], 1))
        return F.normalize(F.relu(h), dim=1)

    def forward(self, x, adj):
        for lin in self.lin:
            x = self._layer(lin, F.dropout(x, self.dropout, self.training), adj)
        return self.out(F.dropout(x, self.dropout, self.training))

    def embed(self, x, adj):
        with torch.no_grad():
            for lin in self.lin:
                x = self._layer(lin, x, adj)
        return x


ARCHS = {"gcn": (GCN, gcn_norm), "graphsage": (GraphSAGE, mean_norm)}


def make(arch: str, in_dim: int, hidden: int, layers: int, dropout: float):
    cls, norm = ARCHS[arch]
    return cls(in_dim, hidden, layers, dropout), norm


to_torch = _to_torch
