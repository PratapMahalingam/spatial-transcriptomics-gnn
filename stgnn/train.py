"""Full-batch training with early stopping on held-out validation sections."""
from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from .metrics import section_macro_f1


@dataclass
class GraphBatch:
    """Several sections stacked as one disconnected graph."""
    x: torch.Tensor
    edge_index: torch.Tensor
    y: torch.Tensor
    section_ptr: np.ndarray   # section i occupies rows ptr[i]:ptr[i+1]


def stack_sections(feats: list[np.ndarray], edges: list[np.ndarray], ys: list[np.ndarray]) -> GraphBatch:
    xs, eis, offset, ptr = [], [], 0, [0]
    for x, e in zip(feats, edges):
        e = np.asarray(e, dtype=np.int64).reshape(-1, 2)
        if len(e):
            both = np.concatenate([e, e[:, ::-1]], 0).T + offset
            eis.append(both)
        xs.append(x)
        offset += len(x)
        ptr.append(offset)
    ei = np.concatenate(eis, 1) if eis else np.zeros((2, 0), dtype=np.int64)
    return GraphBatch(
        x=torch.as_tensor(np.concatenate(xs, 0), dtype=torch.float32),
        edge_index=torch.as_tensor(np.ascontiguousarray(ei), dtype=torch.long),
        y=torch.as_tensor(np.concatenate(ys, 0), dtype=torch.long),
        section_ptr=np.asarray(ptr),
    )


def class_weights(y: torch.Tensor, n_classes: int) -> torch.Tensor:
    """Inverse-frequency weights (classes absent from training get weight 0)."""
    counts = torch.bincount(y, minlength=n_classes).float()
    w = torch.where(counts > 0, counts.sum() / (n_classes * counts.clamp(min=1)), torch.zeros_like(counts))
    return w


@torch.no_grad()
def predict(model: nn.Module, batch: GraphBatch) -> np.ndarray:
    model.eval()
    return model(batch.x, batch.edge_index).argmax(1).cpu().numpy()


def mean_section_f1(y_true: np.ndarray, y_pred: np.ndarray, ptr: np.ndarray) -> float:
    return float(np.mean([section_macro_f1(y_true[a:b], y_pred[a:b]) for a, b in zip(ptr[:-1], ptr[1:])]))


def fit(model: nn.Module, train: GraphBatch, val: GraphBatch, n_classes: int, *,
        lr: float = 5e-3, weight_decay: float = 5e-4, max_epochs: int = 300,
        patience: int = 30) -> dict:
    """Train on ``train``; keep the weights with the best validation macro-F1."""
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights(train.y, n_classes))
    y_val = val.y.numpy()
    best, best_state, best_epoch, since = -1.0, None, 0, 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        opt.zero_grad()
        loss = loss_fn(model(train.x, train.edge_index), train.y)
        loss.backward()
        opt.step()
        score = mean_section_f1(y_val, predict(model, val), val.section_ptr)
        if score > best + 1e-4:
            best, best_state, best_epoch, since = score, copy.deepcopy(model.state_dict()), epoch, 0
        else:
            since += 1
            if since >= patience:
                break
    model.load_state_dict(best_state)
    return {"best_val_f1": best, "best_epoch": best_epoch, "epochs_run": epoch}
