"""
BFHD (supplementary) - minimal reference implementation.

Includes:
  - NICE-style shared bidirectional backbone (additive coupling)
  - Hard-concrete (L0) output gates
  - Three-stage training losses (stage1 NMSE warm-up, stage2 hinge-feasibility + gate reward, stage3 harden + finetune)

This is intended as a minimal reference implementation of the proposed method.
Dataset-specific preprocessing and scripts are omitted; all hyperparameters and protocols are specified in the paper/SM.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List, Any
import copy
import math
import json
import argparse
import copy
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Optional deps for practical data loading (same style as your prior scripts)
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# -------------------------
# Utils: seeds, stats
# -------------------------

def set_seed(seed: int = 0):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pad_to_dim_np(arr: np.ndarray, target_dim: int) -> np.ndarray:
    if arr.shape[1] == target_dim:
        return arr.astype(np.float32, copy=False)
    if arr.shape[1] > target_dim:
        return arr[:, :target_dim].astype(np.float32, copy=False)
    pad = np.zeros((arr.shape[0], target_dim - arr.shape[1]), dtype=np.float32)
    return np.concatenate([arr.astype(np.float32, copy=False), pad], axis=1)


def pad_and_add_error_dim_np(arr: np.ndarray, orig_dim: int, target_dim: int, error_dim_num: int) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    pad_dim = max(0, target_dim - orig_dim)
    if pad_dim == 0 and error_dim_num == 0:
        return arr
    zeros = np.zeros((arr.shape[0], pad_dim + error_dim_num), dtype=np.float32)
    return np.concatenate([arr, zeros], axis=1)


def featurewise_nmse(y_true: torch.Tensor, y_pred: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    mse = torch.mean((y_true - y_pred) ** 2, dim=0)
    denom = torch.mean(y_true ** 2, dim=0).clamp_min(eps)
    return mse / denom


def ccc_1d_torch(y_true: torch.Tensor, y_pred: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """CCC per feature: input [N,D] -> output [D]."""
    mu_x = torch.mean(y_true, dim=0)
    mu_y = torch.mean(y_pred, dim=0)
    vx = torch.mean((y_true - mu_x) ** 2, dim=0)
    vy = torch.mean((y_pred - mu_y) ** 2, dim=0)
    cov = torch.mean((y_true - mu_x) * (y_pred - mu_y), dim=0)
    return (2 * cov) / (vx + vy + (mu_x - mu_y) ** 2 + eps)


# -------------------------
# NICE backbone (additive coupling)
# -------------------------

class CouplingLayer(nn.Module):
    """Additive coupling (NICE). Invertible by construction."""
    def __init__(self, dim: int, hidden_dim: int, mask: torch.Tensor):
        super().__init__()
        self.register_buffer("mask", mask.view(1, -1))
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x: torch.Tensor, reverse: bool = False) -> torch.Tensor:
        x_masked = x * self.mask
        m = self.net(x_masked)
        if reverse:
            return x_masked + (1.0 - self.mask) * (x - m)
        return x_masked + (1.0 - self.mask) * (x + m)


class NICEBlock(nn.Module):
    def __init__(self, dim: int, valid_dim: int, hidden_dims: List[int]):
        super().__init__()
        self.dim = dim
        self.valid_dim = valid_dim
        self.layers = nn.ModuleList()

        half = max(1, valid_dim // 2)
        for k, h in enumerate(hidden_dims):
            mask = torch.zeros(dim, dtype=torch.float32)
            if k % 2 == 0:
                mask[:half] = 1.0
            else:
                mask[half:valid_dim] = 1.0
            if valid_dim < dim:
                mask[valid_dim:] = 1.0
            self.layers.append(CouplingLayer(dim, h, mask))

    def forward(self, x: torch.Tensor, reverse: bool = False) -> torch.Tensor:
        if not reverse:
            for layer in self.layers:
                x = layer(x, reverse=False)
        else:
            for layer in reversed(self.layers):
                x = layer(x, reverse=True)
        return x


class NiceDualBackbone(nn.Module):
    """
    Shared bidirectional operator:
      X -> Z via NICE_x forward, then Z -> Y via NICE_y inverse
      Y -> Z via NICE_y forward, then Z -> X via NICE_x inverse
    """
    def __init__(self, dim: int, valid_dim_x: int, valid_dim_y: int,
                 hidden_x: List[int], hidden_y: List[int]):
        super().__init__()
        self.dim = dim
        self.valid_dim_x = valid_dim_x
        self.valid_dim_y = valid_dim_y
        self.nice_x = NICEBlock(dim, valid_dim_x, hidden_x)
        self.nice_y = NICEBlock(dim, valid_dim_y, hidden_y)

    def x2y(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.nice_x(x, reverse=False)
        y_pred = self.nice_y(z, reverse=True)
        return y_pred, z

    def y2x(self, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.nice_y(y, reverse=False)
        x_pred = self.nice_x(z, reverse=True)
        return x_pred, z


# -------------------------
# Hard-Concrete (L0) Output Gates
# -------------------------

class HardConcreteGate(nn.Module):
    """
    Feature-wise hard-concrete gates (Louizos et al., 2018).
    """
    def __init__(self, valid_dim: int,
                 droprate_init: float = 0.2,
                 temperature: float = 1.0 / 5.0,
                 limit_a: float = -1.0,
                 limit_b: float = 2.0):
        super().__init__()
        self.valid_dim = valid_dim
        self.temperature = temperature
        self.limit_a = limit_a
        self.limit_b = limit_b
        self.eps = 1e-6

        init_mean = math.log(1 - droprate_init) - math.log(droprate_init)
        self.qz_loga = nn.Parameter(torch.empty(valid_dim).normal_(init_mean, 1e-2))

        self._always_on = False
        self._hard_mask: Optional[torch.Tensor] = None  # [valid_dim] 0/1 when hardened

    def reset_parameters(self):
        with torch.no_grad():
            init_mean = self.qz_loga.mean().item()
            self.qz_loga.normal_(init_mean, 1e-2)
        self._hard_mask = None
        self._always_on = False

    def set_all_open(self, flag: bool = True):
        self._always_on = flag

    def harden(self, threshold: float = 0.5):
        with torch.no_grad():
            probs = torch.sigmoid(self.qz_loga)
            self._hard_mask = (probs > threshold).float()

    def get_gate_probs(self) -> torch.Tensor:
        return torch.sigmoid(self.qz_loga)

    def _sample_stretched(self, batch: int) -> torch.Tensor:
        u = torch.empty(batch, self.valid_dim, device=self.qz_loga.device).uniform_(self.eps, 1 - self.eps)
        s = torch.sigmoid((torch.log(u) - torch.log(1 - u) + self.qz_loga) / self.temperature)
        s = s * (self.limit_b - self.limit_a) + self.limit_a
        return torch.clamp(s, 0.0, 1.0)

    def forward(self, batch_size: int, training: bool) -> torch.Tensor:
        if self._always_on:
            return torch.ones(batch_size, self.valid_dim, device=self.qz_loga.device)
        if self._hard_mask is not None:
            return self._hard_mask.view(1, -1).expand(batch_size, -1)
        if training:
            return self._sample_stretched(batch_size)
        # deterministic mean
        probs = torch.sigmoid(self.qz_loga)
        stretched = probs * (self.limit_b - self.limit_a) + self.limit_a
        return torch.clamp(stretched, 0.0, 1.0).view(1, -1).expand(batch_size, -1)


# -------------------------
# Full model: backbone + output gates
# -------------------------

class BFHDModel(nn.Module):
    def __init__(self, dim: int, valid_dim_x: int, valid_dim_y: int,
                 hidden_x: List[int], hidden_y: List[int],
                 gate_droprate_init: float = 0.2,
                 gate_temperature: float = 1.0 / 5.0):
        super().__init__()
        self.dim = dim
        self.valid_dim_x = valid_dim_x
        self.valid_dim_y = valid_dim_y

        self.backbone = NiceDualBackbone(dim, valid_dim_x, valid_dim_y, hidden_x, hidden_y)
        self.gate_x = HardConcreteGate(valid_dim_x, droprate_init=gate_droprate_init, temperature=gate_temperature)
        self.gate_y = HardConcreteGate(valid_dim_y, droprate_init=gate_droprate_init, temperature=gate_temperature)

    def forward_x2y(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        y_full, _ = self.backbone.x2y(x)
        gy = self.gate_y(batch_size=x.shape[0], training=self.training)
        y_pred = y_full[:, :self.valid_dim_y]
        return y_pred, gy, y_full

    def forward_y2x(self, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x_full, _ = self.backbone.y2x(y)
        gx = self.gate_x(batch_size=y.shape[0], training=self.training)
        x_pred = x_full[:, :self.valid_dim_x]
        return x_pred, gx, x_full

    def harden_gates(self, threshold: float = 0.5, freeze: bool = True):
        self.gate_x.harden(threshold)
        self.gate_y.harden(threshold)
        if freeze:
            for p in self.gate_x.parameters():
                p.requires_grad = False
            for p in self.gate_y.parameters():
                p.requires_grad = False

    def gate_summary(self) -> Dict[str, torch.Tensor]:
        with torch.no_grad():
            px = self.gate_x.get_gate_probs()
            py = self.gate_y.get_gate_probs()
            hx = (px > 0.5).float()
            hy = (py > 0.5).float()
        return {"p_x": px, "p_y": py, "hard_x": hx, "hard_y": hy}


# -------------------------
# Losses
# -------------------------

def stage1_loss_nmse(model: BFHDModel, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Stage1: gates open, train backbone with plain NMSE (no hinge)."""
    model.gate_x.set_all_open(True)
    model.gate_y.set_all_open(True)
    y_pred, _, _ = model.forward_x2y(x)
    x_pred, _, _ = model.forward_y2x(y)
    nmse_y = featurewise_nmse(y[:, :model.valid_dim_y], y_pred).mean()
    nmse_x = featurewise_nmse(x[:, :model.valid_dim_x], x_pred).mean()
    return nmse_x + nmse_y


def hinge_gate_loss(y_true: torch.Tensor, y_pred: torch.Tensor,
                    g: torch.Tensor, eps: float, lam_close: float) -> torch.Tensor:
    """
    Per-feature objective:
      lam_close*(1-g_i) + g_i*ReLU(NMSE_i - eps)
    Use batch-mean gate per feature for stability.
    """
    g_bar = torch.mean(g, dim=0)  # [d]
    nmse = featurewise_nmse(y_true, y_pred)  # [d]
    hinge = F.relu(nmse - eps)
    return torch.mean(lam_close * (1.0 - g_bar) + g_bar * hinge)


def stage2_loss(model: BFHDModel, x: torch.Tensor, y: torch.Tensor,
                eps: float, lam_close: float) -> torch.Tensor:
    model.gate_x.set_all_open(False)
    model.gate_y.set_all_open(False)
    y_pred, gy, _ = model.forward_x2y(x)
    x_pred, gx, _ = model.forward_y2x(y)
    loss_y = hinge_gate_loss(y[:, :model.valid_dim_y], y_pred, gy, eps=eps, lam_close=lam_close)
    loss_x = hinge_gate_loss(x[:, :model.valid_dim_x], x_pred, gx, eps=eps, lam_close=lam_close)
    return loss_x + loss_y


def stage3_loss(model: BFHDModel, x: torch.Tensor, y: torch.Tensor, eps: float) -> torch.Tensor:
    """Stage3: hard gates fixed. Hinge only on selected dims (no lam_close)."""
    y_pred, gy, _ = model.forward_x2y(x)
    x_pred, gx, _ = model.forward_y2x(y)
    my = torch.mean(gy, dim=0)  # 0/1
    mx = torch.mean(gx, dim=0)  # 0/1
    nmse_y = featurewise_nmse(y[:, :model.valid_dim_y], y_pred)
    nmse_x = featurewise_nmse(x[:, :model.valid_dim_x], x_pred)
    loss_y = torch.sum(my * F.relu(nmse_y - eps)) / (my.sum() + 1e-8)
    loss_x = torch.sum(mx * F.relu(nmse_x - eps)) / (mx.sum() + 1e-8)
    return loss_x + loss_y


# -------------------------
# Early stopping
# -------------------------

@dataclass
class EarlyStopConfig:
    patience: int = 1000
    min_delta: float = 1e-6
    warmup: int = 0  # epochs before we start checking


class EarlyStopper:
    def __init__(self, cfg: EarlyStopConfig):
        self.cfg = cfg
        self.best = float("inf")
        self.best_epoch = -1
        self.counter = 0
        self.best_state: Optional[Dict[str, torch.Tensor]] = None

    def step(self, val_loss: float, model: nn.Module, epoch: int) -> bool:
        """Returns True if should stop."""
        if epoch < self.cfg.warmup:
            return False

        if val_loss < self.best - self.cfg.min_delta:
            self.best = val_loss
            self.best_epoch = epoch
            self.counter = 0
            # store CPU clone (safe across devices)
            self.best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1

        return self.counter >= self.cfg.patience

    def restore_best(self, model: nn.Module):
        if self.best_state is not None:
            model.load_state_dict(self.best_state)
        return self.best, self.best_epoch


# -------------------------
# Training config
# -------------------------

@dataclass
class TrainConfig:
    # architecture
    hidden_x: Tuple[int, ...] = 
    hidden_y: Tuple[int, ...] = 
    gate_droprate_init: float = 
    gate_temperature: float = 

    # epochs
    epochs_stage1: int = 
    epochs_stage2: int = 
    epochs_stage3: int = 

    # early stop
    patience_stage1: int = 
    patience_stage2: int =
    patience_stage3: int = 
    warmup_stage1: int = 
    warmup_stage2: int = 
    warmup_stage3: int = 
    min_delta: float =

    # learning rates
    lr_backbone: float = 
    lr_gate: float = 
    weight_decay: float =

    stage2_lr_milestones: Tuple[Tuple[int, float, float], ...] = 

    # loss hyperparams
    lam_close: float = 
    eps_start: float = 
    eps_target: float = 
    gate_hard_threshold: float = 

    # evaluation
    ccc_threshold: float = 


def linear_anneal(start: float, end: float, t: int, T: int) -> float:
    if T <= 1:
        return end
    alpha = t / (T - 1)
    return max(float(start + alpha * (end - start)),end)


def make_optimizers(model: BFHDModel, lr_backbone: float, lr_gate: float, weight_decay: float):
    gate_params = list(model.gate_x.parameters()) + list(model.gate_y.parameters())
    inn_params = [p for n, p in model.named_parameters() if ('gate_' not in n and 'gate.' not in n)]
    # safer split: use module attributes
    inn_params = list(model.backbone.parameters())
    opt = torch.optim.Adam([
        {'params': inn_params, 'lr': lr_backbone},
        {'params': gate_params, 'lr': lr_gate},
    ], weight_decay=weight_decay)
    return opt


def to_tensors_with_padding(
    X: np.ndarray, Y: np.ndarray,
    error_dim_num: int,
    device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, int, int, int]:
    """
    Pads X,Y to common target_dim then appends error_dim_num zeros.
    Returns (X_tensor, Y_tensor, x_dim, y_dim, full_dim)
    """
    x_dim = X.shape[1]
    y_dim = Y.shape[1]
    target_dim = max(x_dim, y_dim)
    full_dim = target_dim + error_dim_num

    Xp = pad_and_add_error_dim_np(X, x_dim, target_dim, error_dim_num)
    Yp = pad_and_add_error_dim_np(Y, y_dim, target_dim, error_dim_num)

    return (
        torch.tensor(Xp, dtype=torch.float32, device=device),
        torch.tensor(Yp, dtype=torch.float32, device=device),
        x_dim, y_dim, full_dim
    )



# -------------------------
# Training loop (three stages + early stop + restore best)
# -------------------------

def train_bfhd(
    X_train: torch.Tensor, Y_train: torch.Tensor,
    X_val: Optional[torch.Tensor], Y_val: Optional[torch.Tensor],
    valid_dim_x: int, valid_dim_y: int,
    cfg: TrainConfig,
    device: Optional[str] = None,
    log_every: int = 100,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)

    D = X_train.shape[1]
    model = BFHDModel(
        dim=D,
        valid_dim_x=valid_dim_x,
        valid_dim_y=valid_dim_y,
        hidden_x=list(cfg.hidden_x),
        hidden_y=list(cfg.hidden_y),
        gate_droprate_init=cfg.gate_droprate_init,
        gate_temperature=cfg.gate_temperature,
    ).to(dev)

    X_train = X_train.to(dev)
    Y_train = Y_train.to(dev)
    if X_val is not None:
        X_val = X_val.to(dev)
        Y_val = Y_val.to(dev)

    hist: Dict[str, List[float]] = {
        "stage1_train_loss": [], "stage1_val_loss": [],
        "stage2_train_loss": [], "stage2_val_loss": [], "stage2_eps": [],
        "stage3_train_loss": [], "stage3_val_loss": [],
    }

    # ============ Stage 1 ============
    print("\n====== Stage 1: backbone pretrain, gates all open ======")
    model.train()
    model.gate_x.set_all_open(True)
    model.gate_y.set_all_open(True)
    for p in model.gate_x.parameters():
        p.requires_grad = False
    for p in model.gate_y.parameters():
        p.requires_grad = False

    opt1 = torch.optim.Adam(model.backbone.parameters(), lr=cfg.lr_backbone, weight_decay=cfg.weight_decay)
    es1 = EarlyStopper(EarlyStopConfig(
        patience=cfg.patience_stage1, min_delta=cfg.min_delta, warmup=cfg.warmup_stage1
    ))

    for epoch in range(cfg.epochs_stage1):
        model.train()
        opt1.zero_grad(set_to_none=True)
        loss = stage1_loss_nmse(model, X_train, Y_train)
        loss.backward()
        opt1.step()
        hist["stage1_train_loss"].append(float(loss.item()))

        # validation
        if X_val is not None:
            model.eval()
            with torch.no_grad():
                vloss = stage1_loss_nmse(model, X_val, Y_val).item()
            hist["stage1_val_loss"].append(float(vloss))
            stop = es1.step(vloss, model, epoch)
        else:
            stop = False

        if log_every and (epoch + 1) % log_every == 0:
            msg = f"[S1 {epoch+1}/{cfg.epochs_stage1}] train={loss.item():.6f}"
            if X_val is not None:
                msg += f" val={vloss:.6f}"
            print(msg)

        if stop:
            best, best_ep = es1.restore_best(model)
            print(f"[S1] Early stop at {epoch}. Restored best @ {best_ep} val={best:.6f}")
            break

    if X_val is not None:
        best, best_ep = es1.restore_best(model)
        print(f"[S1] Done. Loaded best @ {best_ep} val={best:.6f}")
    
    ST1_result = copy.deepcopy(model)
    # ============ Stage 2 ============
    print("\n====== Stage 2: joint train backbone + gates (hinge NMSE-eps), eps anneal ======")
    # enable gates
    model.gate_x.set_all_open(False)
    model.gate_y.set_all_open(False)
    for p in model.gate_x.parameters():
        p.requires_grad = True
    for p in model.gate_y.parameters():
        p.requires_grad = True

    opt2 = make_optimizers(model, cfg.lr_backbone, cfg.lr_gate, cfg.weight_decay)
    es2 = EarlyStopper(EarlyStopConfig(
        patience=cfg.patience_stage2, min_delta=cfg.min_delta, warmup=cfg.warmup_stage2
    ))

    milestones = sorted(list(cfg.stage2_lr_milestones), key=lambda t: t[0])
    next_ms = 0

    for epoch in range(cfg.epochs_stage2):

        if next_ms < len(milestones) and epoch == milestones[next_ms][0]:
            _, new_lr_inn, new_lr_gate = milestones[next_ms]
            opt2 = make_optimizers(model, new_lr_inn, new_lr_gate, cfg.weight_decay)
            next_ms += 1

        eps_t = linear_anneal(cfg.eps_start, cfg.eps_target, epoch, cfg.warmup_stage2)
        hist["stage2_eps"].append(float(eps_t))

        model.train()
        opt2.zero_grad(set_to_none=True)
        loss = stage2_loss(model, X_train, Y_train, eps=eps_t, lam_close=cfg.lam_close)
        loss.backward()
        opt2.step()
        hist["stage2_train_loss"].append(float(loss.item()))

        if X_val is not None:
            model.eval()
            with torch.no_grad():
                vloss = stage2_loss(model, X_val, Y_val, eps=eps_t, lam_close=cfg.lam_close).item()
            hist["stage2_val_loss"].append(float(vloss))
            stop = es2.step(vloss, model, epoch)
        else:
            stop = False

        if log_every and (epoch + 1) % log_every == 0:
            gs = model.gate_summary()
            num_x = int(gs["hard_x"].sum().item())
            num_y = int(gs["hard_y"].sum().item())
            msg = f"[S2 {epoch+1}/{cfg.epochs_stage2}] eps={eps_t:.4f} train={loss.item():.6f} sel~=({num_x},{num_y})"
            if X_val is not None:
                msg += f" val={vloss:.6f}"
            print(msg)

        if stop:
            best, best_ep = es2.restore_best(model)
            print(f"[S2] Early stop at {epoch}. Restored best @ {best_ep} val={best:.6f}")
            break

    if X_val is not None:
        best, best_ep = es2.restore_best(model)
        print(f"[S2] Done. Loaded best @ {best_ep} val={best:.6f}")

    # ============ Stage 3 ============
    print("\n====== Stage 3: harden gates, freeze, finetune backbone ======")
    model.harden_gates(threshold=cfg.gate_hard_threshold, freeze=True)
    # only backbone params
    opt3 = torch.optim.Adam(model.backbone.parameters(), lr=max(cfg.lr_backbone * 0.1, 1e-6), weight_decay=cfg.weight_decay)
    es3 = EarlyStopper(EarlyStopConfig(
        patience=cfg.patience_stage3, min_delta=cfg.min_delta, warmup=cfg.warmup_stage3
    ))

    for epoch in range(cfg.epochs_stage3):
        model.train()
        opt3.zero_grad(set_to_none=True)
        loss = stage3_loss(model, X_train, Y_train, eps=cfg.eps_target)
        loss.backward()
        opt3.step()
        hist["stage3_train_loss"].append(float(loss.item()))

        if X_val is not None:
            model.eval()
            with torch.no_grad():
                vloss = stage3_loss(model, X_val, Y_val, eps=cfg.eps_target).item()
            hist["stage3_val_loss"].append(float(vloss))
            stop = es3.step(vloss, model, epoch)
        else:
            stop = False

        if log_every and (epoch + 1) % log_every == 0:
            msg = f"[S3 {epoch+1}/{cfg.epochs_stage3}] train={loss.item():.6f}"
            if X_val is not None:
                msg += f" val={vloss:.6f}"
            print(msg)

        if stop:
            best, best_ep = es3.restore_best(model)
            print(f"[S3] Early stop at {epoch}. Restored best @ {best_ep} val={best:.6f}")
            break

    if X_val is not None:
        best, best_ep = es3.restore_best(model)
        print(f"[S3] Done. Loaded best @ {best_ep} val={best:.6f}")

    return model, hist, ST1_result
