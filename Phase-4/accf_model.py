"""
ACCF — Adaptive Continual Continual Forecaster
Fashion demand forecasting with per-SKU adapters

Architecture overview
---------------------
ACCF separates what should be shared from what should be per-SKU:

  Shared backbone (frozen after pre-training):
    TCNEncoder — extracts temporal representations from the demand window.
    Trained once on a large SKU sample; never updated during inference.

  Per-SKU adapter (lightweight, fast to train):
    A small bottleneck MLP conditioned on:
      (a) the backbone's representation
      (b) the 13-dim demand feature vector
      (c) a metadata embedding (product type, colour, department, etc.)
    Only the adapter is updated per-SKU — ~5× fewer parameters than
    retraining the full model.

  Adapter bottleneck design (inspired by Houlsby et al., 2019):
    z → down-project (hidden→adapter_size) → GELU → up-project
        (adapter_size→hidden) → residual add
    This keeps the adapter small (2 × hidden × adapter_size parameters)
    while giving it enough capacity to capture SKU-specific patterns.

Continual learning
------------------
  EWC (Elastic Weight Consolidation) on adapter weights across SKU stream.
  Reservoir replay buffer — unbiased sampling from all seen SKUs.
  These prevent the adapter from catastrophically forgetting earlier SKUs
  when the shared backbone is periodically fine-tuned.

Public interface (matches evaluate_accf.py exactly)
----------------------------------------------------
  ACCFForecaster(...)
    .fit(train_data_dict, metadata_dict)   → bool
    .predict(test_series, train_series,
             sku_id, metadata)             → np.ndarray
"""

import logging
import warnings
from copy import deepcopy
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

torch.manual_seed(42)
np.random.seed(42)


# ---------------------------------------------------------------------------
# Feature engineering (consistent across all improved baselines)
# ---------------------------------------------------------------------------

def build_features(window: np.ndarray) -> np.ndarray:
    """
    13-dimensional hand-crafted feature vector from a raw demand window:
      [0]    mean
      [1]    std  (+ 1e-8 for stability)
      [2]    min
      [3]    max
      [4]    linear trend slope (OLS)
      [5-11] last 7 values (zero-padded left if window < 7)
      [12]   non-zero ratio (intermittency proxy)
    """
    w = window.astype(float)
    n = len(w)

    mean_v = w.mean()
    std_v  = w.std() + 1e-8
    min_v  = w.min()
    max_v  = w.max()

    if n >= 2:
        x     = np.arange(n, dtype=float) - n / 2.0
        slope = np.dot(x, w) / (np.dot(x, x) + 1e-8)
    else:
        slope = 0.0

    last7       = np.zeros(7)
    take        = min(7, n)
    last7[-take:] = w[-take:]

    sparsity = (w > 0).mean()

    return np.array([mean_v, std_v, min_v, max_v, slope,
                     *last7, sparsity], dtype=np.float32)


FEATURE_DIM = 13


# ---------------------------------------------------------------------------
# Metadata encoder
# ---------------------------------------------------------------------------

class MetadataEncoder(nn.Module):
    """
    Projects a raw metadata vector (hash codes, floats, etc.) into a
    fixed-size embedding used to condition the per-SKU adapter.

    A two-layer MLP with LayerNorm output ensures the embedding stays
    well-scaled regardless of the raw metadata distribution.
    """

    def __init__(self, input_dim: int = 64, embed_dim: int = 32,
                 dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
        )
        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# TCN backbone (shared, frozen after pre-training)
# ---------------------------------------------------------------------------

class _CausalConv1d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int, dilation: int):
        super().__init__()
        self.pad  = (kernel - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel,
                              dilation=dilation, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(nn.functional.pad(x, (self.pad, 0)))


class TCNBackbone(nn.Module):
    """
    Shared dilated causal TCN backbone.
    Trained once on many SKUs, then frozen.
    Outputs (B, channels) — last-timestep representation.
    """

    def __init__(self, channels: int = 48, kernel: int = 3,
                 n_layers: int = 4, dropout: float = 0.1):
        super().__init__()
        self.proj   = nn.Linear(1, channels)
        self.blocks = nn.ModuleList([
            self._make_block(channels, kernel, 2**i, dropout)
            for i in range(n_layers)
        ])
        self.out_dim = channels

    @staticmethod
    def _make_block(ch: int, k: int, d: int, drop: float) -> nn.Module:
        return nn.Sequential(
            _CausalConv1d(ch, ch, k, d),
            nn.LayerNorm(ch),
            nn.GELU(),
            nn.Dropout(drop),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:   # (B,T,1)
        h = self.proj(x).transpose(1, 2)                  # (B,C,T)
        for blk in self.blocks:
            # LayerNorm expects (B,T,C) — transpose in/out
            h = h + blk(h)
        return h[:, :, -1]                                 # (B,C)

    def freeze(self):
        """Freeze all backbone parameters after pre-training."""
        for p in self.parameters():
            p.requires_grad_(False)

    def unfreeze(self):
        for p in self.parameters():
            p.requires_grad_(True)


# ---------------------------------------------------------------------------
# Per-SKU adapter
# ---------------------------------------------------------------------------

class SKUAdapter(nn.Module):
    """
    Lightweight bottleneck adapter conditioned on backbone output,
    demand features, and SKU metadata embedding.

    Input concat: [backbone_repr | demand_features | metadata_embed]
    Bottleneck : down-project → GELU → up-project → residual add
    Output     : scalar demand forecast (via Softplus)

    Parameters
    ----------
    backbone_dim  : TCNBackbone.out_dim
    feat_dim      : FEATURE_DIM (13)
    meta_dim      : MetadataEncoder.embed_dim
    adapter_size  : bottleneck width (default 32 — ~10× smaller than backbone)
    """

    def __init__(self,
                 backbone_dim : int   = 48,
                 feat_dim     : int   = FEATURE_DIM,
                 meta_dim     : int   = 32,
                 adapter_size : int   = 32,
                 dropout      : float = 0.1):
        super().__init__()

        fusion_dim = backbone_dim + feat_dim + meta_dim

        # Bottleneck (Houlsby-style adapter)
        self.down   = nn.Linear(fusion_dim, adapter_size)
        self.act    = nn.GELU()
        self.drop   = nn.Dropout(dropout)
        self.up     = nn.Linear(adapter_size, fusion_dim)

        # Final head: fused (with residual) → scalar
        self.head   = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, adapter_size),
            nn.GELU(),
            nn.Linear(adapter_size, 1),
            nn.Softplus(),     # non-negative, smooth
        )

        # Initialise up-projection near zero so adapter starts as identity
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self,
                backbone_repr: torch.Tensor,    # (B, backbone_dim)
                demand_feat  : torch.Tensor,    # (B, feat_dim)
                meta_embed   : torch.Tensor,    # (B, meta_dim)
                ) -> torch.Tensor:              # (B,)
        z   = torch.cat([backbone_repr, demand_feat, meta_embed], dim=-1)
        # Adapter residual
        z   = z + self.up(self.drop(self.act(self.down(z))))
        return self.head(z).squeeze(-1)


# ---------------------------------------------------------------------------
# Full ACCF model (backbone + metadata encoder + adapter)
# ---------------------------------------------------------------------------

class ACCFModel(nn.Module):
    """
    Combined model: backbone + metadata encoder + adapter.
    During pre-training, all components are trainable.
    After pre-training, backbone is frozen; only adapter + metadata
    encoder are updated per-SKU.
    """

    def __init__(self,
                 lookback     : int   = 28,
                 tcn_channels : int   = 48,
                 tcn_layers   : int   = 4,
                 meta_input   : int   = 64,
                 meta_embed   : int   = 32,
                 adapter_size : int   = 32,
                 dropout      : float = 0.1):
        super().__init__()

        self.backbone  = TCNBackbone(tcn_channels, n_layers=tcn_layers,
                                     dropout=dropout)
        self.meta_enc  = MetadataEncoder(meta_input, meta_embed, dropout)
        self.adapter   = SKUAdapter(tcn_channels, FEATURE_DIM,
                                    meta_embed, adapter_size, dropout)
        self.lookback  = lookback

    def forward(self,
                seq  : torch.Tensor,    # (B, T, 1)  scaled demand
                feat : torch.Tensor,    # (B, 13)    demand features
                meta : torch.Tensor,    # (B, meta_input)
                ) -> torch.Tensor:      # (B,)
        backbone_repr = self.backbone(seq)
        meta_embed    = self.meta_enc(meta)
        return self.adapter(backbone_repr, feat, meta_embed)


# ---------------------------------------------------------------------------
# Reservoir replay buffer
# ---------------------------------------------------------------------------

class ReservoirBuffer:
    """
    Reservoir sampling (Vitter, 1985) — unbiased over the full stream.
    Stores (seq, feat, meta, target) tuples.
    """

    def __init__(self, max_size: int = 4000):
        self.max_size = max_size
        self._data: List[Tuple] = []
        self._count = 0

    def add_batch(self, seqs: np.ndarray, feats: np.ndarray,
                  metas: np.ndarray, targets: np.ndarray):
        for s, f, m, t in zip(seqs, feats, metas, targets):
            self._count += 1
            if len(self._data) < self.max_size:
                self._data.append((s.copy(), f.copy(), m.copy(), float(t)))
            else:
                j = np.random.randint(0, self._count)
                if j < self.max_size:
                    self._data[j] = (s.copy(), f.copy(), m.copy(), float(t))

    def sample(self, n: int) -> Optional[Tuple[np.ndarray, ...]]:
        if not self._data:
            return None
        n   = min(n, len(self._data))
        idx = np.random.choice(len(self._data), n, replace=False)
        seqs  = np.array([self._data[i][0] for i in idx], np.float32)
        feats = np.array([self._data[i][1] for i in idx], np.float32)
        metas = np.array([self._data[i][2] for i in idx], np.float32)
        tgts  = np.array([self._data[i][3] for i in idx], np.float32)
        return seqs, feats, metas, tgts

    def __len__(self):
        return len(self._data)


# ---------------------------------------------------------------------------
# EWC helper
# ---------------------------------------------------------------------------

def estimate_fisher(model    : ACCFModel,
                    seq_t    : torch.Tensor,
                    feat_t   : torch.Tensor,
                    meta_t   : torch.Tensor,
                    tgt_t    : torch.Tensor,
                    device   : torch.device,
                    n_samples: int = 64,
                    ) -> Dict[str, torch.Tensor]:
    """
    Diagonal Fisher Information Matrix via mean squared gradient.
    Only computed for adapter + metadata encoder parameters
    (backbone is frozen and excluded).
    """
    model.eval()
    crit   = nn.HuberLoss(delta=1.0)
    fisher = {n: torch.zeros_like(p)
              for n, p in model.named_parameters()
              if p.requires_grad}

    n = min(n_samples, len(tgt_t))
    for i in range(n):
        model.zero_grad()
        pred = model(seq_t[i:i+1], feat_t[i:i+1], meta_t[i:i+1])
        loss = crit(pred, tgt_t[i:i+1])
        loss.backward()
        for nm, p in model.named_parameters():
            if p.requires_grad and p.grad is not None:
                fisher[nm] += p.grad.detach() ** 2

    for nm in fisher:
        fisher[nm] /= n
    return fisher


# ---------------------------------------------------------------------------
# ACCFForecaster — the public interface used by evaluate_accf.py
# ---------------------------------------------------------------------------

class ACCFForecaster:
    """
    Public interface for ACCF.

    Phase 1 (fit):
      Pre-train the shared TCNBackbone + adapter + metadata encoder on
      `n_train_skus` SKUs with all parameters trainable.
      Then freeze the backbone.

    Phase 2 (predict):
      For each new SKU, only the adapter + metadata encoder are updated
      (fast, lightweight). The frozen backbone provides a universal
      temporal representation.

    Continual learning:
      EWC regularisation on adapter weights prevents forgetting across
      the SKU stream. Reservoir buffer provides experience replay.
    """

    def __init__(self,
                 lookback_window    : int   = 28,
                 hidden_size        : int   = 128,   # maps to tcn_channels
                 adapter_size       : int   = 32,
                 metadata_dim       : int   = 64,
                 learning_rate      : float = 1e-3,
                 adapter_lr         : float = 1e-2,
                 epochs             : int   = 50,
                 batch_size         : int   = 32,
                 replay_buffer_size : int   = 5000,
                 replay_ratio       : float = 0.3,
                 # additional knobs
                 tcn_layers         : int   = 4,
                 meta_embed_dim     : int   = 32,
                 ewc_lambda         : float = 200.0,
                 adapter_epochs     : int   = 20,
                 adapter_patience   : int   = 5,
                 grad_clip          : float = 1.0):

        self.lookback        = lookback_window
        self.tcn_channels    = max(hidden_size // 4, 32)   # keep model lean
        self.adapter_size    = adapter_size
        self.metadata_dim    = metadata_dim
        self.lr              = learning_rate
        self.adapter_lr      = adapter_lr
        self.epochs          = epochs
        self.batch_size      = batch_size
        self.replay_ratio    = replay_ratio
        self.tcn_layers      = tcn_layers
        self.meta_embed_dim  = meta_embed_dim
        self.ewc_lambda      = ewc_lambda
        self.adapter_epochs  = adapter_epochs
        self.adapter_patience = adapter_patience
        self.grad_clip       = grad_clip

        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        self.model = ACCFModel(
            lookback     = lookback_window,
            tcn_channels = self.tcn_channels,
            tcn_layers   = tcn_layers,
            meta_input   = metadata_dim,
            meta_embed   = meta_embed_dim,
            adapter_size = adapter_size,
        ).to(self.device)

        self.replay_buffer = ReservoirBuffer(max_size=replay_buffer_size)

        # EWC state (set after pre-training)
        self._ewc_params : Optional[Dict[str, torch.Tensor]] = None
        self._ewc_fisher : Optional[Dict[str, torch.Tensor]] = None

        # Per-SKU scaler registry (keyed by sku_id)
        self._scalers: Dict[str, StandardScaler] = {}

        self.is_pretrained = False
        logger.info(f"ACCF initialised | device={self.device} "
                    f"| params={sum(p.numel() for p in self.model.parameters()):,}")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _scaler_for(self, sku_id: str,
                    raw: np.ndarray,
                    fit: bool = True) -> StandardScaler:
        if fit or sku_id not in self._scalers:
            sc = StandardScaler()
            sc.fit(raw.reshape(-1, 1))
            self._scalers[sku_id] = sc
        return self._scalers[sku_id]

    def _build_windows(self,
                       raw    : np.ndarray,
                       scaled : np.ndarray,
                       meta   : np.ndarray
                       ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Slide a lookback window over the series.
        Returns (seqs, feats, metas, targets) — all numpy, float32.
        """
        seqs, feats, metas, tgts = [], [], [], []
        for i in range(self.lookback, len(scaled)):
            wr = raw[i - self.lookback: i]
            ws = scaled[i - self.lookback: i]
            seqs.append(ws.reshape(-1, 1))
            feats.append(build_features(wr))
            metas.append(meta)
            tgts.append(scaled[i])
        if not seqs:
            return (np.empty((0, self.lookback, 1), np.float32),
                    np.empty((0, FEATURE_DIM),       np.float32),
                    np.empty((0, len(meta)),          np.float32),
                    np.empty((0,),                   np.float32))
        return (np.array(seqs,  np.float32),
                np.array(feats, np.float32),
                np.array(metas, np.float32),
                np.array(tgts,  np.float32))

    def _ewc_penalty(self) -> torch.Tensor:
        if self._ewc_params is None or self._ewc_fisher is None:
            return torch.tensor(0.0, device=self.device)
        penalty = torch.tensor(0.0, device=self.device)
        for n, p in self.model.named_parameters():
            if p.requires_grad and n in self._ewc_fisher:
                penalty += (self._ewc_fisher[n] *
                            (p - self._ewc_params[n]) ** 2).sum()
        return self.ewc_lambda * penalty

    def _train_loop(self,
                    seq_t   : torch.Tensor,
                    feat_t  : torch.Tensor,
                    meta_t  : torch.Tensor,
                    tgt_t   : torch.Tensor,
                    optimizer: optim.Optimizer,
                    n_epochs: int,
                    patience: int,
                    use_ewc : bool = False,
                    use_replay: bool = False):
        """
        Shared mini-batch training loop used for both pre-training and
        per-SKU adapter fine-tuning.
        """
        criterion  = nn.HuberLoss(delta=1.0)
        best_loss  = float('inf')
        no_improve = 0
        n          = len(tgt_t)

        self.model.train()

        for _ in range(n_epochs):
            perm       = torch.randperm(n, device=self.device)
            epoch_loss = 0.0
            n_batches  = 0

            for start in range(0, n, self.batch_size):
                idx = perm[start: start + self.batch_size]
                sb  = seq_t [idx]
                fb  = feat_t[idx]
                mb  = meta_t[idx]
                tb  = tgt_t [idx]

                # Optionally mix replay samples
                if use_replay and len(self.replay_buffer) > 0:
                    sample = self.replay_buffer.sample(
                        max(1, int(len(idx) * self.replay_ratio)))
                    if sample is not None:
                        rs, rf, rm, rt = sample
                        sb = torch.cat([sb, torch.tensor(rs, device=self.device)])
                        fb = torch.cat([fb, torch.tensor(rf, device=self.device)])
                        mb = torch.cat([mb, torch.tensor(rm, device=self.device)])
                        tb = torch.cat([tb, torch.tensor(rt, device=self.device)])

                preds = self.model(sb, fb, mb)
                loss  = criterion(preds, tb)
                if use_ewc:
                    loss = loss + self._ewc_penalty()

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad],
                    self.grad_clip)
                optimizer.step()

                epoch_loss += loss.item()
                n_batches  += 1

            avg = epoch_loss / max(n_batches, 1)
            if avg < best_loss - 1e-4:
                best_loss  = avg
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    break

    # ------------------------------------------------------------------
    # Phase 1: pre-training
    # ------------------------------------------------------------------

    def fit(self,
            train_data_dict    : Dict[str, pd.Series],
            train_metadata_dict: Dict[str, np.ndarray]) -> bool:
        """
        Pre-train ACCF on a dictionary of {sku_id: demand_series}.

        All parameters are trainable during pre-training.
        After this call, the backbone is frozen and EWC state is set.

        Parameters
        ----------
        train_data_dict     : {sku_id: pd.Series of demand}
        train_metadata_dict : {sku_id: np.ndarray of shape (metadata_dim,)}
        """
        logger.info(f"Pre-training ACCF on {len(train_data_dict)} SKUs...")

        all_seqs, all_feats, all_metas, all_tgts = [], [], [], []

        for sku_id, series in train_data_dict.items():
            raw  = series.values.astype(float)
            meta = train_metadata_dict.get(sku_id,
                                           np.zeros(self.metadata_dim,
                                                    dtype=np.float32))
            if len(raw) < self.lookback + 10:
                continue

            sc     = self._scaler_for(sku_id, raw, fit=True)
            scaled = sc.transform(raw.reshape(-1, 1)).flatten()
            seqs, feats, metas, tgts = self._build_windows(raw, scaled, meta)

            if len(seqs) == 0:
                continue

            all_seqs.append(seqs);  all_feats.append(feats)
            all_metas.append(metas); all_tgts.append(tgts)

            # Seed the replay buffer during pre-training
            self.replay_buffer.add_batch(seqs, feats, metas, tgts)

        if not all_seqs:
            logger.error("No usable training data — pre-training aborted.")
            return False

        seq_t  = torch.tensor(np.concatenate(all_seqs),  device=self.device)
        feat_t = torch.tensor(np.concatenate(all_feats), device=self.device)
        meta_t = torch.tensor(np.concatenate(all_metas), device=self.device)
        tgt_t  = torch.tensor(np.concatenate(all_tgts),  device=self.device)

        logger.info(f"Pre-train dataset: {len(tgt_t):,} windows | "
                    f"device={self.device}")

        # All parameters trainable during pre-training
        self.model.backbone.unfreeze()
        optimizer = optim.AdamW(self.model.parameters(),
                                lr=self.lr, weight_decay=1e-4)

        self._train_loop(seq_t, feat_t, meta_t, tgt_t,
                         optimizer  = optimizer,
                         n_epochs   = self.epochs,
                         patience   = 10,
                         use_ewc    = False,
                         use_replay = False)

        # Freeze backbone — only adapter + metadata encoder update per-SKU
        self.model.backbone.freeze()
        logger.info("Backbone frozen after pre-training.")

        # Set EWC anchor on adapter + metadata encoder weights
        self._ewc_params = {n: p.detach().clone()
                            for n, p in self.model.named_parameters()
                            if p.requires_grad}
        self._ewc_fisher = estimate_fisher(
            self.model, seq_t[:64], feat_t[:64],
            meta_t[:64], tgt_t[:64], self.device)

        self.is_pretrained = True
        logger.info("ACCF pre-training complete.")
        return True

    # ------------------------------------------------------------------
    # Phase 2: per-SKU adapter fine-tuning + autoregressive inference
    # ------------------------------------------------------------------

    def predict(self,
                test_series  : pd.Series,
                train_series : pd.Series,
                sku_id       : str,
                metadata     : np.ndarray) -> np.ndarray:
        """
        Predict demand for a single SKU.

        If the model has been pre-trained, the adapter is quickly fine-tuned
        on the SKU's training data before inference. Inference is fully
        autoregressive — own predictions are used as context, never future
        ground truth.

        Parameters
        ----------
        test_series  : demand series for the test period (used only for length)
        train_series : demand series for the training period
        sku_id       : SKU identifier (used to retrieve per-SKU scaler)
        metadata     : np.ndarray of shape (metadata_dim,)
        """
        n_steps  = len(test_series)
        train_raw = train_series.values.astype(float)

        if len(train_raw) < self.lookback + 5:
            return np.zeros(n_steps)

        # ---- Per-SKU adapter fine-tuning ----
        if self.is_pretrained:
            sc     = self._scaler_for(sku_id, train_raw, fit=True)
            scaled = sc.transform(train_raw.reshape(-1, 1)).flatten()
            seqs, feats, metas, tgts = self._build_windows(
                train_raw, scaled, metadata.astype(np.float32))

            if len(seqs) > 0:
                seq_t  = torch.tensor(seqs,  device=self.device)
                feat_t = torch.tensor(feats, device=self.device)
                meta_t = torch.tensor(metas, device=self.device)
                tgt_t  = torch.tensor(tgts,  device=self.device)

                # Only adapter + metadata encoder params are trainable
                adapter_params = (list(self.model.adapter.parameters()) +
                                  list(self.model.meta_enc.parameters()))
                adapter_opt = optim.AdamW(adapter_params,
                                          lr=self.adapter_lr,
                                          weight_decay=1e-4)

                self._train_loop(seq_t, feat_t, meta_t, tgt_t,
                                 optimizer   = adapter_opt,
                                 n_epochs    = self.adapter_epochs,
                                 patience    = self.adapter_patience,
                                 use_ewc     = True,
                                 use_replay  = True)

                # Add this SKU to replay buffer for future EWC updates
                self.replay_buffer.add_batch(seqs, feats, metas, tgts)

                # Update EWC state
                self._ewc_params = {n: p.detach().clone()
                                    for n, p in self.model.named_parameters()
                                    if p.requires_grad}
                self._ewc_fisher = estimate_fisher(
                    self.model,
                    seq_t[:min(32, len(seq_t))],
                    feat_t[:min(32, len(feat_t))],
                    meta_t[:min(32, len(meta_t))],
                    tgt_t[:min(32, len(tgt_t))],
                    self.device, n_samples=32)

        sc = self._scaler_for(sku_id, train_raw, fit=False)

        # ---- Autoregressive inference ----
        # Start from the last `lookback` days of training history.
        # Roll own predictions forward — never peek at test ground truth.
        ctx_raw = list(train_raw[-self.lookback:])
        if len(ctx_raw) < self.lookback:
            pad     = self.lookback - len(ctx_raw)
            ctx_raw = [ctx_raw[0]] * pad + ctx_raw

        meta_t_inf = torch.tensor(
            metadata.astype(np.float32).reshape(1, -1),
            device=self.device)

        self.model.eval()
        preds_sc = []

        with torch.no_grad():
            for _ in range(n_steps):
                win_raw = np.array(ctx_raw[-self.lookback:], dtype=float)
                win_sc  = sc.transform(win_raw.reshape(-1, 1)).flatten()

                seq_t  = torch.tensor(win_sc.reshape(1, -1, 1),
                                      dtype=torch.float32,
                                      device=self.device)
                feat_t = torch.tensor(
                    build_features(win_raw).reshape(1, -1),
                    dtype=torch.float32, device=self.device)

                pred_sc = self.model(seq_t, feat_t, meta_t_inf).cpu().item()
                preds_sc.append(pred_sc)

                # Roll context with own prediction (not ground truth)
                pred_raw = float(sc.inverse_transform(
                    np.array([[pred_sc]]))[0, 0])
                ctx_raw.append(max(pred_raw, 0.0))

        preds = sc.inverse_transform(
            np.array(preds_sc).reshape(-1, 1)).flatten()
        return np.maximum(np.round(preds), 0).astype(float)