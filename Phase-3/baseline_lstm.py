"""
LSTM Baseline Model for Fashion Demand Forecasting — IMPROVED
Deep learning approach using recurrent neural networks

Key improvements over the original:
  1. Additive attention over all LSTM timesteps
                                  — last-hidden-state only discards the full
                                    sequence context; attention learns which
                                    timesteps matter most per SKU
  2. Softplus output              — replaces ReLU (no dead-zone problem) for
                                    smooth non-negative demand
  3. HuberLoss (δ=1)              — robust to the demand spikes common in
                                    fashion retail, unlike MSELoss
  4. CosineAnnealingLR            — warm restarts help escape local minima on
                                    short, noisy fashion demand series
  5. Early stopping               — halts training when val-proxy loss stalls,
                                    preventing overfitting on sparse SKUs
  6. Gradient clipping            — standard practice for LSTM stability
  7. Shared pre-trained backbone  — one LSTM is pre-trained on a representative
                                    SKU sample then fine-tuned per-SKU; reduces
                                    cold-start error on short series
  8. Autoregressive multi-step    — inference rolls own predictions forward as
                                    context, not ground-truth test demand
                                    (original leaked future values)
  9. Per-SKU scaler               — no cross-SKU distribution contamination
  10. Rich feature vector         — 13-dim engineered features (mean, std, trend,
                                    last-7, sparsity) fed to a parallel MLP head
                                    fused with the LSTM attention context
"""

import json
import logging
import warnings
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import sys
sys.path.append('.')

from utils.data_loader import BaselineDataLoader
from utils.metrics import calculate_forecasting_metrics, aggregate_sku_metrics
from utils.visualization import plot_predictions, plot_error_distribution

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

torch.manual_seed(42)
np.random.seed(42)


# ---------------------------------------------------------------------------
# Shared feature engineering (consistent with ARIMA / CLeaR / FSNet improved)
# ---------------------------------------------------------------------------

def build_features(window: np.ndarray) -> np.ndarray:
    """
    13-dimensional hand-crafted feature vector from a raw demand window.
      [0]    mean
      [1]    std
      [2]    min
      [3]    max
      [4]    linear trend slope (OLS)
      [5-11] last 7 values  (zero-padded left if window < 7)
      [12]   non-zero ratio (sparsity / intermittency measure)
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

    last7 = np.zeros(7)
    take  = min(7, n)
    last7[-take:] = w[-take:]

    sparsity = (w > 0).mean()

    return np.array([mean_v, std_v, min_v, max_v, slope,
                     *last7, sparsity], dtype=np.float32)


FEATURE_DIM = 13


# ---------------------------------------------------------------------------
# PyTorch dataset
# ---------------------------------------------------------------------------

class TimeSeriesDataset(Dataset):
    """Sequences (T,1), features (F,), target scalar."""

    def __init__(self,
                 sequences : np.ndarray,   # (N, T, 1)
                 features  : np.ndarray,   # (N, F)
                 targets   : np.ndarray):  # (N,)
        self.seq  = torch.tensor(sequences, dtype=torch.float32)
        self.feat = torch.tensor(features,  dtype=torch.float32)
        self.tgt  = torch.tensor(targets,   dtype=torch.float32)

    def __len__(self):
        return len(self.tgt)

    def __getitem__(self, idx):
        return self.seq[idx], self.feat[idx], self.tgt[idx]


# ---------------------------------------------------------------------------
# Additive attention module
# ---------------------------------------------------------------------------

class AdditiveAttention(nn.Module):
    """
    Bahdanau-style additive attention over LSTM hidden states.

    Scores each timestep with a learnable query vector, applies softmax,
    and returns the weighted sum of all hidden states.

    This replaces the original `lstm_out[:, -1, :]` which discarded all
    intermediate temporal information.
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.query  = nn.Linear(hidden_size, hidden_size, bias=False)
        self.energy = nn.Linear(hidden_size, 1,           bias=False)

    def forward(self, lstm_out: torch.Tensor) -> torch.Tensor:
        """
        lstm_out : (B, T, H)
        returns  : (B, H)  — attention-weighted context vector
        """
        scores  = self.energy(torch.tanh(self.query(lstm_out)))  # (B,T,1)
        weights = torch.softmax(scores, dim=1)                   # (B,T,1)
        context = (weights * lstm_out).sum(dim=1)                # (B,H)
        return context


# ---------------------------------------------------------------------------
# LSTM model with attention + feature branch
# ---------------------------------------------------------------------------

class LSTMModel(nn.Module):
    """
    Improved LSTM forecaster:
      - Bidirectional LSTM encoder
      - Additive attention over all timesteps
      - Parallel MLP branch for engineered features
      - Fusion MLP head
      - Softplus output (non-negative, gradient everywhere)
    """

    def __init__(self,
                 hidden_size  : int   = 64,
                 num_layers   : int   = 2,
                 feat_dim     : int   = FEATURE_DIM,
                 dropout      : float = 0.2,
                 bidirectional: bool  = True):
        super().__init__()

        self.bidirectional = bidirectional
        dir_mult = 2 if bidirectional else 1

        # --- sequence encoder ---
        self.lstm = nn.LSTM(
            input_size  = 1,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            dropout     = dropout if num_layers > 1 else 0.0,
            batch_first = True,
            bidirectional = bidirectional,
        )
        lstm_out_dim = hidden_size * dir_mult

        # --- attention ---
        self.attention = AdditiveAttention(lstm_out_dim)

        # --- feature branch ---
        self.feat_net = nn.Sequential(
            nn.Linear(feat_dim, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
        )
        feat_out_dim = hidden_size // 2

        # --- fusion head ---
        fusion_in = lstm_out_dim + feat_out_dim
        self.head = nn.Sequential(
            nn.LayerNorm(fusion_in),
            nn.Linear(fusion_in, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
            nn.Softplus(),       # non-negative, no dead zone
        )

    def forward(self,
                seq : torch.Tensor,   # (B, T, 1)
                feat: torch.Tensor    # (B, F)
                ) -> torch.Tensor:    # (B,)
        lstm_out, _ = self.lstm(seq)           # (B, T, H*dir)
        context     = self.attention(lstm_out) # (B, H*dir)
        feat_out    = self.feat_net(feat)      # (B, H//2)
        fused       = torch.cat([context, feat_out], dim=-1)
        return self.head(fused).squeeze(-1)    # (B,)


# ---------------------------------------------------------------------------
# Per-SKU forecaster
# ---------------------------------------------------------------------------

class LSTMForecaster:
    """
    LSTM forecaster for a single SKU.

    Uses a shared pre-trained model as starting point (warm start),
    then fine-tunes on the SKU's own training data.

    Inference uses autoregressive decoding — own predictions are rolled
    forward as context, not ground-truth test values.
    """

    def __init__(self,
                 model       : LSTMModel,
                 device      : torch.device,
                 lookback    : int   = 28,
                 lr          : float = 5e-4,
                 epochs      : int   = 40,
                 batch_size  : int   = 32,
                 patience    : int   = 8,
                 grad_clip   : float = 1.0,
                 fine_tune   : bool  = True):

        # Deep copy so per-SKU fine-tuning doesn't corrupt the shared weights
        self.model      = deepcopy(model)
        self.device     = device
        self.lookback   = lookback
        self.lr         = lr
        self.epochs     = epochs
        self.batch_size = batch_size
        self.patience   = patience
        self.grad_clip  = grad_clip
        self.fine_tune  = fine_tune

        self.scaler    = StandardScaler()   # per-SKU
        self.is_fitted = False

    # ------------------------------------------------------------------

    def _build_dataset(self, raw: np.ndarray,
                       scaled: np.ndarray) -> TimeSeriesDataset:
        seqs, feats, tgts = [], [], []
        for i in range(self.lookback, len(scaled)):
            wr = raw[i - self.lookback: i]
            ws = scaled[i - self.lookback: i]
            seqs.append(ws.reshape(-1, 1))
            feats.append(build_features(wr))
            tgts.append(scaled[i])
        if not seqs:
            return None
        return TimeSeriesDataset(
            np.array(seqs,  np.float32),
            np.array(feats, np.float32),
            np.array(tgts,  np.float32))

    # ------------------------------------------------------------------

    def _train_on_dataset(self, dataset: TimeSeriesDataset,
                          n_epochs: int, lr: float):
        """Core training loop with cosine LR, early stopping, grad clip."""
        loader    = DataLoader(dataset, batch_size=self.batch_size,
                               shuffle=True, drop_last=False)
        optimizer = optim.AdamW(self.model.parameters(),
                                lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=n_epochs, eta_min=lr * 0.05)
        criterion = nn.HuberLoss(delta=1.0)

        best_loss  = float('inf')
        no_improve = 0
        best_state = None

        self.model.train()
        for _ in range(n_epochs):
            epoch_loss = 0.0
            for seq_b, feat_b, tgt_b in loader:
                seq_b  = seq_b .to(self.device)
                feat_b = feat_b.to(self.device)
                tgt_b  = tgt_b .to(self.device)

                preds = self.model(seq_b, feat_b)
                loss  = criterion(preds, tgt_b)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.grad_clip)
                optimizer.step()
                epoch_loss += loss.item()

            scheduler.step()
            avg = epoch_loss / max(len(loader), 1)

            if avg < best_loss - 1e-4:
                best_loss  = avg
                no_improve = 0
                best_state = {k: v.clone()
                              for k, v in self.model.state_dict().items()}
            else:
                no_improve += 1
                if no_improve >= self.patience:
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)

    # ------------------------------------------------------------------

    def fit(self, train_series: pd.Series) -> bool:
        raw = train_series.values.astype(float)
        if len(raw) < self.lookback + 10:
            return False

        self.scaler.fit(raw.reshape(-1, 1))
        scaled  = self.scaler.transform(raw.reshape(-1, 1)).flatten()
        dataset = self._build_dataset(raw, scaled)
        if dataset is None:
            return False

        self.model.to(self.device)
        if self.fine_tune:
            self._train_on_dataset(dataset, self.epochs, self.lr)
        else:
            self._train_on_dataset(dataset, self.epochs, self.lr)

        self.is_fitted = True
        return True

    # ------------------------------------------------------------------

    def predict(self,
                train_series: pd.Series,
                n_steps     : int) -> np.ndarray:
        """
        Autoregressive multi-step forecast.

        Starts with the last `lookback` days of training data as context,
        then rolls own predictions forward — never peeks at test ground truth.
        """
        if not self.is_fitted:
            return np.zeros(n_steps)

        train_raw = train_series.values.astype(float)

        # Rolling context buffer (raw scale for feature engineering,
        # scaled for LSTM input)
        ctx_raw = list(train_raw[-self.lookback:])
        if len(ctx_raw) < self.lookback:
            pad = self.lookback - len(ctx_raw)
            ctx_raw = [ctx_raw[0]] * pad + ctx_raw

        self.model.eval()
        preds_sc = []

        with torch.no_grad():
            for _ in range(n_steps):
                win_raw = np.array(ctx_raw[-self.lookback:], dtype=float)
                win_sc  = self.scaler.transform(
                    win_raw.reshape(-1, 1)).flatten()

                seq_t  = torch.tensor(win_sc.reshape(1, -1, 1),
                                      dtype=torch.float32,
                                      device=self.device)
                feat_t = torch.tensor(
                    build_features(win_raw).reshape(1, -1),
                    dtype=torch.float32, device=self.device)

                pred_sc = self.model(seq_t, feat_t).cpu().item()
                preds_sc.append(pred_sc)

                # Roll context forward with the prediction (raw scale)
                pred_raw = float(self.scaler.inverse_transform(
                    np.array([[pred_sc]]))[0, 0])
                ctx_raw.append(max(pred_raw, 0.0))

        preds = self.scaler.inverse_transform(
            np.array(preds_sc).reshape(-1, 1)).flatten()
        return np.maximum(np.round(preds), 0).astype(float)


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------

class LSTMBaselineExperiment:
    """
    Full LSTM experiment.

    Phase 1 — pre-training:
      The shared LSTMModel is trained on a sample of SKUs to warm-start
      the weights. This gives every subsequent SKU a better initialisation
      than random, which is especially important for short series.

    Phase 2 — per-SKU fine-tuning:
      Each SKU deep-copies the shared model and fine-tunes for a smaller
      number of epochs. Inference uses autoregressive decoding.
    """

    def __init__(self,
                 processed_data_dir : str   = '../phase2/processed_data',
                 results_dir        : str   = 'results/lstm',
                 n_skus             : int   = None,
                 # model
                 lookback           : int   = 28,
                 hidden_size        : int   = 64,
                 num_layers         : int   = 2,
                 dropout            : float = 0.2,
                 bidirectional      : bool  = True,
                 # pre-training
                 pretrain_n_skus    : int   = 100,
                 pretrain_epochs    : int   = 30,
                 pretrain_lr        : float = 1e-3,
                 # fine-tuning
                 finetune_epochs    : int   = 30,
                 finetune_lr        : float = 3e-4,
                 batch_size         : int   = 32,
                 patience           : int   = 8):

        self.data_loader      = BaselineDataLoader(processed_data_dir)
        self.results_dir      = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.n_skus           = n_skus
        self.lookback         = lookback
        self.pretrain_n_skus  = pretrain_n_skus
        self.pretrain_epochs  = pretrain_epochs
        self.pretrain_lr      = pretrain_lr
        self.finetune_epochs  = finetune_epochs
        self.finetune_lr      = finetune_lr
        self.batch_size       = batch_size
        self.patience         = patience

        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        # Shared model (pre-trained in phase 1)
        self.shared_model = LSTMModel(
            hidden_size   = hidden_size,
            num_layers    = num_layers,
            dropout       = dropout,
            bidirectional = bidirectional,
        ).to(self.device)

        n_params = sum(p.numel() for p in self.shared_model.parameters()
                       if p.requires_grad)

        self.config = {
            'lookback'       : lookback,
            'hidden_size'    : hidden_size,
            'num_layers'     : num_layers,
            'dropout'        : dropout,
            'bidirectional'  : bidirectional,
            'pretrain_n_skus': pretrain_n_skus,
            'pretrain_epochs': pretrain_epochs,
            'pretrain_lr'    : pretrain_lr,
            'finetune_epochs': finetune_epochs,
            'finetune_lr'    : finetune_lr,
            'batch_size'     : batch_size,
            'patience'       : patience,
        }

        logger.info(f"Initialised improved LSTM experiment")
        logger.info(f"Device  : {self.device}")
        logger.info(f"Params  : {n_params:,}")
        logger.info(f"Results : {self.results_dir}")

    # ------------------------------------------------------------------
    # Phase 1: pre-training on a representative SKU sample
    # ------------------------------------------------------------------

    def _pretrain(self, train_data: pd.DataFrame,
                  sample_skus: List[str]):
        """
        Aggregate sequences from multiple SKUs into one large dataset and
        train the shared model. This gives the model a good starting point
        for all fine-tuning runs.
        """
        logger.info(f"Pre-training on {len(sample_skus)} SKUs...")

        all_seqs, all_feats, all_tgts = [], [], []

        for sku in tqdm(sample_skus, desc="Building pre-train dataset"):
            tr = train_data[train_data['article_id'] == sku].sort_values('date')
            raw = tr['demand'].values.astype(float)
            if len(raw) < self.lookback + 10:
                continue
            # Use a temporary scaler per SKU to normalise before pooling
            sc     = StandardScaler()
            scaled = sc.fit_transform(raw.reshape(-1, 1)).flatten()
            for i in range(self.lookback, len(scaled)):
                wr = raw[i - self.lookback: i]
                ws = scaled[i - self.lookback: i]
                all_seqs.append(ws.reshape(-1, 1))
                all_feats.append(build_features(wr))
                all_tgts.append(scaled[i])

        if not all_seqs:
            logger.warning("Pre-train dataset is empty — skipping pre-training")
            return

        dataset   = TimeSeriesDataset(
            np.array(all_seqs,  np.float32),
            np.array(all_feats, np.float32),
            np.array(all_tgts,  np.float32))

        loader    = DataLoader(dataset, batch_size=self.batch_size,
                               shuffle=True, drop_last=True)
        optimizer = optim.AdamW(self.shared_model.parameters(),
                                lr=self.pretrain_lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.pretrain_epochs,
            eta_min=self.pretrain_lr * 0.05)
        criterion = nn.HuberLoss(delta=1.0)

        self.shared_model.train()
        for epoch in range(self.pretrain_epochs):
            epoch_loss = 0.0
            for seq_b, feat_b, tgt_b in loader:
                seq_b  = seq_b .to(self.device)
                feat_b = feat_b.to(self.device)
                tgt_b  = tgt_b .to(self.device)

                preds = self.shared_model(seq_b, feat_b)
                loss  = criterion(preds, tgt_b)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.shared_model.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item()

            scheduler.step()
            if (epoch + 1) % 10 == 0:
                logger.info(f"  Pre-train epoch {epoch+1}/{self.pretrain_epochs}"
                            f"  loss={epoch_loss/len(loader):.4f}")

        logger.info("Pre-training complete.")

    # ------------------------------------------------------------------
    # Phase 2: per-SKU evaluation
    # ------------------------------------------------------------------

    def run_experiment(self) -> Dict:
        logger.info("=" * 80)
        logger.info("STARTING IMPROVED LSTM BASELINE EXPERIMENT")
        logger.info("=" * 80)

        train_data, val_data, test_data = self.data_loader.get_train_test_split()
        final_train = pd.concat([train_data, val_data], ignore_index=True)

        skus = (self.data_loader.get_sample_skus(n=self.n_skus)
                if self.n_skus else self.data_loader.get_all_skus())

        logger.info(f"Total SKUs : {len(skus)}")

        # Phase 1 — pre-train on a sample
        pretrain_skus = skus[:self.pretrain_n_skus]
        self._pretrain(final_train, pretrain_skus)

        # Phase 2 — fine-tune + evaluate every SKU
        all_results  = []
        sku_metrics  = {}
        failed_skus  = []

        for sku in tqdm(skus, desc="LSTM — SKUs"):
            try:
                result = self._evaluate_sku(sku, final_train, test_data)
                if result is not None:
                    all_results.append(result)
                    sku_metrics[sku] = result['metrics']
                else:
                    failed_skus.append(sku)
            except Exception as e:
                logger.warning(f"SKU {sku} failed: {e}")
                failed_skus.append(sku)

        logger.info(f"Succeeded : {len(all_results)}")
        logger.info(f"Failed    : {len(failed_skus)}")

        aggregated = aggregate_sku_metrics(sku_metrics)
        self._save_results(all_results, aggregated)
        self._generate_visualizations(all_results)

        logger.info("=" * 80)
        logger.info("IMPROVED LSTM EXPERIMENT COMPLETE")
        logger.info("=" * 80)

        return aggregated

    # ------------------------------------------------------------------

    def _evaluate_sku(self,
                      sku_id    : str,
                      train_data: pd.DataFrame,
                      test_data : pd.DataFrame) -> Optional[Dict]:

        tr = train_data[train_data['article_id'] == sku_id].sort_values('date')
        te = test_data [test_data ['article_id'] == sku_id].sort_values('date')

        if len(te) == 0 or len(tr) < self.lookback + 10:
            return None

        forecaster = LSTMForecaster(
            model      = self.shared_model,   # warm-started copy
            device     = self.device,
            lookback   = self.lookback,
            lr         = self.finetune_lr,
            epochs     = self.finetune_epochs,
            batch_size = self.batch_size,
            patience   = self.patience,
        )

        ok = forecaster.fit(tr['demand'])
        if not ok:
            return None

        preds  = forecaster.predict(tr['demand'], n_steps=len(te))
        actual = te['demand'].values

        return {
            'sku_id'      : sku_id,
            'predictions' : preds.tolist(),
            'actual'      : actual.tolist(),
            'dates'       : te['date'].dt.strftime('%Y-%m-%d').tolist(),
            'metrics'     : calculate_forecasting_metrics(actual, preds),
            'model_fitted': ok,
        }

    # ------------------------------------------------------------------

    def _save_results(self, all_results: List[Dict], aggregated: Dict):
        summary = {
            'model'             : 'LSTM (improved)',
            'configuration'     : self.config,
            'n_skus_evaluated'  : len(all_results),
            'aggregated_metrics': aggregated,
            'timestamp'         : pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
        }
        with open(self.results_dir / 'results_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)

        pd.DataFrame([{
            'sku_id'      : r['sku_id'],
            'rmse'        : r['metrics']['rmse'],
            'mae'         : r['metrics']['mae'],
            'mape'        : r['metrics']['mape'],
            'smape'       : r['metrics']['smape'],
            'model_fitted': r['model_fitted'],
        } for r in all_results]).to_csv(
            self.results_dir / 'detailed_results.csv', index=False)

        with open(self.results_dir / 'sample_predictions.json', 'w') as f:
            json.dump(all_results[:10], f, indent=2)

        print("\n" + "=" * 80)
        print("IMPROVED LSTM BASELINE — RESULTS SUMMARY")
        print("=" * 80)
        print(f"Model   : Bidirectional LSTM + attention + feature MLP")
        print(f"Device  : {self.device}")
        print(f"SKUs    : {len(all_results)}")
        print(f"\nAggregated Metrics:")
        print(f"  Mean RMSE    : {aggregated['mean_rmse']:.4f}  (±{aggregated['std_rmse']:.4f})")
        print(f"  Mean MAE     : {aggregated['mean_mae']:.4f}  (±{aggregated['std_mae']:.4f})")
        print(f"  Mean MAPE    : {aggregated['mean_mape']:.2f}%")
        print(f"  Median RMSE  : {aggregated['median_rmse']:.4f}")
        print(f"  Median MAE   : {aggregated['median_mae']:.4f}")
        print("=" * 80 + "\n")

    def _generate_visualizations(self, all_results: List[Dict]):
        logger.info("Generating visualisations...")
        for i, r in enumerate(all_results[:5]):
            plot_predictions(
                dates     = pd.to_datetime(r['dates']),
                y_true    = np.array(r['actual']),
                y_pred    = np.array(r['predictions']),
                title     = f"LSTM Predictions — SKU {r['sku_id']}",
                save_path = self.results_dir / f"predictions_sku_{i+1}.png",
            )
        errors = np.concatenate([
            np.array(r['actual']) - np.array(r['predictions'])
            for r in all_results
        ])
        plot_error_distribution(
            errors     = errors,
            model_name = 'LSTM (improved)',
            save_path  = self.results_dir / 'error_distribution.png',
        )
        logger.info(f"Visualisations saved to {self.results_dir}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    PROCESSED_DATA_DIR = '../phase2/processed_data'
    RESULTS_DIR        = 'results/lstm'
    N_SKUS             = None   # None = all

    experiment = LSTMBaselineExperiment(
        processed_data_dir = PROCESSED_DATA_DIR,
        results_dir        = RESULTS_DIR,
        n_skus             = N_SKUS,
        # model
        lookback           = 28,
        hidden_size        = 64,
        num_layers         = 2,
        dropout            = 0.2,
        bidirectional      = True,
        # pre-training (increase pretrain_n_skus for richer warm-start)
        pretrain_n_skus    = 100,
        pretrain_epochs    = 30,
        pretrain_lr        = 1e-3,
        # per-SKU fine-tuning
        finetune_epochs    = 30,
        finetune_lr        = 3e-4,
        batch_size         = 32,
        patience           = 8,
    )

    metrics = experiment.run_experiment()

    print("\nImproved LSTM Baseline Complete!")
    print(f"Results saved to: {RESULTS_DIR}")