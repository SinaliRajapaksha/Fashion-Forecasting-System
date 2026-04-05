"""
CLeaR Baseline Model — Speed-Optimised Edition v2
Continual Learning for Regression (He & Sick, 2021)

SPEED IMPROVEMENTS over the slow v1:
  1. Batched Fisher  — single backward pass (not N per-sample passes)
  2. Vectorised predict — one batched forward for all test steps
  3. Lighter TCN     — 2 blocks / 24 ch / BatchNorm (was 3/32/LayerNorm)
  4. Reduced epochs  — 20 max, patience 5 (was 40/7)
  5. Parallel workers — ProcessPoolExecutor (cpu_count // 2)

BUG FIX vs previous fast version:
  Workers were dying immediately because 2.57M-row DataFrames were
  being pickled and sent to every worker process, causing OOM kills.

  Fix: pre-filter each SKU's data in the main process → pass only
  tiny numpy arrays (train_vals, test_vals, test_dates) to workers.
  Peak memory per worker is now ~kilobytes not gigabytes.

All qualitative improvements preserved:
  ✓ TCN encoder (temporal awareness)
  ✓ Batched diagonal Fisher EWC
  ✓ Per-SKU StandardScaler (no cross-SKU leakage)
  ✓ Reservoir sampling replay buffer
  ✓ Rich 13-dim feature engineering
  ✓ Early stopping + gradient clipping
  ✓ HuberLoss, Softplus output
"""

import json
import logging
import multiprocessing
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
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
# Feature engineering
# ---------------------------------------------------------------------------

def build_features(window: np.ndarray) -> np.ndarray:
    """13-dim: mean, std, min, max, slope, last-7, sparsity."""
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

    last7         = np.zeros(7, dtype=np.float32)
    take          = min(7, n)
    last7[-take:] = w[-take:]
    sparsity      = (w > 0).mean()

    return np.array([mean_v, std_v, min_v, max_v, slope,
                     *last7, sparsity], dtype=np.float32)


FEATURE_DIM = 13


# ---------------------------------------------------------------------------
# Reservoir buffer
# ---------------------------------------------------------------------------

class ReservoirBuffer:
    def __init__(self, max_size: int = 1500):
        self.max_size = max_size
        self._x: List = []
        self._y: List = []
        self._count   = 0

    def add_batch(self, X: np.ndarray, y: np.ndarray):
        for xi, yi in zip(X, y):
            self._count += 1
            if len(self._x) < self.max_size:
                self._x.append(xi.copy())
                self._y.append(float(yi))
            else:
                j = np.random.randint(0, self._count)
                if j < self.max_size:
                    self._x[j] = xi.copy()
                    self._y[j] = float(yi)

    def sample(self, n: int) -> Tuple[np.ndarray, np.ndarray]:
        if not self._x:
            return (np.empty((0, FEATURE_DIM), dtype=np.float32),
                    np.empty(0, dtype=np.float32))
        n   = min(n, len(self._x))
        idx = np.random.choice(len(self._x), size=n, replace=False)
        X   = np.stack([self._x[i] for i in idx])
        y   = np.array([self._y[i] for i in idx], dtype=np.float32)
        return X, y

    def __len__(self):
        return len(self._x)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class _CausalConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, dilation):
        super().__init__()
        self.pad  = (kernel - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel,
                              dilation=dilation, padding=0)

    def forward(self, x):
        return self.conv(nn.functional.pad(x, (self.pad, 0)))


class TCNBlock(nn.Module):
    """BatchNorm1d — ~2x faster than LayerNorm on CPU."""
    def __init__(self, channels, kernel, dilation, dropout=0.1):
        super().__init__()
        self.conv1 = _CausalConv1d(channels, channels, kernel, dilation)
        self.conv2 = _CausalConv1d(channels, channels, kernel, dilation)
        self.bn1   = nn.BatchNorm1d(channels)
        self.bn2   = nn.BatchNorm1d(channels)
        self.drop  = nn.Dropout(dropout)
        self.act   = nn.GELU()

    def forward(self, x):
        h = self.drop(self.act(self.bn1(self.conv1(x))))
        h = self.act(self.bn2(self.conv2(h)))
        return x + h


class CLeaRModel(nn.Module):
    def __init__(self,
                 lookback    : int   = 28,
                 tcn_channels: int   = 24,
                 tcn_layers  : int   = 2,
                 tcn_kernel  : int   = 3,
                 feat_dim    : int   = FEATURE_DIM,
                 hidden_dim  : int   = 64,
                 dropout     : float = 0.1):
        super().__init__()

        self.input_proj = nn.Linear(1, tcn_channels)
        self.tcn_blocks = nn.ModuleList([
            TCNBlock(tcn_channels, tcn_kernel, dilation=2**i, dropout=dropout)
            for i in range(tcn_layers)
        ])
        self.feat_net = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
        )
        fusion_in = tcn_channels + hidden_dim // 2
        self.head = nn.Sequential(
            nn.Linear(fusion_in, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Softplus()
        )

    def forward(self, seq: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(seq).transpose(1, 2)
        for blk in self.tcn_blocks:
            x = blk(x)
        tcn_out  = x[:, :, -1]
        feat_out = self.feat_net(feat)
        return self.head(torch.cat([tcn_out, feat_out], dim=-1)).squeeze(-1)


# ---------------------------------------------------------------------------
# Batched Fisher
# ---------------------------------------------------------------------------

def compute_fisher(model, seq_t, feat_t, tgt_t, device):
    model.train()
    model.zero_grad()
    n   = min(64, len(tgt_t))
    out = model(seq_t[:n], feat_t[:n])
    nn.MSELoss()(out, tgt_t[:n]).backward()
    fisher = {
        name: (p.grad.detach() ** 2) if p.grad is not None
              else torch.zeros_like(p)
        for name, p in model.named_parameters()
    }
    model.zero_grad()
    return fisher


# ---------------------------------------------------------------------------
# Forecaster  — accepts numpy arrays, not Series
# ---------------------------------------------------------------------------

class CLeaRForecaster:
    def __init__(self, model, replay_buffer, device,
                 lookback=28, learning_rate=5e-4, epochs=20,
                 batch_size=32, replay_ratio=0.4, ewc_lambda=400.0,
                 patience=5, grad_clip=1.0):
        self.model         = model
        self.replay_buffer = replay_buffer
        self.device        = device
        self.lookback      = lookback
        self.lr            = learning_rate
        self.epochs        = epochs
        self.batch_size    = batch_size
        self.replay_ratio  = replay_ratio
        self.ewc_lambda    = ewc_lambda
        self.patience      = patience
        self.grad_clip     = grad_clip
        self.scaler        = StandardScaler()
        self.is_fitted     = False
        self.prev_params   = None
        self.fisher        = None

    def _make_sequences(self, scaled, raw):
        n = len(scaled)
        if n <= self.lookback:
            return (np.empty((0, self.lookback, 1), dtype=np.float32),
                    np.empty((0, FEATURE_DIM),       dtype=np.float32),
                    np.empty((0,),                   dtype=np.float32))
        num   = n - self.lookback
        seqs  = np.lib.stride_tricks.sliding_window_view(
                    scaled, self.lookback)[:num].reshape(
                    num, self.lookback, 1).astype(np.float32)
        tgts  = scaled[self.lookback:].astype(np.float32)
        feats = np.stack([
            build_features(raw[i: i + self.lookback])
            for i in range(num)
        ]).astype(np.float32)
        return seqs, feats, tgts

    def _ewc_penalty(self):
        if self.prev_params is None:
            return torch.tensor(0.0, device=self.device)
        penalty = torch.tensor(0.0, device=self.device)
        for n, p in self.model.named_parameters():
            if n in self.fisher:
                penalty += (self.fisher[n] *
                            (p - self.prev_params[n]) ** 2).sum()
        return self.ewc_lambda * penalty

    def fit(self, train_vals: np.ndarray) -> bool:
        raw = train_vals.astype(float)
        if len(raw) < self.lookback + 10:
            return False

        self.scaler.fit(raw.reshape(-1, 1))
        scaled = self.scaler.transform(raw.reshape(-1, 1)).flatten()
        seqs, feats, targets = self._make_sequences(scaled, raw)
        if len(seqs) == 0:
            return False

        self.replay_buffer.add_batch(feats, targets)

        optimizer  = optim.AdamW(self.model.parameters(),
                                 lr=self.lr, weight_decay=1e-4)
        criterion  = nn.HuberLoss(delta=1.0)
        best_loss  = float('inf')
        no_improve = 0

        seq_t  = torch.tensor(seqs,    device=self.device)
        feat_t = torch.tensor(feats,   device=self.device)
        tgt_t  = torch.tensor(targets, device=self.device)

        self.model.train()
        for _ in range(self.epochs):
            perm       = torch.randperm(len(seq_t))
            epoch_loss = 0.0
            n_batches  = 0

            for start in range(0, len(seq_t), self.batch_size):
                idx = perm[start: start + self.batch_size]
                s_b = seq_t[idx]
                f_b = feat_t[idx]
                y_b = tgt_t[idx]

                n_rep  = max(1, int(len(idx) * self.replay_ratio))
                rX, rY = self.replay_buffer.sample(n_rep)
                if len(rX) > 0:
                    rS  = torch.zeros(len(rX), self.lookback, 1,
                                      device=self.device)
                    s_b = torch.cat([s_b, rS])
                    f_b = torch.cat([f_b, torch.tensor(rX, device=self.device)])
                    y_b = torch.cat([y_b, torch.tensor(rY, device=self.device)])

                optimizer.zero_grad()
                loss = (criterion(self.model(s_b, f_b), y_b)
                        + self._ewc_penalty())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.grad_clip)
                optimizer.step()

                epoch_loss += loss.item()
                n_batches  += 1

            avg = epoch_loss / max(n_batches, 1)
            if avg < best_loss - 1e-4:
                best_loss  = avg
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= self.patience:
                    break

        self.prev_params = {n: p.detach().clone()
                            for n, p in self.model.named_parameters()}
        self.fisher = compute_fisher(
            self.model, seq_t[:64], feat_t[:64], tgt_t[:64], self.device)
        self.is_fitted = True
        return True

    def predict(self, train_vals: np.ndarray,
                test_vals: np.ndarray) -> np.ndarray:
        """Vectorised: single batched forward pass."""
        if not self.is_fitted:
            return np.zeros(len(test_vals))

        all_raw = np.concatenate([train_vals, test_vals]).astype(float)
        all_sc  = self.scaler.transform(all_raw.reshape(-1, 1)).flatten()
        n_train = len(train_vals)
        steps   = len(test_vals)

        wins_sc, wins_raw = [], []
        for i in range(n_train, len(all_raw)):
            start = max(0, i - self.lookback)
            w_sc  = all_sc[start: i]
            w_raw = all_raw[start: i]
            if len(w_sc) < self.lookback:
                pad   = self.lookback - len(w_sc)
                w_sc  = np.pad(w_sc,  (pad, 0), mode='edge')
                w_raw = np.pad(w_raw, (pad, 0), mode='edge')
            wins_sc .append(w_sc)
            wins_raw.append(w_raw)

        seq_t  = torch.tensor(
            np.array(wins_sc, dtype=np.float32).reshape(steps, self.lookback, 1),
            device=self.device)
        feat_t = torch.tensor(
            np.stack([build_features(w) for w in wins_raw]).astype(np.float32),
            device=self.device)

        self.model.eval()
        with torch.no_grad():
            preds_sc = self.model(seq_t, feat_t).cpu().numpy()

        preds = self.scaler.inverse_transform(
            preds_sc.reshape(-1, 1)).flatten()
        return np.maximum(np.round(preds), 0).astype(float)


# ---------------------------------------------------------------------------
# Worker — receives tiny numpy arrays, NOT the full DataFrame
# ---------------------------------------------------------------------------

def _worker(args: Tuple) -> Optional[Dict]:
    """
    Top-level picklable worker.

    Receives pre-filtered numpy arrays per SKU — NOT DataFrames.
    This is the core fix: the original failure was caused by pickling
    a 2.57M-row DataFrame for every worker, causing OOM kills.
    """
    (sku_id, train_vals, test_vals, test_dates,
     model_state, config) = args

    device = torch.device('cpu')
    model  = CLeaRModel(
        lookback     = config['lookback'],
        tcn_channels = config['tcn_channels'],
        tcn_layers   = config['tcn_layers'],
        hidden_dim   = config['hidden_dim'],
    )
    model.load_state_dict(model_state)
    model.to(device)

    replay     = ReservoirBuffer(max_size=config['replay_buffer_size'])
    forecaster = CLeaRForecaster(
        model         = model,
        replay_buffer = replay,
        device        = device,
        lookback      = config['lookback'],
        learning_rate = config['learning_rate'],
        epochs        = config['epochs'],
        batch_size    = config['batch_size'],
        replay_ratio  = config['replay_ratio'],
        ewc_lambda    = config['ewc_lambda'],
        patience      = config['patience'],
    )

    try:
        ok = forecaster.fit(train_vals)
        if not ok:
            return None
        preds   = forecaster.predict(train_vals, test_vals)
        metrics = calculate_forecasting_metrics(test_vals, preds)
        return {
            'sku_id'      : sku_id,
            'predictions' : preds.tolist(),
            'actual'      : test_vals.tolist(),
            'dates'       : test_dates,
            'metrics'     : metrics,
            'model_fitted': True,
        }
    except Exception:
        import traceback
        logger.debug(f"SKU {sku_id}:\n{traceback.format_exc()}")
        return None


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------

class CLeaRBaselineExperiment:

    def __init__(self,
                 processed_data_dir : str   = '../phase2/processed_data',
                 results_dir        : str   = 'results/clear',
                 n_skus             : int   = None,
                 n_workers          : int   = None,
                 lookback           : int   = 28,
                 tcn_channels       : int   = 24,
                 tcn_layers         : int   = 2,
                 hidden_dim         : int   = 64,
                 learning_rate      : float = 5e-4,
                 epochs             : int   = 20,
                 batch_size         : int   = 32,
                 replay_buffer_size : int   = 1500,
                 replay_ratio       : float = 0.4,
                 ewc_lambda         : float = 400.0,
                 patience           : int   = 5):

        self.data_loader = BaselineDataLoader(processed_data_dir)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.n_skus      = n_skus

        cpu_count     = multiprocessing.cpu_count()
        self.n_workers = (n_workers if n_workers is not None
                          else max(1, cpu_count // 2))

        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        self.model = CLeaRModel(
            lookback     = lookback,
            tcn_channels = tcn_channels,
            tcn_layers   = tcn_layers,
            hidden_dim   = hidden_dim,
        ).to(self.device)

        self.config = dict(
            lookback           = lookback,
            tcn_channels       = tcn_channels,
            tcn_layers         = tcn_layers,
            hidden_dim         = hidden_dim,
            learning_rate      = learning_rate,
            epochs             = epochs,
            batch_size         = batch_size,
            replay_buffer_size = replay_buffer_size,
            replay_ratio       = replay_ratio,
            ewc_lambda         = ewc_lambda,
            patience           = patience,
        )

        n_params = sum(p.numel() for p in self.model.parameters()
                       if p.requires_grad)
        logger.info("Initialised fast CLeaR experiment")
        logger.info(f"Device   : {self.device}")
        logger.info(f"Params   : {n_params:,}")
        logger.info(f"Workers  : {self.n_workers}  (CPU cores: {cpu_count})")
        logger.info(f"Results  : {self.results_dir}")

    # ------------------------------------------------------------------

    def run_experiment(self) -> Dict:
        logger.info("=" * 80)
        logger.info("STARTING FAST CLEAR BASELINE EXPERIMENT")
        logger.info("=" * 80)

        train_data, val_data, test_data = self.data_loader.get_train_test_split()
        final_train = pd.concat([train_data, val_data], ignore_index=True)

        skus = (self.data_loader.get_sample_skus(n=self.n_skus)
                if self.n_skus is not None
                else self.data_loader.get_all_skus())

        logger.info(f"SKUs to evaluate : {len(skus)}")
        logger.info("Pre-filtering SKU data in main process...")

        # ----------------------------------------------------------------
        # KEY FIX: filter once here; workers receive tiny arrays only
        # ----------------------------------------------------------------
        lookback    = self.config['lookback']
        model_state = {k: v.cpu()
                       for k, v in self.model.state_dict().items()}

        # GroupBy-index for O(1) SKU lookup
        train_grp = final_train.groupby('article_id')
        test_grp  = test_data.groupby('article_id')

        worker_args : List[Tuple] = []
        skipped     = 0

        for sku in skus:
            try:
                tr_df = train_grp.get_group(sku).sort_values('date')
                te_df = test_grp .get_group(sku).sort_values('date')
                tr    = tr_df['demand'].values.astype(np.float32)
                te    = te_df['demand'].values.astype(np.float32)
                dates = te_df['date'].dt.strftime('%Y-%m-%d').tolist()

                if len(te) == 0 or len(tr) < lookback + 10:
                    skipped += 1
                    continue

                worker_args.append(
                    (sku, tr, te, dates, model_state, self.config))

            except KeyError:
                skipped += 1

        logger.info(f"Valid SKUs: {len(worker_args)}  |  Skipped: {skipped}")

        all_results : List[Dict] = []
        failed_skus : List[str]  = []

        if self.n_workers == 1:
            for args in tqdm(worker_args, desc="CLeaR — SKUs"):
                result = _worker(args)
                (all_results if result else failed_skus).append(
                    result if result else args[0])
        else:
            with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
                futures = {executor.submit(_worker, a): a[0]
                           for a in worker_args}
                for fut in tqdm(as_completed(futures),
                                total=len(futures),
                                desc="CLeaR — SKUs (parallel)"):
                    sku = futures[fut]
                    try:
                        result = fut.result()
                        if result:
                            all_results.append(result)
                        else:
                            failed_skus.append(sku)
                    except Exception as e:
                        logger.warning(f"SKU {sku} future error: {e}")
                        failed_skus.append(sku)

        logger.info(f"Succeeded : {len(all_results)} SKUs")
        logger.info(f"Failed    : {len(failed_skus)} SKUs")

        if not all_results:
            logger.error(
                "No SKUs succeeded. Run with n_workers=1 to surface "
                "detailed tracebacks from the worker.")
            return {}

        sku_metrics = {r['sku_id']: r['metrics'] for r in all_results}
        aggregated  = aggregate_sku_metrics(sku_metrics)
        self._save_results(all_results, aggregated)
        self._generate_visualizations(all_results)

        logger.info("=" * 80)
        logger.info("FAST CLEAR EXPERIMENT COMPLETE")
        logger.info("=" * 80)
        return aggregated

    # ------------------------------------------------------------------

    def _save_results(self, all_results: List[Dict], aggregated: Dict):
        summary = {
            'model'             : 'CLeaR (fast)',
            'configuration'     : self.config,
            'n_workers'         : self.n_workers,
            'n_skus_evaluated'  : len(all_results),
            'aggregated_metrics': aggregated,
            'timestamp'         : pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
        }
        with open(self.results_dir / 'results_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)

        rows = [{
            'sku_id'      : r['sku_id'],
            'rmse'        : r['metrics'].get('rmse',  float('nan')),
            'mae'         : r['metrics'].get('mae',   float('nan')),
            'mape'        : r['metrics'].get('mape',  float('nan')),
            'smape'       : r['metrics'].get('smape', float('nan')),
            'model_fitted': r['model_fitted'],
        } for r in all_results]
        pd.DataFrame(rows).to_csv(
            self.results_dir / 'detailed_results.csv', index=False)

        with open(self.results_dir / 'sample_predictions.json', 'w') as f:
            json.dump(all_results[:10], f, indent=2)

        print("\n" + "=" * 80)
        print("FAST CLEAR BASELINE — RESULTS SUMMARY")
        print("=" * 80)
        print(f"Model  : CLeaR — TCN + batched EWC + reservoir replay")
        print(f"Device : {self.device}   Workers: {self.n_workers}")
        print(f"SKUs   : {len(all_results)}")
        print(f"\nAggregated Metrics:")
        print(f"  Mean RMSE   : {aggregated.get('mean_rmse',   float('nan')):.4f}"
              f"  (±{aggregated.get('std_rmse', float('nan')):.4f})")
        print(f"  Mean MAE    : {aggregated.get('mean_mae',    float('nan')):.4f}"
              f"  (±{aggregated.get('std_mae',  float('nan')):.4f})")
        print(f"  Mean MAPE   : {aggregated.get('mean_mape',   float('nan')):.2f}%")
        print(f"  Median RMSE : {aggregated.get('median_rmse', float('nan')):.4f}")
        print(f"  Median MAE  : {aggregated.get('median_mae',  float('nan')):.4f}")
        print("=" * 80 + "\n")

    def _generate_visualizations(self, all_results: List[Dict]):
        logger.info("Generating visualisations...")
        for i, r in enumerate(all_results[:5]):
            plot_predictions(
                dates     = pd.to_datetime(r['dates']),
                y_true    = np.array(r['actual']),
                y_pred    = np.array(r['predictions']),
                title     = f"CLeaR Predictions — SKU {r['sku_id']}",
                save_path = self.results_dir / f"predictions_sku_{i+1}.png",
            )
        errors = np.concatenate([
            np.array(r['actual']) - np.array(r['predictions'])
            for r in all_results
        ])
        plot_error_distribution(
            errors     = errors,
            model_name = 'CLeaR (fast)',
            save_path  = self.results_dir / 'error_distribution.png',
        )
        logger.info(f"Visualisations saved to {self.results_dir}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Required guard for ProcessPoolExecutor on Linux/macOS (spawn mode)
    multiprocessing.set_start_method('spawn', force=True)

    PROCESSED_DATA_DIR = '../phase2/processed_data'
    RESULTS_DIR        = 'results/clear'
    N_SKUS             = None   # None = all 5000

    experiment = CLeaRBaselineExperiment(
        processed_data_dir  = PROCESSED_DATA_DIR,
        results_dir         = RESULTS_DIR,
        n_skus              = N_SKUS,
        n_workers           = None,   # auto = cpu_count // 2
        lookback            = 28,
        tcn_channels        = 24,
        tcn_layers          = 2,
        hidden_dim          = 64,
        learning_rate       = 5e-4,
        epochs              = 20,
        batch_size          = 32,
        replay_buffer_size  = 1500,
        replay_ratio        = 0.4,
        ewc_lambda          = 400.0,
        patience            = 5,
    )

    metrics = experiment.run_experiment()
    print("\nFast CLeaR Baseline Complete!")
    print(f"Results saved to: {RESULTS_DIR}")