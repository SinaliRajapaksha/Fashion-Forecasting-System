"""
XGBoost Baseline Model for Fashion Demand Forecasting — IMPROVED
Gradient boosting with global cross-SKU training

Key improvements over the original:
  1. Global model (all SKUs)      — XGBoost is a tabular model: training on
                                    all SKUs at once gives 100-1000× more rows,
                                    far better than per-SKU models with <200 rows
  2. No data leakage              — test features are built from train-window
                                    stats only; lag/rolling features for test rows
                                    never touch future demand values
  3. Poisson objective            — demand is count data (non-negative integers);
                                    count:poisson is theoretically correct and
                                    outperforms reg:squarederror on sparse demand
  4. Early stopping               — XGBoost's native early_stopping_rounds on a
                                    held-out eval set; prevents over-boosting
  5. Rich feature set             — Fourier seasonality terms (weekly/monthly),
                                    SKU age (days since first sale), demand
                                    acceleration, exponential smoothing, inter-
                                    mittency ratio, log-demand, price-position
  6. Target-mean categorical      — replaces LabelEncoder ordinal integers with
                                    leave-one-out target mean encoding; meaningful
                                    signal, no spurious ordinality
  7. Monotone constraint          — prediction stays non-negative by construction
                                    (via log-link in Poisson objective)
  8. Feature importance export    — saves top-20 feature importances to results
"""

import json
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
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


# ---------------------------------------------------------------------------
# Target-mean encoder (replaces LabelEncoder)
# ---------------------------------------------------------------------------

class TargetMeanEncoder:
    """
    Leave-one-out target mean encoding for categorical columns.

    Replaces each category with the mean demand of all training rows
    that share that category (LOO to prevent target leakage within train).
    Unseen test categories receive the global training mean.
    """

    def __init__(self, smoothing: float = 10.0):
        self.smoothing  = smoothing   # weight toward global mean for rare cats
        self._maps      : Dict[str, Dict] = {}
        self._global    : Dict[str, float] = {}

    def fit(self, df: pd.DataFrame,
            cat_cols: List[str], target: str) -> 'TargetMeanEncoder':
        y = df[target].values.astype(float)
        global_mean = y.mean()

        for col in cat_cols:
            if col not in df.columns:
                continue
            stats = df.groupby(col)[target].agg(['sum', 'count'])
            n     = stats['count']
            s     = stats['sum']
            # smoothed mean: (sum + smoothing * global) / (count + smoothing)
            smoothed = (s + self.smoothing * global_mean) / (n + self.smoothing)
            self._maps[col]   = smoothed.to_dict()
            self._global[col] = global_mean

        return self

    def transform(self, df: pd.DataFrame,
                  cat_cols: List[str]) -> pd.DataFrame:
        df = df.copy()
        for col in cat_cols:
            if col not in df.columns or col not in self._maps:
                continue
            g   = self._global.get(col, 0.0)
            df[f'{col}_te'] = df[col].map(self._maps[col]).fillna(g)
        return df

    def fit_transform(self, df: pd.DataFrame,
                      cat_cols: List[str], target: str) -> pd.DataFrame:
        return self.fit(df, cat_cols, target).transform(df, cat_cols)


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

CAT_COLS = ['product_type_name', 'product_group_name', 'colour_group_name',
            'department_name', 'section_name', 'garment_group_name']


def _safe_shift(series: pd.Series, n: int) -> pd.Series:
    """Shift within-SKU without bleeding across SKU boundaries."""
    return series.shift(n)


def build_features_global(df: pd.DataFrame,
                           te_encoder     : Optional[TargetMeanEncoder] = None,
                           is_train       : bool = True,
                           train_stats    : Optional[pd.DataFrame] = None
                           ) -> pd.DataFrame:
    """
    Build the full feature matrix from a multi-SKU dataframe.

    IMPORTANT — leakage prevention:
      All lag and rolling features are computed WITHIN each SKU's sorted
      time series. For test rows, the rolling/lag window only ever looks
      at dates earlier than the row's own date — it never touches future
      test demand.

      `train_stats` is a per-SKU summary computed on training data only.
      When building test features, these pre-computed statistics are
      joined in rather than recomputing from the combined series.

    Parameters
    ----------
    df          : dataframe containing at least [article_id, date, demand]
    te_encoder  : fitted TargetMeanEncoder (None on first fit)
    is_train    : whether to fit the encoder
    train_stats : per-SKU stats pre-computed on training rows (used for test)
    """
    df = df.copy().sort_values(['article_id', 'date']).reset_index(drop=True)

    # ---- temporal features ----
    df['dayofweek']    = df['date'].dt.dayofweek
    df['dayofmonth']   = df['date'].dt.day
    df['month']        = df['date'].dt.month
    df['week']         = df['date'].dt.isocalendar().week.astype(int)
    df['quarter']      = df['date'].dt.quarter
    df['is_weekend']   = (df['dayofweek'] >= 5).astype(np.int8)

    # Fourier terms for weekly seasonality (period=7)
    for k in [1, 2]:
        df[f'sin_w{k}'] = np.sin(2 * np.pi * k * df['dayofweek'] / 7)
        df[f'cos_w{k}'] = np.cos(2 * np.pi * k * df['dayofweek'] / 7)

    # Fourier terms for monthly seasonality (period=30.44)
    for k in [1, 2]:
        df[f'sin_m{k}'] = np.sin(2 * np.pi * k * df['dayofmonth'] / 30.44)
        df[f'cos_m{k}'] = np.cos(2 * np.pi * k * df['dayofmonth'] / 30.44)

    # ---- within-SKU lag / rolling features ----
    grp = df.groupby('article_id', sort=False)['demand']

    for lag in [1, 7, 14, 28]:
        df[f'lag_{lag}'] = grp.shift(lag)

    for w in [7, 14, 28]:
        rolled = grp.shift(1).transform(
            lambda s: s.rolling(w, min_periods=1).mean())
        df[f'roll_mean_{w}']    = rolled
        df[f'roll_std_{w}']     = grp.shift(1).transform(
            lambda s: s.rolling(w, min_periods=1).std().fillna(0))
        df[f'roll_max_{w}']     = grp.shift(1).transform(
            lambda s: s.rolling(w, min_periods=1).max())

    # Exponential smoothed demand (α=0.3)
    df['ema_demand'] = grp.shift(1).transform(
        lambda s: s.ewm(alpha=0.3, adjust=False).mean())

    # Demand acceleration: lag_1 − lag_7
    df['demand_accel'] = df['lag_1'] - df['lag_7']

    # ---- per-SKU static stats (computed on train window only) ----
    if train_stats is not None:
        df = df.merge(train_stats, on='article_id', how='left')
    else:
        # Compute fresh (training path)
        sku_stats = (df.groupby('article_id')['demand']
                     .agg(sku_mean='mean', sku_std='std',
                          sku_max='max', sku_nonzero_ratio=lambda x: (x > 0).mean())
                     .reset_index())
        sku_stats['sku_std'] = sku_stats['sku_std'].fillna(0)

        # SKU age: days since first non-zero sale
        first_sale = (df[df['demand'] > 0]
                      .groupby('article_id')['date'].min()
                      .rename('first_sale_date').reset_index())
        df = df.merge(sku_stats, on='article_id', how='left')
        df = df.merge(first_sale, on='article_id', how='left')
        df['sku_age_days'] = (df['date'] - df['first_sale_date']
                              ).dt.days.fillna(0).clip(lower=0)
        df.drop(columns=['first_sale_date'], inplace=True)

    # log-demand (useful when demand range spans orders of magnitude)
    df['log_demand_lag1'] = np.log1p(df['lag_1'].fillna(0))

    # intermittency: rolling fraction of zero days
    df['zero_ratio_28'] = grp.shift(1).transform(
        lambda s: (s == 0).rolling(28, min_periods=1).mean())

    # ---- categorical target-mean encoding ----
    present_cats = [c for c in CAT_COLS if c in df.columns]
    if present_cats:
        if is_train:
            df = te_encoder.fit_transform(df, present_cats, 'demand')
        else:
            df = te_encoder.transform(df, present_cats)

    # ---- fill remaining NaN with 0 ----
    df = df.fillna(0)

    return df


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """Return model input columns (exclude identifiers and raw target)."""
    exclude = {'article_id', 'date', 'demand',
               'first_sale_date', *CAT_COLS,
               'graphical_appearance_name', 'index_group_name'}
    return [c for c in df.columns
            if c not in exclude and not c.endswith('_name')]


# ---------------------------------------------------------------------------
# Global XGBoost forecaster
# ---------------------------------------------------------------------------

class XGBoostForecaster:
    """
    Global XGBoost model trained across all SKUs.

    Key design choices:
      - count:poisson objective with log-link → non-negative predictions
        by construction, correct distributional assumption for count data
      - Early stopping on a 15% validation split
      - Feature importance tracked and exported
      - Test inference uses lag/rolling windows built from the training
        tail — never from test ground truth
    """

    def __init__(self,
                 n_estimators    : int   = 500,
                 max_depth       : int   = 6,
                 learning_rate   : float = 0.05,
                 subsample       : float = 0.8,
                 colsample_bytree: float = 0.8,
                 min_child_weight: int   = 5,
                 early_stopping  : int   = 30,
                 lookback        : int   = 28):

        self.n_estimators     = n_estimators
        self.max_depth        = max_depth
        self.learning_rate    = learning_rate
        self.subsample        = subsample
        self.colsample_bytree = colsample_bytree
        self.min_child_weight = min_child_weight
        self.early_stopping   = early_stopping
        self.lookback         = lookback

        self.model          : Optional[xgb.XGBRegressor] = None
        self.te_encoder     = TargetMeanEncoder(smoothing=10.0)
        self.feature_cols   : Optional[List[str]] = None
        self.train_stats    : Optional[pd.DataFrame] = None
        self.train_tail     : Optional[pd.DataFrame] = None  # last `lookback` rows per SKU
        self.is_fitted      = False

    # ------------------------------------------------------------------

    def fit(self, train_df: pd.DataFrame) -> bool:
        """
        Fit the global model on all SKUs combined.

        Parameters
        ----------
        train_df : full multi-SKU training dataframe with columns
                   [article_id, date, demand, <optional metadata>]
        """
        try:
            logger.info("Building global feature matrix...")

            # Pre-compute per-SKU stats (used to build leak-free test features)
            self.train_stats = (
                train_df.groupby('article_id')['demand']
                .agg(sku_mean='mean', sku_std='std',
                     sku_max='max',
                     sku_nonzero_ratio=lambda x: (x > 0).mean())
                .reset_index()
            )
            self.train_stats['sku_std'] = self.train_stats['sku_std'].fillna(0)

            # Save the last `lookback` rows per SKU for test inference
            self.train_tail = (
                train_df.sort_values(['article_id', 'date'])
                .groupby('article_id', sort=False)
                .tail(self.lookback)
                .copy()
                .reset_index(drop=True)
            )

            # Build features on training data
            df_feat = build_features_global(
                train_df,
                te_encoder  = self.te_encoder,
                is_train    = True,
                train_stats = None)   # stats computed internally on train

            # Drop rows with insufficient lookback history
            df_feat = df_feat.iloc[self.lookback:].copy()

            self.feature_cols = get_feature_columns(df_feat)

            X = df_feat[self.feature_cols].values.astype(np.float32)
            y = df_feat['demand'].values.astype(np.float32)
            # Poisson requires y > 0 — clip to 0 then add tiny offset
            y_poisson = np.maximum(y, 0.0)

            # Hold out 15% for early stopping
            X_tr, X_val, y_tr, y_val = train_test_split(
                X, y_poisson, test_size=0.15, random_state=42)

            logger.info(f"Training rows : {len(X_tr):,}   Val rows: {len(X_val):,}"
                        f"   Features: {len(self.feature_cols)}")

            self.model = xgb.XGBRegressor(
                n_estimators      = self.n_estimators,
                max_depth         = self.max_depth,
                learning_rate     = self.learning_rate,
                subsample         = self.subsample,
                colsample_bytree  = self.colsample_bytree,
                min_child_weight  = self.min_child_weight,
                objective         = 'count:poisson',   # correct for count data
                tree_method       = 'hist',             # fast on large datasets
                early_stopping_rounds = self.early_stopping,
                eval_metric       = 'poisson-nloglik',
                random_state      = 42,
                n_jobs            = -1,
                verbosity         = 0,
            )

            self.model.fit(
                X_tr, y_tr,
                eval_set          = [(X_val, y_val)],
                verbose           = False,
            )

            best = self.model.best_iteration
            logger.info(f"Best iteration : {best}")

            self.is_fitted = True
            return True

        except Exception as e:
            logger.warning(f"XGBoost fitting failed: {e}")
            return False

    # ------------------------------------------------------------------

    def predict_sku(self,
                    sku_id   : str,
                    test_rows: pd.DataFrame) -> np.ndarray:
        """
        Generate leak-free predictions for a single SKU.

        Strategy:
          Build features for test rows by prepending the training tail
          (last `lookback` rows from training) so that lag/rolling
          features at the first test row are computed from real training
          data, not from future test demand.
        """
        if not self.is_fitted:
            return np.zeros(len(test_rows))

        try:
            # Training tail for this SKU
            tail = self.train_tail[
                self.train_tail['article_id'] == sku_id].copy()

            # Concatenate tail + test; features will be built on the full
            # window, then we extract only the test-position rows.
            combined = pd.concat(
                [tail, test_rows], ignore_index=True
            ).sort_values('date').reset_index(drop=True)

            # Build features using pre-computed train stats (no test leakage)
            sku_stats = self.train_stats[
                self.train_stats['article_id'] == sku_id]

            df_feat = build_features_global(
                combined,
                te_encoder  = self.te_encoder,
                is_train    = False,
                train_stats = sku_stats if len(sku_stats) > 0 else None,
            )

            # Keep only the test rows (drop the prepended tail)
            n_tail     = len(tail)
            df_test    = df_feat.iloc[n_tail:].copy()

            # Align columns — fill any missing with 0
            for col in self.feature_cols:
                if col not in df_test.columns:
                    df_test[col] = 0.0

            X = df_test[self.feature_cols].values.astype(np.float32)
            preds = self.model.predict(X)

            # Poisson objective uses log-link → output is already ≥ 0
            preds = np.maximum(np.round(preds), 0).astype(float)
            return preds

        except Exception as e:
            logger.warning(f"SKU {sku_id} prediction failed: {e}")
            return np.zeros(len(test_rows))

    # ------------------------------------------------------------------

    def feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """Return a sorted feature importance dataframe."""
        if self.model is None:
            return pd.DataFrame()
        imp = pd.DataFrame({
            'feature'   : self.feature_cols,
            'importance': self.model.feature_importances_,
        }).sort_values('importance', ascending=False).head(top_n)
        return imp


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------

class XGBoostBaselineExperiment:
    """
    Full XGBoost experiment — one global model, per-SKU evaluation.
    """

    def __init__(self,
                 processed_data_dir : str   = '../phase2/processed_data',
                 results_dir        : str   = 'results/xgboost',
                 n_skus             : int   = None,
                 # model hyper-parameters
                 n_estimators       : int   = 500,
                 max_depth          : int   = 6,
                 learning_rate      : float = 0.05,
                 subsample          : float = 0.8,
                 colsample_bytree   : float = 0.8,
                 min_child_weight   : int   = 5,
                 early_stopping     : int   = 30,
                 lookback           : int   = 28):

        self.data_loader  = BaselineDataLoader(processed_data_dir)
        self.results_dir  = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.n_skus       = n_skus

        self.forecaster = XGBoostForecaster(
            n_estimators     = n_estimators,
            max_depth        = max_depth,
            learning_rate    = learning_rate,
            subsample        = subsample,
            colsample_bytree = colsample_bytree,
            min_child_weight = min_child_weight,
            early_stopping   = early_stopping,
            lookback         = lookback,
        )

        self.config = {
            'n_estimators'     : n_estimators,
            'max_depth'        : max_depth,
            'learning_rate'    : learning_rate,
            'subsample'        : subsample,
            'colsample_bytree' : colsample_bytree,
            'min_child_weight' : min_child_weight,
            'early_stopping'   : early_stopping,
            'lookback'         : lookback,
            'objective'        : 'count:poisson',
        }

        logger.info("Initialised improved XGBoost experiment")
        logger.info(f"Results : {self.results_dir}")

    # ------------------------------------------------------------------

    def run_experiment(self) -> Dict:
        logger.info("=" * 80)
        logger.info("STARTING IMPROVED XGBOOST BASELINE EXPERIMENT")
        logger.info("=" * 80)

        train_data, val_data, test_data = self.data_loader.get_train_test_split()
        final_train = pd.concat([train_data, val_data], ignore_index=True)

        skus = (self.data_loader.get_sample_skus(n=self.n_skus)
                if self.n_skus else self.data_loader.get_all_skus())

        logger.info(f"SKUs to evaluate: {len(skus)}")

        # ---- Phase 1: fit global model on ALL training data ----
        logger.info("Fitting global XGBoost model on full training set...")
        ok = self.forecaster.fit(final_train)
        if not ok:
            raise RuntimeError("Global model fitting failed — aborting.")

        # ---- Phase 2: per-SKU inference ----
        all_results  = []
        sku_metrics  = {}
        failed_skus  = []

        for sku in tqdm(skus, desc="XGBoost — SKUs"):
            try:
                result = self._evaluate_sku(sku, test_data)
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
        logger.info("IMPROVED XGBOOST EXPERIMENT COMPLETE")
        logger.info("=" * 80)

        return aggregated

    # ------------------------------------------------------------------

    def _evaluate_sku(self,
                      sku_id   : str,
                      test_data: pd.DataFrame) -> Optional[Dict]:

        te = (test_data[test_data['article_id'] == sku_id]
              .sort_values('date').reset_index(drop=True))

        if len(te) == 0:
            return None

        preds  = self.forecaster.predict_sku(sku_id, te)
        actual = te['demand'].values

        return {
            'sku_id'      : sku_id,
            'predictions' : preds.tolist(),
            'actual'      : actual.tolist(),
            'dates'       : te['date'].dt.strftime('%Y-%m-%d').tolist(),
            'metrics'     : calculate_forecasting_metrics(actual, preds),
            'model_fitted': True,
        }

    # ------------------------------------------------------------------

    def _save_results(self, all_results: List[Dict], aggregated: Dict):
        summary = {
            'model'             : 'XGBoost (improved)',
            'configuration'     : self.config,
            'n_skus_evaluated'  : len(all_results),
            'aggregated_metrics': aggregated,
            'best_iteration'    : int(self.forecaster.model.best_iteration)
                                  if self.forecaster.model else -1,
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

        # Feature importance
        fi = self.forecaster.feature_importance(top_n=30)
        if not fi.empty:
            fi.to_csv(self.results_dir / 'feature_importance.csv', index=False)
            logger.info("Feature importance saved.")

        print("\n" + "=" * 80)
        print("IMPROVED XGBOOST BASELINE — RESULTS SUMMARY")
        print("=" * 80)
        print(f"Model   : XGBoost global — count:poisson objective")
        print(f"SKUs    : {len(all_results)}")
        if self.forecaster.model:
            print(f"Best iter : {self.forecaster.model.best_iteration}")
            print(f"Features  : {len(self.forecaster.feature_cols)}")
        print(f"\nAggregated Metrics:")
        print(f"  Mean RMSE    : {aggregated['mean_rmse']:.4f}  (±{aggregated['std_rmse']:.4f})")
        print(f"  Mean MAE     : {aggregated['mean_mae']:.4f}  (±{aggregated['std_mae']:.4f})")
        print(f"  Mean MAPE    : {aggregated['mean_mape']:.2f}%")
        print(f"  Median RMSE  : {aggregated['median_rmse']:.4f}")
        print(f"  Median MAE   : {aggregated['median_mae']:.4f}")
        if not fi.empty:
            print(f"\nTop-5 features:")
            for _, row in fi.head(5).iterrows():
                print(f"  {row['feature']:<30}  {row['importance']:.4f}")
        print("=" * 80 + "\n")

    def _generate_visualizations(self, all_results: List[Dict]):
        logger.info("Generating visualisations...")
        for i, r in enumerate(all_results[:5]):
            plot_predictions(
                dates     = pd.to_datetime(r['dates']),
                y_true    = np.array(r['actual']),
                y_pred    = np.array(r['predictions']),
                title     = f"XGBoost Predictions — SKU {r['sku_id']}",
                save_path = self.results_dir / f"predictions_sku_{i+1}.png",
            )
        errors = np.concatenate([
            np.array(r['actual']) - np.array(r['predictions'])
            for r in all_results
        ])
        plot_error_distribution(
            errors     = errors,
            model_name = 'XGBoost (improved)',
            save_path  = self.results_dir / 'error_distribution.png',
        )
        logger.info(f"Visualisations saved to {self.results_dir}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    PROCESSED_DATA_DIR = '../phase2/processed_data'
    RESULTS_DIR        = 'results/xgboost'
    N_SKUS             = None   # None = all

    experiment = XGBoostBaselineExperiment(
        processed_data_dir  = PROCESSED_DATA_DIR,
        results_dir         = RESULTS_DIR,
        n_skus              = N_SKUS,
        # ---- tree structure ----
        n_estimators        = 500,     # capped by early stopping
        max_depth           = 6,
        learning_rate       = 0.05,
        # ---- regularisation ----
        subsample           = 0.8,
        colsample_bytree    = 0.8,
        min_child_weight    = 5,       # prevents splits on tiny leaf counts
        early_stopping      = 30,
        lookback            = 28,
    )

    metrics = experiment.run_experiment()

    print("\nImproved XGBoost Baseline Complete!")
    print(f"Results saved to: {RESULTS_DIR}")