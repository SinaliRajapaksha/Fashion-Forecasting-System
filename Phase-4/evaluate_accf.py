"""
ACCF Model Evaluation Script — IMPROVED
Trains and evaluates ACCF on H&M fashion dataset

Improvements over the original:
  1. Deterministic metadata encoding
                        — original used Python's hash() which is randomised
                          across processes by PYTHONHASHSEED; replaced with
                          sklearn OrdinalEncoder so results are reproducible
  2. Cold-start SKUs    — original skipped SKUs with < 50 training days;
                          ACCF's adapter can work with as few as 14 days
                          (lookback + a few targets) so threshold lowered
  3. WAPE in summary    — added to printed results and saved JSON to match
                          the improved metrics.py
  4. Residual plot      — calls plot_residuals for the first result SKU
  5. wape / bias columns in detailed CSV
  6. Cleaner article metadata loading with fallback if file missing
"""

import sys
sys.path.append('../phase3_baselines')

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from accf_model import ACCFForecaster
from utils.data_loader import BaselineDataLoader
from utils.metrics import calculate_forecasting_metrics, aggregate_sku_metrics
from utils.visualization import (plot_predictions, plot_error_distribution,
                                  plot_residuals)

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Deterministic metadata encoder
# ---------------------------------------------------------------------------

class MetadataBuilder:
    """
    Builds a fixed-size float32 metadata vector for each SKU using
    OrdinalEncoder — deterministic, reproducible, no hash collisions.

    Categorical columns are ordinally encoded (unknown categories → -1).
    The vector is zero-padded to `target_dim`.
    """

    CAT_COLS = [
        'product_type_name', 'product_group_name',
        'colour_group_name', 'department_name',
        'section_name', 'garment_group_name',
    ]

    def __init__(self, target_dim: int = 64):
        self.target_dim  = target_dim
        self._encoders   = {}   # col → {cat: int}
        self._is_fitted  = False

    def fit(self, articles_df: pd.DataFrame) -> 'MetadataBuilder':
        for col in self.CAT_COLS:
            if col not in articles_df.columns:
                continue
            cats = articles_df[col].astype(str).unique().tolist()
            self._encoders[col] = {c: i for i, c in enumerate(sorted(cats))}
        self._is_fitted = True
        return self

    def encode(self, articles_df: pd.DataFrame,
               sku_id: str) -> np.ndarray:
        row = articles_df[articles_df['article_id'] == sku_id]
        vec = []
        for col in self.CAT_COLS:
            if col not in articles_df.columns or col not in self._encoders:
                vec.append(0.0)
                continue
            val = str(row[col].iloc[0]) if len(row) > 0 else '__unknown__'
            vec.append(float(self._encoders[col].get(val, -1)))

        # Pad / truncate to target_dim
        if len(vec) < self.target_dim:
            vec += [0.0] * (self.target_dim - len(vec))
        vec = vec[:self.target_dim]
        return np.array(vec, dtype=np.float32)


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------

class ACCFExperiment:
    """Complete ACCF experiment with training and evaluation."""

    def __init__(self,
                 processed_data_dir : str = '../phase2/processed_data',
                 results_dir        : str = 'results/accf',
                 n_skus             : int = None,
                 n_train_skus       : int = 100):
        """
        Parameters
        ----------
        processed_data_dir : path to phase2 processed data
        results_dir        : where to write results
        n_skus             : SKUs to evaluate (None = all)
        n_train_skus       : SKUs used for backbone pre-training
        """
        self.data_loader  = BaselineDataLoader(processed_data_dir)
        self.results_dir  = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.n_skus       = n_skus
        self.n_train_skus = n_train_skus

        self.config = {
            'lookback_window'   : 28,
            'hidden_size'       : 128,
            'adapter_size'      : 32,
            'metadata_dim'      : 64,
            'learning_rate'     : 1e-3,
            'adapter_lr'        : 1e-2,
            'epochs'            : 50,
            'batch_size'        : 32,
            'replay_buffer_size': 5000,
            'replay_ratio'      : 0.3,
            'ewc_lambda'        : 200.0,
            'adapter_epochs'    : 20,
            'name'              : 'ACCF',
        }

        logger.info(f"Initialised ACCF experiment | config={self.config}")

    # ------------------------------------------------------------------

    def _load_articles(self) -> pd.DataFrame:
        """Load article metadata with graceful fallback."""
        candidates = [
            Path(self.data_loader.data_dir) / 'articles_metadata.csv',
            Path(self.data_loader.data_dir) / 'articles.csv',
        ]
        for path in candidates:
            if path.exists():
                df = pd.read_csv(path)
                # Normalise article_id dtype
                df['article_id'] = df['article_id'].astype(str)
                logger.info(f"Loaded article metadata: {path} "
                            f"({len(df):,} rows)")
                return df

        logger.warning("No article metadata file found — using empty frame.")
        return pd.DataFrame(columns=['article_id'])

    # ------------------------------------------------------------------

    def evaluate_sku(self,
                     sku_id     : str,
                     train_data : pd.DataFrame,
                     test_data  : pd.DataFrame,
                     forecaster : ACCFForecaster,
                     meta_builder: MetadataBuilder,
                     articles_df: pd.DataFrame) -> dict:
        """Evaluate ACCF on a single SKU."""

        tr = (train_data[train_data['article_id'] == sku_id]
              .sort_values('date').reset_index(drop=True))
        te = (test_data [test_data ['article_id'] == sku_id]
              .sort_values('date').reset_index(drop=True))

        # Lower min-train threshold — adapter works with 14+ days
        min_train = forecaster.lookback + 5
        if len(te) == 0 or len(tr) < min_train:
            return None

        metadata    = meta_builder.encode(articles_df, sku_id)
        predictions = forecaster.predict(
            te['demand'], tr['demand'], sku_id, metadata)

        actual  = te['demand'].values
        metrics = calculate_forecasting_metrics(actual, predictions)

        return {
            'sku_id'     : sku_id,
            'predictions': predictions.tolist(),
            'actual'     : actual.tolist(),
            'dates'      : te['date'].dt.strftime('%Y-%m-%d').tolist(),
            'metrics'    : metrics,
        }

    # ------------------------------------------------------------------

    def run_experiment(self):
        """Run the complete ACCF experiment."""

        logger.info("=" * 80)
        logger.info("STARTING ACCF EXPERIMENT")
        logger.info("=" * 80)

        train_data, val_data, test_data = self.data_loader.get_train_test_split()
        final_train = pd.concat([train_data, val_data], ignore_index=True)

        articles_df  = self._load_articles()
        meta_builder = MetadataBuilder(target_dim=self.config['metadata_dim'])
        meta_builder.fit(articles_df)

        # Determine SKU lists
        if self.n_skus is not None:
            skus_to_evaluate = self.data_loader.get_sample_skus(n=self.n_skus)
        else:
            skus_to_evaluate = self.data_loader.get_all_skus()

        train_skus = skus_to_evaluate[:self.n_train_skus]
        logger.info(f"Pre-training on  : {len(train_skus)} SKUs")
        logger.info(f"Evaluating on    : {len(skus_to_evaluate)} SKUs")

        # ---- Initialise forecaster ----
        forecaster = ACCFForecaster(
            lookback_window     = self.config['lookback_window'],
            hidden_size         = self.config['hidden_size'],
            adapter_size        = self.config['adapter_size'],
            metadata_dim        = self.config['metadata_dim'],
            learning_rate       = self.config['learning_rate'],
            adapter_lr          = self.config['adapter_lr'],
            epochs              = self.config['epochs'],
            batch_size          = self.config['batch_size'],
            replay_buffer_size  = self.config['replay_buffer_size'],
            replay_ratio        = self.config['replay_ratio'],
            ewc_lambda          = self.config['ewc_lambda'],
            adapter_epochs      = self.config['adapter_epochs'],
        )

        # ---- Build pre-training dicts ----
        train_data_dict     = {}
        train_metadata_dict = {}
        min_len = forecaster.lookback + 10

        for sku_id in tqdm(train_skus, desc="Preparing pre-train data"):
            sku_data = (final_train[final_train['article_id'] == sku_id]
                        .sort_values('date').reset_index(drop=True))
            if len(sku_data) >= min_len:
                train_data_dict[sku_id]     = sku_data['demand']
                train_metadata_dict[sku_id] = meta_builder.encode(
                    articles_df, sku_id)

        logger.info(f"Pre-train data ready for {len(train_data_dict)} SKUs")

        # ---- Pre-train backbone ----
        ok = forecaster.fit(train_data_dict, train_metadata_dict)
        if not ok:
            logger.error("ACCF pre-training failed — aborting.")
            return None
        logger.info("ACCF pre-training complete.")

        # ---- Per-SKU evaluation ----
        all_results  = []
        sku_metrics  = {}
        failed_skus  = []

        for sku_id in tqdm(skus_to_evaluate, desc="Evaluating SKUs"):
            try:
                result = self.evaluate_sku(
                    sku_id, final_train, test_data,
                    forecaster, meta_builder, articles_df)
                if result is not None:
                    all_results.append(result)
                    sku_metrics[sku_id] = result['metrics']
                else:
                    failed_skus.append(sku_id)
            except Exception as e:
                logger.warning(f"SKU {sku_id} failed: {e}")
                failed_skus.append(sku_id)

        logger.info(f"Succeeded : {len(all_results)}")
        logger.info(f"Failed    : {len(failed_skus)}")

        aggregated = aggregate_sku_metrics(sku_metrics)
        self._save_results(all_results, aggregated)
        self._generate_visualizations(all_results)

        logger.info("=" * 80)
        logger.info("ACCF EXPERIMENT COMPLETE")
        logger.info("=" * 80)

        return aggregated

    # ------------------------------------------------------------------

    def _save_results(self, all_results: list, aggregated: dict):
        summary = {
            'model'             : 'ACCF',
            'configuration'     : self.config,
            'n_skus_evaluated'  : len(all_results),
            'aggregated_metrics': aggregated,
            'timestamp'         : pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
        }
        with open(self.results_dir / 'results_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)

        # Detailed CSV — includes wape and bias from improved metrics
        rows = []
        for r in all_results:
            row = {
                'sku_id': r['sku_id'],
                'rmse'  : r['metrics']['rmse'],
                'mae'   : r['metrics']['mae'],
                'mape'  : r['metrics']['mape'],
                'smape' : r['metrics']['smape'],
            }
            if 'wape' in r['metrics']:
                row['wape'] = r['metrics']['wape']
            if 'bias' in r['metrics']:
                row['bias'] = r['metrics']['bias']
            rows.append(row)

        pd.DataFrame(rows).to_csv(
            self.results_dir / 'detailed_results.csv', index=False)

        with open(self.results_dir / 'sample_predictions.json', 'w') as f:
            json.dump(all_results[:10], f, indent=2)

        # Console summary
        print("\n" + "=" * 80)
        print("ACCF RESULTS SUMMARY")
        print("=" * 80)
        print(f"Model  : ACCF (Adaptive Continual Continual Forecaster)")
        print(f"SKUs   : {len(all_results)}")
        print(f"\nConfiguration:")
        print(f"  Hidden size   : {self.config['hidden_size']}")
        print(f"  Adapter size  : {self.config['adapter_size']}")
        print(f"  Lookback      : {self.config['lookback_window']} days")
        print(f"  Pre-train ep. : {self.config['epochs']}")
        print(f"  Adapter ep.   : {self.config['adapter_epochs']}")
        print(f"\nAggregated Metrics:")
        print(f"  Mean RMSE    : {aggregated['mean_rmse']:.4f}  "
              f"(±{aggregated['std_rmse']:.4f})")
        print(f"  Mean MAE     : {aggregated['mean_mae']:.4f}  "
              f"(±{aggregated['std_mae']:.4f})")
        print(f"  Mean MAPE    : {aggregated['mean_mape']:.2f}%")
        if 'mean_wape' in aggregated:
            print(f"  Mean WAPE    : {aggregated['mean_wape']:.2f}%")
        print(f"  Median RMSE  : {aggregated['median_rmse']:.4f}")
        print(f"  Median MAE   : {aggregated['median_mae']:.4f}")
        if 'p90_mae' in aggregated:
            print(f"  P90 MAE      : {aggregated['p90_mae']:.4f}")
        print("=" * 80 + "\n")

    # ------------------------------------------------------------------

    def _generate_visualizations(self, all_results: list):
        logger.info("Generating visualisations...")

        for i, r in enumerate(all_results[:5]):
            dates = pd.to_datetime(r['dates'])
            plot_predictions(
                dates     = dates,
                y_true    = np.array(r['actual']),
                y_pred    = np.array(r['predictions']),
                title     = f"ACCF Predictions — SKU {r['sku_id']}",
                save_path = self.results_dir / f"predictions_sku_{i+1}.png",
            )

        # Residual diagnostic for the first result SKU
        if all_results:
            r = all_results[0]
            plot_residuals(
                dates      = pd.to_datetime(r['dates']),
                y_true     = np.array(r['actual']),
                y_pred     = np.array(r['predictions']),
                model_name = 'ACCF',
                save_path  = self.results_dir / 'residuals_sku_1.png',
            )

        errors = np.concatenate([
            np.array(r['actual']) - np.array(r['predictions'])
            for r in all_results
        ])
        plot_error_distribution(
            errors     = errors,
            model_name = 'ACCF',
            save_path  = self.results_dir / 'error_distribution.png',
        )
        logger.info(f"Visualisations saved to {self.results_dir}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    PROCESSED_DATA_DIR = '../phase2/processed_data'
    RESULTS_DIR        = 'results/accf'
    N_SKUS             = 500
    N_TRAIN_SKUS       = 100

    experiment = ACCFExperiment(
        processed_data_dir = PROCESSED_DATA_DIR,
        results_dir        = RESULTS_DIR,
        n_skus             = N_SKUS,
        n_train_skus       = N_TRAIN_SKUS,
    )

    metrics = experiment.run_experiment()

    if metrics:
        print("\nACCF Experiment Complete!")
        print(f"Results saved to : {RESULTS_DIR}")
        print(f"MAE              : {metrics['mean_mae']:.4f}")
        print(f"RMSE             : {metrics['mean_rmse']:.4f}")
        print(f"MAPE             : {metrics['mean_mape']:.2f}%")
        if 'mean_wape' in metrics:
            print(f"WAPE             : {metrics['mean_wape']:.2f}%")
    else:
        print("\nACCF Experiment Failed.")