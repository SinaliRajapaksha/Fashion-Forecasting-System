"""
ARIMA Baseline Model for Fashion Demand Forecasting
Improved: auto-ARIMA, parallelization, edge case handling, Croston's method
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import warnings
from tqdm import tqdm
import logging
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

import pmdarima as pm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tools.sm_exceptions import ConvergenceWarning
import sys
sys.path.append('.')

from utils.data_loader import BaselineDataLoader
from utils.metrics import calculate_forecasting_metrics, aggregate_sku_metrics
from utils.visualization import plot_predictions, plot_error_distribution

warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=ConvergenceWarning)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Demand pattern classification
# ---------------------------------------------------------------------------

class DemandClassifier:
    """
    Classifies a demand series into one of four patterns so the right
    forecasting strategy is selected automatically.

    Patterns (Syntetos-Boylan quadrant):
        - SMOOTH       : regular, low variability
        - ERRATIC      : regular but high variability
        - INTERMITTENT : sparse but stable when non-zero
        - LUMPY        : sparse AND high variability  ← hardest to forecast
    """

    SMOOTH       = 'smooth'
    ERRATIC      = 'erratic'
    INTERMITTENT = 'intermittent'
    LUMPY        = 'lumpy'
    ZERO         = 'zero'

    # Thresholds (Syntetos & Boylan, 2005)
    ADI_THRESHOLD = 1.32   # Average Demand Interval
    CV2_THRESHOLD = 0.49   # Squared Coefficient of Variation

    @staticmethod
    def classify(series: np.ndarray) -> str:
        """Return demand pattern label for a 1-D demand array."""
        if series.sum() == 0:
            return DemandClassifier.ZERO

        non_zero = series[series > 0]

        # ADI: average gap between non-zero observations
        intervals = np.diff(np.where(series > 0)[0])
        adi = intervals.mean() + 1 if len(intervals) > 0 else 1.0

        # CV²: squared coefficient of variation of non-zero demand
        cv2 = (non_zero.std() / non_zero.mean()) ** 2 if non_zero.mean() > 0 else 0.0

        if adi <= DemandClassifier.ADI_THRESHOLD and cv2 <= DemandClassifier.CV2_THRESHOLD:
            return DemandClassifier.SMOOTH
        elif adi <= DemandClassifier.ADI_THRESHOLD and cv2 > DemandClassifier.CV2_THRESHOLD:
            return DemandClassifier.ERRATIC
        elif adi > DemandClassifier.ADI_THRESHOLD and cv2 <= DemandClassifier.CV2_THRESHOLD:
            return DemandClassifier.INTERMITTENT
        else:
            return DemandClassifier.LUMPY


# ---------------------------------------------------------------------------
# Specialised fallback forecasters
# ---------------------------------------------------------------------------

class CrostonForecaster:
    """
    Croston's method for intermittent / lumpy demand.
    Separately smooths demand size and inter-arrival interval,
    then divides to get the per-period rate.

    Reference: Croston (1972), Syntetos & Boylan (2005) bias correction.
    """

    def __init__(self, alpha: float = 0.1, use_sba: bool = True):
        """
        Args:
            alpha    : Smoothing constant (0 < alpha < 1)
            use_sba  : Apply Syntetos-Boylan-Approximation bias correction
        """
        self.alpha   = alpha
        self.use_sba = use_sba
        self._rate   = None          # estimated demand per period

    def fit(self, series: np.ndarray) -> 'CrostonForecaster':
        non_zero_idx = np.where(series > 0)[0]

        if len(non_zero_idx) == 0:
            self._rate = 0.0
            return self

        # Initialise with first non-zero observation
        z  = float(series[non_zero_idx[0]])   # smoothed demand size
        p  = float(non_zero_idx[0] + 1)       # smoothed interval

        for k in range(1, len(non_zero_idx)):
            i          = non_zero_idx[k]
            prev_i     = non_zero_idx[k - 1]
            interval_k = float(i - prev_i)
            demand_k   = float(series[i])

            z = self.alpha * demand_k + (1 - self.alpha) * z
            p = self.alpha * interval_k + (1 - self.alpha) * p

        # SBA bias correction: multiply rate by (1 - alpha/2)
        rate = z / p
        if self.use_sba:
            rate *= (1 - self.alpha / 2)

        self._rate = max(rate, 0.0)
        return self

    def predict(self, steps: int) -> np.ndarray:
        if self._rate is None:
            raise RuntimeError("Call fit() before predict().")
        return np.full(steps, self._rate)


class SeasonalNaiveForecaster:
    """
    Seasonal Naïve: repeat the last observed season.
    Used as last-resort fallback for very short or constant series.
    """

    def __init__(self, period: int = 7):
        self.period = period
        self._last_season: Optional[np.ndarray] = None

    def fit(self, series: np.ndarray) -> 'SeasonalNaiveForecaster':
        if len(series) >= self.period:
            self._last_season = series[-self.period:].astype(float)
        else:
            # Not enough data – use the whole series tiled
            self._last_season = series.astype(float)
        return self

    def predict(self, steps: int) -> np.ndarray:
        if self._last_season is None:
            return np.zeros(steps)
        reps   = int(np.ceil(steps / len(self._last_season)))
        tiled  = np.tile(self._last_season, reps)
        return tiled[:steps]


# ---------------------------------------------------------------------------
# Preprocessing helpers
# ---------------------------------------------------------------------------

def winsorise(series: pd.Series, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
    """
    Clip extreme values to the [lower, upper] quantile range.
    Reduces the influence of demand spikes on ARIMA parameter estimation.
    """
    lo = series.quantile(lower)
    hi = series.quantile(upper)
    return series.clip(lower=lo, upper=hi)


def detect_seasonality(series: np.ndarray, period: int = 7) -> bool:
    """
    Simple autocorrelation-based seasonality check.
    Returns True if the lag-period autocorrelation exceeds 0.20.
    """
    if len(series) < 2 * period:
        return False
    try:
        acf_val = pd.Series(series).autocorr(lag=period)
        return bool(acf_val > 0.20)
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Per-SKU worker (must be top-level for multiprocessing pickling)
# ---------------------------------------------------------------------------

def _evaluate_sku_worker(args: Tuple) -> Dict:
    """
    Standalone worker function evaluated in a child process.
    All imports are local so pickling works cleanly on Linux/macOS.
    """
    (sku_id,
     train_values, train_dates,
     test_values,  test_dates,
     min_train_len,
     arima_max_p, arima_max_q,
     forecast_horizon) = args

    import warnings, numpy as np, pandas as pd
    from statsmodels.tools.sm_exceptions import ConvergenceWarning
    warnings.filterwarnings('ignore')
    warnings.filterwarnings('ignore', category=ConvergenceWarning)

    # ---- local copies of helpers (avoid cross-process import issues) ------

    def _winsorise(arr, lo=0.01, hi=0.99):
        s  = pd.Series(arr)
        return s.clip(lower=s.quantile(lo), upper=s.quantile(hi)).values

    def _detect_seasonality(arr, period=7):
        if len(arr) < 2 * period:
            return False
        try:
            return pd.Series(arr).autocorr(lag=period) > 0.20
        except Exception:
            return False

    def _classify(arr):
        if arr.sum() == 0:
            return 'zero'
        nz        = arr[arr > 0]
        intervals = np.diff(np.where(arr > 0)[0])
        adi       = intervals.mean() + 1 if len(intervals) > 0 else 1.0
        cv2       = (nz.std() / nz.mean()) ** 2 if nz.mean() > 0 else 0.0
        if adi <= 1.32 and cv2 <= 0.49:  return 'smooth'
        if adi <= 1.32 and cv2 >  0.49:  return 'erratic'
        if adi >  1.32 and cv2 <= 0.49:  return 'intermittent'
        return 'lumpy'

    def _croston(arr, alpha=0.1, sba=True):
        nz_idx = np.where(arr > 0)[0]
        if len(nz_idx) == 0:
            return 0.0
        z = float(arr[nz_idx[0]])
        p = float(nz_idx[0] + 1)
        for k in range(1, len(nz_idx)):
            i  = nz_idx[k]
            pi = nz_idx[k - 1]
            z  = alpha * float(arr[i])    + (1 - alpha) * z
            p  = alpha * float(i - pi)    + (1 - alpha) * p
        rate = z / p
        if sba:
            rate *= (1 - alpha / 2)
        return max(rate, 0.0)

    def _seasonal_naive(arr, steps, period=7):
        season = arr[-period:] if len(arr) >= period else arr
        reps   = int(np.ceil(steps / len(season)))
        return np.tile(season, reps)[:steps].astype(float)

    def _safe_metrics(actual, predicted):
        """Compute MAE / RMSE / MAPE / sMAPE safely."""
        actual    = np.array(actual,    dtype=float)
        predicted = np.array(predicted, dtype=float)
        predicted = np.maximum(predicted, 0)          # non-negative demand

        mae  = float(np.mean(np.abs(actual - predicted)))
        rmse = float(np.sqrt(np.mean((actual - predicted) ** 2)))

        mask = actual > 0
        mape  = float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100) if mask.any() else 0.0
        denom = (np.abs(actual) + np.abs(predicted)) / 2
        smask = denom > 0
        smape = float(np.mean(np.abs(actual[smask] - predicted[smask]) / denom[smask]) * 100) if smask.any() else 0.0

        return {'mae': mae, 'rmse': rmse, 'mape': mape, 'smape': smape}

    # -----------------------------------------------------------------------
    train_arr = np.array(train_values, dtype=float)
    test_arr  = np.array(test_values,  dtype=float)
    steps     = len(test_arr)

    result_base = {
        'sku_id'      : sku_id,
        'actual'      : test_arr.tolist(),
        'dates'       : test_dates,
        'model_fitted': False,
        'strategy'    : 'unknown',
    }

    # ------------------------------------------------------------------
    # Guard: too short to model
    # ------------------------------------------------------------------
    if len(train_arr) < min_train_len:
        preds = _seasonal_naive(train_arr, steps)
        return {**result_base,
                'predictions': preds.tolist(),
                'metrics'    : _safe_metrics(test_arr, preds),
                'strategy'   : 'seasonal_naive_short',
                'model_fitted': False}

    # ------------------------------------------------------------------
    # Classify demand pattern
    # ------------------------------------------------------------------
    pattern = _classify(train_arr)

    # ------------------------------------------------------------------
    # Zero-demand series → predict zeros
    # ------------------------------------------------------------------
    if pattern == 'zero':
        preds = np.zeros(steps)
        return {**result_base,
                'predictions': preds.tolist(),
                'metrics'    : _safe_metrics(test_arr, preds),
                'strategy'   : 'zero_demand',
                'model_fitted': True}

    # ------------------------------------------------------------------
    # Intermittent / Lumpy → Croston's method
    # ------------------------------------------------------------------
    if pattern in ('intermittent', 'lumpy'):
        rate  = _croston(train_arr)
        preds = np.full(steps, rate)
        return {**result_base,
                'predictions': preds.tolist(),
                'metrics'    : _safe_metrics(test_arr, preds),
                'strategy'   : f'croston_{pattern}',
                'model_fitted': True}

    # ------------------------------------------------------------------
    # Smooth / Erratic → Auto-ARIMA (with optional SARIMA)
    # ------------------------------------------------------------------
    import pmdarima as pm

    # Winsorise before fitting (reduces spike influence)
    train_clean = _winsorise(train_arr)

    has_seasonality = _detect_seasonality(train_arr)
    seasonal        = has_seasonality
    m               = 7 if has_seasonality else 1

    try:
        auto_model = pm.auto_arima(
            train_clean,
            start_p=0, max_p=arima_max_p,
            start_q=0, max_q=arima_max_q,
            d=None,                    # auto-select differencing
            seasonal=seasonal,
            m=m,
            D=1 if seasonal else None,
            max_P=1, max_Q=1,          # keep SARIMA lightweight
            information_criterion='aic',
            stepwise=True,             # stepwise search = much faster
            suppress_warnings=True,
            error_action='ignore',
            n_jobs=1,                  # already parallelised at SKU level
        )

        raw_preds = auto_model.predict(n_periods=steps)
        # Clip to non-negative and round to nearest integer (discrete demand)
        preds = np.maximum(np.round(raw_preds), 0).astype(float)

        order     = auto_model.order
        s_order   = auto_model.seasonal_order if seasonal else None
        strategy  = f'auto_arima({"S" if seasonal else ""}ARIMA{order}{"x" + str(s_order) if s_order else ""})'

        return {**result_base,
                'predictions' : preds.tolist(),
                'metrics'     : _safe_metrics(test_arr, preds),
                'strategy'    : strategy,
                'model_fitted': True,
                'arima_order' : order,
                'seasonal_order': s_order}

    except Exception as e:
        # Fallback: seasonal naive when auto-ARIMA fails completely
        preds = _seasonal_naive(train_arr, steps)
        return {**result_base,
                'predictions': preds.tolist(),
                'metrics'    : _safe_metrics(test_arr, preds),
                'strategy'   : f'seasonal_naive_fallback({str(e)[:60]})',
                'model_fitted': False}


# ---------------------------------------------------------------------------
# Main experiment class
# ---------------------------------------------------------------------------

class ARIMABaselineExperiment:
    """
    Complete ARIMA baseline experiment with:
      - Demand pattern classification (smooth / erratic / intermittent / lumpy)
      - Auto-ARIMA order selection per SKU
      - Croston's method for intermittent / lumpy demand
      - Seasonal Naïve fallback for very short or zero series
      - Parallel SKU evaluation via ProcessPoolExecutor
      - Outlier-robust pre-processing (IQR winsorisation)
    """

    def __init__(self,
                 processed_data_dir : str = '../phase2/processed_data',
                 results_dir        : str = 'results/arima',
                 n_skus             : int = None,
                 n_workers          : int = None,
                 min_train_len      : int = 14,
                 arima_max_p        : int = 3,
                 arima_max_q        : int = 3):
        """
        Args:
            processed_data_dir : Path to processed data directory
            results_dir        : Directory to save results
            n_skus             : SKUs to evaluate (None = all)
            n_workers          : Parallel workers (None = CPU count − 1)
            min_train_len      : Minimum training days before modelling
            arima_max_p        : Maximum AR order for auto-ARIMA search
            arima_max_q        : Maximum MA order for auto-ARIMA search
        """
        self.data_loader    = BaselineDataLoader(processed_data_dir)
        self.results_dir    = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.n_skus         = n_skus
        self.n_workers      = n_workers or max(1, multiprocessing.cpu_count() - 1)
        self.min_train_len  = min_train_len
        self.arima_max_p    = arima_max_p
        self.arima_max_q    = arima_max_q

        logger.info(f"Initialised improved ARIMA experiment")
        logger.info(f"Workers : {self.n_workers}")
        logger.info(f"Results : {self.results_dir}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_experiment(self) -> Dict:
        """Run the complete experiment and return aggregated metrics."""

        logger.info("=" * 80)
        logger.info("STARTING IMPROVED ARIMA BASELINE EXPERIMENT")
        logger.info("=" * 80)

        # Load data
        train_data, val_data, test_data = self.data_loader.get_train_test_split()

        # Determine SKUs to evaluate
        if self.n_skus is not None:
            skus = self.data_loader.get_sample_skus(n=self.n_skus)
        else:
            skus = self.data_loader.get_all_skus()

        logger.info(f"SKUs to evaluate: {len(skus)}")

        # Build per-SKU argument tuples for the worker
        worker_args = self._build_worker_args(skus, train_data, test_data)

        # Parallel evaluation
        all_results, failed_skus = self._parallel_evaluate(worker_args)

        logger.info(f"Succeeded : {len(all_results)} SKUs")
        logger.info(f"Failed    : {len(failed_skus)} SKUs")

        # Aggregate metrics
        sku_metrics        = {r['sku_id']: r['metrics'] for r in all_results}
        aggregated_metrics = aggregate_sku_metrics(sku_metrics)

        # Strategy breakdown
        strategy_counts = {}
        for r in all_results:
            key = r.get('strategy', 'unknown').split('(')[0]   # strip params
            strategy_counts[key] = strategy_counts.get(key, 0) + 1

        # Save + visualise
        self._save_results(all_results, aggregated_metrics, strategy_counts)
        self._generate_visualizations(all_results)

        logger.info("=" * 80)
        logger.info("EXPERIMENT COMPLETE")
        logger.info("=" * 80)

        return aggregated_metrics

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_worker_args(self,
                           skus      : List[str],
                           train_data: pd.DataFrame,
                           test_data : pd.DataFrame) -> List[Tuple]:
        """Pre-package per-SKU data into picklable tuples."""
        args = []
        for sku in skus:
            tr = train_data[train_data['article_id'] == sku].sort_values('date')
            te = test_data [test_data ['article_id'] == sku].sort_values('date')

            if len(te) == 0:
                continue

            args.append((
                sku,
                tr['demand'].values.tolist(),
                tr['date'].dt.strftime('%Y-%m-%d').tolist(),
                te['demand'].values.tolist(),
                te['date'].dt.strftime('%Y-%m-%d').tolist(),
                self.min_train_len,
                self.arima_max_p,
                self.arima_max_q,
                len(te),             # forecast_horizon
            ))
        return args

    def _parallel_evaluate(self,
                            worker_args: List[Tuple]) -> Tuple[List[Dict], List[str]]:
        """Dispatch SKU evaluation across worker pool."""
        all_results  = []
        failed_skus  = []

        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            futures = {executor.submit(_evaluate_sku_worker, a): a[0]
                       for a in worker_args}

            for fut in tqdm(as_completed(futures),
                            total=len(futures),
                            desc="Evaluating SKUs"):
                sku_id = futures[fut]
                try:
                    result = fut.result()
                    all_results.append(result)
                except Exception as e:
                    logger.warning(f"SKU {sku_id} worker crashed: {e}")
                    failed_skus.append(sku_id)

        return all_results, failed_skus

    def _save_results(self,
                      all_results       : List[Dict],
                      aggregated_metrics: Dict,
                      strategy_counts   : Dict):
        """Persist summary JSON and detailed CSV."""

        summary = {
            'model'             : 'ARIMA (improved)',
            'variant'           : 'auto_arima + croston + seasonal_naive',
            'n_skus_evaluated'  : len(all_results),
            'aggregated_metrics': aggregated_metrics,
            'strategy_breakdown': strategy_counts,
            'timestamp'         : pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
        }

        with open(self.results_dir / 'results_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)

        # Detailed per-SKU CSV
        rows = []
        for r in all_results:
            rows.append({
                'sku_id'      : r['sku_id'],
                'rmse'        : r['metrics']['rmse'],
                'mae'         : r['metrics']['mae'],
                'mape'        : r['metrics']['mape'],
                'smape'       : r['metrics']['smape'],
                'strategy'    : r.get('strategy', ''),
                'model_fitted': r.get('model_fitted', False),
            })

        pd.DataFrame(rows).to_csv(self.results_dir / 'detailed_results.csv', index=False)

        # Sample predictions (first 10)
        sample = all_results[:10]
        with open(self.results_dir / 'sample_predictions.json', 'w') as f:
            json.dump(sample, f, indent=2)

        # Console summary
        print("\n" + "=" * 80)
        print("IMPROVED ARIMA BASELINE — RESULTS SUMMARY")
        print("=" * 80)
        print(f"SKUs Evaluated  : {len(all_results)}")
        print(f"\nAggregated Metrics:")
        print(f"  Mean RMSE    : {aggregated_metrics['mean_rmse']:.4f}  (±{aggregated_metrics['std_rmse']:.4f})")
        print(f"  Mean MAE     : {aggregated_metrics['mean_mae']:.4f}  (±{aggregated_metrics['std_mae']:.4f})")
        print(f"  Mean MAPE    : {aggregated_metrics['mean_mape']:.2f}%")
        print(f"  Median RMSE  : {aggregated_metrics['median_rmse']:.4f}")
        print(f"  Median MAE   : {aggregated_metrics['median_mae']:.4f}")
        print(f"\nStrategy Breakdown:")
        for strategy, count in sorted(strategy_counts.items(), key=lambda x: -x[1]):
            pct = count / len(all_results) * 100
            print(f"  {strategy:<45} {count:>5}  ({pct:.1f}%)")
        print("=" * 80 + "\n")

    def _generate_visualizations(self, all_results: List[Dict]):
        """Prediction plots + error distribution."""
        logger.info("Generating visualisations...")

        for i, result in enumerate(all_results[:5]):
            dates    = pd.to_datetime(result['dates'])
            actual   = np.array(result['actual'])
            predicted = np.array(result['predictions'])

            plot_predictions(
                dates     = dates,
                y_true    = actual,
                y_pred    = predicted,
                title     = f"ARIMA Predictions - SKU {result['sku_id']} [{result.get('strategy','')}]",
                save_path = self.results_dir / f"predictions_sku_{i+1}.png"
            )

        all_errors = []
        for r in all_results:
            actual    = np.array(r['actual'])
            predicted = np.array(r['predictions'])
            all_errors.extend((actual - predicted).tolist())

        plot_error_distribution(
            errors     = np.array(all_errors),
            model_name = 'ARIMA (improved)',
            save_path  = self.results_dir / 'error_distribution.png'
        )

        logger.info(f"Visualisations saved to {self.results_dir}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    PROCESSED_DATA_DIR = '../phase2/processed_data'
    RESULTS_DIR        = 'results/arima'
    N_SKUS             = None    # None = all SKUs

    experiment = ARIMABaselineExperiment(
        processed_data_dir = PROCESSED_DATA_DIR,
        results_dir        = RESULTS_DIR,
        n_skus             = N_SKUS,
        n_workers          = None,      # auto-detect CPU count
        min_train_len      = 14,        # SKUs with < 14 training days → seasonal naïve
        arima_max_p        = 3,         # increase to 5 for more accuracy (slower)
        arima_max_q        = 3,
    )

    metrics = experiment.run_experiment()

    print("\nImproved ARIMA Baseline Complete!")
    print(f"Results saved to: {RESULTS_DIR}")