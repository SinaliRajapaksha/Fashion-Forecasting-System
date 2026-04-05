"""
Comprehensive Model Comparison Script — IMPROVED
Compares all baseline models + ACCF with detailed metrics and visualizations

Improvements over the original:
  1. Bug fix: best-bar highlight       — original used idxmin() (label) as a
                                         positional bar index; now uses iloc
                                         position to avoid wrong-bar highlight
  2. Bug fix: ACCF-missing guard       — create_detailed_report no longer
                                         crashes if ACCF results are absent
  3. Missing-model reporting           — load_all_results clearly separates
                                         loaded vs missing models and returns
                                         a status dict alongside results
  4. WAPE + sMAPE + p90 MAE columns   — pulled from improved metrics.py output
  5. DRY plotting                      — single _plot_metric_bars() helper
                                         replaces three near-identical functions
  6. Per-SKU distribution box plot     — loads detailed_results.csv from each
                                         model; compares error distributions,
                                         not just means
  7. Radar / spider chart              — multi-metric tradeoff in one figure,
                                         standard in thesis comparison sections
  8. Cold-start vs warm-start split    — separate MAE rows for warm/cold SKUs
                                         using the sku_split from data loader
  9. ACCF improvement % vs baselines   — computed and printed in console + report
  10. No global rcParams mutation      — all figure sizes set per-figure
"""

import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings('ignore')
sns.set_style("whitegrid")

_SAVE_DPI  = 300
_PALETTE   = sns.color_palette('tab10', n_colors=10)

# Consistent model → colour mapping so every chart uses the same colours
_MODEL_COLORS = {
    'ARIMA'  : _PALETTE[0],
    'XGBoost': _PALETTE[1],
    'LSTM'   : _PALETTE[2],
    'FSNet'  : _PALETTE[3],
    'CLeaR'  : _PALETTE[4],
    'ACCF'   : _PALETTE[5],
}
_ACCF_HIGHLIGHT = '#e74c3c'   # red  — best model highlight
_BEST_HIGHLIGHT = '#2ecc71'   # green — ACCF highlight when not best


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _save(fig: plt.Figure, path: Path, label: str):
    fig.savefig(path, dpi=_SAVE_DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path.name}")


def _style_ax(ax: plt.Axes, title: str,
              xlabel: str = '', ylabel: str = ''):
    ax.set_title(title, fontsize=13, pad=10)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=11)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=11)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3, linewidth=0.7)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class ModelComparison:
    """
    Load, compare, and visualise results from all six models.
    Produces charts and a Markdown report suitable for a thesis.
    """

    MODEL_ORDER = ['ARIMA', 'XGBoost', 'LSTM', 'CLeaR', 'FSNet', 'ACCF']

    MODEL_TYPES = {
        'ARIMA'  : 'Statistical',
        'XGBoost': 'Machine Learning',
        'LSTM'   : 'Deep Learning',
        'FSNet'  : 'Continual Learning',
        'CLeaR'  : 'Continual Learning',
        'ACCF'   : 'Proposed (CL + Adapters)',
    }

    def __init__(self,
                 phase3_results_dir: str = '../phase3_baselines/results',
                 phase4_results_dir: str = 'results',
                 sku_split_path    : str = '../phase2/processed_data/sku_split.json'):

        self.phase3_dir   = Path(phase3_results_dir)
        self.phase4_dir   = Path(phase4_results_dir)
        self.output_dir   = Path('evaluation_report')
        self.output_dir.mkdir(exist_ok=True)

        self.model_dirs = {
            'ARIMA'  : self.phase3_dir / 'arima',
            'XGBoost': self.phase3_dir / 'xgboost',
            'LSTM'   : self.phase3_dir / 'lstm',
            'FSNet'  : self.phase3_dir / 'fsnet',
            'CLeaR'  : self.phase3_dir / 'clear',
            'ACCF'   : self.phase4_dir / 'accf',
        }

        # Optional: warm/cold SKU split for stratified comparison
        self._warm_skus: Optional[set] = None
        self._cold_skus: Optional[set] = None
        try:
            with open(sku_split_path) as f:
                split = json.load(f)
            self._warm_skus = set(split.get('warm_start_skus', []))
            self._cold_skus = set(split.get('cold_start_skus',  []))
        except FileNotFoundError:
            pass   # silently skip stratified analysis if file absent

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load_all_results(self) -> Tuple[Dict, List[str]]:
        """
        Load results_summary.json for every model.

        Returns
        -------
        results  : {model_name: summary_dict}  — only models with data
        missing  : list of model names whose JSON was not found
        """
        results = {}
        missing = []

        for name in self.MODEL_ORDER:
            path = self.model_dirs[name] / 'results_summary.json'
            if path.exists():
                with open(path) as f:
                    results[name] = json.load(f)
            else:
                missing.append(name)
                print(f"  [MISSING] {name} — {path}")

        return results, missing

    def load_detailed_csvs(self) -> Dict[str, pd.DataFrame]:
        """Load per-SKU detailed_results.csv for distribution plots."""
        dfs = {}
        for name in self.MODEL_ORDER:
            path = self.model_dirs[name] / 'detailed_results.csv'
            if path.exists():
                df = pd.read_csv(path)
                df['model'] = name
                dfs[name] = df
        return dfs

    # ------------------------------------------------------------------
    # Comparison table
    # ------------------------------------------------------------------

    def create_comparison_table(self, results: Dict) -> pd.DataFrame:
        """
        Build the main comparison DataFrame.
        Includes WAPE, sMAPE, p90 MAE when available (from improved metrics.py).
        """
        rows = []
        for name in self.MODEL_ORDER:
            if name not in results:
                continue
            m = results[name]['aggregated_metrics']
            row = {
                'Model'          : name,
                'Type'           : self.MODEL_TYPES.get(name, ''),
                'Mean MAE'       : m.get('mean_mae',    np.nan),
                'Std MAE'        : m.get('std_mae',     np.nan),
                'Median MAE'     : m.get('median_mae',  np.nan),
                'P90 MAE'        : m.get('p90_mae',     np.nan),
                'Mean RMSE'      : m.get('mean_rmse',   np.nan),
                'Std RMSE'       : m.get('std_rmse',    np.nan),
                'Median RMSE'    : m.get('median_rmse', np.nan),
                'Mean MAPE (%)'  : m.get('mean_mape',   np.nan),
                'Mean WAPE (%)'  : m.get('mean_wape',   np.nan),
                'Mean sMAPE (%)' : m.get('mean_smape',  np.nan),
                'SKUs Evaluated' : results[name].get('n_skus_evaluated', np.nan),
            }
            rows.append(row)

        df = (pd.DataFrame(rows)
              .sort_values('Mean MAE', na_position='last')
              .reset_index(drop=True))
        df.insert(0, 'Rank', range(1, len(df) + 1))
        return df

    # ------------------------------------------------------------------
    # Stratified (cold-start / warm-start) table
    # ------------------------------------------------------------------

    def create_stratified_table(self,
                                 detailed: Dict[str, pd.DataFrame]
                                 ) -> Optional[pd.DataFrame]:
        """
        Per-model MAE split by warm-start vs cold-start SKUs.
        Returns None if sku_split.json was not loaded.
        """
        if not self._warm_skus and not self._cold_skus:
            return None

        rows = []
        for name, df in detailed.items():
            if 'sku_id' not in df.columns or 'mae' not in df.columns:
                continue
            df['sku_id'] = df['sku_id'].astype(str)
            warm = df[df['sku_id'].isin(self._warm_skus)]['mae']
            cold = df[df['sku_id'].isin(self._cold_skus)]['mae']
            rows.append({
                'Model'          : name,
                'Warm MAE (mean)': warm.mean() if len(warm) else np.nan,
                'Cold MAE (mean)': cold.mean() if len(cold) else np.nan,
                'Warm SKUs'      : len(warm),
                'Cold SKUs'      : len(cold),
            })

        if not rows:
            return None
        return pd.DataFrame(rows).sort_values('Warm MAE (mean)')

    # ------------------------------------------------------------------
    # ACCF improvement summary
    # ------------------------------------------------------------------

    def compute_improvements(self, df: pd.DataFrame) -> Dict:
        """
        Compute how much ACCF improves over each baseline in MAE.
        Returns dict of {model: pct_improvement} (positive = ACCF is better).
        """
        if 'ACCF' not in df['Model'].values:
            return {}
        accf_mae = df.loc[df['Model'] == 'ACCF', 'Mean MAE'].iloc[0]
        out = {}
        for _, row in df[df['Model'] != 'ACCF'].iterrows():
            baseline = row['Mean MAE']
            if pd.notna(baseline) and baseline > 0:
                out[row['Model']] = (baseline - accf_mae) / baseline * 100
        return out

    # ------------------------------------------------------------------
    # DRY single-metric bar chart
    # ------------------------------------------------------------------

    def _plot_metric_bars(self,
                          df        : pd.DataFrame,
                          col       : str,
                          std_col   : Optional[str],
                          title     : str,
                          ylabel    : str,
                          fname     : str,
                          fmt       : str = '{:.4f}'):
        """
        Generic bar chart for one metric across all models.
        Highlights ACCF in green and the best model in red.
        Uses iloc for bar indexing — no label/position mismatch.
        """
        sub = df[df[col].notna()].reset_index(drop=True)
        if sub.empty:
            return

        models = sub['Model'].tolist()
        vals   = sub[col].values
        errs   = sub[std_col].values if std_col and std_col in sub else None

        # Best = lowest (all metrics are "lower is better")
        best_pos = int(np.nanargmin(vals))

        colors = [_MODEL_COLORS.get(m, '#aaaaaa') for m in models]

        fig, ax = plt.subplots(figsize=(max(9, len(models) * 1.6), 5))
        bars = ax.bar(models, vals, yerr=errs, capsize=5,
                      color=colors, alpha=0.82, edgecolor='white',
                      error_kw=dict(elinewidth=1.2, ecolor='#444'))

        # Colour overrides
        bars[best_pos].set_color(_ACCF_HIGHLIGHT)
        bars[best_pos].set_alpha(1.0)
        if 'ACCF' in models:
            accf_pos = models.index('ACCF')
            if accf_pos != best_pos:
                bars[accf_pos].set_color(_BEST_HIGHLIGHT)
                bars[accf_pos].set_alpha(1.0)

        # Value labels
        max_err = float(np.nanmax(errs)) if errs is not None else 0.0
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max_err * 0.08 + max(vals) * 0.01,
                    fmt.format(val),
                    ha='center', va='bottom', fontsize=9)

        # Legend patches
        patches = [
            mpatches.Patch(color=_ACCF_HIGHLIGHT, label='Best model'),
            mpatches.Patch(color=_BEST_HIGHLIGHT,  label='ACCF (proposed)'),
        ]
        ax.legend(handles=patches, fontsize=9, framealpha=0.9)

        plt.setp(ax.get_xticklabels(), rotation=35, ha='right')
        _style_ax(ax, title, ylabel=ylabel)
        fig.tight_layout()
        _save(fig, self.output_dir / fname, fname)

    # ------------------------------------------------------------------
    # Per-SKU distribution box plot
    # ------------------------------------------------------------------

    def plot_sku_distributions(self, detailed: Dict[str, pd.DataFrame]):
        """
        Box-and-whisker of per-SKU MAE for each model.
        Shows spread and outliers — means alone are misleading.
        """
        frames = [df[['model', 'mae']].rename(columns={'mae': 'MAE'})
                  for df in detailed.values() if 'mae' in df.columns]
        if not frames:
            return

        combined = pd.concat(frames, ignore_index=True)
        order    = [m for m in self.MODEL_ORDER
                    if m in combined['model'].unique()]
        palette  = {m: _MODEL_COLORS.get(m, '#aaaaaa') for m in order}

        fig, ax = plt.subplots(figsize=(max(10, len(order) * 1.8), 5))
        sns.boxplot(data=combined, x='model', y='MAE',
                    order=order, palette=palette, ax=ax,
                    flierprops=dict(marker='o', markersize=2,
                                    markerfacecolor='grey', alpha=0.4),
                    linewidth=1.1)

        plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
        _style_ax(ax, 'Per-SKU MAE Distribution by Model',
                  ylabel='MAE per SKU (lower is better)')
        fig.tight_layout()
        _save(fig, self.output_dir / 'sku_mae_distribution.png',
              'sku_mae_distribution.png')

    # ------------------------------------------------------------------
    # Radar chart
    # ------------------------------------------------------------------

    def plot_radar(self, df: pd.DataFrame):
        """
        Spider / radar chart comparing all models across five metrics.
        Lower-is-better metrics are inverted so "outward = better" on all axes.
        """
        metrics_cfg = [
            ('Mean MAE',      'MAE\n(inv)'),
            ('Mean RMSE',     'RMSE\n(inv)'),
            ('Mean MAPE (%)', 'MAPE\n(inv)'),
            ('Mean WAPE (%)', 'WAPE\n(inv)'),
            ('P90 MAE',       'P90 MAE\n(inv)'),
        ]

        # Keep only metrics that exist in the data
        available = [(col, lbl) for col, lbl in metrics_cfg
                     if col in df.columns and df[col].notna().any()]
        if len(available) < 3:
            return   # not enough metrics to draw a meaningful radar

        cols, labels = zip(*available)
        n    = len(cols)
        sub  = df[['Model'] + list(cols)].dropna().reset_index(drop=True)
        if sub.empty:
            return

        # Normalise each metric to [0,1] then invert (1 = best)
        norm = sub[list(cols)].copy()
        for c in cols:
            mn, mx = norm[c].min(), norm[c].max()
            if mx > mn:
                norm[c] = 1 - (norm[c] - mn) / (mx - mn)
            else:
                norm[c] = 1.0

        angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
        angles += angles[:1]   # close the polygon

        fig, ax = plt.subplots(figsize=(7, 7),
                               subplot_kw=dict(polar=True))

        for _, row in sub.iterrows():
            model  = row['Model']
            values = norm.loc[row.name, list(cols)].tolist()
            values += values[:1]
            color  = _MODEL_COLORS.get(model, '#aaaaaa')
            lw     = 2.5 if model == 'ACCF' else 1.4
            ax.plot(angles, values, color=color, linewidth=lw, label=model)
            ax.fill(angles, values, color=color,
                    alpha=0.12 if model == 'ACCF' else 0.05)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_yticklabels([])
        ax.set_title('Multi-metric comparison\n(outward = better)',
                     fontsize=12, pad=18)
        ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.15),
                  fontsize=9, framealpha=0.9)

        fig.tight_layout()
        _save(fig, self.output_dir / 'radar_comparison.png',
              'radar_comparison.png')

    # ------------------------------------------------------------------
    # Model-type summary
    # ------------------------------------------------------------------

    def plot_model_types(self, df: pd.DataFrame):
        """Horizontal bar chart: mean MAE and RMSE per model-type category."""
        type_perf = (df.groupby('Type', sort=False)
                     .agg({'Mean MAE': 'mean', 'Mean RMSE': 'mean'})
                     .reset_index()
                     .sort_values('Mean MAE'))

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        palette = sns.color_palette('Set2', n_colors=len(type_perf))

        ax1.barh(type_perf['Type'], type_perf['Mean MAE'],
                 color=palette, alpha=0.85, edgecolor='white')
        for i, (val, y) in enumerate(zip(type_perf['Mean MAE'],
                                          type_perf['Type'])):
            ax1.text(val + type_perf['Mean MAE'].max() * 0.01,
                     i, f'{val:.4f}', va='center', fontsize=9)
        _style_ax(ax1, 'Mean MAE by Model Category', xlabel='Mean MAE')
        ax1.grid(axis='x', alpha=0.3)
        ax1.grid(axis='y', alpha=0)

        ax2.barh(type_perf['Type'], type_perf['Mean RMSE'],
                 color=palette, alpha=0.85, edgecolor='white')
        for i, (val, y) in enumerate(zip(type_perf['Mean RMSE'],
                                          type_perf['Type'])):
            ax2.text(val + type_perf['Mean RMSE'].max() * 0.01,
                     i, f'{val:.4f}', va='center', fontsize=9)
        _style_ax(ax2, 'Mean RMSE by Model Category', xlabel='Mean RMSE')
        ax2.grid(axis='x', alpha=0.3)
        ax2.grid(axis='y', alpha=0)

        fig.tight_layout()
        _save(fig, self.output_dir / 'performance_by_type.png',
              'performance_by_type.png')

    # ------------------------------------------------------------------
    # Markdown report
    # ------------------------------------------------------------------

    def create_detailed_report(self,
                                df          : pd.DataFrame,
                                results     : Dict,
                                improvements: Dict,
                                strat_df    : Optional[pd.DataFrame] = None):
        """Write a structured Markdown evaluation report."""

        lines = []
        T = lambda s: lines.append(s + '\n')

        T('# Fashion Demand Forecasting — Model Evaluation Report')
        T(f'**Generated:** {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")}')
        T('')
        T('---')
        T('')

        # ---- Executive summary ----
        T('## Executive Summary')
        T('')
        best_row  = df.iloc[0]
        n_models  = len(df)

        T(f'- **Best overall model:** {best_row["Model"]} '
          f'(Mean MAE = {best_row["Mean MAE"]:.4f})')

        if 'ACCF' in df['Model'].values:
            accf_row = df[df['Model'] == 'ACCF'].iloc[0]
            T(f'- **ACCF rank:** #{int(accf_row["Rank"])} of {n_models} '
              f'(Mean MAE = {accf_row["Mean MAE"]:.4f})')
        else:
            T('- **ACCF:** results not available')

        T(f'- **Models compared:** {n_models}')
        T(f'- **SKUs evaluated:** {int(df["SKUs Evaluated"].max()):,}')
        T('')

        if improvements:
            T('### ACCF improvement over baselines (Mean MAE)')
            T('')
            T('| Baseline | ACCF improvement |')
            T('|----------|-----------------|')
            for model, pct in sorted(improvements.items(),
                                     key=lambda x: -x[1]):
                sign = '+' if pct >= 0 else ''
                T(f'| {model} | {sign}{pct:.1f}% |')
            T('')

        # ---- Overall performance table ----
        T('## Overall Performance Comparison')
        T('')

        # Dynamic headers based on what columns actually exist
        base_cols = ['Rank', 'Model', 'Type',
                     'Mean MAE', 'Median MAE', 'P90 MAE',
                     'Mean RMSE', 'Mean MAPE (%)', 'Mean WAPE (%)']
        show_cols = [c for c in base_cols if c in df.columns
                     and df[c].notna().any()]

        T('| ' + ' | '.join(show_cols) + ' |')
        T('|' + '|'.join(['---'] * len(show_cols)) + '|')

        for _, row in df.iterrows():
            vals = []
            for c in show_cols:
                v = row[c]
                if c == 'Rank':
                    vals.append(str(int(v)))
                elif c in ('Model', 'Type'):
                    vals.append(str(v))
                elif c == 'Mean MAPE (%)' or c == 'Mean WAPE (%)':
                    vals.append(f'{v:.2f}' if pd.notna(v) else '—')
                else:
                    vals.append(f'{v:.4f}' if pd.notna(v) else '—')
            T('| ' + ' | '.join(vals) + ' |')
        T('')

        # ---- Cold/warm split ----
        if strat_df is not None:
            T('## Warm-start vs Cold-start Performance')
            T('')
            T('| Model | Warm MAE | Cold MAE | Warm SKUs | Cold SKUs |')
            T('|-------|----------|----------|-----------|-----------|')
            for _, row in strat_df.iterrows():
                T(f"| {row['Model']} "
                  f"| {row['Warm MAE (mean)']:.4f} "
                  f"| {row['Cold MAE (mean)']:.4f} "
                  f"| {int(row['Warm SKUs'])} "
                  f"| {int(row['Cold SKUs'])} |")
            T('')

        # ---- Per-model details ----
        T('## Individual Model Details')
        T('')

        for name in self.MODEL_ORDER:
            if name not in results:
                continue
            r      = results[name]
            m      = r['aggregated_metrics']
            config = r.get('configuration', {})

            T(f'### {name}')
            T('')
            T(f'**Category:** {self.MODEL_TYPES.get(name, "Unknown")}  ')
            T('')

            if config:
                T('**Key configuration:**')
                T('')
                for k, v in config.items():
                    if k not in ('name',) and not isinstance(v, dict):
                        T(f'- `{k}`: {v}')
                T('')

            T('**Metrics:**')
            T('')
            T(f'| Metric | Value |')
            T(f'|--------|-------|')
            metric_rows = [
                ('Mean MAE',   f"{m.get('mean_mae',   np.nan):.4f} "
                               f"(±{m.get('std_mae', np.nan):.4f})"),
                ('Median MAE', f"{m.get('median_mae', np.nan):.4f}"),
                ('P90 MAE',    f"{m.get('p90_mae',   np.nan):.4f}"
                               if 'p90_mae' in m else '—'),
                ('Mean RMSE',  f"{m.get('mean_rmse',  np.nan):.4f} "
                               f"(±{m.get('std_rmse', np.nan):.4f})"),
                ('Mean MAPE',  f"{m.get('mean_mape',  np.nan):.2f}%"),
                ('Mean WAPE',  f"{m.get('mean_wape',  np.nan):.2f}%"
                               if 'mean_wape' in m else '—'),
                ('Mean sMAPE', f"{m.get('mean_smape', np.nan):.2f}%"
                               if 'mean_smape' in m else '—'),
            ]
            for metric, val in metric_rows:
                T(f'| {metric} | {val} |')
            T('')

        # ---- ACCF innovation section ----
        T('## ACCF Architecture and Innovation')
        T('')
        T('ACCF (Adaptive Continual Continual Forecaster) distinguishes itself '
          'from the five baselines through three integrated mechanisms:')
        T('')
        T('1. **Shared TCN Backbone** — pre-trained once across all SKUs to '
          'capture universal temporal patterns (weekly seasonality, trend shapes). '
          'Frozen after pre-training; never updated per-SKU.')
        T('2. **Lightweight Per-SKU Adapters** — Houlsby-style bottleneck MLP '
          '(~10× fewer parameters than the backbone) trained independently per '
          'SKU in seconds. Conditioned on demand history, engineered features, '
          'and product metadata simultaneously.')
        T('3. **Continual Learning** — EWC (Elastic Weight Consolidation) on '
          'adapter weights plus reservoir replay buffer prevents catastrophic '
          'forgetting as the adapter stream processes thousands of SKUs.')
        T('')
        T('**Key properties:**')
        T('')
        T('- Cold-start SKUs handled from as few as 33 training days')
        T('- No full retraining required for new SKUs or demand shifts')
        T('- Metadata conditioning enables cross-category knowledge transfer')
        T('- Autoregressive inference — never leaks future demand into context')
        T('')

        # ---- Generated files ----
        T('## Generated Outputs')
        T('')
        T('| File | Description |')
        T('|------|-------------|')
        files = [
            ('model_comparison.csv',      'Full metrics table, all models'),
            ('mae_comparison.png',        'MAE bar chart'),
            ('rmse_comparison.png',       'RMSE bar chart'),
            ('mape_comparison.png',       'MAPE bar chart'),
            ('wape_comparison.png',       'WAPE bar chart (retail standard)'),
            ('p90_mae_comparison.png',    'P90 MAE — worst-10% SKU performance'),
            ('sku_mae_distribution.png',  'Per-SKU MAE box plots'),
            ('radar_comparison.png',      'Multi-metric radar chart'),
            ('performance_by_type.png',   'MAE/RMSE by model category'),
            ('evaluation_report.md',      'This report'),
        ]
        for fname, desc in files:
            T(f'| `{fname}` | {desc} |')
        T('')

        report_path = self.output_dir / 'evaluation_report.md'
        with open(report_path, 'w') as f:
            f.writelines(lines)
        print(f'  Saved: evaluation_report.md')

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def generate_full_evaluation(self):
        """Generate complete evaluation report and all visualisations."""

        print('\n' + '=' * 80)
        print('GENERATING COMPREHENSIVE EVALUATION REPORT')
        print('=' * 80 + '\n')

        # ---- Load ----
        print('Loading results...')
        results, missing = self.load_all_results()
        if missing:
            print(f'  Missing models: {missing} (skipped in comparison)')
        print(f'  Loaded: {list(results.keys())}\n')

        if not results:
            print('ERROR: No model results found. Check directory paths.')
            return

        detailed = self.load_detailed_csvs()

        # ---- Tables ----
        print('Building comparison table...')
        df = self.create_comparison_table(results)
        df.to_csv(self.output_dir / 'model_comparison.csv', index=False)
        print('  Saved: model_comparison.csv')

        strat_df     = self.create_stratified_table(detailed)
        improvements = self.compute_improvements(df)

        # Console summary
        print('\n' + '=' * 80)
        print('MODEL COMPARISON SUMMARY')
        print('=' * 80)
        pd.set_option('display.float_format', '{:.4f}'.format)
        pd.set_option('display.max_columns', 20)
        pd.set_option('display.width', 160)
        print(df.to_string(index=False))
        print('=' * 80)

        if improvements:
            print('\nACCF improvement over baselines (Mean MAE):')
            for model, pct in sorted(improvements.items(), key=lambda x: -x[1]):
                sign = '+' if pct >= 0 else ''
                print(f'  vs {model:<10} {sign}{pct:.1f}%')
        print()

        # ---- Visualisations ----
        print('Generating visualisations...')

        # DRY bar charts via single helper
        self._plot_metric_bars(
            df, 'Mean MAE', 'Std MAE',
            'Model Comparison — MAE (lower is better)',
            'Mean Absolute Error', 'mae_comparison.png', '{:.4f}')

        self._plot_metric_bars(
            df, 'Mean RMSE', 'Std RMSE',
            'Model Comparison — RMSE (lower is better)',
            'Root Mean Squared Error', 'rmse_comparison.png', '{:.4f}')

        self._plot_metric_bars(
            df, 'Mean MAPE (%)', None,
            'Model Comparison — MAPE (lower is better)',
            'Mean Absolute Percentage Error (%)',
            'mape_comparison.png', '{:.2f}%')

        self._plot_metric_bars(
            df, 'Mean WAPE (%)', None,
            'Model Comparison — WAPE (lower is better)',
            'Weighted Absolute Percentage Error (%)',
            'wape_comparison.png', '{:.2f}%')

        self._plot_metric_bars(
            df, 'P90 MAE', None,
            'Model Comparison — P90 MAE (worst-10% SKUs)',
            'P90 MAE (lower is better)',
            'p90_mae_comparison.png', '{:.4f}')

        self.plot_sku_distributions(detailed)
        self.plot_radar(df)
        self.plot_model_types(df)

        # ---- Markdown report ----
        print('\nWriting evaluation report...')
        self.create_detailed_report(df, results, improvements, strat_df)

        # ---- Summary ----
        print('\n' + '=' * 80)
        print('EVALUATION COMPLETE')
        print('=' * 80)
        print(f'\nAll outputs saved to: {self.output_dir}')
        generated = [
            'model_comparison.csv', 'evaluation_report.md',
            'mae_comparison.png', 'rmse_comparison.png',
            'mape_comparison.png', 'wape_comparison.png',
            'p90_mae_comparison.png', 'sku_mae_distribution.png',
            'radar_comparison.png', 'performance_by_type.png',
        ]
        for f in generated:
            path = self.output_dir / f
            status = 'OK' if path.exists() else 'MISSING'
            print(f'  [{status}] {f}')
        print('=' * 80 + '\n')


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    comparison = ModelComparison(
        phase3_results_dir = '../phase3_baselines/results',
        phase4_results_dir = 'results',
        sku_split_path     = '../phase2/processed_data/sku_split.json',
    )
    comparison.generate_full_evaluation()