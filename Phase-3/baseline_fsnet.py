"""
FSNet-Inspired Baseline Model for Fashion Demand Forecasting
Fast and Slow learner for continual forecasting
Simplified implementation based on Pham et al., 2022
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import warnings
from tqdm import tqdm
import logging
from typing import Dict, List, Tuple
import sys
sys.path.append('.')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

from utils.data_loader import BaselineDataLoader
from utils.metrics import calculate_forecasting_metrics, aggregate_sku_metrics
from utils.visualization import plot_predictions, plot_error_distribution

warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

torch.manual_seed(42)
np.random.seed(42)


class FSNetModel(nn.Module):
    """
    Simplified FSNet: Fast and Slow learners
    Fast learner: Adapts quickly to recent patterns
    Slow learner: Maintains long-term knowledge
    """
    
    def __init__(self, 
                 input_size: int = 1,
                 fast_hidden: int = 32,
                 slow_hidden: int = 64,
                 dropout: float = 0.2):
        super(FSNetModel, self).__init__()
        
        # Fast learner - shallow network for quick adaptation
        self.fast_learner = nn.Sequential(
            nn.Linear(input_size, fast_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fast_hidden, fast_hidden),
            nn.ReLU()
        )
        
        # Slow learner - deeper network for stable patterns
        self.slow_learner = nn.Sequential(
            nn.Linear(input_size, slow_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(slow_hidden, slow_hidden),
            nn.ReLU(),
            nn.Linear(slow_hidden, slow_hidden),
            nn.ReLU()
        )
        
        # Combination layer
        self.combiner = nn.Linear(fast_hidden + slow_hidden, 1)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        fast_out = self.fast_learner(x)
        slow_out = self.slow_learner(x)
        
        # Concatenate fast and slow
        combined = torch.cat([fast_out, slow_out], dim=1)
        
        # Final prediction
        output = self.combiner(combined)
        output = self.relu(output)
        
        return output


class FSNetForecaster:
    """
    FSNet-inspired forecaster with continual learning
    """
    
    def __init__(self,
                 lookback_window: int = 28,
                 fast_hidden: int = 32,
                 slow_hidden: int = 64,
                 learning_rate_fast: float = 0.01,
                 learning_rate_slow: float = 0.001,
                 epochs: int = 30,
                 batch_size: int = 32):
        
        self.lookback_window = lookback_window
        self.fast_hidden = fast_hidden
        self.slow_hidden = slow_hidden
        self.learning_rate_fast = learning_rate_fast
        self.learning_rate_slow = learning_rate_slow
        self.epochs = epochs
        self.batch_size = batch_size
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.scaler = StandardScaler()
        self.is_fitted = False
    
    def create_sequences(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        X, y = [], []
        for i in range(len(data) - self.lookback_window):
            # For simplicity, use the last value as feature
            X.append(data[i:i + self.lookback_window].mean())
            y.append(data[i + self.lookback_window])
        return np.array(X).reshape(-1, 1), np.array(y)
    
    def fit(self, train_data: pd.Series) -> bool:
        try:
            data = train_data.values.reshape(-1, 1)
            
            if len(data) < self.lookback_window + 10:
                return False
            
            data_scaled = self.scaler.fit_transform(data)
            X, y = self.create_sequences(data_scaled)
            
            if len(X) == 0:
                return False
            
            # Create model
            self.model = FSNetModel(
                input_size=1,
                fast_hidden=self.fast_hidden,
                slow_hidden=self.slow_hidden
            ).to(self.device)
            
            # Separate optimizers for fast and slow learners
            fast_params = list(self.model.fast_learner.parameters())
            slow_params = list(self.model.slow_learner.parameters())
            combiner_params = list(self.model.combiner.parameters())
            
            optimizer_fast = optim.Adam(fast_params + combiner_params, lr=self.learning_rate_fast)
            optimizer_slow = optim.Adam(slow_params, lr=self.learning_rate_slow)
            
            criterion = nn.MSELoss()
            
            # Training
            X_tensor = torch.FloatTensor(X).to(self.device)
            y_tensor = torch.FloatTensor(y).to(self.device)
            
            self.model.train()
            for epoch in range(self.epochs):
                # Update fast learner more frequently
                for _ in range(2):
                    outputs = self.model(X_tensor)
                    loss = criterion(outputs.squeeze(), y_tensor)
                    
                    optimizer_fast.zero_grad()
                    loss.backward(retain_graph=True)
                    optimizer_fast.step()
                
                # Update slow learner once
                outputs = self.model(X_tensor)
                loss = criterion(outputs.squeeze(), y_tensor)
                
                optimizer_slow.zero_grad()
                loss.backward()
                optimizer_slow.step()
            
            self.is_fitted = True
            return True
            
        except Exception as e:
            logger.warning(f"FSNet fitting failed: {str(e)}")
            return False
    
    def predict(self, test_data: pd.Series, train_data: pd.Series) -> np.ndarray:
        try:
            if not self.is_fitted:
                return np.zeros(len(test_data))
            
            self.model.eval()
            
            combined_data = pd.concat([train_data, test_data], ignore_index=True)
            data = combined_data.values.reshape(-1, 1)
            data_scaled = self.scaler.transform(data)
            
            predictions = []
            
            with torch.no_grad():
                for i in range(len(train_data), len(combined_data)):
                    if i >= self.lookback_window:
                        sequence = data_scaled[i - self.lookback_window:i]
                    else:
                        sequence = np.pad(
                            data_scaled[:i],
                            ((self.lookback_window - i, 0), (0, 0)),
                            mode='edge'
                        )
                    
                    feature = np.array([[sequence.mean()]])
                    feature_tensor = torch.FloatTensor(feature).to(self.device)
                    pred = self.model(feature_tensor)
                    predictions.append(pred.cpu().numpy()[0, 0])
            
            predictions = np.array(predictions).reshape(-1, 1)
            predictions = self.scaler.inverse_transform(predictions).flatten()
            predictions = np.maximum(predictions, 0)
            
            return predictions
            
        except Exception as e:
            logger.warning(f"FSNet prediction failed: {str(e)}")
            return np.zeros(len(test_data))
    
    def update(self, new_data: pd.Series):
        """
        Continual update with new data
        Fast learner adapts quickly, slow learner updates gradually
        """
        if not self.is_fitted:
            return self.fit(new_data)
        
        # Fine-tune with new data
        try:
            data = new_data.values.reshape(-1, 1)
            data_scaled = self.scaler.transform(data)
            X, y = self.create_sequences(data_scaled)
            
            if len(X) == 0:
                return False
            
            X_tensor = torch.FloatTensor(X).to(self.device)
            y_tensor = torch.FloatTensor(y).to(self.device)
            
            # Update fast learner more aggressively
            fast_params = list(self.model.fast_learner.parameters())
            combiner_params = list(self.model.combiner.parameters())
            optimizer_fast = optim.Adam(fast_params + combiner_params, lr=self.learning_rate_fast)
            
            criterion = nn.MSELoss()
            
            self.model.train()
            for _ in range(5):  # Quick adaptation
                outputs = self.model(X_tensor)
                loss = criterion(outputs.squeeze(), y_tensor)
                
                optimizer_fast.zero_grad()
                loss.backward()
                optimizer_fast.step()
            
            return True
            
        except Exception as e:
            logger.warning(f"FSNet update failed: {str(e)}")
            return False


class FSNetBaselineExperiment:
    """
    FSNet baseline experiment with continual learning evaluation
    """
    
    def __init__(self,
                 processed_data_dir: str = '../phase2/processed_data',
                 results_dir: str = 'results/fsnet',
                 n_skus: int = None):
        
        self.data_loader = BaselineDataLoader(processed_data_dir)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.n_skus = n_skus
        
        self.config = {
            'fast_hidden': 32,
            'slow_hidden': 64,
            'learning_rate_fast': 0.01,
            'learning_rate_slow': 0.001,
            'epochs': 30,
            'name': 'FSNet'
        }
        
        logger.info(f"Initialized FSNet experiment")
        logger.info(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    
    def evaluate_sku(self,
                    sku_id: str,
                    train_data: pd.DataFrame,
                    test_data: pd.DataFrame) -> Dict:
        
        train_sku = train_data[train_data['article_id'] == sku_id].copy()
        test_sku = test_data[test_data['article_id'] == sku_id].copy()
        
        train_sku = train_sku.sort_values('date').reset_index(drop=True)
        test_sku = test_sku.sort_values('date').reset_index(drop=True)
        
        if len(test_sku) == 0 or len(train_sku) < 50:
            return None
        
        forecaster = FSNetForecaster(
            fast_hidden=self.config['fast_hidden'],
            slow_hidden=self.config['slow_hidden'],
            learning_rate_fast=self.config['learning_rate_fast'],
            learning_rate_slow=self.config['learning_rate_slow'],
            epochs=self.config['epochs']
        )
        
        success = forecaster.fit(train_sku['demand'])
        
        if not success:
            return None
        
        predictions = forecaster.predict(test_sku['demand'], train_sku['demand'])
        actual = test_sku['demand'].values
        
        metrics = calculate_forecasting_metrics(actual, predictions)
        
        return {
            'sku_id': sku_id,
            'predictions': predictions.tolist(),
            'actual': actual.tolist(),
            'dates': test_sku['date'].dt.strftime('%Y-%m-%d').tolist(),
            'metrics': metrics,
            'model_fitted': success
        }
    
    def run_experiment(self):
        logger.info("="*80)
        logger.info("STARTING FSNET BASELINE EXPERIMENT")
        logger.info("="*80)
        
        train_data, val_data, test_data = self.data_loader.get_train_test_split()
        
        if self.n_skus is not None:
            skus_to_evaluate = self.data_loader.get_sample_skus(n=self.n_skus)
        else:
            skus_to_evaluate = self.data_loader.get_all_skus()
        
        logger.info(f"Evaluating {len(skus_to_evaluate)} SKUs")
        
        final_train = pd.concat([train_data, val_data], ignore_index=True)
        
        all_results = []
        sku_metrics = {}
        failed_skus = []
        
        for sku in tqdm(skus_to_evaluate, desc="Evaluating SKUs"):
            try:
                result = self.evaluate_sku(sku, final_train, test_data)
                
                if result is not None:
                    all_results.append(result)
                    sku_metrics[sku] = result['metrics']
                else:
                    failed_skus.append(sku)
                    
            except Exception as e:
                logger.warning(f"SKU {sku} failed: {str(e)}")
                failed_skus.append(sku)
        
        logger.info(f"Successfully evaluated: {len(all_results)} SKUs")
        logger.info(f"Failed: {len(failed_skus)} SKUs")
        
        aggregated_metrics = aggregate_sku_metrics(sku_metrics)
        
        self._save_results(all_results, aggregated_metrics)
        self._generate_visualizations(all_results)
        
        logger.info("="*80)
        logger.info("FSNET BASELINE EXPERIMENT COMPLETED")
        logger.info("="*80)
        
        return aggregated_metrics
    
    def _save_results(self, all_results: List[Dict], aggregated_metrics: Dict):
        results_summary = {
            'model': 'FSNet',
            'configuration': self.config,
            'n_skus_evaluated': len(all_results),
            'aggregated_metrics': aggregated_metrics,
            'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        with open(self.results_dir / 'results_summary.json', 'w') as f:
            json.dump(results_summary, f, indent=2)
        
        detailed_results = pd.DataFrame([
            {
                'sku_id': r['sku_id'],
                'rmse': r['metrics']['rmse'],
                'mae': r['metrics']['mae'],
                'mape': r['metrics']['mape'],
                'smape': r['metrics']['smape']
            }
            for r in all_results
        ])
        
        detailed_results.to_csv(self.results_dir / 'detailed_results.csv', index=False)
        
        print("\n" + "="*80)
        print("FSNET BASELINE RESULTS SUMMARY")
        print("="*80)
        print(f"Model: FSNet (Fast & Slow Learners)")
        print(f"SKUs Evaluated: {len(all_results)}")
        print(f"\nAggregated Metrics:")
        print(f"  Mean RMSE: {aggregated_metrics['mean_rmse']:.4f} (±{aggregated_metrics['std_rmse']:.4f})")
        print(f"  Mean MAE:  {aggregated_metrics['mean_mae']:.4f} (±{aggregated_metrics['std_mae']:.4f})")
        print(f"  Mean MAPE: {aggregated_metrics['mean_mape']:.2f}%")
        print("="*80 + "\n")
    
    def _generate_visualizations(self, all_results: List[Dict]):
        logger.info("Generating visualizations...")
        
        for i, result in enumerate(all_results[:5]):
            dates = pd.to_datetime(result['dates'])
            plot_predictions(
                dates=dates,
                y_true=np.array(result['actual']),
                y_pred=np.array(result['predictions']),
                title=f"FSNet Predictions - SKU {result['sku_id']}",
                save_path=self.results_dir / f"predictions_sku_{i+1}.png"
            )
        
        all_errors = []
        for result in all_results:
            errors = np.array(result['actual']) - np.array(result['predictions'])
            all_errors.extend(errors)
        
        plot_error_distribution(
            errors=np.array(all_errors),
            model_name='FSNet',
            save_path=self.results_dir / 'error_distribution.png'
        )


if __name__ == "__main__":
    PROCESSED_DATA_DIR = '../phase2/processed_data'
    RESULTS_DIR = 'results/fsnet'
    N_SKUS = 500
    
    experiment = FSNetBaselineExperiment(
        processed_data_dir=PROCESSED_DATA_DIR,
        results_dir=RESULTS_DIR,
        n_skus=N_SKUS
    )
    
    metrics = experiment.run_experiment()
    
    print("\nFSNet Baseline Complete!")
    print(f"Results saved to: {RESULTS_DIR}")