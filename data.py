"""
Phase 2: Memory-Optimized Data Preparation for ACCF Model
H&M Fashion Demand Forecasting - Continual Few-Shot Learning
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import warnings
import json
from typing import Dict, List, Tuple
from tqdm import tqdm
import logging
import gc

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class HMDataPreprocessor:
    """
    Memory-optimized data preprocessing pipeline for H&M Fashion Dataset
    """
    
    def __init__(self, data_dir: str, output_dir: str = 'processed_data'):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Configuration - CRITICAL: Filter to top SKUs to manage memory
        self.config = {
            'top_n_skus': 5000,              # Only process top 5000 SKUs by volume
            'min_transactions_per_sku': 50,  # Minimum transactions to include SKU
            'cold_start_threshold': 200,     # SKUs with < 200 transactions are cold-start
            'aggregation_period': 'D',       # Daily ('D') or Weekly ('W')
            'date_format': '%Y-%m-%d',
            'streaming_split_ratio': 0.7,    # 70% for initial training, 30% for streaming
            'test_days': 7,                  # Last 7 days for testing
            'chunk_size': 1000,              # Process SKUs in chunks
        }
        
        logger.info(f"Initialized preprocessor with data_dir: {data_dir}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Will process top {self.config['top_n_skus']} SKUs")
    
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load CSV files with optimized dtypes"""
        logger.info("Loading datasets...")
        
        # Load transactions
        logger.info("Loading transactions_train.csv...")
        transactions = pd.read_csv(
            self.data_dir / 'transactions_train.csv',
            dtype={
                'article_id': str,
                'customer_id': str,
                'price': np.float32,
                'sales_channel_id': np.int8
            },
            parse_dates=['t_dat']
        )
        logger.info(f"Transactions loaded: {len(transactions):,} rows")
        
        # Load articles
        logger.info("Loading articles.csv...")
        articles = pd.read_csv(
            self.data_dir / 'articles.csv',
            dtype={'article_id': str}
        )
        logger.info(f"Articles loaded: {len(articles):,} rows")
        
        return transactions, articles
    
    def select_top_skus(self, transactions: pd.DataFrame) -> List[str]:
        """Select top N SKUs by transaction volume"""
        logger.info("Selecting top SKUs by transaction volume...")
        
        sku_counts = transactions['article_id'].value_counts()
        
        # Filter by minimum threshold
        min_threshold = self.config['min_transactions_per_sku']
        sku_counts = sku_counts[sku_counts >= min_threshold]
        
        # Select top N
        top_n = self.config['top_n_skus']
        top_skus = sku_counts.head(top_n).index.tolist()
        
        logger.info(f"Selected {len(top_skus)} SKUs (min {min_threshold} transactions)")
        logger.info(f"Top SKU has {sku_counts.iloc[0]:,} transactions")
        logger.info(f"Median SKU has {sku_counts.median():,.0f} transactions")
        
        return top_skus
    
    def clean_transactions(self, transactions: pd.DataFrame, selected_skus: List[str]) -> pd.DataFrame:
        """Clean and filter transaction data"""
        logger.info("Cleaning transaction data...")
        
        initial_count = len(transactions)
        
        # Filter to selected SKUs first (major memory reduction)
        transactions = transactions[transactions['article_id'].isin(selected_skus)].copy()
        logger.info(f"Filtered to selected SKUs: {len(transactions):,} rows ({len(transactions)/initial_count:.1%})")
        
        # Remove duplicates
        transactions = transactions.drop_duplicates()
        logger.info(f"After removing duplicates: {len(transactions):,} rows")
        
        # Remove invalid prices
        transactions = transactions[transactions['price'] > 0]
        
        # Sort by date
        transactions = transactions.sort_values('t_dat').reset_index(drop=True)
        
        logger.info(f"Date range: {transactions['t_dat'].min()} to {transactions['t_dat'].max()}")
        
        return transactions
    
    def clean_articles(self, articles: pd.DataFrame, selected_skus: List[str]) -> pd.DataFrame:
        """Clean and prepare article metadata"""
        logger.info("Cleaning article metadata...")
        
        # Filter to selected SKUs
        articles = articles[articles['article_id'].isin(selected_skus)].copy()
        
        # Fill missing values
        text_columns = articles.select_dtypes(include=['object']).columns
        for col in text_columns:
            articles[col] = articles[col].fillna('unknown')
        
        # Select key metadata features
        key_features = [
            'article_id',
            'product_type_name',
            'product_group_name',
            'graphical_appearance_name',
            'colour_group_name',
            'department_name',
            'section_name',
            'garment_group_name'
        ]
        
        available_features = [col for col in key_features if col in articles.columns]
        articles_clean = articles[available_features].copy()
        
        logger.info(f"Article metadata features: {len(available_features)}")
        
        return articles_clean
    
    def aggregate_daily_demand(self, transactions: pd.DataFrame) -> pd.DataFrame:
        """Aggregate transactions into daily demand time series per SKU"""
        logger.info("Aggregating to daily demand per SKU...")
        
        daily_demand = transactions.groupby(['article_id', 't_dat']).agg({
            'customer_id': 'count',
            'price': 'mean',
            'sales_channel_id': lambda x: x.mode()[0] if len(x) > 0 else 1
        }).reset_index()
        
        daily_demand.columns = ['article_id', 'date', 'demand', 'price_mean', 'sales_channel']
        
        logger.info(f"Daily demand records: {len(daily_demand):,}")
        logger.info(f"Unique SKUs: {daily_demand['article_id'].nunique():,}")
        
        return daily_demand
    
    def create_complete_time_series_chunked(self, daily_demand: pd.DataFrame) -> pd.DataFrame:
        """Create complete time series in memory-efficient chunks"""
        logger.info("Creating complete time series (chunked processing)...")
        
        min_date = daily_demand['date'].min()
        max_date = daily_demand['date'].max()
        date_range = pd.date_range(start=min_date, end=max_date, freq='D')
        
        unique_skus = daily_demand['article_id'].unique()
        chunk_size = self.config['chunk_size']
        
        logger.info(f"Processing {len(unique_skus)} SKUs over {len(date_range)} days")
        logger.info(f"Chunk size: {chunk_size} SKUs")
        
        all_chunks = []
        
        # Process in chunks
        for i in tqdm(range(0, len(unique_skus), chunk_size), desc="Processing SKU chunks"):
            chunk_skus = unique_skus[i:i+chunk_size]
            
            # Create complete index for this chunk
            chunk_index = pd.MultiIndex.from_product(
                [chunk_skus, date_range],
                names=['article_id', 'date']
            )
            
            chunk_df = pd.DataFrame(index=chunk_index).reset_index()
            
            # Merge with actual demand for this chunk
            chunk_demand = daily_demand[daily_demand['article_id'].isin(chunk_skus)]
            chunk_df = chunk_df.merge(chunk_demand, on=['article_id', 'date'], how='left')
            
            # Fill missing values
            chunk_df['demand'] = chunk_df['demand'].fillna(0)
            chunk_df = chunk_df.sort_values(['article_id', 'date'])
            chunk_df['price_mean'] = chunk_df.groupby('article_id')['price_mean'].fillna(method='ffill')
            chunk_df['price_mean'] = chunk_df.groupby('article_id')['price_mean'].fillna(method='bfill')
            chunk_df['sales_channel'] = chunk_df['sales_channel'].fillna(1).astype(np.int8)
            
            all_chunks.append(chunk_df)
            
            # Clear memory
            del chunk_df, chunk_demand, chunk_index
            gc.collect()
        
        # Combine all chunks
        logger.info("Combining chunks...")
        complete_series = pd.concat(all_chunks, ignore_index=True)
        
        # Fill any remaining NaN
        complete_series['price_mean'] = complete_series['price_mean'].fillna(complete_series['price_mean'].mean())
        
        logger.info(f"Complete time series created: {len(complete_series):,} records")
        
        return complete_series
    
    def add_temporal_features(self, complete_series: pd.DataFrame) -> pd.DataFrame:
        """Add temporal and rolling features"""
        logger.info("Adding temporal features...")
        
        # Ensure sorted
        complete_series = complete_series.sort_values(['article_id', 'date']).reset_index(drop=True)
        
        # Date components
        complete_series['year'] = complete_series['date'].dt.year.astype(np.int16)
        complete_series['month'] = complete_series['date'].dt.month.astype(np.int8)
        complete_series['week'] = complete_series['date'].dt.isocalendar().week.astype(np.int8)
        complete_series['day_of_week'] = complete_series['date'].dt.dayofweek.astype(np.int8)
        complete_series['day_of_month'] = complete_series['date'].dt.day.astype(np.int8)
        complete_series['is_weekend'] = (complete_series['day_of_week'] >= 5).astype(np.int8)
        
        logger.info("Computing rolling statistics (this may take a while)...")
        
        # Rolling features - process in chunks by SKU
        unique_skus = complete_series['article_id'].unique()
        chunk_size = 500
        
        # Initialize columns
        complete_series['demand_rolling_mean_7d'] = 0.0
        complete_series['demand_rolling_mean_28d'] = 0.0
        complete_series['demand_lag_1d'] = 0.0
        complete_series['demand_lag_7d'] = 0.0
        
        for i in tqdm(range(0, len(unique_skus), chunk_size), desc="Computing features"):
            chunk_skus = unique_skus[i:i+chunk_size]
            mask = complete_series['article_id'].isin(chunk_skus)
            
            for window in [7, 28]:
                complete_series.loc[mask, f'demand_rolling_mean_{window}d'] = (
                    complete_series[mask].groupby('article_id')['demand']
                    .transform(lambda x: x.rolling(window=window, min_periods=1).mean())
                )
            
            for lag in [1, 7]:
                complete_series.loc[mask, f'demand_lag_{lag}d'] = (
                    complete_series[mask].groupby('article_id')['demand'].shift(lag).fillna(0)
                )
        
        logger.info(f"Features added. Total columns: {len(complete_series.columns)}")
        
        return complete_series
    
    def merge_metadata(self, complete_series: pd.DataFrame, articles: pd.DataFrame) -> pd.DataFrame:
        """Merge article metadata"""
        logger.info("Merging article metadata...")
        
        merged = complete_series.merge(articles, on='article_id', how='left')
        
        logger.info(f"Merged data shape: {merged.shape}")
        
        return merged
    
    def identify_cold_start_skus(self, complete_series: pd.DataFrame) -> Dict:
        """Identify cold-start SKUs"""
        logger.info("Identifying cold-start SKUs...")
        
        threshold = self.config['cold_start_threshold']
        sku_totals = complete_series.groupby('article_id')['demand'].sum()
        
        cold_start_skus = sku_totals[sku_totals < threshold].index.tolist()
        warm_start_skus = sku_totals[sku_totals >= threshold].index.tolist()
        
        logger.info(f"Cold-start SKUs (< {threshold} transactions): {len(cold_start_skus):,}")
        logger.info(f"Warm-start SKUs (>= {threshold} transactions): {len(warm_start_skus):,}")
        
        return {
            'cold_start_skus': cold_start_skus,
            'warm_start_skus': warm_start_skus,
            'cold_start_threshold': threshold,
            'cold_start_count': len(cold_start_skus),
            'warm_start_count': len(warm_start_skus)
        }
    
    def create_streaming_splits(self, complete_series: pd.DataFrame) -> Dict:
        """Create temporal splits for streaming simulation"""
        logger.info("Creating streaming data splits...")
        
        min_date = complete_series['date'].min()
        max_date = complete_series['date'].max()
        total_days = (max_date - min_date).days + 1
        
        test_days = self.config['test_days']
        streaming_ratio = self.config['streaming_split_ratio']
        
        test_start_date = max_date - timedelta(days=test_days - 1)
        train_end_date = test_start_date - timedelta(days=1)
        streaming_split_date = min_date + timedelta(days=int(total_days * streaming_ratio))
        
        splits = {
            'min_date': min_date.strftime('%Y-%m-%d'),
            'max_date': max_date.strftime('%Y-%m-%d'),
            'total_days': total_days,
            'streaming_split_date': streaming_split_date.strftime('%Y-%m-%d'),
            'test_start_date': test_start_date.strftime('%Y-%m-%d'),
            'train_end_date': train_end_date.strftime('%Y-%m-%d'),
            'initial_train_days': (streaming_split_date - min_date).days + 1,
            'streaming_days': (train_end_date - streaming_split_date).days,
            'test_days': test_days
        }
        
        logger.info(f"Initial training: {splits['min_date']} to {splits['streaming_split_date']} ({splits['initial_train_days']} days)")
        logger.info(f"Streaming period: {streaming_split_date.strftime('%Y-%m-%d')} to {splits['train_end_date']} ({splits['streaming_days']} days)")
        logger.info(f"Test period: {splits['test_start_date']} to {splits['max_date']} ({splits['test_days']} days)")
        
        return splits
    
    def save_processed_data(self, final_data: pd.DataFrame, sku_split: Dict, 
                           temporal_splits: Dict, articles: pd.DataFrame):
        """Save all processed data"""
        logger.info("Saving processed data...")
        
        # Save main dataset in parquet (more efficient)
        parquet_file = self.output_dir / 'time_series_data.parquet'
        final_data.to_parquet(parquet_file, index=False, compression='snappy')
        logger.info(f"Saved parquet file: {parquet_file}")
        
        # Save SKU lists
        sku_file = self.output_dir / 'sku_split.json'
        with open(sku_file, 'w') as f:
            json.dump(sku_split, f, indent=2)
        logger.info(f"Saved SKU split: {sku_file}")
        
        # Save temporal splits
        splits_file = self.output_dir / 'temporal_splits.json'
        with open(splits_file, 'w') as f:
            json.dump(temporal_splits, f, indent=2)
        logger.info(f"Saved temporal splits: {splits_file}")
        
        # Save article metadata
        articles_file = self.output_dir / 'articles_metadata.csv'
        articles.to_csv(articles_file, index=False)
        logger.info(f"Saved article metadata: {articles_file}")
        
        # Save summary statistics
        self._save_summary_statistics(final_data, sku_split, temporal_splits)
    
    def _save_summary_statistics(self, final_data: pd.DataFrame, 
                                 sku_split: Dict, temporal_splits: Dict):
        """Save summary statistics"""
        
        summary = {
            'dataset_info': {
                'total_records': len(final_data),
                'unique_skus': final_data['article_id'].nunique(),
                'date_range': {
                    'start': final_data['date'].min().strftime('%Y-%m-%d'),
                    'end': final_data['date'].max().strftime('%Y-%m-%d'),
                    'days': (final_data['date'].max() - final_data['date'].min()).days + 1
                }
            },
            'demand_statistics': {
                'total_demand': float(final_data['demand'].sum()),
                'mean_daily_demand': float(final_data['demand'].mean()),
                'std_daily_demand': float(final_data['demand'].std()),
                'median_daily_demand': float(final_data['demand'].median()),
                'max_daily_demand': float(final_data['demand'].max()),
                'zero_demand_ratio': float((final_data['demand'] == 0).sum() / len(final_data))
            },
            'sku_split': sku_split,
            'temporal_splits': temporal_splits,
            'feature_columns': final_data.columns.tolist()
        }
        
        summary_file = self.output_dir / 'summary_statistics.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Saved summary: {summary_file}")
        
        # Print summary
        print("\n" + "="*80)
        print("DATA PREPARATION SUMMARY")
        print("="*80)
        print(f"Total Records: {summary['dataset_info']['total_records']:,}")
        print(f"Unique SKUs: {summary['dataset_info']['unique_skus']:,}")
        print(f"Date Range: {summary['dataset_info']['date_range']['start']} to {summary['dataset_info']['date_range']['end']}")
        print(f"Total Days: {summary['dataset_info']['date_range']['days']}")
        print(f"\nDemand Statistics:")
        print(f"  Total Demand: {summary['demand_statistics']['total_demand']:,.0f}")
        print(f"  Mean Daily Demand: {summary['demand_statistics']['mean_daily_demand']:.2f}")
        print(f"  Zero Demand Ratio: {summary['demand_statistics']['zero_demand_ratio']:.2%}")
        print(f"\nSKU Split:")
        print(f"  Cold-start SKUs: {sku_split['cold_start_count']:,}")
        print(f"  Warm-start SKUs: {sku_split['warm_start_count']:,}")
        print(f"\nTemporal Splits:")
        print(f"  Initial Training: {temporal_splits['initial_train_days']} days")
        print(f"  Streaming Period: {temporal_splits['streaming_days']} days")
        print(f"  Test Period: {temporal_splits['test_days']} days")
        print("="*80 + "\n")
    
    def run_full_pipeline(self):
        """Execute the complete pipeline"""
        logger.info("="*80)
        logger.info("STARTING MEMORY-OPTIMIZED DATA PREPARATION PIPELINE")
        logger.info("="*80)
        
        try:
            # Load data
            transactions, articles = self.load_data()
            
            # Select top SKUs
            selected_skus = self.select_top_skus(transactions)
            
            # Clean data
            transactions_clean = self.clean_transactions(transactions, selected_skus)
            del transactions
            gc.collect()
            
            articles_clean = self.clean_articles(articles, selected_skus)
            del articles
            gc.collect()
            
            # Aggregate daily demand
            daily_demand = self.aggregate_daily_demand(transactions_clean)
            del transactions_clean
            gc.collect()
            
            # Create complete time series (chunked)
            complete_series = self.create_complete_time_series_chunked(daily_demand)
            del daily_demand
            gc.collect()
            
            # Add features
            featured_series = self.add_temporal_features(complete_series)
            del complete_series
            gc.collect()
            
            # Merge metadata
            final_data = self.merge_metadata(featured_series, articles_clean)
            del featured_series
            gc.collect()
            
            # Identify cold-start SKUs
            sku_split = self.identify_cold_start_skus(final_data)
            
            # Create streaming splits
            temporal_splits = self.create_streaming_splits(final_data)
            
            # Save everything
            self.save_processed_data(final_data, sku_split, temporal_splits, articles_clean)
            
            logger.info("="*80)
            logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
            logger.info("="*80)
            
            return final_data, sku_split, temporal_splits
            
        except Exception as e:
            logger.error(f"Error in pipeline: {str(e)}")
            raise


class StreamingDataLoader:
    """Utility class to load prepared data"""
    
    def __init__(self, processed_data_dir: str = 'processed_data'):
        self.data_dir = Path(processed_data_dir)
        
        with open(self.data_dir / 'sku_split.json', 'r') as f:
            self.sku_split = json.load(f)
        
        with open(self.data_dir / 'temporal_splits.json', 'r') as f:
            self.temporal_splits = json.load(f)
        
        with open(self.data_dir / 'summary_statistics.json', 'r') as f:
            self.summary = json.load(f)
    
    def load_full_data(self) -> pd.DataFrame:
        """Load full dataset"""
        logger.info("Loading data from parquet...")
        data = pd.read_parquet(self.data_dir / 'time_series_data.parquet')
        logger.info(f"Loaded {len(data):,} records")
        return data
    
    def get_initial_training_data(self) -> pd.DataFrame:
        """Get initial training data"""
        data = self.load_full_data()
        split_date = pd.to_datetime(self.temporal_splits['streaming_split_date'])
        return data[data['date'] <= split_date].copy()
    
    def get_streaming_data(self) -> pd.DataFrame:
        """Get streaming data"""
        data = self.load_full_data()
        split_date = pd.to_datetime(self.temporal_splits['streaming_split_date'])
        test_start = pd.to_datetime(self.temporal_splits['test_start_date'])
        return data[(data['date'] > split_date) & (data['date'] < test_start)].copy()
    
    def get_test_data(self) -> pd.DataFrame:
        """Get test data"""
        data = self.load_full_data()
        test_start = pd.to_datetime(self.temporal_splits['test_start_date'])
        return data[data['date'] >= test_start].copy()
    
    def stream_by_day(self, start_date: str = None):
        """Stream data day by day"""
        data = self.load_full_data()
        
        if start_date is None:
            start_date = self.temporal_splits['streaming_split_date']
        
        start = pd.to_datetime(start_date)
        test_start = pd.to_datetime(self.temporal_splits['test_start_date'])
        
        current_date = start
        while current_date < test_start:
            daily_data = data[data['date'] == current_date].copy()
            yield current_date, daily_data
            current_date += timedelta(days=1)


if __name__ == "__main__":
    DATA_DIR = "."
    OUTPUT_DIR = "processed_data"
    
    preprocessor = HMDataPreprocessor(data_dir=DATA_DIR, output_dir=OUTPUT_DIR)
    final_data, sku_split, temporal_splits = preprocessor.run_full_pipeline()
    
    # Example usage
    print("\n" + "="*80)
    print("EXAMPLE: Loading Processed Data")
    print("="*80)
    
    loader = StreamingDataLoader(OUTPUT_DIR)
    train_data = loader.get_initial_training_data()
    print(f"\nInitial training data: {len(train_data):,} records")
    
    stream_data = loader.get_streaming_data()
    print(f"Streaming data: {len(stream_data):,} records")
    
    test_data = loader.get_test_data()
    print(f"Test data: {len(test_data):,} records")
    
    print("\n" + "="*80)
    print("Phase 2 Complete!")
    print("="*80)