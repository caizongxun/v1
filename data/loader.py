import pandas as pd
import numpy as np
from pathlib import Path
import warnings
from typing import Optional, Dict, List
from loguru import logger
from datasets import load_dataset
from huggingface_hub import hf_hub_download, snapshot_download
import os

warnings.filterwarnings('ignore')


class HFDataLoader:
    def __init__(self, dataset_id: str = "zongowo111/v2-crypto-ohlcv-data", cache_dir: str = ".cache/hf_data"):
        """
        Initialize HuggingFace data loader
        
        Args:
            dataset_id: HuggingFace dataset ID
            cache_dir: Local cache directory
        """
        self.dataset_id = dataset_id
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"HFDataLoader initialized with dataset: {dataset_id}")

    def fetch_pair_data(
        self,
        pair: str,
        timeframe: str = "15m",
        use_cache: bool = True,
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Fetch cryptocurrency pair data from HuggingFace
        
        Args:
            pair: Cryptocurrency pair (e.g., 'BTCUSDT')
            timeframe: Timeframe ('15m' or '1h')
            use_cache: Use local cache if available
            limit: Maximum number of rows to load
        
        Returns:
            DataFrame with OHLCV data
        """
        try:
            # Construct file path pattern
            symbol = pair.replace('USDT', '')
            filename = f"{symbol}_{timeframe}.parquet"
            file_path = f"klines/{pair}/{filename}"
            
            cache_path = self.cache_dir / f"{pair}_{timeframe}.parquet"
            
            # Check local cache
            if use_cache and cache_path.exists():
                logger.info(f"Loading {pair} {timeframe} from cache")
                df = pd.read_parquet(cache_path)
            else:
                # Download from HuggingFace
                logger.info(f"Downloading {pair} {timeframe} from HuggingFace")
                df = self._download_from_hf(file_path)
                
                # Cache locally
                df.to_parquet(cache_path)
                logger.info(f"Cached {pair} {timeframe} to {cache_path}")
            
            # Limit rows if specified
            if limit:
                df = df.tail(limit)
            
            # Ensure correct column names and types
            df = self._standardize_dataframe(df)
            
            logger.info(f"Loaded {len(df)} rows for {pair} {timeframe}")
            return df
        
        except Exception as e:
            logger.error(f"Error loading {pair} {timeframe}: {str(e)}")
            raise

    def _download_from_hf(self, file_path: str) -> pd.DataFrame:
        """
        Download parquet file from HuggingFace
        
        Args:
            file_path: Path within dataset
        
        Returns:
            DataFrame
        """
        try:
            # Try loading from HuggingFace hub
            local_path = hf_hub_download(
                repo_id=self.dataset_id,
                filename=file_path,
                repo_type="dataset",
                cache_dir=self.cache_dir
            )
            df = pd.read_parquet(local_path)
            return df
        except Exception as e:
            logger.error(f"Failed to download {file_path}: {str(e)}")
            raise

    def _standardize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize dataframe column names and types
        
        Args:
            df: Input dataframe
        
        Returns:
            Standardized dataframe
        """
        # Handle various column naming conventions
        column_mapping = {
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume',
            'timestamp': 'Time',
            'date': 'Time',
            'open_time': 'Time'
        }
        
        # Rename columns (case-insensitive)
        for old, new in column_mapping.items():
            for col in df.columns:
                if col.lower() == old:
                    df = df.rename(columns={col: new})
        
        # Ensure required columns
        required = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Convert to numeric types
        for col in required:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Handle datetime index
        if 'Time' in df.columns:
            df['Time'] = pd.to_datetime(df['Time'])
            df = df.set_index('Time')
        else:
            df.index = pd.to_datetime(df.index)
        
        # Sort by index
        df = df.sort_index()
        
        # Remove duplicates
        df = df[~df.index.duplicated(keep='last')]
        
        return df

    def list_available_pairs(self) -> List[str]:
        """
        List all available cryptocurrency pairs in dataset
        
        Returns:
            List of pair names
        """
        try:
            # Common pairs in typical datasets
            pairs = [
                'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'XRPUSDT', 'ADAUSDT',
                'DOGEUSDT', 'MATICUSDT', 'AVAXUSDT', 'LTCUSDT', 'LINKUSDT',
                'SOLUSDT', 'UNIUSDT', 'FILUSDT', 'ATOMUSDT', 'APTUSDT'
            ]
            return pairs
        except Exception as e:
            logger.error(f"Error listing pairs: {str(e)}")
            return []

    def get_multiple_pairs(
        self,
        pairs: List[str],
        timeframe: str = "15m",
        **kwargs
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple pairs
        
        Args:
            pairs: List of pair names
            timeframe: Timeframe for all pairs
            **kwargs: Additional arguments for fetch_pair_data
        
        Returns:
            Dictionary mapping pair names to DataFrames
        """
        data = {}
        for pair in pairs:
            try:
                data[pair] = self.fetch_pair_data(pair, timeframe, **kwargs)
            except Exception as e:
                logger.warning(f"Failed to load {pair}: {str(e)}")
        return data


if __name__ == "__main__":
    # Example usage
    loader = HFDataLoader()
    
    # Load single pair
    df = loader.fetch_pair_data('BTCUSDT', timeframe='15m', limit=1000)
    print(f"Loaded shape: {df.shape}")
    print(df.head())
    
    # Load multiple pairs
    pairs = ['BTCUSDT', 'ETHUSDT']
    data = loader.get_multiple_pairs(pairs, timeframe='15m')
    for pair, df in data.items():
        print(f"{pair}: {df.shape}")
