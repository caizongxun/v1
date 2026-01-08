#!/usr/bin/env python3
"""
資料集探索腳本 - 理解 HuggingFace 上的資料結構
"""

import pandas as pd
import numpy as np
from huggingface_hub import hf_hub_download

print("\n" + "="*80)
print("資料集探索 - HuggingFace Crypto OHLCV Data")
print("="*80)

# 設定
REPO_ID = "zongowo111/v2-crypto-ohlcv-data"
CRYPTO = "BTC"
TIMEFRAME = "15m"  # 先用 15m 測試

print(f"\n目標資料:")
print(f"  Repository: {REPO_ID}")
print(f"  幣種: {CRYPTO}")
print(f"  時間框架: {TIMEFRAME}")

print("\n" + "="*80)
print("第 1 步: 下載資料")
print("="*80)

try:
    filename = f"klines/{CRYPTO}USDT/{CRYPTO}_{TIMEFRAME.upper()}.parquet"
    print(f"\n下載路徑: {filename}")
    
    filepath = hf_hub_download(
        repo_id=REPO_ID,
        filename=filename,
        repo_type="dataset"
    )
    
    print(f"✓ 下載成功: {filepath}")
    
except Exception as e:
    print(f"✗ 下載失敗: {e}")
    exit(1)

print("\n" + "="*80)
print("第 2 步: 讀取資料")
print("="*80)

try:
    df = pd.read_parquet(filepath)
    print(f"✓ 成功讀取 parquet 文件")
except Exception as e:
    print(f"✗ 讀取失敗: {e}")
    exit(1)

print("\n" + "="*80)
print("第 3 步: 資料概覽")
print("="*80)

print(f"\n資料形狀: {df.shape}")
print(f"  行數 (K線數): {len(df)}")
print(f"  列數 (特徵數): {df.shape[1]}")

print(f"\n時間範圍:")
print(f"  開始: {df.index[0]}")
print(f"  結束: {df.index[-1]}")
print(f"  時長: {df.index[-1] - df.index[0]}")

print("\n" + "="*80)
print("第 4 步: 列名和資料型別")
print("="*80)

print(f"\n所有列名 ({len(df.columns)} 個):")
for i, col in enumerate(df.columns, 1):
    print(f"  {i:2d}. {col:20s} (dtype: {df[col].dtype})")

print("\n" + "="*80)
print("第 5 步: 資料樣本")
print("="*80)

print(f"\n前 5 筆資料:")
print(df.head())

print(f"\n後 5 筆資料:")
print(df.tail())

print("\n" + "="*80)
print("第 6 步: 統計信息")
print("="*80)

print(f"\n數值列的統計信息:")
print(df.describe())

print(f"\n缺失值:")
missing = df.isnull().sum()
if missing.sum() == 0:
    print("  ✓ 沒有缺失值")
else:
    print(missing[missing > 0])

print("\n" + "="*80)
print("第 7 步: 列名映射")
print("="*80)

print(f"\n嘗試識別 OHLCV 列...")

# 列名映射字典
column_mapping = {}
standard_cols = ['open', 'high', 'low', 'close', 'volume']
actual_cols = [col.lower().strip() for col in df.columns]

print(f"\n實際列名 (小寫): {actual_cols}")

# 嘗試自動映射
for std_col in standard_cols:
    # 完全匹配
    if std_col in actual_cols:
        column_mapping[std_col] = std_col
        print(f"  ✓ {std_col:10s} -> {std_col}")
    # 包含匹配
    else:
        matches = [col for col in actual_cols if std_col in col]
        if matches:
            column_mapping[std_col] = matches[0]
            print(f"  ✓ {std_col:10s} -> {matches[0]}")
        else:
            print(f"  ✗ {std_col:10s} -> 找不到對應的列")

print(f"\n" + "="*80)
print("第 8 步: 重新命名列")
print("="*80)

if len(column_mapping) == 5:
    print(f"\n✓ 找到所有 5 個列!")
    print(f"\n映射方案:")
    for std, actual in column_mapping.items():
        print(f"  {std:10s} <- {actual}")
    
    # 重新命名
    df_renamed = df.rename(columns={v: k for k, v in column_mapping.items()})
    print(f"\n重新命名後的列名: {list(df_renamed.columns)}")
    
    print(f"\n重新命名後的前 5 筆資料:")
    print(df_renamed.head())
else:
    print(f"\n✗ 只找到 {len(column_mapping)} 個列，需要手動映射")
    print(f"\n實際可用的列:")
    for col in actual_cols:
        print(f"  - {col}")

print("\n" + "="*80)
print("資料探索完成!")
print("="*80)
