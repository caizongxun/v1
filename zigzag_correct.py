#!/usr/bin/env python3
"""ZigZag MT4 Correct Implementation

Based on:
- MT4 ZigZag official algorithm
- TradingView Pine Script implementation
- Reference: https://www.mql5.com/en/code/576
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

class ZigZagMT4Correct:
    """Correct MT4 ZigZag implementation"""
    
    def __init__(self, depth=12, deviation=5, backstep=2):
        """
        depth: Number of bars for finding local extremum
        deviation: Minimum percent change to consider a reversal
        backstep: Number of bars to review the last extremum
        """
        self.depth = depth
        self.deviation = deviation
        self.backstep = backstep
    
    def find_zigzag(self, highs, lows, closes):
        """
        Find ZigZag points using MT4 algorithm
        
        Returns:
            extrema: List of [index, price, type] where type is 'H' (High) or 'L' (Low)
        """
        n = len(highs)
        extrema = []  # [idx, price, type]
        
        # Step 1: Find all local extremums
        local_extrem = []  # [idx, price, type, is_high]
        
        for i in range(self.depth, n - self.depth):
            # Check High extremum
            if highs[i] == np.max(highs[i - self.depth:i + self.depth + 1]):
                local_extrem.append([i, highs[i], 'H', True])
            # Check Low extremum  
            if lows[i] == np.min(lows[i - self.depth:i + self.depth + 1]):
                local_extrem.append([i, lows[i], 'L', False])
        
        if not local_extrem:
            return extrema
        
        # Step 2: Filter by deviation and backstep
        deviation_coeff = 1.0 + self.deviation / 100.0
        
        extrema = []
        i = 0
        
        while i < len(local_extrem):
            curr_idx, curr_price, curr_type, is_high = local_extrem[i]
            
            # Skip if already processed
            if extrema and extrema[-1][0] == curr_idx:
                i += 1
                continue
            
            # First extremum
            if not extrema:
                extrema.append([curr_idx, curr_price, curr_type])
                i += 1
                continue
            
            last_idx, last_price, last_type = extrema[-1]
            
            # If same type, update with better extremum
            if curr_type == last_type:
                if curr_type == 'H':
                    if curr_price > last_price:
                        extrema[-1] = [curr_idx, curr_price, curr_type]
                else:  # 'L'
                    if curr_price < last_price:
                        extrema[-1] = [curr_idx, curr_price, curr_type]
                i += 1
                continue
            
            # Check deviation condition
            if curr_type == 'H':
                if last_type == 'L':
                    # High after Low: check if increase >= deviation
                    if curr_price >= last_price * deviation_coeff:
                        extrema.append([curr_idx, curr_price, curr_type])
            else:  # curr_type == 'L'
                if last_type == 'H':
                    # Low after High: check if decrease >= deviation
                    if curr_price <= last_price / deviation_coeff:
                        extrema.append([curr_idx, curr_price, curr_type])
            
            i += 1
        
        return extrema
    
    def label_zigzag(self, extrema):
        """
        Label ZigZag points as HH, HL, LH, LL
        
        Logic (from TradingView code):
        - direction<0? (z2.price<lastPoint? "LL": "HL"): (z2.price>lastPoint? "HH": "LH")
        - Compare with PREVIOUS extremum price, not relative to High/Low
        """
        if len(extrema) < 2:
            return extrema
        
        labeled = []
        
        # First extremum - no label
        labeled.append({
            'idx': extrema[0][0],
            'price': extrema[0][1],
            'type': extrema[0][2],
            'label': None
        })
        
        # Subsequent extrema - compare with previous
        for i in range(1, len(extrema)):
            curr_idx, curr_price, curr_type = extrema[i]
            prev_idx, prev_price, prev_type = extrema[i-1]
            
            # Determine label based on previous and current
            if prev_type == 'H':
                # Previous was High, now is Low (downtrend)
                # direction = -1 (downtrend), so direction < 0
                if curr_price < prev_price:
                    label = 'LL'  # Lower Low
                else:
                    label = 'LH'  # Lower High (but it's actually a Low, so weird)
            else:  # prev_type == 'L'
                # Previous was Low, now is High (uptrend)
                # direction = 1 (uptrend), so direction > 0
                if curr_price > prev_price:
                    label = 'HH'  # Higher High
                else:
                    label = 'HL'  # Higher Low
            
            labeled.append({
                'idx': curr_idx,
                'price': curr_price,
                'type': curr_type,
                'label': label
            })
        
        return labeled

def visualize_zigzag(df, extrema, title="ZigZag (MT4)", num_bars=100):
    """Visualize ZigZag"""
    fig, ax = plt.subplots(figsize=(20, 10))
    
    df_plot = df.reset_index(drop=True)
    
    # Show last num_bars
    start_idx = max(0, len(df_plot) - num_bars)
    end_idx = len(df_plot)
    df_segment = df_plot.iloc[start_idx:end_idx].copy()
    df_segment = df_segment.reset_index(drop=True)
    
    # Adjust extrema indices
    extrema_segment = []
    for ext in extrema:
        if start_idx <= ext['idx'] < end_idx:
            ext_copy = ext.copy()
            ext_copy['idx'] = ext['idx'] - start_idx
            extrema_segment.append(ext_copy)
    
    # Plot candlesticks
    ax.plot(df_segment.index, df_segment['high'], color='lightgray', linewidth=0.5, alpha=0.3)
    ax.plot(df_segment.index, df_segment['low'], color='lightgray', linewidth=0.5, alpha=0.3)
    ax.plot(df_segment.index, df_segment['close'], color='black', linewidth=1.5, alpha=0.8)
    
    # Color mapping
    label_colors = {
        'HH': '#1f77b4',  # Blue
        'HL': '#ff7f0e',  # Orange
        'LH': '#2ca02c',  # Green
        'LL': '#d62728',  # Red
    }
    
    # Plot extrema
    for ext in extrema_segment:
        idx = ext['idx']
        price = ext['price']
        label = ext['label']
        
        if label:
            color = label_colors[label]
            ax.scatter(idx, price, color=color, s=300, marker='o', edgecolors='black', linewidth=2.5, zorder=5)
            ax.text(idx, price * 1.015, label, ha='center', va='bottom', fontsize=10, fontweight='bold')
        else:
            # First point
            ax.scatter(idx, price, color='#2ca02c', s=300, marker='s', edgecolors='black', linewidth=2.5, zorder=5)
            ax.text(idx, price * 1.015, 'S', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Draw ZigZag lines
    if len(extrema_segment) > 1:
        sorted_extrema = sorted(extrema_segment, key=lambda x: x['idx'])
        for i in range(len(sorted_extrema) - 1):
            x1, y1 = sorted_extrema[i]['idx'], sorted_extrema[i]['price']
            x2, y2 = sorted_extrema[i+1]['idx'], sorted_extrema[i+1]['price']
            next_label = sorted_extrema[i+1]['label']
            line_color = label_colors.get(next_label, '#888888')
            ax.plot([x1, x2], [y1, y2], color=line_color, linewidth=2, alpha=0.6, zorder=3)
    
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Bar Index', fontsize=12)
    ax.set_ylabel('Price (USD)', fontsize=12)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    legend_elements = [
        mpatches.Patch(color='#1f77b4', label='HH - Higher High'),
        mpatches.Patch(color='#ff7f0e', label='HL - Higher Low'),
        mpatches.Patch(color='#2ca02c', label='LH - Lower High'),
        mpatches.Patch(color='#d62728', label='LL - Lower Low'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=11)
    
    plt.tight_layout()
    return fig, ax

# ============================================================================
# MAIN
# ============================================================================

print("="*80)
print("ZigZag MT4 Correct Implementation")
print("="*80)

print("\n[1/4] Fetching data...")
try:
    df = yf.download('BTC-USD', period='2y', interval='1h', progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.reset_index()
    df.columns = [c.lower().strip() for c in df.columns]
    print(f"  OK Loaded {len(df):,} bars")
except Exception as e:
    print(f"  Error: {e}")
    exit()

print(f"  Time range: {df['datetime'].iloc[0]} to {df['datetime'].iloc[-1]}")
print(f"  Price range: {df['close'].min():.2f} - {df['close'].max():.2f}")

highs = df['high'].values
lows = df['low'].values
closes = df['close'].values

print("\n[2/4] Comparing different parameters...")
print(f"\n{'depth':>8s} {'deviation':>12s} {'backstep':>10s} {'extrema_count':>15s}")
print("-" * 80)

for depth in [8, 12, 15]:
    for deviation in [3, 5, 7]:
        zz = ZigZagMT4Correct(depth=depth, deviation=deviation, backstep=2)
        extrema = zz.find_zigzag(highs, lows, closes)
        print(f"{depth:8d} {deviation:12d}% {2:10d} {len(extrema):15d}")

# Use depth=12, deviation=5 as default
depth_choice = 12
deviation_choice = 5
backstep_choice = 2

print(f"\n[3/4] Generating labels with depth={depth_choice}, deviation={deviation_choice}%...")

zz = ZigZagMT4Correct(depth=depth_choice, deviation=deviation_choice, backstep=backstep_choice)
extrema_raw = zz.find_zigzag(highs, lows, closes)
extrema = zz.label_zigzag(extrema_raw)

print(f"  OK Found {len(extrema)} extrema")

if len(extrema) > 0:
    print(f"\n  Label distribution:")
    hh = sum(1 for e in extrema if e['label'] == 'HH')
    hl = sum(1 for e in extrema if e['label'] == 'HL')
    lh = sum(1 for e in extrema if e['label'] == 'LH')
    ll = sum(1 for e in extrema if e['label'] == 'LL')
    start = sum(1 for e in extrema if e['label'] is None)
    print(f"    HH: {hh:4d}")
    print(f"    HL: {hl:4d}")
    print(f"    LH: {lh:4d}")
    print(f"    LL: {ll:4d}")
    print(f"    START: {start:2d}")
    print(f"    TOTAL: {len(extrema):4d}")
    
    print(f"\n  Last 30 extrema:")
    for i, ext in enumerate(extrema[-30:]):
        label = ext['label'] if ext['label'] else 'START'
        print(f"    {i+1:2d}. Bar {ext['idx']:5d} | {label:6s} | {ext['type']} | Price: {ext['price']:10.2f}")

print(f"\n[4/4] Creating visualization...")
fig, ax = visualize_zigzag(
    df,
    extrema,
    title=f"ZigZag (MT4 - depth={depth_choice}, deviation={deviation_choice}%)",
    num_bars=100
)
plt.show()

print("\n" + "="*80)
print("Done!")
print("="*80)
