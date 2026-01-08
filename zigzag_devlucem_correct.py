#!/usr/bin/env python3
"""ZigZag DevLucem Correct Implementation

Key insight from EliteTrader & StackOverflow implementations:
- Compare current extrema with TWO bars back (zigzag.get(0) vs zigzag.get(2))
- Not with the previous extrema!

Logic:
- dir == 1 (uptrend): if current.price > get(2).price then HH else LH
- dir == -1 (downtrend): if current.price < get(2).price then LL else HL
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

class ZigZagDevLucem:
    """ZigZag implementation based on DevLucem/EliteTrader logic"""
    
    def __init__(self, depth=12, deviation=5, backstep=2):
        self.depth = depth
        self.deviation = deviation
        self.backstep = backstep
    
    def find_zigzag(self, highs, lows):
        """
        Find ZigZag extrema
        Returns: [(idx, price, type), ...] where type is 'H' or 'L'
        """
        n = len(highs)
        extrema = []
        
        # Find all local extremums
        for i in range(self.depth, n - self.depth):
            if highs[i] == np.max(highs[i - self.depth:i + self.depth + 1]):
                extrema.append((i, highs[i], 'H'))
            if lows[i] == np.min(lows[i - self.depth:i + self.depth + 1]):
                extrema.append((i, lows[i], 'L'))
        
        # Filter by deviation and remove consecutive same-type
        deviation_coeff = 1.0 + self.deviation / 100.0
        filtered = []
        
        for idx, price, ext_type in extrema:
            if not filtered:
                filtered.append((idx, price, ext_type))
                continue
            
            last_idx, last_price, last_type = filtered[-1]
            
            # Same type - keep better one
            if ext_type == last_type:
                if ext_type == 'H' and price > last_price:
                    filtered[-1] = (idx, price, ext_type)
                elif ext_type == 'L' and price < last_price:
                    filtered[-1] = (idx, price, ext_type)
            else:
                # Different type - check deviation
                if ext_type == 'H':
                    if price >= last_price * deviation_coeff:
                        filtered.append((idx, price, ext_type))
                else:  # L
                    if price <= last_price / deviation_coeff:
                        filtered.append((idx, price, ext_type))
        
        return filtered
    
    def label_extrema(self, extrema):
        """
        Label extrema as HH, HL, LH, LL
        
        CORRECT LOGIC (from EliteTrader):
        - Keep a list of zigzag points
        - For each new point, compare with 2 bars back:
          - if current.price > get(2).price: HH (if uptrend) or LL (if downtrend)  
          - else: LH (if uptrend) or HL (if downtrend)
        """
        if len(extrema) < 3:
            return extrema
        
        labeled = []
        
        # First point - no label
        labeled.append({
            'idx': extrema[0][0],
            'price': extrema[0][1],
            'type': extrema[0][2],
            'label': None,
            'direction': None
        })
        
        # Determine initial direction from first two points
        if extrema[1][1] > extrema[0][1]:
            direction = 1  # Up
        else:
            direction = -1  # Down
        
        labeled.append({
            'idx': extrema[1][0],
            'price': extrema[1][1],
            'type': extrema[1][2],
            'label': None,
            'direction': direction
        })
        
        # Process remaining points
        for i in range(2, len(extrema)):
            curr_idx, curr_price, curr_type = extrema[i]
            prev_idx, prev_price, prev_type = extrema[i-1]
            prev2_idx, prev2_price, prev2_type = extrema[i-2]
            
            # Determine new direction
            if curr_type != prev_type:
                if curr_type == 'H':
                    direction = 1  # Changed to uptrend
                else:
                    direction = -1  # Changed to downtrend
            else:
                # Same type, direction continues
                direction = labeled[i-1]['direction']
            
            # Label based on direction and comparison with 2 bars back
            if direction == 1:  # Uptrend
                # We're placing a High
                if curr_price > prev2_price:
                    label = 'HH'  # Higher High
                else:
                    label = 'LH'  # Lower High
            else:  # Downtrend (-1)
                # We're placing a Low
                if curr_price < prev2_price:
                    label = 'LL'  # Lower Low
                else:
                    label = 'HL'  # Higher Low
            
            labeled.append({
                'idx': curr_idx,
                'price': curr_price,
                'type': curr_type,
                'label': label,
                'direction': direction
            })
        
        return labeled

def visualize_zigzag(df, extrema, title="ZigZag (DevLucem Correct)", num_bars=100):
    """Visualize ZigZag"""
    fig, ax = plt.subplots(figsize=(20, 10))
    
    df_plot = df.reset_index(drop=True)
    
    start_idx = max(0, len(df_plot) - num_bars)
    end_idx = len(df_plot)
    df_segment = df_plot.iloc[start_idx:end_idx].copy()
    df_segment = df_segment.reset_index(drop=True)
    
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
print("ZigZag DevLucem Correct Implementation")
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

print("\n[2/4] Comparing parameters...")
print(f"\n{'depth':>8s} {'deviation':>12s} {'extrema':>15s}")
print("-" * 80)

for depth in [8, 12, 15]:
    for deviation in [3, 5, 7]:
        zz = ZigZagDevLucem(depth=depth, deviation=deviation, backstep=2)
        extrema = zz.find_zigzag(highs, lows)
        print(f"{depth:8d} {deviation:12d}% {len(extrema):15d}")

# Use default parameters
depth_choice = 12
deviation_choice = 5

print(f"\n[3/4] Generating labels with depth={depth_choice}, deviation={deviation_choice}%...")

zz = ZigZagDevLucem(depth=depth_choice, deviation=deviation_choice, backstep=2)
extrema_raw = zz.find_zigzag(highs, lows)
extrema = zz.label_extrema(extrema_raw)

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
    title=f"ZigZag (DevLucem - depth={depth_choice}, deviation={deviation_choice}%)",
    num_bars=100
)
plt.show()

print("\n" + "="*80)
print("Done!")
print("="*80)
