#!/usr/bin/env python3
"""
Colab ZigZag Visualization - Ready-to-execute in Google Colab
Executes the complete ZigZag labeling pipeline with visualization
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

print("\n" + "="*80)
print("ZigZag Visualization Pipeline - Colab Ready")
print("="*80)

class ZigZagPineScript:
    """ZigZag implementation following exact Pine Script logic"""
    
    def __init__(self, depth=12, deviation=5, backstep=2):
        self.depth = depth
        self.deviation = deviation / 100.0
        self.backstep = backstep
    
    def find_pivots(self, highs, lows):
        """Find pivot points using MT4 ZigZag algorithm"""
        n = len(highs)
        pivots = []
        
        if n < self.depth * 2 + 1:
            return pivots
        
        extrema_candidates = []
        
        for i in range(self.depth, n - self.depth):
            is_high = highs[i] == np.max(highs[i - self.depth:i + self.depth + 1])
            is_low = lows[i] == np.min(lows[i - self.depth:i + self.depth + 1])
            
            if is_high:
                extrema_candidates.append((i, highs[i], 'H'))
            if is_low:
                extrema_candidates.append((i, lows[i], 'L'))
        
        if not extrema_candidates:
            return pivots
        
        extrema_dict = {}
        for idx, price, ptype in extrema_candidates:
            key = idx
            if key not in extrema_dict:
                extrema_dict[key] = (idx, price, ptype)
            else:
                existing_price = extrema_dict[key][1]
                existing_type = extrema_dict[key][2]
                
                if ptype == 'H' and existing_type == 'L':
                    if price > existing_price:
                        extrema_dict[key] = (idx, price, ptype)
                elif ptype == 'L' and existing_type == 'H':
                    if price < existing_price:
                        extrema_dict[key] = (idx, price, ptype)
        
        extrema = sorted(extrema_dict.values(), key=lambda x: x[0])
        
        filtered = []
        
        for idx, price, ptype in extrema:
            if not filtered:
                filtered.append((idx, price, ptype))
                continue
            
            last_idx, last_price, last_type = filtered[-1]
            
            if ptype == last_type:
                if ptype == 'H' and price > last_price:
                    filtered[-1] = (idx, price, ptype)
                elif ptype == 'L' and price < last_price:
                    filtered[-1] = (idx, price, ptype)
                continue
            
            if ptype == 'H':
                threshold = last_price * (1 + self.deviation)
                if price >= threshold:
                    filtered.append((idx, price, ptype))
            else:
                threshold = last_price * (1 - self.deviation)
                if price <= threshold:
                    filtered.append((idx, price, ptype))
        
        return filtered
    
    def label_pivots_accurate(self, pivots):
        """Label pivots as HH, HL, LH, LL using Pine Script logic"""
        if len(pivots) < 2:
            labeled = []
            for idx, price, ptype in pivots:
                labeled.append({
                    'idx': idx,
                    'price': price,
                    'type': ptype,
                    'label': None,
                    'direction': None
                })
            return labeled
        
        labeled = []
        
        labeled.append({
            'idx': pivots[0][0],
            'price': pivots[0][1],
            'type': pivots[0][2],
            'label': None,
            'direction': None
        })
        
        if pivots[1][1] > pivots[0][1]:
            current_direction = 1
        else:
            current_direction = -1
        
        last_point_price = None
        
        labeled.append({
            'idx': pivots[1][0],
            'price': pivots[1][1],
            'type': pivots[1][2],
            'label': None,
            'direction': current_direction
        })
        
        for i in range(2, len(pivots)):
            curr_idx, curr_price, curr_type = pivots[i]
            prev_idx, prev_price, prev_type = pivots[i-1]
            
            new_direction = current_direction
            
            if curr_type != prev_type:
                if curr_type == 'H':
                    new_direction = 1
                else:
                    new_direction = -1
                
                if new_direction != current_direction:
                    last_point_price = prev_price
                    current_direction = new_direction
            
            if last_point_price is not None:
                if current_direction > 0:
                    label = 'HH' if curr_price > last_point_price else 'LH'
                else:
                    label = 'LL' if curr_price < last_point_price else 'HL'
            else:
                label = None
            
            labeled.append({
                'idx': curr_idx,
                'price': curr_price,
                'type': curr_type,
                'label': label,
                'direction': current_direction
            })
        
        return labeled


def visualize_zigzag_accurate(df, labeled_pivots, title="ZigZag Analysis", num_bars=150):
    """Create publication-quality ZigZag visualization"""
    fig, ax = plt.subplots(figsize=(22, 11))
    
    df_plot = df.reset_index(drop=True).copy()
    
    start_idx = max(0, len(df_plot) - num_bars)
    end_idx = len(df_plot)
    df_segment = df_plot.iloc[start_idx:end_idx].copy()
    df_segment = df_segment.reset_index(drop=True)
    
    pivots_segment = []
    for pivot in labeled_pivots:
        if start_idx <= pivot['idx'] < end_idx:
            pivot_copy = pivot.copy()
            pivot_copy['idx'] = pivot['idx'] - start_idx
            pivots_segment.append(pivot_copy)
    
    # Plot candlesticks
    ax.plot(df_segment.index, df_segment['high'], color='#e8e8e8', linewidth=0.4, alpha=0.6)
    ax.plot(df_segment.index, df_segment['low'], color='#e8e8e8', linewidth=0.4, alpha=0.6)
    ax.plot(df_segment.index, df_segment['close'], color='#333333', linewidth=1.3, alpha=0.8, label='Close Price')
    
    label_colors = {
        'HH': '#0052cc',  # Blue
        'HL': '#ff6b35',  # Orange
        'LH': '#16a34a',  # Green
        'LL': '#dc2626',  # Red
    }
    
    for pivot in pivots_segment:
        idx = pivot['idx']
        price = pivot['price']
        label = pivot['label']
        
        if label and label in label_colors:
            color = label_colors[label]
            marker = 'o'
            size = 300
            ax.scatter(idx, price, color=color, s=size, marker=marker, 
                      edgecolors='black', linewidth=2.8, zorder=5, alpha=0.95)
            ax.text(idx, price * 1.008, label, ha='center', va='bottom', 
                   fontsize=12, fontweight='bold', color=color, 
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='none'))
        else:
            ax.scatter(idx, price, color='#999999', s=120, marker='s', 
                      edgecolors='black', linewidth=1.8, zorder=5, alpha=0.5)
    
    # Draw ZigZag lines
    if len(pivots_segment) > 1:
        sorted_pivots = sorted(pivots_segment, key=lambda x: x['idx'])
        for i in range(len(sorted_pivots) - 1):
            x1 = sorted_pivots[i]['idx']
            y1 = sorted_pivots[i]['price']
            x2 = sorted_pivots[i+1]['idx']
            y2 = sorted_pivots[i+1]['price']
            
            next_label = sorted_pivots[i+1]['label']
            line_color = label_colors.get(next_label, '#aaaaaa')
            
            ax.plot([x1, x2], [y1, y2], color=line_color, linewidth=2.8, alpha=0.75, zorder=3)
    
    ax.set_title(title, fontsize=18, fontweight='bold', pad=20, color='#1a1a1a')
    ax.set_xlabel('Bar Index (from right)', fontsize=13, fontweight='600')
    ax.set_ylabel('Price', fontsize=13, fontweight='600')
    ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.6, color='#cccccc')
    ax.set_facecolor('#f9f9f9')
    
    legend_elements = [
        mpatches.Patch(facecolor='#0052cc', edgecolor='black', label='HH - Higher High (Uptrend Strength)'),
        mpatches.Patch(facecolor='#ff6b35', edgecolor='black', label='HL - Higher Low (Downtrend)'),
        mpatches.Patch(facecolor='#16a34a', edgecolor='black', label='LH - Lower High (Downtrend)'),
        mpatches.Patch(facecolor='#dc2626', edgecolor='black', label='LL - Lower Low (Downtrend Strength)'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=12, framealpha=0.98, 
             edgecolor='black', fancybox=True, shadow=True)
    
    plt.tight_layout()
    return fig, ax


# ============================================================================
# MAIN PIPELINE
# ============================================================================

print("\n[Step 1/5] Installing dependencies...")
try:
    import yfinance
    import pandas
    import numpy
    import matplotlib
    print("   OK - All packages available")
except ImportError:
    print("   Installing missing packages...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", 
                          "yfinance", "pandas", "numpy", "matplotlib"])
    print("   OK - Installation complete")

print("\n[Step 2/5] Fetching market data...")
try:
    df = yf.download('BTC-USD', period='2y', interval='1h', progress=False)
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    df = df.reset_index()
    df.columns = [c.lower().strip() for c in df.columns]
    df = df.rename(columns={
        'open': 'open', 'high': 'high', 'low': 'low', 'close': 'close',
        'volume': 'volume', 'date': 'datetime', 'index': 'datetime'
    })
    
    print(f"   OK - Loaded {len(df):,} bars (BTC-USD, 1h)")
    print(f"        Date: {df['datetime'].min()} to {df['datetime'].max()}")
    print(f"        Range: {df['close'].min():.2f} - {df['close'].max():.2f}")
    
except Exception as e:
    print(f"   ERROR: {e}")
    sys.exit(1)

print("\n[Step 3/5] Detecting pivot points...")
zz = ZigZagPineScript(depth=12, deviation=5, backstep=2)
highs = df['high'].values
lows = df['low'].values

pivots = zz.find_pivots(highs, lows)
print(f"   OK - Found {len(pivots):,} pivot points")

print("\n[Step 4/5] Labeling pivots (HH/HL/LH/LL)...")
labeled_pivots = zz.label_pivots_accurate(pivots)

hh_count = sum(1 for p in labeled_pivots if p['label'] == 'HH')
hl_count = sum(1 for p in labeled_pivots if p['label'] == 'HL')
lh_count = sum(1 for p in labeled_pivots if p['label'] == 'LH')
ll_count = sum(1 for p in labeled_pivots if p['label'] == 'LL')
start_count = sum(1 for p in labeled_pivots if p['label'] is None)

total = len(labeled_pivots)

print(f"   OK - Label distribution:")
print(f"        HH (Higher High):     {hh_count:6d}  ({100*hh_count/total:5.2f}%)")
print(f"        HL (Higher Low):      {hl_count:6d}  ({100*hl_count/total:5.2f}%)")
print(f"        LH (Lower High):      {lh_count:6d}  ({100*lh_count/total:5.2f}%)")
print(f"        LL (Lower Low):       {ll_count:6d}  ({100*ll_count/total:5.2f}%)")
print(f"        START (Unlabeled):    {start_count:6d}  ({100*start_count/total:5.2f}%)")
print(f"        TOTAL:                {total:6d}")

print(f"\n   Last 25 pivot points:")
print(f"   {'#':>3s} {'Bar':>6s} {'Label':>8s} {'Type':>5s} {'Price':>12s} {'Direction':>10s}")
print("   " + "-"*65)

for i, pivot in enumerate(labeled_pivots[-25:], 1):
    label = pivot['label'] if pivot['label'] else 'START'
    direction = 'UP' if pivot['direction'] == 1 else ('DOWN' if pivot['direction'] == -1 else 'N/A')
    print(f"   {i:3d} {pivot['idx']:6d} {label:>8s} {pivot['type']:>5s} {pivot['price']:12.2f} {direction:>10s}")

print("\n[Step 5/5] Rendering visualization...")
fig, ax = visualize_zigzag_accurate(
    df,
    labeled_pivots,
    title="BTC-USD ZigZag Analysis (1h bars, depth=12, deviation=5%)",
    num_bars=150
)

plt.savefig('zigzag_analysis.png', dpi=150, bbox_inches='tight', facecolor='white')
print("   OK - Chart saved as 'zigzag_analysis.png'")

plt.show()

# Save pivot data
pivot_df = pd.DataFrame([
    {
        'bar_index': p['idx'],
        'price': p['price'],
        'type': p['type'],
        'label': p['label'],
        'direction': 'UP' if p['direction'] == 1 else ('DOWN' if p['direction'] == -1 else None)
    }
    for p in labeled_pivots
])

pivot_df.to_csv('zigzag_pivots.csv', index=False)
print("\n   Pivot data saved to 'zigzag_pivots.csv'")

print("\n" + "="*80)
print("Success! All components executed correctly.")
print("="*80 + "\n")
