#!/usr/bin/env python3
"""
Accurate ZigZag Implementation Following Pine Script Logic

Pine Script Reference (Dev Lucem):
- direction: >0 (uptrend), <0 (downtrend)
- z1: previous pivot point
- z2: current pivot point
- lastPoint: price of previous direction-changed pivot

HH/HL/LH/LL Logic:
if direction < 0 (downtrend):
    LL: if z2.price < lastPoint
    HL: else
if direction > 0 (uptrend):
    HH: if z2.price > lastPoint
    LH: else

Key Insight:
Compare current pivot price with LAST PIVOT FROM DIFFERENT DIRECTION
Not with the previous pivot in same direction
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

class ZigZagPineScript:
    """ZigZag implementation following exact Pine Script logic"""
    
    def __init__(self, depth=12, deviation=5, backstep=2):
        self.depth = depth
        self.deviation = deviation / 100.0
        self.backstep = backstep
    
    def find_pivots(self, highs, lows):
        """
        Find pivot points using MT4 ZigZag algorithm
        Returns list of (index, price, pivot_type) tuples
        pivot_type: 'H' for high, 'L' for low
        """
        n = len(highs)
        pivots = []
        
        if n < self.depth * 2 + 1:
            return pivots
        
        # Identify local extrema
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
        
        # Sort by index and remove duplicates at same bar
        extrema_dict = {}
        for idx, price, ptype in extrema_candidates:
            key = idx
            if key not in extrema_dict:
                extrema_dict[key] = (idx, price, ptype)
            else:
                # If same bar has both H and L, prefer the more extreme one
                existing_price = extrema_dict[key][1]
                existing_type = extrema_dict[key][2]
                
                if ptype == 'H' and existing_type == 'L':
                    if price > existing_price:
                        extrema_dict[key] = (idx, price, ptype)
                elif ptype == 'L' and existing_type == 'H':
                    if price < existing_price:
                        extrema_dict[key] = (idx, price, ptype)
        
        extrema = sorted(extrema_dict.values(), key=lambda x: x[0])
        
        # Filter using deviation and backstep
        filtered = []
        
        for idx, price, ptype in extrema:
            if not filtered:
                filtered.append((idx, price, ptype))
                continue
            
            last_idx, last_price, last_type = filtered[-1]
            
            # Same type as previous - keep the more extreme one
            if ptype == last_type:
                if ptype == 'H' and price > last_price:
                    filtered[-1] = (idx, price, ptype)
                elif ptype == 'L' and price < last_price:
                    filtered[-1] = (idx, price, ptype)
                continue
            
            # Different type - check deviation
            if ptype == 'H':
                threshold = last_price * (1 + self.deviation)
                if price >= threshold:
                    filtered.append((idx, price, ptype))
            else:  # 'L'
                threshold = last_price * (1 - self.deviation)
                if price <= threshold:
                    filtered.append((idx, price, ptype))
        
        return filtered
    
    def label_pivots_accurate(self, pivots):
        """
        Label pivots as HH, HL, LH, LL using Pine Script logic
        
        Pine Logic:
        - Track direction (1=up, -1=down)
        - When direction changes, lastPoint = previous pivot price
        - Compare current pivot with lastPoint:
            * uptrend (dir>0): HH if price > lastPoint, else LH
            * downtrend (dir<0): LL if price < lastPoint, else HL
        """
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
        
        # First pivot
        labeled.append({
            'idx': pivots[0][0],
            'price': pivots[0][1],
            'type': pivots[0][2],
            'label': None,
            'direction': None
        })
        
        # Determine initial direction
        if pivots[1][1] > pivots[0][1]:
            current_direction = 1  # Uptrend
        else:
            current_direction = -1  # Downtrend
        
        last_point_price = None
        
        labeled.append({
            'idx': pivots[1][0],
            'price': pivots[1][1],
            'type': pivots[1][2],
            'label': None,
            'direction': current_direction
        })
        
        # Process remaining pivots
        for i in range(2, len(pivots)):
            curr_idx, curr_price, curr_type = pivots[i]
            prev_idx, prev_price, prev_type = pivots[i-1]
            
            # Determine if direction changed
            new_direction = current_direction
            
            if curr_type != prev_type:
                # Type changed = direction changed
                if curr_type == 'H':
                    new_direction = 1
                else:
                    new_direction = -1
                
                # When direction changes, update lastPoint
                if new_direction != current_direction:
                    last_point_price = prev_price
                    current_direction = new_direction
            
            # Label this pivot
            if last_point_price is not None:
                if current_direction > 0:  # Uptrend
                    label = 'HH' if curr_price > last_point_price else 'LH'
                else:  # Downtrend
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

def visualize_zigzag_accurate(df, labeled_pivots, title="ZigZag (Pine Script Accurate)", num_bars=100):
    """Visualize ZigZag with accurate labels"""
    fig, ax = plt.subplots(figsize=(20, 10))
    
    df_plot = df.reset_index(drop=True).copy()
    
    start_idx = max(0, len(df_plot) - num_bars)
    end_idx = len(df_plot)
    df_segment = df_plot.iloc[start_idx:end_idx].copy()
    df_segment = df_segment.reset_index(drop=True)
    
    # Filter pivots in range
    pivots_segment = []
    for pivot in labeled_pivots:
        if start_idx <= pivot['idx'] < end_idx:
            pivot_copy = pivot.copy()
            pivot_copy['idx'] = pivot['idx'] - start_idx
            pivots_segment.append(pivot_copy)
    
    # Plot candlesticks
    ax.plot(df_segment.index, df_segment['high'], color='#cccccc', linewidth=0.5, alpha=0.5)
    ax.plot(df_segment.index, df_segment['low'], color='#cccccc', linewidth=0.5, alpha=0.5)
    ax.plot(df_segment.index, df_segment['close'], color='black', linewidth=1.2, alpha=0.7, label='Close')
    
    # Color mapping for labels
    label_colors = {
        'HH': '#1f77b4',  # Blue - Higher High
        'HL': '#ff7f0e',  # Orange - Higher Low
        'LH': '#2ca02c',  # Green - Lower High
        'LL': '#d62728',  # Red - Lower Low
    }
    
    # Plot labeled pivots
    for pivot in pivots_segment:
        idx = pivot['idx']
        price = pivot['price']
        label = pivot['label']
        
        if label and label in label_colors:
            color = label_colors[label]
            marker = 'o'
            size = 250
            ax.scatter(idx, price, color=color, s=size, marker=marker, 
                      edgecolors='black', linewidth=2.5, zorder=5)
            ax.text(idx, price * 1.012, label, ha='center', va='bottom', 
                   fontsize=11, fontweight='bold', color=color)
        else:
            # Start point or unlabeled
            ax.scatter(idx, price, color='gray', s=150, marker='s', 
                      edgecolors='black', linewidth=2, zorder=5, alpha=0.6)
    
    # Draw ZigZag lines connecting pivots
    if len(pivots_segment) > 1:
        sorted_pivots = sorted(pivots_segment, key=lambda x: x['idx'])
        for i in range(len(sorted_pivots) - 1):
            x1 = sorted_pivots[i]['idx']
            y1 = sorted_pivots[i]['price']
            x2 = sorted_pivots[i+1]['idx']
            y2 = sorted_pivots[i+1]['price']
            
            next_label = sorted_pivots[i+1]['label']
            line_color = label_colors.get(next_label, '#888888')
            
            ax.plot([x1, x2], [y1, y2], color=line_color, linewidth=2.5, alpha=0.7, zorder=3)
    
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Bar Index (from right)', fontsize=12)
    ax.set_ylabel('Price', fontsize=12)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    # Legend
    legend_elements = [
        mpatches.Patch(facecolor='#1f77b4', edgecolor='black', label='HH - Higher High (Uptrend)'),
        mpatches.Patch(facecolor='#ff7f0e', edgecolor='black', label='HL - Higher Low (Downtrend)'),
        mpatches.Patch(facecolor='#2ca02c', edgecolor='black', label='LH - Lower High (Downtrend)'),
        mpatches.Patch(facecolor='#d62728', edgecolor='black', label='LL - Lower Low (Downtrend)'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=11, framealpha=0.95)
    
    plt.tight_layout()
    return fig, ax

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == '__main__':
    print("="*80)
    print("ZigZag Implementation - Pine Script Accurate Logic")
    print("="*80)
    
    # Fetch data
    print("\n[Step 1/4] Fetching market data...")
    try:
        df = yf.download('BTC-USD', period='2y', interval='1h', progress=False)
        
        # Handle MultiIndex columns
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        df = df.reset_index()
        df.columns = [c.lower().strip() for c in df.columns]
        
        # Ensure proper column names
        df = df.rename(columns={
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'close': 'close',
            'volume': 'volume',
            'date': 'datetime',
            'index': 'datetime'
        })
        
        print(f"  OK - Loaded {len(df):,} bars")
        print(f"      Date range: {df['datetime'].min()} to {df['datetime'].max()}")
        print(f"      Price range: {df['close'].min():.2f} - {df['close'].max():.2f}")
    
    except Exception as e:
        print(f"  ERROR: {e}")
        exit(1)
    
    # Initialize ZigZag
    print("\n[Step 2/4] Finding pivot points...")
    
    zz = ZigZagPineScript(depth=12, deviation=5, backstep=2)
    
    highs = df['high'].values
    lows = df['low'].values
    
    pivots = zz.find_pivots(highs, lows)
    print(f"  OK - Found {len(pivots):,} pivot points")
    
    # Label pivots
    print("\n[Step 3/4] Labeling pivots (HH/HL/LH/LL)...")
    
    labeled_pivots = zz.label_pivots_accurate(pivots)
    
    # Statistics
    hh_count = sum(1 for p in labeled_pivots if p['label'] == 'HH')
    hl_count = sum(1 for p in labeled_pivots if p['label'] == 'HL')
    lh_count = sum(1 for p in labeled_pivots if p['label'] == 'LH')
    ll_count = sum(1 for p in labeled_pivots if p['label'] == 'LL')
    start_count = sum(1 for p in labeled_pivots if p['label'] is None)
    
    print(f"  OK - Label distribution:")
    print(f"       HH (Higher High):    {hh_count:6d}  ({100*hh_count/len(labeled_pivots):5.2f}%)")
    print(f"       HL (Higher Low):     {hl_count:6d}  ({100*hl_count/len(labeled_pivots):5.2f}%)")
    print(f"       LH (Lower High):     {lh_count:6d}  ({100*lh_count/len(labeled_pivots):5.2f}%)")
    print(f"       LL (Lower Low):      {ll_count:6d}  ({100*ll_count/len(labeled_pivots):5.2f}%)")
    print(f"       START (Unlabeled):   {start_count:6d}  ({100*start_count/len(labeled_pivots):5.2f}%)")
    print(f"       TOTAL:               {len(labeled_pivots):6d}")
    
    # Show recent pivots
    print(f"\n  Last 20 pivots:")
    print(f"  {'#':>3s} {'Bar':>6s} {'Label':>8s} {'Type':>5s} {'Price':>12s} {'Direction':>10s}")
    print("  " + "-"*60)
    
    for i, pivot in enumerate(labeled_pivots[-20:], 1):
        label = pivot['label'] if pivot['label'] else 'START'
        direction = 'UP' if pivot['direction'] == 1 else ('DOWN' if pivot['direction'] == -1 else 'N/A')
        print(f"  {i:3d} {pivot['idx']:6d} {label:>8s} {pivot['type']:>5s} {pivot['price']:12.2f} {direction:>10s}")
    
    # Visualization
    print("\n[Step 4/4] Creating visualization...")
    
    fig, ax = visualize_zigzag_accurate(
        df,
        labeled_pivots,
        title="ZigZag Pattern - Pine Script Accurate (depth=12, deviation=5%)",
        num_bars=150
    )
    
    plt.savefig('zigzag_visualization.png', dpi=150, bbox_inches='tight')
    print(f"  OK - Chart saved as 'zigzag_visualization.png'")
    
    plt.show()
    
    print("\n" + "="*80)
    print("Complete!")
    print("="*80)
    
    # Create and save pivot data
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
    print("\nPivot data saved to 'zigzag_pivots.csv'")
