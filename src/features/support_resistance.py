from typing import Dict, Any, List
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
from sklearn.cluster import AgglomerativeClustering
from .base import Feature, FeatureOutput, LevelOutput, FeatureResult

class SupportResistance(Feature):
    @property
    def name(self) -> str:
        return "Support & Resistance"

    @property
    def description(self) -> str:
        return "Identifies key price clusters using Local Extrema, Fractals, or ZigZag."

    @property
    def category(self) -> str:
        return "Price Levels"

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "method": ["ZigZag", "Savitzky-Golay", "Bill Williams"], 
            "threshold_pct": 0.015, # For identifying the pivot itself (ZigZag)
            "window": 2, # For Bill Williams Fractals
            "clustering_pct": 0.02, # For merging nearby pivots
            "min_strength": 1.0
        }

    def compute(self, df: pd.DataFrame, params: Dict[str, Any]) -> FeatureResult:
        method = params.get("method", "ZigZag")
        threshold = float(params.get("threshold_pct", 0.015))
        window = int(params.get("window", 2))
        cluster_thresh = float(params.get("clustering_pct", 0.02))
        min_str = float(params.get("min_strength", 1.0))

        pivots = []
        if method == "Savitzky-Golay":
            pivots = self.get_pivots_smoothed(df, window=5) # Fixed window for stability
        elif method == "Bill Williams":
            pivots = self.get_pivots_bill_williams_vectorized(df, window=window)
        else: # ZigZag (Iterative by nature for live, but we process historically here)
            pivots = self.get_pivots_zigzag(df, deviation_pct=threshold) 

        clusters = self.cluster_pivots(pivots, cluster_thresh)
        
        visuals = []
        for c in clusters:
            if c['strength'] >= min_str:
                visuals.append(LevelOutput(
                    name=f"Level {c['price']}",
                    price=c['price'],
                    min_price=c['min_price'],
                    max_price=c['max_price'],
                    strength=c['strength'],
                    color='#0000ff'
                ))
        
        if not clusters:
             return FeatureResult(visuals=visuals, data={})

        # Vectorized distance to nearest support and resistance
        prices = df['Close'].values
        all_levels = np.sort([c['price'] for c in clusters])
        
        # Use searchsorted for O(log n) lookup
        idx = np.searchsorted(all_levels, prices)
        
        # Support: Highest level <= current price
        supp_idx = np.maximum(idx - 1, 0)
        supp_levels = all_levels[supp_idx]
        dist_to_supp = (prices - supp_levels) / prices
        # Handle prices below all levels
        dist_to_supp[prices < all_levels[0]] = 0.0
        
        # Resistance: Lowest level >= current price
        res_idx = np.minimum(idx, len(all_levels) - 1)
        res_levels = all_levels[res_idx]
        dist_to_res = (res_levels - prices) / prices
        # Handle prices above all levels
        dist_to_res[prices > all_levels[-1]] = 0.0

        return FeatureResult(visuals=visuals, data={
            "Dist_to_Support": pd.Series(dist_to_supp, index=df.index),
            "Dist_to_Resistance": pd.Series(dist_to_res, index=df.index)
        })

    def get_pivots_bill_williams_vectorized(self, df, window=2):
        """Purely vectorized Bill Williams Fractals with T-Zero Rule enforcement."""
        lows = df['Low']
        highs = df['High']
        
        # Bullish Fractal (Support)
        is_support = True
        for j in range(1, window + 1):
            is_support &= (lows < lows.shift(j)) & (lows < lows.shift(-j))
        
        # Shift forward by 'window' bars to record ONLY when confirmed
        # (e.g., a 2-bar fractal is confirmed after the 2nd candle to the right closes)
        confirmed_support = is_support.shift(window)
        
        # Bearish Fractal (Resistance)
        is_resistance = True
        for j in range(1, window + 1):
            is_resistance &= (highs > highs.shift(j)) & (highs > highs.shift(-j))
            
        confirmed_resistance = is_resistance.shift(window)
        
        pivots = []
        # Support Pivots
        supp_indices = np.where(confirmed_support == True)[0]
        for idx in supp_indices:
            # The actual pivot price is from 'window' bars ago
            pivots.append({'price': df['Low'].iloc[idx - window], 'index': idx, 'type': 'support'})
            
        # Resistance Pivots
        res_indices = np.where(confirmed_resistance == True)[0]
        for idx in res_indices:
            pivots.append({'price': df['High'].iloc[idx - window], 'index': idx, 'type': 'resistance'})
            
        return pivots

    def get_pivots_smoothed(self, df, window=5, polyorder=3):
        """Savitzky-Golay with confirmation delay (T-Zero)."""
        if window % 2 == 0: window += 1
        if len(df) <= window: return []

        smoothed_high = savgol_filter(df['High'], window, polyorder)
        smoothed_low = savgol_filter(df['Low'], window, polyorder)
        
        pivots = []
        half = window // 2
        # We start looking only after we have enough data to confirm the center point
        for i in range(half, len(df) - half):
            # Confirmation happens at index i + half
            confirmation_idx = i + half
            
            # Support
            if smoothed_low[i] == min(smoothed_low[i-half:i+half+1]):
                pivots.append({'price': df['Low'].iloc[i], 'index': confirmation_idx, 'type': 'support'})
            # Resistance
            if smoothed_high[i] == max(smoothed_high[i-half:i+half+1]):
                pivots.append({'price': df['High'].iloc[i], 'index': confirmation_idx, 'type': 'resistance'})
        return pivots

    def get_pivots_zigzag(self, df, deviation_pct=0.05):
        """ZigZag is naturally iterative but enforces the T-Zero rule."""
        pivots = []
        last_pivot_price = df['Close'].iloc[0]
        last_pivot_type = None 
        
        for i in range(1, len(df)):
            price_high = df['High'].iloc[i]
            price_low = df['Low'].iloc[i]
            
            diff_high = (price_high - last_pivot_price) / last_pivot_price
            diff_low = (price_low - last_pivot_price) / last_pivot_price
            
            if last_pivot_type is None:
                if diff_high >= deviation_pct:
                    last_pivot_type = 'H'
                    last_pivot_price = price_high
                    pivots.append({'price': price_high, 'index': i, 'type': 'resistance'})
                elif diff_low <= -deviation_pct:
                    last_pivot_type = 'L'
                    last_pivot_price = price_low
                    pivots.append({'price': price_low, 'index': i, 'type': 'support'})
            elif last_pivot_type == 'H':
                if price_high > last_pivot_price:
                    last_pivot_price = price_high
                    pivots[-1] = {'price': price_high, 'index': i, 'type': 'resistance'}
                elif diff_low <= -deviation_pct:
                    last_pivot_type = 'L'
                    last_pivot_price = price_low
                    pivots.append({'price': price_low, 'index': i, 'type': 'support'})
            elif last_pivot_type == 'L':
                if price_low < last_pivot_price:
                    last_pivot_price = price_low
                    pivots[-1] = {'price': price_low, 'index': i, 'type': 'support'}
                elif diff_high >= deviation_pct:
                    last_pivot_type = 'H'
                    last_pivot_price = price_high
                    pivots.append({'price': price_high, 'index': i, 'type': 'resistance'})
        return pivots

    def cluster_pivots(self, pivots, threshold_pct):
        if not pivots:
            return []

        if len(pivots) == 1:
            p = pivots[0]
            return [{
                'price': round(float(p['price']), 2),
                'min_price': round(float(p['price']), 2),
                'max_price': round(float(p['price']), 2),
                'strength': 1
            }]

        prices = np.array([p['price'] for p in pivots]).reshape(-1, 1)
        avg_price = np.mean(prices)
        dist_threshold = avg_price * threshold_pct

        model = AgglomerativeClustering(
            n_clusters=None, 
            distance_threshold=dist_threshold, 
            linkage='complete'
        )
        
        clusters = model.fit_predict(prices)
        levels = []
        for cluster_id in np.unique(clusters):
            cluster_prices = prices[clusters == cluster_id]
            level_price = np.mean(cluster_prices)
            
            levels.append({
                'price': round(float(level_price), 2),
                'min_price': round(float(np.min(cluster_prices)), 2),
                'max_price': round(float(np.max(cluster_prices)), 2),
                'strength': len(cluster_prices)
            })
            
        return sorted(levels, key=lambda x: x['strength'], reverse=True)
