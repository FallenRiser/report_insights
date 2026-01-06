"""
Trend Detector

Time series trend detection with change point analysis.
Implements multiple algorithms for robust trend identification.
"""

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import polars as pl
from numba import jit
from scipy import stats as scipy_stats
from scipy.signal import find_peaks

from config import get_settings


@dataclass
class TrendResult:
    """Result of trend analysis."""
    
    column: str
    direction: str  # increasing, decreasing, stable
    slope: float
    intercept: float
    r_squared: float
    p_value: float
    is_significant: bool
    strength: str  # weak, moderate, strong
    percent_change: float
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "column": self.column,
            "direction": self.direction,
            "slope": round(self.slope, 6),
            "intercept": round(self.intercept, 4),
            "r_squared": round(self.r_squared, 4),
            "p_value": round(self.p_value, 6),
            "is_significant": self.is_significant,
            "strength": self.strength,
            "percent_change": round(self.percent_change, 2),
        }


@dataclass
class ChangePoint:
    """A detected change point in the time series."""
    
    index: int
    value: float
    change_magnitude: float
    direction: str  # increase, decrease
    confidence: float
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "value": round(self.value, 4),
            "change_magnitude": round(self.change_magnitude, 4),
            "direction": self.direction,
            "confidence": round(self.confidence, 4),
        }


@dataclass
class MovingAverageResult:
    """Moving average analysis result."""
    
    column: str
    window_size: int
    sma: list[float]  # Simple moving average
    ema: list[float]  # Exponential moving average
    trend_direction: str
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "column": self.column,
            "window_size": self.window_size,
            "trend_direction": self.trend_direction,
            "sma_last": round(self.sma[-1], 4) if self.sma else None,
            "ema_last": round(self.ema[-1], 4) if self.ema else None,
        }


@jit(nopython=True, cache=True)
def _cumsum_numba(arr: np.ndarray) -> np.ndarray:
    """Numba-accelerated cumulative sum."""
    result = np.empty_like(arr)
    result[0] = arr[0]
    for i in range(1, len(arr)):
        result[i] = result[i-1] + arr[i]
    return result


@jit(nopython=True, cache=True)
def _ema_numba(arr: np.ndarray, alpha: float) -> np.ndarray:
    """Numba-accelerated exponential moving average."""
    n = len(arr)
    result = np.empty(n, dtype=np.float64)
    result[0] = arr[0]
    
    for i in range(1, n):
        result[i] = alpha * arr[i] + (1 - alpha) * result[i-1]
    
    return result


@jit(nopython=True, cache=True)
def _sma_numba(arr: np.ndarray, window: int) -> np.ndarray:
    """Numba-accelerated simple moving average."""
    n = len(arr)
    result = np.empty(n, dtype=np.float64)
    
    # Fill initial values with NaN
    for i in range(window - 1):
        result[i] = np.nan
    
    # Compute moving average
    window_sum = np.sum(arr[:window])
    result[window - 1] = window_sum / window
    
    for i in range(window, n):
        window_sum = window_sum - arr[i - window] + arr[i]
        result[i] = window_sum / window
    
    return result


class TrendDetector:
    """Trend detection engine with multiple algorithms."""
    
    def __init__(self):
        self.settings = get_settings()
    
    def detect_linear_trend(
        self,
        df: pl.DataFrame,
        column: str,
        time_column: Optional[str] = None,
    ) -> TrendResult:
        """
        Detect linear trend using regression.
        
        Args:
            df: DataFrame
            column: Value column
            time_column: Optional time column (uses index if None)
            
        Returns:
            TrendResult with slope, r-squared, and significance
        """
        values = df[column].drop_nulls().to_numpy().astype(np.float64)
        
        if len(values) < self.settings.analysis.trend_min_points:
            return TrendResult(
                column=column,
                direction="insufficient_data",
                slope=0, intercept=0, r_squared=0, p_value=1,
                is_significant=False, strength="none", percent_change=0
            )
        
        # Create time index
        if time_column and time_column in df.columns:
            x = np.arange(len(values))  # Use ordinal for simplicity
        else:
            x = np.arange(len(values))
        
        # Linear regression
        slope, intercept, r_value, p_value, std_err = scipy_stats.linregress(x, values)
        r_squared = r_value ** 2
        
        # Determine direction
        significance_level = self.settings.analysis.correlation_significance_level
        is_significant = p_value < significance_level
        
        if not is_significant:
            direction = "stable"
        elif slope > 0:
            direction = "increasing"
        else:
            direction = "decreasing"
        
        # Determine strength
        if r_squared >= 0.7:
            strength = "strong"
        elif r_squared >= 0.4:
            strength = "moderate"
        elif r_squared >= 0.2:
            strength = "weak"
        else:
            strength = "very_weak"
        
        # Calculate percent change
        start_val = values[0]
        end_val = values[-1]
        if start_val != 0:
            percent_change = ((end_val - start_val) / abs(start_val)) * 100
        else:
            percent_change = 0
        
        return TrendResult(
            column=column,
            direction=direction,
            slope=float(slope),
            intercept=float(intercept),
            r_squared=float(r_squared),
            p_value=float(p_value),
            is_significant=is_significant,
            strength=strength,
            percent_change=float(percent_change),
        )
    
    def mann_kendall_test(
        self,
        df: pl.DataFrame,
        column: str,
    ) -> dict[str, Any]:
        """
        Mann-Kendall test for monotonic trend.
        
        Non-parametric test that's robust to outliers.
        """
        values = df[column].drop_nulls().to_numpy().astype(np.float64)
        n = len(values)
        
        if n < 4:
            return {
                "column": column,
                "trend": "insufficient_data",
                "p_value": 1.0,
                "z_score": 0,
                "significant": False,
            }
        
        # Calculate S statistic
        s = 0
        for i in range(n - 1):
            for j in range(i + 1, n):
                diff = values[j] - values[i]
                if diff > 0:
                    s += 1
                elif diff < 0:
                    s -= 1
        
        # Calculate variance
        # Account for ties
        unique, counts = np.unique(values, return_counts=True)
        ties = counts[counts > 1]
        
        var_s = (n * (n - 1) * (2 * n + 5)) / 18
        if len(ties) > 0:
            tie_correction = np.sum(ties * (ties - 1) * (2 * ties + 5)) / 18
            var_s -= tie_correction
        
        # Calculate z-score
        if s > 0:
            z = (s - 1) / np.sqrt(var_s)
        elif s < 0:
            z = (s + 1) / np.sqrt(var_s)
        else:
            z = 0
        
        # Two-tailed p-value
        p_value = 2 * (1 - scipy_stats.norm.cdf(abs(z)))
        
        # Determine trend
        significance = self.settings.analysis.correlation_significance_level
        if p_value < significance:
            trend = "increasing" if z > 0 else "decreasing"
        else:
            trend = "no_trend"
        
        return {
            "column": column,
            "trend": trend,
            "s_statistic": int(s),
            "z_score": round(float(z), 4),
            "p_value": round(float(p_value), 6),
            "significant": p_value < significance,
        }
    
    def detect_change_points(
        self,
        df: pl.DataFrame,
        column: str,
        min_segment_length: int = 5,
        penalty: float = 3.0,
    ) -> list[ChangePoint]:
        """
        Detect change points using PELT-like algorithm.
        
        Identifies points where the statistical properties change significantly.
        """
        values = df[column].drop_nulls().to_numpy().astype(np.float64)
        n = len(values)
        
        if n < min_segment_length * 2:
            return []
        
        # Use binary segmentation approach (simpler but effective)
        change_points = self._binary_segmentation(
            values, 
            min_segment_length, 
            penalty
        )
        
        # Create ChangePoint objects
        results = []
        for idx in change_points:
            if idx > 0 and idx < n - 1:
                before_mean = np.mean(values[max(0, idx-min_segment_length):idx])
                after_mean = np.mean(values[idx:min(n, idx+min_segment_length)])
                change_mag = after_mean - before_mean
                
                # Calculate confidence based on segment difference
                segment_std = np.std(values)
                if segment_std > 0:
                    confidence = min(1.0, abs(change_mag) / (2 * segment_std))
                else:
                    confidence = 1.0
                
                results.append(ChangePoint(
                    index=int(idx),
                    value=float(values[idx]),
                    change_magnitude=float(change_mag),
                    direction="increase" if change_mag > 0 else "decrease",
                    confidence=float(confidence),
                ))
        
        return results
    
    def _binary_segmentation(
        self,
        values: np.ndarray,
        min_length: int,
        penalty: float,
    ) -> list[int]:
        """Binary segmentation for change point detection."""
        n = len(values)
        
        def cost(start: int, end: int) -> float:
            """Compute cost of a segment (negative log-likelihood for normal)."""
            if end - start < 2:
                return 0
            segment = values[start:end]
            var = np.var(segment)
            if var == 0:
                return 0
            return (end - start) * np.log(var)
        
        def find_best_split(start: int, end: int) -> tuple[int, float]:
            """Find best split point in a segment."""
            best_idx = -1
            best_gain = 0
            
            base_cost = cost(start, end)
            
            for i in range(start + min_length, end - min_length):
                split_cost = cost(start, i) + cost(i, end)
                gain = base_cost - split_cost - penalty
                
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i
            
            return best_idx, best_gain
        
        # Recursively find change points
        change_points = []
        segments = [(0, n)]
        
        while segments:
            start, end = segments.pop()
            if end - start < 2 * min_length:
                continue
            
            best_idx, best_gain = find_best_split(start, end)
            
            if best_idx > 0 and best_gain > 0:
                change_points.append(best_idx)
                segments.append((start, best_idx))
                segments.append((best_idx, end))
        
        return sorted(change_points)
    
    def compute_moving_averages(
        self,
        df: pl.DataFrame,
        column: str,
        window: int = 7,
    ) -> MovingAverageResult:
        """
        Compute simple and exponential moving averages.
        
        Uses Numba-accelerated functions for speed.
        """
        values = df[column].drop_nulls().to_numpy().astype(np.float64)
        
        if len(values) < window:
            return MovingAverageResult(
                column=column,
                window_size=window,
                sma=[],
                ema=[],
                trend_direction="insufficient_data",
            )
        
        # Simple moving average (Numba accelerated)
        sma = _sma_numba(values, window)
        sma_list = [float(x) if not np.isnan(x) else None for x in sma]
        
        # Exponential moving average (Numba accelerated)
        alpha = 2 / (window + 1)
        ema = _ema_numba(values, alpha)
        ema_list = [float(x) for x in ema]
        
        # Determine trend from EMA
        if len(ema) >= 2:
            recent_ema = ema[-window:]
            if np.mean(np.diff(recent_ema)) > 0:
                trend = "increasing"
            elif np.mean(np.diff(recent_ema)) < 0:
                trend = "decreasing"
            else:
                trend = "stable"
        else:
            trend = "insufficient_data"
        
        return MovingAverageResult(
            column=column,
            window_size=window,
            sma=sma_list,
            ema=ema_list,
            trend_direction=trend,
        )
    
    def analyze_all_trends(
        self,
        df: pl.DataFrame,
        columns: Optional[list[str]] = None,
    ) -> dict[str, TrendResult]:
        """Analyze trends for multiple columns."""
        if columns is None:
            columns = [
                col for col in df.columns
                if df[col].dtype in (
                    pl.Int8, pl.Int16, pl.Int32, pl.Int64,
                    pl.Float32, pl.Float64,
                )
            ]
        
        results = {}
        for col in columns:
            results[col] = self.detect_linear_trend(df, col)
        
        return results


# Global instance
trend_detector = TrendDetector()
