"""
Seasonality Analyzer

Time series seasonality detection using STL decomposition
and Fourier analysis.
"""

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import polars as pl
from scipy import signal
from scipy.fft import fft, fftfreq

from config import get_settings


@dataclass
class SeasonalComponent:
    """A detected seasonal component."""
    
    period: int
    strength: float  # 0-1, how strong this seasonality is
    interpretation: str
    peak_indices: list[int]
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "period": self.period,
            "strength": round(self.strength, 4),
            "interpretation": self.interpretation,
            "peak_indices": self.peak_indices[:10],  # Limit for response
        }


@dataclass
class STLResult:
    """Result of STL decomposition."""
    
    column: str
    trend: list[float]
    seasonal: list[float]
    residual: list[float]
    seasonal_strength: float
    trend_strength: float
    period: int
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "column": self.column,
            "trend": [round(v, 4) for v in self.trend[-100:]],  # Last 100 values
            "seasonal": [round(v, 4) for v in self.seasonal[-100:]],
            "residual": [round(v, 4) for v in self.residual[-100:]],
            "seasonal_strength": round(self.seasonal_strength, 4),
            "trend_strength": round(self.trend_strength, 4),
            "period": self.period,
        }


@dataclass
class SeasonalityResult:
    """Complete seasonality analysis result."""
    
    column: str
    has_seasonality: bool
    primary_period: Optional[int]
    seasonal_components: list[SeasonalComponent]
    stl_decomposition: Optional[STLResult]
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "column": self.column,
            "has_seasonality": self.has_seasonality,
            "primary_period": self.primary_period,
            "seasonal_components": [c.to_dict() for c in self.seasonal_components],
            "stl_decomposition": self.stl_decomposition.to_dict() if self.stl_decomposition else None,
        }


class SeasonalityAnalyzer:
    """Seasonality detection engine."""
    
    COMMON_PERIODS = {
        7: "weekly",
        12: "monthly (within year)",
        24: "hourly (within day)",
        30: "monthly",
        52: "weekly (within year)",
        365: "yearly",
    }
    
    def __init__(self):
        self.settings = get_settings()
    
    def analyze(
        self,
        df: pl.DataFrame,
        column: str,
        time_column: Optional[str] = None,
    ) -> SeasonalityResult:
        """
        Analyze seasonality in a time series.
        
        Args:
            df: DataFrame with time series data
            column: Value column
            time_column: Optional time column (uses index if None)
            
        Returns:
            SeasonalityResult with detected patterns
        """
        values = df[column].drop_nulls().to_numpy().astype(np.float64)
        
        if len(values) < 14:  # Need at least 2 weeks of data
            return SeasonalityResult(
                column=column,
                has_seasonality=False,
                primary_period=None,
                seasonal_components=[],
                stl_decomposition=None,
            )
        
        # Detect periods using FFT
        components = self.detect_periods_fft(values)
        
        # Perform STL decomposition with the primary period
        stl_result = None
        primary_period = None
        
        if components:
            primary_period = components[0].period
            stl_result = self.stl_decomposition(values, primary_period)
        
        has_seasonality = (
            len(components) > 0 and
            components[0].strength > 0.1
        )
        
        return SeasonalityResult(
            column=column,
            has_seasonality=has_seasonality,
            primary_period=primary_period,
            seasonal_components=components,
            stl_decomposition=stl_result,
        )
    
    def detect_periods_fft(
        self,
        values: np.ndarray,
        min_period: int = 2,
        max_period: Optional[int] = None,
        top_n: int = 5,
    ) -> list[SeasonalComponent]:
        """
        Detect seasonal periods using Fast Fourier Transform.
        
        More robust than autocorrelation for complex seasonality.
        """
        n = len(values)
        
        if max_period is None:
            max_period = n // 2
        
        # Remove trend using differencing
        detrended = np.diff(values)
        
        # Apply FFT
        fft_values = fft(detrended)
        frequencies = fftfreq(len(detrended))
        
        # Get power spectrum (magnitude squared)
        power = np.abs(fft_values) ** 2
        
        # Only look at positive frequencies
        positive_mask = frequencies > 0
        positive_freqs = frequencies[positive_mask]
        positive_power = power[positive_mask]
        
        # Convert to periods
        periods = 1 / positive_freqs
        
        # Filter by period range
        valid_mask = (periods >= min_period) & (periods <= max_period)
        valid_periods = periods[valid_mask]
        valid_power = positive_power[valid_mask]
        
        if len(valid_power) == 0:
            return []
        
        # Normalize power
        max_power = np.max(valid_power)
        normalized_power = valid_power / max_power if max_power > 0 else valid_power
        
        # Find peaks in power spectrum
        peak_indices, properties = signal.find_peaks(
            normalized_power,
            height=0.1,
            distance=2,
        )
        
        if len(peak_indices) == 0:
            return []
        
        # Sort by power and take top N
        peak_powers = normalized_power[peak_indices]
        sorted_indices = np.argsort(peak_powers)[::-1][:top_n]
        
        components = []
        for idx in sorted_indices:
            peak_idx = peak_indices[idx]
            period = int(round(valid_periods[peak_idx]))
            strength = float(normalized_power[peak_idx])
            
            # Skip if already have this period
            if any(c.period == period for c in components):
                continue
            
            # Get interpretation
            interpretation = self._interpret_period(period)
            
            # Find peak indices in original series
            peaks = self._find_seasonal_peaks(values, period)
            
            components.append(SeasonalComponent(
                period=period,
                strength=strength,
                interpretation=interpretation,
                peak_indices=peaks,
            ))
        
        return components
    
    def _interpret_period(self, period: int) -> str:
        """Get human-readable interpretation of a period."""
        # Check common periods
        if period in self.COMMON_PERIODS:
            return self.COMMON_PERIODS[period]
        
        # Check close matches
        for known_period, interpretation in self.COMMON_PERIODS.items():
            if abs(period - known_period) <= 1:
                return f"approximately {interpretation}"
        
        # Generic description
        if period <= 7:
            return f"{period}-day cycle"
        elif period <= 31:
            return f"{period}-day cycle (intra-month)"
        elif period <= 90:
            return f"{period}-day cycle (quarterly)"
        elif period <= 180:
            return f"{period}-day cycle (semi-annual)"
        else:
            return f"{period}-day cycle"
    
    def _find_seasonal_peaks(
        self,
        values: np.ndarray,
        period: int,
    ) -> list[int]:
        """Find indices of seasonal peaks."""
        # Simple: find peaks that are approximately 'period' apart
        peaks, _ = signal.find_peaks(values, distance=period // 2)
        return [int(p) for p in peaks[:20]]  # Limit
    
    def stl_decomposition(
        self,
        values: np.ndarray,
        period: int,
    ) -> STLResult:
        """
        Perform STL (Seasonal and Trend decomposition using LOESS).
        
        This is a simplified implementation; statsmodels has a more robust version.
        """
        try:
            from statsmodels.tsa.seasonal import STL
            
            # Need at least 2 full periods
            if len(values) < period * 2:
                period = max(2, len(values) // 2)
            
            stl = STL(values, period=period, robust=True)
            result = stl.fit()
            
            trend = result.trend
            seasonal = result.seasonal
            residual = result.resid
            
            # Calculate strength measures
            # Seasonal strength: 1 - Var(R) / Var(S + R)
            var_residual = np.var(residual)
            var_seasonal_residual = np.var(seasonal + residual)
            seasonal_strength = max(0, 1 - var_residual / var_seasonal_residual) if var_seasonal_residual > 0 else 0
            
            # Trend strength: 1 - Var(R) / Var(T + R)
            var_trend_residual = np.var(trend + residual)
            trend_strength = max(0, 1 - var_residual / var_trend_residual) if var_trend_residual > 0 else 0
            
            return STLResult(
                column="",
                trend=[float(v) for v in trend],
                seasonal=[float(v) for v in seasonal],
                residual=[float(v) for v in residual],
                seasonal_strength=float(seasonal_strength),
                trend_strength=float(trend_strength),
                period=period,
            )
            
        except ImportError:
            # Fallback to simple moving average decomposition
            return self._simple_decomposition(values, period)
    
    def _simple_decomposition(
        self,
        values: np.ndarray,
        period: int,
    ) -> STLResult:
        """Simple decomposition using moving averages."""
        n = len(values)
        
        # Trend: centered moving average
        if period % 2 == 0:
            # Even period: need to center
            ma = np.convolve(values, np.ones(period) / period, mode='valid')
            pad_left = (n - len(ma)) // 2
            pad_right = n - len(ma) - pad_left
            trend = np.pad(ma, (pad_left, pad_right), mode='edge')
        else:
            ma = np.convolve(values, np.ones(period) / period, mode='same')
            trend = ma
        
        # Detrended
        detrended = values - trend
        
        # Seasonal: average for each position in the cycle
        seasonal = np.zeros(n)
        for i in range(period):
            indices = np.arange(i, n, period)
            seasonal_value = np.mean(detrended[indices])
            seasonal[indices] = seasonal_value
        
        # Residual
        residual = values - trend - seasonal
        
        # Calculate strengths
        var_residual = np.var(residual)
        var_seasonal_residual = np.var(seasonal + residual)
        var_trend_residual = np.var(trend + residual)
        
        seasonal_strength = max(0, 1 - var_residual / var_seasonal_residual) if var_seasonal_residual > 0 else 0
        trend_strength = max(0, 1 - var_residual / var_trend_residual) if var_trend_residual > 0 else 0
        
        return STLResult(
            column="",
            trend=[float(v) for v in trend],
            seasonal=[float(v) for v in seasonal],
            residual=[float(v) for v in residual],
            seasonal_strength=float(seasonal_strength),
            trend_strength=float(trend_strength),
            period=period,
        )
    
    def detect_autocorrelation_periods(
        self,
        values: np.ndarray,
        max_lag: Optional[int] = None,
    ) -> list[int]:
        """
        Detect periods using autocorrelation.
        
        Complements FFT for period detection.
        """
        n = len(values)
        if max_lag is None:
            max_lag = n // 3
        
        # Normalize
        values_centered = values - np.mean(values)
        
        # Compute autocorrelation
        autocorr = np.correlate(values_centered, values_centered, mode='full')
        autocorr = autocorr[n-1:n-1+max_lag]  # Take positive lags only
        autocorr = autocorr / autocorr[0]  # Normalize
        
        # Find peaks in autocorrelation
        peaks, _ = signal.find_peaks(autocorr, height=0.3, distance=2)
        
        return [int(p) for p in peaks if p > 1]


# Global instance
seasonality_analyzer = SeasonalityAnalyzer()
