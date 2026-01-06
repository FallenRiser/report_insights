"""
Statistical Analyzer

High-performance descriptive statistics using NumPy and Polars.
Vectorized operations for maximum speed.
"""

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import polars as pl
from numba import jit
from scipy import stats as scipy_stats

from config import get_settings


@dataclass
class DescriptiveStats:
    """Descriptive statistics for a numeric column."""
    
    column: str
    count: int
    mean: float
    std: float
    min: float
    max: float
    median: float
    q25: float
    q75: float
    iqr: float
    skewness: float
    kurtosis: float
    variance: float
    range: float
    cv: Optional[float]  # Coefficient of variation
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "column": self.column,
            "count": self.count,
            "mean": round(self.mean, 4),
            "std": round(self.std, 4),
            "min": round(self.min, 4),
            "max": round(self.max, 4),
            "median": round(self.median, 4),
            "q25": round(self.q25, 4),
            "q75": round(self.q75, 4),
            "iqr": round(self.iqr, 4),
            "skewness": round(self.skewness, 4),
            "kurtosis": round(self.kurtosis, 4),
            "variance": round(self.variance, 4),
            "range": round(self.range, 4),
            "cv": round(self.cv, 4) if self.cv else None,
        }


@dataclass 
class DistributionInfo:
    """Distribution characteristics of a column."""
    
    column: str
    distribution_type: str  # normal, skewed_left, skewed_right, uniform, bimodal
    normality_pvalue: float
    is_normal: bool
    modality: int  # Number of modes
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "column": self.column,
            "distribution_type": self.distribution_type,
            "normality_pvalue": round(self.normality_pvalue, 6),
            "is_normal": self.is_normal,
            "modality": self.modality,
        }


@jit(nopython=True, cache=True)
def _fast_percentile(arr: np.ndarray, percentile: float) -> float:
    """Numba-accelerated percentile calculation."""
    sorted_arr = np.sort(arr)
    n = len(sorted_arr)
    idx = (n - 1) * percentile / 100.0
    lower = int(np.floor(idx))
    upper = int(np.ceil(idx))
    
    if lower == upper:
        return sorted_arr[lower]
    
    weight = idx - lower
    return sorted_arr[lower] * (1 - weight) + sorted_arr[upper] * weight


@jit(nopython=True, cache=True)
def _fast_iqr(arr: np.ndarray) -> tuple[float, float, float]:
    """Numba-accelerated IQR calculation."""
    sorted_arr = np.sort(arr)
    n = len(sorted_arr)
    
    q25_idx = int(n * 0.25)
    q75_idx = int(n * 0.75)
    
    q25 = sorted_arr[q25_idx]
    q75 = sorted_arr[q75_idx]
    iqr = q75 - q25
    
    return q25, q75, iqr


class StatisticalAnalyzer:
    """High-performance statistical analysis engine."""
    
    def __init__(self):
        self.settings = get_settings()
    
    def compute_descriptive_stats(
        self,
        df: pl.DataFrame,
        column: str,
    ) -> DescriptiveStats:
        """
        Compute comprehensive descriptive statistics.
        
        Uses vectorized operations for speed.
        """
        col = df[column].drop_nulls()
        arr = col.to_numpy().astype(np.float64)
        
        if len(arr) == 0:
            return DescriptiveStats(
                column=column, count=0, mean=0, std=0, min=0, max=0,
                median=0, q25=0, q75=0, iqr=0, skewness=0, kurtosis=0,
                variance=0, range=0, cv=None
            )
        
        # Basic stats using NumPy (vectorized)
        count = len(arr)
        mean = float(np.mean(arr))
        std = float(np.std(arr, ddof=1)) if count > 1 else 0
        variance = std ** 2
        min_val = float(np.min(arr))
        max_val = float(np.max(arr))
        median = float(np.median(arr))
        val_range = max_val - min_val
        
        # Quartiles using Numba-accelerated function
        q25, q75, iqr = _fast_iqr(arr)
        
        # Skewness and kurtosis
        try:
            skewness = float(scipy_stats.skew(arr, nan_policy="omit"))
            kurtosis = float(scipy_stats.kurtosis(arr, nan_policy="omit"))
        except Exception:
            skewness = 0.0
            kurtosis = 0.0
        
        # Coefficient of variation
        cv = (std / mean) if mean != 0 else None
        
        return DescriptiveStats(
            column=column,
            count=count,
            mean=mean,
            std=std,
            min=min_val,
            max=max_val,
            median=median,
            q25=float(q25),
            q75=float(q75),
            iqr=float(iqr),
            skewness=skewness,
            kurtosis=kurtosis,
            variance=variance,
            range=val_range,
            cv=cv,
        )
    
    def compute_all_stats(
        self,
        df: pl.DataFrame,
        columns: Optional[list[str]] = None,
    ) -> dict[str, DescriptiveStats]:
        """
        Compute statistics for multiple columns.
        
        Uses Polars for parallel computation.
        """
        if columns is None:
            # Get all numeric columns
            columns = [
                col for col in df.columns
                if df[col].dtype in (
                    pl.Int8, pl.Int16, pl.Int32, pl.Int64,
                    pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
                    pl.Float32, pl.Float64,
                )
            ]
        
        results = {}
        for col in columns:
            results[col] = self.compute_descriptive_stats(df, col)
        
        return results
    
    def analyze_distribution(
        self,
        df: pl.DataFrame,
        column: str,
    ) -> DistributionInfo:
        """
        Analyze the distribution of a numeric column.
        
        Detects normality, skewness direction, and modality.
        """
        col = df[column].drop_nulls()
        arr = col.to_numpy().astype(np.float64)
        
        if len(arr) < 8:
            return DistributionInfo(
                column=column,
                distribution_type="insufficient_data",
                normality_pvalue=1.0,
                is_normal=False,
                modality=1,
            )
        
        # Normality test (Shapiro-Wilk for small samples, D'Agostino for large)
        if len(arr) <= 5000:
            # Use sample for large datasets
            sample = arr if len(arr) <= 5000 else arr[np.random.choice(len(arr), 5000, replace=False)]
            try:
                _, p_value = scipy_stats.shapiro(sample)
            except Exception:
                p_value = 0.0
        else:
            try:
                _, p_value = scipy_stats.normaltest(arr[:5000])
            except Exception:
                p_value = 0.0
        
        is_normal = p_value > 0.05
        
        # Determine skewness direction
        skewness = scipy_stats.skew(arr, nan_policy="omit")
        
        # Detect modality using kernel density estimation
        modality = self._estimate_modality(arr)
        
        # Classify distribution
        if is_normal:
            dist_type = "normal"
        elif modality > 1:
            dist_type = "bimodal" if modality == 2 else "multimodal"
        elif abs(skewness) < 0.5:
            dist_type = "symmetric"
        elif skewness > 0.5:
            dist_type = "skewed_right"
        elif skewness < -0.5:
            dist_type = "skewed_left"
        else:
            dist_type = "unknown"
        
        return DistributionInfo(
            column=column,
            distribution_type=dist_type,
            normality_pvalue=float(p_value),
            is_normal=is_normal,
            modality=modality,
        )
    
    def _estimate_modality(self, arr: np.ndarray, n_bins: int = 50) -> int:
        """Estimate number of modes using histogram analysis."""
        if len(arr) < 10:
            return 1
        
        # Create histogram
        hist, bin_edges = np.histogram(arr, bins=n_bins)
        
        # Smooth histogram
        from scipy.ndimage import gaussian_filter1d
        smoothed = gaussian_filter1d(hist.astype(float), sigma=2)
        
        # Find peaks (local maxima)
        peaks = 0
        for i in range(1, len(smoothed) - 1):
            if smoothed[i] > smoothed[i-1] and smoothed[i] > smoothed[i+1]:
                # Only count significant peaks (> 10% of max)
                if smoothed[i] > 0.1 * np.max(smoothed):
                    peaks += 1
        
        return max(1, peaks)
    
    def compute_confidence_interval(
        self,
        df: pl.DataFrame,
        column: str,
        confidence: float = 0.95,
    ) -> tuple[float, float]:
        """
        Compute confidence interval for the mean.
        
        Args:
            df: DataFrame
            column: Column name
            confidence: Confidence level (default 0.95)
            
        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        col = df[column].drop_nulls()
        arr = col.to_numpy().astype(np.float64)
        
        if len(arr) < 2:
            mean = float(np.mean(arr)) if len(arr) > 0 else 0
            return (mean, mean)
        
        mean = np.mean(arr)
        sem = scipy_stats.sem(arr)
        
        # t-distribution for small samples
        t_val = scipy_stats.t.ppf((1 + confidence) / 2, len(arr) - 1)
        margin = sem * t_val
        
        return (float(mean - margin), float(mean + margin))
    
    def compare_groups(
        self,
        df: pl.DataFrame,
        value_column: str,
        group_column: str,
    ) -> dict[str, Any]:
        """
        Compare statistics across groups.
        
        Performs ANOVA or t-test depending on number of groups.
        """
        groups = df.group_by(group_column).agg(
            pl.col(value_column).alias("values")
        )
        
        group_data = []
        for row in groups.iter_rows(named=True):
            values = row["values"]
            if values is not None:
                group_data.append(np.array(values, dtype=np.float64))
        
        if len(group_data) < 2:
            return {
                "test": "insufficient_groups",
                "p_value": 1.0,
                "significant": False,
            }
        
        # Filter out empty groups
        group_data = [g for g in group_data if len(g) > 0]
        
        if len(group_data) == 2:
            # Two groups: t-test
            stat, p_value = scipy_stats.ttest_ind(group_data[0], group_data[1])
            test_name = "t_test"
        else:
            # Multiple groups: ANOVA
            stat, p_value = scipy_stats.f_oneway(*group_data)
            test_name = "anova"
        
        return {
            "test": test_name,
            "statistic": float(stat),
            "p_value": float(p_value),
            "significant": p_value < self.settings.analysis.correlation_significance_level,
            "n_groups": len(group_data),
        }


# Global instance
statistical_analyzer = StatisticalAnalyzer()
