"""
Outlier Detector

Multiple algorithms for anomaly and outlier detection.
From simple IQR to sophisticated ML-based methods.
"""

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import polars as pl
from numba import jit
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

from config import get_settings


@dataclass
class OutlierResult:
    """Result of outlier detection."""
    
    column: str
    method: str
    outlier_count: int
    total_count: int
    outlier_percentage: float
    outlier_indices: list[int]
    outlier_values: list[float]
    lower_bound: Optional[float]
    upper_bound: Optional[float]
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "column": self.column,
            "method": self.method,
            "outlier_count": self.outlier_count,
            "total_count": self.total_count,
            "outlier_percentage": round(self.outlier_percentage, 2),
            "outlier_indices": self.outlier_indices[:50],  # Limit for response size
            "outlier_values": [round(v, 4) for v in self.outlier_values[:50]],
            "lower_bound": round(self.lower_bound, 4) if self.lower_bound else None,
            "upper_bound": round(self.upper_bound, 4) if self.upper_bound else None,
        }


@dataclass
class MultivariateOutlierResult:
    """Result of multivariate outlier detection."""
    
    method: str
    outlier_count: int
    total_count: int
    outlier_percentage: float
    outlier_indices: list[int]
    anomaly_scores: list[float]
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "outlier_count": self.outlier_count,
            "total_count": self.total_count,
            "outlier_percentage": round(self.outlier_percentage, 2),
            "outlier_indices": self.outlier_indices[:100],
            "anomaly_scores": [round(s, 4) for s in self.anomaly_scores[:100]],
        }


@jit(nopython=True, cache=True)
def _iqr_bounds_numba(arr: np.ndarray, multiplier: float) -> tuple[float, float, float, float]:
    """Numba-accelerated IQR bounds calculation."""
    sorted_arr = np.sort(arr)
    n = len(sorted_arr)
    
    q25_idx = int(n * 0.25)
    q75_idx = int(n * 0.75)
    
    q25 = sorted_arr[q25_idx]
    q75 = sorted_arr[q75_idx]
    iqr = q75 - q25
    
    lower = q25 - multiplier * iqr
    upper = q75 + multiplier * iqr
    
    return q25, q75, lower, upper


@jit(nopython=True, cache=True)
def _detect_iqr_outliers_numba(
    arr: np.ndarray,
    lower: float,
    upper: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Numba-accelerated IQR outlier detection."""
    n = len(arr)
    mask = np.zeros(n, dtype=np.bool_)
    
    for i in range(n):
        if arr[i] < lower or arr[i] > upper:
            mask[i] = True
    
    indices = np.where(mask)[0]
    values = arr[mask]
    
    return indices, values


class OutlierDetector:
    """Multi-algorithm outlier detection engine."""
    
    def __init__(self):
        self.settings = get_settings()
    
    def detect_iqr(
        self,
        df: pl.DataFrame,
        column: str,
        multiplier: Optional[float] = None,
    ) -> OutlierResult:
        """
        Detect outliers using Interquartile Range (IQR) method.
        
        Fast and interpretable, good for normally distributed data.
        """
        if multiplier is None:
            multiplier = self.settings.analysis.outlier_iqr_multiplier
        
        col = df[column].drop_nulls()
        arr = col.to_numpy().astype(np.float64)
        
        if len(arr) < 4:
            return OutlierResult(
                column=column, method="iqr", outlier_count=0,
                total_count=len(arr), outlier_percentage=0,
                outlier_indices=[], outlier_values=[],
                lower_bound=None, upper_bound=None,
            )
        
        # Numba-accelerated computation
        q25, q75, lower, upper = _iqr_bounds_numba(arr, multiplier)
        indices, values = _detect_iqr_outliers_numba(arr, lower, upper)
        
        return OutlierResult(
            column=column,
            method="iqr",
            outlier_count=len(indices),
            total_count=len(arr),
            outlier_percentage=len(indices) / len(arr) * 100 if len(arr) > 0 else 0,
            outlier_indices=[int(i) for i in indices],
            outlier_values=[float(v) for v in values],
            lower_bound=float(lower),
            upper_bound=float(upper),
        )
    
    def detect_zscore(
        self,
        df: pl.DataFrame,
        column: str,
        threshold: Optional[float] = None,
        use_modified: bool = True,
    ) -> OutlierResult:
        """
        Detect outliers using Z-score method.
        
        Args:
            df: DataFrame
            column: Column name
            threshold: Z-score threshold (default from config)
            use_modified: Use modified Z-score (more robust to outliers)
        """
        if threshold is None:
            threshold = self.settings.analysis.outlier_zscore_threshold
        
        col = df[column].drop_nulls()
        arr = col.to_numpy().astype(np.float64)
        
        if len(arr) < 3:
            return OutlierResult(
                column=column, method="zscore", outlier_count=0,
                total_count=len(arr), outlier_percentage=0,
                outlier_indices=[], outlier_values=[],
                lower_bound=None, upper_bound=None,
            )
        
        if use_modified:
            # Modified Z-score using median (more robust)
            median = np.median(arr)
            mad = np.median(np.abs(arr - median))
            
            # Scale factor for normal distribution
            if mad == 0:
                z_scores = np.zeros_like(arr)
            else:
                z_scores = 0.6745 * (arr - median) / mad
        else:
            # Standard Z-score
            mean = np.mean(arr)
            std = np.std(arr)
            
            if std == 0:
                z_scores = np.zeros_like(arr)
            else:
                z_scores = (arr - mean) / std
        
        # Find outliers
        outlier_mask = np.abs(z_scores) > threshold
        indices = np.where(outlier_mask)[0]
        values = arr[outlier_mask]
        
        # Calculate bounds
        if use_modified:
            median = np.median(arr)
            mad = np.median(np.abs(arr - median))
            lower = median - threshold * mad / 0.6745
            upper = median + threshold * mad / 0.6745
        else:
            mean = np.mean(arr)
            std = np.std(arr)
            lower = mean - threshold * std
            upper = mean + threshold * std
        
        return OutlierResult(
            column=column,
            method="modified_zscore" if use_modified else "zscore",
            outlier_count=len(indices),
            total_count=len(arr),
            outlier_percentage=len(indices) / len(arr) * 100,
            outlier_indices=[int(i) for i in indices],
            outlier_values=[float(v) for v in values],
            lower_bound=float(lower),
            upper_bound=float(upper),
        )
    
    def detect_isolation_forest(
        self,
        df: pl.DataFrame,
        columns: list[str],
        contamination: float = 0.05,
        n_estimators: int = 100,
    ) -> MultivariateOutlierResult:
        """
        Detect multivariate outliers using Isolation Forest.
        
        Excellent for high-dimensional data and finding anomalies
        that aren't obvious in individual dimensions.
        """
        # Get numeric data
        data = df.select(columns).drop_nulls().to_numpy().astype(np.float64)
        
        if len(data) < 10:
            return MultivariateOutlierResult(
                method="isolation_forest",
                outlier_count=0,
                total_count=len(data),
                outlier_percentage=0,
                outlier_indices=[],
                anomaly_scores=[],
            )
        
        # Standardize features
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)
        
        # Fit Isolation Forest
        iso_forest = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=42,
            n_jobs=-1,  # Use all cores
        )
        
        predictions = iso_forest.fit_predict(data_scaled)
        scores = iso_forest.decision_function(data_scaled)
        
        # Outliers are labeled as -1
        outlier_indices = np.where(predictions == -1)[0]
        
        return MultivariateOutlierResult(
            method="isolation_forest",
            outlier_count=len(outlier_indices),
            total_count=len(data),
            outlier_percentage=len(outlier_indices) / len(data) * 100,
            outlier_indices=[int(i) for i in outlier_indices],
            anomaly_scores=[float(s) for s in scores],
        )
    
    def detect_lof(
        self,
        df: pl.DataFrame,
        columns: list[str],
        n_neighbors: int = 20,
        contamination: float = 0.05,
    ) -> MultivariateOutlierResult:
        """
        Detect outliers using Local Outlier Factor (LOF).
        
        Good for detecting local anomalies in variable-density data.
        """
        data = df.select(columns).drop_nulls().to_numpy().astype(np.float64)
        
        if len(data) < n_neighbors + 1:
            return MultivariateOutlierResult(
                method="lof",
                outlier_count=0,
                total_count=len(data),
                outlier_percentage=0,
                outlier_indices=[],
                anomaly_scores=[],
            )
        
        # Standardize
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)
        
        # Fit LOF
        lof = LocalOutlierFactor(
            n_neighbors=min(n_neighbors, len(data) - 1),
            contamination=contamination,
            n_jobs=-1,
        )
        
        predictions = lof.fit_predict(data_scaled)
        scores = lof.negative_outlier_factor_
        
        outlier_indices = np.where(predictions == -1)[0]
        
        return MultivariateOutlierResult(
            method="lof",
            outlier_count=len(outlier_indices),
            total_count=len(data),
            outlier_percentage=len(outlier_indices) / len(data) * 100,
            outlier_indices=[int(i) for i in outlier_indices],
            anomaly_scores=[float(s) for s in scores],
        )
    
    def detect_all_methods(
        self,
        df: pl.DataFrame,
        column: str,
    ) -> dict[str, OutlierResult]:
        """
        Run all univariate outlier detection methods.
        
        Returns results from IQR, Z-score, and modified Z-score.
        """
        return {
            "iqr": self.detect_iqr(df, column),
            "zscore": self.detect_zscore(df, column, use_modified=False),
            "modified_zscore": self.detect_zscore(df, column, use_modified=True),
        }
    
    def get_consensus_outliers(
        self,
        df: pl.DataFrame,
        column: str,
        min_agreement: int = 2,
    ) -> OutlierResult:
        """
        Get outliers agreed upon by multiple methods.
        
        More robust than single-method detection.
        """
        methods = self.detect_all_methods(df, column)
        
        # Count votes for each index
        vote_counts: dict[int, int] = {}
        for result in methods.values():
            for idx in result.outlier_indices:
                vote_counts[idx] = vote_counts.get(idx, 0) + 1
        
        # Get indices with sufficient agreement
        consensus_indices = [
            idx for idx, count in vote_counts.items()
            if count >= min_agreement
        ]
        
        # Get values
        col = df[column].drop_nulls()
        arr = col.to_numpy()
        values = [float(arr[i]) for i in consensus_indices if i < len(arr)]
        
        return OutlierResult(
            column=column,
            method=f"consensus_{min_agreement}",
            outlier_count=len(consensus_indices),
            total_count=len(arr),
            outlier_percentage=len(consensus_indices) / len(arr) * 100 if len(arr) > 0 else 0,
            outlier_indices=sorted(consensus_indices),
            outlier_values=values,
            lower_bound=None,
            upper_bound=None,
        )


# Global instance
outlier_detector = OutlierDetector()
