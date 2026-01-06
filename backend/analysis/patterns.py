"""
Pattern Recognizer

Detects various patterns in data including majorities,
distributions, categorical frequencies, and clustering.
"""

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import polars as pl
from scipy import stats as scipy_stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from config import get_settings


@dataclass
class MajorityPattern:
    """Majority pattern detection result."""
    
    column: str
    dominant_value: Any
    dominant_count: int
    dominant_percentage: float
    is_majority: bool  # >50%
    is_supermajority: bool  # >66%
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "column": self.column,
            "dominant_value": self.dominant_value,
            "dominant_count": self.dominant_count,
            "dominant_percentage": round(self.dominant_percentage, 2),
            "is_majority": self.is_majority,
            "is_supermajority": self.is_supermajority,
        }


@dataclass
class DistributionPattern:
    """Distribution pattern detection result."""
    
    column: str
    pattern_type: str  # uniform, normal, exponential, bimodal, power_law
    confidence: float
    parameters: dict[str, float]
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "column": self.column,
            "pattern_type": self.pattern_type,
            "confidence": round(self.confidence, 4),
            "parameters": {k: round(v, 4) for k, v in self.parameters.items()},
        }


@dataclass
class FrequencyPattern:
    """Categorical frequency pattern."""
    
    column: str
    distribution_type: str  # uniform, zipf, pareto, concentrated
    entropy: float
    normalized_entropy: float  # 0-1, 1 = uniform
    concentration_ratio: float  # Top 20% / Bottom 80%
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "column": self.column,
            "distribution_type": self.distribution_type,
            "entropy": round(self.entropy, 4),
            "normalized_entropy": round(self.normalized_entropy, 4),
            "concentration_ratio": round(self.concentration_ratio, 4),
        }


@dataclass
class ClusterPattern:
    """Clustering pattern result."""
    
    columns: list[str]
    n_clusters: int
    cluster_sizes: list[int]
    cluster_centers: list[list[float]]
    silhouette_score: float
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "columns": self.columns,
            "n_clusters": self.n_clusters,
            "cluster_sizes": self.cluster_sizes,
            "cluster_centers": [
                [round(v, 4) for v in center]
                for center in self.cluster_centers
            ],
            "silhouette_score": round(self.silhouette_score, 4),
        }


class PatternRecognizer:
    """Pattern detection engine."""
    
    def __init__(self):
        self.settings = get_settings()
    
    def detect_majority(
        self,
        df: pl.DataFrame,
        column: str,
    ) -> MajorityPattern:
        """
        Detect majority patterns in categorical data.
        
        Identifies if a single value dominates the column.
        """
        col = df[column].drop_nulls()
        total = len(col)
        
        if total == 0:
            return MajorityPattern(
                column=column,
                dominant_value=None,
                dominant_count=0,
                dominant_percentage=0,
                is_majority=False,
                is_supermajority=False,
            )
        
        # Get value counts
        value_counts = col.value_counts().sort("count", descending=True)
        
        if len(value_counts) == 0:
            return MajorityPattern(
                column=column,
                dominant_value=None,
                dominant_count=0,
                dominant_percentage=0,
                is_majority=False,
                is_supermajority=False,
            )
        
        first_row = value_counts.row(0, named=True)
        value_col = [c for c in value_counts.columns if c != "count"][0]
        
        dominant_value = first_row[value_col]
        dominant_count = first_row["count"]
        percentage = dominant_count / total * 100
        
        return MajorityPattern(
            column=column,
            dominant_value=dominant_value,
            dominant_count=dominant_count,
            dominant_percentage=percentage,
            is_majority=percentage > 50,
            is_supermajority=percentage > 66.67,
        )
    
    def detect_distribution_pattern(
        self,
        df: pl.DataFrame,
        column: str,
    ) -> DistributionPattern:
        """
        Detect the distribution pattern of numeric data.
        
        Tests for normal, exponential, uniform, and other distributions.
        """
        col = df[column].drop_nulls()
        arr = col.to_numpy().astype(np.float64)
        
        if len(arr) < 10:
            return DistributionPattern(
                column=column,
                pattern_type="insufficient_data",
                confidence=0,
                parameters={},
            )
        
        # Standardize for testing
        mean, std = np.mean(arr), np.std(arr)
        
        results = {}
        
        # Test normal distribution
        try:
            _, normal_p = scipy_stats.normaltest(arr[:5000])
            results["normal"] = normal_p
        except Exception:
            results["normal"] = 0
        
        # Test exponential (only for positive data)
        if np.min(arr) >= 0:
            try:
                # Fit exponential and test
                rate = 1 / np.mean(arr) if np.mean(arr) > 0 else 1
                _, exp_p = scipy_stats.kstest(arr, 'expon', args=(0, 1/rate))
                results["exponential"] = exp_p
            except Exception:
                results["exponential"] = 0
        
        # Test uniform
        try:
            _, uniform_p = scipy_stats.kstest(
                arr, 'uniform',
                args=(np.min(arr), np.max(arr) - np.min(arr))
            )
            results["uniform"] = uniform_p
        except Exception:
            results["uniform"] = 0
        
        # Check for bimodality using dip test approximation
        # Use histogram for simple bimodality check
        hist, _ = np.histogram(arr, bins=30)
        peaks = self._count_peaks(hist)
        
        if peaks >= 2:
            results["bimodal"] = 0.5  # Moderate confidence
        
        # Find best fit
        best_type = "unknown"
        best_p = 0
        
        for dist_type, p_value in results.items():
            if p_value > best_p:
                best_p = p_value
                best_type = dist_type
        
        # Get parameters for the best distribution
        params = {"mean": float(mean), "std": float(std)}
        
        if best_type == "exponential":
            params["rate"] = 1 / mean if mean > 0 else 0
        elif best_type == "uniform":
            params["min"] = float(np.min(arr))
            params["max"] = float(np.max(arr))
        
        return DistributionPattern(
            column=column,
            pattern_type=best_type,
            confidence=float(best_p),
            parameters=params,
        )
    
    def _count_peaks(self, hist: np.ndarray) -> int:
        """Count significant peaks in histogram."""
        from scipy.ndimage import gaussian_filter1d
        
        # Smooth
        smoothed = gaussian_filter1d(hist.astype(float), sigma=2)
        
        # Find local maxima
        peaks = 0
        threshold = 0.1 * np.max(smoothed)
        
        for i in range(1, len(smoothed) - 1):
            if smoothed[i] > smoothed[i-1] and smoothed[i] > smoothed[i+1]:
                if smoothed[i] > threshold:
                    peaks += 1
        
        return peaks
    
    def detect_frequency_pattern(
        self,
        df: pl.DataFrame,
        column: str,
    ) -> FrequencyPattern:
        """
        Analyze frequency distribution of categorical data.
        
        Detects patterns like uniform, Zipf's law, Pareto, etc.
        """
        col = df[column].drop_nulls()
        value_counts = col.value_counts().sort("count", descending=True)
        
        if len(value_counts) < 2:
            return FrequencyPattern(
                column=column,
                distribution_type="single_value",
                entropy=0,
                normalized_entropy=0,
                concentration_ratio=float("inf"),
            )
        
        counts = value_counts["count"].to_numpy().astype(float)
        total = np.sum(counts)
        probabilities = counts / total
        
        # Calculate entropy
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
        max_entropy = np.log2(len(counts))
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
        
        # Calculate concentration ratio (80/20 rule check)
        n_values = len(counts)
        top_20_pct = max(1, int(n_values * 0.2))
        top_20_sum = np.sum(counts[:top_20_pct])
        bottom_80_sum = np.sum(counts[top_20_pct:])
        
        concentration = top_20_sum / bottom_80_sum if bottom_80_sum > 0 else float("inf")
        
        # Classify distribution type
        if normalized_entropy > 0.9:
            dist_type = "uniform"
        elif concentration > 4:  # 80/20 rule
            dist_type = "pareto"
        elif normalized_entropy < 0.5:
            dist_type = "concentrated"
        else:
            # Check for Zipf's law
            if self._check_zipf(counts):
                dist_type = "zipf"
            else:
                dist_type = "irregular"
        
        return FrequencyPattern(
            column=column,
            distribution_type=dist_type,
            entropy=float(entropy),
            normalized_entropy=float(normalized_entropy),
            concentration_ratio=float(concentration),
        )
    
    def _check_zipf(self, counts: np.ndarray, threshold: float = 0.8) -> bool:
        """Check if distribution follows Zipf's law."""
        if len(counts) < 5:
            return False
        
        # Zipf's law: frequency ∝ 1/rank
        ranks = np.arange(1, len(counts) + 1)
        expected = counts[0] / ranks
        
        # Calculate correlation
        try:
            r, _ = scipy_stats.pearsonr(counts, expected)
            return r > threshold
        except Exception:
            return False
    
    def detect_clusters(
        self,
        df: pl.DataFrame,
        columns: list[str],
        n_clusters: Optional[int] = None,
        max_clusters: int = 10,
    ) -> ClusterPattern:
        """
        Detect natural clusters in numeric data.
        
        Uses K-means with automatic cluster number selection.
        """
        # Get numeric data
        data = df.select(columns).drop_nulls().to_numpy().astype(np.float64)
        
        if len(data) < 10:
            return ClusterPattern(
                columns=columns,
                n_clusters=1,
                cluster_sizes=[len(data)],
                cluster_centers=[list(np.mean(data, axis=0))],
                silhouette_score=0,
            )
        
        # Standardize
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)
        
        # Find optimal number of clusters if not specified
        if n_clusters is None:
            n_clusters = self._find_optimal_clusters(
                data_scaled, max_clusters
            )
        
        # Fit final model
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(data_scaled)
        
        # Calculate silhouette score
        from sklearn.metrics import silhouette_score
        if n_clusters > 1:
            try:
                silhouette = silhouette_score(data_scaled, labels)
            except Exception:
                silhouette = 0
        else:
            silhouette = 0
        
        # Get cluster sizes
        unique, counts = np.unique(labels, return_counts=True)
        cluster_sizes = [int(c) for c in counts]
        
        # Transform centers back to original scale
        centers_original = scaler.inverse_transform(kmeans.cluster_centers_)
        
        return ClusterPattern(
            columns=columns,
            n_clusters=n_clusters,
            cluster_sizes=cluster_sizes,
            cluster_centers=[list(center) for center in centers_original],
            silhouette_score=float(silhouette),
        )
    
    def _find_optimal_clusters(
        self,
        data: np.ndarray,
        max_clusters: int,
    ) -> int:
        """Find optimal number of clusters using elbow method."""
        from sklearn.metrics import silhouette_score
        
        max_k = min(max_clusters, len(data) - 1, 10)
        
        if max_k < 2:
            return 1
        
        best_k = 2
        best_score = -1
        
        for k in range(2, max_k + 1):
            try:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = kmeans.fit_predict(data)
                score = silhouette_score(data, labels)
                
                if score > best_score:
                    best_score = score
                    best_k = k
            except Exception:
                continue
        
        return best_k
    
    def detect_all_patterns(
        self,
        df: pl.DataFrame,
    ) -> dict[str, Any]:
        """
        Detect all patterns in the dataset.
        """
        patterns = {
            "majorities": [],
            "distributions": [],
            "frequencies": [],
        }
        
        for col in df.columns:
            dtype = df[col].dtype
            
            if dtype in (pl.Int8, pl.Int16, pl.Int32, pl.Int64,
                        pl.Float32, pl.Float64):
                # Numeric patterns
                dist = self.detect_distribution_pattern(df, col)
                patterns["distributions"].append(dist.to_dict())
            else:
                # Categorical patterns
                majority = self.detect_majority(df, col)
                freq = self.detect_frequency_pattern(df, col)
                
                patterns["majorities"].append(majority.to_dict())
                patterns["frequencies"].append(freq.to_dict())
        
        return patterns


# Global instance
pattern_recognizer = PatternRecognizer()
