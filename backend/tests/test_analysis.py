"""
Test Analysis Engines

Unit tests for statistical, trend, and outlier analysis.
"""

import numpy as np
import polars as pl
import pytest

from analysis.statistical import StatisticalAnalyzer
from analysis.trends import TrendDetector
from analysis.outliers import OutlierDetector
from analysis.correlations import CorrelationAnalyzer


@pytest.fixture
def numeric_df():
    """Create a sample DataFrame with numeric data."""
    np.random.seed(42)
    return pl.DataFrame({
        "normal": np.random.normal(100, 15, 1000),
        "trending_up": np.arange(1000) * 0.5 + np.random.normal(0, 5, 1000),
        "trending_down": -np.arange(1000) * 0.3 + 500 + np.random.normal(0, 3, 1000),
        "with_outliers": np.concatenate([
            np.random.normal(50, 5, 980),
            np.array([200, 250, 300, -50, -100, 500, 600, 700, 800, 1000])  # outliers
        ]),
        "correlated": np.random.normal(100, 15, 1000) + np.arange(1000) * 0.1,
    })


@pytest.fixture
def analyzer():
    return StatisticalAnalyzer()


@pytest.fixture
def trend_detector():
    return TrendDetector()


@pytest.fixture
def outlier_detector():
    return OutlierDetector()


class TestStatisticalAnalyzer:
    def test_descriptive_stats(self, analyzer, numeric_df):
        """Test descriptive statistics computation."""
        stats = analyzer.compute_descriptive_stats(numeric_df, "normal")
        
        assert stats.count == 1000
        assert 95 < stats.mean < 105  # Should be around 100
        assert 13 < stats.std < 17  # Should be around 15
        assert stats.min < stats.max
        assert stats.q25 < stats.median < stats.q75
    
    def test_distribution_detection_normal(self, analyzer, numeric_df):
        """Test normal distribution detection."""
        dist = analyzer.analyze_distribution(numeric_df, "normal")
        
        # Should detect as approximately normal
        assert dist.distribution_type in ("normal", "symmetric")
        assert dist.normality_pvalue > 0.01  # Likely normal
    
    def test_distribution_detection_skewed(self, analyzer):
        """Test skewed distribution detection."""
        np.random.seed(42)
        skewed_df = pl.DataFrame({
            "skewed": np.random.exponential(10, 1000)
        })
        
        dist = analyzer.analyze_distribution(skewed_df, "skewed")
        
        assert dist.distribution_type in ("skewed_right", "exponential", "unknown")
    
    def test_confidence_interval(self, analyzer, numeric_df):
        """Test confidence interval computation."""
        lower, upper = analyzer.compute_confidence_interval(
            numeric_df, "normal", confidence=0.95
        )
        
        assert lower < upper
        assert 95 < lower < 105  # Should contain true mean of 100
        assert 95 < upper < 105


class TestTrendDetector:
    def test_detect_increasing_trend(self, trend_detector, numeric_df):
        """Test detection of increasing trend."""
        result = trend_detector.detect_linear_trend(numeric_df, "trending_up")
        
        assert result.direction == "increasing"
        assert result.slope > 0
        assert result.is_significant
        assert result.r_squared > 0.5
    
    def test_detect_decreasing_trend(self, trend_detector, numeric_df):
        """Test detection of decreasing trend."""
        result = trend_detector.detect_linear_trend(numeric_df, "trending_down")
        
        assert result.direction == "decreasing"
        assert result.slope < 0
        assert result.is_significant
    
    def test_no_trend(self, trend_detector, numeric_df):
        """Test no trend in random data."""
        result = trend_detector.detect_linear_trend(numeric_df, "normal")
        
        # Random normal data should not have significant trend
        # (though it might by chance)
        assert result.r_squared < 0.2  # Weak relationship
    
    def test_mann_kendall(self, trend_detector, numeric_df):
        """Test Mann-Kendall trend test."""
        result = trend_detector.mann_kendall_test(numeric_df, "trending_up")
        
        assert result["trend"] == "increasing"
        assert result["significant"]
        assert result["z_score"] > 0
    
    def test_moving_averages(self, trend_detector, numeric_df):
        """Test moving average computation."""
        result = trend_detector.compute_moving_averages(
            numeric_df, "trending_up", window=10
        )
        
        assert len(result.sma) == 1000
        assert len(result.ema) == 1000
        assert result.trend_direction == "increasing"


class TestOutlierDetector:
    def test_iqr_detection(self, outlier_detector, numeric_df):
        """Test IQR outlier detection."""
        result = outlier_detector.detect_iqr(numeric_df, "with_outliers")
        
        assert result.outlier_count > 0
        assert result.outlier_percentage > 0
        assert result.lower_bound < result.upper_bound
        assert len(result.outlier_indices) == result.outlier_count
    
    def test_zscore_detection(self, outlier_detector, numeric_df):
        """Test Z-score outlier detection."""
        result = outlier_detector.detect_zscore(numeric_df, "with_outliers")
        
        assert result.outlier_count > 0
        assert result.method in ("zscore", "modified_zscore")
    
    def test_consensus_outliers(self, outlier_detector, numeric_df):
        """Test consensus outlier detection."""
        result = outlier_detector.get_consensus_outliers(
            numeric_df, "with_outliers", min_agreement=2
        )
        
        # Extreme outliers should be detected by multiple methods
        assert result.outlier_count > 0
        assert "consensus" in result.method
    
    def test_isolation_forest(self, outlier_detector, numeric_df):
        """Test multivariate outlier detection."""
        result = outlier_detector.detect_isolation_forest(
            numeric_df, 
            ["normal", "trending_up", "correlated"],
            contamination=0.05
        )
        
        assert result.method == "isolation_forest"
        assert 0 <= result.outlier_percentage <= 100


class TestCorrelationAnalyzer:
    def test_correlation_pair(self):
        """Test correlation between two columns."""
        analyzer = CorrelationAnalyzer()
        
        # Create perfectly correlated data
        df = pl.DataFrame({
            "a": list(range(100)),
            "b": list(range(100)),  # Perfect correlation
            "c": list(range(99, -1, -1)),  # Perfect negative correlation
        })
        
        # Test positive correlation
        result = analyzer.compute_correlation_pair(df, "a", "b")
        assert result.pearson > 0.99
        assert result.direction == "positive"
        assert result.strength in ("very_strong", "strong")
        
        # Test negative correlation  
        result = analyzer.compute_correlation_pair(df, "a", "c")
        assert result.pearson < -0.99
        assert result.direction == "negative"
    
    def test_correlation_matrix(self, numeric_df):
        """Test full correlation matrix."""
        analyzer = CorrelationAnalyzer()
        
        matrix = analyzer.compute_correlation_matrix(
            numeric_df,
            ["normal", "trending_up", "correlated"]
        )
        
        assert len(matrix.columns) == 3
        assert len(matrix.pearson_matrix) == 3
        assert len(matrix.pearson_matrix[0]) == 3
        
        # Diagonal should be 1
        for i in range(3):
            assert matrix.pearson_matrix[i][i] == 1.0
