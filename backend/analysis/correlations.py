"""
Correlation Analyzer

Multi-method correlation analysis including linear, non-linear,
and partial correlations with statistical significance testing.
"""

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import polars as pl
from scipy import stats as scipy_stats
from sklearn.feature_selection import mutual_info_regression

from config import get_settings


@dataclass
class CorrelationPair:
    """Correlation between two columns."""
    
    column1: str
    column2: str
    pearson: float
    pearson_pvalue: float
    spearman: float
    spearman_pvalue: float
    is_significant: bool
    strength: str  # weak, moderate, strong, very_strong
    direction: str  # positive, negative, none
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "column1": self.column1,
            "column2": self.column2,
            "pearson": round(self.pearson, 4),
            "pearson_pvalue": round(self.pearson_pvalue, 6),
            "spearman": round(self.spearman, 4),
            "spearman_pvalue": round(self.spearman_pvalue, 6),
            "is_significant": self.is_significant,
            "strength": self.strength,
            "direction": self.direction,
        }


@dataclass
class CorrelationMatrix:
    """Full correlation matrix result."""
    
    columns: list[str]
    pearson_matrix: list[list[float]]
    spearman_matrix: list[list[float]]
    significant_pairs: list[CorrelationPair]
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "columns": self.columns,
            "pearson_matrix": [[round(v, 4) for v in row] for row in self.pearson_matrix],
            "spearman_matrix": [[round(v, 4) for v in row] for row in self.spearman_matrix],
            "significant_pairs": [p.to_dict() for p in self.significant_pairs],
        }


@dataclass
class MutualInformationResult:
    """Mutual information between columns."""
    
    target_column: str
    features: list[str]
    mi_scores: list[float]
    normalized_scores: list[float]  # 0-1 scale
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "target_column": self.target_column,
            "features": self.features,
            "mi_scores": [round(s, 4) for s in self.mi_scores],
            "normalized_scores": [round(s, 4) for s in self.normalized_scores],
        }


class CorrelationAnalyzer:
    """Multi-method correlation analysis engine."""
    
    def __init__(self):
        self.settings = get_settings()
    
    def compute_correlation_pair(
        self,
        df: pl.DataFrame,
        col1: str,
        col2: str,
    ) -> CorrelationPair:
        """
        Compute correlation between two columns.
        
        Calculates both Pearson (linear) and Spearman (monotonic) correlations.
        """
        # Get aligned non-null data
        data = df.select([col1, col2]).drop_nulls()
        arr1 = data[col1].to_numpy().astype(np.float64)
        arr2 = data[col2].to_numpy().astype(np.float64)
        
        if len(arr1) < 3:
            return CorrelationPair(
                column1=col1, column2=col2,
                pearson=0, pearson_pvalue=1,
                spearman=0, spearman_pvalue=1,
                is_significant=False, strength="insufficient_data", direction="none"
            )
        
        # Pearson correlation (linear)
        try:
            pearson_r, pearson_p = scipy_stats.pearsonr(arr1, arr2)
        except Exception:
            pearson_r, pearson_p = 0, 1
        
        # Spearman correlation (monotonic, robust to outliers)
        try:
            spearman_r, spearman_p = scipy_stats.spearmanr(arr1, arr2)
        except Exception:
            spearman_r, spearman_p = 0, 1
        
        # Determine significance
        sig_level = self.settings.analysis.correlation_significance_level
        is_significant = (
            pearson_p < sig_level or spearman_p < sig_level
        )
        
        # Use the stronger correlation for classification
        stronger = pearson_r if abs(pearson_r) >= abs(spearman_r) else spearman_r
        
        # Determine strength
        abs_corr = abs(stronger)
        if abs_corr >= 0.8:
            strength = "very_strong"
        elif abs_corr >= 0.6:
            strength = "strong"
        elif abs_corr >= 0.4:
            strength = "moderate"
        elif abs_corr >= 0.2:
            strength = "weak"
        else:
            strength = "negligible"
        
        # Determine direction
        if not is_significant or abs_corr < self.settings.analysis.min_correlation_strength:
            direction = "none"
        elif stronger > 0:
            direction = "positive"
        else:
            direction = "negative"
        
        return CorrelationPair(
            column1=col1,
            column2=col2,
            pearson=float(pearson_r),
            pearson_pvalue=float(pearson_p),
            spearman=float(spearman_r),
            spearman_pvalue=float(spearman_p),
            is_significant=is_significant,
            strength=strength,
            direction=direction,
        )
    
    def compute_correlation_matrix(
        self,
        df: pl.DataFrame,
        columns: Optional[list[str]] = None,
    ) -> CorrelationMatrix:
        """
        Compute full correlation matrix for all numeric columns.
        
        Returns both Pearson and Spearman matrices plus significant pairs.
        """
        if columns is None:
            columns = [
                col for col in df.columns
                if df[col].dtype in (
                    pl.Int8, pl.Int16, pl.Int32, pl.Int64,
                    pl.Float32, pl.Float64,
                )
            ]
        
        n = len(columns)
        pearson_matrix = [[0.0] * n for _ in range(n)]
        spearman_matrix = [[0.0] * n for _ in range(n)]
        significant_pairs = []
        
        # Get numpy array for faster computation
        data = df.select(columns).drop_nulls().to_numpy().astype(np.float64)
        
        if len(data) < 3:
            return CorrelationMatrix(
                columns=columns,
                pearson_matrix=pearson_matrix,
                spearman_matrix=spearman_matrix,
                significant_pairs=[],
            )
        
        # Compute all pairwise correlations
        for i in range(n):
            for j in range(i, n):
                if i == j:
                    pearson_matrix[i][j] = 1.0
                    spearman_matrix[i][j] = 1.0
                else:
                    pair = self.compute_correlation_pair(df, columns[i], columns[j])
                    
                    pearson_matrix[i][j] = pair.pearson
                    pearson_matrix[j][i] = pair.pearson
                    spearman_matrix[i][j] = pair.spearman
                    spearman_matrix[j][i] = pair.spearman
                    
                    # Track significant correlations
                    if pair.is_significant and abs(pair.pearson) >= self.settings.analysis.min_correlation_strength:
                        significant_pairs.append(pair)
        
        # Sort by correlation strength
        significant_pairs.sort(key=lambda p: abs(p.pearson), reverse=True)
        
        return CorrelationMatrix(
            columns=columns,
            pearson_matrix=pearson_matrix,
            spearman_matrix=spearman_matrix,
            significant_pairs=significant_pairs,
        )
    
    def compute_kendall_tau(
        self,
        df: pl.DataFrame,
        col1: str,
        col2: str,
    ) -> dict[str, Any]:
        """
        Compute Kendall's tau correlation.
        
        Good for ordinal data and small samples.
        """
        data = df.select([col1, col2]).drop_nulls()
        arr1 = data[col1].to_numpy().astype(np.float64)
        arr2 = data[col2].to_numpy().astype(np.float64)
        
        if len(arr1) < 3:
            return {
                "column1": col1,
                "column2": col2,
                "tau": 0,
                "p_value": 1,
                "significant": False,
            }
        
        tau, p_value = scipy_stats.kendalltau(arr1, arr2)
        
        return {
            "column1": col1,
            "column2": col2,
            "tau": round(float(tau), 4),
            "p_value": round(float(p_value), 6),
            "significant": p_value < self.settings.analysis.correlation_significance_level,
        }
    
    def compute_mutual_information(
        self,
        df: pl.DataFrame,
        target_column: str,
        feature_columns: Optional[list[str]] = None,
    ) -> MutualInformationResult:
        """
        Compute mutual information between target and features.
        
        Captures non-linear relationships that correlations miss.
        """
        if feature_columns is None:
            feature_columns = [
                col for col in df.columns
                if col != target_column and df[col].dtype in (
                    pl.Int8, pl.Int16, pl.Int32, pl.Int64,
                    pl.Float32, pl.Float64,
                )
            ]
        
        if not feature_columns:
            return MutualInformationResult(
                target_column=target_column,
                features=[],
                mi_scores=[],
                normalized_scores=[],
            )
        
        # Prepare data
        all_cols = feature_columns + [target_column]
        data = df.select(all_cols).drop_nulls()
        
        X = data.select(feature_columns).to_numpy().astype(np.float64)
        y = data[target_column].to_numpy().astype(np.float64)
        
        if len(X) < 10:
            return MutualInformationResult(
                target_column=target_column,
                features=feature_columns,
                mi_scores=[0.0] * len(feature_columns),
                normalized_scores=[0.0] * len(feature_columns),
            )
        
        # Compute mutual information
        mi_scores = mutual_info_regression(X, y, random_state=42)
        
        # Normalize to 0-1 (divide by max)
        max_mi = max(mi_scores) if max(mi_scores) > 0 else 1
        normalized = [s / max_mi for s in mi_scores]
        
        return MutualInformationResult(
            target_column=target_column,
            features=feature_columns,
            mi_scores=[float(s) for s in mi_scores],
            normalized_scores=normalized,
        )
    
    def compute_partial_correlation(
        self,
        df: pl.DataFrame,
        col1: str,
        col2: str,
        control_cols: list[str],
    ) -> dict[str, Any]:
        """
        Compute partial correlation controlling for other variables.
        
        Shows correlation after removing the effect of control variables.
        """
        all_cols = [col1, col2] + control_cols
        data = df.select(all_cols).drop_nulls().to_numpy().astype(np.float64)
        
        if len(data) < len(all_cols) + 2:
            return {
                "column1": col1,
                "column2": col2,
                "control_columns": control_cols,
                "partial_correlation": 0,
                "p_value": 1,
                "significant": False,
            }
        
        # Compute partial correlation using regression residuals
        from sklearn.linear_model import LinearRegression
        
        X_control = data[:, 2:]  # Control variables
        
        # Residualize col1
        reg1 = LinearRegression().fit(X_control, data[:, 0])
        resid1 = data[:, 0] - reg1.predict(X_control)
        
        # Residualize col2
        reg2 = LinearRegression().fit(X_control, data[:, 1])
        resid2 = data[:, 1] - reg2.predict(X_control)
        
        # Correlate residuals
        r, p = scipy_stats.pearsonr(resid1, resid2)
        
        return {
            "column1": col1,
            "column2": col2,
            "control_columns": control_cols,
            "partial_correlation": round(float(r), 4),
            "p_value": round(float(p), 6),
            "significant": p < self.settings.analysis.correlation_significance_level,
        }
    
    def find_strongest_correlations(
        self,
        df: pl.DataFrame,
        top_n: int = 10,
        columns: Optional[list[str]] = None,
    ) -> list[CorrelationPair]:
        """
        Find the top N strongest correlations in the dataset.
        """
        matrix = self.compute_correlation_matrix(df, columns)
        return matrix.significant_pairs[:top_n]


# Global instance
correlation_analyzer = CorrelationAnalyzer()
