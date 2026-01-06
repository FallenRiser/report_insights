"""
Data Profiler

Comprehensive data profiling with type detection and statistics.
Optimized for performance using Polars and NumPy.
"""

from typing import Any, Optional

import numpy as np
import polars as pl
from scipy import stats as scipy_stats

from api.schemas.responses import ColumnProfile, DataProfile


class DataProfiler:
    """High-performance data profiler."""
    
    # Type mappings from Polars to our categories
    NUMERIC_TYPES = {
        pl.Int8, pl.Int16, pl.Int32, pl.Int64,
        pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
        pl.Float32, pl.Float64,
    }
    DATETIME_TYPES = {pl.Date, pl.Datetime, pl.Time, pl.Duration}
    CATEGORICAL_TYPES = {pl.Categorical, pl.Enum}
    TEXT_TYPES = {pl.Utf8, pl.String}
    
    def profile(self, df: pl.DataFrame) -> DataProfile:
        """
        Generate comprehensive data profile.
        
        Args:
            df: Polars DataFrame
            
        Returns:
            DataProfile with column-level statistics
        """
        columns = []
        numeric_cols = []
        categorical_cols = []
        datetime_cols = []
        text_cols = []
        
        for col_name in df.columns:
            col = df[col_name]
            dtype = col.dtype
            
            # Determine column category
            if dtype in self.NUMERIC_TYPES:
                numeric_cols.append(col_name)
                profile = self._profile_numeric(col, col_name)
            elif dtype in self.DATETIME_TYPES:
                datetime_cols.append(col_name)
                profile = self._profile_datetime(col, col_name)
            elif dtype in self.CATEGORICAL_TYPES:
                categorical_cols.append(col_name)
                profile = self._profile_categorical(col, col_name)
            else:
                # Check if it's a low-cardinality string (categorical)
                unique_ratio = col.n_unique() / len(col) if len(col) > 0 else 0
                if unique_ratio < 0.05 and col.n_unique() < 100:
                    categorical_cols.append(col_name)
                    profile = self._profile_categorical(col, col_name)
                else:
                    text_cols.append(col_name)
                    profile = self._profile_text(col, col_name)
            
            columns.append(profile)
        
        # Calculate memory usage
        memory_mb = df.estimated_size("mb")
        
        return DataProfile(
            row_count=len(df),
            column_count=len(df.columns),
            memory_usage_mb=round(memory_mb, 2),
            columns=columns,
            numeric_columns=numeric_cols,
            categorical_columns=categorical_cols,
            datetime_columns=datetime_cols,
            text_columns=text_cols,
        )
    
    def _profile_numeric(self, col: pl.Series, name: str) -> ColumnProfile:
        """Profile numeric column with statistical measures."""
        count = len(col)
        null_count = col.null_count()
        
        # Get non-null values as numpy for scipy
        non_null = col.drop_nulls()
        
        if len(non_null) == 0:
            return ColumnProfile(
                name=name,
                dtype=str(col.dtype),
                count=count,
                null_count=null_count,
                null_percentage=100.0,
                unique_count=0,
            )
        
        # Convert to numpy for scipy operations
        arr = non_null.to_numpy()
        
        # Compute statistics
        try:
            skewness = float(scipy_stats.skew(arr, nan_policy="omit"))
            kurtosis = float(scipy_stats.kurtosis(arr, nan_policy="omit"))
        except Exception:
            skewness = None
            kurtosis = None
        
        # Polars quantiles
        try:
            q25 = float(non_null.quantile(0.25))
            q75 = float(non_null.quantile(0.75))
            median = float(non_null.median())
        except Exception:
            q25 = q75 = median = None
        
        return ColumnProfile(
            name=name,
            dtype=str(col.dtype),
            count=count,
            null_count=null_count,
            null_percentage=round(null_count / count * 100, 2) if count > 0 else 0,
            unique_count=col.n_unique(),
            mean=round(float(non_null.mean()), 4) if len(non_null) > 0 else None,
            std=round(float(non_null.std()), 4) if len(non_null) > 1 else None,
            min=float(non_null.min()),
            max=float(non_null.max()),
            median=round(median, 4) if median else None,
            q25=round(q25, 4) if q25 else None,
            q75=round(q75, 4) if q75 else None,
            skewness=round(skewness, 4) if skewness else None,
            kurtosis=round(kurtosis, 4) if kurtosis else None,
        )
    
    def _profile_categorical(self, col: pl.Series, name: str) -> ColumnProfile:
        """Profile categorical column with value counts."""
        count = len(col)
        null_count = col.null_count()
        
        # Get top values
        value_counts = (
            col.drop_nulls()
            .value_counts()
            .sort("count", descending=True)
            .head(10)
        )
        
        top_values = []
        for row in value_counts.iter_rows(named=True):
            value_col = [c for c in value_counts.columns if c != "count"][0]
            top_values.append({
                "value": row[value_col],
                "count": row["count"],
                "percentage": round(row["count"] / count * 100, 2) if count > 0 else 0
            })
        
        return ColumnProfile(
            name=name,
            dtype=str(col.dtype),
            count=count,
            null_count=null_count,
            null_percentage=round(null_count / count * 100, 2) if count > 0 else 0,
            unique_count=col.n_unique(),
            top_values=top_values,
        )
    
    def _profile_datetime(self, col: pl.Series, name: str) -> ColumnProfile:
        """Profile datetime column with date range."""
        count = len(col)
        null_count = col.null_count()
        non_null = col.drop_nulls()
        
        min_date = None
        max_date = None
        date_range_days = None
        
        if len(non_null) > 0:
            try:
                min_val = non_null.min()
                max_val = non_null.max()
                min_date = str(min_val)
                max_date = str(max_val)
                
                # Calculate range in days
                if hasattr(min_val, "days") or col.dtype == pl.Date:
                    diff = (max_val - min_val)
                    if hasattr(diff, "days"):
                        date_range_days = diff.days
                    else:
                        date_range_days = int(diff)
            except Exception:
                pass
        
        return ColumnProfile(
            name=name,
            dtype=str(col.dtype),
            count=count,
            null_count=null_count,
            null_percentage=round(null_count / count * 100, 2) if count > 0 else 0,
            unique_count=col.n_unique(),
            min_date=min_date,
            max_date=max_date,
            date_range_days=date_range_days,
        )
    
    def _profile_text(self, col: pl.Series, name: str) -> ColumnProfile:
        """Profile text column."""
        count = len(col)
        null_count = col.null_count()
        
        # For text columns, get sample values
        non_null = col.drop_nulls()
        sample_values = []
        
        if len(non_null) > 0:
            # Get a few sample values
            samples = non_null.head(5).to_list()
            sample_values = [{"value": str(v)[:100], "count": 1} for v in samples]
        
        return ColumnProfile(
            name=name,
            dtype=str(col.dtype),
            count=count,
            null_count=null_count,
            null_percentage=round(null_count / count * 100, 2) if count > 0 else 0,
            unique_count=col.n_unique(),
            top_values=sample_values,
        )
    
    def get_column_type(self, df: pl.DataFrame, column: str) -> str:
        """
        Get simplified column type.
        
        Args:
            df: Polars DataFrame
            column: Column name
            
        Returns:
            Type string: 'numeric', 'categorical', 'datetime', or 'text'
        """
        dtype = df[column].dtype
        
        if dtype in self.NUMERIC_TYPES:
            return "numeric"
        elif dtype in self.DATETIME_TYPES:
            return "datetime"
        elif dtype in self.CATEGORICAL_TYPES:
            return "categorical"
        else:
            # Check cardinality for string columns
            col = df[column]
            unique_ratio = col.n_unique() / len(col) if len(col) > 0 else 0
            if unique_ratio < 0.05 and col.n_unique() < 100:
                return "categorical"
            return "text"
    
    def get_numeric_columns(self, df: pl.DataFrame) -> list[str]:
        """Get list of numeric column names."""
        return [
            col for col in df.columns
            if df[col].dtype in self.NUMERIC_TYPES
        ]
    
    def get_categorical_columns(self, df: pl.DataFrame) -> list[str]:
        """Get list of categorical column names."""
        result = []
        for col in df.columns:
            dtype = df[col].dtype
            if dtype in self.CATEGORICAL_TYPES:
                result.append(col)
            elif dtype in self.TEXT_TYPES or dtype == pl.Utf8:
                unique_ratio = df[col].n_unique() / len(df) if len(df) > 0 else 0
                if unique_ratio < 0.05 and df[col].n_unique() < 100:
                    result.append(col)
        return result
    
    def get_datetime_columns(self, df: pl.DataFrame) -> list[str]:
        """Get list of datetime column names."""
        return [
            col for col in df.columns
            if df[col].dtype in self.DATETIME_TYPES
        ]


# Global profiler instance
data_profiler = DataProfiler()
