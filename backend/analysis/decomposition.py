"""
Decomposition Engine

Hierarchical measure decomposition across dimensions.
Similar to Power BI's Decomposition Tree visual.
"""

from dataclasses import dataclass, field
from typing import Any, Optional
from uuid import uuid4

import numpy as np
import polars as pl

from config import get_settings


@dataclass
class DecompNode:
    """A node in the decomposition tree."""
    
    id: str
    label: str
    value: float
    percentage: float
    count: int
    parent_id: Optional[str]
    depth: int
    dimension: Optional[str] = None
    dimension_value: Optional[Any] = None
    children: list["DecompNode"] = field(default_factory=list)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "value": round(self.value, 4),
            "percentage": round(self.percentage, 2),
            "count": self.count,
            "parent_id": self.parent_id,
            "depth": self.depth,
            "dimension": self.dimension,
            "dimension_value": self.dimension_value,
            "children": [c.to_dict() for c in self.children],
        }


@dataclass
class DecompositionResult:
    """Result of decomposition analysis."""
    
    measure_column: str
    aggregation: str
    total_value: float
    total_count: int
    root: DecompNode
    suggested_dimensions: list[dict[str, Any]]
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "measure_column": self.measure_column,
            "aggregation": self.aggregation,
            "total_value": round(self.total_value, 4),
            "total_count": self.total_count,
            "root": self.root.to_dict(),
            "suggested_dimensions": self.suggested_dimensions,
        }


class DecompositionEngine:
    """
    Decomposition tree generator.
    
    Breaks down a measure across multiple dimensions,
    identifying the most significant contributors.
    """
    
    AGGREGATIONS = {
        "sum": lambda x: x.sum(),
        "mean": lambda x: x.mean(),
        "count": lambda x: x.count(),
        "min": lambda x: x.min(),
        "max": lambda x: x.max(),
    }
    
    def __init__(self):
        self.settings = get_settings()
    
    def decompose(
        self,
        df: pl.DataFrame,
        measure_column: str,
        dimension_columns: list[str],
        aggregation: str = "sum",
        max_depth: int = 5,
        min_percentage: float = 1.0,
        max_children: int = 10,
    ) -> DecompositionResult:
        """
        Decompose a measure across dimensions.
        
        Args:
            df: DataFrame
            measure_column: Numeric column to decompose
            dimension_columns: Categorical columns for decomposition
            aggregation: Aggregation function (sum, mean, count, min, max)
            max_depth: Maximum tree depth
            min_percentage: Minimum percentage to include a node
            max_children: Maximum children per node
            
        Returns:
            DecompositionResult with tree structure
        """
        if aggregation not in self.AGGREGATIONS:
            aggregation = "sum"
        
        # Calculate total
        agg_func = self.AGGREGATIONS[aggregation]
        total_value = float(agg_func(df[measure_column]))
        total_count = len(df)
        
        # Create root node
        root = DecompNode(
            id=str(uuid4())[:8],
            label="Total",
            value=total_value,
            percentage=100.0,
            count=total_count,
            parent_id=None,
            depth=0,
        )
        
        # Get suggested dimensions (ordered by variance explained)
        suggested = self._suggest_dimensions(
            df, measure_column, dimension_columns, aggregation
        )
        
        # Build tree recursively
        if dimension_columns:
            self._build_tree(
                df=df,
                node=root,
                measure_column=measure_column,
                remaining_dimensions=dimension_columns.copy(),
                aggregation=aggregation,
                total_value=total_value,
                max_depth=max_depth,
                min_percentage=min_percentage,
                max_children=max_children,
            )
        
        return DecompositionResult(
            measure_column=measure_column,
            aggregation=aggregation,
            total_value=total_value,
            total_count=total_count,
            root=root,
            suggested_dimensions=suggested,
        )
    
    def _build_tree(
        self,
        df: pl.DataFrame,
        node: DecompNode,
        measure_column: str,
        remaining_dimensions: list[str],
        aggregation: str,
        total_value: float,
        max_depth: int,
        min_percentage: float,
        max_children: int,
    ) -> None:
        """Recursively build decomposition tree."""
        
        if node.depth >= max_depth or not remaining_dimensions:
            return
        
        # Use the first remaining dimension
        dim = remaining_dimensions[0]
        rest_dims = remaining_dimensions[1:]
        
        if dim not in df.columns:
            return
        
        agg_func = self.AGGREGATIONS[aggregation]
        
        # Group by dimension
        grouped = (
            df.group_by(dim)
            .agg([
                agg_func(pl.col(measure_column)).alias("value"),
                pl.count().alias("count"),
            ])
            .sort("value", descending=True)
        )
        
        # Create child nodes
        children = []
        for row in grouped.iter_rows(named=True):
            dim_value = row[dim]
            value = row["value"]
            count = row["count"]
            
            if value is None:
                continue
            
            percentage = abs(value / total_value * 100) if total_value != 0 else 0
            
            if percentage < min_percentage:
                continue
            
            child = DecompNode(
                id=str(uuid4())[:8],
                label=f"{dim}={dim_value}",
                value=float(value),
                percentage=percentage,
                count=int(count),
                parent_id=node.id,
                depth=node.depth + 1,
                dimension=dim,
                dimension_value=dim_value,
            )
            
            # Filter data for this branch
            child_df = df.filter(pl.col(dim) == dim_value)
            
            # Recurse
            if rest_dims and child.depth < max_depth:
                self._build_tree(
                    df=child_df,
                    node=child,
                    measure_column=measure_column,
                    remaining_dimensions=rest_dims,
                    aggregation=aggregation,
                    total_value=total_value,
                    max_depth=max_depth,
                    min_percentage=min_percentage,
                    max_children=max_children,
                )
            
            children.append(child)
            
            if len(children) >= max_children:
                break
        
        node.children = children
    
    def _suggest_dimensions(
        self,
        df: pl.DataFrame,
        measure_column: str,
        dimension_columns: list[str],
        aggregation: str,
    ) -> list[dict[str, Any]]:
        """
        Suggest best dimensions for decomposition.
        
        Ranks dimensions by variance explained (similar to AI splits in Power BI).
        """
        suggestions = []
        agg_func = self.AGGREGATIONS[aggregation]
        
        total_value = float(agg_func(df[measure_column]))
        
        for dim in dimension_columns:
            if dim not in df.columns:
                continue
            
            n_unique = df[dim].n_unique()
            
            # Skip high cardinality
            if n_unique > 100 or n_unique < 2:
                continue
            
            # Calculate variance explained using between-group variance
            grouped = (
                df.group_by(dim)
                .agg([
                    agg_func(pl.col(measure_column)).alias("group_value"),
                    pl.count().alias("count"),
                ])
            )
            
            group_values = grouped["group_value"].to_numpy()
            group_counts = grouped["count"].to_numpy()
            
            # Calculate weighted variance
            if len(group_values) > 1:
                overall_mean = np.average(group_values, weights=group_counts)
                between_var = np.average(
                    (group_values - overall_mean) ** 2,
                    weights=group_counts
                )
                
                # Normalize by total variance
                total_var = np.var(df[measure_column].to_numpy())
                variance_explained = between_var / total_var if total_var > 0 else 0
            else:
                variance_explained = 0
            
            suggestions.append({
                "dimension": dim,
                "unique_values": n_unique,
                "variance_explained": round(float(variance_explained), 4),
                "suggested_order": 1 if variance_explained > 0.3 else (2 if variance_explained > 0.1 else 3),
            })
        
        # Sort by variance explained
        suggestions.sort(key=lambda x: x["variance_explained"], reverse=True)
        
        return suggestions[:10]  # Top 10 suggestions
    
    def auto_decompose(
        self,
        df: pl.DataFrame,
        measure_column: str,
        aggregation: str = "sum",
        max_dimensions: int = 3,
    ) -> DecompositionResult:
        """
        Automatically decompose using AI-suggested dimension order.
        
        Selects the best dimensions based on variance explained.
        """
        # Find categorical columns
        categorical_columns = [
            col for col in df.columns
            if col != measure_column and (
                df[col].dtype in (pl.Utf8, pl.Categorical) or
                (df[col].n_unique() < 50 and df[col].n_unique() >= 2)
            )
        ]
        
        # Get suggestions
        suggestions = self._suggest_dimensions(
            df, measure_column, categorical_columns, aggregation
        )
        
        # Use top dimensions
        best_dims = [s["dimension"] for s in suggestions[:max_dimensions]]
        
        return self.decompose(
            df=df,
            measure_column=measure_column,
            dimension_columns=best_dims,
            aggregation=aggregation,
        )


# Global instance
decomposition_engine = DecompositionEngine()
