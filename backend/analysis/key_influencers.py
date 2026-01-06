"""
Key Influencers Analyzer

Power BI-style key influencers using Random Forest feature importance
and SHAP values for explainability.
"""

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import polars as pl
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from config import get_settings


@dataclass
class Influencer:
    """A single key influencer."""
    
    feature: str
    importance: float
    direction: str  # positive, negative, mixed
    impact_description: str
    shap_mean: Optional[float] = None
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "feature": self.feature,
            "importance": round(self.importance, 4),
            "direction": self.direction,
            "impact_description": self.impact_description,
            "shap_mean": round(self.shap_mean, 4) if self.shap_mean else None,
        }


@dataclass
class Segment:
    """A segment of data with shared characteristics."""
    
    segment_id: int
    size: int
    percentage: float
    rules: list[dict[str, Any]]
    target_mean: float
    overall_mean: float
    lift: float
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "segment_id": self.segment_id,
            "size": self.size,
            "percentage": round(self.percentage, 2),
            "rules": self.rules,
            "target_mean": round(self.target_mean, 4),
            "overall_mean": round(self.overall_mean, 4),
            "lift": round(self.lift, 4),
        }


@dataclass
class KeyInfluencersResult:
    """Complete key influencers analysis result."""
    
    target_column: str
    task_type: str  # regression or classification
    influencers: list[Influencer]
    model_score: float
    top_segments: list[Segment]
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "target_column": self.target_column,
            "task_type": self.task_type,
            "influencers": [i.to_dict() for i in self.influencers],
            "model_score": round(self.model_score, 4),
            "top_segments": [s.to_dict() for s in self.top_segments],
        }


class KeyInfluencersAnalyzer:
    """
    Key influencers analysis engine.
    
    Identifies which features most strongly influence a target variable,
    similar to Power BI's Key Influencers visual.
    """
    
    def __init__(self):
        self.settings = get_settings()
    
    def analyze(
        self,
        df: pl.DataFrame,
        target_column: str,
        feature_columns: Optional[list[str]] = None,
        top_n: Optional[int] = None,
    ) -> KeyInfluencersResult:
        """
        Perform key influencers analysis.
        
        Args:
            df: DataFrame with target and features
            target_column: Column to predict/explain
            feature_columns: Features to consider (auto-detected if None)
            top_n: Number of top influencers to return
            
        Returns:
            KeyInfluencersResult with ranked influencers and segments
        """
        if top_n is None:
            top_n = self.settings.analysis.feature_importance_top_n
        
        # Identify feature columns
        if feature_columns is None:
            feature_columns = [
                col for col in df.columns
                if col != target_column
            ]
        
        # Determine task type
        target_dtype = df[target_column].dtype
        unique_values = df[target_column].n_unique()
        
        if target_dtype in (pl.Utf8, pl.Categorical) or unique_values <= 10:
            task_type = "classification"
        else:
            task_type = "regression"
        
        # Prepare data
        X, y, feature_names, label_encoder = self._prepare_data(
            df, target_column, feature_columns
        )
        
        if X is None or len(X) < 10:
            return KeyInfluencersResult(
                target_column=target_column,
                task_type=task_type,
                influencers=[],
                model_score=0,
                top_segments=[],
            )
        
        # Train model and get importances
        influencers, model_score = self._compute_importances(
            X, y, feature_names, task_type, top_n
        )
        
        # Find top segments
        segments = self._find_segments(
            df, target_column, feature_columns[:5], task_type
        )
        
        return KeyInfluencersResult(
            target_column=target_column,
            task_type=task_type,
            influencers=influencers,
            model_score=model_score,
            top_segments=segments[:5],  # Top 5 segments
        )
    
    def _prepare_data(
        self,
        df: pl.DataFrame,
        target_column: str,
        feature_columns: list[str],
    ) -> tuple[Optional[np.ndarray], Optional[np.ndarray], list[str], Optional[LabelEncoder]]:
        """Prepare data for modeling."""
        
        # Filter to valid columns
        valid_features = []
        for col in feature_columns:
            if col in df.columns:
                dtype = df[col].dtype
                # Accept numeric and low-cardinality categorical
                if dtype in (pl.Int8, pl.Int16, pl.Int32, pl.Int64,
                           pl.Float32, pl.Float64):
                    valid_features.append(col)
                elif df[col].n_unique() < 50:  # Low cardinality
                    valid_features.append(col)
        
        if not valid_features:
            return None, None, [], None
        
        # Drop nulls
        all_cols = valid_features + [target_column]
        clean_df = df.select(all_cols).drop_nulls()
        
        if len(clean_df) < 10:
            return None, None, [], None
        
        # Encode features
        X_list = []
        feature_names = []
        
        for col in valid_features:
            dtype = clean_df[col].dtype
            
            if dtype in (pl.Int8, pl.Int16, pl.Int32, pl.Int64,
                        pl.Float32, pl.Float64):
                X_list.append(clean_df[col].to_numpy().reshape(-1, 1))
                feature_names.append(col)
            else:
                # One-hot encode categorical
                unique_vals = clean_df[col].unique().to_list()
                for val in unique_vals[:10]:  # Limit categories
                    X_list.append(
                        (clean_df[col] == val).to_numpy().astype(float).reshape(-1, 1)
                    )
                    feature_names.append(f"{col}={val}")
        
        X = np.hstack(X_list)
        
        # Prepare target
        target_dtype = clean_df[target_column].dtype
        label_encoder = None
        
        if target_dtype in (pl.Utf8, pl.Categorical):
            label_encoder = LabelEncoder()
            y = label_encoder.fit_transform(clean_df[target_column].to_list())
        else:
            y = clean_df[target_column].to_numpy().astype(np.float64)
        
        return X, y, feature_names, label_encoder
    
    def _compute_importances(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: list[str],
        task_type: str,
        top_n: int,
    ) -> tuple[list[Influencer], float]:
        """Compute feature importances using Random Forest."""
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        if task_type == "classification":
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1,
            )
        else:
            model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1,
            )
        
        model.fit(X_train_scaled, y_train)
        model_score = model.score(X_test_scaled, y_test)
        
        # Get feature importances (MDI)
        mdi_importance = model.feature_importances_
        
        # Get permutation importance (more reliable)
        perm_importance = permutation_importance(
            model, X_test_scaled, y_test,
            n_repeats=10, random_state=42, n_jobs=-1
        )
        
        # Combine importances (average of MDI and permutation)
        combined_importance = (
            mdi_importance + perm_importance.importances_mean
        ) / 2
        
        # Try to get SHAP values if available
        shap_values = self._compute_shap(model, X_test_scaled, feature_names)
        
        # Create influencers
        influencers = []
        indices = np.argsort(combined_importance)[::-1][:top_n]
        
        for idx in indices:
            if combined_importance[idx] < 0.01:  # Skip negligible
                continue
            
            feature = feature_names[idx]
            importance = combined_importance[idx]
            
            # Determine direction from correlation or SHAP
            if shap_values is not None and idx < len(shap_values):
                mean_shap = shap_values[idx]
                if mean_shap > 0:
                    direction = "positive"
                elif mean_shap < 0:
                    direction = "negative"
                else:
                    direction = "mixed"
            else:
                direction = "mixed"
                mean_shap = None
            
            # Generate description
            if "=" in feature:
                base_feature, value = feature.split("=", 1)
                description = f"When {base_feature} is '{value}', it influences the target"
            else:
                description = f"{feature} has {direction} impact on the target"
            
            influencers.append(Influencer(
                feature=feature,
                importance=float(importance),
                direction=direction,
                impact_description=description,
                shap_mean=mean_shap,
            ))
        
        return influencers, float(model_score)
    
    def _compute_shap(
        self,
        model,
        X: np.ndarray,
        feature_names: list[str],
    ) -> Optional[list[float]]:
        """Compute SHAP values if library is available."""
        try:
            import shap
            
            # Limit samples for speed
            max_samples = min(len(X), self.settings.analysis.shap_max_samples)
            X_sample = X[:max_samples]
            
            # Use TreeExplainer for Random Forest
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)
            
            # Handle classification (returns list of arrays per class)
            if isinstance(shap_values, list):
                # Use the positive class for binary, or average for multi-class
                if len(shap_values) == 2:
                    shap_values = shap_values[1]
                else:
                    shap_values = np.mean(
                        [np.abs(sv) for sv in shap_values], axis=0
                    )
            
            # Mean absolute SHAP value per feature
            mean_shap = np.mean(shap_values, axis=0)
            
            return [float(s) for s in mean_shap]
            
        except ImportError:
            return None
        except Exception:
            return None
    
    def _find_segments(
        self,
        df: pl.DataFrame,
        target_column: str,
        feature_columns: list[str],
        task_type: str,
    ) -> list[Segment]:
        """
        Find top segments using decision tree rules.
        
        Similar to Power BI's "Top Segments" view.
        """
        from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
        
        # Prepare simple numeric data
        numeric_features = [
            col for col in feature_columns
            if df[col].dtype in (pl.Int8, pl.Int16, pl.Int32, pl.Int64,
                                pl.Float32, pl.Float64)
        ][:5]  # Limit features
        
        if not numeric_features:
            return []
        
        all_cols = numeric_features + [target_column]
        clean_df = df.select(all_cols).drop_nulls()
        
        if len(clean_df) < 20:
            return []
        
        X = clean_df.select(numeric_features).to_numpy().astype(np.float64)
        
        if task_type == "classification":
            # Encode target
            label_encoder = LabelEncoder()
            y = label_encoder.fit_transform(clean_df[target_column].to_list())
            tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=20)
        else:
            y = clean_df[target_column].to_numpy().astype(np.float64)
            tree = DecisionTreeRegressor(max_depth=3, min_samples_leaf=20)
        
        tree.fit(X, y)
        
        # Extract leaf nodes as segments
        leaf_ids = tree.apply(X)
        unique_leaves = np.unique(leaf_ids)
        
        overall_mean = float(np.mean(y))
        segments = []
        
        for i, leaf_id in enumerate(unique_leaves[:5]):  # Top 5 segments
            mask = leaf_ids == leaf_id
            segment_size = int(np.sum(mask))
            segment_mean = float(np.mean(y[mask]))
            
            lift = segment_mean / overall_mean if overall_mean != 0 else 1
            
            # Get decision path (simplified)
            rules = self._extract_rules(tree, leaf_id, numeric_features)
            
            segments.append(Segment(
                segment_id=i + 1,
                size=segment_size,
                percentage=segment_size / len(y) * 100,
                rules=rules,
                target_mean=segment_mean,
                overall_mean=overall_mean,
                lift=lift,
            ))
        
        # Sort by absolute lift
        segments.sort(key=lambda s: abs(s.lift - 1), reverse=True)
        
        return segments
    
    def _extract_rules(
        self,
        tree,
        leaf_id: int,
        feature_names: list[str],
    ) -> list[dict[str, Any]]:
        """Extract decision rules for a leaf node."""
        rules = []
        
        # Get node path
        feature = tree.tree_.feature
        threshold = tree.tree_.threshold
        children_left = tree.tree_.children_left
        children_right = tree.tree_.children_right
        
        # Find path to leaf
        def find_path(node_id: int, path: list) -> Optional[list]:
            if node_id == leaf_id:
                return path
            
            if children_left[node_id] != -1:
                left_path = find_path(
                    children_left[node_id],
                    path + [(node_id, "<=", threshold[node_id], feature[node_id])]
                )
                if left_path:
                    return left_path
            
            if children_right[node_id] != -1:
                right_path = find_path(
                    children_right[node_id],
                    path + [(node_id, ">", threshold[node_id], feature[node_id])]
                )
                if right_path:
                    return right_path
            
            return None
        
        path = find_path(0, [])
        
        if path:
            for node_id, operator, thresh, feat_idx in path:
                if feat_idx >= 0 and feat_idx < len(feature_names):
                    rules.append({
                        "feature": feature_names[feat_idx],
                        "operator": operator,
                        "threshold": round(float(thresh), 4),
                    })
        
        return rules


# Global instance
key_influencers_analyzer = KeyInfluencersAnalyzer()
