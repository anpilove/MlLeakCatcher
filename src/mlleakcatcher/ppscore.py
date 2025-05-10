import pandas as pd
import numpy as np
import logging
from typing import Dict, Optional, Literal, Union, List, Tuple
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import r2_score, accuracy_score
from catboost import CatBoostClassifier, CatBoostRegressor

logger = logging.getLogger(__name__)

class EnhancedPPS:
    """
    Enhanced Predictive Power Score (PPS) calculator with support for multiple model types.
    This class allows calculation of PPS for features using different machine learning models.
    """

    AVAILABLE_MODELS = {
        "classification": ["random_forest", "logistic_regression", "svm", "decision_tree", "catboost"],
        "regression": ["random_forest", "linear_regression", "svm", "decision_tree", "catboost"]
    }

    MODEL_COMPLEXITY = {
        "simple": ["logistic_regression", "linear_regression", "decision_tree"],
        "complex": ["random_forest", "catboost", "svm"]
    }

    MODEL_SIZES = {
        "small": {
            "n_estimators": 10,
            "max_depth": 3,
            "max_iter": 100,
            "iterations": 50,
            "learning_rate": 0.1
        },
        "medium": {
            "n_estimators": 50,
            "max_depth": 5,
            "max_iter": 200,
            "iterations": 100,
            "learning_rate": 0.05
        },
        "large": {
            "n_estimators": 100,
            "max_depth": 10,
            "max_iter": 500,
            "iterations": 200,
            "learning_rate": 0.03
        }
    }

    @staticmethod
    def calculate_pps(
        X: pd.DataFrame,
        y: pd.Series,
        task_type: Literal["classification", "regression"] = "classification",
        model_type: str = "random_forest",
        model_size: Literal["small", "medium", "large"] = "medium",
        n_samples: int = 10000,
        test_size: float = 0.3,
        random_state: int = 42
    ) -> Dict[str, float]:
        """
        Calculate custom PPS (Predictive Power Score) for all features against target using specified model.

        Args:
            X (pd.DataFrame): Feature dataframe
            y (pd.Series): Target series
            task_type (str): Task type ("classification" or "regression")
            model_type (str): Type of model to use (e.g., "random_forest", "logistic_regression")
            model_size (str): Size of model to use ("small", "medium", "large")
            n_samples (int): Number of samples to use for calculation (for large datasets)
            test_size (float): Proportion of data to use for testing
            random_state (int): Random seed for reproducibility

        Returns:
            Dict[str, float]: Dictionary mapping feature names to PPS scores
        """
        results = {}
        is_classification = task_type.lower() == "classification"

        valid_models = EnhancedPPS.AVAILABLE_MODELS["classification" if is_classification else "regression"]
        if model_type not in valid_models:
            logger.warning(f"Unknown model type: {model_type}, using random_forest")
            model_type = "random_forest"

        if model_size not in EnhancedPPS.MODEL_SIZES:
            logger.warning(f"Unknown model size: {model_size}, using medium")
            model_size = "medium"

        model_params = EnhancedPPS.MODEL_SIZES[model_size]

        if len(X) > n_samples:
            sample_indices = np.random.choice(len(X), n_samples, replace=False)
            X_sample = X.iloc[sample_indices]
            y_sample = y.iloc[sample_indices]
        else:
            X_sample = X
            y_sample = y

        if is_classification:
            if not pd.api.types.is_numeric_dtype(y_sample):
                label_encoder = LabelEncoder()
                y_sample = label_encoder.fit_transform(y_sample)

        if is_classification:
            baseline_score = max(np.bincount(y_sample)) / len(y_sample)
        else:
            y_mean = np.mean(y_sample)
            baseline_score = np.mean((y_sample - y_mean) ** 2)  # MSE

        for col in X.columns:
            try:
                X_col = X_sample[[col]]

                if X_col[col].isna().mean() > 0.5:
                    results[col] = 0.0
                    continue

                if not pd.api.types.is_numeric_dtype(X_col[col]):
                    try:
                        X_col = pd.get_dummies(X_col, columns=[col], drop_first=True)
                    except Exception:
                        le = LabelEncoder()
                        X_col[col] = le.fit_transform(X_col[col].astype(str))

                for feature in X_col.columns:
                    if X_col[feature].isna().any():
                        if pd.api.types.is_numeric_dtype(X_col[feature]):
                            X_col[feature] = X_col[feature].fillna(X_col[feature].mean())
                        else:
                            X_col[feature] = X_col[feature].fillna(X_col[feature].mode()[0])

                model = EnhancedPPS._get_model(
                    model_type=model_type,
                    task_type=task_type,
                    params=model_params,
                    random_state=random_state
                )

                if model is None:
                    logger.error(f"Failed to create model for {model_type}/{task_type}")
                    results[col] = 0.0
                    continue

                X_train, X_test, y_train, y_test = train_test_split(
                    X_col, y_sample, test_size=test_size, random_state=random_state
                )

                model.fit(X_train, y_train)

                if is_classification:
                    # For classification, use accuracy
                    y_pred = model.predict(X_test)
                    model_score = accuracy_score(y_test, y_pred)
                    pps_score = (model_score - baseline_score) / (1 - baseline_score)
                else:
                    pps_score = model.score(X_test, y_test)

                pps_score = max(0, min(1, pps_score))
                results[col] = pps_score

            except Exception as e:
                logger.warning(f"Error calculating PPS for {col}: {str(e)}")
                results[col] = 0.0

        return results

    @staticmethod
    def _get_model(
        model_type: str,
        task_type: str,
        params: dict,
        random_state: int
    ) -> Optional[object]:
        """
        Get the appropriate model instance based on model type and task type.

        Args:
            model_type (str): Type of model
            task_type (str): Type of task (classification or regression)
            params (dict): Model parameters
            random_state (int): Random state for reproducibility

        Returns:
            Model instance or None if invalid configuration
        """
        is_classification = task_type.lower() == "classification"

        # Random Forest
        if model_type == "random_forest":
            if is_classification:
                return RandomForestClassifier(
                    n_estimators=params["n_estimators"],
                    max_depth=params["max_depth"],
                    random_state=random_state
                )
            else:
                return RandomForestRegressor(
                    n_estimators=params["n_estimators"],
                    max_depth=params["max_depth"],
                    random_state=random_state
                )

        # Linear models
        elif model_type in ["logistic_regression", "linear_regression"]:
            if is_classification:
                return LogisticRegression(
                    max_iter=params["max_iter"],
                    random_state=random_state
                )
            else:
                return LinearRegression()

        # SVM
        elif model_type == "svm":
            if is_classification:
                return SVC(
                    random_state=random_state
                )
            else:
                return SVR()

        # Decision Trees
        elif model_type == "decision_tree":
            if is_classification:
                return DecisionTreeClassifier(
                    max_depth=params["max_depth"],
                    random_state=random_state
                )
            else:
                return DecisionTreeRegressor(
                    max_depth=params["max_depth"],
                    random_state=random_state
                )

        # CatBoost
        elif model_type == "catboost":
            if is_classification:
                return CatBoostClassifier(
                    iterations=params["iterations"],
                    depth=params["max_depth"],
                    learning_rate=params["learning_rate"],
                    random_seed=random_state,
                    verbose=False
                )
            else:
                return CatBoostRegressor(
                    iterations=params["iterations"],
                    depth=params["max_depth"],
                    learning_rate=params["learning_rate"],
                    random_seed=random_state,
                    verbose=False
                )

        return None

    @staticmethod
    def compare_models(
        X: pd.DataFrame,
        y: pd.Series,
        task_type: Literal["classification", "regression"] = "classification",
        model_size: Literal["small", "medium", "large"] = "medium",
        n_samples: int = 10000,
        test_size: float = 0.3,
        random_state: int = 42,
        models_to_use: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare PPS scores for all features using different models.

        Args:
            X (pd.DataFrame): Feature dataframe
            y (pd.Series): Target series
            task_type (str): Task type ("classification" or "regression")
            model_size (str): Size of models to use ("small", "medium", "large")
            n_samples (int): Number of samples to use
            test_size (float): Test size proportion
            random_state (int): Random seed
            models_to_use (List[str], optional): Specific models to use. If None, all available models are used.

        Returns:
            Dict[str, Dict[str, float]]: Dictionary mapping model types to feature PPS scores
        """
        results = {}

        available_models = EnhancedPPS.AVAILABLE_MODELS[task_type]

        if models_to_use:
            models = [m for m in models_to_use if m in available_models]
            if not models:
                logger.warning(f"None of the specified models are available for {task_type}. Using all available models.")
                models = available_models
        else:
            models = available_models

        for model_type in models:
            try:
                model_results = EnhancedPPS.calculate_pps(
                    X=X,
                    y=y,
                    task_type=task_type,
                    model_type=model_type,
                    model_size=model_size,
                    n_samples=n_samples,
                    test_size=test_size,
                    random_state=random_state
                )
                results[model_type] = model_results
            except Exception as e:
                logger.error(f"Error calculating PPS for model {model_type}: {str(e)}")
                results[model_type] = {col: 0.0 for col in X.columns}

        return results

    @staticmethod
    def compare_simple_vs_complex(
        X: pd.DataFrame,
        y: pd.Series,
        task_type: Literal["classification", "regression"] = "classification",
        model_size: Literal["small", "medium", "large"] = "medium",
        n_samples: int = 10000,
        test_size: float = 0.3,
        random_state: int = 42,
        simple_model: Optional[str] = None,
        complex_model: Optional[str] = None
    ) -> Dict[str, Dict[str, Dict[str, float]]]:
        """
        Compare PPS scores between simple and complex models.

        Args:
            X (pd.DataFrame): Feature dataframe
            y (pd.Series): Target series
            task_type (str): Task type ("classification" or "regression")
            model_size (str): Size of models to use ("small", "medium", "large")
            n_samples (int): Number of samples to use
            test_size (float): Test size proportion
            random_state (int): Random seed
            simple_model (str, optional): Specific simple model to use. If None, the first available simple model is used.
            complex_model (str, optional): Specific complex model to use. If None, the first available complex model is used.

        Returns:
            Dict: Dictionary with 'simple', 'complex', and 'difference' PPS scores for each feature
        """
        available_models = EnhancedPPS.AVAILABLE_MODELS[task_type]

        if simple_model is not None and simple_model in available_models and simple_model in EnhancedPPS.MODEL_COMPLEXITY["simple"]:
            selected_simple_model = simple_model
        else:
            simple_models = [m for m in EnhancedPPS.MODEL_COMPLEXITY["simple"] if m in available_models]
            if not simple_models:
                raise ValueError(f"No simple models available for {task_type}")
            selected_simple_model = simple_models[0]

        if complex_model is not None and complex_model in available_models and complex_model in EnhancedPPS.MODEL_COMPLEXITY["complex"]:
            selected_complex_model = complex_model
        else:
            complex_models = [m for m in EnhancedPPS.MODEL_COMPLEXITY["complex"] if m in available_models]
            if not complex_models:
                raise ValueError(f"No complex models available for {task_type}")
            selected_complex_model = complex_models[0]

        simple_results = EnhancedPPS.calculate_pps(
            X=X,
            y=y,
            task_type=task_type,
            model_type=selected_simple_model,
            model_size=model_size,
            n_samples=n_samples,
            test_size=test_size,
            random_state=random_state
        )

        complex_results = EnhancedPPS.calculate_pps(
            X=X,
            y=y,
            task_type=task_type,
            model_type=selected_complex_model,
            model_size=model_size,
            n_samples=n_samples,
            test_size=test_size,
            random_state=random_state
        )

        difference_results = {}
        for feature in X.columns:
            simple_score = simple_results.get(feature, 0.0)
            complex_score = complex_results.get(feature, 0.0)
            difference_results[feature] = complex_score - simple_score

        return {
            "simple": {selected_simple_model: simple_results},
            "complex": {selected_complex_model: complex_results},
            "difference": difference_results
        }

    @staticmethod
    def get_best_features(
        pps_results: Dict[str, float],
        threshold: float = 0.1,
        top_n: Optional[int] = None
    ) -> List[str]:
        """
        Get best features based on PPS scores.

        Args:
            pps_results (Dict[str, float]): PPS scores by feature
            threshold (float): Minimum PPS score to consider
            top_n (int, optional): Return only top N features

        Returns:
            List[str]: List of best feature names
        """
        filtered_features = [(feat, score) for feat, score in pps_results.items() if score >= threshold]

        sorted_features = sorted(filtered_features, key=lambda x: x[1], reverse=True)

        if top_n is not None and top_n > 0:
            sorted_features = sorted_features[:top_n]

        return [feat for feat, _ in sorted_features]

    @staticmethod
    def get_feature_importance_report(
        comparison_results: Dict[str, Dict[str, Dict[str, float]]],
        threshold: float = 0.05,
    ) -> Dict[str, List[Dict[str, Union[str, float]]]]:
        """
        Generate a comprehensive feature importance report comparing simple and complex models.

        Args:
            comparison_results: Output from compare_simple_vs_complex method
            threshold: Minimum improvement threshold to consider a feature as "complex-dependent"

        Returns:
            Dict containing categorized features by their behavior with different model complexities
        """
        if not all(k in comparison_results for k in ["simple", "complex", "difference"]):
            raise ValueError("Invalid comparison results format")

        simple_model = list(comparison_results["simple"].keys())[0]
        complex_model = list(comparison_results["complex"].keys())[0]

        simple_results = comparison_results["simple"][simple_model]
        complex_results = comparison_results["complex"][complex_model]
        diff_results = comparison_results["difference"]

        basic_features = []
        complex_features = []
        strong_features = []
        weak_features = []

        for feature, diff_score in diff_results.items():
            simple_score = simple_results.get(feature, 0.0)
            complex_score = complex_results.get(feature, 0.0)

            feature_info = {
                "feature": feature,
                "simple_score": simple_score,
                "complex_score": complex_score,
                "improvement": diff_score
            }

            if simple_score > 0.3 and complex_score > 0.3:
                strong_features.append(feature_info)
            elif simple_score < 0.1 and complex_score < 0.1:
                weak_features.append(feature_info)
            elif diff_score > threshold:
                complex_features.append(feature_info)
            else:
                basic_features.append(feature_info)

        strong_features = sorted(strong_features, key=lambda x: x["simple_score"] + x["complex_score"], reverse=True)
        weak_features = sorted(weak_features, key=lambda x: x["complex_score"])
        complex_features = sorted(complex_features, key=lambda x: x["improvement"], reverse=True)
        basic_features = sorted(basic_features, key=lambda x: x["simple_score"], reverse=True)

        return {
            "strong_features": strong_features,
            "complex_dependent_features": complex_features,
            "basic_features": basic_features,
            "weak_features": weak_features
        }
