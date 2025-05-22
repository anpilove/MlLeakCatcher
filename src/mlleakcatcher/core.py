import logging
from typing import Union, Optional, List, Dict, Set
from .dataset import Dataset
import polars as pl
import pandas as pd
from pyspark.sql.dataframe import DataFrame
from .plot_utils import plot_top_features_by_pps, plot_top_features_train_test_difference
from .checks import (CheckResult, IdentifierTargetPPSCheck,FeatureTargetPPSCheck,FeatureTargetPPSChangeCheck)
from .log_utils import set_verbosity


logger = logging.getLogger("mlleakcatcher")

TYPING_DATAFRAME = Union[pd.DataFrame, pl.DataFrame, DataFrame]

class MlLeakCatcher:

    # Default checks
    CHECK_REGISTRY = {
        "identifier_target_pps": IdentifierTargetPPSCheck,
        "feature_target_pps": FeatureTargetPPSCheck,
        "feature_target_pps_change": FeatureTargetPPSChangeCheck,
    }

    DEFAULT_OPTIONS = {
        "identifier_target_pps": {
            "warning_threshold": 0.01,
            "model_type": "random_forest",
            "model_size": "medium",
            "n_samples": 10_000,
        },
        "feature_target_pps": {
            "deletion_threshold": 0.8,
            "warning_threshold": 0.5,
            "model_type": "random_forest",
            "model_size": "medium",
            "n_samples": 30_000,
        },
        "feature_target_pps_change": {
            "deletion_threshold": 0.8,
            "warning_threshold": 0.5,
            "warning_threshold_difference": 0.3,
            "model_type": "random_forest",
            "model_size": "medium",
            "n_samples": 30_000,
        },
        "model_target_check": {
            "deletion_threshold": 0.9,
            "warning_threshold": 0.7,
            "eval_metric": "auto",
            "n_folds": 3,
            "test_size": 0.2
        },
    }
    TASK_TYPE = ['classification', 'regression']


    def __init__(
        self,
        task_type: str,
        verbosity: int = 0,
        report_path: str = "mlleakcatcher",
        **kwargs
    ):

        self.task_type = task_type
        self.verbosity = verbosity
        self.report_path = report_path
        self.test_results = []
        set_verbosity(self.verbosity)

        self.options = {}
        for check_name, default_opts in self.DEFAULT_OPTIONS.items():
            if kwargs.get(check_name) is not False:  # Only include if not explicitly disabled
                check_opts = kwargs.get(check_name)
                if isinstance(check_opts, dict):
                    self.options[check_name] = {**default_opts, **check_opts}
                else:
                    self.options[check_name] = default_opts


    def run(
        self,
        train_data: TYPING_DATAFRAME,
        target_col: str,
        id_cols: Optional[List[str]] = None,
        test_data: Optional[TYPING_DATAFRAME] = None,
    ):


        train_dataset = Dataset(train_data, target_col, id_cols)
        test_dataset = None
        if test_data:
            test_dataset = Dataset(test_data, target_col, id_cols)


        self.test_results = []
        for check_name, check_options in self.options.items():
            check_class = self.CHECK_REGISTRY.get(check_name)
            if not test_data and check_name == "feature_target_pps_change":
                logger.warning(f"Test dataset is required for {check_name}")
                continue
            if check_class:
                check_instance = check_class(
                    task_type = self.task_type,
                    options = check_options,
                    report_path = self.report_path
                )
                result = check_instance.run(train_dataset, test_dataset)
                self.test_results.append(result)
            else:
                logger.warning(f"Check '{check_name}' not found in registry")


        deleted_features: Set[str] = set()
        for result in self.test_results:
            deleted_features.update(result.deleted_features)

        excluded_columns = [target_col, id_cols]
        remaining_features = [col for col in train_data.columns
                             if col not in excluded_columns and col not in deleted_features]

        return remaining_features, self.test_results