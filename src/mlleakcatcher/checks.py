import logging
from abc import ABC, abstractmethod
from typing import Union, Optional, List, Dict
from dataclasses import dataclass, field
from .dataset import Dataset
import polars as pl
import pandas as pd
from pyspark.sql.dataframe import DataFrame
from .ppscore import EnhancedPPS



logger = logging.getLogger("mlleakcatcher")

TYPING_DATAFRAME = Union[pd.DataFrame, pl.DataFrame, DataFrame]


@dataclass
class CheckResult:
    """
    Structure for storing feature check results and deletion reasons.

    Attributes:
        name (str): Name of the test
        deleted_features (List[str]): List of features that were removed
        warning_features (List[str]): List of features that raised warnings
        result_df (pd.DataFrame): DataFrame containing detailed check results
        deletion_reasons (Dict[str, str]): Mapping of features to their deletion reasons
    """

    name: Optional[str] = None
    deleted_features: List[str] = field(default_factory=list)
    warning_features: List[str] = field(default_factory=list)
    result_df: Optional[pd.DataFrame] = None
    deletion_reasons: Dict[str, str] = field(default_factory=dict)

    def __repr__(self):
        df_preview = (
            "\n".join(
                "  " + line
                for line in self.result_df.head(3).to_string(index=False).split("\n")
            )
            if self.result_df is not None
            else "  None"
        )
        reasons = (
            "\n".join(f"  {k}: {v}" for k, v in self.deletion_reasons.items())
            if self.deletion_reasons
            else "  None"
        )

        return (
            f"\n"
            f"CheckResult:\n"
            f"  Name: {self.name}\n"
            f"  Deleted Features: {', '.join(self.deleted_features) or 'None'}\n"
            f"  Warning Features: {', '.join(self.warning_features) or 'None'}\n"
            f"  Result DataFrame (Preview):\n{df_preview}\n"
            f"  Deletion Reasons:\n{reasons}"
            f"\n"
        )


class BaseCheck(ABC):
    def __init__(self, task_type: str, options: dict, report_path: str = "mlleakcatcher"):
            """
            Initialize a base check.

            Args:
                options (dict): Configuration options for the check
                report_path (str): Base path for report artifacts
            """
            self.task_type = task_type
            print(options)
            self.options = options
            self.report_path = f"{report_path}/tests_targetleak"
            self.name = self.__class__.__name__.lower().replace("check", "")
            self.deletion_threshold = self.options.get("warning_threshold", 0.8)
            self.warning_threshold = self.options.get("warning_threshold", 0.5)
            self.model_type =  "decision_tree"
            self.model_size = "medium"
            self.del_features = []
            self.warning_features = []
            self.deletion_reasons = {}


    @abstractmethod
    def run(self, train_ds: Dataset, test_ds: Optional[Dataset] = None) -> CheckResult:
        """
        Run the check and return results.

        Args:
            train_ds (Dataset): Training dataset
            test_ds (Optional[Dataset]): Test dataset, if applicable

        Returns:
            CheckResult: Results of the check
        """
        pass

    def _save_result_csv(self, result_df: pd.DataFrame, filename: str = None):
        """
        Save results DataFrame to CSV.

        Args:
            result_df (pd.DataFrame): Results DataFrame
            filename (str, optional): Custom filename. Defaults to check name.
        """
        if filename is None:
            filename = f"{self.name}.csv"

        result_df.to_csv(
            f"{self.report_path}/{filename}",
            index=False,
        )


class ModelTargetCheck(BaseCheck):
    pass

class IdentifierTargetPPSCheck(BaseCheck):
    """
    Checks for correlation between identifier columns and the target variable using EnhancedPPS.
    """

    def __init__(self, task_type : str, options: dict = None, report_path: str = "mlleakcatcher"):
        super().__init__(task_type, options, report_path)


    def run(self, train_ds: Dataset, test_ds: Optional[Dataset] = None) -> CheckResult:
        """
        Calculate PPS between identifier columns and target using EnhancedPPS.
        """

        identifiers = train_ds.id_cols
        if not identifiers:
            return CheckResult(name=self.name, result_df=pd.DataFrame())

        X = train_ds.data[identifiers]
        y = train_ds.data[train_ds.target_col]

        print(self.options)
        # Calculate PPS using EnhancedPPS
        pps_scores = EnhancedPPS.calculate_pps(
            X=X,
            y=y,
            task_type=self.task_type,
            model_type=self.model_type,
            model_size=self.model_size,
            n_samples=self.options.get("n_samples", 10000)
        )

        # Create result DataFrame
        result_df = pd.DataFrame(
            list(pps_scores.items()),
            columns=["Identifier", "Predictive Power Score (PPS)"],
        )

        print(pps_scores)
        print(self.warning_threshold)
        # Identify warning features
        warning_features = [
            col for col, score in pps_scores.items()
            if score >= self.warning_threshold
        ]

        return CheckResult(
            name=self.name,
            warning_features=warning_features,
            result_df=result_df,
        )



class FeatureTargetPPSCheck(BaseCheck):
    """
    Checks feature-target correlations using EnhancedPPS.
    """

    def __init__(self, task_type : str, options: dict = None, report_path: str = "mlleakcatcher"):
        super().__init__(task_type, options, report_path)

    def run(self, train_ds: Dataset, test_ds: Optional[Dataset] = None) -> CheckResult:
        """
        Calculate PPS between features and target using EnhancedPPS.
        """

        features = [col for col in train_ds.data.columns if col != train_ds.target_col]
        X = train_ds.data[features]
        y = train_ds.data[train_ds.target_col]

        # Calculate PPS using EnhancedPPS
        pps_scores = EnhancedPPS.calculate_pps(
            X=X,
            y=y,
            task_type=self.task_type,
            model_type=self.model_type,
            model_size=self.model_size,
            n_samples=self.options.get("n_samples", 10000)
        )

        result_df = pd.DataFrame(
            list(pps_scores.items()),
            columns=["Feature", "Predictive Power Score (PPS)"],
        )

        deleted_features = [
            col for col, score in pps_scores.items()
            if score >= self.deletion_threshold
        ]

        warning_features = [
            col for col, score in pps_scores.items()
            if score >= self.warning_threshold and score < self.deletion_threshold
        ]

        deletion_reasons = {
            col: f"High PPS: {pps_scores[col]:.3f}"
            for col in deleted_features
        }

        return CheckResult(
            name=self.name,
            deleted_features=deleted_features,
            warning_features=warning_features,
            result_df=result_df,
            deletion_reasons=deletion_reasons,
        )
class FeatureTargetPPSChangeCheck(BaseCheck):
    """
    Checks feature-target correlation changes between train and test using EnhancedPPS.
    """

    def __init__(self, task_type : str, options: dict = None, report_path: str = "mlleakcatcher"):
        super().__init__(task_type, options, report_path)

    def run(self, train_ds: Dataset, test_ds: Optional[Dataset] = None) -> CheckResult:
        """
        Calculate PPS changes between train and test using EnhancedPPS.
        """
        if test_ds is None:
            raise ValueError("Test dataset is required for this check")

        common_features = [
            col for col in train_ds.data.columns
            if col != train_ds.target_name and col in test_ds.data.columns
        ]

        # Calculate EnhancedPPS for train
        train_pps = EnhancedPPS.calculate_pps(
            X=train_ds.data[common_features],
            y=train_ds.data[train_ds.target_name],
            task_type=self.task_type,
            model_type=self.model_type,
            model_size=self.model_size,
            n_samples=self.options.get("n_samples", 10000)
        )

        # Calculate EnhancedPPS for test
        test_pps = EnhancedPPS.calculate_pps(
            X=test_ds.data[common_features],
            y=test_ds.data[test_ds.target_name],
            task_type=self.task_type,
            model_type=self.model_type,
            model_size=self.model_size,
            n_samples=self.options.get("n_samples", 10000)
        )

        diff_pps = {
            col: abs(train_pps[col] - test_pps[col])
            for col in common_features
        }

        result_data = []
        for col in common_features:
            result_data.append({
                "Feature": col,
                "Train_PPS": train_pps[col],
                "Test_PPS": test_pps[col],
                "Difference": diff_pps[col],
                "data_type": "train"
            })
            result_data.append({
                "Feature": col,
                "Train_PPS": train_pps[col],
                "Test_PPS": test_pps[col],
                "Difference": diff_pps[col],
                "data_type": "test"
            })

        result_df = pd.DataFrame(result_data)

        deleted_features = []
        warning_features = []
        deletion_reasons = {}

        for col, score in train_pps.items():
            if score >= self.deletion_threshold:
                deleted_features.append(col)
                deletion_reasons[col] = f"High train PPS: {score:.3f}"
            elif score >= self.warning_threshold:
                warning_features.append(col)

        for col, score in test_pps.items():
            if score >= self.deletion_threshold and col not in deleted_features:
                deleted_features.append(col)
                deletion_reasons[col] = f"High test PPS: {score:.3f}"
            elif score >= self.warning_threshold and col not in warning_features:
                warning_features.append(col)

        for col, diff in diff_pps.items():
            if diff >= self.warning_threshold_difference and col not in deleted_features:
                deleted_features.append(col)
                deletion_reasons[col] = f"Large train-test PPS difference: {diff:.3f}"
            elif diff >= (self.warning_threshold_difference * 0.5) and col not in warning_features:
                warning_features.append(col)

        return CheckResult(
            name=self.name,
            deleted_features=deleted_features,
            warning_features=warning_features,
            result_df=result_df,
            deletion_reasons=deletion_reasons,
        )
