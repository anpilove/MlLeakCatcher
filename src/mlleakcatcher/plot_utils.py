import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_top_features_by_pps(result_dt: pd.DataFrame, report_path: str):
    """
    Plot and save a bar visualization of the top 10 features ranked by their Predictive Power Score (PPS).

    Args:
        result_dt (pd.DataFrame): DataFrame containing 'Feature' and 'PPS' columns.
        report_path (str): report path

    Returns:
        matplotlib.figure.Figure: The generated plot figure object
    """

    top_n = 10

    top_features = result_dt.nlargest(top_n, "PPS")

    data_type = "train"

    if "data_type" in result_dt.columns:
        data_type = result_dt["data_type"].iloc[0]  # type: ignore[assignment]  # result_dt["data_type"] always str

    with plt.style.context("seaborn-notebook"):
        fig, ax = plt.subplots(figsize=(8, 8))

        sns.barplot(
            data=top_features,
            x="Feature",
            y="PPS",
            palette="viridis",
            ax=ax,
            hue="Feature",
            dodge=False,
        )

        ax.set_xlabel("Feature", fontsize=12)
        ax.set_ylabel("Predictive Power Score (PPS)", fontsize=12)
        ax.set_title(
            f"Top {top_n} Features by Predictive Power Score (PPS)", fontsize=14
        )

        ax.set_xticks(range(len(top_features["Feature"])))
        ax.set_xticklabels(
            top_features["Feature"], rotation=45, ha="right", fontsize=10
        )

    plt.subplots_adjust(bottom=0.5)
    plt.savefig(f"{report_path}/top_feature_label_correlation_{data_type}.png")
    plt.close(fig)

    return fig


def plot_top_features_train_test_difference(
    result_df: pd.DataFrame,
    report_path: str,
):
    """
    Plot and save a comparison visualization showing the top 10 features with largest PPS differences between train and test sets.

    Args:
        result_df (pd.DataFrame): DataFrame containing columns 'Feature', 'PPS', and 'data_type'
        report_path (str): report path

    Returns:
        matplotlib.figure.Figure: The generated plot figure object
    """
    top_n = 10

    diff_df = result_df[result_df["data_type"] == "train-test difference"]
    top_diff_features = diff_df.nlargest(top_n, "PPS")

    train_test_df = result_df[
        result_df["Feature"].isin(top_diff_features["Feature"])
        & result_df["data_type"].isin(["train", "test"])
    ]

    train_test_df = train_test_df.sort_values(by="PPS", ascending=False)

    with plt.style.context("seaborn-notebook"):
        fig, ax = plt.subplots(figsize=(8, 8))

        sns.barplot(
            data=train_test_df,
            x="Feature",
            y="PPS",
            hue="data_type",
            palette="viridis",
            ax=ax,
        )

        ax.set_title(
            f"Top {top_n} Features by Train-Test Difference in PPS", fontsize=14
        )
        ax.set_xlabel("Feature", fontsize=12)
        ax.set_ylabel("Predictive Power Score (PPS)", fontsize=12)

        feature_order = train_test_df["Feature"]  # type: ignore[operator] # train_test_df["Feature"] always Series[Any]
        ax.set_xticks(range(len(feature_order)))
        ax.set_xticklabels(feature_order, rotation=45, ha="right", fontsize=10)

    plt.subplots_adjust(bottom=0.5)
    plt.savefig(
        f"{report_path}/top_feature_label_correlation_train_test_difference.png"
    )
    plt.close(fig)

    return fig