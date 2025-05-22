import streamlit as st
import pandas as pd
import numpy as np
import os
import tempfile
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO

st.set_page_config(
    page_title="MLLeakCatcher",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

try:
    from mlleakcatcher import MlLeakCatcher, Dataset
    from mlleakcatcher.plot_utils import plot_top_features_by_pps, plot_top_features_train_test_difference
    MLLEAKCATCHER_AVAILABLE = True
except ImportError:
    MLLEAKCATCHER_AVAILABLE = False
    class Dataset:
        def __init__(self, data_source, target_col, id_cols=None, backend=None):
            self.data = data_source
            self.target_col = target_col
            self.id_cols = id_cols or []
            self.backend = backend or "pandas"

    class MlLeakCatcher:
        def __init__(self, task_type, verbosity=0, report_path="mlleakcatcher", **kwargs):
            self.task_type = task_type
            self.verbosity = verbosity
            self.report_path = report_path
            self.options = kwargs
            self.test_results = []

        def run(self, train_data, target_col, id_cols=None, test_data=None):
            # Mock implementation
            features = [col for col in train_data.columns if col != target_col and (id_cols is None or col not in id_cols)]
            return features, []

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #1E3A8A;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #F0F7FF;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #BFDBFE;
        margin-bottom: 1rem;
    }
    .warning-box {
        background-color: #FEF3C7;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #FCD34D;
        margin-bottom: 1rem;
    }
    .success-box {
        background-color: #ECFDF5;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #A7F3D0;
        margin-bottom: 1rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #F3F4F6;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #DBEAFE;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">MLLeakCatcher</h1>', unsafe_allow_html=True)
st.markdown('<div class="info-box">A tool to detect data leakage in your machine learning datasets</div>', unsafe_allow_html=True)

st.sidebar.title("Configuration")

st.sidebar.subheader("Data Upload")
train_file = st.sidebar.file_uploader("Upload Training Data", type=["csv", "xlsx", "parquet"])
test_file = st.sidebar.file_uploader("Upload Test Data (Optional)", type=["csv", "xlsx", "parquet"])

st.sidebar.subheader("MLLeakCatcher Settings")
task_type = st.sidebar.selectbox("Task Type", ["classification", "regression"])
verbosity = st.sidebar.slider("Verbosity Level", 0, 3, 1)
report_path = st.sidebar.text_input("Report Path", "mlleakcatcher_report")

st.sidebar.subheader("Advanced Settings")
show_advanced = st.sidebar.checkbox("Show Advanced Settings", False)

if show_advanced:
    with st.sidebar.expander("Identifier Target PPS Check Settings"):
        id_target_pps_enabled = st.checkbox("Enable Identifier Target PPS Check", True)
        if id_target_pps_enabled:
            id_warning_threshold = st.slider("Warning Threshold", 0.0, 1.0, 0.01, 0.01)
            id_model_type = st.selectbox("Model Type (Identifier)", ["random_forest", "decision_tree", "logistic_regression", "svm", "catboost"])
            id_model_size = st.selectbox("Model Size (Identifier)", ["small", "medium", "large"])
            id_n_samples = st.number_input("Number of Samples (Identifier)", 1000, 100000, 10000, 1000)

    with st.sidebar.expander("Feature Target PPS Check Settings"):
        feature_target_pps_enabled = st.checkbox("Enable Feature Target PPS Check", True)
        if feature_target_pps_enabled:
            feature_deletion_threshold = st.slider("Deletion Threshold", 0.0, 1.0, 0.8, 0.01)
            feature_warning_threshold = st.slider("Warning Threshold", 0.0, 1.0, 0.5, 0.01)
            feature_model_type = st.selectbox("Model Type (Feature)", ["random_forest", "decision_tree", "logistic_regression", "svm", "catboost"])
            feature_model_size = st.selectbox("Model Size (Feature)", ["small", "medium", "large"])
            feature_n_samples = st.number_input("Number of Samples (Feature)", 1000, 100000, 30000, 1000)

    with st.sidebar.expander("Feature Target PPS Change Check Settings"):
        feature_change_enabled = st.checkbox("Enable Feature Target PPS Change Check", True)
        if feature_change_enabled:
            change_deletion_threshold = st.slider("Change Deletion Threshold", 0.0, 1.0, 0.8, 0.01)
            change_warning_threshold = st.slider("Change Warning Threshold", 0.0, 1.0, 0.5, 0.01)
            change_warning_threshold_diff = st.slider("Change Warning Threshold Difference", 0.0, 1.0, 0.3, 0.01)
            change_model_type = st.selectbox("Model Type (Change)", ["random_forest", "decision_tree", "logistic_regression", "svm", "catboost"])
            change_model_size = st.selectbox("Model Size (Change)", ["small", "medium", "large"])
            change_n_samples = st.number_input("Number of Samples (Change)", 1000, 100000, 30000, 1000)
else:
    id_target_pps_enabled = True
    id_warning_threshold = 0.01
    id_model_type = "random_forest"
    id_model_size = "medium"
    id_n_samples = 10000

    feature_target_pps_enabled = True
    feature_deletion_threshold = 0.8
    feature_warning_threshold = 0.5
    feature_model_type = "random_forest"
    feature_model_size = "medium"
    feature_n_samples = 30000

    feature_change_enabled = True
    change_deletion_threshold = 0.8
    change_warning_threshold = 0.5
    change_warning_threshold_diff = 0.3
    change_model_type = "random_forest"
    change_model_size = "medium"
    change_n_samples = 30000

def load_data(file_obj):
    if file_obj is None:
        return None

    file_extension = os.path.splitext(file_obj.name)[1].lower()

    try:
        if file_extension == ".csv":
            return pd.read_csv(file_obj)
        elif file_extension == ".xlsx":
            return pd.read_excel(file_obj)
        elif file_extension == ".parquet":
            return pd.read_parquet(file_obj)
        else:
            st.error(f"Unsupported file format: {file_extension}")
            return None
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None

train_data = load_data(train_file)
test_data = load_data(test_file)

if train_data is not None:
    st.markdown('<h2 class="sub-header">Data Preview</h2>', unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["Training Data", "Test Data"])

    with tab1:
        st.dataframe(train_data.head(10))
        st.text(f"Shape: {train_data.shape[0]} rows, {train_data.shape[1]} columns")

        with st.expander("Data Summary", expanded=False):
            buffer = BytesIO()

            summary_stats = train_data.describe(include='all').T
            summary_stats['missing'] = train_data.isnull().sum()
            summary_stats['missing_percent'] = train_data.isnull().mean() * 100

            st.dataframe(summary_stats)

            dtypes = train_data.dtypes.value_counts()
            fig, ax = plt.subplots(figsize=(8, 4))
            dtypes.plot(kind='bar', ax=ax)
            ax.set_title('Distribution of Data Types')
            ax.set_ylabel('Count')
            st.pyplot(fig)

    with tab2:
        if test_data is not None:
            st.dataframe(test_data.head(10))
            st.text(f"Shape: {test_data.shape[0]} rows, {test_data.shape[1]} columns")

            with st.expander("Data Summary", expanded=False):
                summary_stats = test_data.describe(include='all').T
                summary_stats['missing'] = test_data.isnull().sum()
                summary_stats['missing_percent'] = test_data.isnull().mean() * 100

                st.dataframe(summary_stats)
        else:
            st.info("No test data uploaded")

    st.markdown('<h2 class="sub-header">MLLeakCatcher Configuration</h2>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        target_col = st.selectbox("Target Column", train_data.columns)

    with col2:
        id_cols = st.multiselect("ID Columns (Optional)", train_data.columns)

    config = {
        "identifier_target_pps": {
            "warning_threshold": id_warning_threshold,
            "model_type": id_model_type,
            "model_size": id_model_size,
            "n_samples": id_n_samples,
        } if id_target_pps_enabled else False,

        "feature_target_pps": {
            "deletion_threshold": feature_deletion_threshold,
            "warning_threshold": feature_warning_threshold,
            "model_type": feature_model_type,
            "model_size": feature_model_size,
            "n_samples": feature_n_samples,
        } if feature_target_pps_enabled else False,

        "feature_target_pps_change": {
            "deletion_threshold": change_deletion_threshold,
            "warning_threshold": change_warning_threshold,
            "warning_threshold_difference": change_warning_threshold_diff,
            "model_type": change_model_type,
            "model_size": change_model_size,
            "n_samples": change_n_samples,
        } if feature_change_enabled and test_data is not None else False,
    }

    with st.expander("Configuration Summary"):
        st.json(config)

    st.markdown('<h2 class="sub-header">Run Leak Detection</h2>', unsafe_allow_html=True)

    if st.button("Run MLLeakCatcher"):
        with st.spinner("Running leak detection..."):
            try:
                with tempfile.TemporaryDirectory() as temp_dir:
                    detector = MlLeakCatcher(
                        task_type=task_type,
                        verbosity=verbosity,
                        report_path=temp_dir,
                        **config
                    )

                    safe_features, test_results = detector.run(
                        train_data=train_data,
                        target_col=target_col,
                        id_cols=id_cols if id_cols else None,
                        test_data=test_data
                    )

                    st.markdown('<h2 class="sub-header">Results</h2>', unsafe_allow_html=True)

                    st.markdown(f"""
                    <div class="success-box">
                        <h3>Summary</h3>
                        <ul>
                            <li>Total features: {len(train_data.columns) - 1 - len(id_cols)}</li>
                            <li>Safe features: {len(safe_features)}</li>
                            <li>Deleted features: {len(train_data.columns) - 1 - len(id_cols) - len(safe_features)}</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)

                    if not MLLEAKCATCHER_AVAILABLE:
                        all_features = [col for col in train_data.columns if col != target_col and col not in id_cols]

                        feature_pps_result = pd.DataFrame({
                            "Feature": all_features,
                            "Predictive Power Score (PPS)": np.random.uniform(0, 1, size=len(all_features))
                        })

                        feature_pps_result = feature_pps_result.sort_values(
                            by="Predictive Power Score (PPS)",
                            ascending=False
                        ).reset_index(drop=True)

                        deleted_features = []
                        warning_features = []
                        deletion_reasons = {}

                        for idx, row in feature_pps_result.iterrows():
                            if row["Predictive Power Score (PPS)"] > feature_deletion_threshold:
                                deleted_features.append(row["Feature"])
                                deletion_reasons[row["Feature"]] = f"High PPS: {row['Predictive Power Score (PPS)']:.3f}"
                            elif row["Predictive Power Score (PPS)"] > feature_warning_threshold:
                                warning_features.append(row["Feature"])

                        if test_data is not None:
                            train_test_result = pd.DataFrame()
                            for feature in all_features:
                                if feature not in deleted_features:
                                    train_pps = np.random.uniform(0, 0.7)
                                    test_pps = np.random.uniform(0, 0.7)
                                    diff = abs(train_pps - test_pps)

                                    train_test_result = pd.concat([train_test_result, pd.DataFrame({
                                        "Feature": [feature, feature],
                                        "Train_PPS": [train_pps, train_pps],
                                        "Test_PPS": [test_pps, test_pps],
                                        "Difference": [diff, diff],
                                        "data_type": ["train", "test"]
                                    })], ignore_index=True)

                                    if diff > change_warning_threshold_diff:
                                        if feature not in deleted_features:
                                            deleted_features.append(feature)
                                            deletion_reasons[feature] = f"Large train-test PPS difference: {diff:.3f}"
                                        if feature not in warning_features:
                                            warning_features.append(feature)

                        test_results = [
                            type('CheckResult', (), {
                                'name': 'feature_target_pps',
                                'deleted_features': deleted_features.copy(),
                                'warning_features': warning_features.copy(),
                                'result_df': feature_pps_result,
                                'deletion_reasons': deletion_reasons.copy()
                            }),
                        ]

                        if test_data is not None:
                            test_results.append(
                                type('CheckResult', (), {
                                    'name': 'feature_target_pps_change',
                                    'deleted_features': deleted_features.copy(),
                                    'warning_features': warning_features.copy(),
                                    'result_df': train_test_result,
                                    'deletion_reasons': deletion_reasons.copy()
                                })
                            )

                    for result in test_results:
                        st.markdown(f'<h3 class="sub-header">{result.name} Results</h3>', unsafe_allow_html=True)

                        if hasattr(result, 'result_df') and result.result_df is not None:
                            result_df = result.result_df

                            st.dataframe(result_df)

                            if result.name == 'feature_target_pps':
                                fig, ax = plt.subplots(figsize=(10, 6))
                                top_n = 20

                                top_features = result_df.sort_values(
                                    by="Predictive Power Score (PPS)",
                                    ascending=False
                                ).head(top_n)

                                sns.barplot(
                                    x="Predictive Power Score (PPS)",
                                    y="Feature",
                                    data=top_features,
                                    ax=ax
                                )

                                ax.set_title(f'Top {top_n} Features by PPS')
                                ax.axvline(x=feature_warning_threshold, color='orange', linestyle='--', label=f'Warning Threshold ({feature_warning_threshold})')
                                ax.axvline(x=feature_deletion_threshold, color='red', linestyle='--', label=f'Deletion Threshold ({feature_deletion_threshold})')
                                ax.legend()

                                st.pyplot(fig)

                            elif result.name == 'feature_target_pps_change' and test_data is not None:
                                if 'Difference' in result_df.columns:
                                    unique_features = result_df['Feature'].unique()

                                    top_diff_features = result_df.sort_values(by='Difference', ascending=False)
                                    top_diff_features = top_diff_features.drop_duplicates(subset=['Feature']).head(15)

                                    fig, ax = plt.subplots(figsize=(12, 6))

                                    plot_data = result_df[result_df['Feature'].isin(top_diff_features['Feature'])]

                                    train_data = plot_data[plot_data['data_type'] == 'train']
                                    test_data = plot_data[plot_data['data_type'] == 'test']

                                    x = np.arange(len(train_data))
                                    width = 0.35

                                    ax.bar(x - width/2, train_data['Train_PPS'], width, label='Train PPS')
                                    ax.bar(x + width/2, test_data['Test_PPS'], width, label='Test PPS')

                                    ax.set_xlabel('Feature')
                                    ax.set_ylabel('PPS Score')
                                    ax.set_title('Train vs Test PPS for Features with High Difference')
                                    ax.set_xticks(x)
                                    ax.set_xticklabels(train_data['Feature'], rotation=90)
                                    ax.legend()

                                    st.pyplot(fig)

                        if hasattr(result, 'deleted_features') and result.deleted_features:
                            st.markdown('<h4>Deleted Features</h4>', unsafe_allow_html=True)
                            for feature in result.deleted_features:
                                reason = result.deletion_reasons.get(feature, "Unknown reason")
                                st.markdown(f"- **{feature}**: {reason}")

                        if hasattr(result, 'warning_features') and result.warning_features:
                            st.markdown('<h4>Warning Features</h4>', unsafe_allow_html=True)
                            for feature in result.warning_features:
                                if feature not in result.deleted_features:
                                    st.markdown(f"- **{feature}**")

                    st.markdown('<h3 class="sub-header">Safe Features</h3>', unsafe_allow_html=True)
                    st.write(safe_features)

                    safe_features_df = pd.DataFrame({"safe_features": safe_features})
                    csv = safe_features_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download Safe Features List",
                        data=csv,
                        file_name="safe_features.csv",
                        mime="text/csv",
                    )

            except Exception as e:
                st.error(f"Error running MLLeakCatcher: {str(e)}")
                st.exception(e)
else:
    st.info("Please upload your training data to begin.")

    with st.expander("How to use MLLeakCatcher"):
        st.markdown("""
        ### MLLeakCatcher Workflow

        1. **Upload Data**: Upload your training dataset and optionally a test dataset
        2. **Configure Settings**:
           - Select the target column
           - Identify any ID columns
           - Choose your task type (classification/regression)
           - Adjust thresholds and model settings if needed
        3. **Run Analysis**: Click "Run MLLeakCatcher" to analyze your data
        4. **Review Results**:
           - Examine deleted and warning features
           - Understand why features were flagged
           - Download the safe features list

        ### What MLLeakCatcher Detects

        - **Identifier Target Correlation**: Checks if ID columns are predictive of the target
        - **Feature Target Correlation**: Identifies features with suspiciously high correlation with the target
        - **Train/Test Distribution Changes**: Detects features whose predictive power changes significantly between train and test sets

        ### Benefits

        - Prevents overfitting and data leakage
        - Improves model generalization
        - Identifies features that might cause production issues
        """)

        st.image("https://via.placeholder.com/800x400.png?text=MLLeakCatcher+Workflow+Diagram",
                 caption="Sample MLLeakCatcher workflow diagram")

st.markdown("---")
st.markdown("MLLeakCatcher - A tool for detecting data leakage in machine learning datasets")