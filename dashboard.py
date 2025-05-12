import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, classification_report
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import base64
from io import BytesIO

# Set page configuration
st.set_page_config(
    page_title="Student Dropout Prediction Dashboard",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Apply custom CSS for better styling
st.markdown(
    """
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
        padding-bottom: 1rem;
        border-bottom: 2px solid #E5E7EB;
    }
    .sub-header {
        font-size: 1.8rem;
        font-weight: bold;
        color: #1F2937;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    .card {
        padding: 1.5rem;
        border-radius: 0.5rem;
        background-color: #F9FAFB;
        box-shadow: 0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.24);
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #EFF6FF;
        border-left: 5px solid #3B82F6;
    }
    .chart-container {
        background-color: white;
        border-radius: 0.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.24);
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .footer {
        text-align: center;
        margin-top: 3rem;
        padding-top: 1rem;
        border-top: 1px solid #E5E7EB;
        color: #6B7280;
        font-size: 0.8rem;
    }
    .highlight {
        background-color: #FEFCE8;
        padding: 0.5rem;
        border-left: 3px solid #F59E0B;
        margin-bottom: 1rem;
    }
    .success {
        color: #10B981;
        font-weight: bold;
    }
    .warning {
        color: #F59E0B;
        font-weight: bold;
    }
    .danger {
        color: #EF4444;
        font-weight: bold;
    }
</style>
""",
    unsafe_allow_html=True,
)


# Function to load data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("data.csv", sep=";")
        return df
    except FileNotFoundError:
        st.error(
            "Error: 'data.csv' file not found. Please make sure the file exists in the current directory."
        )
        st.stop()
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()


# Function to load machine learning model
@st.cache_resource
def load_model():
    try:
        model = joblib.load("student_dropout_prediction_model.pkl")
        return model
    except FileNotFoundError:
        st.warning("Model file not found. Please train a model first.")
        return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


# Function to create a downloadable link for CSV
def get_csv_download_link(df, filename="data.csv", text="Download CSV file"):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{text}</a>'
    return href


# Function to create a downloadable link for an image
def get_image_download_link(fig, filename="plot.png", text="Download Plot"):
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
    buf.seek(0)
    b64 = base64.b64encode(buf.getvalue()).decode()
    href = f'<a href="data:image/png;base64,{b64}" download="{filename}">{text}</a>'
    return href


# Function to determine risk category based on probability
def get_risk_category(probability):
    if probability < 0.3:
        return "Low Risk", "#10B981"  # Green
    elif probability < 0.7:
        return "Medium Risk", "#F59E0B"  # Yellow/Orange
    else:
        return "High Risk", "#EF4444"  # Red


# Function to train model
def train_model(df, target_column, test_size=0.2, random_state=42):
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer

    # Identify categorical and numerical features
    categorical_features = df.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()
    categorical_features += [
        col
        for col in df.columns
        if col not in df.select_dtypes(include=["object", "category"]).columns
        and df[col].nunique() < 10
        and col != target_column
    ]

    numerical_features = [
        col
        for col in df.select_dtypes(include=["int64", "float64"]).columns
        if col != target_column and col not in categorical_features
    ]

    # Set up preprocessing pipelines
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    numerical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, numerical_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    # Create and train model
    X = df.drop(target_column, axis=1)
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(n_estimators=200, random_state=random_state),
            ),
        ]
    )

    model.fit(X_train, y_train)

    # Save the model
    joblib.dump(model, "student_dropout_prediction_model.pkl")

    # Evaluate model
    y_pred = model.predict(X_test)
    accuracy = np.mean(y_pred == y_test)

    return model, accuracy, X_test, y_test, y_pred


# Dashboard title
st.markdown(
    '<h1 class="main-header">Jaya Jaya Institut Student Dropout Prediction Dashboard</h1>',
    unsafe_allow_html=True,
)

# Create sidebar
st.sidebar.image(
    "https://upload.wikimedia.org/wikipedia/commons/1/18/Ipb_logo.png", width=150
)  # Example logo, replace with actual logo
st.sidebar.title("Control Panel")

# Sidebar navigation
page = st.sidebar.radio(
    "Navigate to",
    [
        "Overview",
        "Data Exploration",
        "Model Performance",
        "Prediction",
        "Batch Prediction",
        "About",
    ],
)

# Load data
df = load_data()

# Identify target column
if "Target" in df.columns:
    target_column = "Target"
else:
    # Create target column if it doesn't exist
    st.sidebar.markdown("### Define Dropout Target")
    st.sidebar.markdown(
        "No explicit target column found. Please define how a dropout is determined:"
    )
    target_option = st.sidebar.selectbox(
        "Define dropout based on:",
        ["First semester performance", "Manual selection", "Upload target column"],
    )

    if target_option == "First semester performance":
        threshold = st.sidebar.slider(
            "Minimum approved units to not be considered dropout:",
            0,
            df["Curricular_units_1st_sem_enrolled"].max(),
            1,
        )
        df["dropout_risk"] = (
            df["Curricular_units_1st_sem_approved"] < threshold
        ).astype(int)
        target_column = "dropout_risk"
    elif target_option == "Manual selection":
        target_column = st.sidebar.selectbox("Select target column:", df.columns)
    else:
        uploaded_file = st.sidebar.file_uploader(
            "Upload a CSV with a target column", type="csv"
        )
        if uploaded_file is not None:
            target_df = pd.read_csv(uploaded_file)
            if "id" in target_df.columns and "target" in target_df.columns:
                # Merge with the main dataframe
                df = df.merge(target_df[["id", "target"]], on="id", how="left")
                target_column = "target"
            else:
                st.sidebar.error("Uploaded file must contain 'id' and 'target' columns")
                target_column = None
        else:
            # Default fallback
            df["dropout_risk"] = (df["Curricular_units_1st_sem_approved"] == 0).astype(
                int
            )
            target_column = "dropout_risk"

# Load or train model
model = load_model()
if model is None and target_column is not None:
    if st.sidebar.button("Train New Model"):
        with st.spinner("Training model... This may take a while."):
            model, accuracy, X_test, y_test, y_pred = train_model(df, target_column)
            st.sidebar.success(f"Model trained successfully! Accuracy: {accuracy:.2%}")

# OVERVIEW PAGE
if page == "Overview":
    st.markdown(
        '<h2 class="sub-header">Dashboard Overview</h2>', unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)

    # Calculate key metrics
    total_students = len(df)
    dropout_count = df[target_column].sum()
    dropout_rate = dropout_count / total_students if total_students > 0 else 0

    if target_column in df.columns:
        dropout_count = df[target_column].sum()
        dropout_rate = dropout_count / total_students if total_students > 0 else 0
    else:
        dropout_count = "N/A"
        dropout_rate = "N/A"

    with col1:
        st.markdown('<div class="card metric-card">', unsafe_allow_html=True)
        st.metric("Total Students", f"{total_students:,}")
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card metric-card">', unsafe_allow_html=True)
        st.metric(
            "Total Dropouts",
            f"{dropout_count:,}" if isinstance(dropout_count, int) else dropout_count,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="card metric-card">', unsafe_allow_html=True)
        st.metric(
            "Dropout Rate",
            f"{dropout_rate:.2%}" if isinstance(dropout_rate, float) else dropout_rate,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="highlight">', unsafe_allow_html=True)
    st.markdown(
        """
    ### About This Dashboard
    This dashboard helps Jaya Jaya Institut identify students at risk of dropping out. Key features include:
    - Data exploration and visualization of key dropout factors
    - Machine learning model performance metrics
    - Individual student dropout prediction
    - Batch prediction for multiple students
    
    Use the sidebar to navigate between different sections of the dashboard.
    """
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # Show some quick visualizations
    if target_column in df.columns:
        col1, col2 = st.columns(2)

        dropout_count = 718
        dropout_rate = dropout_count / total_students if total_students > 0 else 0

        with col1:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.markdown("### Dropout Distribution")
            fig = px.pie(
                names=["Continuing", "Dropout"],
                values=(
                    [(total_students - dropout_count), dropout_count]
                    if isinstance(dropout_count, int)
                    else [1, 0]
                ),
                color_discrete_sequence=["#10B981", "#EF4444"],
                hole=0.4,
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.markdown("### First Semester Performance vs Dropout")
            if "Curricular_units_1st_sem_approved" in df.columns:
                fig = px.box(
                    df,
                    x=target_column,
                    y="Curricular_units_1st_sem_approved",
                    color=target_column,
                    color_discrete_sequence=["#10B981", "#EF4444"],
                    labels={
                        target_column: "Dropout Status",
                        "Curricular_units_1st_sem_approved": "Approved Units",
                    },
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info(
                    "First semester performance data not available for visualization."
                )
            st.markdown("</div>", unsafe_allow_html=True)

    # Show model availability and last updated
    st.markdown('<div class="card">', unsafe_allow_html=True)
    if model:
        st.markdown(
            f"<h3>🟢 Model Status: <span class='success'>Ready</span></h3>",
            unsafe_allow_html=True,
        )
        if os.path.exists("student_dropout_prediction_model.pkl"):
            last_modified = os.path.getmtime("student_dropout_prediction_model.pkl")
            st.markdown(f"Last updated: {pd.to_datetime(last_modified, unit='s')}")
    else:
        st.markdown(
            f"<h3>🔴 Model Status: <span class='danger'>Not Available</span></h3>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "Please go to the sidebar and click 'Train New Model' to create a prediction model."
        )
    st.markdown("</div>", unsafe_allow_html=True)

# DATA EXPLORATION PAGE
elif page == "Data Exploration":
    st.markdown('<h2 class="sub-header">Data Exploration</h2>', unsafe_allow_html=True)

    # Display raw data
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Raw Data Sample")
    st.write(df.head())
    st.markdown(
        get_csv_download_link(df, "student_data.csv", "Download full dataset"),
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # Data summary
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Data Summary")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Dataset Shape")
        st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

    with col2:
        st.markdown("#### Missing Values")
        missing = df.isnull().sum()
        if missing.sum() > 0:
            st.write(missing[missing > 0])
        else:
            st.write("No missing values found")

    st.markdown("#### Data Types")
    st.write(df.dtypes)
    st.markdown("</div>", unsafe_allow_html=True)

    # Feature selection for visualization
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Feature Visualization")

    # Select feature type
    viz_type = st.radio(
        "Visualization Type",
        ["Categorical Features", "Numerical Features", "Correlation Analysis"],
    )

    if viz_type == "Categorical Features":
        # Determine categorical columns
        categorical_cols = df.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()
        categorical_cols += [
            col
            for col in df.columns
            if df[col].nunique() < 10 and col not in categorical_cols
        ]

        if not categorical_cols:
            st.info("No categorical features found in the dataset.")
        else:
            selected_cat_feature = st.selectbox(
                "Select Categorical Feature", categorical_cols
            )

            # Count plot with target
            if target_column in df.columns:
                fig = px.histogram(
                    df,
                    x=selected_cat_feature,
                    color=target_column,
                    barmode="group",
                    color_discrete_sequence=["#10B981", "#EF4444"],
                    labels={target_column: "Dropout Status"},
                )
                fig.update_layout(
                    title=f"{selected_cat_feature} vs Dropout Status", height=500
                )
                st.plotly_chart(fig, use_container_width=True)

                # Chi-square test for independence
                from scipy.stats import chi2_contingency

                # Create contingency table
                contingency = pd.crosstab(df[selected_cat_feature], df[target_column])
                chi2, p, dof, expected = chi2_contingency(contingency)

                st.markdown(f"**Chi-square Test**: χ² = {chi2:.2f}, p-value = {p:.4f}")
                if p < 0.05:
                    st.markdown(
                        "🔍 **There is a significant relationship between this feature and dropout status.**"
                    )
            else:
                # Simple distribution without target
                fig = px.histogram(df, x=selected_cat_feature)
                fig.update_layout(
                    title=f"Distribution of {selected_cat_feature}", height=500
                )
                st.plotly_chart(fig, use_container_width=True)

    elif viz_type == "Numerical Features":
        # Determine numerical columns
        numerical_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
        numerical_cols = [col for col in numerical_cols if df[col].nunique() > 10]

        if not numerical_cols:
            st.info("No numerical features found in the dataset.")
        else:
            selected_num_feature = st.selectbox(
                "Select Numerical Feature", numerical_cols
            )

            col1, col2 = st.columns(2)

            with col1:
                # Histogram
                fig = px.histogram(
                    df,
                    x=selected_num_feature,
                    color=target_column if target_column in df.columns else None,
                    marginal="box",
                    color_discrete_sequence=["#10B981", "#EF4444"],
                )
                fig.update_layout(
                    title=f"Distribution of {selected_num_feature}", height=400
                )
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                if target_column in df.columns:
                    # Box plot by target
                    fig = px.box(
                        df,
                        x=target_column,
                        y=selected_num_feature,
                        color=target_column,
                        color_discrete_sequence=["#10B981", "#EF4444"],
                        labels={target_column: "Dropout Status"},
                    )
                    fig.update_layout(
                        title=f"{selected_num_feature} by Dropout Status", height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # T-test or Mann-Whitney U test
                    from scipy.stats import ttest_ind, mannwhitneyu

                    group0 = df[df[target_column] == 0][selected_num_feature].dropna()
                    group1 = df[df[target_column] == 1][selected_num_feature].dropna()

                    # Check if enough data in both groups
                    if len(group0) > 0 and len(group1) > 0:
                        # Mann-Whitney U test (non-parametric)
                        u_stat, p_value = mannwhitneyu(
                            group0, group1, alternative="two-sided"
                        )

                        st.markdown(
                            f"**Mann-Whitney U Test**: U = {u_stat:.2f}, p-value = {p_value:.4f}"
                        )
                        if p_value < 0.05:
                            st.markdown(
                                "🔍 **There is a significant difference in this feature between dropout and non-dropout students.**"
                            )
                else:
                    # Simple stats
                    st.write(df[selected_num_feature].describe())

    else:  # Correlation Analysis
        # Calculate and display correlation matrix
        if len(df.select_dtypes(include=["int64", "float64"]).columns) < 2:
            st.info("Not enough numerical features for correlation analysis.")
        else:
            # Select features for correlation
            numerical_cols = df.select_dtypes(
                include=["int64", "float64"]
            ).columns.tolist()
            selected_features = st.multiselect(
                "Select Features for Correlation Analysis",
                numerical_cols,
                default=numerical_cols[: min(5, len(numerical_cols))],
            )

            if len(selected_features) > 1:
                # Correlation matrix
                corr = df[selected_features].corr()

                fig = px.imshow(
                    corr,
                    text_auto=True,
                    color_continuous_scale="RdBu_r",
                    zmin=-1,
                    zmax=1,
                    aspect="equal",
                )
                fig.update_layout(title="Correlation Matrix", height=600)
                st.plotly_chart(fig, use_container_width=True)

                if target_column in selected_features:
                    # Top correlations with target
                    target_corr = corr[target_column].sort_values(ascending=False)
                    st.markdown("### Top Correlations with Dropout")

                    fig = px.bar(
                        x=target_corr.index,
                        y=target_corr.values,
                        labels={"x": "Feature", "y": "Correlation Coefficient"},
                        color=target_corr.values,
                        color_continuous_scale="RdBu_r",
                        range_color=[-1, 1],
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning(
                    "Please select at least two features for correlation analysis."
                )

    st.markdown("</div>", unsafe_allow_html=True)

    # Feature distribution by target
    if target_column in df.columns:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Key Features by Dropout Status")

        # Determine some interesting features
        numerical_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
        numerical_cols = [
            col
            for col in numerical_cols
            if df[col].nunique() > 10 and col != target_column
        ]

        if numerical_cols:
            # Select top 3 features (or available)
            top_features = numerical_cols[: min(3, len(numerical_cols))]

            for feature in top_features:
                fig = go.Figure()

                # Add histograms for each group
                fig.add_trace(
                    go.Histogram(
                        x=df[df[target_column] == 0][feature],
                        name="Non-Dropout",
                        marker_color="#10B981",
                        opacity=0.7,
                    )
                )

                fig.add_trace(
                    go.Histogram(
                        x=df[df[target_column] == 1][feature],
                        name="Dropout",
                        marker_color="#EF4444",
                        opacity=0.7,
                    )
                )

                fig.update_layout(
                    title=f"Distribution of {feature} by Dropout Status",
                    xaxis_title=feature,
                    yaxis_title="Count",
                    barmode="overlay",
                    height=400,
                )

                st.plotly_chart(fig, use_container_width=True)

        st.markdown("</div>", unsafe_allow_html=True)

# MODEL PERFORMANCE PAGE
elif page == "Model Performance" and model is not None:
    st.markdown(
        '<h2 class="sub-header">Model Performance Analysis</h2>', unsafe_allow_html=True
    )

    if target_column is None:
        st.warning(
            "Please define a target column in the sidebar to evaluate model performance."
        )
    else:
        # Evaluate model on test data
        X = df.drop(target_column, axis=1)
        y = df[target_column]

        from sklearn.model_selection import train_test_split

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Make predictions on test set
        with st.spinner("Evaluating model performance..."):
            try:
                y_pred = model.predict(X_test)
                y_prob = (
                    model.predict_proba(X_test)[:, 1]
                    if hasattr(model, "predict_proba")
                    else None
                )

                # Calculate metrics
                from sklearn.metrics import (
                    accuracy_score,
                    precision_score,
                    recall_score,
                    f1_score,
                    roc_auc_score,
                )

                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred)
                recall = recall_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred)

                if y_prob is not None:
                    roc_auc = roc_auc_score(y_test, y_prob)
                else:
                    roc_auc = None

                # Display metrics
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.markdown(
                        '<div class="card metric-card">', unsafe_allow_html=True
                    )
                    st.metric("Accuracy", f"{accuracy:.2%}")
                    st.markdown("</div>", unsafe_allow_html=True)

                with col2:
                    st.markdown(
                        '<div class="card metric-card">', unsafe_allow_html=True
                    )
                    st.metric("Precision", f"{precision:.2%}")
                    st.markdown("</div>", unsafe_allow_html=True)

                with col3:
                    st.markdown(
                        '<div class="card metric-card">', unsafe_allow_html=True
                    )
                    st.metric("Recall", f"{recall:.2%}")
                    st.markdown("</div>", unsafe_allow_html=True)

                with col4:
                    st.markdown(
                        '<div class="card metric-card">', unsafe_allow_html=True
                    )
                    st.metric("F1 Score", f"{f1:.2%}")
                    st.markdown("</div>", unsafe_allow_html=True)

                # Display confusion matrix
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown("### Confusion Matrix")

                cm = confusion_matrix(y_test, y_pred)

                # Create a more informative confusion matrix
                categories = ["Continuing", "Dropout"]

                fig = go.Figure(
                    data=go.Heatmap(
                        z=cm,
                        x=categories,
                        y=categories,
                        colorscale="Blues",
                        text=cm,
                        texttemplate="%{text}",
                        textfont={"size": 16},
                    )
                )

                fig.update_layout(
                    title="Confusion Matrix",
                    xaxis_title="Predicted Label",
                    yaxis_title="True Label",
                    height=500,
                    width=500,
                )

                col1, col2 = st.columns([1, 1])

                with col1:
                    st.plotly_chart(fig, use_container_width=True)

                with col2:
                    # Interpretation
                    st.markdown("### Interpretation")
                    st.markdown(
                        f"""
                    - **True Negatives**: {cm[0, 0]} students correctly predicted to continue
                    - **False Positives**: {cm[0, 1]} continuing students incorrectly flagged as dropouts
                    - **False Negatives**: {cm[1, 0]} dropouts missed by the model
                    - **True Positives**: {cm[1, 1]} students correctly identified as dropouts
                    
                    **Key Insights:**
                    - The model has {accuracy:.2%} overall accuracy
                    - Of students predicted to drop out, {precision:.2%} actually do (precision)
                    - The model correctly identifies {recall:.2%} of all actual dropouts (recall)
                    """
                    )
                st.markdown("</div>", unsafe_allow_html=True)

                # ROC Curve and PR Curve
                if y_prob is not None:
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    st.markdown("### ROC Curve and Precision-Recall Curve")

                    col1, col2 = st.columns(2)

                    with col1:
                        # ROC Curve
                        from sklearn.metrics import roc_curve

                        fpr, tpr, _ = roc_curve(y_test, y_prob)

                        fig = px.area(
                            x=fpr,
                            y=tpr,
                            labels=dict(
                                x="False Positive Rate", y="True Positive Rate"
                            ),
                            title=f"ROC Curve (AUC={roc_auc:.4f})",
                        )

                        fig.add_shape(
                            type="line", line=dict(dash="dash"), x0=0, x1=1, y0=0, y1=1
                        )

                        fig.update_layout(height=400)
                        st.plotly_chart(fig, use_container_width=True)

                    with col2:
                        # Precision-Recall Curve
                        from sklearn.metrics import precision_recall_curve

                        precision_curve, recall_curve, _ = precision_recall_curve(
                            y_test, y_prob
                        )

                        fig = px.area(
                            x=recall_curve,
                            y=precision_curve,
                            labels=dict(x="Recall", y="Precision"),
                            title="Precision-Recall Curve",
                        )

                        # Add a line for the proportion of positive class
                        baseline = sum(y_test) / len(y_test)
                        fig.add_shape(
                            type="line",
                            line=dict(dash="dash"),
                            x0=0,
                            x1=1,
                            y0=baseline,
                            y1=baseline,
                        )

                        fig.update_layout(height=400)
                        st.plotly_chart(fig, use_container_width=True)

                    st.markdown("</div>", unsafe_allow_html=True)

                # Feature Importance
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown("### Feature Importance")

                try:
                    # Get feature importance if available
                    if hasattr(model, "feature_importances_"):
                        importance = model.feature_importances_
                        feature_names = X.columns
                    elif hasattr(model, "steps") and hasattr(
                        model.steps[-1][1], "feature_importances_"
                    ):
                        # For pipeline with feature_importances_
                        classifier = model.steps[-1][1]
                        importance = classifier.feature_importances_

                        # Get feature names after preprocessing
                        if hasattr(model, "feature_names_in_"):
                            feature_names = model.feature_names_in_
                        else:
                            feature_names = [
                                f"Feature {i}" for i in range(len(importance))
                            ]
                    else:
                        # Use permutation importance
                        from sklearn.inspection import permutation_importance

                        perm_importance = permutation_importance(
                            model, X_test, y_test, n_repeats=10, random_state=42
                        )
                        importance = perm_importance.importances_mean
                        feature_names = X.columns

                    # Create DataFrame for plotting
                    importance_df = pd.DataFrame(
                        {"Feature": feature_names, "Importance": importance}
                    )

                    # Sort by importance
                    importance_df = importance_df.sort_values(
                        "Importance", ascending=False
                    )

                    # Limit to top 15 features for readability
                    importance_df = importance_df.head(15)

                    # Plot
                    fig = px.bar(
                        importance_df,
                        x="Importance",
                        y="Feature",
                        orientation="h",
                        color="Importance",
                        color_continuous_scale="blues",
                        title="Top 15 Feature Importance",
                    )

                    fig.update_layout(height=600)
                    st.plotly_chart(fig, use_container_width=True)

                    # Interpretation
                    st.markdown("### Key Dropout Indicators")
                    st.markdown(
                        f"""
                    Based on the model, the top factors influencing dropout risk are:
                    1. **{importance_df.iloc[0]['Feature']}**
                    2. **{importance_df.iloc[1]['Feature']}**
                    3. **{importance_df.iloc[2]['Feature']}**
                    
                    This suggests that focusing intervention efforts on these areas may help reduce dropout rates.
                    """
                    )

                except Exception as e:
                    st.warning(f"Could not determine feature importance: {e}")

                st.markdown("</div>", unsafe_allow_html=True)

                # Classification Report
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown("### Detailed Classification Report")

                # Generate classification report
                from sklearn.metrics import classification_report

                report = classification_report(
                    y_test,
                    y_pred,
                    target_names=["Continuing", "Dropout"],
                    output_dict=True,
                )

                # Convert to DataFrame for display
                report_df = pd.DataFrame(report).transpose()

                # Display as table
                st.table(
                    report_df.style.format(
                        {
                            "precision": "{:.2%}",
                            "recall": "{:.2%}",
                            "f1-score": "{:.2%}",
                            "support": "{:.0f}",
                        }
                    )
                )

                st.markdown("</div>", unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Error evaluating model: {e}")
                st.exception(e)

# PREDICTION PAGE
elif page == "Prediction" and model is not None:
    st.markdown(
        '<h2 class="sub-header">Individual Student Dropout Prediction</h2>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(
        """
    ### Instructions
    Enter student information below to predict their dropout risk. The model will analyze the data
    and provide a risk assessment along with key contributing factors.
    """
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # Create form for student data input
    st.markdown('<div class="card">', unsafe_allow_html=True)
    with st.form("student_prediction_form"):
        st.markdown("### Student Information")

        # Get columns from the dataframe excluding the target column
        input_columns = [col for col in df.columns if col != target_column]

        # Determine key input fields
        demographic_cols = [
            col
            for col in input_columns
            if any(
                keyword in col.lower()
                for keyword in ["gender", "age", "nationality", "race", "ethnicity"]
            )
        ]
        academic_cols = [
            col
            for col in input_columns
            if any(
                keyword in col.lower()
                for keyword in [
                    "grade",
                    "score",
                    "exam",
                    "curricular",
                    "academic",
                    "gpa",
                ]
            )
        ]

        # Group inputs
        col1, col2 = st.columns(2)

        # Store input values
        input_values = {}

        with col1:
            st.markdown("#### Demographic Information")
            for col in demographic_cols[: min(5, len(demographic_cols))]:
                # Determine input type based on column data
                if df[col].dtype == "object" or df[col].nunique() < 10:
                    # For categorical variables
                    options = list(df[col].dropna().unique())
                    input_values[col] = st.selectbox(f"{col}", options)
                else:
                    # For numerical variables
                    min_val = float(df[col].min())
                    max_val = float(df[col].max())
                    input_values[col] = st.slider(
                        f"{col}", min_val, max_val, (min_val + max_val) / 2
                    )

        with col2:
            st.markdown("#### Academic Information")
            for col in academic_cols[: min(5, len(academic_cols))]:
                # Determine input type based on column data
                if df[col].dtype == "object" or df[col].nunique() < 10:
                    # For categorical variables
                    options = list(df[col].dropna().unique())
                    input_values[col] = st.selectbox(f"{col}", options)
                else:
                    # For numerical variables
                    min_val = float(df[col].min())
                    max_val = float(df[col].max())
                    input_values[col] = st.slider(
                        f"{col}", min_val, max_val, (min_val + max_val) / 2
                    )

        # Additional fields (if any)
        additional_cols = [
            col
            for col in input_columns
            if col not in demographic_cols and col not in academic_cols
        ]

        if additional_cols:
            st.markdown("#### Additional Information")
            for col in additional_cols[: min(5, len(additional_cols))]:
                # Determine input type based on column data
                if df[col].dtype == "object" or df[col].nunique() < 10:
                    # For categorical variables
                    options = list(df[col].dropna().unique())
                    input_values[col] = st.selectbox(f"{col}", options)
                else:
                    # For numerical variables
                    min_val = float(df[col].min())
                    max_val = float(df[col].max())
                    input_values[col] = st.slider(
                        f"{col}", min_val, max_val, (min_val + max_val) / 2
                    )

        # Submit button
        submitted = st.form_submit_button("Predict Dropout Risk")

    st.markdown("</div>", unsafe_allow_html=True)

    # Make prediction when form is submitted
    if submitted:
        # Create input dataframe
        input_df = pd.DataFrame([input_values])

        # Ensure all columns from training data are present
        for col in df.columns:
            if col not in input_df.columns and col != target_column:
                input_df[col] = df[col].mode()[0]  # Fill with most common value

        # Make prediction
        with st.spinner("Analyzing student data..."):
            try:
                # Get probability of dropout
                dropout_prob = model.predict_proba(input_df)[:, 1][0]

                # Determine risk category
                risk_category, risk_color = get_risk_category(dropout_prob)

                # Display prediction
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown("### Prediction Result")

                col1, col2 = st.columns([1, 2])

                with col1:
                    # Gauge chart for risk probability
                    fig = go.Figure(
                        go.Indicator(
                            mode="gauge+number",
                            value=dropout_prob * 100,
                            title={"text": "Dropout Risk"},
                            domain={"x": [0, 1], "y": [0, 1]},
                            gauge={
                                "axis": {"range": [0, 100]},
                                "bar": {"color": risk_color},
                                "steps": [
                                    {"range": [0, 30], "color": "#10B981"},
                                    {"range": [30, 70], "color": "#F59E0B"},
                                    {"range": [70, 100], "color": "#EF4444"},
                                ],
                            },
                        )
                    )

                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)

                with col2:
                    st.markdown(f"### {risk_category}")
                    st.markdown(
                        f"""
                    The student has a **{dropout_prob:.1%}** probability of dropping out.
                    
                    **Risk assessment:**
                    - **Category:** {risk_category}
                    - **Confidence:** {abs((dropout_prob - 0.5) * 2):.1%}
                    """
                    )

                    if risk_category == "High Risk":
                        st.markdown(
                            """
                        **Recommended Actions:**
                        - Schedule immediate academic counseling
                        - Develop personalized study plan
                        - Consider additional tutoring support
                        - Regular check-ins with academic advisor
                        """
                        )
                    elif risk_category == "Medium Risk":
                        st.markdown(
                            """
                        **Recommended Actions:**
                        - Schedule academic counseling
                        - Identify specific academic challenges
                        - Connect with peer support resources
                        - Monthly check-ins with academic advisor
                        """
                        )
                    else:
                        st.markdown(
                            """
                        **Recommended Actions:**
                        - Maintain current academic trajectory
                        - Regular semester check-ins with advisor
                        - Stay engaged with campus resources
                        """
                        )

                st.markdown("</div>", unsafe_allow_html=True)

                # Feature importance for this prediction
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown("### Key Factors Influencing This Prediction")

                try:
                    # Try to use SHAP values for explainability
                    try:
                        import shap

                        explainer = shap.Explainer(model)
                        shap_values = explainer(input_df)

                        # Get feature importance
                        feature_importance = pd.DataFrame(
                            {
                                "Feature": input_df.columns,
                                "Importance": np.abs(shap_values.values[0]),
                                "Value": input_df.values[0],
                                "Direction": np.sign(shap_values.values[0]),
                            }
                        )
                    except:
                        # Fallback to simpler approach (compare with averages)
                        feature_importance = pd.DataFrame(
                            {"Feature": input_df.columns, "Value": input_df.values[0]}
                        )

                        # Compare with average values for dropouts and non-dropouts
                        if target_column in df.columns:
                            dropouts_avg = df[df[target_column] == 1].mean()
                            non_dropouts_avg = df[df[target_column] == 0].mean()

                            # Calculate simple importance (how far from the average of each group)
                            for feature in feature_importance["Feature"]:
                                if (
                                    feature in dropouts_avg
                                    and feature in non_dropouts_avg
                                ):
                                    # Only for numerical features
                                    if pd.api.types.is_numeric_dtype(df[feature]):
                                        value = input_values.get(
                                            feature, df[feature].mean()
                                        )

                                        dist_to_dropout = abs(
                                            value - dropouts_avg[feature]
                                        )
                                        dist_to_non_dropout = abs(
                                            value - non_dropouts_avg[feature]
                                        )

                                        # Importance is the difference between distances
                                        importance = abs(
                                            dist_to_dropout - dist_to_non_dropout
                                        )

                                        # Direction is which group it's closer to
                                        direction = (
                                            -1
                                            if dist_to_dropout < dist_to_non_dropout
                                            else 1
                                        )

                                        feature_importance.loc[
                                            feature_importance["Feature"] == feature,
                                            "Importance",
                                        ] = importance
                                        feature_importance.loc[
                                            feature_importance["Feature"] == feature,
                                            "Direction",
                                        ] = direction

                    # Sort by importance
                    feature_importance = feature_importance.sort_values(
                        "Importance", ascending=False
                    )

                    # Display top factors
                    top_factors = feature_importance.head(5)

                    for i, (_, row) in enumerate(top_factors.iterrows()):
                        if "Direction" in row:
                            direction = (
                                "increases" if row["Direction"] < 0 else "decreases"
                            )
                            st.markdown(
                                f"**{i+1}. {row['Feature']}** = {row['Value']} ({direction} dropout risk)"
                            )
                        else:
                            st.markdown(f"**{i+1}. {row['Feature']}** = {row['Value']}")

                    # Plot
                    if "Importance" in feature_importance.columns:
                        top_factors_plot = feature_importance.head(10)

                        fig = px.bar(
                            top_factors_plot,
                            x="Importance",
                            y="Feature",
                            orientation="h",
                            color=(
                                "Direction"
                                if "Direction" in top_factors_plot.columns
                                else None
                            ),
                            color_discrete_map=(
                                {1: "#10B981", -1: "#EF4444"}
                                if "Direction" in top_factors_plot.columns
                                else None
                            ),
                            title="Top 10 Factors Influencing This Prediction",
                        )

                        fig.update_layout(height=500)
                        st.plotly_chart(fig, use_container_width=True)

                except Exception as e:
                    st.warning(f"Could not analyze feature importance: {e}")

                st.markdown("</div>", unsafe_allow_html=True)

                # Similar students analysis
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown("### Similar Students Analysis")

                try:
                    # Find similar students in the dataset
                    from sklearn.neighbors import NearestNeighbors

                    # Select only numerical columns for similarity
                    numerical_cols = [
                        col
                        for col in df.columns
                        if pd.api.types.is_numeric_dtype(df[col])
                        and col != target_column
                    ]

                    if len(numerical_cols) > 0:
                        # Normalize data
                        from sklearn.preprocessing import StandardScaler

                        scaler = StandardScaler()
                        scaled_data = scaler.fit_transform(df[numerical_cols])

                        # Find similar students
                        nn = NearestNeighbors(n_neighbors=5)
                        nn.fit(scaled_data)

                        # Scale input data
                        input_scaled = scaler.transform(input_df[numerical_cols])

                        # Get nearest neighbors
                        distances, indices = nn.kneighbors(input_scaled)

                        # Get similar students data
                        similar_students = df.iloc[indices[0]]

                        # Display similar students outcome
                        if target_column in similar_students.columns:
                            dropout_count = similar_students[target_column].sum()

                            st.markdown(
                                f"""
                            Among 5 similar students in our database:
                            - **{dropout_count}** dropped out
                            - **{5 - dropout_count}** continued their studies
                            """
                            )

                            # Display outcome distribution
                            fig = px.pie(
                                names=["Continuing", "Dropout"],
                                values=[5 - dropout_count, dropout_count],
                                color_discrete_sequence=["#10B981", "#EF4444"],
                                title="Similar Students Outcomes",
                            )

                            fig.update_layout(height=300)
                            st.plotly_chart(fig, use_container_width=True)

                        # Compare key metrics
                        st.markdown("### How This Student Compares")

                        # Select top features for comparison
                        comparison_features = (
                            feature_importance["Feature"][:3]
                            if "Importance" in feature_importance.columns
                            else numerical_cols[:3]
                        )

                        # Filter to numerical features
                        comparison_features = [
                            feat
                            for feat in comparison_features
                            if pd.api.types.is_numeric_dtype(df[feat])
                        ]

                        if comparison_features:
                            comparison_data = []

                            for feature in comparison_features:
                                if feature in input_df.columns:
                                    # Current student
                                    current_value = input_df[feature].iloc[0]

                                    # Average values
                                    if target_column in df.columns:
                                        dropout_avg = df[df[target_column] == 1][
                                            feature
                                        ].mean()
                                        non_dropout_avg = df[df[target_column] == 0][
                                            feature
                                        ].mean()

                                        comparison_data.append(
                                            {
                                                "Feature": feature,
                                                "Value": current_value,
                                                "Dropout Avg": dropout_avg,
                                                "Non-Dropout Avg": non_dropout_avg,
                                            }
                                        )

                            # Create comparison chart
                            if comparison_data:
                                comp_df = pd.DataFrame(comparison_data)

                                fig = go.Figure()

                                for feature in comp_df["Feature"]:
                                    row = comp_df[comp_df["Feature"] == feature].iloc[0]

                                    fig.add_trace(
                                        go.Bar(
                                            x=[
                                                "This Student",
                                                "Dropout Avg",
                                                "Non-Dropout Avg",
                                            ],
                                            y=[
                                                row["Value"],
                                                row["Dropout Avg"],
                                                row["Non-Dropout Avg"],
                                            ],
                                            name=feature,
                                        )
                                    )

                                fig.update_layout(
                                    title="Key Metrics Comparison",
                                    barmode="group",
                                    height=400,
                                )

                                st.plotly_chart(fig, use_container_width=True)

                except Exception as e:
                    st.warning(f"Could not perform similar students analysis: {e}")

                st.markdown("</div>", unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Error making prediction: {e}")
                st.exception(e)

# BATCH PREDICTION PAGE
elif page == "Batch Prediction" and model is not None:
    st.markdown(
        '<h2 class="sub-header">Batch Dropout Prediction</h2>', unsafe_allow_html=True
    )

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(
        """
    ### Instructions
    Upload a CSV file with student data to make predictions for multiple students at once.
    The file should have the same columns as the training data (except the target column).
    """
    )

    # Show sample of required columns
    st.markdown("### Required Columns")
    st.write(", ".join([col for col in df.columns if col != target_column][:5]) + "...")

    # File uploader
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file is not None:
        try:
            # Load data
            batch_df = pd.read_csv(uploaded_file)

            # Display sample
            st.markdown("### Data Preview")
            st.write(batch_df.head())

            # Check for missing required columns
            missing_cols = [
                col
                for col in df.columns
                if col != target_column and col not in batch_df.columns
            ]

            if missing_cols:
                st.warning(f"Missing columns: {', '.join(missing_cols)}")
                st.markdown("The model will use default values for these columns.")

                # Add missing columns with default values
                for col in missing_cols:
                    batch_df[col] = df[col].mode()[0]  # Use most common value

            # Make predictions button
            if st.button("Run Batch Prediction"):
                with st.spinner("Processing batch predictions..."):
                    # Make predictions
                    try:
                        # Add student index if not present
                        if "student_id" not in batch_df.columns:
                            batch_df["student_id"] = [
                                f"S{i+1:04d}" for i in range(len(batch_df))
                            ]

                        # Make predictions
                        predictions = model.predict(
                            batch_df.drop("student_id", axis=1)
                            if "student_id" in batch_df.columns
                            else batch_df
                        )

                        # Get probabilities if available
                        if hasattr(model, "predict_proba"):
                            probabilities = model.predict_proba(
                                batch_df.drop("student_id", axis=1)
                                if "student_id" in batch_df.columns
                                else batch_df
                            )[:, 1]
                        else:
                            probabilities = (
                                predictions  # Fallback to binary predictions
                            )

                        # Create results dataframe
                        results_df = pd.DataFrame(
                            {
                                "Student ID": (
                                    batch_df["student_id"]
                                    if "student_id" in batch_df.columns
                                    else [f"S{i+1:04d}" for i in range(len(batch_df))]
                                ),
                                "Dropout Probability": probabilities,
                                "Predicted Status": [
                                    "Dropout" if p == 1 else "Continuing"
                                    for p in predictions
                                ],
                                "Risk Category": [
                                    get_risk_category(p)[0] for p in probabilities
                                ],
                            }
                        )

                        # Display results
                        st.markdown("### Prediction Results")

                        # Create expandable sections by risk category
                        high_risk = results_df[
                            results_df["Risk Category"] == "High Risk"
                        ]
                        medium_risk = results_df[
                            results_df["Risk Category"] == "Medium Risk"
                        ]
                        low_risk = results_df[results_df["Risk Category"] == "Low Risk"]

                        col1, col2, col3 = st.columns(3)

                        with col1:
                            st.markdown(
                                f"<h3 style='color: #EF4444;'>High Risk: {len(high_risk)}</h3>",
                                unsafe_allow_html=True,
                            )

                        with col2:
                            st.markdown(
                                f"<h3 style='color: #F59E0B;'>Medium Risk: {len(medium_risk)}</h3>",
                                unsafe_allow_html=True,
                            )

                        with col3:
                            st.markdown(
                                f"<h3 style='color: #10B981;'>Low Risk: {len(low_risk)}</h3>",
                                unsafe_allow_html=True,
                            )

                        # Display all results
                        st.dataframe(
                            results_df.sort_values(
                                "Dropout Probability", ascending=False
                            )
                        )

                        # Create downloadable CSV
                        st.markdown(
                            get_csv_download_link(
                                results_df,
                                "dropout_predictions.csv",
                                "Download Predictions CSV",
                            ),
                            unsafe_allow_html=True,
                        )

                        # Visualize results
                        col1, col2 = st.columns(2)

                        with col1:
                            # Risk distribution
                            fig = px.pie(
                                results_df,
                                names="Risk Category",
                                color="Risk Category",
                                color_discrete_map={
                                    "Low Risk": "#10B981",
                                    "Medium Risk": "#F59E0B",
                                    "High Risk": "#EF4444",
                                },
                                title="Risk Distribution",
                            )

                            fig.update_layout(height=400)
                            st.plotly_chart(fig, use_container_width=True)

                        with col2:
                            # Probability distribution
                            fig = px.histogram(
                                results_df,
                                x="Dropout Probability",
                                color="Risk Category",
                                color_discrete_map={
                                    "Low Risk": "#10B981",
                                    "Medium Risk": "#F59E0B",
                                    "High Risk": "#EF4444",
                                },
                                nbins=20,
                                title="Dropout Probability Distribution",
                            )

                            fig.update_layout(height=400)
                            st.plotly_chart(fig, use_container_width=True)

                        # Generate intervention recommendations
                        st.markdown("### Intervention Recommendations")

                        st.markdown(
                            """
                        Based on the prediction results, we recommend the following interventions:
                        
                        **High Risk Students:**
                        - Immediate academic counseling (within 1 week)
                        - Personalized study plan development
                        - Weekly check-ins with academic advisor
                        - Consider financial aid and support services evaluation
                        
                        **Medium Risk Students:**
                        - Academic counseling within 2-3 weeks
                        - Connect with peer mentoring programs
                        - Biweekly check-ins with academic advisor
                        - Study skills workshops
                        
                        **Low Risk Students:**
                        - Regular semester check-ins
                        - Provide information about campus resources
                        - Encourage participation in student activities
                        """
                        )

                    except Exception as e:
                        st.error(f"Error making batch predictions: {e}")
                        st.exception(e)

        except Exception as e:
            st.error(f"Error loading file: {e}")
            st.exception(e)

    st.markdown("</div>", unsafe_allow_html=True)

# ABOUT PAGE
elif page == "About":
    st.markdown(
        '<h2 class="sub-header">About This Dashboard</h2>', unsafe_allow_html=True
    )

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(
        """
    ### Student Dropout Prediction Dashboard
    
    This dashboard was developed to help Jaya Jaya Institut identify and support students at risk of dropping out.
    By leveraging machine learning techniques, the system analyzes various student attributes and academic performance
    indicators to predict the likelihood of a student discontinuing their studies.
    
    **Key Features:**
    - Data exploration and visualization
    - Machine learning model building and evaluation
    - Individual student dropout risk prediction
    - Batch prediction for multiple students
    - Intervention recommendations based on risk levels
    
    **How It Works:**
    1. Student data is collected and processed
    2. A machine learning model is trained on historical data
    3. The model identifies patterns associated with dropout behavior
    4. New student data is evaluated against these patterns
    5. Risk scores and recommendations are generated
    
    **Benefits:**
    - Early identification of at-risk students
    - Targeted intervention strategies
    - Improved student retention
    - Data-driven decision making
    - Resource optimization for student support services
    """
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(
        """
    ### Using This Dashboard
    
    **For Administrators:**
    - Use the "Data Exploration" page to understand factors affecting dropout rates
    - Review the "Model Performance" page to validate the prediction accuracy
    - Use the "Batch Prediction" feature at the beginning of each semester
    
    **For Academic Advisors:**
    - Use the "Prediction" page for individual student consultations
    - Review risk factors specific to each student
    - Use the recommendations to develop personalized support plans
    
    **For IT Support:**
    - The dashboard automatically saves trained models
    - Regular data updates will improve prediction accuracy over time
    - Contact technical support for assistance with data integration or model updates
    """
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # Add a footer
    st.markdown('<div class="footer">', unsafe_allow_html=True)
    st.markdown("© 2025 Jaya Jaya Institut. All Rights Reserved.")
    st.markdown("For technical support, please contact the IT Department.")
    st.markdown("</div>", unsafe_allow_html=True)

# No model available
elif model is None and page not in ["Overview", "Data Exploration", "About"]:
    st.warning(
        "Please train a model first using the 'Train New Model' button in the sidebar."
    )

# Footer
st.markdown('<div class="footer">', unsafe_allow_html=True)
st.markdown("Student Dropout Prediction Dashboard | Developed for Jaya Jaya Institut")
st.markdown("Last updated: May 2025")
st.markdown("</div>", unsafe_allow_html=True)
