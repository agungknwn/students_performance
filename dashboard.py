import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import joblib
from PIL import Image

# Page config
st.set_page_config(page_title="Student Dropout Prediction", layout="wide")

# Title and description
st.title("Student Dropout Prediction Dashboard")
st.markdown("### Jaya Jaya Institut")
st.write(
    "This dashboard helps identify students at risk of dropping out based on various factors."
)

# Sidebar
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["Overview", "Prediction", "Insights"])


# Load data (in a real app, replace this with your actual data loading)
@st.cache_data
def load_data():
    # This is a placeholder - in real app load your actual data
    # For demo, creating synthetic data based on your script
    np.random.seed(42)
    data = pd.read_csv("data.csv", sep=";")
    return data


data = load_data()

# Overview page
if page == "Overview":
    st.header("Dataset Overview")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Data Sample")
        st.dataframe(data.head())

    with col2:
        st.subheader("Status Distribution")
        fig, ax = plt.subplots(figsize=(8, 6))
        status_counts = data["Status"].value_counts()
        sns.barplot(
            x=status_counts.index,
            y=status_counts.values,
            ax=ax,
            palette=sns.color_palette("Set2", n_colors=len(status_counts)),
        )
        st.pyplot(fig)

    st.subheader("Key Statistics")
    st.dataframe(data.describe())

# Prediction page
elif page == "Prediction":
    st.header("Dropout Risk Prediction")

    st.write("Enter student information to predict dropout risk")

    col1, col2 = st.columns(2)

    with col1:
        prev_grade = st.slider("Previous Qualification Grade", 10.0, 20.0, 15.0)
        admission_grade = st.slider("Admission Grade", 10.0, 20.0, 15.0)
        first_sem_grade = st.slider("1st Semester Grade", 0.0, 20.0, 12.0)
        second_sem_grade = st.slider("2nd Semester Grade", 0.0, 20.0, 12.0)

    with col2:
        age = st.slider("Age at Enrollment", 17, 60, 19)
        gender = st.selectbox("Gender", ["M", "F"])
        scholarship = st.selectbox("Scholarship Holder", ["Yes", "No"])
        economic_factor = st.slider("Economic Factor (1-10)", 1, 10, 5)

    if st.button("Predict"):
        # In a real app, you would load your trained model and make predictions
        # Here we'll simulate a prediction

        # Convert scholarship to binary
        scholarship_binary = 1 if scholarship == "Yes" else 0

        # Features that would typically influence dropout risk
        risk_score = 0
        if first_sem_grade < 10:
            risk_score += 30
        if second_sem_grade < 10:
            risk_score += 25
        if prev_grade < 14:
            risk_score += 10
        if admission_grade < 13:
            risk_score += 15
        if age > 25:
            risk_score += 5
        if scholarship_binary == 0:
            risk_score += 10
        if economic_factor < 4:
            risk_score += 15

        # Normalize to 0-100 scale
        risk_score = min(100, risk_score)

        # Display results
        st.subheader("Prediction Results")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Dropout Risk Score", f"{risk_score}%")

            if risk_score >= 70:
                st.error("High risk of dropout. Immediate intervention recommended.")
            elif risk_score >= 40:
                st.warning("Moderate risk of dropout. Close monitoring recommended.")
            else:
                st.success(
                    "Low risk of dropout. Student likely to continue successfully."
                )

        with col2:
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.barh("Risk", risk_score, color="red")
            ax.barh("Safe", 100 - risk_score, color="green")
            ax.set_xlim(0, 100)
            ax.set_title("Risk Assessment")
            ax.axis("off")
            st.pyplot(fig)

        st.write("Key Factors Contributing to Risk:")
        factors = []
        if first_sem_grade < 10:
            factors.append("Low 1st semester performance")
        if second_sem_grade < 10:
            factors.append("Low 2nd semester performance")
        if admission_grade < 14:
            factors.append("Below average admission grade")
        if scholarship_binary == 0 and economic_factor < 4:
            factors.append("Economic challenges")

        for factor in factors:
            st.write(f"• {factor}")

# Insights page
elif page == "Insights":
    st.header("Data Insights")

    chart_type = st.selectbox(
        "Select Chart Type",
        [
            "Grade Distribution",
            "Academic Performance by Status",
            "Age Distribution",
            "Correlation Matrix",
        ],
    )

    if chart_type == "Grade Distribution":
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(data=data, x="Curricular_units_1st_sem_grade", kde=True, ax=ax)
        st.pyplot(fig)

    elif chart_type == "Academic Performance by Status":
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(x="Status", y="Curricular_units_1st_sem_grade", data=data, ax=ax)
        st.pyplot(fig)

    elif chart_type == "Age Distribution":
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(
            data=data, x="Age_at_enrollment", hue="Status", multiple="stack", ax=ax
        )
        st.pyplot(fig)

    elif chart_type == "Correlation Matrix":
        numeric_cols = data.select_dtypes(include=["float64", "int64"]).columns
        correlation = data[numeric_cols].corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(correlation, annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

    st.subheader("Key Findings")
    st.write(
        """
    - First and second semester academic performance are strong predictors of dropout risk
    - Students with scholarship support are less likely to drop out
    - Admission grade shows moderate correlation with academic success
    - Students who are older at enrollment may face additional challenges
    """
    )

    st.subheader("Recommendations")
    st.write(
        """
    1. Implement early warning system based on first semester performance
    2. Provide targeted academic support for at-risk students
    3. Consider additional financial aid options for students with economic challenges
    4. Develop special mentoring programs for non-traditional students
    """
    )

# Footer
st.sidebar.markdown("---")
st.sidebar.info("Student Dropout Prediction Tool v1.0")
