import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

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
    data = pd.read_csv("data/data.csv", sep=";")
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
    from model_inference import (
        load_best_model,
        predict_single_student,
        create_example_student,
    )

    # load trained model
    model, model_name = load_best_model()

    st.header("Dropout Risk Prediction")

    st.write("Enter student information to predict dropout risk")

    col1, col2 = st.columns(2)

    with col1:
        scholarship = st.selectbox("Scholarship Holder", ["Yes", "No"])
        daytime_attendance = st.selectbox(
            "Attendance Type", ["Daytime", "Evening"]
        )  # 1 = Daytime, 0 = Evening
        prev_qualification = st.selectbox(
            "Previous Qualification",
            [
                "Secondary education",
                "Higher education - bachelor's degree",
                "Higher education - degree",
                "Higher education - master's",
                "Higher education - doctorate",
                "Frequency of higher education",
                "12th year - not completed",
                "11th year - not completed",
                "Other - 11th year",
                "10th year",
                "10th year - not completed",
                "Basic education 3rd cycle",
                "Basic education 2nd cycle",
                "Technological specialization",
                "Higher education - 1st cycle",
                "Professional technical course",
                "Higher education - master (2nd cycle)",
            ],
        )
        prev_grade = st.slider("Previous Qualification Grade (0–200)", 0, 200, 150)
        admission_grade = st.slider("Admission Grade (10–20)", 10.0, 20.0, 15.0)
        course = st.selectbox(
            "Course",
            [
                "Biofuel Production Technologies",
                "Animation and Multimedia Design",
                "Social Service (evening)",
                "Agronomy",
                "Communication Design",
                "Veterinary Nursing",
                "Informatics Engineering",
                "Equinculture",
                "Management",
                "Social Service",
                "Tourism",
                "Nursing",
                "Oral Hygiene",
                "Advertising and Marketing Management",
                "Journalism and Communication",
                "Basic Education",
                "Management (evening)",
            ],
        )

        # 1st Semester performance
        first_sem_grade = st.slider("1st Semester GPA", 0.0, 20.0, 12.0)
        first_sem_credited = st.slider("1st Sem: Credited Subjects", 0, 10, 5)
        first_sem_enrolled = st.slider("1st Sem: Enrolled Subjects", 0, 10, 6)
        first_sem_evaluations = st.slider("1st Sem: Evaluations Taken", 0, 10, 5)
        first_sem_no_evaluations = st.slider("1st Sem: Evaluations Not Taken", 0, 10, 0)
        first_sem_approved = st.slider("1st Sem: Approved Subjects", 0, 10, 4)

        # 2nd Semester performance
        second_sem_grade = st.slider("2nd Semester GPA", 0.0, 20.0, 12.0)
        second_sem_credited = st.slider("2nd Sem: Credited Subjects", 0, 10, 5)
        second_sem_enrolled = st.slider("2nd Sem: Enrolled Subjects", 0, 10, 6)
        second_sem_evaluations = st.slider("2nd Sem: Evaluations Taken", 0, 10, 5)
        second_sem_no_evaluations = st.slider(
            "2nd Sem: Evaluations Not Taken", 0, 10, 0
        )
        second_sem_approved = st.slider("2nd Sem: Approved Subjects", 0, 10, 4)

        unemployment_rate = st.slider("Unemployment rate", 0, 20, 1)
        inflation_rate = st.slider("Inflation Rate", 0, 10, 1)
        gdp = st.slider("GDP", 0, 10, 1)

    with col2:
        age = st.slider("Age at Enrollment", 17, 60, 19)
        gender = st.selectbox("Gender", ["M", "F"])
        nationality = st.selectbox(
            "Nationality",
            [
                "Portuguese",
                "German",
                "Spanish",
                "Italian",
                "Dutch",
                "English",
                "Lithuanian",
                "Angolan",
                "Cape Verdean",
                "Guinean",
                "Mozambican",
                "Santomean",
                "Turkish",
                "Brazilian",
                "Romanian",
                "Moldova",
                "Mexican",
                "Ukrainian",
                "Russian",
                "Cuban",
                "Colombian",
            ],
        )
        isForeigner = st.selectbox("International Students ?", ["Yes", "No"])
        marital_status = st.selectbox(
            "Marital Status",
            [
                "Single",
                "Married",
                "Widower",
                "Divorced",
                "Facto Union",
                "Legally Separated",
            ],
        )

        tuition_fees_status = st.selectbox("Tuition Fees Up to Date", ["Yes", "No"])
        displaced = st.selectbox("Student Displaced", ["Yes", "No"])
        isDebtor = st.selectbox("Student Is Debtor", ["Yes", "No"])
        educational_struggle = st.selectbox("Need Educational Support", ["Yes", "No"])

        mother_qualification = st.selectbox(
            "Mother's Qualification",
            [
                "Secondary Education",
                "Bachelor's Degree",
                "Degree",
                "Master's",
                "Doctorate",
                "12th Year Not Completed",
                "11th Year Not Completed",
                "10th Year",
                "Basic Education",
                "Unknown",
                "Other",
            ],
        )
        mother_occupation = st.selectbox(
            "Mother's Occupation",
            [
                "Student",
                "Legislative/Executive",
                "Intellectual/Scientific",
                "Technicians",
                "Admin Staff",
                "Service Workers",
                "Skilled Workers",
                "Machine Operators",
                "Elementary",
                "Unemployed",
            ],
        )
        father_qualification = st.selectbox(
            "Father's Qualification",
            [
                "Secondary Education",
                "Bachelor's Degree",
                "Degree",
                "Master's",
                "Doctorate",
                "12th Year Not Completed",
                "11th Year Not Completed",
                "10th Year",
                "Basic Education",
                "Unknown",
                "Other",
            ],
        )
        father_occupation = st.selectbox(
            "Father's Occupation",
            [
                "Student",
                "Legislative/Executive",
                "Intellectual/Scientific",
                "Technicians",
                "Admin Staff",
                "Service Workers",
                "Skilled Workers",
                "Machine Operators",
                "Elementary",
                "Unemployed",
            ],
        )

        application_mode = st.selectbox(
            "Application Mode",
            [
                "1st phase - general",
                "Ord. No. 612/93",
                "Special (Azores)",
                "Other higher courses",
                "Ord. No. 854-B/99",
                "International",
                "Special (Madeira)",
                "2nd phase - general",
                "3rd phase - general",
                "Ord. No. 533-A/99 b2",
                "Ord. No. 533-A/99 b3",
                "Over 23",
                "Transfer",
                "Change of course",
                "Tech diploma",
                "Change of inst/course",
                "Short cycle diploma",
                "Change inst/course (Intl)",
            ],
        )
        application_order = st.slider("Application Order (0 = 1st Choice)", 0, 9, 0)

    if st.button("Predict"):
        # Convert selected data to binary
        daytime_attendance_binary = 1 if daytime_attendance == "Daytime" else 0
        gender_binary = 1 if gender == "M" else 0
        international_binary = 1 if isForeigner == "Yes" else 0
        scholarship_binary = 1 if scholarship == "Yes" else 0
        tuition_status_binary = 1 if tuition_fees_status == "Yes" else 0
        displaced_binary = 1 if displaced == "Yes" else 0
        debtor_binary = 1 if isDebtor == "Yes" else 0
        educational_struggle_binary = 1 if educational_struggle == "Yes" else 0

        # decode data
        marital_status_map = {
            "Single": 1,
            "Married": 2,
            "Widower": 3,
            "Divorced": 4,
            "Facto Union": 5,
            "Legally Separated": 6,
        }
        marital_status_decoded = marital_status_map[marital_status]

        application_mode_map = {
            "1st phase - general": 1,
            "Ord. No. 612/93": 2,
            "Special (Azores)": 5,
            "Other higher courses": 7,
            "Ord. No. 854-B/99": 10,
            "International": 15,
            "Special (Madeira)": 16,
            "2nd phase - general": 17,
            "3rd phase - general": 18,
            "Ord. No. 533-A/99 b2": 26,
            "Ord. No. 533-A/99 b3": 27,
            "Over 23": 39,
            "Transfer": 42,
            "Change of course": 43,
            "Tech diploma": 44,
            "Change of inst/course": 51,
            "Short cycle diploma": 53,
            "Change inst/course (Intl)": 57,
        }
        application_mode_decoded = application_mode_map[application_mode]

        course_map = {
            "Biofuel Production Technologies": 33,
            "Animation and Multimedia Design": 171,
            "Social Service (evening)": 8014,
            "Agronomy": 9003,
            "Communication Design": 9070,
            "Veterinary Nursing": 9085,
            "Informatics Engineering": 9119,
            "Equinculture": 9130,
            "Management": 9147,
            "Social Service": 9238,
            "Tourism": 9254,
            "Nursing": 9500,
            "Oral Hygiene": 9556,
            "Advertising and Marketing Management": 9670,
            "Journalism and Communication": 9773,
            "Basic Education": 9853,
            "Management (evening)": 9991,
        }
        course_decoded = course_map[course]

        prev_qualification_map = {
            "Secondary education": 1,
            "Higher education - bachelor's degree": 2,
            "Higher education - degree": 3,
            "Higher education - master's": 4,
            "Higher education - doctorate": 5,
            "Frequency of higher education": 6,
            "12th year - not completed": 9,
            "11th year - not completed": 10,
            "Other - 11th year": 12,
            "10th year": 14,
            "10th year - not completed": 15,
            "Basic education 3rd cycle": 19,
            "Basic education 2nd cycle": 38,
            "Technological specialization": 39,
            "Higher education - 1st cycle": 40,
            "Professional technical course": 42,
            "Higher education - master (2nd cycle)": 43,
        }
        prev_qualification_decoded = prev_qualification_map[prev_qualification]

        nationality_map = {
            "Portuguese": 1,
            "German": 2,
            "Spanish": 6,
            "Italian": 11,
            "Dutch": 13,
            "English": 14,
            "Lithuanian": 17,
            "Angolan": 21,
            "Cape Verdean": 22,
            "Guinean": 24,
            "Mozambican": 25,
            "Santomean": 26,
            "Turkish": 32,
            "Brazilian": 41,
            "Romanian": 62,
            "Moldova": 100,
            "Mexican": 101,
            "Ukrainian": 103,
            "Russian": 105,
            "Cuban": 108,
            "Colombian": 109,
        }
        nationality_decoded = nationality_map[nationality]

        qualification_map = {
            "Secondary Education": 1,
            "Bachelor's Degree": 2,
            "Degree": 3,
            "Master's": 4,
            "Doctorate": 5,
            "12th Year Not Completed": 9,
            "11th Year Not Completed": 10,
            "10th Year": 14,
            "Basic Education": 19,
            "Unknown": 34,
            "Other": 12,
        }
        mother_qualification_decoded = qualification_map[mother_qualification]
        father_qualification_decoded = qualification_map[father_qualification]

        occupation_map = {
            "Student": 1,
            "Legislative/Executive": 2,
            "Intellectual/Scientific": 3,
            "Technicians": 4,
            "Admin Staff": 5,
            "Service Workers": 6,
            "Skilled Workers": 7,
            "Machine Operators": 8,
            "Elementary": 9,
            "Unemployed": 10,
        }
        mother_occupation_decoded = occupation_map[mother_occupation]
        father_occupation_decoded = occupation_map[father_occupation]

        # Features that would typically influence dropout risk
        student_data = create_example_student(
            marital_status_decoded,
            application_mode_decoded,
            application_order,
            course_decoded,
            prev_qualification_decoded,
            prev_grade,
            mother_qualification_decoded,
            father_qualification_decoded,
            mother_occupation_decoded,
            father_occupation_decoded,
            admission_grade,
            displaced_binary,
            educational_struggle_binary,
            debtor_binary,
            tuition_status_binary,
            gender_binary,
            scholarship_binary,
            age,
            international_binary,
            first_sem_credited,
            first_sem_enrolled,
            first_sem_evaluations,
            first_sem_approved,
            first_sem_grade,
            second_sem_credited,
            second_sem_enrolled,
            second_sem_evaluations,
            second_sem_approved,
            second_sem_grade,
            unemployment_rate,
            inflation_rate,
            gdp,
            nationality_decoded,
            first_sem_no_evaluations,
            daytime_attendance_binary,
            second_sem_no_evaluations,
        )

        result = predict_single_student(model, student_data)

        # Normalize to 0-100 scale
        risk_score = min(100, result["confidence"] * 100)

        # Display results
        st.subheader("Prediction Results")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Dropout Risk Score", f"{risk_score}%")

            if risk_score >= 75:
                st.error("High risk of dropout. Immediate intervention recommended.")
            elif risk_score >= 50:
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
        if risk_score >= 50:
            if first_sem_grade < 10:
                factors.append("Low 1st semester performance")
            if second_sem_grade < 10:
                factors.append("Low 2nd semester performance")
            if admission_grade < 14:
                factors.append("Below average admission grade")
            if scholarship_binary == 0 and isDebtor == "Yes":
                factors.append("Economic challenges")
            if unemployment_rate > 13:
                factors.append("High Unemployment rate")
            if inflation_rate > 2.6:
                factors.append("High Inflation rate")
            if educational_struggle == "Yes":
                factors.append("Need Educational Support")

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
