# Simple inference script for the best dropout prediction model
import joblib
import pandas as pd
import numpy as np


# Load your best model (update the filename based on your actual saved model)
def load_best_model():
    """Load the best model from your training results"""

    # Try to load different possible model filenames
    possible_filenames = [
        "random_forest_best_model.pkl",
        "gradient_boosting_best_model.pkl",
        "logistic_regression_best_model.pkl",
        "best_model.pkl",
    ]

    for filename in possible_filenames:
        try:
            model = joblib.load(filename)
            print(f"Successfully loaded model: {filename}")
            return model, filename
        except FileNotFoundError:
            continue

    print(
        "No model file found. Make sure to run the training code first and save the model."
    )
    return None, None


def predict_single_student(model, student_data):
    """
    Predict dropout risk for a single student

    Args:
        model: Trained model pipeline
        student_data: Dictionary with student features

    Returns:
        Dictionary with prediction results
    """
    # Convert to DataFrame
    df = pd.DataFrame([student_data])

    # Make prediction
    prediction = model.predict(df)[0]
    probabilities = model.predict_proba(df)[0]

    # Get class labels
    if hasattr(model, "classes_"):
        class_labels = model.classes_
    else:
        class_labels = model.named_steps["classifier"].classes_

    # Create probability dictionary
    prob_dict = {label: prob for label, prob in zip(class_labels, probabilities)}

    # Determine risk level and recommendation
    confidence = np.max(probabilities)

    if prediction == "Dropout":
        risk_level = "⚠️  HIGH RISK"
        recommendation = (
            "Immediate intervention needed! Provide academic counseling and support."
        )
        color = "🔴"
    elif prediction == "Enrolled":
        risk_level = "✅ CONTINUING"
        recommendation = "Student likely to continue. Monitor regularly."
        color = "🟡"
    else:  # Graduate
        risk_level = "🎓 SUCCESS TRACK"
        recommendation = "Student on track for graduation. Maintain current support."
        color = "🟢"

    return {
        "prediction": prediction,
        "risk_level": risk_level,
        "confidence": confidence,
        "recommendation": recommendation,
        "probabilities": prob_dict,
        "color": color,
    }


def create_example_student(
    Marital_status,
    Application_mode,
    Application_order,
    Course,
    Previous_qualification,
    Previous_qualification_grade,
    Mothers_qualification,
    Fathers_qualification,
    Mothers_occupation,
    Fathers_occupation,
    Admission_grade,
    Displaced,
    Educational_special_needs,
    Debtor,
    Tuition_fees_up_to_date,
    Gender,
    Scholarship_holder,
    Age_at_enrollment,
    International,
    Curricular_units_1st_sem_credited,
    Curricular_units_1st_sem_enrolled,
    Curricular_units_1st_sem_evaluations,
    Curricular_units_1st_sem_approved,
    Curricular_units_1st_sem_grade,
    Curricular_units_2nd_sem_credited,
    Curricular_units_2nd_sem_enrolled,
    Curricular_units_2nd_sem_evaluations,
    Curricular_units_2nd_sem_approved,
    Curricular_units_2nd_sem_grade,
    Unemployment_rate,
    Inflation_rate,
    GDP,
    Nacionality,
    Curricular_units_1st_sem_without_evaluations,
    Daytime_evening_attendance,
    Curricular_units_2nd_sem_without_evaluations,
):
    """Create example student data for testing"""
    return {
        "Marital_status": Marital_status,
        "Application_mode": Application_mode,
        "Application_order": Application_order,
        "Course": Course,
        "Previous_qualification": Previous_qualification,
        "Previous_qualification_grade": Previous_qualification_grade,
        "Mothers_qualification": Mothers_qualification,
        "Fathers_qualification": Fathers_qualification,
        "Mothers_occupation": Mothers_occupation,
        "Fathers_occupation": Fathers_occupation,
        "Admission_grade": Admission_grade,
        "Displaced": Displaced,
        "Educational_special_needs": Educational_special_needs,
        "Debtor": Debtor,
        "Tuition_fees_up_to_date": Tuition_fees_up_to_date,
        "Gender": Gender,
        "Scholarship_holder": Scholarship_holder,
        "Age_at_enrollment": Age_at_enrollment,
        "International": International,
        "Curricular_units_1st_sem_credited": Curricular_units_1st_sem_credited,
        "Curricular_units_1st_sem_enrolled": Curricular_units_1st_sem_enrolled,
        "Curricular_units_1st_sem_evaluations": Curricular_units_1st_sem_evaluations,
        "Curricular_units_1st_sem_approved": Curricular_units_1st_sem_approved,
        "Curricular_units_1st_sem_grade": Curricular_units_1st_sem_grade,
        "Curricular_units_2nd_sem_credited": Curricular_units_2nd_sem_credited,
        "Curricular_units_2nd_sem_enrolled": Curricular_units_2nd_sem_enrolled,
        "Curricular_units_2nd_sem_evaluations": Curricular_units_2nd_sem_evaluations,
        "Curricular_units_2nd_sem_approved": Curricular_units_2nd_sem_approved,
        "Curricular_units_2nd_sem_grade": Curricular_units_2nd_sem_grade,
        "Unemployment_rate": Unemployment_rate,
        "Inflation_rate": Inflation_rate,
        "GDP": GDP,
        "Nacionality": Nacionality,
        "Curricular_units_1st_sem_without_evaluations": Curricular_units_1st_sem_without_evaluations,
        "Daytime_evening_attendance": Daytime_evening_attendance,
        "Curricular_units_2nd_sem_without_evaluations": Curricular_units_2nd_sem_without_evaluations,
    }


def dummy_input():
    """Create example student data for testing"""
    return {
        "Marital_status": 1,
        "Application_mode": 7,
        "Application_order": 2,
        "Course": 33,
        "Previous_qualification": 1,
        "Previous_qualification_grade": 160.0,
        "Mothers_qualification": 19,
        "Fathers_qualification": 13,
        "Mothers_occupation": 4,
        "Fathers_occupation": 10,
        "Admission_grade": 142.5,
        "Displaced": 1,
        "Educational_special_needs": 0,
        "Debtor": 0,
        "Tuition_fees_up_to_date": 1,
        "Gender": 1,
        "Scholarship_holder": 0,
        "Age_at_enrollment": 17,
        "International": 0,
        "Curricular_units_1st_sem_credited": 0,
        "Curricular_units_1st_sem_enrolled": 6,
        "Curricular_units_1st_sem_evaluations": 6,
        "Curricular_units_1st_sem_approved": 6,
        "Curricular_units_1st_sem_grade": 13.67,
        "Curricular_units_2nd_sem_credited": 0,
        "Curricular_units_2nd_sem_enrolled": 6,
        "Curricular_units_2nd_sem_evaluations": 6,
        "Curricular_units_2nd_sem_approved": 6,
        "Curricular_units_2nd_sem_grade": 13.75,
        "Unemployment_rate": 10.8,
        "Inflation_rate": 1.4,
        "GDP": 1.74,
        "Nacionality": 1,
        "Curricular_units_1st_sem_without_evaluations": 2,
        "Daytime_evening_attendance": 1,
        "Curricular_units_2nd_sem_without_evaluations": 3,
    }


def print_prediction_results(results):
    """Print formatted prediction results"""
    print("\n" + "=" * 60)
    print("🎓 JAYA JAYA INSTITUT - DROPOUT PREDICTION SYSTEM 🎓")
    print("=" * 60)

    print(f"\n{results['color']} PREDICTION: {results['prediction']}")
    print(f"📊 RISK LEVEL: {results['risk_level']}")
    print(f"🎯 CONFIDENCE: {results['confidence']:.1%}")
    print(f"💡 RECOMMENDATION: {results['recommendation']}")

    print(f"\n📈 DETAILED PROBABILITIES:")
    print("-" * 30)
    for status, prob in results["probabilities"].items():
        bar_length = int(prob * 20)  # Scale to 20 characters
        bar = "█" * bar_length + "░" * (20 - bar_length)
        print(f"{status:12s}: {bar} {prob:.1%}")


# Main execution
if __name__ == "__main__":
    print("Loading best model...")

    # Load the model
    model, model_name = load_best_model()

    if model is None:
        print(
            "❌ Could not load model. Please check that you have trained and saved a model."
        )
        exit()

    print(f"✅ Model loaded: {model_name}")

    # Test with example student
    print("\n🧪 Testing with example student data...")
    example_student = dummy_input()

    try:
        # Make prediction
        results = predict_single_student(model, example_student)

        # Print results
        print_prediction_results(results)

        print(f"\n{'='*60}")
        print("✅ Inference completed successfully!")
        print("💡 To predict for new students, modify the student_data dictionary")
        print("   with the actual student information and run predict_single_student()")

    except Exception as e:
        print(f"❌ Error during prediction: {e}")
        print("Make sure the student data contains all required features.")


# Function to predict multiple students from a DataFrame
def predict_multiple_students(model, students_df):
    """
    Predict for multiple students at once

    Args:
        model: Trained model
        students_df: DataFrame with student data

    Returns:
        DataFrame with predictions added
    """
    predictions = model.predict(students_df)
    probabilities = model.predict_proba(students_df)

    # Add predictions to DataFrame
    result_df = students_df.copy()
    result_df["Predicted_Status"] = predictions
    result_df["Prediction_Confidence"] = np.max(probabilities, axis=1)

    # Add risk levels
    risk_levels = []
    for pred in predictions:
        if pred == "Dropout":
            risk_levels.append("HIGH RISK")
        elif pred == "Enrolled":
            risk_levels.append("CONTINUING")
        else:
            risk_levels.append("SUCCESS TRACK")

    result_df["Risk_Level"] = risk_levels

    return result_df


# Example usage for batch prediction
"""
# To predict for multiple students from a CSV file:

# 1. Load your data
new_students = pd.read_csv('new_students.csv', sep=';')

# 2. Make predictions
model, _ = load_best_model()
if model:
    predictions_df = predict_multiple_students(model, new_students)
    
    # 3. Save results
    predictions_df.to_csv('student_predictions.csv', index=False)
    
    # 4. Print summary
    print("Prediction Summary:")
    print(predictions_df['Risk_Level'].value_counts())
"""
