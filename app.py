import streamlit as st
import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score

st.set_page_config(page_title="Student Performance Predictor", layout="centered")

st.title("🎓 Student Performance Prediction (Lasso Regression)")

# -------- LOAD DATA --------
@st.cache_data
def load_data():
    import os
    files = os.listdir()
    
    st.write("📁 Files:", files)

    if "student_performance.xlsx" in files:
        return pd.read_excel("student_performance.xlsx")
    else:
        st.error("❌ File not found. Check file name in GitHub.")
        st.stop()
data = load_data()

st.write("📊 Dataset Preview")
st.write(data.head())

# -------- DEFINE FEATURES --------
features = [
    "attendance_percentage",
    "previous_gpa",
    "study_hours_per_day",
    "assignment_score",
    "midterm_marks"
]

target = "final_result"

X = data[features]
y = data[target]

# -------- TRAIN MODEL --------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = Lasso(alpha=0.5)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

# -------- MODEL PERFORMANCE --------
st.subheader("📊 Model Performance")
st.write(f"**MSE:** {mean_squared_error(y_test, y_pred):.2f}")
st.write(f"**R² Score:** {r2_score(y_test, y_pred):.2f}")

# -------- DATASET PREDICTIONS --------
st.subheader("📈 Predicted Final Scores (Dataset)")

results = X_test.copy()
results["Actual"] = y_test.values
results["Predicted"] = y_pred

st.write(results.head())

# -------- FEATURE IMPORTANCE --------
st.subheader("🔍 Feature Importance")

coeff_df = pd.DataFrame({
    "Feature": features,
    "Coefficient": model.coef_
})

st.write(coeff_df)

# -------- USER PREDICTION --------
st.subheader("🎯 Predict Your Own Score")

if st.checkbox("👉 Enter your data"):

    attendance = st.number_input("Attendance (%)", 0.0, 100.0)
    gpa = st.number_input("Previous GPA", 0.0)
    study = st.number_input("Study Hours per Day", 0.0)
    assignment = st.number_input("Assignment Score", 0.0)
    midterm = st.number_input("Midterm Marks", 0.0)

    if st.button("Predict"):

        input_data = np.array([[attendance, gpa, study, assignment, midterm]])
        input_scaled = scaler.transform(input_data)

        prediction = model.predict(input_scaled)

        st.success(f"🎉 Predicted Final Result: {prediction[0]:.2f}")
