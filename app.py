import streamlit as st
import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score

st.set_page_config(page_title="Student Exam Score Predictor", layout="centered")

st.title("🎓 Student Exam Score Prediction (Lasso Regression)")

# -------- LOAD DATA --------
@st.cache_data
def load_data():
    if "student_exam_scores.csv" in os.listdir():
        return pd.read_csv("student_exam_scores.csv")
    else:
        st.error("❌ Dataset not found")
        st.stop()

data = load_data()

st.subheader("📊 Dataset Preview")
st.write(data.head())

# -------- FEATURES & TARGET --------
target = "exam_score"

# Automatically take all other columns as features
features = [col for col in data.columns if col != target]

X = data[features]
y = data[target]

# Ensure numeric
X = X.apply(pd.to_numeric, errors='coerce')
X = X.fillna(0)

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

# -------- PERFORMANCE --------
st.subheader("📊 Model Performance")
st.write(f"**MSE:** {mean_squared_error(y_test, y_pred):.2f}")
st.write(f"**R² Score:** {r2_score(y_test, y_pred):.2f}")

# -------- DATASET PREDICTIONS --------
st.subheader("📈 Dataset with Predicted Exam Scores")

data["Predicted_Exam_Score"] = model.predict(
    scaler.transform(X)
)

st.write(data.head())

# -------- FEATURE IMPORTANCE --------
st.subheader("🔍 Feature Importance (Lasso)")

coeff_df = pd.DataFrame({
    "Feature": features,
    "Coefficient": model.coef_
})

st.write(coeff_df)

# -------- USER INPUT --------
st.subheader("🎯 Predict Your Own Score")

if st.checkbox("👉 Enter your data"):

    user_input = []

    for feature in features:
        val = st.number_input(f"{feature}", 0.0)
        user_input.append(val)

    if st.button("Predict"):

        input_array = np.array([user_input])
        input_scaled = scaler.transform(input_array)

        prediction = model.predict(input_scaled)

        st.success(f"🎉 Predicted Exam Score: {prediction[0]:.2f}")
