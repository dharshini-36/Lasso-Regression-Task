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
    files = os.listdir()

    if "student_data.csv" in files:
        return pd.read_csv("student_data.csv")
    elif "student_data.xlsx" in files:
        return pd.read_excel("student_data.xlsx")
    elif "student_performance.xlsx" in files:
        return pd.read_excel("student_performance.xlsx")
    else:
        st.error("❌ Dataset file not found. Upload it to GitHub.")
        st.stop()

data = load_data()

# -------- CLEAN COLUMN NAMES --------
data.columns = data.columns.str.strip()

st.write("📌 Available Columns:", list(data.columns))

# -------- SHOW DATA --------
if st.checkbox("👀 Show Dataset"):
    st.write(data.head())

# -------- SELECT FEATURES --------
st.subheader("🛠️ Select Features and Target")

selected_features = st.multiselect(
    "Select 5 Feature Columns",
    data.columns
)

target = st.selectbox(
    "Select Target Column",
    data.columns
)

# -------- AUTO TRAIN MODEL --------
if len(selected_features) == 5 and target:

    st.success("✅ Model trained successfully!")

    X = data[selected_features]
    y = data[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = Lasso(alpha=0.5)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)

    # -------- METRICS --------
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    st.subheader("📊 Model Performance")
    st.write(f"**MSE:** {mse:.2f}")
    st.write(f"**R² Score:** {r2:.2f}")

    # -------- RESULTS --------
    results = pd.DataFrame({
        "Actual": y_test.values,
        "Predicted": y_pred
    })

    st.subheader("📈 Predictions on Test Data")
    st.write(results.head())

    # -------- FEATURE IMPORTANCE --------
    coeff_df = pd.DataFrame({
        "Feature": selected_features,
        "Coefficient": model.coef_
    })

    st.subheader("🔍 Feature Importance")
    st.write(coeff_df)

    # Save model
    st.session_state["model"] = model
    st.session_state["scaler"] = scaler
    st.session_state["features"] = selected_features

# -------- USER PREDICTION --------
st.subheader("🎯 Try Your Own Prediction")

if "model" in st.session_state:

    inputs = []

    for feature in st.session_state["features"]:
        val = st.number_input(f"{feature}", 0.0)
        inputs.append(val)

    if st.button("Predict Score"):

        input_data = np.array([inputs])
        input_scaled = st.session_state["scaler"].transform(input_data)

        prediction = st.session_state["model"].predict(input_scaled)

        st.success(f"🎉 Predicted Final Score: {prediction[0]:.2f}")

else:
    st.info("⚠️ Please select 5 features and a target to train the model.")
