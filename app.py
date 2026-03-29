import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score

st.set_page_config(page_title="Student Performance Predictor", layout="centered")

st.title("🎓 Student Performance Prediction (Lasso Regression)")

# Load dataset
@st.cache_data
def load_data():
    return pd.read_xlsx("student_data.xlsx")

data = load_data()

# Show dataset
if st.checkbox("👀 Show Dataset"):
    st.write(data.head())

# Features
features = ["Hours_Studied", "Attendance", "Sleep_Hours", "Previous_Scores", "Internet_Usage"]
target = "Final_Score"

X = data[features]
y = data[target]

# Train model button
if st.button("🚀 Train Model"):

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Model
    model = Lasso(alpha=0.5)
    model.fit(X_train_scaled, y_train)

    # Predict on test set
    y_pred = model.predict(X_test_scaled)

    # Metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    st.subheader("📊 Model Performance")
    st.write(f"**MSE:** {mse:.2f}")
    st.write(f"**R² Score:** {r2:.2f}")

    # Show predictions vs actual
    results = pd.DataFrame({
        "Actual": y_test.values,
        "Predicted": y_pred
    })

    st.subheader("📈 Predictions on Test Data")
    st.write(results.head())

    # Feature importance
    coeff_df = pd.DataFrame({
        "Feature": features,
        "Coefficient": model.coef_
    })

    st.subheader("🔍 Important Features (Lasso)")
    st.write(coeff_df)

    # Store model in session
    st.session_state["model"] = model
    st.session_state["scaler"] = scaler

# ---------- USER PREDICTION ----------
st.subheader("🎯 Try Your Own Prediction")

if "model" in st.session_state:

    if st.checkbox("👉 Enable Custom Prediction"):

        hours = st.number_input("Hours Studied", 0.0)
        attendance = st.number_input("Attendance (%)", 0.0, 100.0)
        sleep = st.number_input("Sleep Hours", 0.0)
        previous = st.number_input("Previous Scores", 0.0)
        internet = st.number_input("Internet Usage (hrs)", 0.0)

        if st.button("Predict Score"):

            input_data = np.array([[hours, attendance, sleep, previous, internet]])
            input_scaled = st.session_state["scaler"].transform(input_data)

            prediction = st.session_state["model"].predict(input_scaled)

            st.success(f"🎉 Predicted Final Score: {prediction[0]:.2f}")

else:
    st.info("⚠️ Please train the model first to enable prediction.")
