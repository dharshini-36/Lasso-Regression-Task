import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score

st.set_page_config(page_title="Student Performance Predictor", layout="centered")

st.title("🎓 Student Performance Prediction (Lasso Regression)")

# -------- LOAD DATA --------
@st.cache_data
def load_data():
    try:
        return pd.read_csv("student_data.csv")
    except:
        return pd.read_excel("student_data.xlsx")

data = load_data()

# -------- CLEAN COLUMN NAMES (VERY IMPORTANT) --------
data.columns = data.columns.str.strip()
data.columns = data.columns.str.replace(" ", "_")
data.columns = data.columns.str.replace("%", "")

# Debug (optional)
st.write("📌 Columns in dataset:", list(data.columns))

# -------- SHOW DATA --------
if st.checkbox("👀 Show Dataset"):
    st.write(data.head())

# -------- FEATURES --------
features = ["Hours_Studied", "Attendance", "Sleep_Hours", "Previous_Scores", "Internet_Usage"]
target = "Final_Score"

# -------- CHECK IF COLUMNS EXIST --------
missing_cols = [col for col in features + [target] if col not in data.columns]

if missing_cols:
    st.error(f"❌ Missing columns in dataset: {missing_cols}")
    st.stop()

X = data[features]
y = data[target]

# -------- TRAIN MODEL --------
if st.button("🚀 Train Model"):

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

    # -------- PREDICTIONS TABLE --------
    results = pd.DataFrame({
        "Actual": y_test.values,
        "Predicted": y_pred
    })

    st.subheader("📈 Predictions on Test Data")
    st.write(results.head())

    # -------- FEATURE IMPORTANCE --------
    coeff_df = pd.DataFrame({
        "Feature": features,
        "Coefficient": model.coef_
    })

    st.subheader("🔍 Feature Importance (Lasso)")
    st.write(coeff_df)

    # Save model
    st.session_state["model"] = model
    st.session_state["scaler"] = scaler

# -------- USER PREDICTION --------
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
    st.info("⚠️ Please train the model first.")
