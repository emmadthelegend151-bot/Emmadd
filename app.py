
%%writefile app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# 1. Load the saved model and scaler
# Note: Ensure these files are uploaded to the same folder on GitHub
model = joblib.load('heart_model.pkl')
scaler = joblib.load('scaler.pkl')

st.set_page_config(page_title="Heart Health AI", layout="wide")

st.title("❤️ Heart Disease Diagnostic Dashboard")
st.markdown("Enter patient clinical data to evaluate risk and receive health recommendations.")

# Sidebar for inputs
with st.sidebar:
    st.header("Patient Clinical Data")
    age = st.number_input("Age", 1, 120, 60)
    sex = st.selectbox("Sex", [0, 1], format_func=lambda x: "Male (1)" if x==1 else "Female (0)")
    cp = st.selectbox("Chest Pain Type (0-3)", [0, 1, 2, 3], index=2)
    trestbps = st.number_input("Resting Blood Pressure", 50, 250, 130)
    chol = st.number_input("Serum Cholesterol", 100, 600, 230)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
    restecg = st.selectbox("Resting ECG (0-2)", [0, 1, 2], index=1)
    thalach = st.number_input("Max Heart Rate", 60, 220, 168)
    exang = st.selectbox("Exercise Induced Angina", [0, 1])
    oldpeak = st.number_input("ST Depression (Oldpeak)", 0.0, 6.0, 1.0)
    slope = st.selectbox("ST Slope (0-2)", [0, 1, 2], index=2)
    ca = st.selectbox("Number of Major Vessels", [0, 1, 2, 3, 4], index=4)
    thal = st.selectbox("Thal (0-3)", [0, 1, 2, 3], index=3)

if st.button("Run Diagnostic Analysis"):
    # Prepare and scale data
    input_data = [[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]]
    scaled_data = scaler.transform(input_data)
    prediction = model.predict(scaled_data)[0]

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Result & Visualization")
        if prediction == 1:
            st.error("### STATUS: High Risk Detected")
        else:
            st.success("### STATUS: Low Risk Detected")

        # Radar Chart Logic
        labels = ['BP', 'Cholesterol', 'Max HR', 'ST Dep.', 'Vessels']
        stats = [trestbps/200, chol/400, thalach/200, oldpeak/6, ca/4]
        healthy = [120/200, 200/400, 150/200, 0.1/6, 0.1/4]
        angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
        stats = np.concatenate((stats, [stats[0]]))
        healthy = np.concatenate((healthy, [healthy[0]]))
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
        ax.fill(angles, stats, color='red', alpha=0.3, label='Patient')
        ax.fill(angles, healthy, color='green', alpha=0.1, label='Healthy Baseline')
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels)
        st.pyplot(fig)

    with col2:
        st.subheader("Personalized Health Advice")
        advice_found = False
        if trestbps > 130: st.warning("**BP:** High. Reduce salt intake.")
        if chol > 200: st.warning("**Cholesterol:** Elevated. Focus on heart-healthy fats.")
        if not advice_found and prediction == 0:
            st.success("All metrics are within healthy ranges.")
