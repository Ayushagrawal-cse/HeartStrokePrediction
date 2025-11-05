import streamlit as st
import pandas as pd
import joblib

# Load model and preprocessing tools
model = joblib.load("Logistic_heart.pkl")
scaler = joblib.load("scaler.pkl")
expected_columns = joblib.load("columns.pkl")

# Page config
st.set_page_config(page_title="Heart Risk Predictor", page_icon="❤️", layout="centered")

# Title and description
st.title("❤️ Heart Disease Risk Predictor")
st.markdown("### Please provide the following health details to assess your heart disease risk.")

# Sidebar for inputs
st.sidebar.header("🩺 Patient Information")

age = st.sidebar.slider("Age", 18, 100, 40)
sex = st.sidebar.radio("Sex", ['M', 'F'], horizontal=True)
chestpain = st.sidebar.selectbox("Chest Pain Type", ['ATA', 'NAP', 'TA', 'ASY'])
restingbp = st.sidebar.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
cholesterol = st.sidebar.number_input("Cholesterol (mg/dl)", 100, 600, 200)
fastingbs = st.sidebar.radio("Fasting Blood Sugar > 120 mg/dl", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
restingecg = st.sidebar.selectbox("Resting ECG", ['Normal', 'ST', 'LVH'])
maxhr = st.sidebar.slider("Max Heart Rate", 60, 220, 150)
exerciseangina = st.sidebar.radio("Exercise-Induced Angina", ['Y', 'N'], horizontal=True)
oldpeak = st.sidebar.slider("Oldpeak (ST Depression)", 0.0, 6.0, 1.0, step=0.1)
stslope = st.sidebar.selectbox("ST Slope", ['Up', 'Flat', 'Down'])

# Predict button
if st.button("🔍 Predict Risk"):
    # Prepare input
    raw_input = {
        'age': age,
        'restingbp': restingbp,
        'cholesterol': cholesterol,
        'fastingbs': fastingbs,
        'maxhr': maxhr,
        'oldpeak': oldpeak,
        'sex' + sex: 1,
        'chestpain' + chestpain: 1,
        'restingECG' + restingecg: 1,
        'exerciseAngina' + exerciseangina: 1,
        'st_slope' + stslope: 1
    }

    input_df = pd.DataFrame([raw_input])
    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[expected_columns]

    # Scale and predict
    scaled_input = scaler.transform(input_df)
    prediction = model.predict(scaled_input)[0]

    # Display result
    st.markdown("---")
    if prediction == 1:
        st.error("⚠️ **High Risk of Heart Disease**\n\nPlease consult a healthcare professional for further evaluation.")
    else:
        st.success("✅ **Low Risk of Heart Disease**\n\nKeep maintaining a healthy lifestyle!")


    st.markdown("💡 _This prediction is based on statistical modeling and should not replace medical advice._")
