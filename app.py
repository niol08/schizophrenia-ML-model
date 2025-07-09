import streamlit as st
import joblib
import numpy as np

# Load the model and encoders
bundle = joblib.load('schizo_model.pkl')
model = bundle['model']
encoders = bundle['encoders']

st.title("Schizophrenia Prediction App")

# --- Feature list for reference ---
feature_names = [
    'AGE', 'DUR_EPIS', 'SEX', 'OCCUP', 'MAR_STA',
    'P_PSY_HX', 'FAM_P_HX', 'P_SOC_HX', 'EEG', 'INSIGHT'
]

# Collect user inputs
inputs = []

# AGE
age = st.number_input("Age", min_value=1, max_value=100, value=30)
inputs.append(age)

# DURATION OF EPISODE
duration = st.number_input("Duration of Episode (months)", min_value=0.0, max_value=120.0, value=1.0)
inputs.append(duration)

# SEX
sex = st.selectbox("Sex", encoders['SEX'].classes_)
sex_encoded = encoders['SEX'].transform([sex])[0]
inputs.append(sex_encoded)

# OCCUPATION
occup = st.selectbox("Occupation", encoders['OCCUP'].classes_)
occup_encoded = encoders['OCCUP'].transform([occup])[0]
inputs.append(occup_encoded)

# MARITAL STATUS
marsta = st.selectbox("Marital Status", encoders['MAR_STA'].classes_)
marsta_encoded = encoders['MAR_STA'].transform([marsta])[0]
inputs.append(marsta_encoded)

# P_PSY_HX
ppsy = st.selectbox("Past Psychological History", encoders['P_PSY_HX'].classes_)
ppsy_encoded = encoders['P_PSY_HX'].transform([ppsy])[0]
inputs.append(ppsy_encoded)

# FAM_P_HX
fphy = st.selectbox("Family Psychiatric History", encoders['FAM_P_HX'].classes_)
fphy_encoded = encoders['FAM_P_HX'].transform([fphy])[0]
inputs.append(fphy_encoded)

# P_SOC_HX
psoc = st.selectbox("Past Social History", encoders['P_SOC_HX'].classes_)
psoc_encoded = encoders['P_SOC_HX'].transform([psoc])[0]
inputs.append(psoc_encoded)

# EEG
eeg = st.selectbox("EEG Result", encoders['EEG'].classes_)
eeg_encoded = encoders['EEG'].transform([eeg])[0]
inputs.append(eeg_encoded)

# INSIGHT
insight = st.selectbox("Insight Level", encoders['INSIGHT'].classes_)
insight_encoded = encoders['INSIGHT'].transform([insight])[0]
inputs.append(insight_encoded)

# Convert inputs into array for the model
input_array = np.array(inputs).reshape(1, -1)

# --- Prediction and Confidence ---
if st.button("Predict"):
    pred = model.predict(input_array)[0]
    proba = model.predict_proba(input_array)[0]

    classes = model.classes_
    schizo_index = list(classes).index(1)
    confidence = proba[schizo_index] * 100

    if pred == 1:
        st.success(f"Prediction: SCHIZOPHRENIA (Confidence: {confidence:.2f}%)")
    else:
        st.info(f"Prediction: OTHERS (Confidence of SCHIZ: {confidence:.2f}%)")
