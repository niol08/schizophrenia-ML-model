
import streamlit as st
import joblib
import numpy as np

# Load model and encoders
bundle = joblib.load('schizo_model.pkl')
model = bundle['model']
encoders = bundle['encoders']

st.title("Schizophrenia Prediction App")

# Get the full list of expected features from the model
expected_features = model.feature_names_in_

# Prepare input list for prediction
inputs = []

st.subheader("Enter Patient Details")

for feature in expected_features:
    if feature in encoders:
        # Categorical feature
        options = list(encoders[feature].classes_)
        user_input = st.selectbox(f"{feature}", options, key=feature)
        encoded = encoders[feature].transform([user_input])[0]
        inputs.append(encoded)
    else:
        # Numeric feature
        user_input = st.number_input(f"{feature}", step=1.0, key=feature)
        inputs.append(user_input)

# Convert inputs to array
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
