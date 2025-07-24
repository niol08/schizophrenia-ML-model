import streamlit as st
import numpy as np
import joblib
import pathlib


bundle = joblib.load("schizo_model.pkl")
model = bundle["model"]
encoders = bundle["encoders"]
defaults = bundle["defaults"]
features = bundle["feature_names"]


REQUIRED_FEATURES = {
    "PREMOBD_HX": ("Premorbid History",         "cat"),
    "INSIGHT"   : ("Insight Level",             "cat"),
    "AGE"       : ("Age (years)",               "int"),
    "TH_STRM"   : ("Thought Stream",             "cat"),
    "P_PSY_HX"  : ("Past Psychiatric History",  "cat"),
    "TH_FORM"   : ("Thought Form Abnormalities","cat"),
    "PSE"       : ("Present State Examination", "cat"),
    "SPEECH"    : ("Speech Characteristics",     "cat"),
    "PERCEP"    : ("Perceptual Symptoms",       "cat"),
    "MOOD"      : ("Mood",                      "cat"),
}

ALL_FEATURES = {
    "AGE"       : ("Age (years)",                                      "int"),
    "SEX"       : ("Biological Sex",                                   "cat"),
    "OCCUP"     : ("Occupation",                                       "cat"),
    "MAR_STA"   : ("Marital Status",                                   "cat"),
    "DUR_EPIS"  : ("Episode Duration (months)",                        "float"),
    "P_PSY_HX"  : ("Past Psychiatric History",                         "cat"),
    "P_MED_HX"  : ("Past Medical History",                             "cat"),
    "FAM_P_HX"  : ("Family Psychiatric History",                       "cat"),
    "P_SOC_HX"  : ("Past Social History",                              "cat"),
    "P_SEX_HX"  : ("Past Sexual History",                              "cat"),
    "FOR_HX"    : ("Forensic History",                                 "cat"),
    "PREMOBD_HX": ("Premorbid History",                                "cat"),
    "MSE"       : ("Mental State Examination Summary",                 "cat"),
    "SPEECH"    : ("Speech Characteristics",                           "cat"),
    "MOOD"      : ("Mood",                                             "cat"),
    "AFFECT"    : ("Emotional Affect",                                 "cat"),
    "TH_FORM"   : ("Thought Form Abnormalities",                       "cat"),
    "TH_STRM"   : ("Thought Stream",                                   "cat"),
    "TH_CONTENT": ("Thought Content",                                  "cat"),
    "PERCEP"    : ("Perceptual Symptoms",                              "cat"),
    "ORIENT"    : ("Orientation",                                      "cat"),
    "ATTEN"     : ("Attention",                                        "cat"),
    "CONC"      : ("Concentration",                                    "cat"),
    "MEM_IR"    : ("Memory – Immediate Recall",                        "cat"),
    "MEM_ST"    : ("Memory – Short‑Term",                              "cat"),
    "MEM_LT"    : ("Memory – Long‑Term",                               "cat"),
    "JUDGMT"    : ("Judgment",                                         "cat"),
    "INSIGHT"   : ("Insight Level",                                    "cat"),
    "PSE"       : ("Present State Examination",                        "cat"),
    "EEG"       : ("Electroencephalogram Result",                      "cat"),
}

for feat in features:
    if feat in REQUIRED_FEATURES:
        ALL_FEATURES[feat] = REQUIRED_FEATURES[feat]
    elif feat in encoders:
        label = feat.replace("_", " ").title()
        ALL_FEATURES[feat] = (label, "cat")
    else:
        label = feat.replace("_", " ").title()
        
        if any(tok in feat.upper() for tok in ["AGE", "DUR", "RATE", "SCORE"]):
            ALL_FEATURES[feat] = (label, "float")
        else:
            ALL_FEATURES[feat] = (label, "int")

OPTIONAL_FEATURES = {
    k: v for k, v in ALL_FEATURES.items() if k not in REQUIRED_FEATURES
}

st.set_page_config(page_title="Schizophrenia Predictor", layout="centered")
st.title("Schizophrenia Prediction App")
st.markdown("Provide required patient details and optionally others to predict diagnosis.")

with st.form("schizo_form"):
    st.subheader("Required Patient Inputs")
    user_vals = {}

    for feat, (label, typ) in REQUIRED_FEATURES.items():
        if typ == "cat":
            opts = encoders[feat].classes_
            val = st.selectbox(label, opts, key=feat)
        elif typ == "int":
            val = st.number_input(label, step=1, format="%d", key=feat)
        elif typ == "float":
            val = st.number_input(label, step=0.1, format="%.1f", key=feat)
        user_vals[feat] = val

    st.subheader("Other Features")
    with st.expander("Show optional inputs"):
        for feat, (label, typ) in OPTIONAL_FEATURES.items():
            if typ == "cat":
                opts = encoders[feat].classes_
                val = st.selectbox(label, opts, key=feat+"_opt")
                user_vals[feat] = val
            elif typ == "int":
                val = st.number_input(label, step=1, format="%d", key=feat+"_opt")
                user_vals[feat] = val
            elif typ == "float":
                val = st.number_input(label, step=0.1, format="%.1f", key=feat+"_opt")
                user_vals[feat] = val

    submitted = st.form_submit_button("🧾 Predict")


if submitted:
    vector = []

    for feat in features:
        if feat in user_vals:
            val = user_vals[feat]
            if feat in encoders:
                
                if isinstance(val, str):
                    val = encoders[feat].transform([val])[0]
        else:
            val = defaults[feat]
        vector.append(val)

    input_array = np.array(vector).reshape(1, -1)
    pred = model.predict(input_array)[0]
    proba = model.predict_proba(input_array)[0]

    schiz_code = encoders["CLASS"].transform(["SCHIZ"])[0]
    confidence = proba[list(model.classes_).index(schiz_code)] * 100

    if pred == schiz_code:
        st.success(f"🧠 **SCHIZOPHRENIA** predicted (Confidence: {confidence:.2f}%)")
    else:
        st.info(f"Others (Confidence SCHIZOPHRENIA: {confidence:.2f}%)")
