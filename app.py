import streamlit as st
import numpy as np
import pickle

# Load saved model and scaler
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

# Page Configuration
st.set_page_config(page_title="Placement Predictor", page_icon="🎓", layout="centered")

# Header
st.title("🎓 Student Placement Predictor")
st.write("Predict student placement likelihood based on academic performance and aptitude scores.")
st.markdown("---")

# Input Form
col1, col2 = st.columns(2)

with col1:
    cgpa = st.number_input("Enter CGPA (0.0 - 10.0)", min_value=0.0, max_value=10.0, value=7.0, step=0.1)

with col2:
    iq = st.number_input("Enter IQ Score (40 - 200)", min_value=40, max_value=200, value=100, step=1)

st.markdown("---")

# Predict Button
if st.button("Predict Outcome", use_container_width=True):
    # Scale and predict
    input_data = np.array([[cgpa, iq]])
    scaled_data = scaler.transform(input_data)
    prediction = model.predict(scaled_data)[0]
    probabilities = model.predict_proba(scaled_data)[0]
    
    # Display Result
    if prediction == 1:
        st.success(f"🎉 **High Chance of Placement!** (Confidence: {probabilities[1]*100:.1f}%)")
    else:
        st.error(f"⚠️ **Low Chance of Placement.** (Confidence: {probabilities[0]*100:.1f}%)")