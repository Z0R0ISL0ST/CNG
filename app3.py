import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load trained model
model = joblib.load("h2_cng_model.pkl")

# Streamlit UI
st.title("Hydrogen-CNG Blend Optimization")
st.subheader("Optimize H2-CNG blends for maximum efficiency and low emissions")

# User Inputs
H2_ratio = st.slider("Hydrogen Ratio (%)", 0, 100, 20)
pressure = st.number_input("Pressure (bar)", value=5.0, step=0.1)
temperature = st.number_input("Temperature (°C)", value=300, step=5)

# Predict Output
if st.button("Predict"):
    input_data = np.array([[H2_ratio, pressure, temperature]])
    efficiency, emissions = model.predict(input_data)[0]

    st.write(f"Predicted Efficiency: {efficiency:.2f}%")
    st.write(f"Predicted Emissions: {emissions:.2f} g/km")