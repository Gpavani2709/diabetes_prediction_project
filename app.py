import streamlit as st
import pickle
import numpy as np

model = pickle.load(open('model.pkl', 'rb'))

st.title("Diabetes Prediction App")

# Input fields
preg = st.number_input("Pregnancies", 0)
glu = st.number_input("Glucose", 0)
bp = st.number_input("Blood Pressure", 0)
skin = st.number_input("Skin Thickness", 0)
insulin = st.number_input("Insulin", 0)
bmi = st.number_input("BMI", 0.0)
dpf = st.number_input("Diabetes Pedigree Function", 0.0)
age = st.number_input("Age", 1)

if st.button("Predict"):
    data = np.array([[preg, glu, bp, skin, insulin, bmi, dpf, age]])
    prediction = model.predict(data)[0]
    st.success("Result: Diabetic" if prediction == 1 else "Result: Not Diabetic")
