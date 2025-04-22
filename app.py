
import streamlit as st
import pickle
import pandas as pd

# Load the trained model
with open('student_performance_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Streamlit UI
st.title("Student Performance Analysis")
st.write("This app predicts student performance based on various factors.")

# Example input fields
school = st.selectbox("School", ["GP", "MS"])
sex = st.selectbox("Sex", ["F", "M"])
age = st.slider("Age", 15, 20, 18)
address = st.selectbox("Address", ["U", "R"])
famsize = st.selectbox("Family Size", ["GT3", "LE3"])

# Encoding the inputs for prediction
input_data = pd.DataFrame([[school, sex, age, address, famsize]], columns=["school", "sex", "age", "address", "famsize"])
input_data["school"] = input_data["school"].map({"GP": 0, "MS": 1})
input_data["sex"] = input_data["sex"].map({"F": 0, "M": 1})
input_data["address"] = input_data["address"].map({"U": 0, "R": 1})
input_data["famsize"] = input_data["famsize"].map({"GT3": 0, "LE3": 1})

# Make prediction based on input
prediction = model.predict(input_data)

st.write(f"Predicted Student Performance (G3): {prediction[0]}")
