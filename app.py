import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

st.title("🎓 Student Grade Predictor")

@st.cache_data
def load_data():
    math_df = pd.read_csv("C:/Users/Arbaz Khan/Downloads/student+performance/student/student-mat.csv", sep=';')
    port_df = pd.read_csv("C:/Users/Arbaz Khan/Downloads/student+performance/student/student-por.csv", sep=';')
    merge_columns = ["school", "sex", "age", "address", "famsize", "Pstatus",
                     "Medu", "Fedu", "Mjob", "Fjob", "reason", "nursery", "internet"]
    df = pd.merge(math_df, port_df, on=merge_columns, suffixes=('_math', '_por'))
    return df

df = load_data()

features = ['studytime_python', 'failures_python', 'absences_python']
X = df[features]
y = df['G3_math']

model = LinearRegression()
model.fit(X, y)

study = st.slider("📚 Study Time", 1, 6)
failures = st.slider("❌ Past Class Failures", 0, 4)
absences = st.slider("🏫 School Absences", 0, 100)

if st.button("Predict Final python Grade"):
    prediction = model.predict([[study, failures, absences]])
    st.success(f"🎯 Predicted Grade: {prediction[0]:.2f} / 20")
