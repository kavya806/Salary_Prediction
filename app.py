import streamlit as st
import joblib
import pandas as pd

# Load trained model
model = joblib.load("model/salary_prediction_model.pkl")

# Page title
st.title("Salary Prediction App")

st.write("Enter the details below to predict the salary.")

# Inputs
age = st.number_input(
    "Age",
    min_value=18,
    max_value=70,
    value=25
)

education = st.selectbox(
    "Education Level",
    ["High School", "Bachelor's", "Master's", "PhD"]
)

experience = st.number_input(
    "Years of Experience",
    min_value=0,
    max_value=50,
    value=2
)

# Feature engineering
experience_age_ratio = experience / age

# Create input data
input_data = pd.DataFrame({
    "Age": [age],
    "Education Level": [education],
    "Years of Experience": [experience],
    "Experience_Age_Ratio": [experience_age_ratio]
})

# Prediction
if st.button("Predict Salary"):

    prediction = model.predict(input_data)

    st.success(
        f"Predicted Salary: ${prediction[0]:,.2f}"
    )