import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os

# Load model and scaler\
thresholds = joblib.load('optimal_threshold.pkl')
model = joblib.load("random_forest.pkl") 

# Streamlit UI
st.title(" Loan Approval Prediction App")

st.markdown("Fill the form below to predict if the loan will be approved:")

#Set minimun loan amount
loan_amount = 1000

# Input fields
gender = st.selectbox("Gender", ['Male', 'Female'])
married = st.selectbox("Married", ['Yes', 'No'])
education = st.selectbox("Education", ['Graduate', 'Not Graduate'])
self_employed = st.selectbox("Self Employed", ['Yes', 'No'])
applicant_income = st.number_input("Applicant Income", min_value=0)
coapplicant_income = st.number_input("Coapplicant Income", min_value=0)
loan_amount = st.number_input("Loan Amount (in 1000s)", min_value=1000)
loan_term = st.selectbox("Loan Term", [360, 180, 120, 84, 60])
credit_history = st.selectbox("Credit History", [1.0, 0.0])
property_area = st.selectbox("Property Area", ['Urban', 'Semiurban', 'Rural'])
dependents = st.selectbox("Dependents", ['0', '1', '2', '3+'])


total_income = applicant_income + coapplicant_income
income_loan_ratio = total_income / loan_amount


# Preprocess input
input_data = np.array([[
    1 if gender == 'Male' else 0,
    1 if married == 'Yes' else 0,
    1 if education == 'Graduate' else 0,
    1 if self_employed == 'Yes' else 0,
    loan_term,
    credit_history,
    total_income,
    income_loan_ratio,
    loan_amount,
    True if property_area =='Semiurban' else False,
    True if property_area == 'Urban' else False,
    True if dependents == 1 else False,
    True if dependents == 2 else False,
    True if dependents == '3+' else False,
]])

# Apply scaler if used
# input_data = scaler.transform(input_data)

# Prediction
if st.button("Predict Loan Approval"):
    prediction_probs = model.predict_proba(input_data)[:,1]
    prediction = (prediction_probs >= thresholds).astype(int)
    if prediction == 1:
        st.success(" Loan will likely be Approved!")
    else:
        st.error(" Loan will likely NOT be Approved.")
