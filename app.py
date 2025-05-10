import streamlit as st
import joblib
import pandas as pd
import numpy as np
import math
from sklearn.preprocessing import StandardScaler

# Load model and scaler\
thresholds = joblib.load('threshold_precision.pkl')
model = joblib.load("best_model.pkl") 

# Streamlit UI
st.title(" Loan Approval Prediction App")
st.markdown("""This app predicts whether a loan application is likely to be **approved or not** based on applicant details.
            It uses a trained Machine Learnings model (e.g., Logistic Regression) on realworld loan data.""")
st.markdown("Fill the form below to predict if the loan will be approved:")

#Set minimun loan amount
loan_amount = 1000

## Input Data
with st.form("loan_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        gender = st.selectbox("Gender",["Male","Female"])
        married = st.selectbox("Married",["Yes","No"])
        education = st.selectbox("Education", ["Graduate","Not Graduate"])
        applicant_income = st.number_input("Applicant Income", min_value=1000)
        coapplicant_income = st.number_input("Coapplicant Income", min_value = 0)
        
    with col2: 
        dependents = st.selectbox("Dependents",["0","1","2","3+"])
        self_employed = st.selectbox("Self Employed",["Yes","No"])
        loan_amount = st.number_input("Loan Amount (in 1000s)", min_value=1000)
        loan_term = st.number_input("Loan Term (in days)", min_value =60)
        property_area = st.selectobx("Property Area", ["Urban","Semi Urban","Rural"])
            
    credit_history = st.selectbox("Credit History", ['1','0'])
    
    
    submitted = st.form_submit_button("Predict Loan Status")      
    
# Preprocessing Input 
total_income = applicant_income + coapplicant_income    
income_loan_ratio = total_income / loan_amount
log_loan_amount = math.log(loan_amount)
scaler_arr =np.array([total_income,income_loan_ratio,log_loan_amount]).reshape(-1, 1)
scaler = StandardScaler()
data_arr = scaler.fit_transform(scaler_arr)

input_data = np.array([[
    1 if gender == 'Male' else 0,
    1 if married == 'Yes' else 0,
    1 if education == 'Graduate' else 0,
    1 if self_employed == 'Yes' else 0,
    loan_term,
    credit_history,
    total_income,
    data_arr[0],
    data_arr[1],
    data_arr[2],
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
