#!/usr/bin/env python
# coding: utf-8

# In[29]:


import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load model and scaler
model = joblib.load("logmodel.pkl")
scaler = joblib.load("stscaler.pkl")

st.set_page_config(page_title="Fraud Prediction", page_icon="🤞")

st.title("😵 Fraud Prediction")

st.write("Enter the values below to predict fraud.")

# User Inputs
def label_enco(x):
    if x == 1:
        return "Yes"
    else:
        return "No"

avg_amount_days = st.number_input("avg_amount_days", min_value=0.0, format="%.4f")
is_declined = st.selectbox("is_declined", options=[1,0], format_func=label_enco)
foreign_transaction = st.selectbox("foreign_transaction", options=[1,0], format_func=label_enco)
transaction_amount = st.number_input("transaction_amount", min_value=0.0)
high_risk_countries = st.selectbox("high_risk_countries", options=[1,0], format_func=label_enco)

if st.button("Predict Fraud"):
    # Create a DataFrame matching training features
    input_data = pd.DataFrame([{
        "avg_amount_days": avg_amount_days,
        "is_declined": is_declined,
        "foreign_transaction": foreign_transaction,
        "transaction_amount": transaction_amount,
        "high_risk_countries": high_risk_countries
    }])

    # Scale the input
    cols1=['avg_amount_days','transaction_amount']
    input_data[cols1] = scaler.transform(input_data[cols1])

    # Predict
    prediction = model.predict(input_data)[0]

    # Display Result
    if prediction== 0:
        st.error("🔴 Prediction: fraudulent")
    else:
        st.success("🟢 Prediction: non-fraudulent")


# In[ ]:




