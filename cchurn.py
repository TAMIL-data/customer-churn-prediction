import pandas as pd
import joblib
import streamlit as st

import traceback

try:
    model = joblib.load("churn_model.pkl")
except Exception:
    st.code(traceback.format_exc())
scaler = joblib.load("scaler.pkl")
features = joblib.load("features.pkl")
st.title("Customer churn prediction 📊")
st.write("Enter customer details ")

customer_id      = st.text_input("Enter your id ")
contract         = st.selectbox("contract type",["Month-to-month", "One year", "Two year"])
SeniorCitizen    = st.selectbox("SeniorCitizen",["YES","NO"])
tenure           = st.number_input("Enter your tenure,(months)",min_value = 0)
internet         = st.selectbox("internet service type",["Fiber optic","DSL","NO"])
MonthlyCharges   = st.number_input("Monthly Charges",min_value = 0)
TotalCharges     = st.number_input("Total Charges",min_value = 0)

if st.button("Check Churn"):
    
    X = pd.DataFrame(0, index =[0], columns=features) 

    X["SeniorCitizen"] = 1 if SeniorCitizen == "YES" else 0 
    X ["tenure"]= tenure
    X ["MonthlyCharges"]=MonthlyCharges
    X["TotalCharges"]=TotalCharges

    if MonthlyCharges == 0 or TotalCharges == 0:
        st.warning("Please enter valid charges")
        st.stop()


    if contract =="One year":
        X["Contract_One year"] = 1
    elif contract =="Two year":
        X["Contract_Two year"] = 1   

       
    if internet == "DSL":
        X["InternetService_DSL"] = 1
    elif internet == "Fiber optic":
        X["InternetService_Fiber optic"] = 1 


    X_scaled = scaler.transform(X)    
    proba = model.predict_proba(X_scaled)[0][1]

    st.write("customer id",customer_id)
    st.write("📊 Churn Probability:", round(proba, 2))

    THRESHOLD = 0.30   # business-friendly threshold

    if proba >= THRESHOLD:
        st.error("🚨 Customer WILL Churn!")
    else:
        st.success("✅ Customer WILL NOT Churn")

