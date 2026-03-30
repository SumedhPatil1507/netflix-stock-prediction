import streamlit as st
import joblib
import pandas as pd
import os

st.title("📈 Netflix Stock Prediction App")
st.write("Enter stock details")

model_path = os.path.join(os.path.dirname(__file__), "..", "models", "model.pkl")
model = joblib.load(model_path)

# Inputs
open_price = st.number_input("Open Price")
high_price = st.number_input("High Price")
low_price = st.number_input("Low Price")
close_price = st.number_input("Close Price")
volume = st.number_input("Volume")

ma7 = st.number_input("MA7")
ma21 = st.number_input("MA21")
ret = st.number_input("Return")

# Feature dataframe
features = pd.DataFrame([[open_price, high_price, low_price, close_price, volume, ma7, ma21, ret]],
                        columns=['Open','High','Low','Close','Volume','MA7','MA21','Return'])

# Predict
if st.button("Predict"):
    prediction = model.predict(features)[0]
    st.success(f"📈 Predicted Next Value: {prediction:.2f}")