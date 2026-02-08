import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the trained model
model_path = hf_hub_download(repo_id="shyam92/TPP", filename="tppshydn.joblib")
model = joblib.load(model_path)

# Streamlit UI
st.title("Tourism Package Prediction")
st.write("""
This application predicts the expected PitchSatisfactionScore of a Tourist Place
based on its characteristics such as DurationOfPitch, NumberOfFollowups, MonthlyIncome,
ProdTaken, CityTier, ProductPitched, and OwnCar.
Please enter the app details below to get a revenue prediction.
""")

# User input
'DurationOfPitch', 'NumberOfFollowups', 'MonthlyIncome'
ProdTaken = st.selectbox("ProdTaken", ["Yes", "No"])
CityTier = st.selectbox("CityTier", ["Tier 1", "Tier 2", "Tier 3"])
ProductPitched = st.selectbox("ProductPitched", ["Package 1", "Package 2", "Package 3"])
OwnCar= st.selectbox("OwnCar", ["Yes", "No"])

MonthlyIncome = st.number_input("Price (USD)", min_value=0.0, max_value=100.0, value=0.0, step=0.1)
DurationOfPitch = st.number_input("Pitch Duration",min_value=0.0, max_value=100.0, value=0.0, step=0.1)
NumberOfFollowups = st.number_input("Followups", min_value=0.0, max_value=100.0, value=0.0, step=0.1 )

# Assemble input into DataFrame
input_data = pd.DataFrame([{
    'ProdTaken': ProdTaken,
    'CityTier': CityTier,
    'ProductPitched': ProductPitched,
    'OwnCar': OwnCar,
    'DurationOfPitch': DurationOfPitch,
    'NumberOfFollowups': NumberOfFollowups,
    'MonthlyIncome': MonthlyIncome

}])

# Predict button
if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    st.subheader("Prediction Result:")
    st.success(f"Estimated SatisfactionScore : ${prediction:,.2f}")
