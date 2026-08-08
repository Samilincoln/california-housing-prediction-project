import joblib
import pandas as pd
import streamlit as st

model = joblib.load('housing_regression_model.joblib')

st.title("California Housing Regression Model")

med_inc = st.number_input("Median Income", value=3.5)
house_age = st.number_input("House Age", value=25.0)
ave_rooms = st.number_input("Average Rooms", value=5.5)
ave_bedrms = st.number_input("Average Bedrooms", value=1.1)
population = st.number_input("Population", value=1000.0)
ave_occup = st.number_input("Average Occupancy", value=3.0)
latitude = st.number_input("Latitude", value=34.0)
longitude = st.number_input("Longitude", value=-118.0)


if st.button("Predict"):
    prediction = model.predict([[med_inc, house_age, ave_rooms, ave_bedrms, population, ave_occup, latitude, longitude]])
    #st.write(f"Predicted House Price: {prediction[0]:.2f}")
    st.success(f"Predicted House Value: ${prediction[0]*100000:,.2f}")
