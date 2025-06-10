import streamlit as st
import joblib
import numpy as np

# Load saved model pipeline
model = joblib.load('hotel_booking_predictor.pkl')

st.title("Nepal Kathmandu Hotel Booking Predictor")

# Collect user inputs
star_rating = st.slider("Star Rating", 1, 5, 3)
user_rating = st.slider("User Rating", 1.0, 10.0, 7.5, 0.1)
price_per_night = st.number_input("Price per Night (NPR)", min_value=0)
is_luxury = st.selectbox("Is Luxury Hotel?", ["No", "Yes"])
has_wifi = st.selectbox("Has Wi-Fi?", ["No", "Yes"])
has_breakfast = st.selectbox("Has Breakfast Included?", ["No", "Yes"])
has_cancellation = st.selectbox("Has Free Cancellation?", ["No", "Yes"])
distance_from_city_center = st.number_input("Distance from City Center (km)", min_value=0.0, step=0.1)
amenities_score = st.slider("Amenities Score", 0, 100, 50)
location = st.selectbox("Location", model.named_steps['preprocessor'].transformers_[0][1].categories_[0])

# Map Yes/No to 1/0
def yes_no_to_int(x):
    return 1 if x == "Yes" else 0

input_data = {
    'star_rating': star_rating,
    'user_rating': user_rating,
    'price_per_night': price_per_night,
    'is_luxury': yes_no_to_int(is_luxury),
    'has_wifi': yes_no_to_int(has_wifi),
    'has_breakfast_included': yes_no_to_int(has_breakfast),
    'has_free_cancellation': yes_no_to_int(has_cancellation),
    'distance_from_city_center': distance_from_city_center,
    'amenities_score': amenities_score,
    'location': location
}

# Convert input to DataFrame
import pandas as pd
input_df = pd.DataFrame([input_data])

if st.button("Predict Bookings"):
    prediction = model.predict(input_df)[0]
    st.success(f"Predicted Total Bookings: {int(prediction)}")

