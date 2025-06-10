import streamlit as st
import pandas as pd
import joblib

st.title("Best Hotel Predictor - Kathmandu, Nepal")

# Load model
model = joblib.load("hotel_booking_predictor.pkl")

# File upload
uploaded_file = st.file_uploader("Upload Hotel Dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    # Preview
    st.subheader("Uploaded Hotel Data")
    st.write(data.head())

    # Check required columns
    required_columns = ['star_rating', 'user_rating', 'price_per_night', 'is_luxury',
                        'has_wifi', 'has_breakfast_included', 'has_free_cancellation',
                        'distance_from_city_center', 'amenities_score', 'location']

    if all(col in data.columns for col in required_columns):
        # Predict total bookings
        predictions = model.predict(data[required_columns])
        data['predicted_bookings'] = predictions

        # Show top 5 hotels
        best_hotels = data.sort_values(by='predicted_bookings', ascending=False).head(5)

        st.subheader("Top Recommended Hotels")
        st.write(best_hotels[['hotel_name', 'location', 'user_rating', 'price_per_night',
                              'amenities_score', 'predicted_bookings']])
    else:
        st.error("Dataset is missing required columns.")
else:
    st.info("Please upload a hotel dataset to get recommendations.")
