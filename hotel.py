import streamlit as st
import joblib
import pandas as pd
import os

# File paths
model_path = 'hotel_booking_predictor.pkl'
dataset_path = 'Kathamandu.csv'

st.title("🏨 Nepal Kathmandu Hotel Booking Predictor & Recommender")

# Check if model file exists
if not os.path.exists(model_path):
    st.error(f"❌ Model file not found: {model_path}")
    st.stop()

# Load model
model = joblib.load(model_path)

# Check if dataset exists
if not os.path.exists(dataset_path):
    st.warning(f"⚠️ Dataset not found: {dataset_path}")
    st.info("Please upload or create a CSV named `Kathamandu.csv` with hotel data.")
    st.stop()

# Load hotel dataset
hotels_df = pd.read_csv(dataset_path)

# Required columns for prediction
required_columns = [
    'hotel_name', 'location', 'star_rating', 'user_rating',
    'price_per_night', 'is_luxury', 'has_wifi', 'has_breakfast_included',
    'has_free_cancellation', 'distance_from_city_center', 'amenities_score'
]

# Check for missing columns
missing_columns = [col for col in required_columns if col not in hotels_df.columns]

if missing_columns:
    st.error(f"🚫 Missing columns in CSV: {', '.join(missing_columns)}")
    st.stop()

# Predict and recommend top 3 hotels
if st.button("🔍 Recommend Top 3 Hotels"):
    try:
        # Predict bookings using the model
        hotels_df['predicted_bookings'] = model.predict(hotels_df[[
            'star_rating', 'user_rating', 'price_per_night', 'is_luxury',
            'has_wifi', 'has_breakfast_included', 'has_free_cancellation',
            'distance_from_city_center', 'amenities_score', 'location'
        ]])
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.stop()

    # Get top 3 hotels with highest predicted bookings
    top_3_hotels = hotels_df.sort_values(by='predicted_bookings', ascending=False).head(3)

    st.success("✅ Top 3 Recommended Hotels:")
    for i, (_, hotel) in enumerate(top_3_hotels.iterrows(), start=1):
        st.markdown(f"""
        ### 🏅 Rank {i}
        - ** Hotel Name:** {hotel['hotel_name']}
        - ** Location:** {hotel['location']}
        - **  User Rating:** {hotel['user_rating']}
        - ** Price per Night:** NPR {hotel['price_per_night']}
        - ** Wi-Fi:** {"Yes" if hotel['has_wifi'] else "No"}
        - ** Breakfast Included:** {"Yes" if hotel['has_breakfast_included'] else "No"}
        - ** Free Cancellation:** {"Yes" if hotel['has_free_cancellation'] else "No"}
        - ** Distance from City Center:** {hotel['distance_from_city_center']} km
        - ** Amenities Score:** {hotel['amenities_score']}
        - ** Predicted Bookings:** {int(hotel['predicted_bookings'])}
        ---
        """)

# Optional: Show full dataset
if st.checkbox("Show full hotel dataset with predicted bookings"):
    if 'predicted_bookings' not in hotels_df.columns:
        try:
            hotels_df['predicted_bookings'] = model.predict(hotels_df[[
                'star_rating', 'user_rating', 'price_per_night', 'is_luxury',
                'has_wifi', 'has_breakfast_included', 'has_free_cancellation',
                'distance_from_city_center', 'amenities_score', 'location'
            ]])
        except Exception as e:
            st.error(f"Failed to predict bookings: {e}")
            st.stop()

    st.dataframe(hotels_df)
