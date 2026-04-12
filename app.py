import streamlit as st
import pandas as pd
import os
from datetime import datetime
from model import train_model
from weather import get_weather, adjust_for_location

# ------------------ SEASON FUNCTION ------------------
def get_season(month_num):
    if month_num in [6,7,8,9,10]:
        return "Kharif"
    elif month_num in [10,11,12,1,2,3]:
        return "Rabi"
    else:
        return "Zaid"

# ------------------ RAINFALL ESTIMATION ------------------
def estimate_rainfall(month):
    if month in [6,7,8,9]:  
        return 200
    elif month in [10,11]:
        return 100
    elif month in [12,1,2]:
        return 20
    else:
        return 50

# ------------------ PROFIT DATA ------------------
crop_data = {
    "rice": {"yield": 25, "price": 2000, "cost": 20000},
    "wheat": {"yield": 20, "price": 2200, "cost": 18000},
    "maize": {"yield": 30, "price": 1700, "cost": 15000},
    "cotton": {"yield": 15, "price": 6000, "cost": 25000},
    "mustard": {"yield": 12, "price": 5000, "cost": 15000},
    "watermelon": {"yield": 40, "price": 1000, "cost": 12000},
    "cucumber": {"yield": 35, "price": 1200, "cost": 10000}
}

def calculate_profit(crop):
    data = crop_data.get(crop.lower())
    if not data:
        return None
    revenue = data["yield"] * data["price"]
    profit = revenue - data["cost"]
    return profit

# ------------------ LOAD MODEL ------------------
@st.cache_resource
def load_model():
    return train_model()

model, accuracy = load_model()

# ------------------ UI ------------------
st.set_page_config(page_title="Smart Crop AI", page_icon="🌾", layout="wide")
st.title("🌾 Smart Crop Recommendation System (AUTO + PROFIT AI)")

# ------------------ LOAD LOCATION DATA ------------------
base_path = os.path.dirname(__file__)
file_path = os.path.join(base_path, "india_districts.csv")

df_loc = pd.read_csv(file_path)
df_loc.columns = df_loc.columns.str.strip().str.lower()

# ------------------ DROPDOWN ------------------
state = st.selectbox("🌍 Select State", sorted(df_loc['state'].unique()))

district = st.selectbox(
    "🏙️ Select District",
    sorted(df_loc[df_loc['state'] == state]['district'].unique())
)

# ------------------ AUTO MONTH ------------------
current_month = datetime.now().month
season = get_season(current_month)

st.info(f"📅 Current Month: {current_month}")
st.info(f"🌱 Season: {season}")

# ------------------ AUTO RAINFALL ------------------
rainfall = estimate_rainfall(current_month)

# ------------------ SOIL INPUT ------------------
N = st.slider("Nitrogen (N)", 0, 100, 50)
P = st.slider("Phosphorus (P)", 0, 100, 50)
K = st.slider("Potassium (K)", 0, 100, 50)
ph = st.slider("Soil pH", 0.0, 14.0, 6.5)

# ------------------ PREDICTION ------------------
if st.button("Predict Crop"):
    try:
        temp, humidity = get_weather(district)
        rainfall_adj = adjust_for_location(district, rainfall)

        sample = pd.DataFrame(
            [[temp, rainfall_adj, humidity, N, P, K, ph]],
            columns=['temperature','rainfall','humidity','N','P','K','ph']
        )

        probs = model.predict_proba(sample)[0]
        classes = model.classes_
        top3_idx = probs.argsort()[-3:][::-1]

        # ------------------ SEASON FILTER ------------------
        season_crops = {
            "Kharif": ["rice","maize","cotton"],
            "Rabi": ["wheat","mustard"],
            "Zaid": ["watermelon","cucumber"]
        }

        filtered_results = [
            (classes[i], probs[i])
            for i in top3_idx
            if classes[i].lower() in season_crops[season]
        ]

        # ------------------ OUTPUT ------------------
        st.subheader("🌾 Recommended Crops + Profit")

        results = filtered_results if filtered_results else [(classes[i], probs[i]) for i in top3_idx]

        for crop, prob in results:
            st.write(f"👉 {crop} ({prob*100:.2f}%)")

            profit = calculate_profit(crop)
            if profit:
                st.success(f"💰 Estimated Profit: ₹{profit}")

        st.success(f"📊 Model Accuracy: {accuracy*100:.2f}%")
        st.info(f"🌡️ Temp: {temp}°C | 💧 Humidity: {humidity}%")
        st.info(f"🌧️ Estimated Rainfall: {rainfall_adj} mm")

    except Exception as e:
        st.error(f"❌ Error: {e}")