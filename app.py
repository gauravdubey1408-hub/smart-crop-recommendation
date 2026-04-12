import streamlit as st
import pandas as pd
import os
from model import train_model
from weather import get_weather, adjust_for_location

# Cache model
@st.cache_resource
def load_model():
    return train_model()

model, accuracy = load_model()

st.set_page_config(page_title="Smart Crop AI", page_icon="🌾")

st.title("🌾 Smart Crop Recommendation System (AI Pro)")

# ✅ FIXED CSV LOADING (NO ERROR NOW)
base_path = os.path.dirname(__file__)
file_path = os.path.join(base_path, "india_districts.csv")

df_loc = pd.read_csv(file_path)

# ✅ DYNAMIC DROPDOWN
state = st.selectbox("🌍 Select State", sorted(df_loc['state'].unique()))

district = st.selectbox(
    "🏙️ Select District",
    df_loc[df_loc['state'] == state]['district']
)

# INPUTS
rainfall = st.slider("Rainfall (mm)", 0, 300, 150)
N = st.slider("Nitrogen (N)", 0, 100, 50)
P = st.slider("Phosphorus (P)", 0, 100, 50)
K = st.slider("Potassium (K)", 0, 100, 50)
ph = st.slider("Soil pH", 0.0, 14.0, 6.5)

# PREDICTION
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

        st.subheader("🌾 Top 3 Recommended Crops")

        for i in top3_idx:
            st.write(f"👉 {classes[i]} ({probs[i]*100:.2f}%)")

        st.success(f"📊 Model Accuracy: {accuracy*100:.2f}%")
        st.info(f"🌡️ Temp: {temp}°C | 💧 Humidity: {humidity}%")

    except Exception as e:
        st.error(f"❌ Error: {e}")