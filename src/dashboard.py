"""
dashboard.py
NYC Taxi Trip Duration Prediction
"""

import streamlit as st
import joblib
import pandas as pd

# =========================
# PATH
# =========================

MODEL_PATH    = "models/linear_regression.pkl"
ZONE_PATH     = "data/taxi_zone_lookup.csv"
DIST_PATH     = "data/zone_distances.csv"        # ← hasil build_distance_matrix.py

# =========================
# LOAD MODEL
# =========================

model = joblib.load(MODEL_PATH)

# =========================
# LOAD DATA
# =========================

@st.cache_data
def load_data():
    zone_df = pd.read_csv(ZONE_PATH)
    zone_df["display_name"] = zone_df["Borough"] + " - " + zone_df["Zone"]

    # Tabel jarak pra-komputasi (100% coverage, jarak jalan nyata via OSRM)
    dist_df = pd.read_csv(
        DIST_PATH,
        dtype={"PULocationID": int, "DOLocationID": int, "distance_miles": float}
    )

    # Index untuk lookup O(1)
    dist_index = dist_df.set_index(["PULocationID", "DOLocationID"])["distance_miles"]

    return zone_df, dist_index


zone_df, dist_index = load_data()

# =========================
# HELPER: LOOKUP JARAK
# =========================

def get_distance(pu_id: int, do_id: int) -> tuple[float, str]:
    """
    Return (distance_miles, source_label).
    Selalu tersedia karena coverage 100%.
    """
    try:
        miles = dist_index.loc[(pu_id, do_id)]
        return float(miles), "Haversine estimation"
    except KeyError:
        # Seharusnya tidak terjadi setelah matrix dibangun lengkap,
        # tapi tetap ada fallback aman
        avg = dist_index.mean()
        return float(avg), "global average (fallback)"

# =========================
# DASHBOARD UI
# =========================

st.title("NYC Taxi Trip Duration Prediction")

st.write(
    "Masukkan zona asal, zona tujuan, waktu, dan kondisi cuaca "
    "untuk memprediksi durasi perjalanan taksi."
)

zone_options = zone_df["display_name"].tolist()

pickup_zone  = st.selectbox("Pickup Zone",  zone_options)
dropoff_zone = st.selectbox("Dropoff Zone", zone_options)

pickup_hour = st.number_input(
    "Pickup Hour (0-23)", min_value=0, max_value=23, value=18
)

temperature   = st.number_input("Temperature (°C)",   value=8.0)
precipitation = st.number_input("Precipitation (mm)",  min_value=0.0, value=0.0)

# =========================
# CONVERT ZONE NAME → ID
# =========================

pu_location = int(
    zone_df.loc[zone_df["display_name"] == pickup_zone,  "LocationID"].values[0]
)
do_location = int(
    zone_df.loc[zone_df["display_name"] == dropoff_zone, "LocationID"].values[0]
)

# =========================
# LOOKUP JARAK (AKURAT)
# =========================

estimated_distance, dist_source = get_distance(pu_location, do_location)

st.info(
    f"Estimated Trip Distance: **{estimated_distance:.2f} miles**  \n"
    f"<small>Source: {dist_source}</small>",
    icon="🗺️"
)

# =========================
# PREDICTION
# =========================

if st.button("Predict Duration"):
    input_data = pd.DataFrame([{
        "pickup_hour":    pickup_hour,
        "trip_distance":  estimated_distance,
        "PULocationID":   pu_location,
        "DOLocationID":   do_location,
        "temperature_2m": temperature,
        "precipitation":  precipitation,
    }])

    prediction = model.predict(input_data)[0]
    prediction = max(1.0, prediction)

    st.success(f"Predicted Trip Duration: {prediction:.2f} minutes")