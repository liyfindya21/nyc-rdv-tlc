import os
import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="NYC Taxi Analytics - RDV Project",
    page_icon="🚕",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# PATH CONFIGURATION
# ==========================================
DATA_PATH  = "data/dashboard_sample_2025.parquet" # Pakai file sampel 500rb baris
MODEL_PATH = "models/linear_regression.pkl"
ZONE_PATH  = "data/taxi_zone_lookup.csv"
DIST_PATH  = "data/zone_distances.csv"

# ==========================================
# LOAD DATA & RESOURCES
# ==========================================
@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None

@st.cache_data
def load_resources():
    zone_df = pd.DataFrame()
    dist_index = None
    if os.path.exists(ZONE_PATH):
        zone_df = pd.read_csv(ZONE_PATH)
        zone_df["display_name"] = zone_df["Borough"] + " - " + zone_df["Zone"]
    if os.path.exists(DIST_PATH):
        d_df = pd.read_csv(DIST_PATH, dtype={"PULocationID": int, "DOLocationID": int})
        dist_index = d_df.set_index(["PULocationID", "DOLocationID"])["distance_miles"]
    return zone_df, dist_index

@st.cache_data
def load_dashboard_data():
    if os.path.exists(DATA_PATH):
        df = pd.read_parquet(DATA_PATH)
        # Mapping Wilayah (Borough)
        z_df = pd.read_csv(ZONE_PATH)
        mapping = z_df.set_index('LocationID')['Borough'].to_dict()
        df['pickup_borough'] = df['PULocationID'].map(mapping)
        df['dropoff_borough'] = df['DOLocationID'].map(mapping)
        
        # Kategori Waktu
        def get_time_cat(hour):
            if 5 <= hour < 11: return "Pagi"
            elif 11 <= hour < 17: return "Siang"
            else: return "Malam"
        df['time_category'] = df['pickup_hour'].apply(get_time_cat)
        return df
    return pd.DataFrame()

# Inisialisasi
model = load_model()
zone_df, dist_index = load_resources()
df_real = load_dashboard_data()
zone_options = zone_df["display_name"].tolist() if not zone_df.empty else []

# ==========================================
# SIDEBAR: ML PREDICTOR
# ==========================================
with st.sidebar:
    st.header("🚕 Prediksi Durasi")
    st.write("Gunakan model Linear Regression untuk estimasi waktu.")
    st.markdown("---")
    
    pu_zone = st.selectbox("Pickup Zone", zone_options)
    do_zone = st.selectbox("Dropoff Zone", zone_options, index=min(5, len(zone_options)-1))
    p_hour = st.number_input("Jam (0-23)", 0, 23, 18)
    temp = st.number_input("Suhu (°C)", value=10.0)
    rain = st.number_input("Hujan (mm)", min_value=0.0, value=0.0)

    # Hitung Jarak
    dist_val = 5.0
    if not zone_df.empty and dist_index is not None:
        try:
            p_id = int(zone_df.loc[zone_df["display_name"] == pu_zone, "LocationID"].values[0])
            d_id = int(zone_df.loc[zone_df["display_name"] == do_zone, "LocationID"].values[0])
            dist_val = float(dist_index.loc[(p_id, d_id)])
        except: pass
    
    st.info(f"📍 Jarak: **{dist_val:.2f} miles**")

    if st.button("⚡ Hitung Prediksi", type="primary"):
        if model:
            inp = pd.DataFrame([{"pickup_hour": p_hour, "trip_distance": dist_val, "PULocationID": p_id, "DOLocationID": d_id, "temperature_2m": temp, "precipitation": rain}])
            pred = max(1.0, model.predict(inp)[0])
            st.success(f"### ⏱️ {pred:.1f} Menit")

# ==========================================
# MAIN DASHBOARD
# ==========================================
st.title("📊 NYC Taxi Analytics Dashboard")
st.caption("Analisis data nyata (500,000 sampel) untuk Rekayasa Data & Visualisasi")
st.markdown("---")

if df_real.empty:
    st.error("Data sampel tidak ditemukan! Jalankan script create_sample.py dulu.")
else:
    # 1. KPI Metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Sampel", f"{len(df_real):,} baris")
    m2.metric("Rata-rata Jarak", f"{df_real['trip_distance'].mean():.2f} Mil")
    m3.metric("Rata-rata Durasi", f"{df_real['trip_duration'].mean():.1f} Mnt")
    m4.metric("Rata-rata Tarif", f"${df_real['fare_amount'].mean():.2f}")

    st.markdown("<br>", unsafe_allow_html=True)

    # 2. Heatmap Wilayah & Pie Chart
    col_a, col_b = st.columns([2, 1])

    with col_a:
        st.subheader("🔥 Heatmap Aliran Perjalanan Antar Wilayah (Borough)")
        st.write("Menunjukkan kepadatan rute dari zona asal ke zona tujuan.")
        # Agregasi Borough to Borough
        hm_data = df_real.groupby(['pickup_borough', 'dropoff_borough']).size().unstack(fill_value=0)
        fig_hm = px.imshow(hm_data, text_auto=True, color_continuous_scale="Viridis",
                           labels=dict(x="Tujuan (Dropoff)", y="Asal (Pickup)", color="Jumlah"))
        st.plotly_chart(fig_hm, use_container_width=True)

    with col_b:
        st.subheader("🍕 Share Waktu")
        pie_data = df_real['time_category'].value_counts()
        fig_pie = px.pie(values=pie_data.values, names=pie_data.index, hole=0.4,
                         color_discrete_sequence=px.colors.qualitative.Pastel)
        st.plotly_chart(fig_pie, use_container_width=True)

    # 3. Karakteristik & Tren
    st.markdown("---")
    st.subheader("📋 Ringkasan Karakteristik Perjalanan")
    summ = df_real.groupby('time_category').agg({
        'trip_distance': 'mean',
        'trip_duration': 'mean',
        'fare_amount': 'mean'
    }).reindex(["Pagi", "Siang", "Malam"])
    summ.columns = ["Jarak (Mil)", "Durasi (Menit)", "Tarif (USD)"]
    st.table(summ.style.format("{:.2f}"))

    col_c, col_d = st.columns(2)
    with col_c:
        st.subheader("📈 Tren Jam Sibuk")
        hourly = df_real['pickup_hour'].value_counts().sort_index()
        st.line_chart(hourly)
    
    with col_d:
        st.subheader("📏 Distribusi Jarak")
        fig_hist = px.histogram(df_real[df_real['trip_distance'] < 20], x="trip_distance", nbins=30, color_discrete_sequence=['#f39c12'])
        st.plotly_chart(fig_hist, use_container_width=True)

    st.caption("Dashboard ini dirancang untuk menunjukkan kemampuan analisis data geospasial dan integrasi model machine learning.")