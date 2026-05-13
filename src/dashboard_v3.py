import os
import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ==========================================
# 1. KONFIGURASI HALAMAN & CSS
# ==========================================
st.set_page_config(
    page_title="NYC Taxi RDV Dashboard",
    page_icon="🚕",
    layout="wide"
)

# Custom CSS BUNGlon (Adaptif Light/Dark Mode)
st.markdown("""
    <style>
    /* Menggunakan variabel bawaan Streamlit agar warna otomatis berubah */
    [data-testid="metric-container"] { 
        background-color: var(--secondary-background-color); 
        padding: 15px; 
        border-radius: 10px; 
        border: 1px solid rgba(128, 128, 128, 0.2);
        box-shadow: 0 2px 4px rgba(0,0,0,0.05); 
    }
    .prediction-card { 
        background-color: var(--secondary-background-color); 
        padding: 20px; 
        border-radius: 15px; 
        border-left: 5px solid #2196f3;
        margin-top: 20px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .prediction-title { 
        color: var(--text-color); 
        font-weight: bold; 
        margin-bottom: 0; 
        opacity: 0.8;
    }
    .prediction-value { 
        color: #2196f3; 
        margin-top: 5px; 
        margin-bottom: 5px; 
        font-size: 2.5em; 
        font-weight: bold; 
    }
    .prediction-sub { 
        color: var(--text-color); 
        opacity: 0.6; 
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. LOAD DATA & MODEL
# ==========================================
DATA_PATH  = "data/dashboard_sample_2025.parquet"
MODEL_PATH = "models/linear_regression.pkl"
ZONE_PATH  = "data/taxi_zone_lookup.csv"
DIST_PATH  = "data/zone_distances.csv"

@st.cache_resource
def load_assets():
    model = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None
    df = pd.read_parquet(DATA_PATH) if os.path.exists(DATA_PATH) else pd.DataFrame()
    
    # Mapping Wilayah & Kategori Waktu
    if not df.empty:
        z_df = pd.read_csv(ZONE_PATH)
        mapping = z_df.set_index('LocationID')['Borough'].to_dict()
        df['pickup_borough'] = df['PULocationID'].map(mapping)
        df['dropoff_borough'] = df['DOLocationID'].map(mapping)
        df['time_category'] = df['pickup_hour'].apply(lambda x: "Pagi" if 5<=x<11 else ("Siang" if 11<=x<17 else "Malam"))
    
    dist_index = pd.read_csv(DIST_PATH).set_index(["PULocationID", "DOLocationID"])["distance_miles"] if os.path.exists(DIST_PATH) else None
    zone_options = (pd.read_csv(ZONE_PATH)["Borough"] + " - " + pd.read_csv(ZONE_PATH)["Zone"]).tolist() if os.path.exists(ZONE_PATH) else []
    
    return model, df, dist_index, zone_options

model, df_real, dist_index, zone_options = load_assets()

# ==========================================
# 3. SIDEBAR: KONTROL PREDIKSI
# ==========================================
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/d/d3/New_York_City_Taxi_logo.svg", width=100)
    st.title("Navigator Prediksi")
    st.write("Sesuaikan parameter di bawah untuk melihat estimasi waktu tempuh.")
    st.markdown("---")
    
    pu_choice = st.selectbox("Titik Penjemputan", zone_options)
    do_choice = st.selectbox("Titik Tujuan", zone_options, index=10 if len(zone_options) > 10 else 0)
    hour = st.slider("Jam Keberangkatan", 0, 23, 12)
    temp = st.slider("Suhu Lingkungan (°C)", -10, 40, 20)
    rain = st.selectbox("Kondisi Hujan", [0.0, 1.0, 5.0, 10.0], format_func=lambda x: "Tidak Hujan" if x==0 else f"Hujan ({x}mm)")

    # Hitung Jarak & Prediksi
    dist_val = 5.0
    try:
        z_raw = pd.read_csv(ZONE_PATH)
        p_id = z_raw.loc[(z_raw["Borough"] + " - " + z_raw["Zone"]) == pu_choice, "LocationID"].values[0]
        d_id = z_raw.loc[(z_raw["Borough"] + " - " + z_raw["Zone"]) == do_choice, "LocationID"].values[0]
        dist_val = dist_index.loc[(p_id, d_id)]
    except: pass

    st.markdown(f"**Jarak Terdeteksi:** `{dist_val:.2f} miles`")
    
    if st.button("⚡ Jalankan Prediksi", use_container_width=True):
        if model:
            inp = pd.DataFrame([{"pickup_hour": hour, "trip_distance": dist_val, "PULocationID": p_id, "DOLocationID": d_id, "temperature_2m": temp, "precipitation": rain}])
            res = max(1.0, model.predict(inp)[0])
            
            # OUTPUT PREDIKSI YANG INTERAKTIF (Warna Adaptif)
            st.markdown(f"""
                <div class="prediction-card">
                    <p class="prediction-title">HASIL PREDIKSI DURASI</p>
                    <div class="prediction-value">{res:.1f} Menit</div>
                    <small class="prediction-sub">Berdasarkan pola trafik & cuaca</small>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.error("Model .pkl tidak ditemukan!")

# ==========================================
# 4. MAIN DASHBOARD: VISUALISASI LENGKAP
# ==========================================
st.title("🚕 NYC Yellow Taxi: Analisis Mobilitas 2025")
st.markdown(f"Dashboard ini merupakan bagian dari projek **Rekayasa Data & Visualisasi (RDV)**. (Data aktual: {len(df_real):,} baris)")

# KPI Row
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("Dataset Size", f"{len(df_real):,} Rows")
kpi2.metric("Avg. Distance", f"{df_real['trip_distance'].mean():.2f} Mi")
kpi3.metric("Avg. Duration", f"{df_real['trip_duration'].mean():.1f} Min")
kpi4.metric("Avg. Fare", f"${df_real['fare_amount'].mean():.2f}")

# TABS UNTUK ORGANISASI PLOT
tab1, tab2, tab3 = st.tabs(["📈 Tren & Distribusi", "🔥 Analisis Wilayah", "📋 Karakteristik"])

with tab1:
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Pola Keberangkatan per Jam")
        fig_h = px.line(df_real.groupby('pickup_hour').size().reset_index(name='count'), 
                        x='pickup_hour', y='count', markers=True, color_discrete_sequence=['#f7c948'])
        fig_h.update_layout(xaxis_title="Jam", yaxis_title="Jumlah Perjalanan")
        st.plotly_chart(fig_h, use_container_width=True)
    
    with col_b:
        st.subheader("Hubungan Jarak vs Waktu")
        fig_sc = px.scatter(df_real.sample(min(1000, len(df_real))), x="trip_distance", y="trip_duration", 
                            color="pickup_hour", opacity=0.5, color_continuous_scale="Viridis")
        st.plotly_chart(fig_sc, use_container_width=True)

    st.subheader("Distribusi Durasi Perjalanan")
    fig_hist = px.histogram(df_real[df_real['trip_duration']<60], x="trip_duration", nbins=50, color_discrete_sequence=['#2ecc71'])
    fig_hist.update_layout(xaxis_title="Durasi (Menit)", yaxis_title="Frekuensi")
    st.plotly_chart(fig_hist, use_container_width=True)

with tab2:
    st.subheader("Heatmap Aliran Perjalanan Antar Borough")
    hm_data = df_real.groupby(['pickup_borough', 'dropoff_borough']).size().unstack(fill_value=0)
    # Ganti "Blues" jadi "Plasma" biar nyala di mode gelap dan tetap bagus di mode terang
    fig_hm = px.imshow(hm_data, text_auto=True, color_continuous_scale="Plasma", aspect="auto")
    st.plotly_chart(fig_hm, use_container_width=True)
    st.write("Insight: *Gunakan heatmap ini untuk mengidentifikasi rute antar-wilayah yang paling sibuk.*")

with tab3:
    c_left, c_right = st.columns([1, 1])
    with c_left:
        st.subheader("Proporsi Waktu Perjalanan")
        fig_pie = px.pie(df_real, names='time_category', hole=0.5, color_discrete_sequence=px.colors.qualitative.Set3)
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with c_right:
        st.subheader("Rata-rata per Kategori Waktu")
        summ = df_real.groupby('time_category')[['trip_distance', 'trip_duration', 'fare_amount']].mean().reindex(["Pagi", "Siang", "Malam"])
        st.dataframe(summ.style.highlight_max(axis=0, color='rgba(255, 236, 179, 0.5)'), use_container_width=True)

st.markdown("---")
st.caption("Dikembangkan oleh tim | Informatika UB | 2026")