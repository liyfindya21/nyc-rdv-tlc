import pandas as pd

print("⏳ Sedang membaca dataset utama (21.8 Juta baris)... Sabar ya!")
# Sesuaikan path jika file utamanya ada di tempat lain
df = pd.read_parquet("data/cleaned_nyc_taxi_weather_2025.parquet")

print("✂️ Mengambil sampel 1.000.000 baris acak...")
# Kita ambil 1.000.000 baris sesuai permintaan
df_sample = df.sample(n=1000000, random_state=42)

print("💾 Menyimpan ke file khusus dashboard...")
df_sample.to_parquet("data/dashboard_sample_2025.parquet", index=False)

print("✅ Selesai! File dashboard_sample_2025.parquet siap digunakan.")