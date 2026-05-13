import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)

DATA_PATH = "data/cleaned_nyc_taxi_weather_2025.parquet"

df = pd.read_parquet(DATA_PATH)

print("Jumlah data:")
print(df.shape)

print("\nDaftar kolom:")
for col in df.columns:
    print("-", col)

print("\n5 Data pertama:")
print(df.head())