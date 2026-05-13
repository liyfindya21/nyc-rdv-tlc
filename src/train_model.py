import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

import joblib

# =========================
# LOAD DATA
# =========================

DATA_PATH = "data/cleaned_nyc_taxi_weather_2025.parquet"

df = pd.read_parquet(DATA_PATH)

print("Data loaded!")
print(df.shape)

# =========================
# PILIH FEATURE
# =========================

features = [
    "pickup_hour",
    "trip_distance",
    #"fare_amount",
    "PULocationID",
    "DOLocationID",
    "temperature_2m",
    "precipitation"
]

target = "trip_duration"

# =========================
# CLEAN DATA
# =========================

data = df[features + [target]].dropna()

print("\nData after cleaning:")
print(data.shape)

# =========================
# SPLIT X & y
# =========================

X = data[features]
y = data[target]

# =========================
# TRAIN TEST SPLIT
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTrain-test split done!")

# =========================
# TRAIN MODEL
# =========================

model = LinearRegression()

model.fit(X_train, y_train)

print("\nModel trained!")

# =========================
# PREDICTION
# =========================

y_pred = model.predict(X_test)

# =========================
# EVALUATION
# =========================

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n=== MODEL EVALUATION ===")
print(f"MAE : {mae:.2f}")
print(f"MSE : {mse:.2f}")
print(f"R2  : {r2:.2f}")
# =========================
# SAVE METRICS
# =========================

os.makedirs("outputs", exist_ok=True)

with open("outputs/model_metrics.txt", "w") as f:
    f.write("=== MODEL EVALUATION ===\n")
    f.write(f"MAE : {mae:.2f}\n")
    f.write(f"MSE : {mse:.2f}\n")
    f.write(f"R2  : {r2:.2f}\n")

print("\nMetrics saved to outputs/model_metrics.txt")


# =========================
# VISUALIZATION
# =========================

# ambil sample biar plot tidak berat
sample_size = min(1000, len(y_test))

y_test_sample = y_test.iloc[:sample_size]
y_pred_sample = y_pred[:sample_size]

plt.figure(figsize=(8, 6))
plt.scatter(y_test_sample, y_pred_sample, alpha=0.5)
plt.xlabel("Actual Trip Duration")
plt.ylabel("Predicted Trip Duration")
plt.title("Actual vs Predicted Trip Duration")

plt.savefig("outputs/actual_vs_predicted.png", dpi=300, bbox_inches="tight")
plt.close()

print("Plot saved to outputs/actual_vs_predicted.png")

# =========================
# SAVE MODEL
# =========================

joblib.dump(model, "models/linear_regression.pkl")

print("\nModel saved to models/linear_regression.pkl")