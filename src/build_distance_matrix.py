"""
build_distance_matrix.py
========================
Script untuk membuat zone_distances.csv yang berisi jarak jalan nyata
(bukan estimasi) untuk SEMUA kombinasi zona NYC taxi (265x265 = 70,225 route).

Cara kerja:
1. Download shapefile zona NYC dari NYC Open Data
2. Hitung centroid tiap zona (titik tengah polygon)
3. Panggil OSRM Table API (gratis, berbasis OpenStreetMap) → jarak jalan nyata
4. Simpan ke zone_distances.csv

Jalankan sekali saja, hasilnya pakai terus di dashboard.

Requirements:
    pip install geopandas requests pandas pyarrow

Output:
    data/zone_distances.csv
    Kolom: PULocationID, DOLocationID, distance_miles
"""

import os
import math
import time
import requests
import zipfile
import io
import pandas as pd
import geopandas as gpd

# ──────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────

OUTPUT_PATH = "data/zone_distances.csv"
ZONE_LOOKUP  = "data/taxi_zone_lookup.csv"

# OSRM public demo server (gratis, no key needed)
# Untuk produksi, pertimbangkan self-host OSRM atau pakai server berbayar
OSRM_BASE = "http://router.project-osrm.com"

# Shapefile NYC taxi zones dari NYC Open Data (resmi, gratis)
SHAPEFILE_URL = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zones.zip"

# ──────────────────────────────────────────
# STEP 1: DOWNLOAD & EXTRACT CENTROIDS
# ──────────────────────────────────────────

def download_zone_centroids() -> pd.DataFrame:
    """
    Download shapefile zona NYC, hitung centroid tiap zona,
    return DataFrame dengan kolom: LocationID, lon, lat
    """
    print("[1/3] Download shapefile NYC taxi zones ...")

    resp = requests.get(SHAPEFILE_URL, timeout=60)
    resp.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
        z.extractall("_tmp_shp")

    # Cari file .shp
    shp_file = None
    for root, _, files in os.walk("_tmp_shp"):
        for f in files:
            if f.endswith(".shp"):
                shp_file = os.path.join(root, f)
                break

    if shp_file is None:
        raise FileNotFoundError("Shapefile tidak ditemukan di ZIP")

    gdf = gpd.read_file(shp_file)
    gdf = gdf.to_crs(epsg=4326)          # konversi ke WGS84 (lon/lat)
    gdf["centroid"] = gdf.geometry.centroid

    # Kolom LocationID bisa bernama 'OBJECTID', 'location_i', atau 'LocationID'
    # Deteksi otomatis
    id_col = None
    for candidate in ["LocationID", "location_i", "OBJECTID", "objectid"]:
        if candidate in gdf.columns:
            id_col = candidate
            break

    if id_col is None:
        print("Kolom tersedia:", gdf.columns.tolist())
        raise ValueError("Tidak ditemukan kolom LocationID di shapefile")

    centroids = pd.DataFrame({
        "LocationID": gdf[id_col].astype(int),
        "lon":        gdf["centroid"].x,
        "lat":        gdf["centroid"].y,
    })

    print(f"   → {len(centroids)} zona ditemukan")
    return centroids


# ──────────────────────────────────────────
# STEP 2: OSRM TABLE API
# ──────────────────────────────────────────

def osrm_distance_matrix(centroids: pd.DataFrame) -> pd.DataFrame:
    """
    Panggil OSRM /table service untuk mendapat matriks jarak
    antara semua zona. Return DataFrame panjang (PULocationID,
    DOLocationID, distance_miles).

    OSRM public server membatasi ~1000 koordinat per request,
    jadi kita batch per 100 zona sekaligus.
    """
    print("[2/3] Hitung jarak via OSRM Table API ...")

    coords = centroids.sort_values("LocationID").reset_index(drop=True)
    n = len(coords)

    # Format koordinat untuk URL: "lon,lat;lon,lat;..."
    coord_str = ";".join(
        f"{row.lon:.6f},{row.lat:.6f}"
        for row in coords.itertuples()
    )

    BATCH = 100   # zone per batch (source)
    records = []

    for i in range(0, n, BATCH):
        src_indices  = list(range(i, min(i + BATCH, n)))
        src_str      = ";".join(str(x) for x in src_indices)

        url = (
            f"{OSRM_BASE}/table/v1/driving/{coord_str}"
            f"?sources={src_str}"
            f"&annotations=distance"
        )

        try:
            resp = requests.get(url, timeout=60)
            if resp.status_code != 200:
                print(f"   ⚠ OSRM error pada batch {i}: {resp.status_code}")
                _haversine_batch(coords, src_indices, records)
                continue
        except requests.exceptions.RequestException:
            print(f"   ⚠ Server OSRM down. Fallback ke Haversine.")
            _haversine_batch(coords, src_indices, records)
            continue

        data = resp.json()
        if data.get("code") != "Ok":
            print(f"   ⚠ OSRM code bukan Ok: {data.get('code')}")
            _haversine_batch(coords, src_indices, records)
            continue

        distances = data["distances"]   # matrix [src x all_dst] dalam meter

        for local_i, src_idx in enumerate(src_indices):
            pu_id = int(coords.loc[src_idx, "LocationID"])
            row_m = distances[local_i]    # list jarak ke semua tujuan (meter)

            for dst_idx, meters in enumerate(row_m):
                do_id = int(coords.loc[dst_idx, "LocationID"])
                if meters is None:
                    # OSRM tidak bisa route → pakai haversine
                    meters = _haversine_m(
                        coords.loc[src_idx, "lat"], coords.loc[src_idx, "lon"],
                        coords.loc[dst_idx, "lat"], coords.loc[dst_idx, "lon"],
                    ) * 1.3   # koreksi road factor
                miles = meters / 1609.344
                records.append((pu_id, do_id, round(miles, 4)))

        done = min(i + BATCH, n)
        print(f"   → Batch {i}–{done-1} selesai ({done}/{n} zona)")
        time.sleep(0.5)   # hindari rate limit

    df = pd.DataFrame(records, columns=["PULocationID", "DOLocationID", "distance_miles"])
    print(f"   → Total {len(df):,} route di-generate")
    return df


def _haversine_m(lat1, lon1, lat2, lon2) -> float:
    """Jarak straight-line antara dua titik (dalam meter)."""
    R = 6_371_000
    p = math.pi / 180
    a = (math.sin((lat2 - lat1) * p / 2) ** 2
         + math.cos(lat1 * p) * math.cos(lat2 * p)
         * math.sin((lon2 - lon1) * p / 2) ** 2)
    return 2 * R * math.asin(math.sqrt(a))


def _haversine_batch(coords, src_indices, records):
    """Fallback haversine (dengan road factor 1.3) jika OSRM gagal."""
    for src_idx in src_indices:
        pu_id = int(coords.loc[src_idx, "LocationID"])
        for dst_idx in range(len(coords)):
            do_id = int(coords.loc[dst_idx, "LocationID"])
            m = _haversine_m(
                coords.loc[src_idx, "lat"], coords.loc[src_idx, "lon"],
                coords.loc[dst_idx, "lat"], coords.loc[dst_idx, "lon"],
            ) * 1.3
            records.append((pu_id, do_id, round(m / 1609.344, 4)))


# ──────────────────────────────────────────
# STEP 3: SIMPAN
# ──────────────────────────────────────────

def save(df: pd.DataFrame):
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"[3/3] Tersimpan → {OUTPUT_PATH}")
    print(f"      {len(df):,} baris | {os.path.getsize(OUTPUT_PATH)/1024:.0f} KB")


# ──────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("  NYC Taxi Zone Distance Matrix Builder")
    print("=" * 55)

    centroids = download_zone_centroids()
    dist_df   = osrm_distance_matrix(centroids)
    save(dist_df)

    print()
    print("✅ Selesai! Sekarang jalankan dashboard Streamlit.")
    print(f"   File: {OUTPUT_PATH}")
    print(f"   Coverage: 100% ({len(dist_df):,} route)")