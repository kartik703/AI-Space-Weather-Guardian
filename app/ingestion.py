import os
import requests
import pandas as pd


DATA_FILE = os.path.join("data", "raw_data.csv")


def fetch_noaa_kp():
    """Fetch NOAA Kp index data."""
    url = "https://services.swpc.noaa.gov/json/planetary_k_index_1m.json"
    df = pd.read_json(url)
    df["time_tag"] = pd.to_datetime(df["time_tag"])
    df = df.rename(columns={"kp_index": "Kp"})
    return df[["time_tag", "Kp"]]


def fetch_solar_wind():
    """Fetch DSCOVR solar wind plasma data."""
    url = "https://services.swpc.noaa.gov/products/solar-wind/plasma-7-day.json"
    raw = requests.get(url, timeout=10).json()
    df = pd.DataFrame(raw[1:], columns=raw[0])
    df["time_tag"] = pd.to_datetime(df["time_tag"])
    for col in ["density", "speed", "temperature"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df[["time_tag", "density", "speed", "temperature"]].dropna()


def load_and_merge():
    """Fetch + merge Kp + solar wind into one DataFrame."""
    kp = fetch_noaa_kp()
    sw = fetch_solar_wind()
    df = pd.merge_asof(kp.sort_values("time_tag"),
                       sw.sort_values("time_tag"),
                       on="time_tag",
                       tolerance=pd.Timedelta("5m"))
    return df.dropna()


def update_dataset():
    """Update local dataset with new fetched data."""
    new_data = load_and_merge()
    if os.path.exists(DATA_FILE):
        old_data = pd.read_csv(DATA_FILE, parse_dates=["time_tag"])
        combined = pd.concat([old_data, new_data]).drop_duplicates("time_tag").sort_values("time_tag")
    else:
        combined = new_data

    combined.to_csv(DATA_FILE, index=False)
    print(f"✅ Saved dataset with {len(combined)} rows at {DATA_FILE}")


if __name__ == "__main__":
    update_dataset()
