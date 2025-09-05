import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib
import os


PROCESSED_FILE = os.path.join("data", "processed.csv")
SCALER_X_FILE = os.path.join("models", "scaler_X.pkl")
SCALER_Y_FILE = os.path.join("models", "scaler_y.pkl")


def build_features(df, target="Kp", max_lag=12):
    """
    Add lag, rolling, and time-based features.
    """
    df = df.copy()

    # Lag features
    for lag in range(1, max_lag + 1):
        df[f"lag_{lag}"] = df[target].shift(lag)

    # Rolling stats
    df["roll_mean_6"] = df[target].rolling(6).mean()
    df["roll_std_6"] = df[target].rolling(6).std()
    df["roll_mean_24"] = df[target].rolling(24).mean()
    df["roll_std_24"] = df[target].rolling(24).std()

    # Time features
    df["hour"] = df["time_tag"].dt.hour
    df["month"] = df["time_tag"].dt.month
    df["dayofyear"] = df["time_tag"].dt.dayofyear

    return df.dropna().reset_index(drop=True)


def scale_and_save(X, y):
    """
    Scale features and target using MinMaxScaler (0-1).
    Save scalers to models/.
    """
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

    # Save scalers for future inference
    os.makedirs("models", exist_ok=True)
    joblib.dump(scaler_X, SCALER_X_FILE)
    joblib.dump(scaler_y, SCALER_Y_FILE)

    return X_scaled, y_scaled


def process_and_save(input_file="data/raw_data.csv"):
    """
    Load raw data, engineer features, scale, and save processed dataset.
    """
    df = pd.read_csv(input_file, parse_dates=["time_tag"])
    df = build_features(df)

    feature_cols = [c for c in df.columns if c not in ["time_tag", "Kp"]]
    X, y = df[feature_cols].values, df["Kp"].values

    X_scaled, y_scaled = scale_and_save(X, y)

    processed = df.copy()
    processed[feature_cols] = X_scaled
    processed["Kp_scaled"] = y_scaled

    processed.to_csv(PROCESSED_FILE, index=False)
    print(f"✅ Processed dataset saved with {len(processed)} rows → {PROCESSED_FILE}")

    return processed


if __name__ == "__main__":
    process_and_save()
