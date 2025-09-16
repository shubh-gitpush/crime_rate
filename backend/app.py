from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import os


app = FastAPI()

# CORS for local frontend dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=False,
    allow_methods=["POST", "OPTIONS"],
    allow_headers=["*"],
)


class TouristData(BaseModel):
    lat: float
    lon: float
    hour: int


# ---------- Data loading and training from CSV ----------
CSV_PATH = os.path.join(os.path.dirname(__file__), "NCRB_Table_1A.1.csv")

df = pd.read_csv(CSV_PATH)
df.columns = [c.strip() for c in df.columns]

pop_lakhs_col = "Mid-Year Projected Population (in Lakhs) (2022)"

# Per-capita crime rates per 100k population
df["rate_2020"] = (df["2020"] / (df[pop_lakhs_col] * 100000)) * 100000
df["rate_2021"] = (df["2021"] / (df[pop_lakhs_col] * 100000)) * 100000
df["rate_2022"] = (df["2022"] / (df[pop_lakhs_col] * 100000)) * 100000

FEATURE_COLS = ["rate_2020", "rate_2021", "rate_2022", "Chargesheeting Rate (2022)"]

# Target labels based on quantiles
q_low = df["rate_2022"].quantile(0.33)
q_high = df["rate_2022"].quantile(0.66)

def label_risk(rate):
    if rate <= q_low:
        return 0  # safe
    elif rate >= q_high:
        return 2  # risky
    else:
        return 1  # average

df["risk"] = df["rate_2022"].apply(label_risk)

# Ensure numeric features
for col in FEATURE_COLS:
    df[col] = pd.to_numeric(df[col], errors="coerce")
df[FEATURE_COLS] = df[FEATURE_COLS].fillna(df[FEATURE_COLS].median(numeric_only=True))

X = df[FEATURE_COLS].to_numpy(dtype=float)
y = df["risk"].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = LogisticRegression(
    multi_class="multinomial",
    class_weight="balanced",
    max_iter=1000,
    C=0.5,
    random_state=42,
)
model.fit(X_scaled, y)

state_to_features = {row["State/UT"]: row[FEATURE_COLS].to_numpy(dtype=float) for _, row in df.iterrows()}

# Rough centroids for states/UTs present in the CSV (lat, lon)
STATE_CENTROIDS = {
    "Andhra Pradesh": (15.9, 80.0),
    "Arunachal Pradesh": (28.1, 94.6),
    "Assam": (26.2, 92.9),
    "Bihar": (25.6, 85.3),
    "Chhattisgarh": (21.3, 82.0),
    "Goa": (15.3, 74.1),
    "Gujarat": (22.7, 71.6),
    "Haryana": (29.0, 76.0),
    "Himachal Pradesh": (31.1, 77.2),
    "Jharkhand": (23.6, 85.3),
    "Karnataka": (15.3, 75.7),
    "Kerala": (10.3, 76.3),
    "Madhya Pradesh": (23.5, 78.5),
    "Maharashtra": (19.7, 75.7),
    "Manipur": (24.7, 93.9),
    "Meghalaya": (25.5, 91.3),
    "Mizoram": (23.3, 92.7),
    "Nagaland": (26.0, 94.5),
    "Odisha": (20.9, 85.1),
    "Punjab": (31.0, 75.3),
    "Rajasthan": (26.9, 73.0),
    "Sikkim": (27.5, 88.5),
    "Tamil Nadu": (11.1, 78.7),
    "Telangana": (17.9, 79.6),
    "Tripura": (23.9, 91.6),
    "Uttar Pradesh": (26.8, 80.9),
    "Uttarakhand": (30.1, 79.0),
    "West Bengal": (23.7, 88.1),
    "Andaman and Nicobar Islands": (11.7, 92.7),
    "Chandigarh": (30.7, 76.8),
    "Dadra and Nagar Haveli and Daman and Diu": (20.3, 73.0),
    "Delhi": (28.6, 77.2),
    "Jammu and Kashmir": (33.5, 75.0),
    "Ladakh": (34.2, 77.6),
    "Lakshadweep": (10.6, 72.6),
    "Puducherry": (11.9, 79.8),
}


def find_nearest_state(lat: float, lon: float) -> str:
    nearest = None
    best_d2 = float("inf")
    for state, (s_lat, s_lon) in STATE_CENTROIDS.items():
        d2 = (lat - s_lat) ** 2 + (lon - s_lon) ** 2
        if d2 < best_d2:
            best_d2 = d2
            nearest = state
    return nearest


@app.post("/predict")
def predict(data: TouristData):
    state = find_nearest_state(data.lat, data.lon)

    features = state_to_features.get(state)
    if features is None:
        features = np.nanmedian(X, axis=0)

    features = np.asarray(features, dtype=float).reshape(1, -1)
    features_scaled = scaler.transform(features)

    proba = model.predict_proba(features_scaled)[0]
    label = int(np.argmax(proba))
    score = int(round(proba[label] * 100))

    # Time-based adjustment
    if 0 <= data.hour <= 4:
        score = min(100, score + 5)
    elif 22 <= data.hour <= 23 or 5 <= data.hour <= 6:
        score = min(100, score + 2)

    if label == 0:
        status = "safe"
    elif label == 1:
        status = "average"
    else:
        status = "risky"

    return {"state": state, "status": status, "score": score}
