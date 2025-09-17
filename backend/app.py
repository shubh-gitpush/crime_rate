from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import os
import glob

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
    prediction_level: str = "state"  # Can be "state" or "district"

# State centroids for location mapping
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

# ---------- Data loading and training ----------
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

# Load district-level data
district_files = glob.glob(os.path.join(DATA_DIR, "*_District_wise_crimes_committed_IPC_20*.csv"))
district_dfs = []

for file in district_files:
    try:
        df_temp = pd.read_csv(file)
        df_temp.columns = [c.strip() for c in df_temp.columns]
        year = int(file.split("_")[-1].split(".")[0])  # Extract year from filename
        df_temp['Year'] = year
        district_dfs.append(df_temp)
    except Exception as e:
        print(f"Error reading {file}: {e}")

if not district_dfs:
    raise ValueError("No suitable district-level CSV files found in data folder.")

# Combine all district data
district_df = pd.concat(district_dfs, ignore_index=True)

# Create features for district model
district_features = ['Murder', 'Rape', 'Kidnapping & Abduction_Total', 'Robbery', 'Theft']

# Calculate normalized risk score based on serious crimes with adjusted weights
district_df['total_population'] = 100000  # Default population if not available

# Define weights for different crime types (adjusted for severity)
district_features_weights = {
    'Murder': 10.0,           # Most severe
    'Rape': 8.0,             # Very severe
    'Kidnapping & Abduction_Total': 6.0,
    'Robbery': 4.0,
    'Theft': 1.0             # Least severe
}

# Calculate per capita crime rates first
for feature in district_features:
    district_df[f'{feature}_per_capita'] = (
        district_df[feature].fillna(0) / district_df['total_population'] * 100000
    )

# Get the median values for scaling
medians = {feature: district_df[f'{feature}_per_capita'].median() 
          for feature in district_features}

# Calculate normalized and weighted risk score
district_df['risk_score'] = 0
for feature, weight in district_features_weights.items():
    # Normalize each crime rate relative to its median and apply weight
    normalized_rate = district_df[f'{feature}_per_capita'] / (medians[feature] if medians[feature] > 0 else 1)
    district_df['risk_score'] += normalized_rate * weight

# Scale scores to 0-100 range using a log transform to handle outliers
district_df['risk_score'] = np.log1p(district_df['risk_score'])  # log1p handles zero values
min_score = district_df['risk_score'].min()
max_score = district_df['risk_score'].quantile(0.98)  # 98th percentile to handle extreme outliers
district_df['risk_score'] = ((district_df['risk_score'] - min_score) / (max_score - min_score) * 100).clip(0, 100)

# Calculate risk levels with more nuanced thresholds
def label_district_risk(score):
    if score <= 20:
        return 0  # very safe
    elif score <= 40:
        return 1  # safe
    elif score <= 60:
        return 2  # moderate
    elif score <= 80:
        return 3  # risky
    else:
        return 4  # very risky

district_df['risk'] = district_df['risk_score'].apply(label_district_risk)

# Prepare district-level model
district_X = district_df[district_features].fillna(0)
district_y = district_df['risk']

district_scaler = StandardScaler()
district_X_scaled = district_scaler.fit_transform(district_X)

district_model = LogisticRegression(
    class_weight='balanced',
    max_iter=1000,
    random_state=42
)
district_model.fit(district_X_scaled, district_y)

# Create district location mapping
DISTRICT_LOCATIONS = {}
for state in STATE_CENTROIDS:
    state_districts = district_df[district_df['States/UTs'] == state]['District'].unique()
    lat, lon = STATE_CENTROIDS[state]
    # Assign slightly offset coordinates to each district
    for i, district in enumerate(state_districts):
        offset = (i * 0.1) - ((len(state_districts) - 1) * 0.05)  # Spread districts around state centroid
        DISTRICT_LOCATIONS[district] = (lat + offset, lon + offset)

def find_nearest_state(lat: float, lon: float) -> str:
    nearest = None
    best_d2 = float("inf")
    for state, (s_lat, s_lon) in STATE_CENTROIDS.items():
        d2 = (lat - s_lat) ** 2 + (lon - s_lon) ** 2
        if d2 < best_d2:
            best_d2 = d2
            nearest = state
    return nearest

def find_nearest_district(lat: float, lon: float, state: str) -> str:
    state_districts = district_df[district_df['States/UTs'] == state]['District'].unique()
    if not len(state_districts):
        return None
    
    nearest = None
    best_d2 = float("inf")
    for district in state_districts:
        if district in DISTRICT_LOCATIONS:
            d_lat, d_lon = DISTRICT_LOCATIONS[district]
            d2 = (lat - d_lat) ** 2 + (lon - d_lon) ** 2
            if d2 < best_d2:
                best_d2 = d2
                nearest = district
    return nearest

@app.post("/predict")
def predict(data: TouristData):
    state = find_nearest_state(data.lat, data.lon)
    response = {"state": state}
    
    if data.prediction_level == "district":
        district = find_nearest_district(data.lat, data.lon, state)
        if district:
            response["district"] = district
            # Get latest district data
            district_data = district_df[
                (district_df['States/UTs'] == state) & 
                (district_df['District'] == district)
            ].iloc[-1]
            
            # Get district-level crime rates
            features = district_data[district_features].fillna(0)
            district_pop = max(district_data['total_population'], 100000)  # Prevent division by zero

            # Get national district averages for comparison
            district_averages = {}
            for feature in district_features:
                dist_avg = district_df[feature].mean()
                district_averages[feature] = max(dist_avg, 1)  # Prevent division by zero
            
            # Calculate weighted ratios comparing to national average
            weighted_ratios = []
            for feature, weight in district_features_weights.items():
                district_rate = (features[feature] / district_pop) * 100000
                avg_rate = (district_averages[feature] / 100000) * 100000
                
                # Calculate ratio but limit extreme values
                ratio = min(district_rate / avg_rate if avg_rate > 0 else 1, 5)
                weighted_ratios.append(ratio * weight)
            
            # Calculate final score
            avg_ratio = sum(weighted_ratios) / sum(district_features_weights.values())
            
            # Convert ratio to score (1.0 ratio = 50 points)
            if avg_ratio <= 0.5:  # Much safer than average
                score = 35
            elif avg_ratio <= 0.75:  # Safer than average
                score = 40 + (avg_ratio - 0.5) * 20
            elif avg_ratio <= 1.25:  # Around average
                score = 45 + (avg_ratio - 0.75) * 20
            elif avg_ratio <= 2.0:  # More dangerous than average
                score = 55 + (avg_ratio - 1.25) * 10
            else:  # Much more dangerous than average
                score = 65
            
            score = int(score)
            
            # Assign risk levels consistently
            if score <= 40:
                label = 0  # very safe
            elif score <= 47:
                label = 1  # safe
            elif score <= 53:
                label = 2  # moderate
            elif score <= 60:
                label = 3  # risky
            else:
                label = 4  # very risky
        else:
            # Fallback to state-level prediction
            response["warning"] = "District-level data not available, using state-level prediction"
            state_data = district_df[district_df['States/UTs'] == state].groupby('States/UTs')[district_features].mean()
            if len(state_data) > 0:
                features = state_data.iloc[0].values.reshape(1, -1)
            else:
                features = np.zeros((1, len(district_features)))
            features_scaled = district_scaler.transform(features)
            proba = district_model.predict_proba(features_scaled)[0]
            label = int(np.argmax(proba))
            score = int(round(proba[label] * 100))
    else:
        # State-level prediction using aggregated district data
        state_data = district_df[district_df['States/UTs'] == state]
        
        if len(state_data) > 0:
            # Get the latest year's data for the state
            latest_year = state_data['Year'].max()
            latest_state_data = state_data[state_data['Year'] == latest_year]
            
            # Sum up all crime counts for the state
            features = latest_state_data[district_features].sum()
            
            # Get state population (using sum of district populations)
            state_population = latest_state_data['total_population'].sum()
            if state_population == 0:
                state_population = 100000  # fallback if no population data
            
            # Simple scoring based on direct crime rates comparison
            total_population = max(state_population, 100000)  # Prevent division by zero
            
            # Get state averages for comparison
            state_averages = {}
            for feature in district_features:
                state_avg = district_df[feature].mean()
                state_averages[feature] = max(state_avg, 1)  # Prevent division by zero
            
            # Calculate score based on deviation from average
            weighted_ratios = []
            for feature, weight in district_features_weights.items():
                state_rate = (features[feature] / total_population) * 100000
                avg_rate = (state_averages[feature] / 100000) * 100000
                
                # Calculate ratio but limit extreme values
                ratio = min(state_rate / avg_rate if avg_rate > 0 else 1, 5)
                weighted_ratios.append(ratio * weight)
            
            # Calculate final score
            avg_ratio = sum(weighted_ratios) / sum(district_features_weights.values())
            
            # Convert ratio to score (1.0 ratio = 50 points)
            if avg_ratio <= 0.5:  # Much safer than average
                score = 35
            elif avg_ratio <= 0.75:  # Safer than average
                score = 40 + (avg_ratio - 0.5) * 20
            elif avg_ratio <= 1.25:  # Around average
                score = 45 + (avg_ratio - 0.75) * 20
            elif avg_ratio <= 2.0:  # More dangerous than average
                score = 55 + (avg_ratio - 1.25) * 10
            else:  # Much more dangerous than average
                score = 65
            
            score = int(score)
            
            # Assign risk levels with wider spread
            if score <= 40:
                label = 0  # very safe
            elif score <= 47:
                label = 1  # safe
            elif score <= 53:
                label = 2  # moderate
            elif score <= 60:
                label = 3  # risky
            else:
                label = 4  # very risky
        else:
            # Default values if no data available
            score = 50
            label = 1  # average

    # Time-based adjustment with refined factors
    base_score = score  # Store original score
    
    # Define time risk factors with smaller adjustments
    if 0 <= data.hour <= 4:
        time_factor = 1.15  # Late night
    elif 5 <= data.hour <= 6:
        time_factor = 1.10  # Early morning
    elif 7 <= data.hour <= 9:
        time_factor = 0.95  # Morning rush (safer due to crowds)
    elif 10 <= data.hour <= 16:
        time_factor = 0.85  # Daytime (safest)
    elif 17 <= data.hour <= 19:
        time_factor = 0.95  # Evening rush
    elif 20 <= data.hour <= 21:
        time_factor = 1.05  # Evening
    else:  # 22-23
        time_factor = 1.10  # Night
        
    # Apply time factor with dampening for extreme scores
    time_adjustment = (time_factor - 1.0) * (1.0 - abs(score - 50) / 50)
    score = int(min(100, max(0, score * (1.0 + time_adjustment))))

    if label == 0:
        status = "very safe"
    elif label == 1:
        status = "safe"
    elif label == 2:
        status = "moderate"
    elif label == 3:
        status = "risky"
    elif label == 4:
        status = "very risky"
    else:
        status = "moderate"  # fallback for unexpected values

    response.update({
        "status": status,
        "score": score,
        "time_risk": "high" if score > 80 else "moderate" if score > 50 else "low",
        "recommendations": [
            "Avoid traveling alone at night" if data.hour >= 22 or data.hour <= 4 else None,
            "Stay in well-lit and populated areas" if score > 70 else None,
            "Keep emergency contacts handy" if score > 50 else None,
            "Safe to explore with normal precautions" if score <= 30 else None
        ]
    })
    
    return {k: v for k, v in response.items() if v is not None}
