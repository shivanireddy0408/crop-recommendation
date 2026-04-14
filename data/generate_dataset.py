"""
Script to generate a realistic synthetic Crop Recommendation Dataset.
Based on real agronomic ranges for 22 common crops.
"""

import numpy as np
import pandas as pd

np.random.seed(42)

# Crop profiles: [N_mean, N_std, P_mean, P_std, K_mean, K_std,
#                  temp_mean, temp_std, humidity_mean, humidity_std,
#                  ph_mean, ph_std, rainfall_mean, rainfall_std, n_samples]
crop_profiles = {
    "rice":        [80, 10, 45, 8,  45, 8,  24, 2, 82, 5, 6.5, 0.4, 200, 20, 100],
    "maize":       [78, 10, 48, 8,  20, 5,  22, 2, 66, 5, 6.2, 0.4, 65,  10, 100],
    "chickpea":    [40, 8,  68, 8,  80, 8,  18, 2, 17, 4, 7.2, 0.4, 80,  10, 100],
    "kidneybeans": [20, 5,  68, 8,  20, 5,  20, 2, 21, 4, 5.7, 0.3, 105, 10, 100],
    "pigeonpeas":  [20, 5,  68, 8,  20, 5,  27, 2, 49, 5, 5.8, 0.3, 150, 15, 100],
    "mothbeans":   [21, 5,  48, 8,  20, 5,  28, 2, 53, 5, 6.8, 0.4, 51,  8,  100],
    "mungbean":    [21, 5,  48, 8,  20, 5,  29, 2, 86, 5, 6.6, 0.4, 49,  8,  100],
    "blackgram":   [40, 8,  68, 8,  20, 5,  30, 2, 66, 5, 7.1, 0.4, 68,  8,  100],
    "lentil":      [19, 5,  68, 8,  19, 5,  24, 2, 65, 5, 6.9, 0.4, 46,  8,  100],
    "pomegranate": [18, 4,  18, 4,  40, 6,  22, 2, 90, 5, 6.0, 0.3, 110, 10, 100],
    "banana":      [100,10, 82, 8,  50, 8,  27, 2, 80, 5, 5.9, 0.3, 105, 10, 100],
    "mango":       [0,  2,  27, 5,  30, 5,  31, 2, 50, 5, 5.7, 0.3, 95,  10, 100],
    "grapes":      [23, 5,  132,10, 200,15, 24, 2, 81, 5, 6.0, 0.3, 70,  8,  100],
    "watermelon":  [99, 10, 17, 4,  50, 8,  25, 2, 85, 5, 6.5, 0.4, 50,  8,  100],
    "muskmelon":   [100,10, 17, 4,  50, 8,  29, 2, 92, 5, 6.4, 0.4, 25,  5,  100],
    "apple":       [21, 5,  134,10, 199,15, 22, 2, 92, 5, 5.9, 0.3, 113, 10, 100],
    "orange":      [0,  2,  16, 4,  10, 3,  23, 2, 92, 5, 7.0, 0.4, 110, 10, 100],
    "papaya":      [49, 8,  59, 8,  50, 8,  34, 2, 92, 5, 6.8, 0.4, 143, 12, 100],
    "coconut":     [22, 5,  16, 4,  30, 5,  27, 2, 95, 4, 6.0, 0.3, 175, 15, 100],
    "cotton":      [118,12, 46, 8,  20, 5,  24, 2, 80, 5, 6.7, 0.4, 80,  10, 100],
    "jute":        [78, 10, 46, 8,  40, 6,  25, 2, 80, 5, 6.8, 0.4, 175, 15, 100],
    "coffee":      [101,10, 28, 5,  30, 5,  25, 2, 58, 5, 6.5, 0.4, 158, 12, 100],
}

rows = []
for crop, params in crop_profiles.items():
    n = params[14]
    N         = np.clip(np.random.normal(params[0],  params[1],  n), 0, 200)
    P         = np.clip(np.random.normal(params[2],  params[3],  n), 0, 145)
    K         = np.clip(np.random.normal(params[4],  params[5],  n), 0, 210)
    temp      = np.clip(np.random.normal(params[6],  params[7],  n), 8,  44)
    humidity  = np.clip(np.random.normal(params[8],  params[9],  n), 14, 100)
    ph        = np.clip(np.random.normal(params[10], params[11], n), 3.5, 9.5)
    rainfall  = np.clip(np.random.normal(params[12], params[13], n), 20, 300)
    for i in range(n):
        rows.append([N[i], P[i], K[i], temp[i], humidity[i], ph[i], rainfall[i], crop])

df = pd.DataFrame(rows, columns=["N","P","K","temperature","humidity","ph","rainfall","label"])
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
df.to_csv("data/Crop_recommendation.csv", index=False)
print(f"Dataset saved: {len(df)} rows, {df['label'].nunique()} crops")
print(df.head())
