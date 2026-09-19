# Uber Fare Price Prediction
# Laboratory Practice III - Machine Learning

# ==============================
# 1. IMPORT LIBRARIES
# ==============================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# ==============================
# 2. LOAD DATASET
# ==============================

df = pd.read_csv("uber.csv")

print("First 5 rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nDataset Information:")
print(df.info())


# ==============================
# 3. DATA PREPROCESSING
# ==============================

print("\nMissing Values:")
print(df.isnull().sum())

# Remove rows containing missing values
df = df.dropna()

# Remove unnecessary column
if "Unnamed: 0" in df.columns:
    df = df.drop("Unnamed: 0", axis=1)

# Remove invalid passenger counts
df = df[(df["passenger_count"] > 0) & (df["passenger_count"] <= 6)]

# Remove invalid fare values
df = df[df["fare_amount"] > 0]

# Remove invalid latitude and longitude values
df = df[
    (df["pickup_latitude"].between(-90, 90)) &
    (df["dropoff_latitude"].between(-90, 90)) &
    (df["pickup_longitude"].between(-180, 180)) &
    (df["dropoff_longitude"].between(-180, 180))
]

print("\nShape after preprocessing:")
print(df.shape)


# ==============================
# 4. IDENTIFY OUTLIERS
# ==============================

plt.figure(figsize=(8, 5))
plt.boxplot(df["fare_amount"])
plt.title("Box Plot of Fare Amount")
plt.ylabel("Fare Amount")
plt.show()

# Using IQR method to identify fare outliers

Q1 = df["fare_amount"].quantile(0.25)
Q3 = df["fare_amount"].quantile(0.75)

IQR = Q3 - Q1

lower_limit = Q1 - 1.5 * IQR
upper_limit = Q3 + 1.5 * IQR

outliers = df[
    (df["fare_amount"] < lower_limit) |
    (df["fare_amount"] > upper_limit)
]

print("\nNumber of Outliers:")
print(len(outliers))

print("\nLower Limit:", lower_limit)
print("Upper Limit:", upper_limit)

# Remove fare outliers
df = df[
    (df["fare_amount"] >= lower_limit) &
    (df["fare_amount"] <= upper_limit)
]

print("\nShape after removing outliers:")
print(df.shape)


# ==============================
# 5. HAVERSINE DISTANCE
# ==============================

def haversine_distance(lat1, lon1, lat2, lon2):

    R = 6371  # Earth radius in kilometers

    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1)
        * np.cos(lat2)
        * np.sin(dlon / 2) ** 2
    )

    c = 2 * np.arcsin(np.sqrt(a))

    return R * c


df["distance_km"] = haversine_distance(
    df["pickup_latitude"],
    df["pickup_longitude"],
    df["dropoff_latitude"],
    df["dropoff_longitude"]
)

print("\nDistance Feature:")
print(df[["pickup_latitude",
          "pickup_longitude",
          "dropoff_latitude",
          "dropoff_longitude",
          "distance_km"]].head())


# ==============================
# 6. REMOVE ZERO-DISTANCE RIDES
# ==============================

df = df[df["distance_km"] > 0]

print("\nShape after removing zero-distance rides:")
print(df.shape)


# ==============================
# 7. CHECK CORRELATION
# ==============================

correlation = df[
    [
        "fare_amount",
        "passenger_count",
        "distance_km"
    ]
].corr()

print("\nCorrelation Matrix:")
print(correlation)

plt.figure(figsize=(7, 5))
plt.imshow(correlation, cmap="coolwarm")
plt.colorbar()

plt.xticks(
    range(len(correlation.columns)),
    correlation.columns,
    rotation=45
)

plt.yticks(
    range(len(correlation.columns)),
    correlation.columns
)

plt.title("Correlation Matrix")

for i in range(len(correlation.columns)):
    for j in range(len(correlation.columns)):
        plt.text(
            j,
            i,
            round(correlation.iloc[i, j], 2),
            ha="center",
            va="center"
        )

plt.tight_layout()
plt.show()


# ==============================
# 8. SELECT FEATURES AND TARGET
# ==============================

X = df[
    [
        "passenger_count",
        "distance_km"
    ]
]

y = df["fare_amount"]


# ==============================
# 9. TRAIN-TEST SPLIT
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining Data:", X_train.shape)
print("Testing Data:", X_test.shape)


# ==============================
# 10. LINEAR REGRESSION
# ==============================

linear_model = LinearRegression()

linear_model.fit(X_train, y_train)

linear_prediction = linear_model.predict(X_test)

linear_mse = mean_squared_error(
    y_test,
    linear_prediction
)

linear_rmse = np.sqrt(linear_mse)

linear_mae = mean_absolute_error(
    y_test,
    linear_prediction
)

linear_r2 = r2_score(
    y_test,
    linear_prediction
)


# ==============================
# 11. RANDOM FOREST REGRESSION
# ==============================

random_forest_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

random_forest_model.fit(
    X_train,
    y_train
)

rf_prediction = random_forest_model.predict(X_test)

rf_mse = mean_squared_error(
    y_test,
    rf_prediction
)

rf_rmse = np.sqrt(rf_mse)

rf_mae = mean_absolute_error(
    y_test,
    rf_prediction
)

rf_r2 = r2_score(
    y_test,
    rf_prediction
)


# ==============================
# 12. MODEL COMPARISON
# ==============================

print("\n========================================")
print("MODEL EVALUATION")
print("========================================")

print("\nLinear Regression:")
print("MSE  :", linear_mse)
print("RMSE :", linear_rmse)
print("MAE  :", linear_mae)
print("R2   :", linear_r2)

print("\nRandom Forest Regression:")
print("MSE  :", rf_mse)
print("RMSE :", rf_rmse)
print("MAE  :", rf_mae)
print("R2   :", rf_r2)


# ==============================
# 13. COMPARISON GRAPH
# ==============================

models = ["Linear Regression", "Random Forest"]

r2_scores = [linear_r2, rf_r2]
rmse_scores = [linear_rmse, rf_rmse]

plt.figure(figsize=(8, 5))

plt.bar(models, r2_scores)

plt.title("R2 Score Comparison")
plt.ylabel("R2 Score")

plt.show()


plt.figure(figsize=(8, 5))

plt.bar(models, rmse_scores)

plt.title("RMSE Comparison")
plt.ylabel("RMSE")

plt.show()


# ==============================
# 14. SAMPLE PREDICTION
# ==============================

sample = pd.DataFrame({
    "passenger_count": [2],
    "distance_km": [5]
})

predicted_fare = random_forest_model.predict(sample)

print("\n========================================")
print("SAMPLE PREDICTION")
print("========================================")

print("Passengers:", sample["passenger_count"].iloc[0])
print("Distance:", sample["distance_km"].iloc[0], "km")
print("Predicted Fare:", predicted_fare[0])