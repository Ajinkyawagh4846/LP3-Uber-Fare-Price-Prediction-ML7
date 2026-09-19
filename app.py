import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

df = pd.read_csv("uber.csv")

print("Dataset Shape:", df.shape)
print(df.head())

# %% CELL 2: Data Preprocessing

df = df.drop(columns=["key", "Unnamed: 0"])
df = df.dropna()

df = df[(df["fare_amount"] > 0) &
        (df["passenger_count"] >= 1) &
        (df["passenger_count"] <= 6)]

def haversine(lat1, lon1, lat2, lon2):
    R = 6371

    lat1, lon1, lat2, lon2 = map(
        np.radians, [lat1, lon1, lat2, lon2]
    )

    a = (np.sin((lat2-lat1)/2)**2 +
         np.cos(lat1) * np.cos(lat2) *
         np.sin((lon2-lon1)/2)**2)

    return 2 * R * np.arcsin(np.sqrt(a))

df["distance_km"] = haversine(
    df["pickup_latitude"],
    df["pickup_longitude"],
    df["dropoff_latitude"],
    df["dropoff_longitude"]
)

df = df[df["distance_km"] > 0]

print("Preprocessing completed!")
print(df.head())

# %% CELL 3: Identify and Remove Outliers

sns.boxplot(x=df["fare_amount"])
plt.title("Fare Amount Outliers")
plt.show()

Q1 = df["fare_amount"].quantile(0.25)
Q3 = df["fare_amount"].quantile(0.75)
IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

outliers = ((df["fare_amount"] < lower) |
            (df["fare_amount"] > upper)).sum()

print("Number of outliers:", outliers)

df = df[(df["fare_amount"] >= lower) &
        (df["fare_amount"] <= upper)]

print("Shape after removing outliers:", df.shape)

# %% CELL 4: Correlation Analysis

corr = df[[
    "fare_amount",
    "passenger_count",
    "distance_km"
]].corr()

print(corr)

sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()

# %% CELL 5: Train Models

X = df[["passenger_count", "distance_km"]]
y = df["fare_amount"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)

# Random Forest Regression
rf = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

print("Models trained successfully!")

# %% CELL 6: Model Evaluation

for name, pred in [
    ("Linear Regression", lr_pred),
    ("Random Forest Regression", rf_pred)
]:

    mse = mean_squared_error(y_test, pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, pred)

    print("\n", name)
    print("R2 Score:", round(r2, 4))
    print("MSE:", round(mse, 4))
    print("RMSE:", round(rmse, 4))

    